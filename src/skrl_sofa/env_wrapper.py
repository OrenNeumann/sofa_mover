"""skrl environment wrapper for the batched GPU SofaEnv.

Wraps sofa_mover.env.SofaEnv (a torchrl EnvBase) and exposes the skrl wrapper
interface: flat (num_envs, obs_dim) observation tensors, Discrete action
space, tuple returns, episode metrics in `info`.

Boundary observation mode only (grid obs is not supported in the skrl port).
"""

from typing import Any

import gymnasium
import torch
from tensordict import TensorDict

from skrl.envs.wrappers.torch.base import Wrapper

from sofa_mover.env import SofaEnv, make_sofa_env
from sofa_mover.training.config import SofaEnvConfig


# Extra info keys surfaced to the training loop for wandb logging.
_INFO_METRIC_KEYS = (
    "terminal_area",
    "episode_length",
    "episode_total_angle",
    "episode_total_distance",
    "reward_erosion",
    "reward_progress",
    "reward_terminal",
)


class SkrlSofaEnv(Wrapper):
    """Batched-GPU env wrapper matching the skrl torch Wrapper interface."""

    def __init__(
        self,
        *,
        num_envs: int,
        total_frames: int,
        cfg: SofaEnvConfig,
        device: torch.device,
    ) -> None:
        assert (
            cfg.observation_type == "boundary"
        ), "skrl_sofa only supports boundary observations"
        inner = make_sofa_env(
            total_frames=total_frames,
            num_envs=num_envs,
            cfg=cfg,
            device=device,
        )
        super().__init__(inner)
        self._inner: SofaEnv = inner
        self._num_envs = num_envs
        self._obs_dim = cfg.boundary_rays + 3 + 1  # rays + pose + progress

        self._observation_space = gymnasium.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(self._obs_dim,),
            dtype="float32",
        )
        self._action_space = gymnasium.spaces.Discrete(27)

        # Buffer for converting int actions -> one-hot for the inner env.
        self._action_onehot = torch.zeros(
            num_envs, 27, dtype=torch.float32, device=device
        )
        self._action_index_tensor = torch.arange(num_envs, device=device)

        # Scratch tensordict reused for each inner step call.
        self._step_td = TensorDict(
            {"action": self._action_onehot}, batch_size=(num_envs,), device=device
        )

        # Fresh-reset observation is a constant: the inner env's initial state
        # is deterministic, so the full boundary vector, pose, and progress at
        # reset are all fixed per-env. Capture them once so the partial
        # auto-reset below can splice them in without rerunning the full-batch
        # boundary extraction + COM computation per step.
        fresh_td = inner.reset()
        self._fresh_obs_row = self._flat_obs(fresh_td["observation"])[0].clone()
        self._fresh_sofa_row = inner._sofa[0].clone()
        self._fresh_pose_row = inner._pose[0].clone()
        self._fresh_goal_dist = inner._goal_dist[0].clone()

    # ---- skrl Wrapper required properties ----

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def observation_space(self) -> gymnasium.Space:
        return self._observation_space

    @property
    def state_space(self) -> gymnasium.Space:
        return self._observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        return self._action_space

    # ---- skrl Wrapper required methods ----

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        td = self._inner.reset()
        obs = self._flat_obs(td["observation"])
        return obs, {}

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        # actions is (num_envs, 1) long (CategoricalMixin convention).
        self._action_onehot.zero_()
        self._action_onehot[self._action_index_tensor, actions.view(-1).long()] = 1.0

        out_td = self._inner._step(self._step_td)
        next_td = out_td  # _step returns the "next" tensordict directly

        reward = next_td["reward"]  # (B, 1)
        terminated = next_td["terminated"]  # (B, 1)
        truncated = next_td["truncated"]  # (B, 1)
        done = next_td["done"].squeeze(-1)  # (B,)

        obs = self._flat_obs(next_td["observation"])

        # Expose per-env episode metrics for the trainer to log.
        info: dict[str, Any] = {key: next_td[key] for key in _INFO_METRIC_KEYS}
        info["done"] = done

        # Partial auto-reset: splice the cached fresh observation for done
        # envs and reset their internal state in-place. The full inner.reset()
        # call used to rerun boundary extraction + COM over the whole batch
        # for every step where any env finished — ~14ms/step of wasted work.
        if done.any():
            inner = self._inner
            inner._sofa[done] = self._fresh_sofa_row
            inner._pose[done] = self._fresh_pose_row
            inner._step_count[done] = 0
            inner._episode_total_angle[done] = 0.0
            inner._episode_total_distance[done] = 0.0
            inner._goal_dist[done] = self._fresh_goal_dist
            obs[done] = self._fresh_obs_row

        return obs, reward, terminated, truncated, info

    def state(self) -> torch.Tensor | None:
        return None

    def render(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def close(self) -> None:
        return None

    # ---- helpers ----

    def _flat_obs(self, obs_td: TensorDict) -> torch.Tensor:
        """Concat the (sofa_view, pose, progress) bundle into one flat tensor."""
        return torch.cat(
            [
                obs_td["sofa_view"],
                obs_td["pose"],
                obs_td["progress"],
            ],
            dim=-1,
        )

    @property
    def inner(self) -> SofaEnv:
        """Access the underlying SofaEnv (for logging, checkpointing)."""
        return self._inner
