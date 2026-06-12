"""Minimal synchronous collector for the training loop."""

from collections.abc import Iterator

import torch
from tensordict import TensorDict

from sofa_mover.env import SofaEnv
from sofa_mover.networks import MultiDiscreteCategorical, SofaActorNet


class RolloutCollector:
    """Synchronous on-device rollout collector.

    A lean replacement for ``torchrl.collectors.Collector``: with one batched
    GPU env and the policy on the same device there is nothing to multiprocess
    or sync, so it calls ``actor_net`` and the env's ``_step``/``_reset``
    directly and skips the TorchRL per-step wrappers.

    Iterating yields contiguous (num_envs, rollout_length) batches in the
    TorchRL collector's layout: pre-step observation, ``action`` and
    ``action_log_prob`` at the root, the full step result under ``next``.
    Done envs are reset with a mask every step, so the loop never syncs with
    the GPU.
    """

    def __init__(
        self,
        env: SofaEnv,
        actor_net: SofaActorNet,
        rollout_length: int,
        total_frames: int,
    ) -> None:
        self.env = env
        self.actor_net = actor_net
        self.rollout_length = rollout_length
        self.total_frames = total_frames

    def _sample_action(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor_net(obs["sofa_view"], obs["pose"], obs["progress"])
        dist = MultiDiscreteCategorical(logits=logits, nvec=self.actor_net.nvec)
        action = dist.sample()
        return action, dist.log_prob(action)

    def _step_and_reset(self, action: torch.Tensor) -> tuple[TensorDict, TensorDict]:
        """Step the env, reset done rows, return (step result, merged next obs)."""
        B = self.env.num_envs
        device = self.env.device
        out = self.env._step(TensorDict({"action": action}, batch_size=(B,)))
        done = out["done"]  # (B, 1)
        reset_td = self.env._reset(TensorDict({"_reset": done}, batch_size=(B,)))
        stepped, fresh = out["observation"], reset_td["observation"]
        next_obs = TensorDict(
            {
                key: torch.where(done, fresh[key], stepped[key])
                for key in ("sofa_view", "pose", "progress")
            },
            batch_size=(B,),
            device=device,
        )
        return out, next_obs

    def __iter__(self) -> Iterator[TensorDict]:
        B = self.env.num_envs
        n_batches = self.total_frames // (B * self.rollout_length)
        obs = self.env.reset()["observation"]
        for _ in range(n_batches):
            steps = []
            # no_grad must not wrap the yield: grad mode is thread-global, and
            # a generator suspended inside the context would leak it into the
            # training loop.
            with torch.no_grad():
                for _t in range(self.rollout_length):
                    action, log_prob = self._sample_action(obs)
                    out, next_obs = self._step_and_reset(action)
                    steps.append(
                        TensorDict(
                            {
                                "observation": obs,
                                "action": action,
                                "action_log_prob": log_prob,
                                "next": out,
                            },
                            batch_size=(B,),
                        )
                    )
                    obs = next_obs
            yield torch.stack(steps, dim=1).contiguous()

    def shutdown(self) -> None:
        """Compatibility no-op (matches torchrl's Collector API)."""
