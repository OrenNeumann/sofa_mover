"""Per-step reward shaper that ports the SB3-style VecNormalize reward scaling.

`sofa_mover.training.normalizer.Normalizer.normalize_rewards` consumes a whole
(B, T, 1) rollout at once. skrl calls the shaper once per env step with rewards
of shape (num_envs, 1), so we keep the running discounted returns on this
instance and update them step by step.

Formula matches the per-step branch of `Normalizer.normalize_rewards`:
    return_t = gamma * return_{t-1} + reward_t
    update running-std on return_t
    output = clip(reward_t / std, -clip, clip)
    return_t[done_t] = 0
"""

import torch

from sofa_mover.training.normalizer import RunningMeanStd


class RunningReturnRewardShaper:
    def __init__(
        self,
        *,
        num_envs: int,
        device: torch.device,
        gamma: float = 0.99,
        clip: float = 10.0,
        epsilon: float = 1e-8,
    ) -> None:
        self.gamma = gamma
        self.clip = clip
        self.epsilon = epsilon
        self.freeze = False

        self._ret_rms = RunningMeanStd(shape=(), device=device)
        self._returns = torch.zeros(num_envs, dtype=torch.float32, device=device)
        # The current dones tensor must be plumbed in from the env loop before
        # each shaper call, because skrl's rewards_shaper callback signature
        # only receives (rewards, timestep, timesteps).
        self._current_done: torch.Tensor | None = None

    def set_done(self, done: torch.Tensor) -> None:
        """Install the done mask for the current env step.

        Must be called before the agent records the transition.
        """
        self._current_done = done

    def __call__(
        self, rewards: torch.Tensor, timestep: int, timesteps: int
    ) -> torch.Tensor:
        # rewards: (num_envs, 1)
        reward_flat = rewards.squeeze(-1)
        if not self.freeze:
            with torch.no_grad():
                self._returns = self._returns * self.gamma + reward_flat
                self._ret_rms.update(self._returns)
                if self._current_done is not None:
                    self._returns[self._current_done] = 0.0

        ret_std = self._ret_rms.var.sqrt().clamp(min=self.epsilon)
        normalized = (reward_flat / ret_std).clamp(-self.clip, self.clip)
        return normalized.unsqueeze(-1)

    def state_dict(self) -> dict:
        return {
            "ret_rms": self._ret_rms.state_dict(),
            "returns": self._returns.clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        self._ret_rms.load_state_dict(state["ret_rms"])
        self._returns = state["returns"].to(self._returns.device)
