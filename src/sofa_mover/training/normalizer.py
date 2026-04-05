"""Shared running normalization for flat observations and rewards.
Basically a PyTorch implementation of SB3's VecNormalize."""

from typing import TypedDict, cast
import warnings

import torch

from sofa_mover.training.config import TrainingConfig


class RunningMeanStdState(TypedDict):
    mean: torch.Tensor
    var: torch.Tensor
    count: float


# TODO: optimize for gpu torch tensors
class RunningMeanStd:
    """Tracks running mean and variance using Welford's parallel algorithm."""

    mean: torch.Tensor
    var: torch.Tensor
    count: float

    def __init__(
        self, shape: tuple[int, ...] = (), device: torch.device = torch.device("cpu")
    ) -> None:
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = 1e-4

    def update(self, batch: torch.Tensor) -> None:
        """Update stats from a batch. Leading dims are treated as batch dims."""
        flat = batch.reshape(-1, *self.mean.shape)
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, correction=0)
        batch_count = flat.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
        batch_count: int,
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta.square() * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count

    def state_dict(self) -> RunningMeanStdState:
        return {
            "mean": self.mean.clone(),
            "var": self.var.clone(),
            "count": self.count,
        }

    def load_state_dict(self, state: RunningMeanStdState) -> None:
        device = self.mean.device
        self.mean = state["mean"].to(device=device, dtype=torch.float32)
        self.var = state["var"].to(device=device, dtype=torch.float32)
        self.count = float(state["count"])


class Normalizer:
    """Shared running normalization for flat observations and rewards.
    Based on SB3's VecNormalize."""

    # TODO: for mypy, remove after refactor. also remove the asserts in this class.
    _obs_rms: RunningMeanStd | None
    _ret_rms: RunningMeanStd | None
    _returns: torch.Tensor | None

    def __init__(
        self,
        obs_dim: int | None,
        num_envs: int,
        device: torch.device,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        norm_obs: bool = True,
        norm_reward: bool = True,
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.freeze = False

        self._obs_rms = (
            RunningMeanStd(shape=(cast(int, obs_dim),), device=device)
            if self.norm_obs
            else None
        )
        self._ret_rms = (
            RunningMeanStd(shape=(), device=device) if self.norm_reward else None
        )
        self._returns = (
            torch.zeros(num_envs, dtype=torch.float32, device=device)
            if self.norm_reward
            else None
        )

    @classmethod
    def from_config(
        cls,
        config: TrainingConfig,
        num_envs: int,
    ) -> "Normalizer":
        """Build a Normalizer from a TrainingConfig.

        Handles obs_dim computation and disables obs normalization for grid mode.
        """
        env_cfg = config.env
        norm_obs = config.normalize_observation
        obs_dim: int | None = None
        if env_cfg.observation_type == "boundary":
            obs_dim = env_cfg.boundary_rays + 3 + 1
        elif norm_obs:
            warnings.warn(
                "Observation normalization is disabled for grid observations. "
                "Grid normalization after the conv path is not implemented.",
                category=RuntimeWarning,
                stacklevel=2,
            )
            norm_obs = False
        return cls(
            obs_dim=obs_dim,
            num_envs=num_envs,
            device=config.device,
            gamma=config.gamma,
            norm_obs=norm_obs,
            norm_reward=config.normalize_reward,
        )

    def normalize_flat_observation(self, observation: torch.Tensor) -> torch.Tensor:
        """Normalize a flat observation vector."""
        if not self.norm_obs:
            return observation
        assert self._obs_rms is not None

        if not self.freeze:
            with torch.no_grad():
                self._obs_rms.update(observation)

        mean = self._obs_rms.mean
        std = self._obs_rms.var.sqrt().clamp(min=self.epsilon)
        return ((observation - mean) / std).clamp(-self.clip_obs, self.clip_obs)

    def normalize_rewards(
        self, reward: torch.Tensor, done: torch.Tensor
    ) -> torch.Tensor:
        """Normalize rewards using running discounted-return variance.

        Expects reward and done shaped (B, T, 1).
        """
        if not self.norm_reward:
            return reward
        assert self._ret_rms is not None
        assert self._returns is not None

        reward_seq = reward.squeeze(-1)  # (B, T)
        done_seq = done.squeeze(-1).bool()

        normalized = torch.empty_like(reward_seq)
        for t in range(reward_seq.shape[1]):
            if not self.freeze:
                with torch.no_grad():
                    self._returns = self._returns * self.gamma + reward_seq[:, t]
                    self._ret_rms.update(self._returns)
                    self._returns[done_seq[:, t]] = 0.0
            ret_std = self._ret_rms.var.sqrt().clamp(min=self.epsilon)
            normalized[:, t] = (reward_seq[:, t] / ret_std).clamp(
                -self.clip_reward, self.clip_reward
            )
        return normalized.unsqueeze(-1)

    def state_dict(self) -> dict[str, RunningMeanStdState]:
        state: dict[str, RunningMeanStdState] = {}
        if self._obs_rms is not None:
            state["obs_rms"] = self._obs_rms.state_dict()
        if self._ret_rms is not None:
            state["ret_rms"] = self._ret_rms.state_dict()
        return state

    def load_state_dict(self, state: dict[str, RunningMeanStdState]) -> None:
        if self._obs_rms is not None:
            self._obs_rms.load_state_dict(state["obs_rms"])
        if self._ret_rms is not None:
            self._ret_rms.load_state_dict(state["ret_rms"])
