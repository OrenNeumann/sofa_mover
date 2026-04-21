"""Shared running normalization for grouped observations and rewards.
Basically a PyTorch implementation of SB3's VecNormalize, extended with a
per-group schema so encoders that exploit a symmetry (e.g. rotation over
rays) can tie stats across elements that share the symmetry."""

from dataclasses import dataclass
from typing import TypedDict
import warnings

import torch

from sofa_mover.training.config import TrainingConfig


@dataclass(frozen=True)
class ObsGroupSpec:
    """A named slice of the observation with its own normalization stats.

    tied=False: one (mean, var) per element — standard per-feature normalization.
    tied=True:  one scalar (mean, var) shared by all `size` elements. Use this
                when elements are samples of the same underlying distribution
                (e.g. radial rays under a rotation-equivariant encoder) so that
                normalized outputs preserve the symmetry.
    """

    name: str
    size: int
    tied: bool = False


class RunningMeanStdState(TypedDict):
    mean: torch.Tensor
    var: torch.Tensor
    count: float


@dataclass
class NormalizerState:
    obs_rms: dict[str, RunningMeanStdState]
    ret_rms: RunningMeanStdState | None


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
        """Update stats from a batch. Leading dims are treated as batch dims.

        With a scalar stat (shape=()), the batch is flattened entirely so every
        element contributes as one sample.
        """
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
    """Shared running normalization for grouped observations and rewards.

    Observations are split into named groups via `obs_groups`; each group owns
    one RunningMeanStd (scalar if `tied`, vector otherwise). Encoders call
    `normalize_group(x, name)` directly on the group tensors they already hold,
    so there is no flat-layout contract between the normalizer and the encoders.
    """

    _obs_rms: dict[str, RunningMeanStd]
    _ret_rms: RunningMeanStd | None
    _returns: torch.Tensor | None

    def __init__(
        self,
        obs_groups: list[ObsGroupSpec],
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

        self._obs_rms = {}
        if self.norm_obs:
            for group in obs_groups:
                shape: tuple[int, ...] = () if group.tied else (group.size,)
                self._obs_rms[group.name] = RunningMeanStd(shape=shape, device=device)

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
        device: torch.device | None = None,
    ) -> "Normalizer":
        """Build a Normalizer with a schema matched to the encoder choice.

        For boundary observations the schema is
        [sofa_rays, corridor_rays, pose, progress]; ray groups are tied iff the
        encoder is rotation-equivariant (`circular_conv`). Grid observations
        disable obs normalization (conv-path normalization isn't implemented).
        """
        env_cfg = config.env
        norm_obs = config.normalize_observation
        obs_groups: list[ObsGroupSpec] = []
        if env_cfg.observation_type == "boundary":
            tied_rays = config.boundary_encoder == "circular_conv"
            n = env_cfg.boundary_rays
            obs_groups = [
                ObsGroupSpec("sofa_rays", n, tied=tied_rays),
                ObsGroupSpec("corridor_rays", n, tied=tied_rays),
                ObsGroupSpec("pose", 3),
                ObsGroupSpec("progress", 1),
            ]
        elif norm_obs:
            warnings.warn(
                "Observation normalization is disabled for grid observations. "
                "Grid normalization after the conv path is not implemented.",
                category=RuntimeWarning,
                stacklevel=2,
            )
            norm_obs = False
        return cls(
            obs_groups=obs_groups,
            num_envs=num_envs,
            device=config.device if device is None else device,
            gamma=config.gamma,
            norm_obs=norm_obs,
            norm_reward=config.normalize_reward,
        )

    def normalize_group(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """Normalize one observation group with its running stats.

        `x` must have its last dim equal to the group's declared size; leading
        dims are treated as batch. For tied groups, the scalar mean/std is
        broadcast across the last dim, preserving any symmetry in `x`.
        """
        if not self.norm_obs:
            return x
        rms = self._obs_rms[name]
        if not self.freeze:
            with torch.no_grad():
                rms.update(x)
        std = rms.var.sqrt().clamp(min=self.epsilon)
        return ((x - rms.mean) / std).clamp(-self.clip_obs, self.clip_obs)

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

        reward_seq = reward.squeeze(-1)
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

    def state_dict(self) -> NormalizerState:
        return NormalizerState(
            obs_rms={name: rms.state_dict() for name, rms in self._obs_rms.items()},
            ret_rms=self._ret_rms.state_dict() if self._ret_rms is not None else None,
        )

    def load_state_dict(self, state: NormalizerState) -> None:
        for name, rms in self._obs_rms.items():
            rms.load_state_dict(state.obs_rms[name])
        if self._ret_rms is not None:
            assert state.ret_rms is not None
            self._ret_rms.load_state_dict(state.ret_rms)
