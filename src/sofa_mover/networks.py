"""Neural network modules for the sofa moving agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from sofa_mover.training.normalizer import Normalizer


class MultiDiscreteCategorical(torch.distributions.Distribution):
    """Independent Categoricals over a concatenated one-hot action vector.

    Each axis is sampled independently; log_prob and entropy are sums across axes.

    Args:
        logits: Concatenated logits of shape (..., sum(nvec)).
        nvec: Number of bins per axis.
    """

    has_enumerate_support = False
    has_rsample = False

    def __init__(self, logits: Tensor, nvec: list[int]) -> None:
        self.nvec = nvec
        split_logits = logits.split(nvec, dim=-1)
        self._dists = [
            torch.distributions.Categorical(logits=lg) for lg in split_logits
        ]
        super().__init__(batch_shape=logits.shape[:-1])

    def sample(
        self, sample_shape: torch.Size | list[int] | tuple[int, ...] = torch.Size()
    ) -> Tensor:
        indices = [d.sample(sample_shape) for d in self._dists]
        one_hots = [F.one_hot(i, n).float() for i, n in zip(indices, self.nvec)]
        return torch.cat(one_hots, dim=-1)

    def rsample(
        self, sample_shape: torch.Size | list[int] | tuple[int, ...] = torch.Size()
    ) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, action: Tensor) -> Tensor:
        """action: (..., sum(nvec)) concatenated one-hot (float or bool)."""
        split_actions = action.split(self.nvec, dim=-1)
        indices = [a.argmax(dim=-1) for a in split_actions]
        log_probs = torch.stack(
            [d.log_prob(i) for d, i in zip(self._dists, indices)], dim=-1
        )
        return log_probs.sum(dim=-1)

    def entropy(self) -> Tensor:
        entropies = torch.stack([d.entropy() for d in self._dists], dim=-1)
        return entropies.sum(dim=-1)

    @property
    def mode(self) -> Tensor:
        """Greedy (mode) action: argmax per axis, returned as concatenated one-hot."""
        modes = [d.logits.argmax(dim=-1) for d in self._dists]
        one_hots = [F.one_hot(m, n).float() for m, n in zip(modes, self.nvec)]
        return torch.cat(one_hots, dim=-1)

    @property
    def mean(self) -> Tensor:
        return self.mode

    @property
    def deterministic_sample(self) -> Tensor:
        return self.mode


class SofaEncoder(nn.Module):
    """CNN encoder for the sofa view tensor.

    Resolution-independent thanks to AdaptiveAvgPool2d.
    Takes 1-channel sofa view + pose (3) + progress (1) as inputs.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        # conv_out (128) + pose (3) + progress (1) = 132
        self.fc = nn.Linear(128 + 3 + 1, feature_dim)
        self.relu = nn.ReLU()

    def forward(
        self,
        sofa_view: Float[Tensor, "*batch 1 H W"],
        pose: Float[Tensor, "*batch 3"],
        progress: Float[Tensor, "*batch 1"],
    ) -> Float[Tensor, "*batch feature_dim"]:
        sofa_view = sofa_view.float()  # grid obs is uint8
        # Flatten any leading batch dims (e.g. [B, T, C, H, W] -> [B*T, C, H, W])
        leading = sofa_view.shape[:-3]
        x = sofa_view.reshape(-1, *sofa_view.shape[-3:])
        x = self.conv(x)
        x = x.flatten(1)
        p = progress.reshape(-1, 1)
        pose_flat = pose.reshape(-1, 3)
        x = torch.cat([x, pose_flat, p], dim=1)
        x = self.relu(self.fc(x))
        return x.reshape(*leading, -1)


class SofaBoundaryEncoder(nn.Module):
    """MLP encoder for boundary-profile sofa views.

    Takes a 1D radial profile (N,) + pose (3) + progress (1) and produces features.
    """

    def __init__(
        self,
        n_rays: int = 128,
        feature_dim: int = 128,
        normalizer: Normalizer | None = None,
    ) -> None:
        super().__init__()
        self.normalizer = normalizer
        self.mlp = nn.Sequential(
            nn.Linear(n_rays + 3 + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        sofa_view: Float[Tensor, "*batch N"],
        pose: Float[Tensor, "*batch 3"],
        progress: Float[Tensor, "*batch 1"],
    ) -> Float[Tensor, "*batch feature_dim"]:
        leading = sofa_view.shape[:-1]
        obs_flat = sofa_view.reshape(-1, sofa_view.shape[-1])
        pose_flat = pose.reshape(-1, 3)
        p_flat = progress.reshape(-1, 1)
        x = torch.cat([obs_flat, pose_flat, p_flat], dim=1)
        if self.normalizer is not None:
            x = self.normalizer.normalize_flat_observation(x)
        x = self.mlp(x)
        return x.reshape(*leading, -1)


class SofaActorNet(nn.Module):
    """Actor network: sofa view + pose + progress -> action logits.

    Uses a shared encoder (call .encoder to get the same instance for the
    critic). Accepts either SofaEncoder (grid) or SofaBoundaryEncoder, so
    sofa_view may be image-shaped or already flattened.

    Args:
        nvec: Bins per axis for the MultiDiscrete action space (from config).
            The head outputs sum(nvec) logits (concatenated per-axis logits).
        feature_dim: Size of the shared encoder output.
        encoder: Optional pre-built encoder to share with the critic.
    """

    def __init__(
        self,
        nvec: list[int],
        feature_dim: int = 128,
        encoder: SofaEncoder | SofaBoundaryEncoder | None = None,
    ) -> None:
        super().__init__()
        self.nvec = nvec
        self.encoder = encoder if encoder is not None else SofaEncoder(feature_dim)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, sum(nvec)),
        )

    def forward(
        self,
        sofa_view: Float[Tensor, "..."],
        pose: Float[Tensor, "*batch 3"],
        progress: Float[Tensor, "*batch 1"],
    ) -> Float[Tensor, "*batch n_actions"]:
        features = self.encoder(sofa_view, pose, progress)
        return self.head(features)


class SofaCriticNet(nn.Module):
    """Critic network that shares an encoder with the actor.

    Pass the actor's encoder at construction to share weights.
    """

    def __init__(
        self, encoder: SofaEncoder | SofaBoundaryEncoder, feature_dim: int = 128
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
        )

    def forward(
        self,
        sofa_view: Float[Tensor, "..."],
        pose: Float[Tensor, "*batch 3"],
        progress: Float[Tensor, "*batch 1"],
    ) -> Float[Tensor, "*batch 1"]:
        features = self.encoder(sofa_view, pose, progress)
        return self.head(features)
