"""Neural network modules for the sofa moving agent."""

from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from sofa_mover.training.config import TrainingConfig
from sofa_mover.training.normalizer import Normalizer


class SofaPolicyEncoder(Protocol):
    """Structural type for actor/critic encoders.

    Concrete encoders must expose `feature_dim` and a 3-input forward.
    """

    feature_dim: int

    def __call__(
        self,
        sofa_view: Tensor,
        pose: Tensor,
        progress: Tensor,
    ) -> Tensor: ...


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
        self.feature_dim = feature_dim
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


def _split_and_normalize(
    sofa_view: Tensor,
    pose: Tensor,
    progress: Tensor,
    normalizer: Normalizer | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Flatten leading dims, split rays into (sofa, corridor), optionally normalize
    each group. Groups come out with last-dim sizes (N, N, 3, 1)."""
    n_rays = sofa_view.shape[-1]
    assert n_rays % 2 == 0, f"n_rays must be even (got {n_rays})"
    n = n_rays // 2
    sofa_flat = sofa_view.reshape(-1, n_rays)
    sofa_rays = sofa_flat[:, :n]
    corridor_rays = sofa_flat[:, n:]
    pose_flat = pose.reshape(-1, 3)
    progress_flat = progress.reshape(-1, 1)
    if normalizer is not None:
        sofa_rays = normalizer.normalize_group(sofa_rays, "sofa_rays")
        corridor_rays = normalizer.normalize_group(corridor_rays, "corridor_rays")
        pose_flat = normalizer.normalize_group(pose_flat, "pose")
        progress_flat = normalizer.normalize_group(progress_flat, "progress")
    return sofa_rays, corridor_rays, pose_flat, progress_flat


class BoundaryEncoder(nn.Module):
    """MLP encoder for boundary-profile sofa views.

    Takes a 1D radial profile (N,) + pose (3) + progress (1) and produces features.
    The trunk is `depth` hidden layers of size `width`, followed by a projection
    to `feature_dim`. All layers use ReLU.
    """

    def __init__(
        self,
        n_rays: int = 128,
        feature_dim: int = 128,
        width: int = 256,
        depth: int = 2,
        normalizer: Normalizer | None = None,
    ) -> None:
        super().__init__()
        self.normalizer = normalizer
        self.feature_dim = feature_dim
        sizes = [n_rays + 3 + 1] + [width] * depth + [feature_dim]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(sizes[:-1], sizes[1:], strict=True):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        sofa_view: Float[Tensor, "*batch N"],
        pose: Float[Tensor, "*batch 3"],
        progress: Float[Tensor, "*batch 1"],
    ) -> Float[Tensor, "*batch feature_dim"]:
        leading = sofa_view.shape[:-1]
        sofa, corridor, pose_flat, progress_flat = _split_and_normalize(
            sofa_view, pose, progress, self.normalizer
        )
        x = torch.cat([sofa, corridor, pose_flat, progress_flat], dim=1)
        x = self.mlp(x)
        return x.reshape(*leading, -1)


class CircularBoundaryEncoder(nn.Module):
    """Circular Conv1d encoder for paired sofa/corridor boundary rays.

    The flat `sofa_view` of length `n_rays` is split into two channels by the
    order produced by `BoundaryExtractor`: the first `n_rays // 2` values are
    sofa rays, the rest are corridor rays, both sampled at identical angles.
    Convolutions run over the angular axis with circular padding so the conv
    stack is equivariant to rotations of the ray origin — with `stride=1`,
    exactly for any shift; with `stride>1`, only for shifts that are multiples
    of `stride`. The last conv carries the stride so intermediate features keep
    full angular resolution while the flattened head stays small.

    Rotation equivariance end-to-end also requires the normalizer to tie stats
    across rays (see `ObsGroupSpec.tied` / `Normalizer.from_config`); otherwise
    per-angle means/stds break the symmetry after warmup.
    """

    def __init__(
        self,
        n_rays: int = 128,
        feature_dim: int = 128,
        channels: int = 16,
        depth: int = 1,
        kernel_size: int = 9,
        stride: int = 4,
        normalizer: Normalizer | None = None,
    ) -> None:
        super().__init__()
        assert n_rays % 2 == 0, f"n_rays must be even (got {n_rays})"
        assert kernel_size % 2 == 1, f"kernel_size must be odd (got {kernel_size})"
        assert depth >= 1, f"depth must be >= 1 (got {depth})"
        assert stride >= 1, f"stride must be >= 1 (got {stride})"
        self.normalizer = normalizer
        self.feature_dim = feature_dim
        self.n_angles = n_rays // 2
        self.out_angles = (self.n_angles - 1) // stride + 1
        self.pad = kernel_size // 2
        conv_channels = [2] + [channels] * depth
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_dim,
                    out_dim,
                    kernel_size=kernel_size,
                    padding=0,
                    stride=stride if idx == depth - 1 else 1,
                )
                for idx, (in_dim, out_dim) in enumerate(
                    zip(conv_channels[:-1], conv_channels[1:], strict=True)
                )
            ]
        )
        self.fc = nn.Linear(conv_channels[-1] * self.out_angles + 3 + 1, feature_dim)
        self.relu = nn.ReLU()

    def _circular_pad(self, profile: Tensor) -> Tensor:
        p = self.pad
        if p == 0:
            return profile
        return torch.cat([profile[..., -p:], profile, profile[..., :p]], dim=-1)

    def forward(
        self,
        sofa_view: Float[Tensor, "*batch N"],
        pose: Float[Tensor, "*batch 3"],
        progress: Float[Tensor, "*batch 1"],
    ) -> Float[Tensor, "*batch feature_dim"]:
        leading = sofa_view.shape[:-1]
        sofa, corridor, pose_flat, progress_flat = _split_and_normalize(
            sofa_view, pose, progress, self.normalizer
        )
        profile = torch.stack([sofa, corridor], dim=1)
        for conv in self.convs:
            profile = self.relu(conv(self._circular_pad(profile)))
        x = torch.cat([profile.flatten(1), pose_flat, progress_flat], dim=1)
        x = self.relu(self.fc(x))
        return x.reshape(*leading, -1)


def _build_mlp_head(in_dim: int, width: int, depth: int, out_dim: int) -> nn.Sequential:
    """MLP with `depth` hidden ReLU layers of size `width`, linear output."""
    sizes = [in_dim] + [width] * depth
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:], strict=True):
        layers.append(nn.Linear(a, b))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(sizes[-1], out_dim))
    return nn.Sequential(*layers)


def build_encoder(
    config: TrainingConfig,
    normalizer: Normalizer | None,
) -> SofaPolicyEncoder:
    """Construct the policy encoder selected by `config`."""
    env_cfg = config.env
    if env_cfg.observation_type == "grid":
        return SofaEncoder()
    n_rays = 2 * env_cfg.boundary_rays
    if config.boundary_encoder == "mlp":
        return BoundaryEncoder(
            n_rays=n_rays,
            width=config.boundary_mlp_width,
            depth=config.boundary_mlp_depth,
            normalizer=normalizer,
        )
    if config.boundary_encoder == "circular_conv":
        return CircularBoundaryEncoder(
            n_rays=n_rays,
            channels=config.boundary_conv_channels,
            depth=config.boundary_conv_depth,
            kernel_size=config.boundary_conv_kernel_size,
            stride=config.boundary_conv_stride,
            normalizer=normalizer,
        )
    raise ValueError(f"Unknown boundary_encoder: {config.boundary_encoder!r}")


class SofaActorNet(nn.Module):
    """Actor network: sofa view + pose + progress -> action logits.

    Uses a shared encoder (call .encoder to get the same instance for the
    critic). Accepts grid and boundary encoders, so sofa_view may be
    image-shaped or already flattened.

    Args:
        nvec: Bins per axis for the MultiDiscrete action space (from config).
            The head outputs sum(nvec) logits (concatenated per-axis logits).
        encoder: Pre-built encoder (shared with the critic). Build via
            `build_encoder(config, normalizer)`.
        width: Hidden layer width in the head MLP.
        depth: Number of hidden layers in the head MLP.
    """

    def __init__(
        self,
        nvec: list[int],
        encoder: SofaPolicyEncoder,
        width: int = 128,
        depth: int = 1,
    ) -> None:
        super().__init__()
        self.nvec = nvec
        self.encoder = encoder
        self.head = _build_mlp_head(encoder.feature_dim, width, depth, sum(nvec))

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
        self,
        encoder: SofaPolicyEncoder,
        width: int = 128,
        depth: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = _build_mlp_head(encoder.feature_dim, width, depth, 1)

    def forward(
        self,
        sofa_view: Float[Tensor, "..."],
        pose: Float[Tensor, "*batch 3"],
        progress: Float[Tensor, "*batch 1"],
    ) -> Float[Tensor, "*batch 1"]:
        features = self.encoder(sofa_view, pose, progress)
        return self.head(features)
