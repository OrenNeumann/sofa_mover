"""Neural network modules for the sofa moving agent."""

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from sofa_mover.training.normalizer import Normalizer


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
    """

    def __init__(
        self,
        feature_dim: int = 128,
        n_actions: int = 27,
        encoder: SofaEncoder | SofaBoundaryEncoder | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else SofaEncoder(feature_dim)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, n_actions),
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
