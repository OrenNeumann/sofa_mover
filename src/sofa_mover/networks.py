"""Neural network modules for the sofa moving agent."""

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class SofaEncoder(nn.Module):
    """CNN encoder for the sofa grid observation.

    Resolution-independent thanks to AdaptiveAvgPool2d.
    """

    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128 + 1, feature_dim)
        self.relu = nn.ReLU()

    def forward(
        self,
        observation: Float[Tensor, "*batch 2 H W"],
        progress: Float[Tensor, "*batch 1"],
    ) -> Float[Tensor, "*batch feature_dim"]:
        # Flatten any leading batch dims (e.g. [B, T, C, H, W] -> [B*T, C, H, W])
        leading = observation.shape[:-3]
        x = observation.reshape(-1, *observation.shape[-3:])
        x = self.conv(x)
        x = x.flatten(1)
        p = progress.reshape(-1, 1)
        x = torch.cat([x, p], dim=1)
        x = self.relu(self.fc(x))
        return x.reshape(*leading, -1)


class SofaActorNet(nn.Module):
    """Actor network: observation + progress -> action logits.

    Uses a shared encoder (call .encoder to get the same instance for the
    critic).
    """

    def __init__(self, feature_dim: int = 128, n_actions: int = 27) -> None:
        super().__init__()
        self.encoder = SofaEncoder(feature_dim)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, n_actions),
        )

    def forward(
        self,
        observation: Float[Tensor, "*batch 2 H W"],
        progress: Float[Tensor, "*batch 1"],
    ) -> Float[Tensor, "*batch n_actions"]:
        features = self.encoder(observation, progress)
        return self.head(features)


class SofaCriticNet(nn.Module):
    """Critic network that shares an encoder with the actor.

    Pass the actor's encoder at construction to share weights.
    """

    def __init__(self, encoder: SofaEncoder, feature_dim: int = 128) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
        )

    def forward(
        self,
        observation: Float[Tensor, "*batch 2 H W"],
        progress: Float[Tensor, "*batch 1"],
    ) -> Float[Tensor, "*batch 1"]:
        features = self.encoder(observation, progress)
        return self.head(features)
