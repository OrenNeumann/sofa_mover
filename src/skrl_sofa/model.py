"""Shared actor-critic model for the skrl-based sofa trainer.

Reuses `sofa_mover.networks.SofaBoundaryEncoder` (including its internal
Normalizer for flat observations) and adds two heads. A single instance is
assigned to both `models["policy"]` and `models["value"]` so the encoder is
trained exactly once per minibatch backward pass.
"""

from typing import Any

import gymnasium
import torch
import torch.nn as nn

from skrl.models.torch import CategoricalMixin, DeterministicMixin, Model

from sofa_mover.networks import SofaBoundaryEncoder
from sofa_mover.training.normalizer import Normalizer


class SharedSofaModel(CategoricalMixin, DeterministicMixin, Model):
    """Shared encoder + categorical policy head + deterministic value head."""

    def __init__(
        self,
        *,
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        device: torch.device,
        n_rays: int,
        feature_dim: int = 128,
        normalizer: Normalizer | None = None,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        CategoricalMixin.__init__(self, unnormalized_log_prob=True, role="policy")
        DeterministicMixin.__init__(self, clip_actions=False, role="value")

        self.n_rays = n_rays
        self.encoder = SofaBoundaryEncoder(
            n_rays=n_rays, feature_dim=feature_dim, normalizer=normalizer
        )
        self.policy_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, int(action_space.n)),
        )
        self.value_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
        )

        # skrl's PPO calls policy.act() then value.act() back-to-back with the
        # same observation tensor (both during rollout collection and once per
        # minibatch during update). Without caching, the shared encoder would
        # run its forward (and backward, during update) twice. We cache by
        # tensor identity so the second call reuses features from the first:
        # autograd handles the branching correctly — gradients from both heads
        # sum at the `features` node, giving a single backward pass through
        # the encoder per minibatch. Also fixes a correctness bug where the
        # obs normalizer's running stats were updated twice per rollout step.
        self._cached_obs_ref: torch.Tensor | None = None
        self._cached_features: torch.Tensor | None = None

    def act(
        self, inputs: dict[str, Any], *, role: str = ""
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if role == "policy":
            return CategoricalMixin.act(self, inputs, role=role)
        if role == "value":
            return DeterministicMixin.act(self, inputs, role=role)
        raise ValueError(f"unknown role: {role!r}")

    def compute(
        self, inputs: dict[str, Any], role: str = ""
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        obs = inputs["observations"]
        if obs is self._cached_obs_ref:
            features = self._cached_features
        else:
            sofa_view = obs[..., : self.n_rays]
            pose = obs[..., self.n_rays : self.n_rays + 3]
            progress = obs[..., self.n_rays + 3 : self.n_rays + 4]
            features = self.encoder(sofa_view, pose, progress)
            self._cached_obs_ref = obs
            self._cached_features = features

        if role == "policy":
            return self.policy_head(features), {}
        if role == "value":
            return self.value_head(features), {}
        raise ValueError(f"unknown role: {role!r}")
