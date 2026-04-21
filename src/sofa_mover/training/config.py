from dataclasses import dataclass, field
import math
from typing import Literal, TypeAlias

import torch


@dataclass(frozen=True)
class GridConfig:
    """Configuration for a 2D grid mapping pixels to world coordinates.

    The grid is centered at the origin, spanning [-world_size/2, world_size/2]
    in both axes.
    """

    grid_size: int = 256
    world_size: float = 3.0

    @property
    def pixels_per_unit(self) -> float:
        return self.grid_size / self.world_size


DEVICE = torch.device("cuda")
SOFA_CONFIG = GridConfig()
ObservationType: TypeAlias = Literal["grid", "boundary"]
BoundaryEncoderType: TypeAlias = Literal["mlp", "circular_conv"]


@dataclass(frozen=True)
class SofaEnvConfig:
    """Hyperparameters for the SofaEnv."""

    sofa_config: GridConfig = SOFA_CONFIG
    corridor_width: float = 1.0
    delta_xy: float = 0.05 / 3
    delta_theta: float = math.pi / (60 * 3)
    max_steps: int = 300
    num_substeps: int = 4
    lambda_erosion: float = 0.5
    lambda_progress: float = 1.0
    min_area_fraction: float = 0.05
    initial_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Length of the sofa in world units along the corridor.
    sofa_length: float = 3.5  # TODO: should be >= 2*(1+sqrt(2)), the upper bound.
    # Max y in corridor coords for the front edge of the initial sofa.
    start_y_max: float = -1.5
    # Goal point in corridor-centric coords (middle of exit end)
    goal_point: tuple[float, float] = (3.5, 0.0)
    goal_radius: float = 0.3
    observation_type: ObservationType = "boundary"
    # Grid sofa-view downscale factor (1 = full res, 2 = half, 4 = quarter)
    # Only used for "grid" observations.
    obs_downscale: int = 1
    # Boundary sofa-view: number of rays sampled from the sofa contour.
    # Only used for "boundary" observations.
    boundary_rays: int = 128
    reward_anneal_time: float = 1000.0
    # Per-step area survival reward: small bonus proportional to current area fraction.
    # Provides dense feedback about current area level throughout the episode.
    lambda_area_step: float = 0.002
    # MultiDiscrete action space: number of non-zero magnitude levels per axis.
    # Each axis gets 2*n_magnitude_levels+1 bins: {-n·δ, ..., -δ, 0, +δ, ..., +n·δ}.
    # Actions are sampled independently per axis (dx, dy, dθ).
    n_magnitude_levels: int = 3


@dataclass(frozen=True)
class TrainingConfig:
    """Public training configuration."""

    env: SofaEnvConfig = field(default_factory=SofaEnvConfig)
    num_envs: int = 512
    total_frames: int = 6_000_000
    rollout_length: int = 64
    num_epochs: int = 4
    minibatch_size: int = 512
    lr: float = 3e-3
    lr_end_factor: float = 0.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    critic_coeff: float = 1.0
    max_grad_norm: float = 0.5
    normalize_observation: bool = True
    normalize_reward: bool = True
    boundary_encoder: BoundaryEncoderType = "circular_conv"
    # Boundary-encoder architecture (ignored for grid observations).
    boundary_mlp_width: int = 256
    boundary_mlp_depth: int = 2
    boundary_conv_channels: int = 16
    boundary_conv_depth: int = 1
    boundary_conv_kernel_size: int = 9
    boundary_conv_stride: int = 4
    # Actor/critic head architecture (shared structure, separate instances).
    head_width: int = 128
    head_depth: int = 1
    device: torch.device = DEVICE
    output_dir: str = "output"
    wandb_project: str = "sofa_mover"
    image_log_interval: int = 50
