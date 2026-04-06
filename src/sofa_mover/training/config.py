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


@dataclass(frozen=True)
class SofaEnvConfig:
    """Hyperparameters for the SofaEnv."""

    sofa_config: GridConfig = SOFA_CONFIG
    compile_rasterizer: bool = True
    corridor_width: float = 1.0
    delta_xy: float = 0.05
    delta_theta: float = math.pi / 60
    max_steps: int = 300
    num_substeps: int = 4
    lambda_erosion: float = 0.1
    lambda_progress: float = 1.0
    min_area_fraction: float = 0.05
    initial_pose: tuple[float, float, float] = (0.0, 2.0, 0.0)
    # Max y in corridor coords for the initial sofa region.
    start_y_max: float = -1.5
    # Goal point in corridor-centric coords (middle of exit end)
    goal_point: tuple[float, float] = (2.0, 0.0)
    goal_radius: float = 0.3
    observation_type: ObservationType = "boundary"
    # Grid sofa-view downscale factor (1 = full res, 2 = half, 4 = quarter)
    # Only used for "grid" observations.
    obs_downscale: int = 1
    # Boundary sofa-view: number of rays sampled from the sofa contour.
    # Only used for "boundary" observations.
    boundary_rays: int = 128


@dataclass(frozen=True)
class TrainingConfig:
    """Public training configuration."""

    env: SofaEnvConfig = field(default_factory=SofaEnvConfig)
    num_envs: int = 512
    total_frames: int = 2_000_000
    rollout_length: int = 64
    num_epochs: int = 4
    minibatch_size: int = 512
    lr: float = 3e-4
    lr_end_factor: float = 0.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    critic_coeff: float = 1.0
    max_grad_norm: float = 0.5
    reward_anneal_time: float = 0.6
    normalize_observation: bool = True
    normalize_reward: bool = True
    device: torch.device = DEVICE
    output_dir: str = "output"
    wandb_project: str = "sofa_mover"
    image_log_interval: int = 50
