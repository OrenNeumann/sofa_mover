"""Training configuration helpers."""

from dataclasses import dataclass, field

import torch

from sofa_mover.corridor import DEVICE
from sofa_mover.env import SofaEnvConfig


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
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    critic_coeff: float = 1.0
    max_grad_norm: float = 0.5
    device: torch.device = DEVICE
    output_dir: str = "output"
    wandb_project: str = "sofa_mover"
    image_log_interval: int = 50
