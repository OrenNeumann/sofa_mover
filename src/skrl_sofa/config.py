"""Training config for the skrl-based pipeline.

Mirrors sofa_mover.training.config.TrainingConfig so the two pipelines can be
benchmarked against each other without hyperparameter drift.
"""

from dataclasses import dataclass, field

import torch

from sofa_mover.training.config import DEVICE, SofaEnvConfig


@dataclass(frozen=True)
class SkrlTrainingConfig:
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
    normalize_observation: bool = True
    normalize_reward: bool = True
    device: torch.device = DEVICE
    output_dir: str = "output_skrl"
    wandb_project: str = "sofa_mover_skrl"
    image_log_interval: int = 50

    @property
    def mini_batches(self) -> int:
        """Number of mini-batches per PPO update epoch (skrl API)."""
        batch_size = self.num_envs * self.rollout_length
        return batch_size // self.minibatch_size
