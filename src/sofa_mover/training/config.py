"""Training configuration helpers."""

from dataclasses import dataclass

import torch

from sofa_mover.corridor import DEVICE
from sofa_mover.env import SofaEnvConfig
from sofa_mover.obs_mode import ObsModeName, estimate_max_num_envs, make_env_config


@dataclass(frozen=True)
class TrainingConfig:
    """Public training configuration."""

    obs_mode: ObsModeName = "aggressive"
    num_envs: int | str = "auto"
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
    log_dir: str = "runs/sofa_ppo"
    image_log_interval: int = 50


def resolve_training_config(
    config: TrainingConfig,
    env_cfg: SofaEnvConfig | None = None,
) -> tuple[SofaEnvConfig, int]:
    """Resolve env config and batch size from training config.

    Returns (env_cfg, num_envs).
    """
    resolved_env_cfg = (
        env_cfg if env_cfg is not None else make_env_config(config.obs_mode)
    )

    # --- Resolve auto batch size ---
    if config.num_envs == "auto":
        num_envs = estimate_max_num_envs(
            resolved_env_cfg,
            config.rollout_length,
            config.device,
        )
        print(f"Auto batch size: {num_envs} (mode={config.obs_mode})")
    else:
        num_envs = int(config.num_envs)

    return resolved_env_cfg, num_envs
