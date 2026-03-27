"""Observation mode configuration: maps mode names to env configs and encoders.

Modes:
  baseline   — full-resolution cropped grid observation (uint8).
  safe       — 2x downscaled grid; smaller obs, faster training.
  aggressive — 128-ray radial boundary profile (float32); most compact.
"""

import dataclasses
from typing import Literal

import torch

from sofa_mover.env import SofaEnvConfig, make_sofa_env
from sofa_mover.networks import SofaBoundaryEncoder, SofaEncoder

ObsModeName = Literal["baseline", "safe", "aggressive"]

# Default parameters per mode (only fields that differ from SofaEnvConfig defaults)
_MODE_DEFAULTS: dict[ObsModeName, dict[str, int]] = {
    "baseline": {"obs_downscale": 1, "boundary_rays": 0},
    "safe": {"obs_downscale": 2, "boundary_rays": 0},
    "aggressive": {"obs_downscale": 1, "boundary_rays": 128},
}


def make_env_config(obs_mode: ObsModeName, **overrides: object) -> SofaEnvConfig:
    """Build a SofaEnvConfig for the given observation mode.

    Mode defaults can be overridden by keyword arguments, so all parameters
    remain tunable.
    """
    if obs_mode not in _MODE_DEFAULTS:
        raise ValueError(
            f"Unknown obs_mode {obs_mode!r}. Choose from: {list(_MODE_DEFAULTS)}"
        )
    base = _MODE_DEFAULTS[obs_mode]
    merged = {**base, **overrides}
    # Start from default SofaEnvConfig, apply mode + overrides
    return dataclasses.replace(SofaEnvConfig(), **merged)  # type: ignore[arg-type]


def make_encoder(
    cfg: SofaEnvConfig, feature_dim: int = 128
) -> SofaEncoder | SofaBoundaryEncoder:
    """Create the correct encoder based on env config.

    Derives encoder type from cfg.boundary_rays so that evaluate.py can
    reconstruct the encoder from a checkpoint's saved config.
    """
    if cfg.boundary_rays > 0:
        return SofaBoundaryEncoder(n_rays=cfg.boundary_rays, feature_dim=feature_dim)
    return SofaEncoder(feature_dim=feature_dim)


# TODO: this function is hacky and hardcoded for my machine, improve.
# also should get a tighter lower bound on batch size (currently 70% then round down to power of 2)
def estimate_max_num_envs(
    cfg: SofaEnvConfig,
    rollout_length: int = 64,
    device: torch.device = torch.device("cuda"),
) -> int:
    """Estimate the maximum num_envs that fits in GPU memory.

    Creates a 1-env probe to read observation dimensions, then uses a
    heuristic with a 70% safety margin.
    """
    props = torch.cuda.get_device_properties(device)
    total_mem = props.total_memory

    # Probe env to get actual observation dimensions
    probe = make_sofa_env(num_envs=1, cfg=cfg, device=device)
    if cfg.boundary_rays > 0:
        obs_bytes_per_step = cfg.boundary_rays * 4  # float32
    else:
        crop_h = probe._crop_y.stop - probe._crop_y.start
        crop_w = probe._crop_x.stop - probe._crop_x.start
        obs_bytes_per_step = (
            crop_h // cfg.obs_downscale * crop_w // cfg.obs_downscale
        )  # uint8
    del probe
    torch.cuda.empty_cache()

    # Fixed overhead: network params + optimizer + loss module + collector internals
    fixed_overhead = 500 * 1024 * 1024  # ~500 MB

    # Per-env per-step: obs + next_obs + action(27 float) + reward(1) + done(1)
    #   + pose(3 float) + progress(1 float) + misc scalars
    scalar_bytes_per_step = (27 + 1 + 1 + 3 + 1 + 5) * 4  # ~152 bytes
    step_bytes = obs_bytes_per_step * 2 + scalar_bytes_per_step  # obs + next_obs

    # Total buffer per env over full rollout
    buffer_per_env = step_bytes * rollout_length

    available = total_mem - fixed_overhead
    max_envs = int(available / buffer_per_env * 0.7)  # 70% safety margin

    # Round down to power of 2 for GPU efficiency
    if max_envs >= 2:
        max_envs = 2 ** (max_envs.bit_length() - 1)

    return max(4, min(max_envs, 1024))
