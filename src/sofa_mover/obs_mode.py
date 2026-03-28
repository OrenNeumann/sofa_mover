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

    Creates a 1-env probe to read observation dimensions, then applies an
    empirically calibrated formula:

        per_env_peak ≈ 2.5 × obs_buffer_per_env + 2.24 MB

    where obs_buffer_per_env = obs_numel × dtype_size × rollout_length × 2
    (obs + next_obs stored by the TorchRL collector).

    The 2.5× multiplier accounts for TorchRL internal copies and transient
    tensors during policy inference. The 2.24 MB constant covers the sofa
    state (float32), rasterizer intermediate peaks, and framework overhead.
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

    # Fixed overhead: network params + optimizer + one PPO minibatch pass
    fixed_overhead = 100 * 1024 * 1024  # ~100 MB

    # Collector stores obs + next_obs per step for the full rollout
    obs_buffer_per_env = obs_bytes_per_step * rollout_length * 2

    # Empirical formula: per_env_peak ≈ 2.5 × obs_buffer + 2.24 MB
    per_env_bytes = int(2.5 * obs_buffer_per_env + 2.24 * 1024 * 1024)

    available = total_mem - fixed_overhead
    max_envs = int(available / per_env_bytes * 0.75)  # 75% safety margin

    # Round down to power of 2. GPU throughput saturates well before the memory
    # limit (benchmarks show FPS flat or declining above 256/512/1024 for
    # baseline/safe/aggressive), so over-provisioning just adds PPO overhead..
    if max_envs >= 2:
        max_envs = 2 ** (max_envs.bit_length() - 1)

    return max(4, min(max_envs, 2048))
