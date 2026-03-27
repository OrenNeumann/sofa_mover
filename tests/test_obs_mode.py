"""Tests for the obs_mode configuration module."""

import pytest
import torch

from sofa_mover.env import SofaEnvConfig
from sofa_mover.networks import SofaBoundaryEncoder, SofaEncoder
from sofa_mover.obs_mode import make_encoder, make_env_config


class TestMakeEnvConfig:
    def test_overrides_take_precedence(self) -> None:
        cfg = make_env_config("safe", obs_downscale=4, max_steps=50)
        assert cfg.obs_downscale == 4
        assert cfg.max_steps == 50
        # Non-overridden mode default still applies
        assert cfg.boundary_rays == 0

    def test_returns_sofa_env_config(self) -> None:
        cfg = make_env_config("baseline")
        assert isinstance(cfg, SofaEnvConfig)

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown obs_mode"):
            make_env_config("nonexistent")  # type: ignore[arg-type]


class TestMakeEncoder:
    def test_grid_mode_returns_sofa_encoder(self) -> None:
        cfg = make_env_config("baseline")
        encoder = make_encoder(cfg)
        assert isinstance(encoder, SofaEncoder)

    def test_boundary_mode_returns_boundary_encoder(self) -> None:
        cfg = make_env_config("aggressive")
        encoder = make_encoder(cfg)
        assert isinstance(encoder, SofaBoundaryEncoder)

    def test_custom_feature_dim(self) -> None:
        cfg = make_env_config("baseline")
        encoder = make_encoder(cfg, feature_dim=64)
        # Verify output dim by running a dummy forward pass
        dummy_obs = torch.zeros(1, 1, 32, 32)
        dummy_pose = torch.zeros(1, 3)
        dummy_progress = torch.zeros(1, 1)
        out = encoder(dummy_obs, dummy_pose, dummy_progress)
        assert out.shape[-1] == 64

    def test_boundary_custom_feature_dim(self) -> None:
        cfg = make_env_config("aggressive")
        encoder = make_encoder(cfg, feature_dim=64)
        dummy_obs = torch.zeros(1, 128)
        dummy_pose = torch.zeros(1, 3)
        dummy_progress = torch.zeros(1, 1)
        out = encoder(dummy_obs, dummy_pose, dummy_progress)
        assert out.shape[-1] == 64

    def test_encoder_derives_from_cfg_not_mode_name(self) -> None:
        """Encoder type should depend on cfg.boundary_rays, not mode name."""
        cfg = make_env_config("baseline", boundary_rays=64)
        encoder = make_encoder(cfg)
        assert isinstance(encoder, SofaBoundaryEncoder)
