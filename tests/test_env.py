"""Tests for the SofaEnv TorchRL environment."""

import math

import torch
import pytest

from sofa_mover.corridor import GridConfig
from sofa_mover.env import SofaEnvConfig, make_sofa_env


# Use small grids + CPU for fast tests
TEST_DEVICE = torch.device("cpu")
TEST_SOFA = GridConfig(grid_size=32, world_size=3.0)
NUM_ENVS = 2
H = TEST_SOFA.grid_size


def _test_cfg(**overrides) -> SofaEnvConfig:
    defaults = dict(
        sofa_config=TEST_SOFA,
        max_steps=20,
        compile_rasterizer=False,
        observation_type="grid",
        boundary_rays=0,
    )
    defaults.update(overrides)
    return SofaEnvConfig(**defaults)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def env():
    """Shared env instance — small grid on CPU for speed."""
    return make_sofa_env(num_envs=NUM_ENVS, cfg=_test_cfg(), device=TEST_DEVICE)


def _noop_action(B: int) -> torch.Tensor:
    """Action index 13 = center of 3x3x3: (dx=0, dy=0, dtheta=0)."""
    action = torch.zeros(B, 27, device=TEST_DEVICE)
    action[:, 13] = 1.0
    return action


def _random_action(B: int) -> torch.Tensor:
    action = torch.zeros(B, 27, device=TEST_DEVICE)
    idx = torch.randint(0, 27, (B,))
    action.scatter_(1, idx.unsqueeze(1), 1.0)
    return action


class TestReset:
    def test_shapes(self, env) -> None:
        td = env.reset()
        obs = td["observation"]
        # 1-channel sofa, cropped to bounding box
        assert obs.shape[0] == NUM_ENVS
        assert obs.shape[1] == 1  # single channel (sofa only)
        assert obs.dtype == torch.uint8
        assert td["pose"].shape == (NUM_ENVS, 3)
        assert td["progress"].shape == (NUM_ENVS, 1)
        assert td["done"].shape == (NUM_ENVS, 1)

    def test_sofa_is_carved(self, env) -> None:
        env.reset()
        sofa = env._sofa
        H = env.cfg.sofa_config.grid_size
        full_grid_pixels = NUM_ENVS * H * H
        assert sofa.sum() < full_grid_pixels
        assert sofa.sum() > 0

    def test_initial_progress_zero(self, env) -> None:
        td = env.reset()
        assert torch.allclose(
            td["progress"], torch.zeros_like(td["progress"]), atol=1e-5
        )

    def test_initial_pose(self, env) -> None:
        env.reset()
        expected = torch.tensor([list(env.cfg.initial_pose)], device=TEST_DEVICE)
        assert torch.allclose(env._pose, expected.expand(NUM_ENVS, -1))

    def test_observation_single_channel_binary(self, env) -> None:
        td = env.reset()
        obs = td["observation"].float()
        assert obs.min() >= 0.0 and obs.max() <= 1.0


class TestStep:
    def test_step_shapes(self, env) -> None:
        td = env.reset()
        td["action"] = _noop_action(NUM_ENVS)
        td_next = env.step(td)["next"]
        obs = td_next["observation"]
        assert obs.shape[0] == NUM_ENVS
        assert obs.shape[1] == 1  # single channel
        assert td_next["pose"].shape == (NUM_ENVS, 3)
        assert td_next["reward"].shape == (NUM_ENVS, 1)
        assert td_next["done"].shape == (NUM_ENVS, 1)
        assert td_next["terminated"].shape == (NUM_ENVS, 1)

    def test_noop_preserves_area(self, env) -> None:
        td = env.reset()
        area_before = env._sofa.sum().item()
        td["action"] = _noop_action(NUM_ENVS)
        env.step(td)
        area_after = env._sofa.sum().item()
        assert area_after == pytest.approx(area_before, rel=1e-5)

    def test_area_monotonically_decreases(self, env) -> None:
        td = env.reset()
        prev_area = env._sofa.sum().item()
        for _ in range(5):
            td["action"] = _random_action(NUM_ENVS)
            td = env.step(td)["next"]
            area = env._sofa.sum().item()
            assert area <= prev_area + 1e-5
            prev_area = area

    def test_truncation_at_max_steps(self, env) -> None:
        td = env.reset()
        for _ in range(env.cfg.max_steps):
            td["action"] = _noop_action(NUM_ENVS)
            td = env.step(td)["next"]
        assert td["done"].all()
        assert td["truncated"].all()

    def test_noop_gives_zero_reward(self, env) -> None:
        """Standing still: no erosion, no terminal bonus."""
        td = env.reset()
        td["action"] = _noop_action(NUM_ENVS)
        td_next = env.step(td)["next"]
        assert (td_next["reward"].abs() < 0.01).all()


class TestGoalDetection:
    def test_goal_reachable_with_large_steps(self) -> None:
        """With very large deltas, the agent can reach done (goal or area death)."""
        cfg = _test_cfg(
            delta_xy=0.5,
            delta_theta=math.pi / 4,
            max_steps=100,
            goal_radius=0.5,
        )
        env = make_sofa_env(num_envs=1, cfg=cfg, device=TEST_DEVICE)
        td = env.reset()
        for _ in range(100):
            # dx=0, dy=-d, dt=+d  →  index = 1*9 + 0*3 + 2 = 11
            action = torch.zeros(1, 27, device=TEST_DEVICE)
            action[0, 11] = 1.0
            td["action"] = action
            td = env.step(td)["next"]
            if td["done"].all():
                break
        assert td["done"].all()

    def test_progress_increases_toward_goal(self) -> None:
        """Moving toward the goal should increase progress."""
        cfg = _test_cfg(delta_xy=0.3, delta_theta=math.pi / 8)
        env = make_sofa_env(num_envs=1, cfg=cfg, device=TEST_DEVICE)
        td = env.reset()
        initial_progress = td["progress"][0].item()
        # Move: translate down + rotate (navigating the bend)
        for _ in range(5):
            action = torch.zeros(1, 27, device=TEST_DEVICE)
            action[0, 1 * 9 + 0 * 3 + 2] = 1.0  # dx=0, dy=-, dt=+
            td["action"] = action
            td = env.step(td)["next"]
        assert td["progress"][0].item() > initial_progress


class TestEpisodeAccumulators:
    def test_accumulators_present_in_step_output(self, env) -> None:
        td = env.reset()
        td["action"] = _noop_action(NUM_ENVS)
        td_next = env.step(td)["next"]
        for key in [
            "episode_total_angle",
            "episode_total_distance",
            "episode_length",
            "terminal_area",
        ]:
            assert key in td_next.keys(), f"Missing key: {key}"
            assert td_next[key].shape == (NUM_ENVS, 1)

    def test_noop_angle_and_distance_zero(self, env) -> None:
        td = env.reset()
        td["action"] = _noop_action(NUM_ENVS)
        td_next = env.step(td)["next"]
        assert (td_next["episode_total_angle"] == 0).all()
        assert (td_next["episode_total_distance"] == 0).all()

    def test_accumulators_increase_over_steps(self) -> None:
        cfg = _test_cfg()
        env = make_sofa_env(num_envs=1, cfg=cfg, device=TEST_DEVICE)
        td = env.reset()
        # Action 0 = (-delta_xy, -delta_xy, -delta_theta) — moves in all axes
        action = torch.zeros(1, 27, device=TEST_DEVICE)
        action[0, 0] = 1.0
        prev_dist = 0.0
        prev_angle = 0.0
        for _ in range(3):
            td["action"] = action
            td = env.step(td)["next"]
            dist = td["episode_total_distance"][0].item()
            angle = td["episode_total_angle"][0].item()
            assert dist > prev_dist
            assert angle > prev_angle
            prev_dist = dist
            prev_angle = angle

    def test_accumulators_reset_on_done(self) -> None:
        cfg = _test_cfg(max_steps=3)
        env = make_sofa_env(num_envs=1, cfg=cfg, device=TEST_DEVICE)
        td = env.reset()
        # Run to truncation with non-noop actions
        action = torch.zeros(1, 27, device=TEST_DEVICE)
        action[0, 0] = 1.0
        for _ in range(3):
            td["action"] = action
            td = env.step(td)["next"]
        assert td["done"].all()
        # Accumulators should be nonzero before reset
        assert td["episode_total_angle"][0].item() > 0
        # Reset and step — accumulators should be fresh
        td2 = env.reset()
        td2["action"] = _noop_action(1)
        td2_next = env.step(td2)["next"]
        assert td2_next["episode_total_angle"][0].item() == 0.0
        assert td2_next["episode_total_distance"][0].item() == 0.0

    def test_terminal_area_zero_on_truncation(self) -> None:
        """terminal_area should be 0 when episode ends by truncation."""
        cfg = _test_cfg(max_steps=3)
        env = make_sofa_env(num_envs=1, cfg=cfg, device=TEST_DEVICE)
        td = env.reset()
        for _ in range(3):
            td["action"] = _noop_action(1)
            td = env.step(td)["next"]
        assert td["done"].all()
        assert td["truncated"].all()
        assert td["terminal_area"][0].item() == 0.0

    def test_terminal_area_nonzero_on_goal(self) -> None:
        """terminal_area should be positive when the goal is reached."""
        # goal_radius=10 guarantees immediate goal reach on first step
        cfg = _test_cfg(goal_radius=10.0)
        env = make_sofa_env(num_envs=1, cfg=cfg, device=TEST_DEVICE)
        td = env.reset()
        td["action"] = _noop_action(1)
        td = env.step(td)["next"]
        assert td["terminated"].all()
        assert td["terminal_area"][0].item() > 0.0


class TestObsModes:
    def test_downscale(self) -> None:
        cfg = _test_cfg(obs_downscale=2)
        env = make_sofa_env(num_envs=1, cfg=cfg, device=TEST_DEVICE)
        td = env.reset()
        obs = td["observation"]
        assert obs.shape[1] == 1
        # Downscaled: each dim halved from crop
        crop_h = env._crop_y.stop - env._crop_y.start
        crop_w = env._crop_x.stop - env._crop_x.start
        assert obs.shape[2] == crop_h // 2
        assert obs.shape[3] == crop_w // 2

    def test_boundary_mode(self) -> None:
        cfg = _test_cfg(observation_type="boundary", boundary_rays=64)
        env = make_sofa_env(num_envs=2, cfg=cfg, device=TEST_DEVICE)
        td = env.reset()
        obs = td["observation"]
        assert obs.shape == (2, 64)
        assert obs.dtype == torch.float32
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0
        # After stepping, boundary should still be valid
        td["action"] = _noop_action(2)
        td_next = env.step(td)["next"]
        assert td_next["observation"].shape == (2, 64)


class TestRollout:
    def test_torchrl_rollout(self, env) -> None:
        td = env.rollout(max_steps=3)
        assert td.shape[0] == NUM_ENVS
        assert td.shape[1] == 3
