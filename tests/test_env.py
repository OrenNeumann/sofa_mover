"""Tests for the SofaEnv TorchRL environment."""

import math

import torch
import pytest

from sofa_mover.env import make_sofa_env
from sofa_mover.training.config import GridConfig, SofaEnvConfig


# Use small grids + CPU for fast tests
TEST_DEVICE = torch.device("cpu")
TEST_SOFA = GridConfig(grid_size=32, world_size=3.0)
NUM_ENVS = 2
H = TEST_SOFA.grid_size
TEST_TOTAL_FRAMES = 1_000_000


def _test_cfg(**overrides: object) -> SofaEnvConfig:
    defaults: dict[str, object] = dict(
        sofa_config=TEST_SOFA,
        max_steps=20,
        observation_type="grid",
        boundary_rays=0,
    )
    defaults.update(overrides)
    return SofaEnvConfig(**defaults)  # type: ignore[arg-type]


def _n_bins(cfg: SofaEnvConfig | None = None) -> int:
    if cfg is None:
        cfg = _test_cfg()
    return 2 * cfg.n_magnitude_levels + 1


def _action(B: int, n_bins: int, dx_idx: int, dy_idx: int, dt_idx: int) -> torch.Tensor:
    """Concatenated one-hot for the (dx, dy, dtheta) bin indices."""
    action = torch.zeros(B, 3 * n_bins, device=TEST_DEVICE)
    action[:, dx_idx] = 1.0
    action[:, n_bins + dy_idx] = 1.0
    action[:, 2 * n_bins + dt_idx] = 1.0
    return action


def _noop_action(B: int, n_bins: int | None = None) -> torch.Tensor:
    """Concatenated one-hot with zero delta on all axes (center bin per axis)."""
    if n_bins is None:
        n_bins = _n_bins()
    mid = n_bins // 2
    return _action(B, n_bins, mid, mid, mid)


def _random_action(B: int, n_bins: int | None = None) -> torch.Tensor:
    """Random independent action per axis."""
    if n_bins is None:
        n_bins = _n_bins()
    action = torch.zeros(B, 3 * n_bins, device=TEST_DEVICE)
    for axis in range(3):
        idx = torch.randint(0, n_bins, (B,))
        action.scatter_(1, (idx + axis * n_bins).unsqueeze(1), 1.0)
    return action


@pytest.fixture(scope="module")
def env():
    """Shared env instance — small grid on CPU for speed."""
    return make_sofa_env(
        total_frames=TEST_TOTAL_FRAMES,
        num_envs=NUM_ENVS,
        cfg=_test_cfg(),
        device=TEST_DEVICE,
    )


class TestReset:
    def test_shapes(self, env) -> None:
        td = env.reset()
        assert set(td["observation"].keys()) == {"sofa_view", "pose", "progress"}
        obs = td["observation", "sofa_view"]
        # 1-channel sofa, cropped to bounding box
        assert obs.shape[0] == NUM_ENVS
        assert obs.shape[1] == 1  # single channel (sofa only)
        assert obs.dtype == torch.uint8
        assert td["observation", "pose"].shape == (NUM_ENVS, 3)
        assert td["observation", "progress"].shape == (NUM_ENVS, 1)
        assert td["done"].shape == (NUM_ENVS, 1)

    def test_sofa_is_fully_present_after_reset(self, env) -> None:
        """The sofa grid is pre-cropped to the corridor's interior, so reset
        starts with every pixel occupied. The grid-mode obs is a 1/0 mask of
        the sofa state."""
        td = env.reset()
        obs = td["observation", "sofa_view"]
        assert (obs == 1).all()

    def test_initial_progress_zero(self, env) -> None:
        td = env.reset()
        assert torch.allclose(
            td["observation", "progress"],
            torch.zeros_like(td["observation", "progress"]),
            atol=1e-5,
        )

    def test_initial_pose(self, env) -> None:
        td = env.reset()
        expected = torch.tensor([list(env.cfg.initial_pose)], device=TEST_DEVICE)
        assert torch.allclose(td["observation", "pose"], expected.expand(NUM_ENVS, -1))

    def test_observation_single_channel_binary(self, env) -> None:
        td = env.reset()
        obs = td["observation", "sofa_view"].float()
        assert obs.min() >= 0.0 and obs.max() <= 1.0


class TestStep:
    def test_step_shapes(self, env) -> None:
        td = env.reset()
        td["action"] = _noop_action(NUM_ENVS, env.n_bins)
        td_next = env.step(td)["next"]
        obs = td_next["observation", "sofa_view"]
        assert obs.shape[0] == NUM_ENVS
        assert obs.shape[1] == 1  # single channel
        assert td_next["observation", "pose"].shape == (NUM_ENVS, 3)
        assert td_next["reward"].shape == (NUM_ENVS, 1)
        assert td_next["done"].shape == (NUM_ENVS, 1)
        assert td_next["terminated"].shape == (NUM_ENVS, 1)

    def test_noop_causes_no_erosion(self, env) -> None:
        """No movement => no area lost. reward_erosion is the public expression
        of area_lost (= -lambda_erosion * area_lost / initial_area)."""
        td = env.reset()
        td["action"] = _noop_action(NUM_ENVS, env.n_bins)
        td_next = env.step(td)["next"]
        assert (td_next["reward_erosion"] == 0).all()

    def test_area_never_increases(self, env) -> None:
        """Erosion is monotonic: every step's reward_erosion is non-positive."""
        td = env.reset()
        for _ in range(5):
            td["action"] = _random_action(NUM_ENVS, env.n_bins)
            td = env.step(td)["next"]
            assert (td["reward_erosion"] <= 0).all()

    def test_truncation_at_max_steps(self, env) -> None:
        td = env.reset()
        for _ in range(env.cfg.max_steps):
            td["action"] = _noop_action(NUM_ENVS, env.n_bins)
            td = env.step(td)["next"]
        assert td["done"].all()
        assert td["truncated"].all()

    def test_noop_gives_zero_reward(self, env) -> None:
        """Standing still: no erosion, no terminal bonus."""
        td = env.reset()
        td["action"] = _noop_action(NUM_ENVS, env.n_bins)
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
        env = make_sofa_env(
            total_frames=TEST_TOTAL_FRAMES,
            num_envs=1,
            cfg=cfg,
            device=TEST_DEVICE,
        )
        n_bins = env.n_bins
        mid = n_bins // 2
        # dx=0, dy=most-negative, dtheta=most-positive
        action = _action(1, n_bins, mid, 0, n_bins - 1)
        td = env.reset()
        for _ in range(100):
            td["action"] = action
            td = env.step(td)["next"]
            if td["done"].all():
                break
        assert td["done"].all()

    def test_progress_increases_toward_goal(self) -> None:
        """Moving toward the goal should increase progress."""
        cfg = _test_cfg(delta_xy=0.3, delta_theta=math.pi / 8)
        env = make_sofa_env(
            total_frames=TEST_TOTAL_FRAMES,
            num_envs=1,
            cfg=cfg,
            device=TEST_DEVICE,
        )
        n_bins = env.n_bins
        mid = n_bins // 2
        # dx=0, dy=-1 magnitude, dtheta=+1 magnitude
        action = _action(1, n_bins, mid, mid - 1, mid + 1)
        td = env.reset()
        initial_progress = td["observation", "progress"][0].item()
        for _ in range(5):
            td["action"] = action
            td = env.step(td)["next"]
        assert td["observation", "progress"][0].item() > initial_progress


class TestEpisodeAccumulators:
    def test_accumulators_present_in_step_output(self, env) -> None:
        td = env.reset()
        td["action"] = _noop_action(NUM_ENVS, env.n_bins)
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
        td["action"] = _noop_action(NUM_ENVS, env.n_bins)
        td_next = env.step(td)["next"]
        assert (td_next["episode_total_angle"] == 0).all()
        assert (td_next["episode_total_distance"] == 0).all()

    def test_accumulators_increase_over_steps(self) -> None:
        cfg = _test_cfg()
        env = make_sofa_env(
            total_frames=TEST_TOTAL_FRAMES,
            num_envs=1,
            cfg=cfg,
            device=TEST_DEVICE,
        )
        n_bins = env.n_bins
        td = env.reset()
        # All axes at most-negative magnitude (dx, dy, dtheta all negative)
        action = _action(1, n_bins, 0, 0, 0)
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
        env = make_sofa_env(
            total_frames=TEST_TOTAL_FRAMES,
            num_envs=1,
            cfg=cfg,
            device=TEST_DEVICE,
        )
        n_bins = env.n_bins
        td = env.reset()
        # Run to truncation with non-noop actions (all axes most negative)
        action = _action(1, n_bins, 0, 0, 0)
        for _ in range(3):
            td["action"] = action
            td = env.step(td)["next"]
        assert td["done"].all()
        # Accumulators should be nonzero before reset
        assert td["episode_total_angle"][0].item() > 0
        # Reset and step — accumulators should be fresh
        td2 = env.reset()
        td2["action"] = _noop_action(1, n_bins)
        td2_next = env.step(td2)["next"]
        assert td2_next["episode_total_angle"][0].item() == 0.0
        assert td2_next["episode_total_distance"][0].item() == 0.0

    def test_terminal_area_zero_on_truncation(self) -> None:
        """terminal_area should be 0 when episode ends by truncation."""
        cfg = _test_cfg(max_steps=3)
        env = make_sofa_env(
            total_frames=TEST_TOTAL_FRAMES,
            num_envs=1,
            cfg=cfg,
            device=TEST_DEVICE,
        )
        td = env.reset()
        for _ in range(3):
            td["action"] = _noop_action(1, env.n_bins)
            td = env.step(td)["next"]
        assert td["done"].all()
        assert td["truncated"].all()
        assert td["terminal_area"][0].item() == 0.0

    def test_terminal_area_nonzero_on_goal(self) -> None:
        """terminal_area should be positive when the goal is reached."""
        # goal_radius=10 guarantees immediate goal reach on first step
        cfg = _test_cfg(goal_radius=10.0)
        env = make_sofa_env(
            total_frames=TEST_TOTAL_FRAMES,
            num_envs=1,
            cfg=cfg,
            device=TEST_DEVICE,
        )
        td = env.reset()
        td["action"] = _noop_action(1, env.n_bins)
        td = env.step(td)["next"]
        assert td["terminated"].all()
        assert td["terminal_area"][0].item() > 0.0


class TestObsModes:
    def test_downscale_halves_obs_spatial_dims(self) -> None:
        """obs_downscale=k must produce sofa_view spatial dims k× smaller than
        the un-downscaled baseline."""
        env_full = make_sofa_env(
            total_frames=TEST_TOTAL_FRAMES,
            num_envs=1,
            cfg=_test_cfg(obs_downscale=1),
            device=TEST_DEVICE,
        )
        env_half = make_sofa_env(
            total_frames=TEST_TOTAL_FRAMES,
            num_envs=1,
            cfg=_test_cfg(obs_downscale=2),
            device=TEST_DEVICE,
        )
        full = env_full.reset()["observation", "sofa_view"]
        half = env_half.reset()["observation", "sofa_view"]
        assert half.shape[1] == 1
        assert half.shape[2] == full.shape[2] // 2
        assert half.shape[3] == full.shape[3] // 2

    def test_boundary_mode(self) -> None:
        cfg = _test_cfg(observation_type="boundary", boundary_rays=64)
        env = make_sofa_env(
            total_frames=TEST_TOTAL_FRAMES,
            num_envs=2,
            cfg=cfg,
            device=TEST_DEVICE,
        )
        td = env.reset()
        obs = td["observation", "sofa_view"]
        assert obs.shape == (2, 128)
        assert obs.dtype == torch.float32
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0
        # After stepping, boundary should still be valid
        td["action"] = _noop_action(2, env.n_bins)
        td_next = env.step(td)["next"]
        assert td_next["observation", "sofa_view"].shape == (2, 128)


class TestRollout:
    def test_torchrl_rollout(self, env) -> None:
        td = env.rollout(max_steps=3)
        assert td.shape[0] == NUM_ENVS
        assert td.shape[1] == 3
