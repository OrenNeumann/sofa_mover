"""Tests for the continuous-trajectory optimizer."""

import math

import pytest
import torch

from sofa_mover.env import make_sofa_env
from sofa_mover.optimize_trajectory import (
    CEMConfig,
    actions_to_deltas,
    cem_optimize,
    simulate_trajectories,
)
from sofa_mover.training.config import GridConfig, SofaEnvConfig


TEST_DEVICE = torch.device("cpu")
TEST_SOFA = GridConfig(grid_size=32, world_size=3.0)
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


def _action(B: int, n_bins: int, dx_idx: int, dy_idx: int, dt_idx: int) -> torch.Tensor:
    action = torch.zeros(B, 3 * n_bins, device=TEST_DEVICE)
    action[:, dx_idx] = 1.0
    action[:, n_bins + dy_idx] = 1.0
    action[:, 2 * n_bins + dt_idx] = 1.0
    return action


def test_actions_to_deltas_roundtrip() -> None:
    """Decoding one-hot actions yields the env's discrete delta values."""
    env = make_sofa_env(
        total_frames=TEST_TOTAL_FRAMES,
        num_envs=1,
        cfg=_test_cfg(),
        device=TEST_DEVICE,
    )
    n = env.n_bins
    # Build a 5-step trajectory of varied bin indices.
    bin_seq = [(0, n - 1, n // 2), (n // 2, n // 2, n // 2), (3, 2, 1), (n - 1, 0, 0)]
    actions = torch.stack(
        [_action(1, n, dx, dy, dt).squeeze(0) for (dx, dy, dt) in bin_seq]
    )  # (T, 3*n)

    deltas = actions_to_deltas(actions, n, env.xy_action_vals, env.theta_action_vals)
    assert deltas.shape == (len(bin_seq), 3)
    for t, (dx, dy, dt) in enumerate(bin_seq):
        assert deltas[t, 0].item() == pytest.approx(env.xy_action_vals[dx].item())
        assert deltas[t, 1].item() == pytest.approx(env.xy_action_vals[dy].item())
        assert deltas[t, 2].item() == pytest.approx(env.theta_action_vals[dt].item())


def test_simulator_matches_env_step_by_step() -> None:
    """Running deltas via simulate_trajectories should produce the same sofa
    state as stepping the env with the matching discrete actions.

    We compare cumulative areas after each step; if the underlying sofa masks
    match step-by-step, area trajectories must match exactly.
    """
    # Use a coarser action grid so a random sequence makes meaningful moves.
    cfg = _test_cfg(
        max_steps=10,
        delta_xy=0.15,
        delta_theta=math.pi / 16,
        goal_radius=0.05,  # tight so episodes rarely fire done early
    )
    env_for_sim = make_sofa_env(
        total_frames=TEST_TOTAL_FRAMES, num_envs=1, cfg=cfg, device=TEST_DEVICE
    )
    env_for_step = make_sofa_env(
        total_frames=TEST_TOTAL_FRAMES, num_envs=1, cfg=cfg, device=TEST_DEVICE
    )

    n = env_for_sim.n_bins
    rng = torch.Generator(device=TEST_DEVICE).manual_seed(0)
    T = 8
    actions = torch.zeros(T, 3 * n, device=TEST_DEVICE)
    for t in range(T):
        # Avoid the most extreme bins so we don't blow through the corridor wall.
        idxs = torch.randint(2, n - 2, (3,), generator=rng)
        actions[t, idxs[0]] = 1.0
        actions[t, n + idxs[1]] = 1.0
        actions[t, 2 * n + idxs[2]] = 1.0

    deltas = actions_to_deltas(
        actions, n, env_for_sim.xy_action_vals, env_for_sim.theta_action_vals
    )

    # Run the env discretely, recording area after each step. Feed each action
    # into the current td, step, then carry the resulting "next" td forward.
    td = env_for_step.reset()
    env_areas: list[float] = []
    for t in range(T):
        td["action"] = actions[t].unsqueeze(0)
        td = env_for_step.step(td)["next"].clone()
        env_areas.append(
            env_for_step._sofa.flatten(1).sum(dim=1, dtype=torch.float32).item()
            * env_for_step.cell_area
        )

    # Replay the same deltas inline to expose per-step area (the simulator's
    # public API returns only a scalar fitness, so we mirror its inner loop).
    sofa = torch.ones(
        1,
        1,
        env_for_sim.x_grid.shape[0],
        env_for_sim.x_grid.shape[1],
        dtype=torch.bool,
        device=TEST_DEVICE,
    )
    pose = env_for_sim._initial_pose.unsqueeze(0).clone()
    sim_areas: list[float] = []
    for t in range(T):
        pose_next = pose + deltas[t : t + 1]
        swept, _ = env_for_sim.rasterizer.swept_mask(pose, pose_next, cfg.num_substeps)
        sofa = sofa & swept
        sim_areas.append(
            sofa.flatten(1).sum(dim=1, dtype=torch.float32).item()
            * env_for_sim.cell_area
        )
        pose = pose_next

    assert len(env_areas) == len(sim_areas)
    for t, (ea, sa) in enumerate(zip(env_areas, sim_areas)):
        assert ea == pytest.approx(
            sa, abs=1e-6
        ), f"Mismatch at t={t}: env={ea}, sim={sa}"


def test_simulator_fitness_zero_when_goal_not_reached() -> None:
    """If the trajectory never sits inside the goal radius, fitness is 0."""
    cfg = _test_cfg(goal_radius=0.01, max_steps=5)
    env = make_sofa_env(
        total_frames=TEST_TOTAL_FRAMES, num_envs=1, cfg=cfg, device=TEST_DEVICE
    )
    # All-zero deltas: pose stays at initial pose; goal is far away.
    deltas = torch.zeros(5, 3, device=TEST_DEVICE)
    fitness = simulate_trajectories(env, deltas.unsqueeze(0))
    assert fitness.item() == 0.0


def test_simulator_fitness_positive_when_starting_at_goal() -> None:
    """With a huge goal radius, the agent is at-goal from step 0; fitness > 0."""
    cfg = _test_cfg(goal_radius=10.0, max_steps=3)
    env = make_sofa_env(
        total_frames=TEST_TOTAL_FRAMES, num_envs=1, cfg=cfg, device=TEST_DEVICE
    )
    deltas = torch.zeros(3, 3, device=TEST_DEVICE)
    fitness = simulate_trajectories(env, deltas.unsqueeze(0))
    assert fitness.item() > 0.0


def test_cem_does_not_regress_fitness() -> None:
    """CEM with elitism never produces a worse result than the seed."""
    cfg = _test_cfg(goal_radius=10.0, max_steps=5)
    env = make_sofa_env(
        total_frames=TEST_TOTAL_FRAMES, num_envs=8, cfg=cfg, device=TEST_DEVICE
    )

    def fitness_fn(candidates: torch.Tensor) -> torch.Tensor:
        return simulate_trajectories(env, candidates)

    # Seed: small random deltas. Goal radius is huge so any seed already
    # achieves positive fitness; CEM should not reduce it.
    torch.manual_seed(0)
    T = 5
    seed = torch.randn(T, 3, device=TEST_DEVICE) * 0.01
    seed_fitness = fitness_fn(seed.unsqueeze(0)).item()

    cem_cfg = CEMConfig(
        n_iters=5,
        pop_size=8,
        elite_frac=0.25,
        sigma_xy_init=0.01,
        sigma_theta_init=0.01,
    )
    _, best_area, history = cem_optimize(seed, fitness_fn, cem_cfg, log_every=0)
    assert best_area >= seed_fitness - 1e-9
    assert history[0] == pytest.approx(seed_fitness, abs=1e-6)
    assert history[-1] == pytest.approx(best_area, abs=1e-6)


def test_cem_improves_when_seed_is_suboptimal() -> None:
    """Give CEM a clearly-suboptimal seed; expect it to improve at least once."""
    cfg = _test_cfg(goal_radius=10.0, max_steps=4)
    env = make_sofa_env(
        total_frames=TEST_TOTAL_FRAMES, num_envs=16, cfg=cfg, device=TEST_DEVICE
    )

    def fitness_fn(candidates: torch.Tensor) -> torch.Tensor:
        return simulate_trajectories(env, candidates)

    # Suboptimal seed: meaningful rotations that erode the sofa.
    torch.manual_seed(1)
    T = 4
    seed = torch.zeros(T, 3, device=TEST_DEVICE)
    seed[:, 2] = 0.2  # rotate aggressively each step → erosion
    seed_fitness = fitness_fn(seed.unsqueeze(0)).item()

    cem_cfg = CEMConfig(
        n_iters=15,
        pop_size=16,
        elite_frac=0.25,
        sigma_xy_init=0.0,  # only theta varies — search the theta axis
        sigma_theta_init=0.1,
    )
    _, best_area, _ = cem_optimize(seed, fitness_fn, cem_cfg, log_every=0)
    # CEM should find a trajectory with less rotation → more remaining area.
    assert best_area > seed_fitness
