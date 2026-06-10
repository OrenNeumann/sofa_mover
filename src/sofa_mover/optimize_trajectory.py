"""Continuous post-hoc optimization of a trained policy's best trajectory.

The trained discrete-action policy outputs trajectories quantized to a coarse
grid, leaving easy local improvements on the table. This module replays such a
trajectory as a sequence of continuous (dx, dy, dθ) deltas, then refines them
with the Cross-Entropy Method (CEM) using the same erosion simulator the env
uses. Fitness is the maximum sofa area achieved at any timestep where the COM
sits inside the goal radius (0 if the goal is never reached).

Entry points:

- ``optimize_trajectory(seed_actions, config, ...)``: optimize a given action
  sequence; called from ``train.py`` at end of training.
- ``optimize_from_checkpoint(checkpoint_path, ...)``: load a checkpoint's
  recorded best trajectory, then forward to the above.
- ``__main__``: CLI wrapper around ``optimize_from_checkpoint``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, fields
from pathlib import Path

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from sofa_mover.env import SofaEnv, _goal_corridor_to_sofa, _sofa_com, make_sofa_env
from sofa_mover.training.config import DEVICE, TrainingConfig
from sofa_mover.visualization.render import (
    FrameData,
    compute_frame_data,
    render_trajectory,
)

DEFAULT_CHECKPOINT_PATH = "output/best_policy.pt"
DEFAULT_OUTPUT_PATH = "output/optimized_trajectory.gif"


def actions_to_deltas(
    actions: Float[Tensor, "T action_dim"],
    n_bins: int,
    xy_vals: Float[Tensor, "n_bins"],  # noqa: F821 (jaxtyping dim name)
    theta_vals: Float[Tensor, "n_bins"],  # noqa: F821 (jaxtyping dim name)
) -> Float[Tensor, "T 3"]:
    """Decode (T, 3*n_bins) one-hot actions into continuous (T, 3) deltas."""
    per_axis = actions.view(actions.shape[0], 3, n_bins).argmax(dim=-1)
    return torch.stack(
        [xy_vals[per_axis[:, 0]], xy_vals[per_axis[:, 1]], theta_vals[per_axis[:, 2]]],
        dim=-1,
    )


_RolloutStep = tuple[
    Bool[Tensor, "B 1 H W"],  # sofa after this step's erosion
    Float[Tensor, "B 3"],  # pose after this step
    Bool[Tensor, "B 1 H W"],  # corridor mask at pose_next
    Float[Tensor, "B"],  # noqa: F821 (jaxtyping dim name) — sofa area
    Bool[Tensor, "B"],  # noqa: F821 (jaxtyping dim name) — at_goal flag
]


def _rollout(env: SofaEnv, deltas: Float[Tensor, "B T 3"]) -> Iterator[_RolloutStep]:
    """Yield per-step (sofa, pose, corridor, area, at_goal) for a batched rollout.

    Single source of truth for the env dynamics under continuous deltas; both
    batched fitness evaluation and single-trajectory frame collection consume it.
    """
    B, T, _ = deltas.shape
    cfg = env.cfg
    H, W = env.x_grid.shape
    sofa = torch.ones(B, 1, H, W, dtype=torch.bool, device=env.device)
    pose = env._initial_pose.unsqueeze(0).expand(B, 3).clone()
    for t in range(T):
        pose_next = pose + deltas[:, t]
        swept, corridor_next = env.rasterizer.swept_mask(
            pose, pose_next, cfg.num_substeps
        )
        sofa = sofa & swept
        area = sofa.flatten(1).sum(dim=1, dtype=torch.float32) * env.cell_area
        com = _sofa_com(sofa, env.x_grid, env.y_grid)
        at_goal = (com - _goal_corridor_to_sofa(env.goal_corridor, pose_next)).norm(
            dim=-1
        ) < cfg.goal_radius
        yield sofa, pose_next, corridor_next, area, at_goal
        pose = pose_next


def simulate_trajectories(
    env: SofaEnv, deltas: Float[Tensor, "B T 3"]
) -> Float[Tensor, "B"]:  # noqa: F821 (jaxtyping dim name)
    """Return fitness = max over timesteps of (area if COM inside goal radius else 0).

    Spanning all timesteps (rather than only the first goal-reach) gives the
    optimizer freedom to find a better goal-arrival timestep — the env's
    discrete dynamics would otherwise pin the optimum to the seed's done-step.
    """
    best = torch.zeros(deltas.shape[0], device=env.device)
    for _, _, _, area, at_goal in _rollout(env, deltas):
        best = torch.maximum(best, area * at_goal.float())
    return best


def _collect_frames(env: SofaEnv, deltas: Float[Tensor, "T 3"]) -> list[FrameData]:
    """Roll a single trajectory and collect FrameData for rendering, stopping at goal."""
    pose0 = env._initial_pose.unsqueeze(0)
    corridor0 = env.rasterizer.corridor_mask(pose0)
    sofa0 = torch.ones(1, 1, *env.x_grid.shape, dtype=torch.bool, device=env.device)
    frames = [
        compute_frame_data(
            0, tuple(pose0[0].tolist()), sofa0.float(), corridor0, env.cell_area
        )
    ]
    for t, (sofa, pose, corridor, _, at_goal) in enumerate(
        _rollout(env, deltas.unsqueeze(0)), start=1
    ):
        frames.append(
            compute_frame_data(
                t, tuple(pose[0].tolist()), sofa.float(), corridor, env.cell_area
            )
        )
        if at_goal.item():
            break
    return frames


@dataclass
class CEMConfig:
    n_iters: int = 1500
    pop_size: int = 128
    elite_frac: float = 0.15
    sigma_xy_init: float = 0.008
    sigma_theta_init: float = 0.008
    sigma_floor: float = 1e-5
    # Inertia on sigma (EMA with elite std) softens premature collapse.
    sigma_momentum: float = 0.5
    # Pad the trajectory with extra zero-delta steps so the optimizer can
    # extend past the seed's final step. Fitness uses max-area-at-goal, so
    # zero padding cannot regress the seed.
    extra_steps: int = 30
    # Trigger a restart when sigma falls below this fraction of initial.
    restart_threshold: float = 0.12
    # k-th restart re-seeds sigma at `initial * decay^k`, floored at
    # `2 * restart_threshold` so a restart always buys runway.
    restart_sigma_decay: float = 0.7


def cem_optimize(
    seed_deltas: Float[Tensor, "T 3"],
    fitness_fn: Callable[[Float[Tensor, "B T 3"]], Float[Tensor, "B"]],  # noqa: F821
    config: CEMConfig = CEMConfig(),
    log_every: int = 10,
) -> tuple[Float[Tensor, "T 3"], float, list[float]]:
    """Refine seed_deltas with the Cross-Entropy Method.

    Returns (best_deltas, best_area, history_of_best_area).
    """
    device = seed_deltas.device
    if config.extra_steps > 0:
        pad = torch.zeros(config.extra_steps, 3, device=device, dtype=seed_deltas.dtype)
        seed_deltas = torch.cat([seed_deltas, pad], dim=0)

    initial_sigma = (
        torch.tensor(
            [config.sigma_xy_init, config.sigma_xy_init, config.sigma_theta_init],
            device=device,
        )
        .expand_as(seed_deltas)
        .contiguous()
    )
    sigma = initial_sigma.clone()
    mean = seed_deltas.clone()
    n_elite = max(2, int(config.pop_size * config.elite_frac))

    best_deltas = seed_deltas.clone()
    best_area = fitness_fn(seed_deltas.unsqueeze(0)).item()
    history = [best_area]
    n_restarts = 0

    for it in range(config.n_iters):
        noise = torch.randn(config.pop_size, *seed_deltas.shape, device=device) * sigma
        candidates = mean + noise
        candidates[0] = best_deltas  # elitism

        areas = fitness_fn(candidates)
        if areas.max().item() > best_area:
            best_area = areas.max().item()
            best_deltas = candidates[areas.argmax()].clone()

        elites = candidates[areas.topk(n_elite).indices]
        new_sigma = elites.std(dim=0).clamp(min=config.sigma_floor)
        sigma = config.sigma_momentum * sigma + (1 - config.sigma_momentum) * new_sigma
        mean = elites.mean(dim=0)

        if (sigma / initial_sigma).mean().item() < config.restart_threshold:
            n_restarts += 1
            shrink = max(
                config.restart_sigma_decay**n_restarts, 2 * config.restart_threshold
            )
            sigma = initial_sigma * shrink
            mean = best_deltas.clone()

        history.append(best_area)
        if log_every > 0 and (it + 1) % log_every == 0:
            print(
                f"  CEM iter {it + 1:4d}/{config.n_iters}  best={best_area:.5f}  "
                f"sigma_xy={sigma[:, :2].mean().item():.5f}  "
                f"sigma_t={sigma[:, 2].mean().item():.5f}  restarts={n_restarts}"
            )

    return best_deltas, best_area, history


def optimize_trajectory(
    seed_actions: Float[Tensor, "T action_dim"],
    config: TrainingConfig,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    cem_config: CEMConfig = CEMConfig(),
    device: torch.device | None = None,
) -> Path:
    """Run CEM on a trajectory seed and render the optimized gif.

    `seed_actions` is the (T, 3*n_bins) one-hot action sequence that the trained
    policy produced (e.g. ``checkpoint["best_trajectory_actions"]``). Returns
    the path of the rendered gif. The optimized deltas are also saved at
    ``output_path.with_suffix(".pt")``.
    """
    device = config.device if device is None else device
    env = make_sofa_env(
        total_frames=config.total_frames,
        num_envs=cem_config.pop_size,
        cfg=config.env,
        device=device,
    )
    seed_deltas = actions_to_deltas(
        seed_actions.to(device), env.n_bins, env.xy_action_vals, env.theta_action_vals
    )
    print(
        f"Optimizing trajectory: T={seed_deltas.shape[0]} steps, "
        f"pop={cem_config.pop_size}, iters={cem_config.n_iters}"
    )
    best_deltas, best_area, history = cem_optimize(
        seed_deltas, lambda c: simulate_trajectories(env, c), cem_config
    )
    print(
        f"Seed area: {history[0]:.5f}  →  optimized: {best_area:.5f}  "
        f"(Δ = {best_area - history[0]:+.5f})"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_env = make_sofa_env(
        total_frames=config.total_frames, num_envs=1, cfg=config.env, device=device
    )
    gif_path = render_trajectory(
        _collect_frames(render_env, best_deltas),
        output_path,
        sofa_extent=render_env.sofa_extent,
        corridor_width=config.env.corridor_width,
    )
    torch.save(
        {
            "deltas": best_deltas.cpu(),
            "seed_area": history[0],
            "optimized_area": best_area,
            "history": history,
            "cem_config": cem_config,
        },
        output_path.with_suffix(".pt"),
    )
    print(f"Saved gif → {gif_path} and deltas → {output_path.with_suffix('.pt')}")
    return gif_path


def optimize_from_checkpoint(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    cem_config: CEMConfig = CEMConfig(),
) -> Path:
    """Optimize the best trajectory recorded in a training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if checkpoint["best_trajectory_actions"] is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has no recorded best trajectory."
        )
    return optimize_trajectory(
        checkpoint["best_trajectory_actions"],
        checkpoint["config"],
        output_path,
        cem_config,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    for f in fields(CEMConfig):
        parser.add_argument(
            f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default
        )
    args = parser.parse_args()
    cem_config = CEMConfig(**{f.name: getattr(args, f.name) for f in fields(CEMConfig)})
    optimize_from_checkpoint(args.checkpoint, args.output, cem_config)


if __name__ == "__main__":
    main()
