"""TorchRL environment for the sofa moving problem."""

import math
from dataclasses import dataclass

import torch
from jaxtyping import Float
from tensordict import TensorDict, TensorDictBase
from torch import Tensor
from torchrl.data.tensor_specs import Bounded, Composite, OneHot, Unbounded
from torchrl.envs import EnvBase

from sofa_mover.corridor import (
    DEVICE,
    SOFA_CONFIG,
    GridConfig,
    Pose,
    make_l_corridor,
)
from sofa_mover.erosion import erode
from sofa_mover.rasterize import Rasterizer


@dataclass(frozen=True)
class SofaEnvConfig:
    """Hyperparameters for the SofaEnv."""

    sofa_config: GridConfig = SOFA_CONFIG
    compile_rasterizer: bool = True
    corridor_width: float = 1.0
    delta_xy: float = 0.05
    delta_theta: float = math.pi / 60
    max_steps: int = 300
    num_substeps: int = 4
    lambda_erosion: float = 0.1
    lambda_progress: float = 1.0
    min_area_fraction: float = 0.05
    initial_pose: tuple[float, float, float] = (0.0, 2.0, 0.0)
    # Goal point in corridor-centric coords (middle of exit end)
    goal_point: tuple[float, float] = (2.0, 0.0)
    goal_radius: float = 0.3


def _goal_corridor_to_sofa(
    goal_corridor: Float[Tensor, "2"], pose: Pose
) -> Float[Tensor, "B 2"]:
    """Transform goal point from corridor-centric to sofa-centric coords."""
    tx, ty, theta = pose[:, 0], pose[:, 1], pose[:, 2]
    gx, gy = goal_corridor[0], goal_corridor[1]
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    # Rotate goal_corridor by θ, then translate by (tx, ty)
    wx = cos_t * gx - sin_t * gy + tx
    wy = sin_t * gx + cos_t * gy + ty
    return torch.stack([wx, wy], dim=-1)


def _sofa_com(
    sofa: Float[Tensor, "B 1 H W"],
    x_grid: Float[Tensor, "H W"],
    y_grid: Float[Tensor, "H W"],
) -> Float[Tensor, "B 2"]:
    """Compute center of mass of sofa grids."""
    mass = sofa[:, 0]  # (B, H, H)
    total = mass.sum(dim=(-2, -1)).clamp(min=1e-8)  # (B,)
    cx = (mass * x_grid).sum(dim=(-2, -1)) / total
    cy = (mass * y_grid).sum(dim=(-2, -1)) / total
    return torch.stack([cx, cy], dim=-1)


class SofaEnv(EnvBase):
    """Batched GPU environment for the moving sofa problem.

    The sofa is stationary. The corridor moves around it (translation +
    rotation). At each step the agent picks a discrete action controlling
    the corridor's incremental movement, and everything outside the corridor
    is permanently eroded from the sofa.

    Done when the sofa's center of mass is within goal_radius of the
    goal point, or when sofa area drops below min_area_fraction, or at
    max steps.
    """

    batch_locked = True

    def __init__(
        self,
        num_envs: int,
        cfg: SofaEnvConfig = SofaEnvConfig(),
        device: torch.device = DEVICE,
    ) -> None:
        super().__init__(device=device, batch_size=torch.Size([num_envs]))
        self.cfg = cfg
        self.num_envs = num_envs

        # Build corridor geometry and rasterizer
        geometry = make_l_corridor(corridor_width=cfg.corridor_width)
        self.rasterizer = Rasterizer(
            geometry,
            cfg.sofa_config,
            device=device,
            compile=cfg.compile_rasterizer,
        )

        # Action decode table: 27 actions → (dx, dy, dθ)
        deltas_xy = torch.tensor([-cfg.delta_xy, 0.0, cfg.delta_xy], device=device)
        deltas_theta = torch.tensor(
            [-cfg.delta_theta, 0.0, cfg.delta_theta], device=device
        )
        dx_idx, dy_idx, dt_idx = torch.meshgrid(
            torch.arange(3, device=device),
            torch.arange(3, device=device),
            torch.arange(3, device=device),
            indexing="ij",
        )
        self.action_to_delta = torch.stack(
            [
                deltas_xy[dx_idx.flatten()],
                deltas_xy[dy_idx.flatten()],
                deltas_theta[dt_idx.flatten()],
            ],
            dim=-1,
        )  # (27, 3)

        # World-coordinate grids for COM calculation
        H = cfg.sofa_config.grid_size
        half = cfg.sofa_config.world_size / 2
        coords = torch.linspace(-half, half, H, device=device)
        y_grid, x_grid = torch.meshgrid(coords, coords, indexing="ij")
        self.x_grid: torch.Tensor = x_grid
        self.y_grid: torch.Tensor = y_grid

        # Goal point in corridor-centric coords (constant)
        self.goal_corridor: torch.Tensor = torch.tensor(cfg.goal_point, device=device)

        # Initial area and goal distance (single mask computation)
        self.cell_area = (cfg.sofa_config.world_size / H) ** 2
        init_pose = torch.tensor([list(cfg.initial_pose)], device=device)
        init_mask = self.rasterizer.corridor_mask(init_pose)
        self.initial_area = init_mask.sum().item() * self.cell_area
        init_sofa = torch.ones(1, 1, H, H, device=device) * init_mask
        init_com = _sofa_com(init_sofa, x_grid, y_grid)
        init_goal_sofa = _goal_corridor_to_sofa(self.goal_corridor, init_pose)
        self.initial_goal_dist = (init_com - init_goal_sofa).norm(dim=-1).item()

        # Internal state — NOT in full_state_spec to avoid huge collector buffers
        self._sofa = torch.ones(num_envs, 1, H, H, device=device)
        self._pose = torch.zeros(num_envs, 3, device=device)
        self._step_count = torch.zeros(num_envs, 1, dtype=torch.int64, device=device)

        # Episode metric accumulators (reset per episode)
        self._episode_total_angle = torch.zeros(num_envs, 1, device=device)
        self._episode_total_distance = torch.zeros(num_envs, 1, device=device)
        self._episode_area_integral = torch.zeros(num_envs, 1, device=device)

        self._make_specs()

    def _make_specs(self) -> None:
        H = self.cfg.sofa_config.grid_size
        B = self.num_envs

        _unbounded_b1 = dict(shape=(B, 1), dtype=torch.float32, device=self.device)
        self.observation_spec = Composite(
            observation=Bounded(
                low=0.0,
                high=1.0,
                shape=(B, 2, H, H),
                dtype=torch.float32,
                device=self.device,
            ),
            progress=Unbounded(**_unbounded_b1),
            terminal_area=Unbounded(**_unbounded_b1),
            episode_length=Unbounded(
                shape=(B, 1), dtype=torch.int64, device=self.device
            ),
            episode_total_angle=Unbounded(**_unbounded_b1),
            episode_total_distance=Unbounded(**_unbounded_b1),
            episode_area_integral=Unbounded(**_unbounded_b1),
            shape=(B,),
        )

        self.action_spec = OneHot(
            n=27,
            shape=(B, 27),
            dtype=torch.float32,
            device=self.device,
        )

        self.reward_spec = Unbounded(
            shape=(B, 1),
            dtype=torch.float32,
            device=self.device,
        )

    def _reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        B = self.num_envs
        H = self.cfg.sofa_config.grid_size
        device = self.device

        # Determine which envs need reset
        if tensordict is not None and "_reset" in tensordict.keys():
            reset_mask = tensordict["_reset"].squeeze(-1)  # (B,)
        else:
            reset_mask = torch.ones(B, dtype=torch.bool, device=device)

        # Reset selected envs in internal state
        n_reset = reset_mask.sum().item()
        if n_reset > 0:
            initial_pose = (
                torch.tensor([list(self.cfg.initial_pose)], device=device)
                .expand(n_reset, -1)
                .contiguous()
            )

            fresh_sofa = torch.ones(n_reset, 1, H, H, device=device)
            initial_mask = self.rasterizer.corridor_mask(initial_pose)
            fresh_sofa = erode(fresh_sofa, initial_mask)

            self._sofa[reset_mask] = fresh_sofa
            self._pose[reset_mask] = initial_pose
            self._step_count[reset_mask] = 0
            self._episode_total_angle[reset_mask] = 0.0
            self._episode_total_distance[reset_mask] = 0.0
            self._episode_area_integral[reset_mask] = 0.0

        # Build observation: (B, 2, H, H) = sofa + corridor_mask
        corridor_mask = self.rasterizer.corridor_mask(self._pose)
        observation = torch.cat([self._sofa, corridor_mask], dim=1)

        # Progress = 1 - (dist_to_goal / initial_dist)
        com = _sofa_com(self._sofa, self.x_grid, self.y_grid)
        goal_sofa = _goal_corridor_to_sofa(self.goal_corridor, self._pose)
        self._goal_dist = (com - goal_sofa).norm(dim=-1)  # (B,)
        progress = 1.0 - self._goal_dist.unsqueeze(1) / max(
            self.initial_goal_dist, 1e-8
        )

        return TensorDict(
            {
                "observation": observation,
                "progress": progress,
                "done": torch.zeros(B, 1, dtype=torch.bool, device=device),
                "terminated": torch.zeros(B, 1, dtype=torch.bool, device=device),
                "truncated": torch.zeros(B, 1, dtype=torch.bool, device=device),
                "terminal_area": torch.zeros(B, 1, device=device),
                "episode_length": torch.zeros(B, 1, dtype=torch.int64, device=device),
                "episode_total_angle": torch.zeros(B, 1, device=device),
                "episode_total_distance": torch.zeros(B, 1, device=device),
                "episode_area_integral": torch.zeros(B, 1, device=device),
            },
            batch_size=(B,),
            device=device,
        )

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        cfg = self.cfg
        B = self.num_envs

        action = tensordict["action"]  # (B, 27) one-hot

        # Decode action
        action_idx = action.argmax(dim=-1)  # (B,)
        delta = self.action_to_delta[action_idx]  # (B, 3)

        # Accumulate episode metrics
        dx, dy, dtheta = delta[:, 0], delta[:, 1], delta[:, 2]
        self._episode_total_distance += torch.sqrt(dx * dx + dy * dy).unsqueeze(1)
        self._episode_total_angle += dtheta.abs().unsqueeze(1)

        pose = self._pose
        pose_next = pose + delta

        # Area before erosion
        area_before = self._sofa.flatten(1).sum(dim=1) * self.cell_area  # (B,)

        # Swept mask erosion (also returns corridor mask at pose_next for free)
        swept, corridor_mask = self.rasterizer.swept_mask(
            pose, pose_next, cfg.num_substeps
        )
        new_sofa = erode(self._sofa, swept)

        # Area after erosion
        area_after = new_sofa.flatten(1).sum(dim=1) * self.cell_area  # (B,)
        self._episode_area_integral += area_after.unsqueeze(1)

        # Goal check: sofa COM close to goal point
        com = _sofa_com(new_sofa, self.x_grid, self.y_grid)  # (B, 2)
        goal_sofa = _goal_corridor_to_sofa(self.goal_corridor, pose_next)  # (B, 2)
        goal_dist = (com - goal_sofa).norm(dim=-1)  # (B,)
        goal_reached = goal_dist < cfg.goal_radius  # (B,)

        # Step count
        self._step_count += 1

        # Done conditions
        area_dead = (area_after / max(self.initial_area, 1e-8)) < cfg.min_area_fraction
        truncated = self._step_count.squeeze(1) >= cfg.max_steps
        terminated = goal_reached | area_dead
        done = terminated | truncated

        # --- Reward ---
        area_lost = (area_before - area_after).clamp(min=0.0)  # (B,)
        erosion_penalty = -cfg.lambda_erosion * area_lost
        # Dense progress: reward for reducing distance to goal
        progress_bonus = (
            cfg.lambda_progress
            * (self._goal_dist - goal_dist)
            / max(self.initial_goal_dist, 1e-8)
        )
        terminal_bonus = goal_reached.float() * area_after
        reward = (erosion_penalty + progress_bonus + terminal_bonus).unsqueeze(1)

        # Update internal state
        self._sofa = new_sofa
        self._pose = pose_next
        self._goal_dist = goal_dist

        # Observation
        observation = torch.cat([new_sofa, corridor_mask], dim=1)

        # Progress = 1 - (dist_to_goal / initial_dist)
        progress = 1.0 - goal_dist.unsqueeze(1) / max(self.initial_goal_dist, 1e-8)

        return TensorDict(
            {
                "observation": observation,
                "progress": progress,
                "reward": reward,
                "done": done.unsqueeze(1),
                "terminated": terminated.unsqueeze(1),
                "truncated": truncated.unsqueeze(1),
                "terminal_area": (goal_reached.float() * area_after).unsqueeze(1),
                "episode_length": self._step_count.clone(),
                "episode_total_angle": self._episode_total_angle.clone(),
                "episode_total_distance": self._episode_total_distance.clone(),
                "episode_area_integral": self._episode_area_integral.clone(),
            },
            batch_size=(B,),
            device=self.device,
        )

    def _set_seed(self, seed: int | None) -> None:
        rng = torch.Generator(device=self.device)
        if seed is not None:
            rng.manual_seed(seed)
        self.rng = rng


def make_sofa_env(
    num_envs: int = 256,
    cfg: SofaEnvConfig = SofaEnvConfig(),
    device: torch.device = DEVICE,
) -> SofaEnv:
    """Factory function for SofaEnv with sensible defaults."""
    return SofaEnv(num_envs=num_envs, cfg=cfg, device=device)
