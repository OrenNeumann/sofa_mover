"""TorchRL environment for the sofa moving problem."""

import torch
import torch.nn.functional as F
from jaxtyping import Float
from sofa_mover.boundary import BoundaryExtractor
from tensordict import TensorDict, TensorDictBase
from torch import Tensor
from torchrl.data.tensor_specs import Bounded, Composite, MultiOneHot, Unbounded
from torchrl.envs import EnvBase

from sofa_mover.corridor import Pose, make_l_corridor
from sofa_mover.rasterize import Rasterizer
from sofa_mover.training.config import DEVICE, SofaEnvConfig


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
    mass = sofa[:, 0].float()  # (B, H, W)
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
        total_frames: int,
        cfg: SofaEnvConfig = SofaEnvConfig(),
        device: torch.device = DEVICE,
    ) -> None:
        super().__init__(device=device, batch_size=torch.Size([num_envs]))
        self.cfg = cfg
        self.num_envs = num_envs
        self.steps_count = 0
        self._anneal_end = total_frames * cfg.reward_anneal_time

        # MultiDiscrete action space: each of (dx, dy, dθ) is chosen independently
        # from n_bins = 2*n_magnitude_levels+1 options: {-n·δ, …, 0, …, +n·δ}.
        n_mag = cfg.n_magnitude_levels
        n_bins = 2 * n_mag + 1
        self.n_bins = n_bins
        # Build as [-n·δ, ..., -δ, 0, +δ, ..., +n·δ] with exact 0 at center.
        xy_pos = (
            torch.arange(1, n_mag + 1, dtype=torch.float32, device=device)
            * cfg.delta_xy
        )
        theta_pos = (
            torch.arange(1, n_mag + 1, dtype=torch.float32, device=device)
            * cfg.delta_theta
        )
        self.xy_action_vals = torch.cat(
            [-xy_pos.flip(0), torch.zeros(1, device=device), xy_pos]
        )  # (n_bins,)
        self.theta_action_vals = torch.cat(
            [-theta_pos.flip(0), torch.zeros(1, device=device), theta_pos]
        )  # (n_bins,)

        # Goal point in corridor-centric coords (constant)
        self.goal_corridor: torch.Tensor = torch.tensor(cfg.goal_point, device=device)

        # Build sofa coordinate grids directly in corridor coords.
        # The sofa frame = corridor frame at t=0 (initial_pose = 0,0,0).
        ppu = cfg.sofa_config.pixels_per_unit
        grid_w = round(cfg.corridor_width * ppu)
        grid_h = round(cfg.sofa_length * ppu)
        self.cell_area = (1.0 / ppu) ** 2

        hw = cfg.corridor_width / 2
        front_y = cfg.start_y_max
        back_y = front_y - cfg.sofa_length

        x_coords = torch.linspace(-hw, hw, grid_w, device=device)
        y_coords = torch.linspace(back_y, front_y, grid_h, device=device)
        self.y_grid, self.x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Build corridor geometry and rasterizer (with the sofa's cropped grid)
        geometry = make_l_corridor(corridor_width=cfg.corridor_width)
        self.rasterizer = Rasterizer(
            geometry,
            cfg.sofa_config,
            device=device,
            x_grid=self.x_grid,
            y_grid=self.y_grid,
        )

        # World-coordinate extent of the sofa grid (for rendering)
        self.sofa_extent: tuple[float, float, float, float] = (
            -hw,
            hw,
            back_y,
            front_y,
        )

        init_pose = torch.tensor([list(cfg.initial_pose)], device=device)
        init_sofa = self.rasterizer.corridor_mask(init_pose)
        self.initial_area = init_sofa.sum() * self.cell_area
        init_com = _sofa_com(init_sofa, self.x_grid, self.y_grid)
        init_goal_sofa = _goal_corridor_to_sofa(self.goal_corridor, init_pose)
        self.initial_goal_dist = (init_com - init_goal_sofa).norm(dim=-1)

        # Internal state
        self._sofa = torch.ones(
            num_envs, 1, grid_h, grid_w, dtype=torch.bool, device=device
        )
        self._pose = torch.zeros(num_envs, 3, device=device)
        self._step_count = torch.zeros(num_envs, 1, dtype=torch.int64, device=device)

        # Episode metric accumulators (reset per episode)
        self._episode_total_angle = torch.zeros(num_envs, 1, device=device)
        self._episode_total_distance = torch.zeros(num_envs, 1, device=device)

        # Boundary ray-casting setup (if enabled)
        if cfg.observation_type == "boundary":
            self._boundary = BoundaryExtractor(
                cfg.boundary_rays, init_sofa[0, 0], device
            )

        self._make_specs()

    def _downscale_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Downscale observation if obs_downscale > 1."""
        if self.cfg.obs_downscale <= 1:
            return obs
        return (
            F.avg_pool2d(obs.float(), kernel_size=self.cfg.obs_downscale)
            .round()
            .to(torch.uint8)
        )

    def _make_specs(self) -> None:
        B = self.num_envs

        _unbounded_b1 = dict(shape=(B, 1), dtype=torch.float32, device=self.device)
        if self.cfg.observation_type == "boundary":
            obs_spec = Bounded(
                low=0.0,
                high=1.0,
                shape=(B, 2 * self.cfg.boundary_rays),
                dtype=torch.float32,
                device=self.device,
            )
        else:  # "grid"
            obs_spec = Bounded(
                low=0,
                high=1,
                shape=(
                    B,
                    1,
                    self._sofa.shape[2] // self.cfg.obs_downscale,
                    self._sofa.shape[3] // self.cfg.obs_downscale,
                ),
                dtype=torch.uint8,
                device=self.device,
            )
        observation_bundle_spec = Composite(
            sofa_view=obs_spec,
            pose=Unbounded(shape=(B, 3), dtype=torch.float32, device=self.device),
            progress=Unbounded(**_unbounded_b1),
            shape=(B,),
        )
        self.observation_spec = Composite(
            observation=observation_bundle_spec,
            terminal_area=Unbounded(**_unbounded_b1),
            episode_length=Unbounded(
                shape=(B, 1), dtype=torch.int64, device=self.device
            ),
            episode_total_angle=Unbounded(**_unbounded_b1),
            episode_total_distance=Unbounded(**_unbounded_b1),
            reward_erosion=Unbounded(**_unbounded_b1),
            reward_progress=Unbounded(**_unbounded_b1),
            reward_terminal=Unbounded(**_unbounded_b1),
            shape=(B,),
        )

        n_bins = self.n_bins
        self.action_spec = MultiOneHot(
            nvec=[n_bins, n_bins, n_bins],
            shape=(B, 3 * n_bins),
            dtype=torch.float32,
            device=self.device,
        )

        self.reward_spec = Unbounded(
            shape=(B, 1),
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def shaping_scale(self) -> float:
        return max(0.0, 1.0 - self.steps_count / self._anneal_end)

    def _reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        B = self.num_envs
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

            h, w = self._sofa.shape[2], self._sofa.shape[3]
            fresh_sofa = torch.ones(n_reset, 1, h, w, dtype=torch.bool, device=device)
            assert self.rasterizer.corridor_mask(
                initial_pose
            ).all(), "Initial pose should place sofa fully inside the corridor"

            self._sofa[reset_mask] = fresh_sofa
            self._pose[reset_mask] = initial_pose
            self._step_count[reset_mask] = 0
            self._episode_total_angle[reset_mask] = 0.0
            self._episode_total_distance[reset_mask] = 0.0

        if self.cfg.observation_type == "boundary":
            # TODO: can we get rid of this op?
            corridor = self.rasterizer.corridor_mask(self._pose)
            sofa_view = self._boundary(self._sofa, corridor)
        else:  # "grid"
            sofa_view = self._downscale_obs(self._sofa.to(torch.uint8))

        # Progress = 1 - (dist_to_goal / initial_dist)
        com = _sofa_com(self._sofa, self.x_grid, self.y_grid)
        goal_sofa = _goal_corridor_to_sofa(self.goal_corridor, self._pose)
        self._goal_dist = (com - goal_sofa).norm(dim=-1)  # (B,)
        progress = 1.0 - self._goal_dist.unsqueeze(1) / self.initial_goal_dist

        return TensorDict(
            {
                "observation": {
                    "sofa_view": sofa_view,
                    "pose": self._pose.clone(),
                    "progress": progress,
                },
                "done": torch.zeros(B, 1, dtype=torch.bool, device=device),
                "terminated": torch.zeros(B, 1, dtype=torch.bool, device=device),
                "truncated": torch.zeros(B, 1, dtype=torch.bool, device=device),
                "terminal_area": torch.zeros(B, 1, device=device),
                "episode_length": torch.zeros(B, 1, dtype=torch.int64, device=device),
                "episode_total_angle": torch.zeros(B, 1, device=device),
                "episode_total_distance": torch.zeros(B, 1, device=device),
                "reward_erosion": torch.zeros(B, 1, device=device),
                "reward_progress": torch.zeros(B, 1, device=device),
                "reward_terminal": torch.zeros(B, 1, device=device),
            },
            batch_size=(B,),
            device=device,
        )

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        cfg = self.cfg
        B = self.num_envs

        action = tensordict["action"]  # (B, 3*n_bins) concatenated one-hot

        # Decode one-hots action: argmax each of (dx,dy,dtheta)
        dx_oh, dy_oh, dt_oh = action.split(self.n_bins, dim=-1)
        dx = self.xy_action_vals[dx_oh.argmax(dim=-1)]  # (B,)
        dy = self.xy_action_vals[dy_oh.argmax(dim=-1)]  # (B,)
        dtheta = self.theta_action_vals[dt_oh.argmax(dim=-1)]  # (B,)
        delta = torch.stack([dx, dy, dtheta], dim=-1)  # (B, 3)

        # Accumulate episode metrics
        dx, dy, dtheta = delta[:, 0], delta[:, 1], delta[:, 2]
        self._episode_total_distance += torch.sqrt(dx * dx + dy * dy).unsqueeze(1)
        self._episode_total_angle += dtheta.abs().unsqueeze(1)

        pose = self._pose
        pose_next = pose + delta

        # Area before erosion
        area_before = (
            self._sofa.flatten(1).sum(dim=1, dtype=torch.float32) * self.cell_area
        )  # (B,)

        # Swept mask erosion (rasterizer already operates on the cropped grid)
        swept, corridor_at_next = self.rasterizer.swept_mask(
            pose, pose_next, cfg.num_substeps
        )
        # Erosion: sofa ∩ corridor removes pixels outside the corridor
        new_sofa = self._sofa & swept

        # Area after erosion
        area_after = (
            new_sofa.flatten(1).sum(dim=1, dtype=torch.float32) * self.cell_area
        )  # (B,)
        # Goal check: sofa COM close to goal point
        com = _sofa_com(new_sofa, self.x_grid, self.y_grid)  # (B, 2)
        goal_sofa = _goal_corridor_to_sofa(self.goal_corridor, pose_next)  # (B, 2)
        goal_dist = (com - goal_sofa).norm(dim=-1)  # (B,)
        goal_reached = goal_dist < cfg.goal_radius  # (B,)

        # Step count
        self._step_count += 1

        # Done conditions
        area_dead = (area_after / self.initial_area) < cfg.min_area_fraction
        truncated = self._step_count.squeeze(1) >= cfg.max_steps
        terminated = goal_reached | area_dead
        done = terminated | truncated

        # --- Reward ---
        area_lost = (area_before - area_after).clamp(min=0.0)  # (B,)
        # penalty for losing area fraction, normalized by initial_area
        erosion_penalty = -cfg.lambda_erosion * area_lost / self.initial_area
        # reward for getting closer to goal
        progress_bonus = (
            cfg.lambda_progress
            * (self._goal_dist - goal_dist)
            / self.initial_goal_dist
            * self.shaping_scale
        )
        # per-step area survival bonus: dense feedback about current area level
        area_step_bonus = cfg.lambda_area_step * area_after / self.initial_area
        # TODO: why does cubic reward work so well??
        terminal_bonus = goal_reached.float() * area_after**3
        reward = (
            erosion_penalty + progress_bonus + area_step_bonus + terminal_bonus
        ).unsqueeze(1)

        # Update internal state
        self.steps_count += B
        self._sofa = new_sofa
        self._pose = pose_next
        self._goal_dist = goal_dist

        if self.cfg.observation_type == "boundary":
            sofa_view = self._boundary(new_sofa, corridor_at_next)
        else:  # "grid"
            sofa_view = self._downscale_obs(new_sofa.to(torch.uint8))

        # Progress = 1 - (dist_to_goal / initial_dist)
        progress = 1.0 - goal_dist.unsqueeze(1) / self.initial_goal_dist

        return TensorDict(
            {
                "observation": {
                    "sofa_view": sofa_view,
                    "pose": self._pose.clone(),
                    "progress": progress,
                },
                "reward": reward,
                "done": done.unsqueeze(1),
                "terminated": terminated.unsqueeze(1),
                "truncated": truncated.unsqueeze(1),
                "terminal_area": (goal_reached.float() * area_after).unsqueeze(1),
                "episode_length": self._step_count.clone(),
                "episode_total_angle": self._episode_total_angle.clone(),
                "episode_total_distance": self._episode_total_distance.clone(),
                "reward_erosion": erosion_penalty.unsqueeze(1),
                "reward_progress": progress_bonus.unsqueeze(1),
                "reward_terminal": terminal_bonus.unsqueeze(1),
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
    total_frames: int,
    num_envs: int = 256,
    cfg: SofaEnvConfig = SofaEnvConfig(),
    device: torch.device = DEVICE,
) -> SofaEnv:
    """Factory function for SofaEnv with sensible defaults."""
    return SofaEnv(
        num_envs=num_envs,
        total_frames=total_frames,
        cfg=cfg,
        device=device,
    )
