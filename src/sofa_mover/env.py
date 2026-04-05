"""TorchRL environment for the sofa moving problem."""

import torch
import torch.nn.functional as F
from jaxtyping import Float
from sofa_mover.boundary import BoundaryExtractor
from tensordict import TensorDict, TensorDictBase
from torch import Tensor
from torchrl.data.tensor_specs import Bounded, Composite, OneHot, Unbounded
from torchrl.envs import EnvBase

from sofa_mover.corridor import Pose, make_l_corridor
from sofa_mover.erosion import erode
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

        # Goal point in corridor-centric coords (constant)
        self.goal_corridor: torch.Tensor = torch.tensor(cfg.goal_point, device=device)

        # Initial area and goal distance (single mask computation)
        self.cell_area = (cfg.sofa_config.world_size / H) ** 2
        init_pose = torch.tensor([list(cfg.initial_pose)], device=device)
        # Clip initial sofa to the start region of the vertical leg
        self._start_mask = self._compute_start_mask(init_pose, cfg, device)
        init_mask = self.rasterizer.corridor_mask(init_pose) * self._start_mask

        # Compute crop bounding box from initial sofa (sofa can only shrink)
        self._crop_y, self._crop_x = self._bbox(init_mask)

        # Crop grids for COM calculation
        self.x_grid: torch.Tensor = x_grid[self._crop_y, self._crop_x]
        self.y_grid: torch.Tensor = y_grid[self._crop_y, self._crop_x]

        init_cropped = self._crop(init_mask)
        self.initial_area = init_mask.sum().item() * self.cell_area
        init_com = _sofa_com(init_cropped, self.x_grid, self.y_grid)
        init_goal_sofa = _goal_corridor_to_sofa(self.goal_corridor, init_pose)
        self.initial_goal_dist = (init_com - init_goal_sofa).norm(dim=-1).item()

        # Internal state — cropped to bbox
        crop_h, crop_w = init_cropped.shape[2], init_cropped.shape[3]
        self._sofa = torch.ones(num_envs, 1, crop_h, crop_w, device=device)
        self._pose = torch.zeros(num_envs, 3, device=device)
        self._step_count = torch.zeros(num_envs, 1, dtype=torch.int64, device=device)

        # Episode metric accumulators (reset per episode)
        self._episode_total_angle = torch.zeros(num_envs, 1, device=device)
        self._episode_total_distance = torch.zeros(num_envs, 1, device=device)

        # Boundary ray-casting setup (if enabled)
        if cfg.observation_type == "boundary":
            self._boundary = BoundaryExtractor(
                cfg.boundary_rays, init_cropped[0, 0], device
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
                shape=(B, self.cfg.boundary_rays),
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
                    (self._crop_y.stop - self._crop_y.start) // self.cfg.obs_downscale,
                    (self._crop_x.stop - self._crop_x.start) // self.cfg.obs_downscale,
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

    @staticmethod
    def _compute_start_mask(
        pose: Pose,
        cfg: SofaEnvConfig,
        device: torch.device,
    ) -> Float[Tensor, "1 1 H W"]:
        """Compute a mask clipping sofa pixels to cy <= start_y_max in corridor coords."""
        H = cfg.sofa_config.grid_size
        half = cfg.sofa_config.world_size / 2
        coords = torch.linspace(-half, half, H, device=device)
        y_grid, x_grid = torch.meshgrid(coords, coords, indexing="ij")

        tx, ty, theta = pose[0, 0], pose[0, 1], pose[0, 2]
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        dx = x_grid - tx
        dy = y_grid - ty
        cy = -sin_t * dx + cos_t * dy
        return (cy <= cfg.start_y_max).float().unsqueeze(0).unsqueeze(0)

    @staticmethod
    def _bbox(mask: Float[Tensor, "1 1 H W"]) -> tuple[slice, slice]:
        """Bounding box (y_slice, x_slice) of nonzero pixels."""
        nz = mask[0, 0].nonzero(as_tuple=True)
        y0, y1 = int(nz[0].min()), int(nz[0].max()) + 1
        x0, x1 = int(nz[1].min()), int(nz[1].max()) + 1
        return slice(y0, y1), slice(x0, x1)

    def _crop(self, t: torch.Tensor) -> torch.Tensor:
        """Crop a (B, C, H, W) tensor to the sofa bounding box."""
        return t[:, :, self._crop_y, self._crop_x]

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

            H = self.cfg.sofa_config.grid_size
            fresh_sofa = torch.ones(n_reset, 1, H, H, device=device)
            initial_mask = self.rasterizer.corridor_mask(initial_pose)
            initial_mask = initial_mask * self._start_mask
            fresh_sofa = self._crop(erode(fresh_sofa, initial_mask))

            self._sofa[reset_mask] = fresh_sofa
            self._pose[reset_mask] = initial_pose
            self._step_count[reset_mask] = 0
            self._episode_total_angle[reset_mask] = 0.0
            self._episode_total_distance[reset_mask] = 0.0

        if self.cfg.observation_type == "boundary":
            sofa_view = self._boundary(self._sofa)
        else:  # "grid"
            sofa_view = self._downscale_obs(self._sofa.to(torch.uint8))

        # Progress = 1 - (dist_to_goal / initial_dist)
        com = _sofa_com(self._sofa, self.x_grid, self.y_grid)
        goal_sofa = _goal_corridor_to_sofa(self.goal_corridor, self._pose)
        self._goal_dist = (com - goal_sofa).norm(dim=-1)  # (B,)
        progress = 1.0 - self._goal_dist.unsqueeze(1) / max(
            self.initial_goal_dist, 1e-8
        )

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

        # Swept mask erosion (rasterizer works on full grid, crop result)
        swept, _corridor_at_next = self.rasterizer.swept_mask(
            pose, pose_next, cfg.num_substeps
        )
        new_sofa = erode(self._sofa, self._crop(swept))

        # Area after erosion
        area_after = new_sofa.flatten(1).sum(dim=1) * self.cell_area  # (B,)
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

        if self.cfg.observation_type == "boundary":
            sofa_view = self._boundary(new_sofa)
        else:  # "grid"
            sofa_view = self._downscale_obs(new_sofa.to(torch.uint8))

        # Progress = 1 - (dist_to_goal / initial_dist)
        progress = 1.0 - goal_dist.unsqueeze(1) / max(self.initial_goal_dist, 1e-8)

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
    num_envs: int = 256,
    cfg: SofaEnvConfig = SofaEnvConfig(),
    device: torch.device = DEVICE,
) -> SofaEnv:
    """Factory function for SofaEnv with sensible defaults."""
    return SofaEnv(num_envs=num_envs, cfg=cfg, device=device)
