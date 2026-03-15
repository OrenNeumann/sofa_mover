import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from sofa_mover.corridor import SOFA_CONFIG, TEMPLATE_CONFIG, GridConfig, Pose


def build_affine_matrix(
    pose: Pose,
    sofa_world_size: float,
    template_world_size: float,
) -> Float[Tensor, "B 2 3"]:
    """Build affine matrix mapping sofa-grid normalized coords to template normalized coords.

    The pose (x, y, theta) describes the corridor's position/orientation relative
    to the sofa. The affine matrix computes: for each sofa pixel, where does it
    land in the corridor template?

    Transform chain:
        sofa_norm → sofa_world → (inverse pose) → corridor_world → template_norm

    With same-resolution grids, the scaling factor s = sofa_world_size / template_world_size.
    """
    s = sofa_world_size / template_world_size
    x, y, theta = pose.T
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    # R(-theta): inverse rotation matrix, scaled by s (sofa→template world ratio)
    R = torch.stack(
        [
            torch.stack([s * cos_t, s * sin_t], dim=-1),
            torch.stack([-s * sin_t, s * cos_t], dim=-1),
        ],
        dim=-2,
    )  # (B, 2, 2)

    # Translation: R(-theta) @ (-x, -y), then convert to template normalized coords
    t = torch.stack([x, y], dim=-1)  # (B, 2)
    t_rotated = torch.einsum("bij,bj->bi", R, -t)  # (B, 2)
    t_norm = t_rotated * (2.0 / sofa_world_size)  # world → normalized coords

    affine = torch.cat([R, t_norm.unsqueeze(-1)], dim=-1)  # (B, 2, 3)
    return affine


class Rasterizer:
    """Transforms a pre-rasterized corridor template onto the sofa grid.

    Holds the corridor template and grid configs. Provides corridor_mask (single
    pose) and swept_mask (interpolated between two poses).
    """

    def __init__(
        self,
        template: Float[Tensor, "1 1 Ht Wt"],
        sofa_config: GridConfig = SOFA_CONFIG,
        template_config: GridConfig = TEMPLATE_CONFIG,
    ) -> None:
        self.template = template
        self.sofa_config = sofa_config
        self.template_config = template_config

    def corridor_mask(self, pose: Pose) -> Float[Tensor, "B 1 Hs Ws"]:
        """Compute corridor mask on the sofa grid for a batch of corridor poses.

        Uses affine_grid + grid_sample to efficiently transform the pre-rasterized
        corridor template onto each sofa grid.

        Args:
            pose: Batch of corridor poses (B, 3) — columns are x, y, theta.

        Returns:
            Binary corridor mask (B, 1, Hs, Ws). 1.0 = passable, 0.0 = wall.
        """
        batch_size = pose.shape[0]
        affine = build_affine_matrix(
            pose, self.sofa_config.world_size, self.template_config.world_size
        )

        grid = F.affine_grid(
            affine,
            [batch_size, 1, self.sofa_config.grid_size, self.sofa_config.grid_size],
            align_corners=False,
        )

        template_batch = self.template.expand(batch_size, -1, -1, -1)

        mask = F.grid_sample(
            template_batch,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )

        return mask

    def swept_mask(
        self,
        pose_prev: Pose,
        pose_next: Pose,
        num_substeps: int,
    ) -> Float[Tensor, "B 1 Hs Ws"]:
        """Compute the swept corridor mask between two poses (fused).

        Samples num_substeps intermediate poses (excluding the previous, including
        the next) via linear interpolation, computes all corridor masks in a single
        batched grid_sample call, and returns their element-wise minimum.

        A sofa pixel survives only if it is inside the corridor at every
        intermediate position. With num_substeps=1, this is equivalent to
        corridor_mask at the next pose.

        Uses a single grid_sample with batch size B*K (K=num_substeps) for better
        GPU utilization. See _swept_mask_loop for a memory-lighter alternative that
        processes substeps sequentially.
        """
        if num_substeps < 1:
            raise ValueError(f"num_substeps must be >= 1, got {num_substeps}")

        B = pose_prev.shape[0]
        device = pose_prev.device

        # t values: exclude 0 (previous pose already applied), include 1 (target)
        t_values = torch.linspace(0.0, 1.0, num_substeps + 1, device=device)[1:]  # (K,)

        delta = pose_next - pose_prev  # (B, 3)

        # Build all interpolated poses: (K, B, 3) → (K*B, 3)
        all_poses = pose_prev.unsqueeze(0) + t_values[:, None, None] * delta.unsqueeze(
            0
        )  # (K, B, 3)
        all_poses_flat = all_poses.reshape(-1, 3)  # (K*B, 3)

        # Single batched corridor_mask call
        affine = build_affine_matrix(
            all_poses_flat, self.sofa_config.world_size, self.template_config.world_size
        )
        KB = all_poses_flat.shape[0]
        H, W = self.sofa_config.grid_size, self.sofa_config.grid_size

        grid = F.affine_grid(affine, [KB, 1, H, W], align_corners=False)
        template_batch = self.template.expand(KB, -1, -1, -1)
        masks_flat = F.grid_sample(
            template_batch,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )  # (K*B, 1, H, W)

        # Reshape and take intersection (min) across substeps
        masks = masks_flat.reshape(num_substeps, B, 1, H, W)
        swept, _ = masks.min(dim=0)  # (B, 1, H, W)
        return swept

    def _swept_mask_loop(
        self,
        pose_prev: Pose,
        pose_next: Pose,
        num_substeps: int,
    ) -> Float[Tensor, "B 1 Hs Ws"]:
        """Compute swept corridor mask using a sequential loop over substeps.

        Lighter on memory than the fused swept_mask (allocates one mask at a time
        instead of K masks simultaneously), but slower at low batch numbers due to K sequential
        kernel launches. roughly same speed as other method.
        """
        if num_substeps < 1:
            raise ValueError(f"num_substeps must be >= 1, got {num_substeps}")

        device = pose_prev.device
        t_values = torch.linspace(0.0, 1.0, num_substeps + 1, device=device)[1:]
        delta = pose_next - pose_prev

        swept: Tensor | None = None

        for t in t_values:
            pose_t = pose_prev + t * delta
            mask_t = self.corridor_mask(pose_t)

            if swept is None:
                swept = mask_t
            else:
                torch.min(swept, mask_t, out=swept)

        assert swept is not None
        return swept
