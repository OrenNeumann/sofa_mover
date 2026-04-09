import torch
from jaxtyping import Float, Bool
from torch import Tensor

from sofa_mover.corridor import CorridorGeometry, Pose
from sofa_mover.training.config import SOFA_CONFIG, GridConfig


def _analytical_corridor_mask(
    pose: Pose,
    x_grid: Float[Tensor, "H W"],
    y_grid: Float[Tensor, "H W"],
    rect_bounds: Float[Tensor, "R 4"],
) -> Bool[Tensor, "B 1 H W"]:
    """Compute corridor mask analytically via per-pixel rectangle membership.

    For each sofa pixel, transforms to corridor-local frame and checks if the
    point falls inside any of the R rectangles.
    """
    tx, ty, theta = pose[:, 0], pose[:, 1], pose[:, 2]
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    # Transform sofa pixels to corridor-local frame: R(-theta) @ (pixel - t)
    dx = x_grid.unsqueeze(0) - tx[:, None, None]  # (B, H, W)
    dy = y_grid.unsqueeze(0) - ty[:, None, None]  # (B, H, W)
    cx = cos_t[:, None, None] * dx + sin_t[:, None, None] * dy  # (B, H, W)
    cy = -sin_t[:, None, None] * dx + cos_t[:, None, None] * dy  # (B, H, W)

    # Check membership in each rectangle: rect_bounds is (R, 4)
    x_min = rect_bounds[:, 0].reshape(-1, 1, 1, 1)  # (R, 1, 1, 1)
    y_min = rect_bounds[:, 1].reshape(-1, 1, 1, 1)
    x_max = rect_bounds[:, 2].reshape(-1, 1, 1, 1)
    y_max = rect_bounds[:, 3].reshape(-1, 1, 1, 1)

    cx_r = cx.unsqueeze(0)  # (1, B, H, W)
    cy_r = cy.unsqueeze(0)  # (1, B, H, W)

    in_rect = (
        (cx_r >= x_min) & (cx_r <= x_max) & (cy_r >= y_min) & (cy_r <= y_max)
    )  # (R, B, H, W)
    in_any = in_rect.any(dim=0)  # (B, H, W)

    return in_any.unsqueeze(1)


def _analytical_swept_mask(
    pose_prev: Pose,
    pose_next: Pose,
    num_substeps: int,
    x_grid: Float[Tensor, "H W"],
    y_grid: Float[Tensor, "H W"],
    rect_bounds: Float[Tensor, "R 4"],
) -> tuple[Float[Tensor, "B 1 H W"], Float[Tensor, "B 1 H W"]]:
    """Compute swept corridor mask between two poses via analytical check.

    Samples num_substeps intermediate poses (excluding previous, including next),
    computes corridor masks analytically, and returns their element-wise minimum.
    """
    B = pose_prev.shape[0]
    device = pose_prev.device
    H, W = x_grid.shape

    # t values: exclude 0 (previous pose already applied), include 1 (target)
    t_values = torch.linspace(0.0, 1.0, num_substeps + 1, device=device)[1:]

    delta = pose_next - pose_prev
    all_poses = pose_prev.unsqueeze(0) + t_values[:, None, None] * delta.unsqueeze(0)
    all_poses_flat = all_poses.reshape(-1, 3)

    masks_flat = _analytical_corridor_mask(all_poses_flat, x_grid, y_grid, rect_bounds)
    masks = masks_flat.reshape(num_substeps, B, 1, H, W)

    swept = masks.all(dim=0)  # pixel kept iff inside corridor at every substep
    corridor_at_next = masks[-1]
    return swept, corridor_at_next


class Rasterizer:
    """Computes corridor masks on the sofa grid using analytical geometry.

    Given a corridor defined as a union of rectangles, checks per-pixel membership
    by transforming sofa-grid coordinates to corridor-local frame and testing
    against rectangle bounds.
    """

    def __init__(
        self,
        geometry: CorridorGeometry,
        sofa_config: GridConfig = SOFA_CONFIG,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        self.geometry = geometry
        self.sofa_config = sofa_config

        half = sofa_config.world_size / 2
        H = sofa_config.grid_size
        coords = torch.linspace(-half, half, H, device=device)
        y_grid, x_grid = torch.meshgrid(coords, coords, indexing="ij")
        self._x_grid: Tensor = x_grid
        self._y_grid: Tensor = y_grid
        self._rect_bounds: Tensor = geometry.to_tensor(device)
        self._corridor_mask_fn = torch.compile(_analytical_corridor_mask)
        self._swept_mask_fn = torch.compile(_analytical_swept_mask)

    @property
    def device(self) -> torch.device:
        return self._x_grid.device

    def set_grids(self, x_grid: Tensor, y_grid: Tensor) -> None:
        """Replace the internal coordinate grids (e.g. with a cropped subset).

        All subsequent corridor_mask / swept_mask calls will operate on the
        new grids and return tensors of the corresponding spatial size.
        """
        self._x_grid = x_grid
        self._y_grid = y_grid

    def corridor_mask(self, pose: Pose) -> Float[Tensor, "B 1 Hs Ws"]:
        """Compute corridor mask on the sofa grid for a batch of corridor poses.

        Args:
            pose: Batch of corridor poses (B, 3) — columns are x, y, theta.

        Returns:
            Binary corridor mask (B, 1, Hs, Ws). 1.0 = passable, 0.0 = wall.
        """
        return self._corridor_mask_fn(
            pose, self._x_grid, self._y_grid, self._rect_bounds
        )

    def swept_mask(
        self,
        pose_prev: Pose,
        pose_next: Pose,
        num_substeps: int,
    ) -> tuple[Float[Tensor, "B 1 Hs Ws"], Float[Tensor, "B 1 Hs Ws"]]:
        """Compute the swept corridor mask between two poses.

        Samples num_substeps intermediate poses (excluding the previous, including
        the next) via linear interpolation, computes all corridor masks, and
        returns their element-wise minimum.

        A sofa pixel survives only if it is inside the corridor at every
        intermediate position.

        Returns:
            (swept_mask, corridor_mask_at_next_pose).
        """
        return self._swept_mask_fn(
            pose_prev,
            pose_next,
            num_substeps,
            self._x_grid,
            self._y_grid,
            self._rect_bounds,
        )
