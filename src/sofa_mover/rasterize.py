import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from sofa_mover.corridor import GridConfig, SOFA_CONFIG, TEMPLATE_CONFIG


def build_affine_matrix(
    x: Float[Tensor, " B"],
    y: Float[Tensor, " B"],
    theta: Float[Tensor, " B"],
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


def get_corridor_mask(
    template: Float[Tensor, "1 1 Ht Wt"],
    x: Float[Tensor, " B"],
    y: Float[Tensor, " B"],
    theta: Float[Tensor, " B"],
    sofa_config: GridConfig = SOFA_CONFIG,
    template_config: GridConfig = TEMPLATE_CONFIG,
) -> Float[Tensor, "B 1 Hs Ws"]:
    """Compute corridor mask on the sofa grid for a batch of corridor poses.

    Uses affine_grid + grid_sample to efficiently transform the pre-rasterized
    corridor template onto each sofa grid.

    Args:
        template: Pre-rasterized corridor template (1, 1, Ht, Wt).
        x, y, theta: Batch of corridor poses (B,).
        sofa_config: Grid config for the sofa grid (output).
        template_config: Grid config for the corridor template (input).

    Returns:
        Binary corridor mask (B, 1, Hs, Ws). 1.0 = passable, 0.0 = wall.
    """
    batch_size = x.shape[0]
    device = template.device
    x, y, theta = x.to(device), y.to(device), theta.to(device)
    affine = build_affine_matrix(
        x, y, theta, sofa_config.world_size, template_config.world_size
    )

    grid = F.affine_grid(
        affine,
        [batch_size, 1, sofa_config.grid_size, sofa_config.grid_size],
        align_corners=False,
    )

    # Expand template to batch size
    template_batch = template.expand(batch_size, -1, -1, -1)

    mask = F.grid_sample(
        template_batch,
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )

    return mask
