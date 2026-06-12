"""Quick visualization of corridor masks and erosion."""

import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from sofa_mover.corridor import make_l_corridor
from sofa_mover.rasterize import Rasterizer
from sofa_mover.training.config import DEVICE, GridConfig, SOFA_CONFIG
from sofa_mover.visualization.render import build_composite


def _show_panel(
    ax: Axes,
    sofa_img: np.ndarray,
    mask_img: np.ndarray,
    pose: tuple[float, float, float],
    world_size: float,
    title: str,
) -> None:
    """Draw one sofa/corridor composite, with the floor texture posed in world space."""
    half = world_size / 2
    coords = np.linspace(-half, half, mask_img.shape[0], dtype=np.float32)
    xs, ys = np.meshgrid(coords, coords, indexing="xy")
    px, py, pt = pose
    cos_t, sin_t = np.float32(np.cos(pt)), np.float32(np.sin(pt))
    world_x = cos_t * xs - sin_t * ys + np.float32(px)
    world_y = sin_t * xs + cos_t * ys + np.float32(py)
    ax.imshow(
        build_composite(sofa_img, mask_img, world_x=world_x, world_y=world_y),
        origin="lower",
        extent=(-half, half, -half, half),
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")


def main(
    device: torch.device = DEVICE,
    sofa_config: GridConfig = SOFA_CONFIG,
) -> None:
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    geometry = make_l_corridor()
    rasterizer = Rasterizer(geometry, sofa_config, device=device)

    def mask_at(r: Rasterizer, pose: tuple[float, float, float]) -> np.ndarray:
        return r.corridor_mask(torch.tensor([pose], device=device))[0, 0].cpu().numpy()

    # --- Figure 1: Corridor masks at various poses ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    overview_config = GridConfig(grid_size=256, world_size=6.0)
    overview_rasterizer = Rasterizer(geometry, overview_config, device=device)
    identity = (0.0, 0.0, 0.0)
    rotated = (0.0, 0.0, math.pi / 4)

    overview_mask = mask_at(overview_rasterizer, identity)
    _show_panel(
        axes[0],
        np.zeros_like(overview_mask),
        overview_mask,
        identity,
        overview_config.world_size,
        "L-Corridor at identity (6x6 world units)",
    )
    identity_mask = mask_at(rasterizer, identity)
    _show_panel(
        axes[1],
        np.zeros_like(identity_mask),
        identity_mask,
        identity,
        sofa_config.world_size,
        "Corridor Mask at (0, 0, 0)\n(sofa's 3x3 view)",
    )
    rotated_mask = mask_at(rasterizer, rotated)
    _show_panel(
        axes[2],
        np.zeros_like(rotated_mask),
        rotated_mask,
        rotated,
        sofa_config.world_size,
        "Corridor Mask at (0, 0, π/4)\n(45° rotation)",
    )

    plt.tight_layout()
    path1 = output_dir / "viz_corridor_and_masks.png"
    plt.savefig(path1, dpi=150)
    print(f"Saved {path1}")

    # --- Figure 2: Erosion walkthrough ---
    trajectory: list[tuple[float, float, float]] = [
        (0.0, 0.0, 0.0),
        (0.0, -0.3, 0.0),
        (0.0, -0.5, math.pi / 8),
        (0.0, -0.5, math.pi / 4),
        (-0.3, -0.5, math.pi / 4),
        (-0.5, -0.5, math.pi / 3),
    ]

    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    sofa = torch.ones(
        1,
        1,
        sofa_config.grid_size,
        sofa_config.grid_size,
        dtype=torch.bool,
        device=device,
    )
    cell_area = (sofa_config.world_size / sofa_config.grid_size) ** 2
    for i, pose in enumerate(trajectory):
        mask = rasterizer.corridor_mask(torch.tensor([pose], device=device))
        sofa = sofa & mask
        area_world = sofa.sum().item() * cell_area
        px, py, pt = pose
        _show_panel(
            axes2.flatten()[i],
            sofa[0, 0].float().cpu().numpy(),
            mask[0, 0].float().cpu().numpy(),
            pose,
            sofa_config.world_size,
            f"Step {i}: pose=({px:.1f}, {py:.1f}, {pt:.2f})\narea={area_world:.2f}",
        )

    plt.suptitle("Sofa Erosion Walkthrough", fontsize=14)
    plt.tight_layout()
    path2 = output_dir / "viz_erosion_walkthrough.png"
    plt.savefig(path2, dpi=150)
    print(f"Saved {path2}")


if __name__ == "__main__":
    main()
