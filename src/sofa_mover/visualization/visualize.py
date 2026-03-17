"""Quick visualization of corridor template, rasterized masks, and erosion."""

import math
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from sofa_mover.corridor import (
    DEVICE,
    make_l_corridor,
    SOFA_CONFIG,
    TEMPLATE_CONFIG,
)
from sofa_mover.rasterize import Rasterizer
from sofa_mover.erosion import erode


def main() -> None:
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    template = make_l_corridor(config=TEMPLATE_CONFIG)
    rasterizer = Rasterizer(template, SOFA_CONFIG, TEMPLATE_CONFIG)
    sofa = torch.ones(1, 1, SOFA_CONFIG.grid_size, SOFA_CONFIG.grid_size, device=DEVICE)

    # --- Figure 1: Corridor template ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(
        template[0, 0].cpu().numpy(),
        origin="lower",
        extent=[
            -TEMPLATE_CONFIG.world_size / 2,
            TEMPLATE_CONFIG.world_size / 2,
            -TEMPLATE_CONFIG.world_size / 2,
            TEMPLATE_CONFIG.world_size / 2,
        ],
        cmap="Greys_r",
    )
    axes[0].set_title("L-Corridor Template (6x6 world units)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")

    # --- Figure 2: Mask at identity pose ---
    mask_identity = rasterizer.corridor_mask(
        torch.tensor([[0.0, 0.0, 0.0]], device=DEVICE)
    )
    axes[1].imshow(
        mask_identity[0, 0].cpu().numpy(),
        origin="lower",
        extent=[
            -SOFA_CONFIG.world_size / 2,
            SOFA_CONFIG.world_size / 2,
            -SOFA_CONFIG.world_size / 2,
            SOFA_CONFIG.world_size / 2,
        ],
        cmap="Greys_r",
    )
    axes[1].set_title("Corridor Mask at (0, 0, 0)\n(sofa's 3x3 view)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")

    # --- Figure 3: Mask at rotated pose ---
    mask_rotated = rasterizer.corridor_mask(
        torch.tensor([[0.0, 0.0, math.pi / 4]], device=DEVICE)
    )
    axes[2].imshow(
        mask_rotated[0, 0].cpu().numpy(),
        origin="lower",
        extent=[
            -SOFA_CONFIG.world_size / 2,
            SOFA_CONFIG.world_size / 2,
            -SOFA_CONFIG.world_size / 2,
            SOFA_CONFIG.world_size / 2,
        ],
        cmap="Greys_r",
    )
    axes[2].set_title("Corridor Mask at (0, 0, π/4)\n(45° rotation)")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_aspect("equal")

    plt.tight_layout()
    path1 = output_dir / "viz_corridor_and_masks.png"
    plt.savefig(path1, dpi=150)
    print(f"Saved {path1}")

    # --- Figure 2: Erosion walkthrough ---
    trajectory = [
        [0.0, 0.0, 0.0],
        [0.0, -0.3, 0.0],
        [0.0, -0.5, math.pi / 8],
        [0.0, -0.5, math.pi / 4],
        [-0.3, -0.5, math.pi / 4],
        [-0.5, -0.5, math.pi / 3],
    ]

    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    axes2 = axes2.flatten()
    extent = [
        -SOFA_CONFIG.world_size / 2,
        SOFA_CONFIG.world_size / 2,
        -SOFA_CONFIG.world_size / 2,
        SOFA_CONFIG.world_size / 2,
    ]

    for i, pose_vals in enumerate(trajectory):
        mask = rasterizer.corridor_mask(torch.tensor([pose_vals], device=DEVICE))
        sofa = erode(sofa, mask)
        area_pixels = sofa.sum().item()
        area_world = area_pixels * (SOFA_CONFIG.world_size / SOFA_CONFIG.grid_size) ** 2

        px, py, pt = pose_vals
        axes2[i].imshow(
            sofa[0, 0].cpu().numpy(), origin="lower", extent=extent, cmap="Blues"
        )
        axes2[i].set_title(
            f"Step {i}: pose=({px:.1f}, {py:.1f}, {pt:.2f})\narea={area_world:.2f}"
        )
        axes2[i].set_xlabel("x")
        axes2[i].set_ylabel("y")
        axes2[i].set_aspect("equal")

    plt.suptitle("Sofa Erosion Walkthrough", fontsize=14)
    plt.tight_layout()
    path2 = output_dir / "viz_erosion_walkthrough.png"
    plt.savefig(path2, dpi=150)
    print(f"Saved {path2}")


if __name__ == "__main__":
    main()
