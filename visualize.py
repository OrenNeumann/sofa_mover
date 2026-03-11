"""Quick visualization of corridor template, rasterized masks, and erosion."""

import math

import torch
import matplotlib.pyplot as plt

from sofa_mover.corridor import (
    make_l_corridor,
    SOFA_CONFIG,
    TEMPLATE_CONFIG,
)
from sofa_mover.rasterize import get_corridor_mask
from sofa_mover.erosion import erode


def main() -> None:
    template = make_l_corridor(config=TEMPLATE_CONFIG)
    sofa = torch.ones(1, 1, SOFA_CONFIG.grid_size, SOFA_CONFIG.grid_size)

    # --- Figure 1: Corridor template ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(
        template[0, 0].numpy(),
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
    mask_identity = get_corridor_mask(
        template,
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1),
        SOFA_CONFIG,
        TEMPLATE_CONFIG,
    )
    axes[1].imshow(
        mask_identity[0, 0].numpy(),
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
    mask_rotated = get_corridor_mask(
        template,
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([math.pi / 4]),
        SOFA_CONFIG,
        TEMPLATE_CONFIG,
    )
    axes[2].imshow(
        mask_rotated[0, 0].numpy(),
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
    plt.savefig("viz_corridor_and_masks.png", dpi=150)
    print("Saved viz_corridor_and_masks.png")

    # --- Figure 2: Erosion walkthrough ---
    trajectory = [
        (0.0, 0.0, 0.0),
        (0.0, -0.3, 0.0),
        (0.0, -0.5, math.pi / 8),
        (0.0, -0.5, math.pi / 4),
        (-0.3, -0.5, math.pi / 4),
        (-0.5, -0.5, math.pi / 3),
    ]

    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    axes2 = axes2.flatten()
    extent = [
        -SOFA_CONFIG.world_size / 2,
        SOFA_CONFIG.world_size / 2,
        -SOFA_CONFIG.world_size / 2,
        SOFA_CONFIG.world_size / 2,
    ]

    for i, (px, py, pt) in enumerate(trajectory):
        mask = get_corridor_mask(
            template,
            torch.tensor([px]),
            torch.tensor([py]),
            torch.tensor([pt]),
            SOFA_CONFIG,
            TEMPLATE_CONFIG,
        )
        sofa = erode(sofa, mask)
        area_pixels = sofa.sum().item()
        area_world = area_pixels * (SOFA_CONFIG.world_size / SOFA_CONFIG.grid_size) ** 2

        axes2[i].imshow(sofa[0, 0].numpy(), origin="lower", extent=extent, cmap="Blues")
        axes2[i].set_title(
            f"Step {i}: pose=({px:.1f}, {py:.1f}, {pt:.2f})\narea={area_world:.2f}"
        )
        axes2[i].set_xlabel("x")
        axes2[i].set_ylabel("y")
        axes2[i].set_aspect("equal")

    plt.suptitle("Sofa Erosion Walkthrough", fontsize=14)
    plt.tight_layout()
    plt.savefig("viz_erosion_walkthrough.png", dpi=150)
    print("Saved viz_erosion_walkthrough.png")


if __name__ == "__main__":
    main()
