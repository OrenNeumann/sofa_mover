"""Rendering utilities for sofa erosion trajectory videos."""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor

from sofa_mover.corridor import GridConfig


@dataclass(frozen=True)
class FrameData:
    """All data needed to render a single animation frame."""

    step: int
    pose: tuple[float, float, float]
    sofa: NDArray[np.float32]
    corridor_mask: NDArray[np.float32]
    area: float


def compute_frame_data(
    step: int,
    pose: tuple[float, float, float],
    sofa: Float[Tensor, "1 1 H W"],
    corridor_mask: Float[Tensor, "1 1 H W"],
    sofa_config: GridConfig,
) -> FrameData:
    """Package tensor data into a FrameData for rendering.

    Converts GPU tensors to CPU numpy and computes sofa area in world units.
    """
    cell_area = (sofa_config.world_size / sofa_config.grid_size) ** 2
    area = sofa.sum().item() * cell_area
    return FrameData(
        step=step,
        pose=pose,
        sofa=sofa[0, 0].cpu().numpy(),
        corridor_mask=corridor_mask[0, 0].cpu().numpy(),
        area=area,
    )


def build_composite(
    sofa: NDArray[np.float32],
    corridor_mask: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Build an (H, W, 3) RGB image from sofa and corridor mask arrays.

    Wall (mask=0): dark gray. Corridor without sofa: light gray. Sofa: blue.
    """
    H, W = sofa.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    wall = corridor_mask == 0
    corridor_empty = (corridor_mask == 1) & (sofa == 0)
    sofa_present = (corridor_mask == 1) & (sofa == 1)

    DARK_GRAY = [0.2, 0.2, 0.2]
    LIGHT_GRAY = [0.8, 0.8, 0.8]
    STEEL_BLUE = [0.27, 0.47, 0.73]

    rgb[wall] = DARK_GRAY
    rgb[corridor_empty] = LIGHT_GRAY
    rgb[sofa_present] = STEEL_BLUE

    return rgb


def render_trajectory(
    frames: Sequence[FrameData],
    sofa_config: GridConfig,
    output_path: Path,
    fps: int = 10,
) -> Path:
    """Render a sequence of frames into an animated gif.

    Produces a two-panel animation: composite sofa/corridor image on the left,
    area-over-time plot on the right.
    """
    fig, (ax_img, ax_plot) = plt.subplots(1, 2, figsize=(12, 5))

    half = sofa_config.world_size / 2
    extent = [-half, half, -half, half]

    # Initialize image panel
    composite = build_composite(frames[0].sofa, frames[0].corridor_mask)
    img_artist = ax_img.imshow(composite, origin="lower", extent=extent)
    ax_img.set_xlabel("x")
    ax_img.set_ylabel("y")
    ax_img.set_aspect("equal")
    title = ax_img.set_title("")

    # Initialize area plot
    initial_area = frames[0].area
    (line,) = ax_plot.plot([], [], "b-", linewidth=2)
    (dot,) = ax_plot.plot([], [], "bo", markersize=8)
    ax_plot.set_xlim(0, max(len(frames) - 1, 1))
    ax_plot.set_ylim(0, initial_area * 1.05)
    ax_plot.set_xlabel("Step")
    ax_plot.set_ylabel("Area (world units\u00b2)")
    ax_plot.set_title("Sofa Area Over Time")
    ax_plot.grid(True, alpha=0.3)

    fig.tight_layout()

    def update(frame_idx: int) -> tuple[object, ...]:
        fd = frames[frame_idx]
        comp = build_composite(fd.sofa, fd.corridor_mask)
        img_artist.set_data(comp)
        title.set_text(
            f"Step {fd.step}: pose=({fd.pose[0]:.2f}, {fd.pose[1]:.2f}, {fd.pose[2]:.2f})\n"
            f"area={fd.area:.3f}"
        )
        steps = [frames[i].step for i in range(frame_idx + 1)]
        areas = [frames[i].area for i in range(frame_idx + 1)]
        line.set_data(steps, areas)
        dot.set_data([fd.step], [fd.area])
        return (img_artist, title, line, dot)

    anim = matplotlib.animation.FuncAnimation(
        fig, update, frames=len(frames), blit=False, interval=1000 // fps
    )

    gif_path = output_path.with_suffix(".gif")
    writer = matplotlib.animation.PillowWriter(fps=fps)
    anim.save(str(gif_path), writer=writer)

    plt.close(fig)
    return gif_path
