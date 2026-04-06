"""Generate a video showing step-by-step sofa erosion through an L-corridor."""

import math
from pathlib import Path

import torch

from sofa_mover.corridor import make_l_corridor
from sofa_mover.erosion import erode
from sofa_mover.rasterize import Rasterizer
from sofa_mover.training.config import DEVICE, GridConfig, SOFA_CONFIG
from sofa_mover.visualization.render import (
    FrameData,
    compute_frame_data,
    render_trajectory,
)


def make_l_bend_trajectory(
    num_frames_per_segment: list[int],
) -> list[list[float]]:
    """Generate a smooth trajectory of corridor poses navigating the L-bend.

    Defines waypoints at key phases of the maneuver and linearly interpolates
    between them. Returns a list of [x, y, theta] poses.
    """
    waypoints: list[list[float]] = [
        [0.0, 0.0, 0.0],  # Start: corridor centered on bend
        [0.0, -0.8, 0.0],  # Translate down vertical leg
        [0.0, -1.0, math.pi / 6],  # Begin rotation
        [-0.2, -1.0, math.pi / 3],  # Mid rotation
        [-0.5, -0.8, math.pi / 2],  # Complete 90-deg turn
        [-1.2, -0.5, math.pi / 2],  # Into horizontal leg
        [-1.8, -0.3, math.pi / 2],  # End
    ]

    if len(num_frames_per_segment) != len(waypoints) - 1:
        raise ValueError(
            f"Expected {len(waypoints) - 1} segment counts, "
            f"got {len(num_frames_per_segment)}"
        )

    trajectory: list[list[float]] = []
    for seg_idx in range(len(waypoints) - 1):
        start = waypoints[seg_idx]
        end = waypoints[seg_idx + 1]
        n = num_frames_per_segment[seg_idx]
        for i in range(n):
            t = i / n
            pose = [s + t * (e - s) for s, e in zip(start, end)]
            trajectory.append(pose)
    trajectory.append(waypoints[-1])
    return trajectory


def main(
    device: torch.device = DEVICE,
    sofa_config: GridConfig = SOFA_CONFIG,
) -> None:
    geometry = make_l_corridor()
    rasterizer = Rasterizer(geometry, sofa_config, device=device)

    trajectory = make_l_bend_trajectory([15, 12, 12, 12, 15, 4])
    sofa = torch.ones(1, 1, sofa_config.grid_size, sofa_config.grid_size, device=device)

    frames: list[FrameData] = []
    for step, pose_vals in enumerate(trajectory):
        pose = torch.tensor([pose_vals], device=device)
        mask = rasterizer.corridor_mask(pose)
        sofa = erode(sofa, mask)
        frames.append(
            compute_frame_data(
                step,
                (pose_vals[0], pose_vals[1], pose_vals[2]),
                sofa,
                mask,
                sofa_config,
            )
        )

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "sofa_erosion.gif"
    written = render_trajectory(frames, sofa_config, output_path, fps=10)
    print(f"Saved {written}")


if __name__ == "__main__":
    main()
