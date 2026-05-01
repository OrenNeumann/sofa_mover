"""Render tests cover only what visual inspection of the GIF would miss.
Visible regressions (colors, alignment, rotation, glow, layout) are caught by
eye when reviewing the output, so we don't pin them here."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from sofa_mover.training.config import GridConfig
from sofa_mover.visualization.render import (
    FrameData,
    compute_frame_data,
    render_trajectory,
)


class TestComputeFrameData:
    """`frame.area` is a scalar consumed by titles/logging — a wrong formula
    (e.g. off by a constant factor) wouldn't be obvious from the rendered image."""

    @pytest.mark.parametrize(
        "step, pose, fill_fraction",
        [(0, (0.0, 0.0, 0.0), 1.0), (5, (1.0, 2.0, 0.5), 0.5)],
    )
    def test_packs_metadata_and_computes_world_area(
        self,
        sofa_config: GridConfig,
        device: torch.device,
        step: int,
        pose: tuple[float, float, float],
        fill_fraction: float,
    ) -> None:
        height = sofa_config.grid_size
        sofa = torch.zeros(1, 1, height, height, device=device)
        sofa[:, :, : int(height * fill_fraction), :] = 1.0
        mask = torch.ones(1, 1, height, height, device=device)
        cell_area = (sofa_config.world_size / sofa_config.grid_size) ** 2

        frame = compute_frame_data(step, pose, sofa, mask, cell_area)

        assert frame.step == step
        assert frame.pose == pose
        assert isinstance(frame.sofa, np.ndarray)
        assert frame.sofa.shape == (height, height)
        assert frame.corridor_mask.shape == (height, height)
        expected_area = sofa_config.world_size**2 * fill_fraction
        assert frame.area == pytest.approx(expected_area, rel=1e-2)


class TestRenderTrajectory:
    """GIF encoding/IO checks — silent encoder regressions (dropped frame,
    unreadable file) and the empty-trajectory error path aren't visual."""

    @staticmethod
    def _uniform_frames(sofa_config: GridConfig, n: int) -> list[FrameData]:
        height = sofa_config.grid_size
        sofa = np.ones((height, height), dtype=np.float32)
        mask = np.ones((height, height), dtype=np.float32)
        return [
            FrameData(
                step=i, pose=(0.0, 0.0, 0.0), sofa=sofa, corridor_mask=mask, area=9.0
            )
            for i in range(n)
        ]

    def test_empty_frames_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            render_trajectory(
                [], tmp_path / "out.gif", sofa_extent=(-1.5, 1.5, -1.5, 1.5)
            )

    def test_writes_decodable_gif_with_correct_frame_count(
        self, tmp_path: Path, sofa_config: GridConfig
    ) -> None:
        n_frames = 3
        frames = self._uniform_frames(sofa_config, n_frames)
        half = sofa_config.world_size / 2

        out_path = render_trajectory(
            frames, tmp_path / "test_video.gif", sofa_extent=(-half, half, -half, half)
        )

        assert out_path.exists()
        with Image.open(out_path) as img:
            assert img.format == "GIF"
            assert img.n_frames == n_frames
