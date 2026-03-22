from pathlib import Path

import numpy as np
import pytest
import torch

from sofa_mover.corridor import DEVICE, GridConfig
from sofa_mover.visualization.render import (
    FrameData,
    build_composite,
    compute_frame_data,
    render_trajectory,
)


class TestComputeFrameData:
    def test_converts_tensors_and_computes_area(self, sofa_config: GridConfig) -> None:
        H = sofa_config.grid_size
        sofa = torch.ones(1, 1, H, H, device=DEVICE)
        mask = torch.ones(1, 1, H, H, device=DEVICE)

        fd = compute_frame_data(0, (0.0, 0.0, 0.0), sofa, mask, sofa_config)

        assert fd.step == 0
        assert fd.pose == (0.0, 0.0, 0.0)
        assert fd.sofa.shape == (H, H)
        assert fd.corridor_mask.shape == (H, H)
        assert isinstance(fd.sofa, np.ndarray)
        # Full sofa should have area = world_size^2
        expected_area = sofa_config.world_size**2
        assert fd.area == pytest.approx(expected_area, rel=1e-2)

    def test_partial_sofa_area(self, sofa_config: GridConfig) -> None:
        H = sofa_config.grid_size
        sofa = torch.zeros(1, 1, H, H, device=DEVICE)
        sofa[:, :, : H // 2, :] = 1.0  # half the pixels
        mask = torch.ones(1, 1, H, H, device=DEVICE)

        fd = compute_frame_data(5, (1.0, 2.0, 0.5), sofa, mask, sofa_config)

        expected_area = sofa_config.world_size**2 / 2
        assert fd.area == pytest.approx(expected_area, rel=1e-2)
        assert fd.step == 5
        assert fd.pose == (1.0, 2.0, 0.5)


class TestBuildComposite:
    def test_shape_and_range(self) -> None:
        H, W = 64, 64
        sofa = np.ones((H, W), dtype=np.float32)
        mask = np.ones((H, W), dtype=np.float32)

        rgb = build_composite(sofa, mask)

        assert rgb.shape == (H, W, 3)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_wall_color(self) -> None:
        """Pixels where mask=0 should be dark gray."""
        H, W = 16, 16
        sofa = np.zeros((H, W), dtype=np.float32)
        mask = np.zeros((H, W), dtype=np.float32)

        rgb = build_composite(sofa, mask)

        np.testing.assert_allclose(rgb[0, 0], [0.2, 0.2, 0.2])

    def test_corridor_empty_color(self) -> None:
        """Pixels where mask=1, sofa=0 should be light gray."""
        H, W = 16, 16
        sofa = np.zeros((H, W), dtype=np.float32)
        mask = np.ones((H, W), dtype=np.float32)

        rgb = build_composite(sofa, mask)

        np.testing.assert_allclose(rgb[0, 0], [0.8, 0.8, 0.8])

    def test_sofa_color(self) -> None:
        """Pixels where mask=1, sofa=1 should be blue."""
        H, W = 16, 16
        sofa = np.ones((H, W), dtype=np.float32)
        mask = np.ones((H, W), dtype=np.float32)

        rgb = build_composite(sofa, mask)

        np.testing.assert_allclose(rgb[0, 0], [0.27, 0.47, 0.73])


class TestRenderTrajectory:
    def test_creates_file(self, tmp_path: Path, sofa_config: GridConfig) -> None:
        H = sofa_config.grid_size
        sofa_arr = np.ones((H, H), dtype=np.float32)
        mask_arr = np.ones((H, H), dtype=np.float32)

        frames = [
            FrameData(
                step=i,
                pose=(0.0, 0.0, 0.0),
                sofa=sofa_arr,
                corridor_mask=mask_arr,
                area=9.0,
            )
            for i in range(3)
        ]

        out = tmp_path / "test_video.gif"
        written = render_trajectory(frames, sofa_config, out, fps=5)
        assert written.exists()
        assert written.stat().st_size > 0
