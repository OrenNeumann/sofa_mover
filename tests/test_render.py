from pathlib import Path

import numpy as np
import pytest
import torch

from sofa_mover.training.config import GridConfig
from sofa_mover.visualization.render import (
    FrameData,
    build_composite,
    build_corridor_mask,
    build_world_corridor_mask,
    compute_frame_data,
    expand_extent,
    render_trajectory,
    rotate_extent_counterclockwise,
    rotate_image_counterclockwise,
    sample_image_in_extent,
    sample_sofa_in_corridor_frame,
)


class TestComputeFrameData:
    def test_converts_tensors_and_computes_area(
        self, sofa_config: GridConfig, device: torch.device
    ) -> None:
        height = sofa_config.grid_size
        sofa = torch.ones(1, 1, height, height, device=device)
        mask = torch.ones(1, 1, height, height, device=device)
        cell_area = (sofa_config.world_size / sofa_config.grid_size) ** 2

        frame = compute_frame_data(0, (0.0, 0.0, 0.0), sofa, mask, cell_area)

        assert frame.step == 0
        assert frame.pose == (0.0, 0.0, 0.0)
        assert frame.sofa.shape == (height, height)
        assert frame.corridor_mask.shape == (height, height)
        assert isinstance(frame.sofa, np.ndarray)
        assert frame.area == pytest.approx(sofa_config.world_size**2, rel=1e-2)

    def test_partial_sofa_area(
        self, sofa_config: GridConfig, device: torch.device
    ) -> None:
        height = sofa_config.grid_size
        sofa = torch.zeros(1, 1, height, height, device=device)
        sofa[:, :, : height // 2, :] = 1.0
        mask = torch.ones(1, 1, height, height, device=device)
        cell_area = (sofa_config.world_size / sofa_config.grid_size) ** 2

        frame = compute_frame_data(5, (1.0, 2.0, 0.5), sofa, mask, cell_area)

        assert frame.area == pytest.approx(sofa_config.world_size**2 / 2, rel=1e-2)
        assert frame.step == 5
        assert frame.pose == (1.0, 2.0, 0.5)


class TestBuildComposite:
    def test_shape_and_range(self) -> None:
        sofa = np.ones((64, 64), dtype=np.float32)
        mask = np.ones((64, 64), dtype=np.float32)

        rgb = build_composite(sofa, mask)

        assert rgb.shape == (64, 64, 3)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_wall_color(self) -> None:
        sofa = np.zeros((16, 16), dtype=np.float32)
        mask = np.zeros((16, 16), dtype=np.float32)

        rgb = build_composite(sofa, mask)

        np.testing.assert_allclose(rgb[0, 0], [0.2, 0.2, 0.2])

    def test_corridor_empty_color(self) -> None:
        sofa = np.zeros((16, 16), dtype=np.float32)
        mask = np.ones((16, 16), dtype=np.float32)

        rgb = build_composite(sofa, mask)

        np.testing.assert_allclose(rgb[0, 0], [0.8, 0.8, 0.8])

    def test_sofa_color(self) -> None:
        sofa = np.ones((16, 16), dtype=np.float32)
        mask = np.ones((16, 16), dtype=np.float32)

        rgb = build_composite(sofa, mask)

        np.testing.assert_allclose(rgb[0, 0], [0.27, 0.47, 0.73])


class TestRenderHelpers:
    def test_sample_sofa_in_corridor_frame_matches_identity_pose(
        self, sofa_config: GridConfig
    ) -> None:
        sofa = np.zeros(
            (sofa_config.grid_size, sofa_config.grid_size), dtype=np.float32
        )
        sofa[48:144, 80:176] = 1.0
        extent = (
            -sofa_config.world_size / 2,
            sofa_config.world_size / 2,
            -sofa_config.world_size / 2,
            sofa_config.world_size / 2,
        )

        sampled = sample_sofa_in_corridor_frame(
            sofa,
            (0.0, 0.0, 0.0),
            extent,
            extent,
            sofa.shape,
        )

        np.testing.assert_array_equal(sampled, sofa)

    def test_sample_image_in_extent_zoom_out_moves_content_inward(
        self, sofa_config: GridConfig
    ) -> None:
        sofa = np.zeros(
            (sofa_config.grid_size, sofa_config.grid_size), dtype=np.float32
        )
        sofa[:, -6:-2] = 1.0
        source_extent = (
            -sofa_config.world_size / 2,
            sofa_config.world_size / 2,
            -sofa_config.world_size / 2,
            sofa_config.world_size / 2,
        )
        target_extent = expand_extent(source_extent, sofa.shape, pad_pixels=8)

        sampled = sample_image_in_extent(
            sofa,
            source_extent,
            target_extent,
            sofa.shape,
        )

        assert sampled[:, -4:].sum() == 0.0
        assert sampled[:, :-4].sum() > 0.0

    def test_sample_sofa_in_corridor_frame_keeps_area_for_grid_aligned_translation(
        self, sofa_config: GridConfig
    ) -> None:
        sofa = np.zeros(
            (sofa_config.grid_size, sofa_config.grid_size), dtype=np.float32
        )
        sofa[60:140, 90:170] = 1.0
        extent = (
            -sofa_config.world_size / 2,
            sofa_config.world_size / 2,
            -sofa_config.world_size / 2,
            sofa_config.world_size / 2,
        )
        pixel_step = sofa_config.world_size / (sofa_config.grid_size - 1)

        sampled_origin = sample_sofa_in_corridor_frame(
            sofa,
            (0.0, 0.0, 0.0),
            extent,
            extent,
            sofa.shape,
        )
        sampled_shifted = sample_sofa_in_corridor_frame(
            sofa,
            (pixel_step, 0.0, 0.0),
            extent,
            extent,
            sofa.shape,
        )

        assert sampled_origin.sum() == pytest.approx(sampled_shifted.sum())

    def test_build_corridor_mask_matches_expected_l_shape(self) -> None:
        extent = (-1.5, 1.5, -1.5, 1.5)
        corridor_mask = build_corridor_mask(extent, (5, 5), corridor_width=1.0)

        assert corridor_mask[2, 2] == 1.0
        assert corridor_mask[0, 0] == 0.0
        assert corridor_mask[0, 2] == 1.0

    def test_build_world_corridor_mask_matches_expected_identity_shape(self) -> None:
        extent = (-1.5, 1.5, -1.5, 1.5)
        corridor_mask = build_world_corridor_mask(
            (0.0, 0.0, 0.0),
            extent,
            (5, 5),
            corridor_width=1.0,
        )

        assert corridor_mask[2, 2] == 1.0
        assert corridor_mask[0, 0] == 0.0
        assert corridor_mask[0, 2] == 1.0

    def test_expand_extent_matches_padding(self) -> None:
        expanded = expand_extent((-1.0, 1.0, -2.0, 2.0), (10, 20), pad_pixels=2)
        assert expanded == pytest.approx((-1.2, 1.2, -2.8, 2.8))

    def test_rotate_image_counterclockwise_rotates_pixels(self) -> None:
        image = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        rotated = rotate_image_counterclockwise(image)

        np.testing.assert_array_equal(
            rotated,
            np.array([[2.0, 4.0], [1.0, 3.0]], dtype=np.float32),
        )

    def test_rotate_extent_counterclockwise_swaps_axes(self) -> None:
        rotated = rotate_extent_counterclockwise((-2.0, 5.0, -3.0, 7.0))
        assert rotated == pytest.approx((-7.0, 3.0, -2.0, 5.0))


class TestRenderTrajectory:
    def test_creates_file(self, tmp_path: Path, sofa_config: GridConfig) -> None:
        height = sofa_config.grid_size
        sofa = np.ones((height, height), dtype=np.float32)
        mask = np.ones((height, height), dtype=np.float32)

        frames = [
            FrameData(
                step=step,
                pose=(0.0, 0.0, 0.0),
                sofa=sofa,
                corridor_mask=mask,
                area=9.0,
            )
            for step in range(3)
        ]

        output = tmp_path / "test_video.gif"
        half = sofa_config.world_size / 2
        sofa_extent = (-half, half, -half, half)
        written = render_trajectory(frames, output, sofa_extent=sofa_extent, fps=5)

        assert written.exists()
        assert written.stat().st_size > 0
