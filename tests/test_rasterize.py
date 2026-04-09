import math

import torch

from sofa_mover.rasterize import Rasterizer
from sofa_mover.training.config import GridConfig


def _make_pose(
    x: float,
    y: float,
    theta: float,
    device: torch.device,
    batch_size: int = 1,
) -> torch.Tensor:
    """Helper to create a (B, 3) pose tensor."""
    return (
        torch.tensor([[x, y, theta]], device=device).expand(batch_size, -1).contiguous()
    )


def test_identity_pose(rasterizer: Rasterizer, sofa_config: GridConfig) -> None:
    """At pose (0,0,0), the sofa grid should see the center of the template."""
    pose = _make_pose(0.0, 0.0, 0.0, device=rasterizer.device)
    mask = rasterizer.corridor_mask(pose)

    assert mask.shape == (1, 1, sofa_config.grid_size, sofa_config.grid_size)
    # The sofa grid (world_size=3.0) views the center 3x3 of the template (world_size=6.0).
    # The L-corridor has material at the center, so mask should have some 1s and some 0s.
    total = mask.numel()
    ones = mask.sum().item()
    assert 0 < ones < total, "Mask should be partially filled at identity pose"


def test_identity_pose_center_is_passable(
    rasterizer: Rasterizer, sofa_config: GridConfig
) -> None:
    """The center of the sofa grid (world origin) should map to the corridor corner."""
    pose = _make_pose(0.0, 0.0, 0.0, device=rasterizer.device)
    mask = rasterizer.corridor_mask(pose)

    # The origin is at the corner of the L-corridor.
    # Pixel just below and to the left of center should be inside the vertical leg.
    center = sofa_config.grid_size // 2
    # Check a pixel slightly inside the vertical leg: world ~(-0.3, -0.3)
    offset = int(0.3 * sofa_config.pixels_per_unit)
    assert mask[0, 0, center - offset, center - offset].item() is True


def test_translation_shifts_mask(rasterizer: Rasterizer) -> None:
    """Translating the corridor should shift the visible mask."""
    d = rasterizer.device
    mask_0 = rasterizer.corridor_mask(_make_pose(0.0, 0.0, 0.0, device=d))
    # Shift corridor to the right by 0.5 world units
    mask_shifted = rasterizer.corridor_mask(_make_pose(0.5, 0.0, 0.0, device=d))
    # The masks should be different
    assert not torch.equal(mask_0, mask_shifted)
    # Shifting corridor right (x=+0.5) means sofa views a region shifted left
    # in corridor space, so we see less of the horizontal leg (x>0) and the
    # total visible area decreases.
    assert mask_shifted.sum() < mask_0.sum()


def test_rotation_90_degrees(rasterizer: Rasterizer) -> None:
    """Rotating the corridor 90 degrees should produce a different mask."""
    d = rasterizer.device
    mask_0 = rasterizer.corridor_mask(_make_pose(0.0, 0.0, 0.0, device=d))
    mask_90 = rasterizer.corridor_mask(_make_pose(0.0, 0.0, math.pi / 2, device=d))
    assert not torch.equal(mask_0, mask_90)
    # Both should have approximately the same total passable area
    # (rotation preserves area), within rasterization tolerance
    area_0 = mask_0.sum().item()
    area_90 = mask_90.sum().item()
    assert abs(area_0 - area_90) / max(area_0, area_90) < 0.05


def test_large_translation_gives_empty_mask(rasterizer: Rasterizer) -> None:
    """Corridor translated far away should produce all-zero mask."""
    mask = rasterizer.corridor_mask(
        _make_pose(10.0, 10.0, 0.0, device=rasterizer.device)
    )
    assert mask.sum().item() == 0.0


def test_batch_identical_poses(rasterizer: Rasterizer) -> None:
    """Batch of identical poses should produce identical masks."""
    pose = _make_pose(0.3, -0.2, 0.1, batch_size=4, device=rasterizer.device)
    masks = rasterizer.corridor_mask(pose)

    assert masks.shape[0] == 4
    for i in range(1, 4):
        assert torch.equal(masks[0], masks[i])


def test_batch_different_poses(rasterizer: Rasterizer) -> None:
    """Batched results should match individual results."""
    d = rasterizer.device
    poses_list = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5)]

    # Run individually
    individual_masks = []
    for px, py, pt in poses_list:
        mask = rasterizer.corridor_mask(_make_pose(px, py, pt, device=d))
        individual_masks.append(mask)

    # Run batched
    pose = torch.tensor(poses_list, device=d)
    batched_masks = rasterizer.corridor_mask(pose)

    for i in range(len(poses_list)):
        assert torch.equal(batched_masks[i], individual_masks[i][0])
