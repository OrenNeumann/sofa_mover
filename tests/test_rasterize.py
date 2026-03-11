import math

import torch

from sofa_mover.corridor import GridConfig, make_l_corridor
from sofa_mover.rasterize import get_corridor_mask


def _make_poses(
    x: float, y: float, theta: float, batch_size: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helper to create single-element pose tensors."""
    return (
        torch.full((batch_size,), x),
        torch.full((batch_size,), y),
        torch.full((batch_size,), theta),
    )


def test_identity_pose(sofa_config: GridConfig, template_config: GridConfig) -> None:
    """At pose (0,0,0), the sofa grid should see the center of the template."""
    template = make_l_corridor(config=template_config)
    x, y, theta = _make_poses(0.0, 0.0, 0.0)
    mask = get_corridor_mask(template, x, y, theta, sofa_config, template_config)

    assert mask.shape == (1, 1, sofa_config.grid_size, sofa_config.grid_size)
    # The sofa grid (world_size=3.0) views the center 3x3 of the template (world_size=6.0).
    # The L-corridor has material at the center, so mask should have some 1s and some 0s.
    total = mask.numel()
    ones = mask.sum().item()
    assert 0 < ones < total, "Mask should be partially filled at identity pose"


def test_identity_pose_center_is_passable(
    sofa_config: GridConfig, template_config: GridConfig
) -> None:
    """The center of the sofa grid (world origin) should map to the corridor corner."""
    template = make_l_corridor(config=template_config)
    x, y, theta = _make_poses(0.0, 0.0, 0.0)
    mask = get_corridor_mask(template, x, y, theta, sofa_config, template_config)

    # The origin is at the corner of the L-corridor.
    # Pixel just below and to the left of center should be inside the vertical leg.
    center = sofa_config.grid_size // 2
    # Check a pixel slightly inside the vertical leg: world ~(-0.3, -0.3)
    offset = int(0.3 * sofa_config.pixels_per_unit)
    assert mask[0, 0, center - offset, center - offset].item() == 1.0


def test_translation_shifts_mask(
    sofa_config: GridConfig, template_config: GridConfig
) -> None:
    """Translating the corridor should shift the visible mask."""
    template = make_l_corridor(config=template_config)
    mask_0 = get_corridor_mask(
        template, *_make_poses(0.0, 0.0, 0.0), sofa_config, template_config
    )
    # Shift corridor to the right by 0.5 world units
    mask_shifted = get_corridor_mask(
        template, *_make_poses(0.5, 0.0, 0.0), sofa_config, template_config
    )
    # The masks should be different
    assert not torch.equal(mask_0, mask_shifted)
    # Shifting corridor right (x=+0.5) means sofa views a region shifted left
    # in corridor space, so we see less of the horizontal leg (x>0) and the
    # total visible area decreases.
    assert mask_shifted.sum() < mask_0.sum()


def test_rotation_90_degrees(
    sofa_config: GridConfig, template_config: GridConfig
) -> None:
    """Rotating the corridor 90 degrees should produce a different mask."""
    template = make_l_corridor(config=template_config)
    mask_0 = get_corridor_mask(
        template, *_make_poses(0.0, 0.0, 0.0), sofa_config, template_config
    )
    mask_90 = get_corridor_mask(
        template,
        *_make_poses(0.0, 0.0, math.pi / 2),
        sofa_config,
        template_config,
    )
    assert not torch.equal(mask_0, mask_90)
    # Both should have approximately the same total passable area
    # (rotation preserves area), within rasterization tolerance
    area_0 = mask_0.sum().item()
    area_90 = mask_90.sum().item()
    assert abs(area_0 - area_90) / max(area_0, area_90) < 0.05


def test_large_translation_gives_empty_mask(
    sofa_config: GridConfig, template_config: GridConfig
) -> None:
    """Corridor translated far away should produce all-zero mask."""
    template = make_l_corridor(config=template_config)
    # Shift corridor way beyond the sofa grid view
    mask = get_corridor_mask(
        template, *_make_poses(10.0, 10.0, 0.0), sofa_config, template_config
    )
    assert mask.sum().item() == 0.0


def test_batch_identical_poses(
    sofa_config: GridConfig, template_config: GridConfig
) -> None:
    """Batch of identical poses should produce identical masks."""
    template = make_l_corridor(config=template_config)
    x, y, theta = _make_poses(0.3, -0.2, 0.1, batch_size=4)
    masks = get_corridor_mask(template, x, y, theta, sofa_config, template_config)

    assert masks.shape[0] == 4
    for i in range(1, 4):
        assert torch.equal(masks[0], masks[i])


def test_batch_different_poses(
    sofa_config: GridConfig, template_config: GridConfig
) -> None:
    """Batched results should match individual results."""
    template = make_l_corridor(config=template_config)
    poses = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5)]

    # Run individually
    individual_masks = []
    for px, py, pt in poses:
        mask = get_corridor_mask(
            template, *_make_poses(px, py, pt), sofa_config, template_config
        )
        individual_masks.append(mask)

    # Run batched
    x = torch.tensor([p[0] for p in poses])
    y = torch.tensor([p[1] for p in poses])
    theta = torch.tensor([p[2] for p in poses])
    batched_masks = get_corridor_mask(
        template, x, y, theta, sofa_config, template_config
    )

    for i in range(len(poses)):
        assert torch.equal(batched_masks[i], individual_masks[i][0])
