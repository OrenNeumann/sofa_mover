import math

import torch

from sofa_mover.corridor import GridConfig, make_l_corridor
from sofa_mover.erosion import erode
from sofa_mover.rasterize import get_corridor_mask


def _make_sofa(
    config: GridConfig, batch_size: int = 1, device: str = "cuda"
) -> torch.Tensor:
    """Create a full (all-ones) sofa grid."""
    return torch.ones(batch_size, 1, config.grid_size, config.grid_size, device=device)


def test_no_erosion_when_fully_inside(
    sofa_config: GridConfig, template_config: GridConfig
) -> None:
    """If the sofa is fully inside the corridor mask, no erosion occurs."""
    template = make_l_corridor(config=template_config)
    # Use a small sofa that fits entirely within the corridor at identity pose
    sofa = _make_sofa(sofa_config)
    mask = get_corridor_mask(
        template,
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1),
        sofa_config,
        template_config,
    )
    # Only keep pixels that are inside the corridor
    sofa_inside = sofa * mask
    # Eroding again with the same mask should not change anything
    result = erode(sofa_inside, mask)
    assert torch.equal(result, sofa_inside)


def test_partial_erosion(sofa_config: GridConfig, template_config: GridConfig) -> None:
    """Erosion with a partial mask reduces area."""
    sofa = _make_sofa(sofa_config)
    template = make_l_corridor(config=template_config)
    mask = get_corridor_mask(
        template,
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1),
        sofa_config,
        template_config,
    )
    # The mask doesn't cover the full sofa grid, so erosion removes pixels
    area_before = sofa.sum().item()
    result = erode(sofa, mask)
    area_after = result.sum().item()
    assert area_after < area_before
    assert area_after > 0


def test_full_erosion_with_empty_mask(sofa_config: GridConfig) -> None:
    """All-zero mask erodes the sofa completely."""
    sofa = _make_sofa(sofa_config)
    mask = torch.zeros_like(sofa)
    result = erode(sofa, mask)
    assert result.sum().item() == 0.0


def test_cumulative_erosion(
    sofa_config: GridConfig, template_config: GridConfig
) -> None:
    """Multiple erosion steps with different poses yield monotonically decreasing area."""
    template = make_l_corridor(config=template_config)
    sofa = _make_sofa(sofa_config)

    poses = [
        (0.0, 0.0, 0.0),
        (0.3, 0.0, 0.0),
        (0.3, 0.0, math.pi / 6),
        (0.5, -0.2, math.pi / 4),
    ]

    areas = [sofa.sum().item()]
    for px, py, pt in poses:
        mask = get_corridor_mask(
            template,
            torch.tensor([px]),
            torch.tensor([py]),
            torch.tensor([pt]),
            sofa_config,
            template_config,
        )
        sofa = erode(sofa, mask)
        areas.append(sofa.sum().item())

    # Area should be monotonically non-increasing
    for i in range(1, len(areas)):
        assert areas[i] <= areas[i - 1]
    # Final area should be strictly less than initial
    assert areas[-1] < areas[0]


def test_erosion_idempotent(
    sofa_config: GridConfig, template_config: GridConfig
) -> None:
    """Applying the same mask twice doesn't change the result after the first application."""
    template = make_l_corridor(config=template_config)
    sofa = _make_sofa(sofa_config)
    mask = get_corridor_mask(
        template,
        torch.tensor([0.2]),
        torch.tensor([-0.1]),
        torch.tensor([0.3]),
        sofa_config,
        template_config,
    )
    once = erode(sofa, mask)
    twice = erode(once, mask)
    assert torch.equal(once, twice)


def test_erosion_batched(sofa_config: GridConfig, template_config: GridConfig) -> None:
    """Erosion works correctly with batched inputs."""
    template = make_l_corridor(config=template_config)
    batch_size = 4
    sofa = _make_sofa(sofa_config, batch_size=batch_size)

    x = torch.tensor([0.0, 0.3, -0.2, 0.5])
    y = torch.tensor([0.0, 0.0, 0.1, -0.3])
    theta = torch.tensor([0.0, 0.2, -0.1, 0.4])

    masks = get_corridor_mask(template, x, y, theta, sofa_config, template_config)
    result = erode(sofa, masks)

    # Each batch element should have different area (different poses)
    areas = [result[i].sum().item() for i in range(batch_size)]
    assert len(set(areas)) > 1, "Different poses should give different areas"


def test_l_corridor_walkthrough(
    sofa_config: GridConfig, template_config: GridConfig
) -> None:
    """Simulate a sofa passing through an L-corridor: translate, rotate, translate."""
    template = make_l_corridor(config=template_config)
    sofa = _make_sofa(sofa_config)

    # Start: corridor at reference position
    # Step 1: move corridor down (sofa enters vertical leg)
    # Step 2-3: rotate corridor (sofa navigates the corner)
    # Step 4: move corridor left (sofa exits through horizontal leg)
    trajectory = [
        (0.0, 0.0, 0.0),
        (0.0, -0.5, 0.0),
        (0.0, -0.5, math.pi / 8),
        (0.0, -0.5, math.pi / 4),
        (-0.5, -0.5, math.pi / 4),
    ]

    areas = []
    for px, py, pt in trajectory:
        mask = get_corridor_mask(
            template,
            torch.tensor([px]),
            torch.tensor([py]),
            torch.tensor([pt]),
            sofa_config,
            template_config,
        )
        sofa = erode(sofa, mask)
        areas.append(sofa.sum().item())

    # Area should decrease over the trajectory
    assert areas[-1] < areas[0]
    # But some material should remain (it's a valid trajectory)
    assert areas[-1] > 0
