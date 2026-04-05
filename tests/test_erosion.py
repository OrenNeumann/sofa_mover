import math

import torch

from sofa_mover.erosion import erode
from sofa_mover.rasterize import Rasterizer
from sofa_mover.training.config import GridConfig


def _make_sofa(
    config: GridConfig,
    device: torch.device,
    batch_size: int = 1,
) -> torch.Tensor:
    """Create a full (all-ones) sofa grid."""
    return torch.ones(batch_size, 1, config.grid_size, config.grid_size, device=device)


def test_no_erosion_when_fully_inside(
    rasterizer: Rasterizer, sofa_config: GridConfig
) -> None:
    """If the sofa is fully inside the corridor mask, no erosion occurs."""
    sofa = _make_sofa(sofa_config, device=rasterizer.device)
    mask = rasterizer.corridor_mask(
        torch.tensor([[0.0, 0.0, 0.0]], device=rasterizer.device)
    )
    # Only keep pixels that are inside the corridor
    sofa_inside = sofa * mask
    # Eroding again with the same mask should not change anything
    result = erode(sofa_inside, mask)
    assert torch.equal(result, sofa_inside)


def test_partial_erosion(rasterizer: Rasterizer, sofa_config: GridConfig) -> None:
    """Erosion with a partial mask reduces area."""
    sofa = _make_sofa(sofa_config, device=rasterizer.device)
    mask = rasterizer.corridor_mask(
        torch.tensor([[0.0, 0.0, 0.0]], device=rasterizer.device)
    )
    # The mask doesn't cover the full sofa grid, so erosion removes pixels
    area_before = sofa.sum().item()
    result = erode(sofa, mask)
    area_after = result.sum().item()
    assert area_after < area_before
    assert area_after > 0


def test_full_erosion_with_empty_mask(
    sofa_config: GridConfig, device: torch.device
) -> None:
    """All-zero mask erodes the sofa completely."""
    sofa = _make_sofa(sofa_config, device=device)
    mask = torch.zeros_like(sofa)
    result = erode(sofa, mask)
    assert result.sum().item() == 0.0


def test_cumulative_erosion(rasterizer: Rasterizer, sofa_config: GridConfig) -> None:
    """Multiple erosion steps with different poses yield monotonically decreasing area."""
    d = rasterizer.device
    sofa = _make_sofa(sofa_config, device=d)

    poses = [
        [0.0, 0.0, 0.0],
        [0.3, 0.0, 0.0],
        [0.3, 0.0, math.pi / 6],
        [0.5, -0.2, math.pi / 4],
    ]

    areas = [sofa.sum().item()]
    for pose_vals in poses:
        mask = rasterizer.corridor_mask(torch.tensor([pose_vals], device=d))
        sofa = erode(sofa, mask)
        areas.append(sofa.sum().item())

    # Area should be monotonically non-increasing
    for i in range(1, len(areas)):
        assert areas[i] <= areas[i - 1]
    # Final area should be strictly less than initial
    assert areas[-1] < areas[0]


def test_erosion_idempotent(rasterizer: Rasterizer) -> None:
    """Applying the same mask twice doesn't change the result after the first application."""
    d = rasterizer.device
    sofa = _make_sofa(rasterizer.sofa_config, device=d)
    mask = rasterizer.corridor_mask(torch.tensor([[0.2, -0.1, 0.3]], device=d))
    once = erode(sofa, mask)
    twice = erode(once, mask)
    assert torch.equal(once, twice)


def test_erosion_batched(rasterizer: Rasterizer, sofa_config: GridConfig) -> None:
    """Erosion works correctly with batched inputs."""
    d = rasterizer.device
    batch_size = 4
    sofa = _make_sofa(sofa_config, batch_size=batch_size, device=d)

    pose = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.3, 0.0, 0.2],
            [-0.2, 0.1, -0.1],
            [0.5, -0.3, 0.4],
        ],
        device=d,
    )

    masks = rasterizer.corridor_mask(pose)
    result = erode(sofa, masks)

    # Each batch element should have different area (different poses)
    areas = [result[i].sum().item() for i in range(batch_size)]
    assert len(set(areas)) > 1, "Different poses should give different areas"


def test_l_corridor_walkthrough(
    rasterizer: Rasterizer, sofa_config: GridConfig
) -> None:
    """Simulate a sofa passing through an L-corridor: translate, rotate, translate."""
    d = rasterizer.device
    sofa = _make_sofa(sofa_config, device=d)

    trajectory = [
        [0.0, 0.0, 0.0],
        [0.0, -0.5, 0.0],
        [0.0, -0.5, math.pi / 8],
        [0.0, -0.5, math.pi / 4],
        [-0.5, -0.5, math.pi / 4],
    ]

    areas = []
    for pose_vals in trajectory:
        mask = rasterizer.corridor_mask(torch.tensor([pose_vals], device=d))
        sofa = erode(sofa, mask)
        areas.append(sofa.sum().item())

    # Area should decrease over the trajectory
    assert areas[-1] < areas[0]
    # But some material should remain (it's a valid trajectory)
    assert areas[-1] > 0
