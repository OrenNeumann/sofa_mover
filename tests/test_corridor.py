import torch

from sofa_mover.corridor import (
    CorridorGeometry,
    Rectangle,
    make_l_corridor,
)
from sofa_mover.rasterize import Rasterizer
from sofa_mover.training.config import GridConfig


def test_make_l_corridor_returns_geometry() -> None:
    geometry = make_l_corridor()
    assert isinstance(geometry, CorridorGeometry)
    assert len(geometry.rectangles) == 2
    for r in geometry.rectangles:
        assert isinstance(r, Rectangle)


def test_make_l_corridor_rectangle_bounds() -> None:
    corridor_width = 1.0
    geometry = make_l_corridor(corridor_width=corridor_width)
    hw = corridor_width / 2

    horiz, vert = geometry.rectangles

    # Horizontal leg: x >= -hw, y in [-hw, hw]
    assert horiz.x_min == -hw
    assert horiz.y_min == -hw
    assert horiz.y_max == hw
    assert horiz.x_max > 10.0  # semi-infinite sentinel

    # Vertical leg: x in [-hw, hw], y <= hw
    assert vert.x_min == -hw
    assert vert.x_max == hw
    assert vert.y_max == hw
    assert vert.y_min < -10.0  # semi-infinite sentinel


def test_make_l_corridor_custom_width() -> None:
    geometry = make_l_corridor(corridor_width=2.0)
    horiz, vert = geometry.rectangles
    assert horiz.x_min == -1.0
    assert horiz.y_max == 1.0
    assert vert.x_max == 1.0


def test_geometry_to_tensor(device: torch.device) -> None:
    geometry = make_l_corridor()
    t = geometry.to_tensor(device)
    assert t.shape == (2, 4)
    assert t.device.type == device.type


def test_corridor_mask_area(device: torch.device) -> None:
    """Corridor mask at identity pose on a large grid should have correct area."""
    corridor_width = 1.0
    geometry = make_l_corridor(corridor_width=corridor_width)
    config = GridConfig(grid_size=256, world_size=6.0)
    rasterizer = Rasterizer(geometry, config, device=device, compile=False)

    mask = rasterizer.corridor_mask(torch.tensor([[0.0, 0.0, 0.0]], device=device))
    assert mask.shape == (1, 1, 256, 256)

    # Binary values only
    unique_vals = torch.unique(mask)
    for v in unique_vals:
        assert v.item() in (0.0, 1.0)

    # Area check: the L-corridor viewed on the 6x6 grid.
    # Horizontal leg area visible: (3.0 + 0.5) * 1.0 = 3.5 (x from -0.5 to 3.0, y from -0.5 to 0.5)
    # Vertical leg area visible: (3.0 + 0.5) * 1.0 = 3.5 (x from -0.5 to 0.5, y from -3.0 to 0.5)
    # Overlap (bend square): 1.0
    # Total: 3.5 + 3.5 - 1.0 = 6.0
    half = config.world_size / 2
    hw = corridor_width / 2
    leg_span = half + hw
    expected_area_world = 2 * leg_span * corridor_width - corridor_width**2
    pixel_area = (config.world_size / config.grid_size) ** 2
    expected_pixels = expected_area_world / pixel_area

    actual_pixels = mask.sum().item()
    assert abs(actual_pixels - expected_pixels) / expected_pixels < 0.03


def test_corridor_corner_is_passable(device: torch.device) -> None:
    """The corner region (around origin) should be passable."""
    geometry = make_l_corridor()
    config = GridConfig(grid_size=256, world_size=6.0)
    rasterizer = Rasterizer(geometry, config, device=device, compile=False)

    mask = rasterizer.corridor_mask(torch.tensor([[0.0, 0.0, 0.0]], device=device))

    # World coords (-0.3, -0.3) should be inside the vertical leg
    ppu = config.pixels_per_unit
    center = config.grid_size // 2
    row = center - int(0.3 * ppu)
    col = center - int(0.3 * ppu)
    assert mask[0, 0, row, col].item() == 1.0
