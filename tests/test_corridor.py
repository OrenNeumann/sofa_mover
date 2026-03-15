import torch

from sofa_mover.corridor import GridConfig, make_l_corridor


def test_l_corridor_shape(template_config: GridConfig) -> None:
    template = make_l_corridor(config=template_config)
    assert template.shape == (
        1,
        1,
        template_config.grid_size,
        template_config.grid_size,
    )


def test_l_corridor_binary(template_config: GridConfig) -> None:
    template = make_l_corridor(config=template_config)
    unique_vals = torch.unique(template)
    assert torch.allclose(unique_vals, torch.tensor([0.0, 1.0], device=template.device))


def test_l_corridor_area(template_config: GridConfig) -> None:
    corridor_width = 1.0
    template = make_l_corridor(
        config=template_config,
        corridor_width=corridor_width,
    )
    half = template_config.world_size / 2
    hw = corridor_width / 2
    # Horizontal leg: (half + hw) * cw, from x=-hw to x=half
    # Vertical leg:   (half + hw) * cw, from y=-half to y=hw
    # Overlap (bend square): cw^2
    leg_span = half + hw
    expected_area_world = 2 * leg_span * corridor_width - corridor_width**2
    pixel_area = (template_config.world_size / template_config.grid_size) ** 2
    expected_pixels = expected_area_world / pixel_area

    actual_pixels = template.sum().item()
    # Allow 3% tolerance — corridor is only ~42px wide at this resolution,
    # so boundary rounding has outsized effect
    assert abs(actual_pixels - expected_pixels) / expected_pixels < 0.03


def test_l_corridor_legs_are_connected(template_config: GridConfig) -> None:
    """The corner region (around origin) should be passable."""
    template = make_l_corridor(config=template_config)
    # Origin is at the center of the grid (pixel center // 2).
    # Row = y-axis: row 0 = y_min (-3), row 255 = y_max (+3).
    # Col = x-axis: col 0 = x_min (-3), col 255 = x_max (+3).
    # A point just inside the corner at world coords (-0.5, -0.5)
    # is inside the vertical leg (x in [-1, 0], y in [-3, 0]).
    ppu = template_config.pixels_per_unit
    center = template_config.grid_size // 2
    # World (-0.5, -0.5): row = center + (-0.5 mapped), but y=-0.5 < 0 means
    # row < center (since row increases with y from -3 to +3).
    row = center - int(0.5 * ppu)  # y = -0.5 → above center row
    col = center - int(0.5 * ppu)  # x = -0.5 → left of center col
    assert template[0, 0, row, col].item() == 1.0
