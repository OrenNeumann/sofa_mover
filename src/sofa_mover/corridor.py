from dataclasses import dataclass
from typing import TypeAlias

import torch
from jaxtyping import Float
from torch import Tensor

# Pose tensor, (batch_size, [x, y, theta])
Pose: TypeAlias = Float[Tensor, "B 3"]


@dataclass(frozen=True)
class GridConfig:
    """Configuration for a 2D grid mapping pixels to world coordinates.

    The grid is centered at the origin, spanning [-world_size/2, world_size/2]
    in both axes.
    """

    grid_size: int = 256
    world_size: float = 3.0

    @property
    def pixels_per_unit(self) -> float:
        return self.grid_size / self.world_size


# Default configs
DEVICE = torch.device("cuda")
SOFA_CONFIG = GridConfig(grid_size=256, world_size=3.0)
TEMPLATE_CONFIG = GridConfig(grid_size=256, world_size=6.0)


def make_l_corridor(
    config: GridConfig = TEMPLATE_CONFIG,
    corridor_width: float = 1.0,
    device: torch.device = DEVICE,
) -> Float[Tensor, "1 1 H W"]:
    """Create a binary template for an L-shaped corridor.

    The L-corridor is the union of two perpendicular rectangular legs, centered
    on the bend (the corridor_width x corridor_width square where the legs meet).
    Both legs extend to the world edges:
      - Horizontal leg: x in [-hw, half], y in [-hw, hw]
      - Vertical leg:   x in [-hw, hw], y in [-half, hw]
    where hw = corridor_width / 2, half = world_size / 2.

    The sofa enters from the bottom of the vertical leg and exits through the
    right end of the horizontal leg.

    Returns:
        Binary float tensor (1, 1, H, W). 1.0 = passable, 0.0 = wall.
    """
    half = config.world_size / 2
    coords = torch.linspace(-half, half, config.grid_size, device=device)
    y_grid, x_grid = torch.meshgrid(coords, coords, indexing="ij")

    hw = corridor_width / 2

    # Horizontal leg: x in [-hw, half], y in [-hw, hw]
    horizontal = (x_grid >= -hw) & (y_grid >= -hw) & (y_grid <= hw)

    # Vertical leg: x in [-hw, hw], y in [-half, hw]
    vertical = (x_grid >= -hw) & (x_grid <= hw) & (y_grid <= hw)

    template = (horizontal | vertical).float()
    return template.unsqueeze(0).unsqueeze(0)
