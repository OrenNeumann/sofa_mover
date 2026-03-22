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


@dataclass(frozen=True)
class Rectangle:
    """Axis-aligned rectangle in corridor-local coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class CorridorGeometry:
    """Corridor defined as the union of axis-aligned rectangles.

    Each rectangle is in corridor-local coordinates. A point is inside the
    corridor if it is inside any of the rectangles.
    """

    rectangles: tuple[Rectangle, ...]

    def to_tensor(self, device: torch.device) -> Float[Tensor, "R 4"]:
        """Convert rectangles to a (R, 4) tensor of [x_min, y_min, x_max, y_max]."""
        return torch.tensor(
            [[r.x_min, r.y_min, r.x_max, r.y_max] for r in self.rectangles],
            device=device,
        )


# Default configs
DEVICE = torch.device("cuda")
SOFA_CONFIG = GridConfig(grid_size=256, world_size=3.0)


def make_l_corridor(
    corridor_width: float = 1.0,
) -> CorridorGeometry:
    """Create geometry for an L-shaped corridor.

    The L-corridor is the union of two perpendicular rectangular legs, centered
    on the bend (the corridor_width x corridor_width square where the legs meet):
      - Horizontal leg: x in [-hw, +inf), y in [-hw, hw]
      - Vertical leg:   x in [-hw, hw],  y in (-inf, hw]
    where hw = corridor_width / 2.

    The sofa enters from the bottom of the vertical leg and exits through the
    right end of the horizontal leg.
    """
    hw = corridor_width / 2

    horizontal = Rectangle(x_min=-hw, y_min=-hw, x_max=float("inf"), y_max=hw)
    vertical = Rectangle(x_min=-hw, y_min=-float("inf"), x_max=hw, y_max=hw)

    return CorridorGeometry(rectangles=(horizontal, vertical))
