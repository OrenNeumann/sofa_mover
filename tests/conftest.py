import pytest
import torch
from sofa_mover.corridor import make_l_corridor
from sofa_mover.rasterize import Rasterizer
from sofa_mover.training.config import GridConfig


@pytest.fixture
def sofa_config() -> GridConfig:
    return GridConfig(grid_size=256, world_size=3.0)


@pytest.fixture
def template_config() -> GridConfig:
    return GridConfig(grid_size=256, world_size=6.0)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def rasterizer(sofa_config: GridConfig, device: torch.device) -> Rasterizer:
    geometry = make_l_corridor()
    return Rasterizer(geometry, sofa_config, device=device, compile=False)
