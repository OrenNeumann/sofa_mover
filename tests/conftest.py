import pytest
import torch
from sofa_mover.corridor import DEVICE, GridConfig, make_l_corridor
from sofa_mover.rasterize import Rasterizer


@pytest.fixture
def sofa_config() -> GridConfig:
    return GridConfig(grid_size=256, world_size=3.0)


@pytest.fixture
def template_config() -> GridConfig:
    return GridConfig(grid_size=256, world_size=6.0)


@pytest.fixture
def device() -> torch.device:
    return DEVICE


@pytest.fixture
def rasterizer(sofa_config: GridConfig) -> Rasterizer:
    geometry = make_l_corridor()
    return Rasterizer(geometry, sofa_config, device=DEVICE, compile=False)
