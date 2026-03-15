import pytest
import torch
from sofa_mover.corridor import GridConfig, make_l_corridor
from sofa_mover.rasterize import Rasterizer


@pytest.fixture
def sofa_config() -> GridConfig:
    return GridConfig(grid_size=256, world_size=3.0)


@pytest.fixture
def template_config() -> GridConfig:
    return GridConfig(grid_size=256, world_size=6.0)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda")


@pytest.fixture
def rasterizer(sofa_config: GridConfig, template_config: GridConfig) -> Rasterizer:
    template = make_l_corridor(config=template_config)
    return Rasterizer(template, sofa_config, template_config)
