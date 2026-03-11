import pytest
import torch
from sofa_mover.corridor import GridConfig


@pytest.fixture
def sofa_config() -> GridConfig:
    return GridConfig(grid_size=256, world_size=3.0)


@pytest.fixture
def template_config() -> GridConfig:
    return GridConfig(grid_size=256, world_size=6.0)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda")
