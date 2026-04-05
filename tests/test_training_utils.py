import pytest
import torch
from tensordict import TensorDict

from sofa_mover.env import make_sofa_env
from sofa_mover.training.config import SofaEnvConfig
from sofa_mover.training.utils import (
    extract_episode_metrics,
    maybe_build_episode_composite,
)


def test_extract_episode_metrics_returns_none_when_no_done() -> None:
    next_td = TensorDict(
        {"done": torch.tensor([[False], [False]])},
        batch_size=(2,),
    )
    data = TensorDict({"next": next_td}, batch_size=(2,))
    assert extract_episode_metrics(data) is None


def test_extract_episode_metrics_aggregates_done_episodes() -> None:
    next_td = TensorDict(
        {
            "done": torch.tensor([[True], [True], [False]]),
            "terminated": torch.tensor([[True], [False], [False]]),
            "terminal_area": torch.tensor([[0.8], [0.0], [0.2]]),
            "episode_length": torch.tensor([[10], [20], [30]]),
            "episode_total_angle": torch.tensor([[1.0], [3.0], [5.0]]),
            "episode_total_distance": torch.tensor([[2.0], [4.0], [6.0]]),
        },
        batch_size=(3,),
    )
    data_flat_log = TensorDict({"next": next_td}, batch_size=(3,))

    metrics = extract_episode_metrics(data_flat_log)

    assert metrics is not None
    assert metrics.n_done == 2
    # area_at_goal averages over ALL done episodes (0.8 + 0.0) / 2 = 0.4
    assert metrics.area_at_goal == pytest.approx(0.4)
    assert metrics.goal_rate == pytest.approx(0.5)
    assert metrics.truncation_rate == pytest.approx(0.5)
    assert metrics.mean_ep_length == pytest.approx(15.0)
    assert metrics.mean_total_angle == pytest.approx(2.0)
    assert metrics.mean_total_distance == pytest.approx(3.0)
    assert metrics.last_done_idx == 1


def test_maybe_build_episode_composite_skips_boundary_mode() -> None:
    env = make_sofa_env(
        num_envs=1,
        cfg=SofaEnvConfig(
            observation_type="boundary",
            boundary_rays=8,
            compile_rasterizer=False,
        ),
        device=torch.device("cpu"),
    )
    next_td = TensorDict(
        {
            "done": torch.tensor([[True]]),
            "observation": TensorDict(
                {
                    "sofa_view": torch.zeros(1, 8),
                    "pose": torch.zeros(1, 3),
                    "progress": torch.zeros(1, 1),
                },
                batch_size=(1,),
            ),
        },
        batch_size=(1,),
    )
    data_flat_log = TensorDict({"next": next_td}, batch_size=(1,))

    composite = maybe_build_episode_composite(
        data_flat_log,
        env,
        batch_idx=0,
        image_log_interval=1,
        last_done_idx=0,
    )

    assert composite is None
