from typing import cast

import pytest
import torch
from tensordict import TensorDict

from sofa_mover.env import make_sofa_env
from sofa_mover.networks import (
    BoundaryEncoder,
    MultiDiscreteCategorical,
    SofaActorNet,
    SofaCriticNet,
)
from sofa_mover.training.config import SofaEnvConfig
from sofa_mover.training.utils import (
    _minibatch_losses,
    compute_gae_direct,
    extract_episode_metrics,
    maybe_build_episode_composite,
)

TEST_TOTAL_FRAMES = 1_000_000


class _PlantedValueCritic(torch.nn.Module):
    """Critic stub that reads back the value planted in sofa_view."""

    def forward(
        self, sofa_view: torch.Tensor, pose: torch.Tensor, progress: torch.Tensor
    ) -> torch.Tensor:
        return sofa_view[:, :1]


def _obs(values: list[float]) -> TensorDict:
    return TensorDict(
        {
            "sofa_view": torch.tensor(values).reshape(1, 3, 1),
            "pose": torch.zeros(1, 3, 3),
            "progress": torch.zeros(1, 3, 1),
        },
        batch_size=(1, 3),
    )


# Hand-computed with v=[1,2,3], v'=[2,3,4], r=1, γ=λ=0.5, episode ending at t=2:
# delta_t = r + γ·(1-term_t)·v'_t - v_t;  A_t = delta_t + γλ·(1-done_t)·A_{t+1}
@pytest.mark.parametrize(
    "terminated, expected_adv",
    [
        (True, [1.0, 0.0, -2.0]),  # terminal: v' masked at the final step
        (False, [1.125, 0.5, 0.0]),  # truncated: v' still bootstrapped
    ],
)
def test_compute_gae_direct(terminated: bool, expected_adv: list[float]) -> None:
    data = TensorDict(
        {
            "observation": _obs([1.0, 2.0, 3.0]),
            "next": TensorDict(
                {
                    "observation": _obs([2.0, 3.0, 4.0]),
                    "reward": torch.ones(1, 3, 1),
                    "done": torch.tensor([False, False, True]).reshape(1, 3, 1),
                    "terminated": torch.tensor([False, False, terminated]).reshape(
                        1, 3, 1
                    ),
                },
                batch_size=(1, 3),
            ),
        },
        batch_size=(1, 3),
    )

    # compute_gae_direct only calls the critic; the stub stands in fine.
    critic = cast(SofaCriticNet, _PlantedValueCritic())
    compute_gae_direct(data, critic, gamma=0.5, gae_lambda=0.5)

    expected = torch.tensor(expected_adv).reshape(1, 3, 1)
    torch.testing.assert_close(data["advantage"], expected)
    values = torch.tensor([1.0, 2.0, 3.0]).reshape(1, 3, 1)
    torch.testing.assert_close(data["value_target"], expected + values)


def test_minibatch_losses_match_distribution_class() -> None:
    """_minibatch_losses inlines MultiDiscreteCategorical's math; keep them equal."""
    torch.manual_seed(0)
    nvec = [5, 5, 5]
    encoder = BoundaryEncoder(n_rays=16, feature_dim=32, width=32, depth=1)
    actor_net = SofaActorNet(nvec=nvec, encoder=encoder, width=32, depth=1)
    critic_head = torch.nn.Linear(32, 1)

    B = 32
    sofa_view, pose, progress = torch.rand(B, 16), torch.randn(B, 3), torch.rand(B, 1)
    advantage, value_target = torch.randn(B, 1), torch.randn(B, 1)
    clip_epsilon, entropy_coeff = 0.2, 0.01
    with torch.no_grad():
        dist = MultiDiscreteCategorical(
            logits=actor_net(sofa_view, pose, progress), nvec=nvec
        )
        action = dist.sample()
        old_log_prob = dist.log_prob(action) - 0.1  # offset for non-trivial ratios

    _, loss_policy, _, loss_entropy, approx_kl, _ = _minibatch_losses(
        actor_net,
        critic_head,
        sofa_view,
        pose,
        progress,
        action,
        old_log_prob,
        advantage,
        value_target,
        clip_epsilon,
        entropy_coeff,
        critic_coeff=1.0,
    )

    ratio = torch.exp(dist.log_prob(action) - old_log_prob)
    surr1 = ratio * advantage.squeeze(-1)
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage.squeeze(
        -1
    )
    torch.testing.assert_close(loss_policy, -torch.min(surr1, surr2).mean())
    torch.testing.assert_close(loss_entropy, -entropy_coeff * dist.entropy().mean())
    expected_kl = ((ratio - 1.0) - ratio.log()).mean()
    torch.testing.assert_close(approx_kl, expected_kl)


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
    assert metrics.best_area_at_goal == pytest.approx(0.8)
    assert metrics.goal_rate == pytest.approx(0.5)
    assert metrics.truncation_rate == pytest.approx(0.5)
    assert metrics.mean_ep_length == pytest.approx(15.0)
    assert metrics.mean_total_angle == pytest.approx(2.0)
    assert metrics.mean_total_distance == pytest.approx(3.0)
    assert metrics.last_done_idx == 1


def test_maybe_build_episode_composite_skips_boundary_mode() -> None:
    env = make_sofa_env(
        total_frames=TEST_TOTAL_FRAMES,
        num_envs=1,
        cfg=SofaEnvConfig(
            observation_type="boundary",
            boundary_rays=8,
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
