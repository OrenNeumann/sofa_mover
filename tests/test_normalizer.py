"""Tests for shared observation/reward normalization."""

from collections.abc import Generator

import pytest
import torch
from tensordict import TensorDictBase  # type: ignore[import-untyped]
from tensordict.nn import TensorDictModule  # type: ignore[import-untyped]

from sofa_mover.env import SofaEnv, make_sofa_env
from sofa_mover.networks import SofaActorNet, SofaBoundaryEncoder, SofaEncoder
from sofa_mover.training.config import (
    GridConfig,
    ObservationType,
    SofaEnvConfig,
    TrainingConfig,
)
from sofa_mover.training.stack import build_training_stack, TrainingStack
from sofa_mover.training.utils import (
    compute_gae_direct,
    normalize_rewards_inplace,
    optimize_ppo_epochs,
)
from sofa_mover.training.normalizer import Normalizer, RunningMeanStd

TEST_DEVICE = torch.device("cpu")
TEST_SOFA = GridConfig(grid_size=32, world_size=3.0)
NUM_ENVS = 4
BOUNDARY_RAYS = 64
TEST_TOTAL_FRAMES = 1_000_000


def _test_cfg(
    *,
    observation_type: ObservationType = "boundary",
    boundary_rays: int = BOUNDARY_RAYS,
) -> SofaEnvConfig:
    return SofaEnvConfig(
        sofa_config=TEST_SOFA,
        max_steps=20,
        observation_type=observation_type,
        boundary_rays=boundary_rays,
    )


def _training_config(
    cfg: SofaEnvConfig | None = None,
    *,
    rollout_length: int = 4,
    num_epochs: int = 1,
    minibatch_size: int = 4,
) -> TrainingConfig:
    return TrainingConfig(
        env=_test_cfg() if cfg is None else cfg,
        num_envs=NUM_ENVS,
        total_frames=NUM_ENVS * rollout_length,
        rollout_length=rollout_length,
        num_epochs=num_epochs,
        minibatch_size=minibatch_size,
        device=TEST_DEVICE,
    )


def _make_env(cfg: SofaEnvConfig | None = None) -> SofaEnv:
    return make_sofa_env(
        total_frames=TEST_TOTAL_FRAMES,
        num_envs=NUM_ENVS,
        cfg=_test_cfg() if cfg is None else cfg,
        device=TEST_DEVICE,
    )


def _make_normalizer(cfg: SofaEnvConfig | None = None) -> Normalizer:
    return Normalizer.from_config(_training_config(cfg), NUM_ENVS)


def _make_actor_module(
    cfg: SofaEnvConfig,
    normalizer: Normalizer,
) -> TensorDictModule:
    encoder: SofaEncoder | SofaBoundaryEncoder
    if cfg.observation_type == "boundary":
        encoder = SofaBoundaryEncoder(
            n_rays=cfg.boundary_rays,
            normalizer=normalizer,
        )
    else:
        encoder = SofaEncoder()
    actor_net = SofaActorNet(encoder=encoder)
    return TensorDictModule(
        actor_net,
        in_keys=[
            ("observation", "sofa_view"),
            ("observation", "pose"),
            ("observation", "progress"),
        ],
        out_keys=["logits"],
    )


def _make_boundary_actor_stack(
    *,
    boundary_rays: int = BOUNDARY_RAYS,
) -> tuple[SofaEnvConfig, SofaEnv, Normalizer, TensorDictModule]:
    cfg = _test_cfg(boundary_rays=boundary_rays)
    env = _make_env(cfg)
    normalizer = _make_normalizer(cfg)
    actor_module = _make_actor_module(cfg, normalizer)
    return cfg, env, normalizer, actor_module


def _random_action(batch_size: int) -> torch.Tensor:
    action = torch.zeros(batch_size, 27, device=TEST_DEVICE)
    idx = torch.randint(0, 27, (batch_size,), device=TEST_DEVICE)
    action.scatter_(1, idx.unsqueeze(1), 1.0)
    return action


def _collect_reward_batch(
    env: SofaEnv, num_steps: int
) -> tuple[torch.Tensor, torch.Tensor]:
    td = env.reset()
    rewards: list[torch.Tensor] = []
    done: list[torch.Tensor] = []
    for _ in range(num_steps):
        td["action"] = _random_action(NUM_ENVS)
        td = env.step(td)["next"]
        rewards.append(td["reward"].clone())
        done.append(td["done"].clone())
    return torch.stack(rewards, dim=1), torch.stack(done, dim=1)


def _flatten_boundary_observation(td: TensorDictBase) -> torch.Tensor:
    sofa_view = td["observation", "sofa_view"]
    pose = td["observation", "pose"]
    progress = td["observation", "progress"]
    return torch.cat([sofa_view, pose, progress], dim=-1)


@pytest.fixture
def training_stack() -> Generator[TrainingStack, None, None]:
    stack = build_training_stack(_training_config())
    yield stack
    stack.collector.shutdown()


class TestRunningMeanStd:
    def test_converges_to_known_stats(self) -> None:
        rms = RunningMeanStd(shape=(3,), device=TEST_DEVICE)
        torch.manual_seed(42)
        mean_true = torch.tensor([1.0, -2.0, 5.0])
        std_true = torch.tensor([0.5, 1.0, 2.0])
        for _ in range(200):
            batch = torch.randn(64, 3) * std_true + mean_true
            rms.update(batch)
        assert torch.allclose(rms.mean, mean_true, atol=0.1)
        assert torch.allclose(rms.var.sqrt(), std_true, atol=0.1)

    def test_state_dict_roundtrip(self) -> None:
        rms = RunningMeanStd(shape=(2,), device=TEST_DEVICE)
        rms.update(torch.randn(50, 2))
        state = rms.state_dict()
        rms_2 = RunningMeanStd(shape=(2,), device=TEST_DEVICE)
        rms_2.load_state_dict(state)
        assert torch.allclose(rms.mean, rms_2.mean)
        assert torch.allclose(rms.var, rms_2.var)
        assert rms.count == pytest.approx(rms_2.count)


class TestObservationNormalization:
    def test_from_config_respects_device_override(self) -> None:
        normalizer = Normalizer.from_config(
            _training_config(),
            NUM_ENVS,
            device=TEST_DEVICE,
        )

        if normalizer._obs_rms is not None:
            assert normalizer._obs_rms.mean.device == TEST_DEVICE
        if normalizer._ret_rms is not None:
            assert normalizer._ret_rms.mean.device == TEST_DEVICE

    def test_boundary_actor_forward_updates_observation_stats(self) -> None:
        _cfg, env, normalizer, actor_module = _make_boundary_actor_stack()

        td = env.reset()
        actor_module(td)

        assert normalizer._obs_rms is not None
        assert normalizer._obs_rms.count > 1e-4

    def test_freeze_stops_observation_stat_updates(self) -> None:
        _cfg, env, normalizer, actor_module = _make_boundary_actor_stack()

        td = env.reset()
        actor_module(td)
        assert normalizer._obs_rms is not None
        count_before = normalizer._obs_rms.count

        normalizer.freeze = True
        actor_module(td)

        assert normalizer._obs_rms.count == pytest.approx(count_before)

    def test_boundary_normalizer_matches_flat_observation_width(self) -> None:
        cfg = _test_cfg(boundary_rays=BOUNDARY_RAYS)
        normalizer = _make_normalizer(cfg)
        assert normalizer._obs_rms is not None
        assert normalizer._obs_rms.mean.shape == torch.Size([BOUNDARY_RAYS + 4])

    def test_grid_observation_normalization_is_disabled_with_warning(self) -> None:
        config = _training_config(
            _test_cfg(observation_type="grid", boundary_rays=0),
            rollout_length=2,
        )
        with pytest.warns(RuntimeWarning, match="disabled for grid observations"):
            stack = build_training_stack(config)

        stack.collector.shutdown()
        assert stack.normalizer.norm_obs is False
        assert stack.normalizer._obs_rms is None


class TestRewardNormalization:
    def test_reward_normalization_changes_values(self) -> None:
        env = _make_env()
        normalizer = _make_normalizer()

        reward, done = _collect_reward_batch(env, num_steps=15)
        normalized = normalizer.normalize_rewards(reward, done)

        assert normalized.dtype == torch.float32
        assert not torch.allclose(normalized, reward)

    def test_normalize_rewards_inplace_keeps_components_raw(
        self, training_stack: TrainingStack
    ) -> None:
        data = next(iter(training_stack.collector))
        raw_reward_components = {
            key: data["next", key].clone()
            for key in ("reward_erosion", "reward_progress", "reward_terminal")
        }
        normalize_rewards_inplace(data, training_stack.normalizer)
        for key, raw_value in raw_reward_components.items():
            assert torch.equal(data["next", key], raw_value)


class TestStateAndIntegration:
    def test_save_load_roundtrip_for_frozen_eval(self) -> None:
        cfg, env, normalizer, actor_module = _make_boundary_actor_stack()

        td = env.reset()
        actor_module(td)
        flat_observation = _flatten_boundary_observation(td)
        state = normalizer.state_dict()

        normalizer.freeze = True
        frozen = _make_normalizer(cfg)
        frozen.load_state_dict(state)
        frozen.freeze = True

        normalized_original = normalizer.normalize_flat_observation(flat_observation)
        normalized_loaded = frozen.normalize_flat_observation(flat_observation)

        assert torch.allclose(normalized_original, normalized_loaded)

    def test_training_stack_boundary_rollouts_and_normalized_ppo(
        self, training_stack: TrainingStack
    ) -> None:
        config = _training_config()
        data = next(iter(training_stack.collector))
        raw_reward = data["next", "reward"].clone()

        assert data["observation", "sofa_view"].dtype == torch.float32
        assert training_stack.normalizer._obs_rms is not None
        assert training_stack.normalizer._obs_rms.count > 1e-4

        normalized_reward = normalize_rewards_inplace(data, training_stack.normalizer)
        assert not torch.allclose(normalized_reward, raw_reward)

        training_stack.normalizer.freeze = True
        compute_gae_direct(
            data,
            training_stack.loss_module,
            training_stack.critic_net,
            config.gamma,
            config.gae_lambda,
        )
        optimization_stats = optimize_ppo_epochs(
            data.reshape(-1),
            training_stack.loss_module,
            training_stack.optimizer,
            config.num_epochs,
            config.minibatch_size,
            config.max_grad_norm,
            config.device,
        )
        training_stack.normalizer.freeze = False

        assert isinstance(optimization_stats.loss_policy, float)
