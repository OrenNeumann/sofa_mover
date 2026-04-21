"""Tests for shared observation/reward normalization."""

from collections.abc import Generator

import pytest
import torch
from tensordict.nn import TensorDictModule  # type: ignore[import-untyped]

from sofa_mover.env import SofaEnv, make_sofa_env
from sofa_mover.networks import (
    SofaActorNet,
    BoundaryEncoder,
    SofaEncoder,
)
from sofa_mover.training.config import (
    BoundaryEncoderType,
    GridConfig,
    ObservationType,
    SofaEnvConfig,
    TrainingConfig,
)
from sofa_mover.training.stack import build_training_stack, TrainingStack
from sofa_mover.training.utils import (
    compute_gae_direct,
    optimize_ppo_epochs,
)
from sofa_mover.training.normalizer import Normalizer, ObsGroupSpec, RunningMeanStd

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
    boundary_encoder: BoundaryEncoderType = "mlp",
) -> TrainingConfig:
    return TrainingConfig(
        env=_test_cfg() if cfg is None else cfg,
        num_envs=NUM_ENVS,
        total_frames=NUM_ENVS * rollout_length,
        rollout_length=rollout_length,
        num_epochs=num_epochs,
        minibatch_size=minibatch_size,
        boundary_encoder=boundary_encoder,
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
    n_bins = 2 * cfg.n_magnitude_levels + 1
    nvec = [n_bins, n_bins, n_bins]
    encoder: SofaEncoder | BoundaryEncoder
    if cfg.observation_type == "boundary":
        encoder = BoundaryEncoder(
            n_rays=2 * cfg.boundary_rays,
            normalizer=normalizer,
        )
    else:
        encoder = SofaEncoder()
    actor_net = SofaActorNet(nvec=nvec, encoder=encoder)
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


def _random_action(batch_size: int, n_bins: int) -> torch.Tensor:
    """Random independent action per axis (MultiDiscrete format)."""
    action = torch.zeros(batch_size, 3 * n_bins, device=TEST_DEVICE)
    for axis in range(3):
        idx = torch.randint(0, n_bins, (batch_size,), device=TEST_DEVICE)
        action.scatter_(1, (idx + axis * n_bins).unsqueeze(1), 1.0)
    return action


def _collect_reward_batch(
    env: SofaEnv, num_steps: int
) -> tuple[torch.Tensor, torch.Tensor]:
    n_bins = env.n_bins
    td = env.reset()
    rewards: list[torch.Tensor] = []
    done: list[torch.Tensor] = []
    for _ in range(num_steps):
        td["action"] = _random_action(NUM_ENVS, n_bins)
        td = env.step(td)["next"]
        rewards.append(td["reward"].clone())
        done.append(td["done"].clone())
    return torch.stack(rewards, dim=1), torch.stack(done, dim=1)


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

        for rms in normalizer._obs_rms.values():
            assert rms.mean.device == TEST_DEVICE
        if normalizer._ret_rms is not None:
            assert normalizer._ret_rms.mean.device == TEST_DEVICE

    def test_boundary_actor_forward_updates_observation_stats(self) -> None:
        _cfg, env, normalizer, actor_module = _make_boundary_actor_stack()

        td = env.reset()
        actor_module(td)

        assert normalizer._obs_rms["sofa_rays"].count > 1e-4

    def test_freeze_stops_observation_stat_updates(self) -> None:
        _cfg, env, normalizer, actor_module = _make_boundary_actor_stack()

        td = env.reset()
        actor_module(td)
        counts_before = {name: rms.count for name, rms in normalizer._obs_rms.items()}

        normalizer.freeze = True
        actor_module(td)

        for name, rms in normalizer._obs_rms.items():
            assert rms.count == pytest.approx(counts_before[name])

    def test_tied_group_normalization_preserves_rotation(self) -> None:
        """Tied stats (scalar mean/std) must produce rotation-equivariant
        normalization — otherwise the circular-conv encoder's equivariance
        is broken downstream of the normalizer."""
        normalizer = Normalizer(
            obs_groups=[ObsGroupSpec("rays", 16, tied=True)],
            num_envs=1,
            device=TEST_DEVICE,
            norm_reward=False,
        )
        torch.manual_seed(0)
        x = torch.randn(8, 16)
        normalizer.normalize_group(x, "rays")
        normalizer.freeze = True
        shift = 3
        orig = normalizer.normalize_group(x, "rays")
        rolled = normalizer.normalize_group(x.roll(shift, dims=-1), "rays")
        assert torch.allclose(rolled, orig.roll(shift, dims=-1))

    def test_grid_observation_normalization_is_disabled_with_warning(self) -> None:
        config = _training_config(
            _test_cfg(observation_type="grid", boundary_rays=0),
            rollout_length=2,
        )
        with pytest.warns(RuntimeWarning, match="disabled for grid observations"):
            stack = build_training_stack(config)

        stack.collector.shutdown()
        assert stack.normalizer.norm_obs is False
        assert stack.normalizer._obs_rms == {}


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
        normalized = training_stack.normalizer.normalize_rewards(
            data["next", "reward"], data["next", "done"]
        )
        data["next"].set("reward", normalized)
        for key, raw_value in raw_reward_components.items():
            assert torch.equal(data["next", key], raw_value)


class TestStateAndIntegration:
    def test_save_load_roundtrip_for_frozen_eval(self) -> None:
        cfg, env, normalizer, actor_module = _make_boundary_actor_stack()

        td = env.reset()
        actor_module(td)
        state = normalizer.state_dict()

        normalizer.freeze = True
        frozen = _make_normalizer(cfg)
        frozen.load_state_dict(state)
        frozen.freeze = True

        sofa_view = td["observation", "sofa_view"]
        n = sofa_view.shape[-1] // 2
        groups = {
            "sofa_rays": sofa_view[..., :n],
            "corridor_rays": sofa_view[..., n:],
            "pose": td["observation", "pose"],
            "progress": td["observation", "progress"],
        }
        for name, group in groups.items():
            assert torch.allclose(
                normalizer.normalize_group(group, name),
                frozen.normalize_group(group, name),
            )

    def test_training_stack_boundary_rollouts_and_normalized_ppo(
        self, training_stack: TrainingStack
    ) -> None:
        config = _training_config()
        data = next(iter(training_stack.collector))
        raw_reward = data["next", "reward"].clone()

        assert data["observation", "sofa_view"].dtype == torch.float32
        assert training_stack.normalizer._obs_rms["sofa_rays"].count > 1e-4

        normalized_reward = training_stack.normalizer.normalize_rewards(
            data["next", "reward"], data["next", "done"]
        )
        data["next"].set("reward", normalized_reward)
        assert not torch.allclose(normalized_reward, raw_reward)

        training_stack.normalizer.freeze = True
        compute_gae_direct(
            data,
            training_stack.critic_net,
            config.gamma,
            config.gae_lambda,
        )
        optimization_stats = optimize_ppo_epochs(
            data.reshape(-1),
            training_stack.actor_net,
            training_stack.critic_net,
            training_stack.optimizer,
            config.num_epochs,
            config.minibatch_size,
            config.max_grad_norm,
            config.device,
            config.clip_epsilon,
            config.entropy_coeff,
            config.critic_coeff,
        )
        training_stack.normalizer.freeze = False

        assert isinstance(optimization_stats.loss_policy, float)
