"""Training stack construction."""

from dataclasses import dataclass

import torch
from tensordict.nn import TensorDictModule
from torchrl.collectors import Collector
from torchrl.modules import OneHotCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from sofa_mover.env import SofaEnv, make_sofa_env
from sofa_mover.networks import (
    SofaActorNet,
    SofaBoundaryEncoder,
    SofaCriticNet,
    SofaEncoder,
)
from sofa_mover.training.config import TrainingConfig


@dataclass
class TrainingStack:
    """Runtime objects used by the training loop."""

    env: SofaEnv
    actor_net: SofaActorNet
    actor: ProbabilisticActor
    critic: ValueOperator
    loss_module: ClipPPOLoss
    optimizer: torch.optim.Optimizer
    collector: Collector


def build_training_stack(
    config: TrainingConfig,
) -> TrainingStack:
    """Build env, modules, loss, optimizer, and collector."""
    device = config.device
    num_envs = config.num_envs
    env_cfg = config.env

    # --- Environment ---
    env = make_sofa_env(
        num_envs=num_envs,
        cfg=env_cfg,
        device=device,
    )

    # --- Networks (encoder selected from cfg) ---
    encoder: SofaEncoder | SofaBoundaryEncoder
    if env_cfg.observation_type == "boundary":
        encoder = SofaBoundaryEncoder(n_rays=env_cfg.boundary_rays)
    else:  # "grid"
        encoder = SofaEncoder()
    actor_net = SofaActorNet(encoder=encoder).to(device)
    critic_net = SofaCriticNet(encoder=actor_net.encoder).to(device)

    # Wrap actor for TorchRL
    actor_module = TensorDictModule(
        actor_net,
        in_keys=["observation", "pose", "progress"],
        out_keys=["logits"],
    )
    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        return_log_prob=True,
    )

    # Wrap critic for TorchRL
    critic = ValueOperator(
        module=critic_net,
        in_keys=["observation", "pose", "progress"],
    )

    # --- PPO Loss ---
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=config.clip_epsilon,
        entropy_bonus=True,
        entropy_coeff=config.entropy_coeff,
        critic_coeff=config.critic_coeff,
    )
    loss_module.make_value_estimator(
        GAE,
        gamma=config.gamma,
        lmbda=config.gae_lambda,
        value_network=critic,
    )

    # --- Optimizer ---
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=config.lr)

    # --- Collector ---
    collector = Collector(
        create_env_fn=env,
        policy=actor,
        frames_per_batch=num_envs * config.rollout_length,
        total_frames=config.total_frames,
        device=device,
    )

    return TrainingStack(
        env=env,
        actor_net=actor_net,
        actor=actor,
        critic=critic,
        loss_module=loss_module,
        optimizer=optimizer,
        collector=collector,
    )
