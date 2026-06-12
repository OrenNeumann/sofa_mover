"""Training stack construction."""

import itertools
from dataclasses import dataclass

import torch

from sofa_mover.env import SofaEnv, make_sofa_env
from sofa_mover.training.collector import RolloutCollector
from sofa_mover.networks import SofaActorNet, SofaCriticNet, build_actor_net
from sofa_mover.training.normalizer import Normalizer
from sofa_mover.training.config import TrainingConfig


@dataclass
class TrainingStack:
    """Runtime objects used by the training loop."""

    env: SofaEnv
    normalizer: Normalizer
    actor_net: SofaActorNet
    critic_net: SofaCriticNet
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LinearLR
    collector: RolloutCollector


def build_training_stack(
    config: TrainingConfig,
) -> TrainingStack:
    """Build env, networks, optimizer, and collector."""
    device = config.device
    num_envs = config.num_envs

    # --- Environment ---
    env = make_sofa_env(
        total_frames=config.total_frames,
        num_envs=num_envs,
        cfg=config.env,
        device=device,
    )
    normalizer = Normalizer.from_config(config, num_envs)

    # --- Networks (encoder selected from cfg, shared between actor & critic) ---
    actor_net = build_actor_net(config, normalizer).to(device)
    critic_net = SofaCriticNet(
        encoder=actor_net.encoder,
        width=config.head_width,
        depth=config.head_depth,
    ).to(device)

    # --- Optimizer + LR schedule ---
    # actor_net.parameters() covers encoder + actor head;
    # critic_net.head covers the critic-only head (encoder is shared).
    optimizer = torch.optim.Adam(
        itertools.chain(actor_net.parameters(), critic_net.head.parameters()),
        lr=config.lr,
        fused=True,
    )
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=config.lr_end_factor,
        total_iters=config.total_frames // (num_envs * config.rollout_length),
    )

    # --- Collector ---
    collector = RolloutCollector(
        env=env,
        actor_net=actor_net,
        rollout_length=config.rollout_length,
        total_frames=config.total_frames,
    )

    return TrainingStack(
        env=env,
        normalizer=normalizer,
        actor_net=actor_net,
        critic_net=critic_net,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        collector=collector,
    )
