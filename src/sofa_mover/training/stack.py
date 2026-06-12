"""Training stack construction."""

import itertools
from dataclasses import dataclass

import torch
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, ValueOperator

from sofa_mover.env import SofaEnv, make_sofa_env
from sofa_mover.training.collector import RolloutCollector
from sofa_mover.networks import (
    MultiDiscreteCategorical,
    SofaActorNet,
    SofaCriticNet,
    build_encoder,
)
from sofa_mover.training.normalizer import Normalizer
from sofa_mover.training.config import TrainingConfig


@dataclass
class TrainingStack:
    """Runtime objects used by the training loop."""

    env: SofaEnv
    normalizer: Normalizer
    actor_net: SofaActorNet
    critic_net: SofaCriticNet
    actor: ProbabilisticActor
    critic: ValueOperator
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LinearLR
    collector: RolloutCollector


def build_actor(
    config: TrainingConfig,
    normalizer: Normalizer | None,
    device: torch.device,
) -> tuple[SofaActorNet, TensorDictModule, ProbabilisticActor]:
    """Build the actor net and its TorchRL wrappers."""
    nvec = config.env.nvec
    encoder = build_encoder(config, normalizer)
    actor_net = SofaActorNet(
        nvec=nvec,
        encoder=encoder,
        width=config.head_width,
        depth=config.head_depth,
    ).to(device)
    actor_module = TensorDictModule(
        actor_net,
        in_keys=[
            ("observation", "sofa_view"),
            ("observation", "pose"),
            ("observation", "progress"),
        ],
        out_keys=["logits"],
    )
    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=MultiDiscreteCategorical,
        distribution_kwargs={"nvec": nvec},
        return_log_prob=True,
    )
    return actor_net, actor_module, actor


def build_training_stack(
    config: TrainingConfig,
) -> TrainingStack:
    """Build env, modules, loss, optimizer, and collector."""
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
    actor_net, _, actor = build_actor(config, normalizer, device)
    critic_net = SofaCriticNet(
        encoder=actor_net.encoder,
        width=config.head_width,
        depth=config.head_depth,
    ).to(device)

    # Wrap critic for TorchRL
    critic = ValueOperator(
        module=critic_net,
        in_keys=[
            ("observation", "sofa_view"),
            ("observation", "pose"),
            ("observation", "progress"),
        ],
    )

    # --- Optimizer + LR schedule ---
    # actor_net.parameters() covers encoder + actor head;
    # critic_net.head covers the critic-only head (encoder is shared).
    optimizer = torch.optim.Adam(
        itertools.chain(actor_net.parameters(), critic_net.head.parameters()),
        lr=config.lr,
        fused=True,
    )
    anneal_frames = (
        config.lr_anneal_frames
        if config.lr_anneal_frames is not None
        else config.total_frames
    )
    anneal_batches = anneal_frames // (num_envs * config.rollout_length)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=config.lr_end_factor,
        total_iters=anneal_batches,
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
        actor=actor,
        critic=critic,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        collector=collector,
    )
