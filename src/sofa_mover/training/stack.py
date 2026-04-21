"""Training stack construction."""

import itertools
from dataclasses import dataclass

import torch
from tensordict.nn import TensorDictModule
from torchrl.collectors import Collector
from torchrl.modules import ProbabilisticActor, ValueOperator

from sofa_mover.env import SofaEnv, make_sofa_env
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
        total_frames=config.total_frames,
        num_envs=num_envs,
        cfg=env_cfg,
        device=device,
    )
    normalizer = Normalizer.from_config(config, num_envs)

    # --- Networks (encoder selected from cfg) ---
    # bins for [-n·δ, ..., -δ, 0, +δ, ..., +n·δ]
    n_bins = 2 * env_cfg.n_magnitude_levels + 1
    nvec = [n_bins, n_bins, n_bins]

    encoder = build_encoder(config, normalizer)
    actor_net = SofaActorNet(
        nvec=nvec,
        encoder=encoder,
        width=config.head_width,
        depth=config.head_depth,
    ).to(device)
    critic_net = SofaCriticNet(
        encoder=actor_net.encoder,
        width=config.head_width,
        depth=config.head_depth,
    ).to(device)

    # Wrap actor for TorchRL
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
    total_batches = config.total_frames // (num_envs * config.rollout_length)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=config.lr_end_factor,
        total_iters=total_batches,
    )

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
        normalizer=normalizer,
        actor_net=actor_net,
        critic_net=critic_net,
        actor=actor,
        critic=critic,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        collector=collector,
    )
