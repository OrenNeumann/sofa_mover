"""Shared PPO agent construction for the skrl_sofa stack."""

from __future__ import annotations

import torch

from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.memories.torch import RandomMemory

from skrl_sofa.config import SkrlTrainingConfig
from skrl_sofa.env_wrapper import SkrlSofaEnv
from skrl_sofa.model import SharedSofaModel
from skrl_sofa.reward_shaper import RunningReturnRewardShaper


def build_ppo_agent(
    config: SkrlTrainingConfig,
    env: SkrlSofaEnv,
    model: SharedSofaModel,
    memory: RandomMemory,
    reward_shaper: RunningReturnRewardShaper,
) -> PPO:
    """Build the PPO agent used by train and benchmark."""
    ppo_cfg = PPO_CFG()
    ppo_cfg.rollouts = config.rollout_length
    ppo_cfg.learning_epochs = config.num_epochs
    ppo_cfg.mini_batches = config.mini_batches
    ppo_cfg.discount_factor = config.gamma
    ppo_cfg.gae_lambda = config.gae_lambda
    ppo_cfg.learning_rate = config.lr
    ppo_cfg.learning_rate_scheduler = torch.optim.lr_scheduler.LinearLR
    total_batches = config.total_frames // (config.num_envs * config.rollout_length)
    ppo_cfg.learning_rate_scheduler_kwargs = {
        "start_factor": 1.0,
        "end_factor": config.lr_end_factor,
        # skrl steps the scheduler once per learning epoch inside each PPO
        # update, not once per rollout like the torchrl path.
        "total_iters": total_batches * config.num_epochs,
    }
    ppo_cfg.grad_norm_clip = config.max_grad_norm
    ppo_cfg.ratio_clip = config.clip_epsilon
    ppo_cfg.value_clip = config.clip_epsilon
    ppo_cfg.entropy_loss_scale = config.entropy_coeff
    ppo_cfg.value_loss_scale = config.critic_coeff
    ppo_cfg.rewards_shaper = reward_shaper
    ppo_cfg.experiment.write_interval = 0
    ppo_cfg.experiment.checkpoint_interval = 0
    ppo_cfg.experiment.wandb = False

    # state_space=None skips allocation + per-step memcpy of a `states` tensor
    # that would just duplicate `observations`.
    agent = PPO(
        models={"policy": model, "value": model},
        memory=memory,
        observation_space=env.observation_space,
        state_space=None,
        action_space=env.action_space,
        device=config.device,
        cfg=ppo_cfg,
    )
    agent.init(trainer_cfg=None)
    agent.enable_training_mode(True)

    agent.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, fused=True)
    agent.checkpoint_modules["optimizer"] = agent.optimizer

    if agent.scheduler is not None:
        scheduler_cls = agent.cfg.learning_rate_scheduler[0]
        assert scheduler_cls is not None
        scheduler_kwargs = agent.cfg.learning_rate_scheduler_kwargs[0]
        agent.scheduler = scheduler_cls(agent.optimizer, **scheduler_kwargs)

    return agent
