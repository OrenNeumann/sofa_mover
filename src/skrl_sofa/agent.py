"""Shared PPO agent construction for the skrl_sofa stack."""

from __future__ import annotations

from typing import Any

import torch

from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.agents.torch.ppo import ppo as _skrl_ppo_module
from skrl.memories.torch import RandomMemory

from skrl_sofa.config import SkrlTrainingConfig
from skrl_sofa.env_wrapper import SkrlSofaEnv
from skrl_sofa.model import SharedSofaModel
from skrl_sofa.reward_shaper import RunningReturnRewardShaper

_SKRL_COMPUTE_GAE = _skrl_ppo_module.compute_gae


def compute_gae(
    *,
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    done: torch.Tensor,
    discount_factor: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE with per-step bootstrapping and a done-masked trace.

    `next_values` is the value of the true next state at every step (not the
    reset state), so truncations bootstrap correctly. `terminated` controls
    whether the bootstrap term is included; `done = terminated | truncated`
    controls where the trace is cut so advantages never leak across episodes.
    """
    delta = rewards + discount_factor * (~terminated).float() * next_values - values
    advantages = torch.empty_like(delta)
    gae = torch.zeros_like(delta[0])
    for t in reversed(range(rewards.shape[0])):
        gae = delta[t] + discount_factor * gae_lambda * (~done[t]).float() * gae
        advantages[t] = gae
    return advantages + values, advantages


class SofaPPO(PPO):
    """PPO agent used by the skrl pipeline.

    Thin wrapper around `skrl.PPO` that swaps in a corrected GAE. Stock skrl
    bootstraps only from the final `next_observations` (and uses `values[t+1]`
    for interior steps, which is wrong at truncation boundaries), cuts the
    trace on `terminated` rather than `done`, and normalizes advantages. Here
    we cache per-step `next_observations` and `done`, compute value estimates
    for the real next states once per rollout, and patch `compute_gae` for the
    duration of the base PPO update so the rest of the minibatch loop runs
    unchanged.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._rollout_done: list[torch.Tensor] = []
        self._rollout_next_obs: list[torch.Tensor] = []

    def record_transition(self, **kwargs: Any) -> None:
        super().record_transition(**kwargs)
        if self.training:
            self._rollout_done.append(kwargs["terminated"] | kwargs["truncated"])
            self._rollout_next_obs.append(kwargs["next_observations"])

    def update(self, *, timestep: int, timesteps: int) -> None:
        assert self.memory is not None
        done = torch.stack(self._rollout_done)
        next_obs = torch.stack(self._rollout_next_obs)
        self._rollout_done.clear()
        self._rollout_next_obs.clear()

        with (
            torch.no_grad(),
            torch.autocast(
                device_type=self._device_type, enabled=self.cfg.mixed_precision
            ),
        ):
            next_values_flat, _ = self.value.act(
                {
                    "observations": next_obs.reshape(-1, *next_obs.shape[2:]),
                    "states": None,
                },
                role="value",
            )

        def _patched_compute_gae(
            *,
            rewards: torch.Tensor,
            terminated: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,  # noqa: ARG001 — stock skrl only passes last step
            discount_factor: float,
            lambda_coefficient: float,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return compute_gae(
                rewards=rewards,
                values=values,
                next_values=next_values_flat.reshape_as(values),
                terminated=terminated,
                done=done,
                discount_factor=discount_factor,
                gae_lambda=lambda_coefficient,
            )

        _skrl_ppo_module.compute_gae = _patched_compute_gae
        try:
            super().update(timestep=timestep, timesteps=timesteps)
        finally:
            _skrl_ppo_module.compute_gae = _SKRL_COMPUTE_GAE


def build_ppo_agent(
    config: SkrlTrainingConfig,
    env: SkrlSofaEnv,
    model: SharedSofaModel,
    memory: RandomMemory,
    reward_shaper: RunningReturnRewardShaper,
) -> SofaPPO:
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
        # skrl's base update steps the scheduler once per learning epoch.
        "total_iters": total_batches * config.num_epochs,
    }
    ppo_cfg.grad_norm_clip = config.max_grad_norm
    ppo_cfg.ratio_clip = config.clip_epsilon
    ppo_cfg.value_clip = 0.0
    ppo_cfg.entropy_loss_scale = config.entropy_coeff
    ppo_cfg.value_loss_scale = config.critic_coeff
    ppo_cfg.rewards_shaper = reward_shaper
    ppo_cfg.experiment.write_interval = 0
    ppo_cfg.experiment.checkpoint_interval = 0
    ppo_cfg.experiment.wandb = False

    # state_space=None skips allocation + per-step memcpy of a `states` tensor
    # that would just duplicate `observations`.
    agent = SofaPPO(
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
