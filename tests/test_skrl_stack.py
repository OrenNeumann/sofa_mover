"""Smoke test for the skrl_sofa stack.

Runs a tiny end-to-end loop: env wrapper -> PPO agent -> record transitions
-> post_interaction -> PPO update. Asserts that loss scalars appear in
agent.tracking_data (i.e. the update actually ran).
"""

import torch
from torch.optim.lr_scheduler import LinearLR

from skrl.memories.torch import RandomMemory

from sofa_mover.training.normalizer import Normalizer

from skrl_sofa.agent import build_ppo_agent
from skrl_sofa.config import SkrlTrainingConfig
from skrl_sofa.env_wrapper import SkrlSofaEnv
from skrl_sofa.model import SharedSofaModel
from skrl_sofa.reward_shaper import RunningReturnRewardShaper


def _build_skrl_components(
    cfg: SkrlTrainingConfig,
) -> tuple[
    SkrlSofaEnv,
    Normalizer,
    RunningReturnRewardShaper,
    SharedSofaModel,
    RandomMemory,
]:
    device = cfg.device
    env = SkrlSofaEnv(
        num_envs=cfg.num_envs,
        total_frames=cfg.total_frames,
        cfg=cfg.env,
        device=device,
    )
    normalizer = Normalizer(
        obs_dim=cfg.env.boundary_rays + 4,
        num_envs=cfg.num_envs,
        device=device,
        norm_obs=True,
        norm_reward=False,
    )
    shaper = RunningReturnRewardShaper(
        num_envs=cfg.num_envs, device=device, gamma=cfg.gamma, clip=10.0
    )
    model = SharedSofaModel(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        n_rays=cfg.env.boundary_rays,
        normalizer=normalizer,
    ).to(device)
    memory = RandomMemory(
        memory_size=cfg.rollout_length,
        num_envs=cfg.num_envs,
        device=device,
        replacement=False,
    )
    return env, normalizer, shaper, model, memory


def test_skrl_stack_smoke() -> None:
    cfg = SkrlTrainingConfig(
        num_envs=8,
        total_frames=4096,
        rollout_length=16,
        minibatch_size=32,
        num_epochs=2,
    )
    env, normalizer, shaper, model, memory = _build_skrl_components(cfg)

    agent = build_ppo_agent(
        config=cfg,
        env=env,
        model=model,
        memory=memory,
        reward_shaper=shaper,
    )

    obs, _ = env.reset()
    assert obs.shape == (cfg.num_envs, cfg.env.boundary_rays + 4)

    total_timesteps = 2 * cfg.rollout_length  # two full PPO updates
    for t in range(total_timesteps):
        with torch.no_grad():
            actions, _ = agent.act(
                obs, states=None, timestep=t, timesteps=total_timesteps
            )
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        shaper.set_done(info["done"])
        agent.record_transition(
            observations=obs,
            states=None,
            actions=actions,
            rewards=rewards,
            next_observations=next_obs,
            next_states=None,
            terminated=terminated,
            truncated=truncated,
            infos=info,
            timestep=t,
            timesteps=total_timesteps,
        )
        normalizer.freeze = True
        shaper.freeze = True
        agent.post_interaction(timestep=t, timesteps=total_timesteps)
        normalizer.freeze = False
        shaper.freeze = False
        obs = next_obs

    # Two PPO updates should have populated the loss trackers.
    losses = agent.tracking_data
    assert len(losses["Loss / Policy loss"]) == 2
    assert len(losses["Loss / Value loss"]) == 2
    # Reward shaper running stats should have advanced from init.
    assert shaper._ret_rms.count > 1e-4


def test_build_ppo_agent_uses_fused_adam() -> None:
    cfg = SkrlTrainingConfig(
        num_envs=8,
        total_frames=4096,
        rollout_length=16,
        minibatch_size=32,
        num_epochs=2,
    )
    env, _normalizer, shaper, model, memory = _build_skrl_components(cfg)

    agent = build_ppo_agent(
        config=cfg,
        env=env,
        model=model,
        memory=memory,
        reward_shaper=shaper,
    )

    assert agent.optimizer.defaults["fused"] is True
    assert agent.checkpoint_modules["optimizer"] is agent.optimizer
    assert agent.scheduler is not None
    assert agent.scheduler.optimizer is agent.optimizer


def test_skrl_linear_lr_spans_all_updates() -> None:
    cfg = SkrlTrainingConfig()
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(()))], lr=cfg.lr)
    total_updates = cfg.total_frames // (cfg.num_envs * cfg.rollout_length)
    scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=cfg.lr_end_factor,
        total_iters=total_updates * cfg.num_epochs,
    )

    for _ in range(total_updates):
        optimizer.step()
        scheduler.step()
        assert scheduler.get_last_lr()[0] > 0.0
        for _ in range(cfg.num_epochs - 1):
            optimizer.step()
            scheduler.step()

    assert scheduler.get_last_lr()[0] == 0.0
