"""FPS benchmark for the skrl_sofa training loop.

Mirrors `sofa_mover.training.bench` so the two pipelines can be compared
directly. No wandb, no eval, no gif. First rollout is excluded from the FPS
calculation as CUDA warmup.
"""

import argparse
import statistics
import time
from dataclasses import replace

import torch

from skrl.memories.torch import RandomMemory

from sofa_mover.training.normalizer import Normalizer

from skrl_sofa.agent import build_ppo_agent
from skrl_sofa.config import SkrlTrainingConfig
from skrl_sofa.env_wrapper import SkrlSofaEnv
from skrl_sofa.model import SharedSofaModel
from skrl_sofa.reward_shaper import RunningReturnRewardShaper


def run_one(total_frames: int) -> float:
    """Run one short training loop, return steps/sec excluding the first batch."""
    config = replace(SkrlTrainingConfig(), total_frames=total_frames)
    device = config.device

    env = SkrlSofaEnv(
        num_envs=config.num_envs,
        total_frames=config.total_frames,
        cfg=config.env,
        device=device,
    )
    obs_dim = config.env.boundary_rays + 3 + 1
    normalizer = Normalizer(
        obs_dim=obs_dim,
        num_envs=config.num_envs,
        device=device,
        gamma=config.gamma,
        norm_obs=config.normalize_observation,
        norm_reward=False,
    )
    shaper = RunningReturnRewardShaper(
        num_envs=config.num_envs,
        device=device,
        gamma=config.gamma,
        clip=10.0,
    )
    model = SharedSofaModel(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        n_rays=config.env.boundary_rays,
        normalizer=normalizer,
    ).to(device)
    memory = RandomMemory(
        memory_size=config.rollout_length,
        num_envs=config.num_envs,
        device=device,
        replacement=False,
    )

    agent = build_ppo_agent(
        config=config,
        env=env,
        model=model,
        memory=memory,
        reward_shaper=shaper,
    )

    num_envs = config.num_envs
    rollout_length = config.rollout_length
    total_timesteps = config.total_frames // num_envs
    batch_frames = rollout_length * num_envs

    obs, _ = env.reset()
    first_batch_end: float | None = None
    total_steps_after_warmup = 0
    batch_idx = 0

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

        if (t + 1) % rollout_length == 0:
            torch.cuda.synchronize()
            if batch_idx == 0:
                first_batch_end = time.perf_counter()
            else:
                total_steps_after_warmup += batch_frames
            batch_idx += 1

    assert first_batch_end is not None
    walltime = time.perf_counter() - first_batch_end
    return total_steps_after_warmup / walltime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--total-frames", type=int, default=200_000)
    args = parser.parse_args()

    fps_samples: list[float] = []
    for i in range(args.trials):
        fps = run_one(args.total_frames)
        fps_samples.append(fps)
        print(f"trial {i + 1}/{args.trials}: {fps:,.0f} fps")

    mean = statistics.mean(fps_samples)
    stdev = statistics.stdev(fps_samples) if len(fps_samples) > 1 else 0.0
    print(f"\nfps mean ± std  = {mean:,.0f} ± {stdev:,.0f}")
    print(f"fps min / max   = {min(fps_samples):,.0f} / {max(fps_samples):,.0f}")


if __name__ == "__main__":
    main()
