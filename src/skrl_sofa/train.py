"""Manual skrl PPO training loop for the sofa moving problem.

Mirrors `sofa_mover.training.train` (wandb logging, best-policy checkpoint,
episode metrics) but uses `skrl`'s `PPO` agent + `RandomMemory` instead of the
torchrl collector + bespoke optimizer.

Why a manual loop (Option B)?
    Option A would wrap everything in a `SequentialTrainer` and attach a
    custom agent callback for wandb. That requires funneling our per-env
    episode metrics through skrl's `track_data` interface and wrestling with
    its experiment dir / tensorboard writer, which is more plumbing than
    running the loop ourselves. Manual gives direct access to PPO internals
    (pre_interaction / act / record_transition / post_interaction) while
    still letting skrl handle GAE + minibatch PPO.

Flow per env step:
    1. agent.pre_interaction            (no-op in PPO but kept for symmetry)
    2. agent.act(obs, states, ...)      -> actions, outputs
    3. shaper.set_done(info["done"])    (plumbs dones into the reward shaper)
    4. env.step(actions)                -> next_obs, reward, term, trunc, info
    5. agent.record_transition(...)     (shaper runs here, then memory.add)
    6. normalizer.freeze = True
       agent.post_interaction()         (triggers update every `rollouts` steps)
       normalizer.freeze = False
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from tqdm import tqdm
from wandb.sdk import init as wandb_init

from skrl.memories.torch import RandomMemory
from skrl.resources.schedulers.torch import KLAdaptiveLR  # noqa: F401  # import sanity

from sofa_mover.training.normalizer import Normalizer

from skrl_sofa.agent import build_ppo_agent
from skrl_sofa.config import SkrlTrainingConfig
from skrl_sofa.env_wrapper import SkrlSofaEnv
from skrl_sofa.model import SharedSofaModel
from skrl_sofa.reward_shaper import RunningReturnRewardShaper


def _build_normalizer(cfg: SkrlTrainingConfig) -> Normalizer:
    """Obs-only normalizer (reward normalization is handled by the shaper)."""
    assert cfg.env.observation_type == "boundary"
    obs_dim = cfg.env.boundary_rays + 3 + 1
    return Normalizer(
        obs_dim=obs_dim,
        num_envs=cfg.num_envs,
        device=cfg.device,
        gamma=cfg.gamma,
        norm_obs=cfg.normalize_observation,
        norm_reward=False,
    )


def _aggregate_episode_metrics(
    step_infos: list[dict[str, torch.Tensor]],
) -> dict[str, float] | None:
    """Aggregate per-step info dicts into wandb-ready episode metrics.

    Returns None if no env finished an episode during the rollout.
    """
    dones = torch.cat([i["done"].reshape(-1) for i in step_infos])  # (T*B,)
    if not dones.any():
        return None

    def _gather(key: str) -> torch.Tensor:
        return torch.cat([i[key].reshape(-1) for i in step_infos])[dones]

    terminal_area = _gather("terminal_area")
    ep_length = _gather("episode_length").float()
    ep_angle = _gather("episode_total_angle")
    ep_dist = _gather("episode_total_distance")
    return {
        "episode/area_at_goal": terminal_area.mean().item(),
        "episode/goal_rate": (terminal_area > 0).float().mean().item(),
        "episode/episode_length": ep_length.mean().item(),
        "episode/total_angle": ep_angle.mean().item(),
        "episode/total_distance": ep_dist.mean().item(),
    }


def _aggregate_rollout_rewards(
    step_infos: list[dict[str, torch.Tensor]],
    raw_rewards: list[torch.Tensor],
) -> dict[str, float]:
    erosion = torch.cat([i["reward_erosion"].reshape(-1) for i in step_infos])
    progress = torch.cat([i["reward_progress"].reshape(-1) for i in step_infos])
    terminal = torch.cat([i["reward_terminal"].reshape(-1) for i in step_infos])
    raw = torch.cat([r.reshape(-1) for r in raw_rewards])
    return {
        "train/mean_reward_raw": raw.mean().item(),
        "reward/erosion": erosion.mean().item(),
        "reward/progress": progress.mean().item(),
        "reward/terminal": terminal.mean().item(),
    }


def main() -> None:
    config = SkrlTrainingConfig()
    output_path = Path(config.output_dir)
    output_path.mkdir(exist_ok=True)

    run = wandb_init(project=config.wandb_project)
    run.define_metric("total_steps")
    run.define_metric("*", step_metric="total_steps")

    device = config.device

    # --- Env, normalizer, reward shaper ---
    env = SkrlSofaEnv(
        num_envs=config.num_envs,
        total_frames=config.total_frames,
        cfg=config.env,
        device=device,
    )
    normalizer = _build_normalizer(config)
    reward_shaper = RunningReturnRewardShaper(
        num_envs=config.num_envs,
        device=device,
        gamma=config.gamma,
        clip=10.0,
    )

    # --- Shared actor-critic model ---
    model = SharedSofaModel(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        n_rays=config.env.boundary_rays,
        normalizer=normalizer,
    )
    # --- Memory ---
    memory = RandomMemory(
        memory_size=config.rollout_length,
        num_envs=config.num_envs,
        device=device,
        replacement=False,
    )

    # --- PPO agent ---
    agent = build_ppo_agent(
        config=config,
        env=env,
        model=model,
        memory=memory,
        reward_shaper=reward_shaper,
    )

    # --- Training loop ---
    num_envs = config.num_envs
    total_frames = config.total_frames
    rollout_length = config.rollout_length
    total_timesteps = total_frames // num_envs

    best_area_at_goal = 0.0
    batch_idx = 0
    total_steps = 0
    training_start = time.perf_counter()

    pbar = tqdm(
        total=total_frames,
        desc="Training",
        unit="step",
        unit_scale=False,
        mininterval=0.1,
    )

    obs, _ = env.reset()

    step_infos: list[dict[str, torch.Tensor]] = []
    raw_rewards: list[torch.Tensor] = []
    rollout_t0 = time.perf_counter()

    for timestep in range(total_timesteps):
        agent.pre_interaction(timestep=timestep, timesteps=total_timesteps)

        with torch.no_grad():
            actions, _outputs = agent.act(
                obs, states=None, timestep=timestep, timesteps=total_timesteps
            )

        next_obs, rewards, terminated, truncated, info = env.step(actions)
        raw_rewards.append(rewards.detach())
        step_infos.append(info)

        # Shaper needs the per-env done mask for this step so it can zero the
        # running return at episode boundaries (skrl's shaper callback
        # signature only receives `(rewards, timestep, timesteps)`).
        reward_shaper.set_done(info["done"])

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
            timestep=timestep,
            timesteps=total_timesteps,
        )

        # Freeze the obs normalizer around the PPO update so replayed
        # minibatches don't double-count into the running stats.
        normalizer.freeze = True
        reward_shaper.freeze = True
        agent.post_interaction(timestep=timestep, timesteps=total_timesteps)
        normalizer.freeze = False
        reward_shaper.freeze = False

        obs = next_obs

        # Rollout boundary: flush logging.
        if (timestep + 1) % rollout_length == 0:
            batch_frames = rollout_length * num_envs
            total_steps += batch_frames
            walltime = time.perf_counter() - training_start
            batch_fps = batch_frames / (time.perf_counter() - rollout_t0)

            log_payload: dict[str, float | int] = {
                "total_steps": total_steps,
                "train/walltime": walltime,
                "train/steps_per_second": total_steps / walltime,
                "train/batch_fps": batch_fps,
                "train/lr": (
                    agent.scheduler.get_last_lr()[0]
                    if agent.scheduler is not None
                    else config.lr
                ),
            }
            log_payload.update(_aggregate_rollout_rewards(step_infos, raw_rewards))

            # skrl stashes loss scalars in agent.tracking_data (deque of floats).
            tracking = agent.tracking_data
            if "Loss / Policy loss" in tracking and tracking["Loss / Policy loss"]:
                log_payload["loss/policy"] = tracking["Loss / Policy loss"][-1]
            if "Loss / Value loss" in tracking and tracking["Loss / Value loss"]:
                log_payload["loss/critic"] = tracking["Loss / Value loss"][-1]
            if "Loss / Entropy loss" in tracking and tracking["Loss / Entropy loss"]:
                log_payload["loss/entropy"] = tracking["Loss / Entropy loss"][-1]

            ep_metrics = _aggregate_episode_metrics(step_infos)
            if ep_metrics is not None:
                log_payload.update(ep_metrics)
                area_at_goal = ep_metrics["episode/area_at_goal"]
                if area_at_goal > best_area_at_goal:
                    best_area_at_goal = area_at_goal
                    torch.save(
                        {
                            "model": dict(model.state_dict()),
                            "vec_normalize": normalizer.state_dict(),
                            "reward_shaper": reward_shaper.state_dict(),
                            "config": config,
                            "batch_idx": batch_idx,
                            "best_area_at_goal": best_area_at_goal,
                        },
                        output_path / "best_policy.pt",
                    )

            run.log(log_payload)
            pbar.update(batch_frames)
            batch_idx += 1
            step_infos = []
            raw_rewards = []
            rollout_t0 = time.perf_counter()

    pbar.close()
    run.finish()

    final_path = output_path / "final_policy.pt"
    torch.save(
        {
            "model": dict(model.state_dict()),
            "vec_normalize": normalizer.state_dict(),
            "reward_shaper": reward_shaper.state_dict(),
            "config": config,
            "batch_idx": batch_idx,
            "best_area_at_goal": best_area_at_goal,
        },
        final_path,
    )
    print(f"Training complete. Best area at goal: {best_area_at_goal:.4f}")
    print(f"Saved to {final_path}")


if __name__ == "__main__":
    main()
