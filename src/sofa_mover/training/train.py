"""Run PPO training for the sofa moving problem."""

import time
from pathlib import Path

import torch
from tqdm.rich import tqdm
from wandb.sdk import init as wandb_init
from wandb.sdk.data_types.image import Image

from sofa_mover.evaluate import evaluate
from sofa_mover.training.config import TrainingConfig
from sofa_mover.training.normalizer import Normalizer
from sofa_mover.training.stack import build_training_stack
from sofa_mover.training.utils import (
    compute_gae_minibatch,
    extract_episode_metrics,
    maybe_build_episode_composite,
    normalize_rewards_inplace,
    optimize_ppo_epochs,
)


config = TrainingConfig()
output_path = Path(config.output_dir)
output_path.mkdir(exist_ok=True)
run = wandb_init(project=config.wandb_project)
run.define_metric("env_steps")
run.define_metric("*", step_metric="env_steps")
stack = build_training_stack(config)
normalizer: Normalizer = stack.normalizer

# --- Training loop ---
best_area_at_goal = 0.0
batch_idx = 0
total_env_steps = 0
pbar = tqdm(
    total=config.total_frames,
    desc="Training",
    unit="step",
    unit_scale=False,
    mininterval=0.1,
)

for data in stack.collector:
    t0 = time.perf_counter()
    next_data = data["next"]
    # Reward shaping: anneal secondary rewards
    anneal_end = config.total_frames * config.reward_anneal_time
    shaping_scale = max(0.0, 1.0 - total_env_steps / anneal_end)
    next_data.set("reward_erosion", next_data["reward_erosion"] * shaping_scale)
    next_data.set("reward_progress", next_data["reward_progress"] * shaping_scale)
    next_data.set(
        "reward",
        next_data["reward_erosion"]
        + next_data["reward_progress"]
        + next_data["reward_terminal"],
    )
    mean_reward = next_data["reward"].flatten().mean().item()
    normalized_reward = normalize_rewards_inplace(data, normalizer)

    normalizer.freeze = True
    # Compute GAE advantages in minibatches (avoids converting the full
    # obs buffer to float32 at once, which OOMs for large B)
    compute_gae_minibatch(data, stack.loss_module, config.minibatch_size)

    # Flatten time dimension for minibatch iteration
    data_flat = data.reshape(-1)
    optimization_stats = optimize_ppo_epochs(
        data_flat,
        stack.loss_module,
        stack.optimizer,
        config.num_epochs,
        config.minibatch_size,
        config.max_grad_norm,
        config.device,
    )
    normalizer.freeze = False
    stack.lr_scheduler.step()

    # --- Logging ---
    batch_frames = data.numel()
    total_env_steps += batch_frames
    elapsed = time.perf_counter() - t0
    fps = batch_frames / elapsed

    mean_erosion = next_data["reward_erosion"].flatten().mean().item()
    mean_progress = next_data["reward_progress"].flatten().mean().item()
    mean_terminal = next_data["reward_terminal"].flatten().mean().item()
    log_payload: dict[str, float | int | Image] = {
        "env_steps": total_env_steps,
        "train/fps": fps,
        "train/mean_reward_normalized": normalized_reward.flatten().mean().item(),
        "train/mean_reward_raw": mean_reward,
        "reward/shaping_scale": shaping_scale,
        "reward/erosion": mean_erosion,
        "reward/progress": mean_progress,
        "reward/terminal": mean_terminal,
        "loss/policy": optimization_stats.loss_policy,
        "loss/critic": optimization_stats.loss_critic,
        "loss/entropy": optimization_stats.loss_entropy,
        "train/grad_norm": optimization_stats.grad_norm,
        "train/lr": stack.lr_scheduler.get_last_lr()[0],
    }

    episode_metrics = extract_episode_metrics(data_flat)
    if episode_metrics is not None:
        log_payload.update(
            {
                "episode/area_at_goal": episode_metrics.area_at_goal,
                "episode/goal_rate": episode_metrics.goal_rate,
                "episode/truncation_rate": episode_metrics.truncation_rate,
                "episode/episode_length": episode_metrics.mean_ep_length,
                "episode/total_angle": episode_metrics.mean_total_angle,
                "episode/total_distance": episode_metrics.mean_total_distance,
            }
        )
        composite = maybe_build_episode_composite(
            data_flat,
            stack.env,
            batch_idx,
            config.image_log_interval,
            last_done_idx=episode_metrics.last_done_idx,
        )
        if composite is not None:
            log_payload["episode/sofa_image"] = Image(composite)

    run.log(log_payload)

    # Save best
    if episode_metrics is not None and episode_metrics.area_at_goal > best_area_at_goal:
        best_area_at_goal = episode_metrics.area_at_goal
        torch.save(
            {
                "actor": dict(stack.actor.state_dict()),
                "critic": dict(stack.critic.state_dict()),
                "encoder": dict(stack.actor_net.encoder.state_dict()),
                "vec_normalize": normalizer.state_dict(),
                "config": config,
                "batch_idx": batch_idx,
                "best_area_at_goal": best_area_at_goal,
            },
            output_path / "best_policy.pt",
        )

    pbar.update(batch_frames)
    batch_idx += 1

run.finish()
stack.collector.shutdown()

# Save final
final_path = output_path / "final_policy.pt"
torch.save(
    {
        "actor": dict(stack.actor.state_dict()),
        "critic": dict(stack.critic.state_dict()),
        "encoder": dict(stack.actor_net.encoder.state_dict()),
        "vec_normalize": normalizer.state_dict(),
        "config": config,
        "batch_idx": batch_idx,
        "best_area_at_goal": best_area_at_goal,
    },
    final_path,
)
print(f"Training complete. Best area at goal: {best_area_at_goal:.4f}")
print(f"Saved to {final_path}")

# Visualize the best policy (fall back to final if no best was saved)
best_path = output_path / "best_policy.pt"
eval_path = best_path if best_path.exists() else final_path
evaluate(
    checkpoint_path=str(eval_path),
    output_path=str(output_path / "agent_trajectory.gif"),
    device=config.device,
)
