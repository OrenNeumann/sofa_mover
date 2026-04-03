"""Run PPO training for the sofa moving problem."""

import time
from pathlib import Path

import torch
from tqdm.rich import tqdm
from wandb.sdk import init as wandb_init
from wandb.sdk.data_types.image import Image

from sofa_mover.evaluate import evaluate
from sofa_mover.training.config import TrainingConfig
from sofa_mover.training.stack import build_training_stack
from sofa_mover.training.utils import (
    compute_gae_minibatch,
    extract_episode_metrics,
    maybe_build_episode_composite,
    optimize_ppo_epochs,
)


config = TrainingConfig()
output_path = Path(config.output_dir)
output_path.mkdir(exist_ok=True)
run = wandb_init(project=config.wandb_project)
run.define_metric("env_steps")
run.define_metric("*", step_metric="env_steps")
stack = build_training_stack(config)

# --- Training loop ---
best_mean_area = 0.0
batch_idx = 0
total_env_steps = 0
pbar = tqdm(
    total=config.total_frames,
    desc="Training",
    unit="step",
    unit_scale=True,
)

for data in stack.collector:
    t0 = time.perf_counter()
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

    # --- Logging ---
    batch_frames = data.numel()
    total_env_steps += batch_frames
    elapsed = time.perf_counter() - t0
    fps = batch_frames / elapsed

    mean_reward = data["next", "reward"].flatten().mean().item()
    log_payload: dict[str, float | int | Image] = {
        "env_steps": total_env_steps,
        "train/fps": fps,
        "train/mean_reward": mean_reward,
        "loss/policy": optimization_stats.loss_policy,
        "loss/critic": optimization_stats.loss_critic,
        "loss/entropy": optimization_stats.loss_entropy,
        "train/grad_norm": optimization_stats.grad_norm,
    }

    episode_metrics = extract_episode_metrics(data_flat)
    if episode_metrics is not None:
        log_payload.update(
            {
                "episode/terminal_area": episode_metrics.mean_terminal_area,
                "episode/truncation_rate": episode_metrics.truncation_rate,
                "episode/episode_length": episode_metrics.mean_ep_length,
                "episode/total_angle": episode_metrics.mean_total_angle,
                "episode/total_distance": episode_metrics.mean_total_distance,
                "episode/mean_area": episode_metrics.mean_area,
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

    pbar.set_postfix(
        fps=f"{fps:,.0f}",
        reward=f"{mean_reward:+.4f}",
        done="0" if episode_metrics is None else f"{episode_metrics.n_done:.0f}",
        area=(
            "0.0000"
            if episode_metrics is None
            else f"{episode_metrics.mean_terminal_area:.4f}"
        ),
    )

    # Save best
    if (
        episode_metrics is not None
        and episode_metrics.mean_terminal_area > best_mean_area
    ):
        best_mean_area = episode_metrics.mean_terminal_area
        torch.save(
            {
                "actor": dict(stack.actor.state_dict()),
                "critic": dict(stack.critic.state_dict()),
                "encoder": dict(stack.actor_net.encoder.state_dict()),
                "config": config,
                "batch_idx": batch_idx,
                "best_mean_area": best_mean_area,
            },
            output_path / "best_policy.pt",
        )

    pbar.update(batch_frames)
    batch_idx += 1

run.finish()

# Save final
final_path = output_path / "final_policy.pt"
torch.save(
    {
        "actor": dict(stack.actor.state_dict()),
        "critic": dict(stack.critic.state_dict()),
        "encoder": dict(stack.actor_net.encoder.state_dict()),
        "config": config,
        "batch_idx": batch_idx,
        "best_mean_area": best_mean_area,
    },
    final_path,
)
print(f"Training complete. Best mean terminal area: {best_mean_area:.4f}")
print(f"Saved to {final_path}")

# Visualize the best policy (fall back to final if no best was saved)
best_path = output_path / "best_policy.pt"
eval_path = best_path if best_path.exists() else final_path
evaluate(
    checkpoint_path=str(eval_path),
    output_path=str(output_path / "agent_trajectory.gif"),
    device=config.device,
)
