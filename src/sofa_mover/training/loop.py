"""Training loop implementation."""

import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.rich import tqdm

from sofa_mover.evaluate import evaluate
from sofa_mover.training.config import TrainingConfig, resolve_training_config
from sofa_mover.training.stack import build_training_stack
from sofa_mover.training.utils import (
    compute_gae_minibatch,
    extract_episode_metrics,
    log_episode_metrics,
    maybe_build_episode_composite,
    optimize_ppo_epochs,
)


def run_training(config: TrainingConfig) -> Path:
    """Run PPO training for the sofa moving problem."""

    env_cfg, num_envs = resolve_training_config(config)
    frames_per_batch = num_envs * config.rollout_length
    output_path = Path(config.output_dir)
    output_path.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=config.log_dir)
    stack = build_training_stack(config, env_cfg, num_envs)

    # --- Training loop ---
    best_mean_area = 0.0
    batch_idx = 0
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
        elapsed = time.perf_counter() - t0
        fps = frames_per_batch / elapsed

        # Per-batch TensorBoard scalars
        mean_reward = data["next", "reward"].flatten().mean().item()
        writer.add_scalar("train/fps", fps, batch_idx)
        writer.add_scalar("train/mean_reward", mean_reward, batch_idx)
        writer.add_scalar("loss/policy", optimization_stats.loss_policy, batch_idx)
        writer.add_scalar("loss/critic", optimization_stats.loss_critic, batch_idx)
        writer.add_scalar("loss/entropy", optimization_stats.loss_entropy, batch_idx)
        writer.add_scalar("train/grad_norm", optimization_stats.grad_norm, batch_idx)

        episode_metrics = extract_episode_metrics(data_flat)
        log_episode_metrics(writer, episode_metrics, batch_idx)

        if episode_metrics is not None:
            composite = maybe_build_episode_composite(
                data_flat,
                stack.env,
                batch_idx,
                config.image_log_interval,
                last_done_idx=episode_metrics.last_done_idx,
            )
            if composite is not None:
                writer.add_image(
                    "episode/sofa_image",
                    composite,
                    batch_idx,
                    dataformats="HWC",
                )

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
                    "cfg": stack.env.cfg,
                    "batch_idx": batch_idx,
                    "best_mean_area": best_mean_area,
                },
                output_path / "best_policy.pt",
            )

        pbar.update(frames_per_batch)
        batch_idx += 1

    writer.close()

    # Save final
    final_path = output_path / "final_policy.pt"
    torch.save(
        {
            "actor": dict(stack.actor.state_dict()),
            "critic": dict(stack.critic.state_dict()),
            "encoder": dict(stack.actor_net.encoder.state_dict()),
            "cfg": stack.env.cfg,
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

    return final_path
