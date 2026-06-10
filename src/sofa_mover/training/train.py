"""Run PPO training for the sofa moving problem."""

import time
from pathlib import Path

import torch
from tqdm import tqdm
from wandb.sdk import init as wandb_init
from wandb.sdk.data_types.image import Image

from sofa_mover.evaluate import evaluate
from sofa_mover.optimize_trajectory import optimize_trajectory
from sofa_mover.training.best_tracker import BestEpisodeTracker
from sofa_mover.training.config import TrainingConfig
from sofa_mover.training.normalizer import Normalizer
from sofa_mover.training.stack import build_training_stack
from sofa_mover.training.utils import (
    compute_gae_direct,
    extract_episode_metrics,
    maybe_build_episode_composite,
    optimize_ppo_epochs,
)


def main() -> None:
    config = TrainingConfig()
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    run = wandb_init(project=config.wandb_project)
    run.define_metric("total_steps")
    run.define_metric("*", step_metric="total_steps")
    stack = build_training_stack(config)
    normalizer: Normalizer = stack.normalizer

    best_tracker = BestEpisodeTracker(
        num_envs=config.num_envs,
        max_steps=config.env.max_steps,
        n_bins=stack.env.n_bins,
        device=config.device,
    )

    best_area_so_far = 0.0
    batch_idx = 0
    total_steps = 0
    training_start = time.perf_counter()

    def checkpoint_payload(include_encoder: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "actor": dict(stack.actor.state_dict()),
            "critic": dict(stack.critic.state_dict()),
            "vec_normalize": normalizer.state_dict(),
            "config": config,
            "batch_idx": batch_idx,
            "best_area_so_far": best_area_so_far,
            "best_trajectory_actions": best_tracker.best_actions,
            "best_trajectory_area": best_tracker.best_area,
        }
        if include_encoder:
            payload["encoder"] = dict(stack.actor_net.encoder.state_dict())
        return payload

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
        mean_reward = next_data["reward"].flatten().mean().item()
        best_tracker.update(
            data["action"],
            next_data["terminal_area"].squeeze(-1),
            next_data["done"].squeeze(-1),
        )
        normalized_reward = normalizer.normalize_rewards(
            data["next", "reward"], data["next", "done"]
        )
        data["next"].set("reward", normalized_reward)

        normalizer.freeze = True
        compute_gae_direct(
            data,
            stack.critic_net,
            config.gamma,
            config.gae_lambda,
        )

        data_flat = data.reshape(-1)
        optimization_stats = optimize_ppo_epochs(
            data_flat,
            stack.actor_net,
            stack.critic_net,
            stack.optimizer,
            config.num_epochs,
            config.minibatch_size,
            config.max_grad_norm,
            config.device,
            config.clip_epsilon,
            config.entropy_coeff,
            config.critic_coeff,
            target_kl=config.target_kl,
        )
        normalizer.freeze = False
        stack.lr_scheduler.step()

        batch_frames = data.numel()
        total_steps += batch_frames
        walltime = time.perf_counter() - training_start
        batch_fps = batch_frames / (time.perf_counter() - t0)

        log_payload: dict[str, float | int | Image] = {
            "total_steps": total_steps,
            "train/walltime": walltime,
            "train/steps_per_second": total_steps / walltime,
            "train/batch_fps": batch_fps,
            "train/mean_reward_normalized": normalized_reward.flatten().mean().item(),
            "train/mean_reward_raw": mean_reward,
            "reward/shaping_scale": stack.env.shaping_scale,
            "reward/erosion": next_data["reward_erosion"].flatten().mean().item(),
            "reward/progress": next_data["reward_progress"].flatten().mean().item(),
            "reward/terminal": next_data["reward_terminal"].flatten().mean().item(),
            "loss/policy": optimization_stats.loss_policy,
            "loss/critic": optimization_stats.loss_critic,
            "loss/entropy": optimization_stats.loss_entropy,
            "train/grad_norm": optimization_stats.grad_norm,
            "train/approx_kl": optimization_stats.approx_kl,
            "train/clip_fraction": optimization_stats.clip_fraction,
            "train/ppo_epochs_completed": optimization_stats.epochs_completed,
            "train/lr": float(stack.lr_scheduler.get_last_lr()[0]),
        }

        episode_metrics = extract_episode_metrics(data_flat)
        if episode_metrics is not None:
            log_payload.update(
                {
                    "episode/area_at_goal": episode_metrics.area_at_goal,
                    "episode/best_area_at_goal": episode_metrics.best_area_at_goal,
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

        if (
            episode_metrics is not None
            and episode_metrics.best_area_at_goal > best_area_so_far
        ):
            best_area_so_far = episode_metrics.best_area_at_goal
            torch.save(
                checkpoint_payload(include_encoder=False),
                output_path / "best_policy.pt",
            )

        pbar.update(batch_frames)
        batch_idx += 1

    pbar.close()
    run.finish()
    stack.collector.shutdown()

    final_path = output_path / "final_policy.pt"
    torch.save(checkpoint_payload(include_encoder=True), final_path)
    print(f"Training complete. Best area at goal: {best_area_so_far:.4f}")
    print(f"Saved to {final_path}")

    best_path = output_path / "best_policy.pt"
    eval_path = best_path if best_path.exists() else final_path
    evaluate(
        checkpoint_path=str(eval_path),
        output_path=str(output_path / "agent_trajectory.gif"),
        device=config.device,
    )

    # Post-training refinement, with vanilla non-ML optimization.
    if best_tracker.best_actions is not None:
        optimize_trajectory(
            best_tracker.best_actions,
            config,
            output_path / "optimized_trajectory.gif",
        )


if __name__ == "__main__":
    main()
