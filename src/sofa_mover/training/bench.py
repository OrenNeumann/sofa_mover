"""FPS benchmark for the training loop.

Runs N short training runs using the existing training stack (no wandb, no
eval, no gif) and prints mean/std FPS. Used to compare against a skrl port.
"""

import argparse
import statistics
import time
from dataclasses import replace

import torch

from sofa_mover.training.config import TrainingConfig
from sofa_mover.training.normalizer import Normalizer
from sofa_mover.training.stack import build_training_stack
from sofa_mover.training.utils import compute_gae_direct, optimize_ppo_epochs


def run_one(total_frames: int) -> float:
    """Run one short training loop, return steps/sec excluding the first batch."""
    config = replace(TrainingConfig(), total_frames=total_frames)
    stack = build_training_stack(config)
    normalizer: Normalizer = stack.normalizer

    first_batch_end: float | None = None
    total_steps_after_warmup = 0
    batch_idx = 0

    for data in stack.collector:
        # not used:
        _next_data = data["next"]
        normalized_reward = normalizer.normalize_rewards(
            data["next", "reward"], data["next", "done"]
        )
        data["next"].set("reward", normalized_reward)

        normalizer.freeze = True
        compute_gae_direct(data, stack.critic_net, config.gamma, config.gae_lambda)
        data_flat = data.reshape(-1)
        optimize_ppo_epochs(
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
        )
        normalizer.freeze = False
        stack.lr_scheduler.step()

        torch.cuda.synchronize()
        if batch_idx == 0:
            first_batch_end = time.perf_counter()
        else:
            total_steps_after_warmup += data.numel()
        batch_idx += 1

    stack.collector.shutdown()
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
