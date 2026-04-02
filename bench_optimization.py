"""Benchmark script for obs mode configurations. Temporary, replace
with proper benchmarking.

Measures per mode:
  1. Max batch size that fits in GPU memory (binary search)
  2. Training FPS at max batch size
  3. Peak GPU memory usage
"""

import gc
import time
from dataclasses import dataclass

import torch

from sofa_mover.corridor import DEVICE
from sofa_mover.env import SofaEnvConfig
from sofa_mover.obs_mode import ObsModeName, make_env_config
from sofa_mover.training.config import TrainingConfig, resolve_training_config
from sofa_mover.training.stack import (
    TrainingStack,
    build_training_stack as build_runtime_training_stack,
)
from sofa_mover.training.utils import compute_gae_minibatch, optimize_ppo_epochs


def gpu_mem_mb() -> float:
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def reset_gpu_stats() -> None:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@dataclass
class BenchResult:
    num_envs: int
    fps: float
    peak_mem_mb: float
    success: bool


def build_training_stack(
    num_envs: int,
    cfg: SofaEnvConfig,
    rollout_length: int = 64,
    device: torch.device = DEVICE,
) -> TrainingStack:
    """Build the full training stack (env + networks + collector + loss)."""
    config = TrainingConfig(
        num_envs=num_envs,
        total_frames=num_envs * rollout_length * 100,
        rollout_length=rollout_length,
        device=device,
    )
    env_cfg, resolved_num_envs = resolve_training_config(config, env_cfg=cfg)
    return build_runtime_training_stack(config, env_cfg, resolved_num_envs)


def _warmup_training_step(stack: TrainingStack, minibatch_size: int) -> None:
    """Run one warmup PPO step."""
    warmup_data = next(iter(stack.collector))
    compute_gae_minibatch(warmup_data, stack.loss_module, minibatch_size)
    warmup_flat = warmup_data.reshape(-1)
    optimize_ppo_epochs(
        warmup_flat,
        stack.loss_module,
        stack.optimizer,
        num_epochs=1,
        minibatch_size=minibatch_size,
        max_grad_norm=0.5,
        device=stack.env.device,
    )


def bench_training(
    num_envs: int,
    cfg: SofaEnvConfig,
    n_batches: int = 5,
    rollout_length: int = 64,
) -> BenchResult:
    """Benchmark full training loop at given batch size."""
    cleanup()
    reset_gpu_stats()

    stack: TrainingStack | None = None
    try:
        stack = build_training_stack(num_envs, cfg=cfg, rollout_length=rollout_length)

        frames_per_batch = num_envs * rollout_length

        # Warmup: 1 batch
        _warmup_training_step(stack, minibatch_size=512)

        # Timed runs
        reset_gpu_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        total_frames = 0

        batch_count = 0
        for data in stack.collector:
            compute_gae_minibatch(data, stack.loss_module, minibatch_size=512)
            data_flat = data.reshape(-1)
            optimize_ppo_epochs(
                data_flat,
                stack.loss_module,
                stack.optimizer,
                num_epochs=4,
                minibatch_size=512,
                max_grad_norm=0.5,
                device=stack.env.device,
            )

            total_frames += frames_per_batch
            batch_count += 1
            if batch_count >= n_batches:
                break

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        fps = total_frames / elapsed
        peak = gpu_mem_mb()

        return BenchResult(num_envs=num_envs, fps=fps, peak_mem_mb=peak, success=True)

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            return BenchResult(num_envs=num_envs, fps=0, peak_mem_mb=0, success=False)
        raise
    finally:
        if stack is not None:
            stack.collector.shutdown()
        cleanup()


def find_max_batch_size(cfg: SofaEnvConfig, lo: int = 4, hi: int = 512) -> int:
    """Binary search for max training batch size."""
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        print(f"  Trying B={mid}...", end=" ", flush=True)
        result = bench_training(mid, cfg=cfg, n_batches=2)
        if result.success:
            print(f"OK (peak={result.peak_mem_mb:.0f} MB, fps={result.fps:.0f})")
            best = mid
            lo = mid + 1
        else:
            print("OOM")
            hi = mid - 1
    return best


def bench_all_modes() -> None:
    """Find max B and benchmark FPS at max B for all three obs modes."""
    modes: list[ObsModeName] = ["baseline", "safe", "aggressive"]
    results: dict[str, tuple[int, BenchResult]] = {}

    for mode in modes:
        cfg = make_env_config(mode)
        print(f"\n{'='*60}")
        print(f"  MODE: {mode}")
        print(
            f"  Config: obs_downscale={cfg.obs_downscale}, boundary_rays={cfg.boundary_rays}"
        )
        print(f"{'='*60}")

        print("\n--- Finding max batch size ---")
        max_b = find_max_batch_size(cfg)
        print(f"Max training batch size: {max_b}")

        print(f"\n--- Benchmarking FPS at B={max_b} ---")
        result = bench_training(max_b, cfg=cfg, n_batches=5)
        if result.success:
            print(f"  FPS={result.fps:,.0f}  peak={result.peak_mem_mb:.0f} MB")
        else:
            print("  OOM at max B (unexpected)")
        results[mode] = (max_b, result)

    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Mode':<14} {'Max B':>6} {'FPS@MaxB':>10} {'Peak MB':>10}")
    print("-" * 44)
    for mode in modes:
        max_b, r = results[mode]
        fps_str = f"{r.fps:,.0f}" if r.success else "OOM"
        peak_str = f"{r.peak_mem_mb:,.0f}" if r.success else "—"
        print(f"{mode:<14} {max_b:>6} {fps_str:>10} {peak_str:>10}")


if __name__ == "__main__":
    bench_all_modes()
