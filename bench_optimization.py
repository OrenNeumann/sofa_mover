"""Benchmark script for obs mode configurations.

Measures per mode:
  1. Max batch size that fits in GPU memory (binary search)
  2. Training FPS at max batch size
  3. Peak GPU memory usage
"""

import gc
import time
from dataclasses import dataclass

import torch
from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.modules import OneHotCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from sofa_mover.corridor import DEVICE
from sofa_mover.env import SofaEnvConfig, make_sofa_env
from sofa_mover.networks import SofaActorNet, SofaCriticNet
from sofa_mover.obs_mode import ObsModeName, make_encoder, make_env_config


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
) -> tuple[SyncDataCollector, ClipPPOLoss, torch.optim.Optimizer, ProbabilisticActor]:
    """Build the full training stack (env + networks + collector + loss)."""
    env = make_sofa_env(num_envs=num_envs, cfg=cfg, device=device)

    encoder = make_encoder(cfg)
    actor_net = SofaActorNet(encoder=encoder).to(device)
    critic_net = SofaCriticNet(encoder=actor_net.encoder).to(device)

    actor_module = TensorDictModule(
        actor_net,
        in_keys=["observation", "pose", "progress"],
        out_keys=["logits"],
    )
    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        return_log_prob=True,
    )

    critic = ValueOperator(
        module=critic_net,
        in_keys=["observation", "pose", "progress"],
    )

    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=0.2,
        entropy_bonus=True,
        entropy_coeff=0.01,
        critic_coeff=1.0,
    )
    loss_module.make_value_estimator(GAE, gamma=0.99, lmbda=0.95, value_network=critic)

    optimizer = torch.optim.Adam(loss_module.parameters(), lr=3e-4)

    frames_per_batch = num_envs * rollout_length
    collector = SyncDataCollector(
        create_env_fn=env,
        policy=actor,
        frames_per_batch=frames_per_batch,
        total_frames=frames_per_batch * 100,
        device=device,
    )

    return collector, loss_module, optimizer, actor


def bench_training(
    num_envs: int,
    cfg: SofaEnvConfig,
    n_batches: int = 5,
    rollout_length: int = 64,
) -> BenchResult:
    """Benchmark full training loop at given batch size."""
    cleanup()
    reset_gpu_stats()

    collector = None
    try:
        collector, loss_module, optimizer, actor = build_training_stack(
            num_envs, cfg=cfg, rollout_length=rollout_length
        )

        frames_per_batch = num_envs * rollout_length

        # Warmup: 1 batch
        warmup_data = next(iter(collector))
        with torch.no_grad():
            loss_module.value_estimator(
                warmup_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )
        data_flat = warmup_data.reshape(-1)
        mb = data_flat[: min(512, data_flat.shape[0])]
        loss_td = loss_module(mb)
        loss = (
            loss_td["loss_objective"] + loss_td["loss_critic"] + loss_td["loss_entropy"]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del warmup_data, data_flat, mb, loss_td, loss

        # Timed runs
        reset_gpu_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        total_frames = 0

        batch_count = 0
        for data in collector:
            with torch.no_grad():
                loss_module.value_estimator(
                    data,
                    params=loss_module.critic_network_params,
                    target_params=loss_module.target_critic_network_params,
                )

            data_flat = data.reshape(-1)
            total_samples = data_flat.shape[0]

            for _epoch in range(4):
                perm = torch.randperm(total_samples, device=DEVICE)
                for mb_start in range(0, total_samples, 512):
                    mb_end = min(mb_start + 512, total_samples)
                    mb_idx = perm[mb_start:mb_end]
                    mb = data_flat[mb_idx]

                    loss_td = loss_module(mb)
                    loss = (
                        loss_td["loss_objective"]
                        + loss_td["loss_critic"]
                        + loss_td["loss_entropy"]
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 0.5)
                    optimizer.step()

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
        if collector is not None:
            collector.shutdown()
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
