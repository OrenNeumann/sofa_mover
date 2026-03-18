"""PPO training script for the sofa moving problem."""

import time
from pathlib import Path

import torch
from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.modules import OneHotCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from sofa_mover.corridor import DEVICE
from sofa_mover.env import SofaEnvConfig, make_sofa_env
from sofa_mover.networks import SofaActorNet, SofaCriticNet


def train(
    num_envs: int = 32,
    total_frames: int = 2_000_000,
    frames_per_batch: int = 2048,  # 32 envs * 64 steps
    num_epochs: int = 4,
    minibatch_size: int = 512,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    entropy_coeff: float = 0.01,
    critic_coeff: float = 1.0,
    max_grad_norm: float = 0.5,
    device: torch.device = DEVICE,
    output_dir: str = "output",
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # --- Environment (128×128 grid to fit in GPU memory) ---
    from sofa_mover.corridor import GridConfig

    cfg = SofaEnvConfig(
        sofa_config=GridConfig(grid_size=128, world_size=3.0),
        template_config=GridConfig(grid_size=128, world_size=6.0),
    )
    env = make_sofa_env(num_envs=num_envs, cfg=cfg, device=device)

    # --- Networks ---
    actor_net = SofaActorNet().to(device)
    # Critic shares the actor's encoder
    critic_net = SofaCriticNet(encoder=actor_net.encoder).to(device)

    # Wrap actor for TorchRL
    actor_module = TensorDictModule(
        actor_net,
        in_keys=["observation", "progress"],
        out_keys=["logits"],
    )
    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        return_log_prob=True,
    )

    # Wrap critic for TorchRL
    critic = ValueOperator(
        module=critic_net,
        in_keys=["observation", "progress"],
    )

    # --- PPO Loss ---
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=clip_epsilon,
        entropy_bonus=True,
        entropy_coeff=entropy_coeff,
        critic_coeff=critic_coeff,
    )
    loss_module.make_value_estimator(
        GAE,
        gamma=gamma,
        lmbda=gae_lambda,
        value_network=critic,
    )

    # --- Optimizer ---
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=lr)

    # --- Collector ---
    collector = SyncDataCollector(
        create_env_fn=env,
        policy=actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
    )

    # --- Training loop ---
    best_mean_area = 0.0
    batch_idx = 0

    for data in collector:
        t0 = time.perf_counter()

        # Compute GAE advantages
        with torch.no_grad():
            loss_module.value_estimator(
                data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )

        # Flatten time dimension for minibatch iteration
        data_flat = data.reshape(-1)
        total_samples = data_flat.shape[0]

        # PPO epochs
        for _epoch in range(num_epochs):
            perm = torch.randperm(total_samples, device=device)
            for mb_start in range(0, total_samples, minibatch_size):
                mb_end = min(mb_start + minibatch_size, total_samples)
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
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optimizer.step()

        # --- Logging ---
        elapsed = time.perf_counter() - t0
        fps = data.numel() / elapsed

        # Episode stats from done signals
        done_mask = data["next", "done"].flatten()
        reward_sum = data["next", "reward"].flatten()

        # Terminal areas: reward on steps where goal was reached
        terminated_mask = data["next", "terminated"].flatten()
        if terminated_mask.any():
            terminal_rewards = reward_sum[terminated_mask]
            mean_terminal = terminal_rewards.mean().item()
        else:
            mean_terminal = 0.0

        n_done = done_mask.sum().item()
        goal_reached = data["next", "terminated"].flatten()
        # Goal reached = terminated AND not area_dead (positive terminal reward)
        n_goal = (goal_reached & (reward_sum > 0.5)).sum().item()

        print(
            f"Batch {batch_idx:3d} | "
            f"fps {fps:,.0f} | "
            f"mean_reward {reward_sum.mean().item():+.4f} | "
            f"episodes_done {n_done:.0f} | "
            f"goals {n_goal:.0f} | "
            f"mean_terminal {mean_terminal:+.4f}"
        )

        # Save best
        if mean_terminal > best_mean_area:
            best_mean_area = mean_terminal
            torch.save(
                {
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "encoder": actor_net.encoder.state_dict(),
                    "cfg": env.cfg,
                    "batch_idx": batch_idx,
                    "best_mean_area": best_mean_area,
                },
                output_path / "best_policy.pt",
            )

        batch_idx += 1

    collector.shutdown()

    # Save final
    final_path = output_path / "final_policy.pt"
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "encoder": actor_net.encoder.state_dict(),
            "cfg": env.cfg,
            "batch_idx": batch_idx,
            "best_mean_area": best_mean_area,
        },
        final_path,
    )
    print(f"Training complete. Best mean terminal area: {best_mean_area:.4f}")
    print(f"Saved to {final_path}")
    return final_path


if __name__ == "__main__":
    train()
