"""PPO training script for the sofa moving problem."""

import time
from pathlib import Path

import torch
from tqdm import tqdm
from tensordict.nn import TensorDictModule
from torch.utils.tensorboard import SummaryWriter
from torchrl.collectors import SyncDataCollector
from torchrl.modules import OneHotCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from sofa_mover.corridor import DEVICE, GridConfig
from sofa_mover.env import SofaEnvConfig, make_sofa_env
from sofa_mover.networks import SofaActorNet, SofaCriticNet
from sofa_mover.evaluate import evaluate
from sofa_mover.visualization.render import build_composite


def train(
    num_envs: int = 8,
    total_frames: int = 2_000_000,
    rollout_length: int = 64,
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
    log_dir: str = "runs/sofa_ppo",
    image_log_interval: int = 50,
) -> Path:
    frames_per_batch = num_envs * rollout_length

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # --- Environment (256×256 grid, 8 envs to fit in 6GB GPU) ---
    cfg = SofaEnvConfig(
        sofa_config=GridConfig(grid_size=256, world_size=3.0),
        template_config=GridConfig(grid_size=256, world_size=6.0),
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
    total_batches = total_frames // frames_per_batch
    pbar = tqdm(collector, total=total_batches, desc="Training", unit="batch")

    for data in pbar:
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
        grad_norm = 0.0
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
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), max_grad_norm
                ).item()
                optimizer.step()

        # --- Logging ---
        elapsed = time.perf_counter() - t0
        fps = frames_per_batch / elapsed

        # Per-batch TensorBoard scalars
        mean_reward = data["next", "reward"].flatten().mean().item()
        writer.add_scalar("train/fps", fps, batch_idx)
        writer.add_scalar("train/mean_reward", mean_reward, batch_idx)
        writer.add_scalar("loss/policy", loss_td["loss_objective"].item(), batch_idx)
        writer.add_scalar("loss/critic", loss_td["loss_critic"].item(), batch_idx)
        writer.add_scalar("loss/entropy", loss_td["loss_entropy"].item(), batch_idx)
        writer.add_scalar("train/grad_norm", grad_norm, batch_idx)

        # Per-episode metrics (from done episodes in this batch)
        data_flat_log = data.reshape(-1)
        done_mask = data_flat_log["next", "done"].squeeze(-1)
        n_done = done_mask.sum().item()

        if done_mask.any():
            ep_terminal_area = data_flat_log["next", "terminal_area"].squeeze(-1)[
                done_mask
            ]
            ep_length = (
                data_flat_log["next", "episode_length"].squeeze(-1)[done_mask].float()
            )
            ep_total_angle = data_flat_log["next", "episode_total_angle"].squeeze(-1)[
                done_mask
            ]
            ep_total_distance = data_flat_log["next", "episode_total_distance"].squeeze(
                -1
            )[done_mask]
            ep_area_integral = data_flat_log["next", "episode_area_integral"].squeeze(
                -1
            )[done_mask]
            # truncated = done AND NOT terminated
            ep_terminated = data_flat_log["next", "terminated"].squeeze(-1)[done_mask]
            truncated = ~ep_terminated

            goal_mask = ep_terminal_area > 0
            mean_terminal_area = (
                ep_terminal_area[goal_mask].mean().item() if goal_mask.any() else 0.0
            )
            truncation_rate = truncated.float().mean().item()
            mean_ep_length = ep_length.mean().item()
            mean_total_angle = ep_total_angle.mean().item()
            mean_total_distance = ep_total_distance.mean().item()
            mean_area = (ep_area_integral / ep_length.clamp(min=1)).mean().item()

            writer.add_scalar("episode/terminal_area", mean_terminal_area, batch_idx)
            writer.add_scalar("episode/truncation_rate", truncation_rate, batch_idx)
            writer.add_scalar("episode/episode_length", mean_ep_length, batch_idx)
            writer.add_scalar("episode/total_angle", mean_total_angle, batch_idx)
            writer.add_scalar("episode/total_distance", mean_total_distance, batch_idx)
            writer.add_scalar("episode/mean_area", mean_area, batch_idx)

            # Periodic sofa image
            if batch_idx % image_log_interval == 0:
                last_done_idx = done_mask.nonzero(as_tuple=False)[-1].item()
                sofa_img = (
                    data_flat_log["next", "observation"][last_done_idx, 0].cpu().numpy()
                )
                mask_img = (
                    data_flat_log["next", "observation"][last_done_idx, 1].cpu().numpy()
                )
                composite = build_composite(sofa_img, mask_img)
                writer.add_image(
                    "episode/sofa_image", composite, batch_idx, dataformats="HWC"
                )
        else:
            mean_terminal_area = 0.0

        pbar.set_postfix(
            fps=f"{fps:,.0f}",
            reward=f"{mean_reward:+.4f}",
            done=f"{n_done:.0f}",
            area=f"{mean_terminal_area:.4f}",
        )

        # Save best
        if mean_terminal_area > best_mean_area:
            best_mean_area = mean_terminal_area
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
    writer.close()

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

    # Visualize the best policy
    evaluate(
        checkpoint_path=str(output_path / "best_policy.pt"),
        output_path=str(output_path / "agent_trajectory.gif"),
        device=device,
    )

    return final_path


if __name__ == "__main__":
    train()
