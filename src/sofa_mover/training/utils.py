"""Standalone training helpers."""

from dataclasses import dataclass

import numpy as np
import torch
from tensordict import TensorDictBase
from torchrl.objectives import ClipPPOLoss

from sofa_mover.env import SofaEnv
from sofa_mover.visualization.render import build_composite


# TODO: some of these are more core logic than helpers, maybe reorganize
@dataclass(frozen=True)
class OptimizationStats:
    """Scalar summaries from PPO optimization."""

    loss_policy: float
    loss_critic: float
    loss_entropy: float
    grad_norm: float


@dataclass(frozen=True)
class EpisodeMetrics:
    """Per-batch episode metrics (only present when at least one episode is done)."""

    n_done: int
    mean_terminal_area: float
    truncation_rate: float
    mean_ep_length: float
    mean_total_angle: float
    mean_total_distance: float
    mean_area: float
    last_done_idx: int


def compute_gae_minibatch(
    data: TensorDictBase,
    loss_module: ClipPPOLoss,
    minibatch_size: int,
) -> None:
    """Compute GAE advantages in env-dimension chunks to avoid OOM.

    Processing the full (BxT) buffer at once converts all uint8 obs to
    float32 simultaneously (~5 GiB for B=256, T=64). Chunking along B
    keeps peak memory proportional to minibatch_size.
    """
    batch_size, rollout_length = data.shape
    env_chunk = max(1, minibatch_size // rollout_length)
    adv_key = loss_module.tensor_keys.advantage
    vt_key = loss_module.tensor_keys.value_target
    advantages: list[torch.Tensor] = []
    value_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, batch_size, env_chunk):
            chunk = data[start : start + env_chunk]
            loss_module.value_estimator(
                chunk,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )
            advantages.append(chunk[adv_key])
            value_targets.append(chunk[vt_key])
    data.set(adv_key, torch.cat(advantages, dim=0))
    data.set(vt_key, torch.cat(value_targets, dim=0))


def optimize_ppo_epochs(
    data_flat: TensorDictBase,
    loss_module: ClipPPOLoss,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    minibatch_size: int,
    max_grad_norm: float,
    device: torch.device,
) -> OptimizationStats:
    """Run PPO minibatch optimization."""
    total_samples = data_flat.shape[0]

    # PPO epochs
    grad_norm = 0.0
    last_loss_td: TensorDictBase | None = None
    for _epoch in range(num_epochs):
        perm = torch.randperm(total_samples, device=device)
        for mb_start in range(0, total_samples, minibatch_size):
            mb_end = min(mb_start + minibatch_size, total_samples)
            mb_idx = perm[mb_start:mb_end]
            minibatch = data_flat[mb_idx]

            loss_td = loss_module(minibatch)
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
            last_loss_td = loss_td

    if last_loss_td is None:
        raise RuntimeError("PPO optimization ran with zero minibatches.")

    return OptimizationStats(
        loss_policy=last_loss_td["loss_objective"].item(),
        loss_critic=last_loss_td["loss_critic"].item(),
        loss_entropy=last_loss_td["loss_entropy"].item(),
        grad_norm=grad_norm,
    )


def extract_episode_metrics(data_flat: TensorDictBase) -> EpisodeMetrics | None:
    """Aggregate episode metrics from done episodes in the current batch.

    Returns None when no episodes completed in this batch.
    """
    # Per-episode metrics (from done episodes in this batch)
    done_mask = data_flat["next", "done"].squeeze(-1)

    if not done_mask.any():
        return None

    n_done = int(done_mask.sum().item())

    ep_terminal_area = data_flat["next", "terminal_area"].squeeze(-1)[done_mask]
    ep_length = data_flat["next", "episode_length"].squeeze(-1)[done_mask].float()
    ep_total_angle = data_flat["next", "episode_total_angle"].squeeze(-1)[done_mask]
    ep_total_distance = data_flat["next", "episode_total_distance"].squeeze(-1)[
        done_mask
    ]
    ep_area_integral = data_flat["next", "episode_area_integral"].squeeze(-1)[done_mask]
    # truncated = done AND NOT terminated
    ep_terminated = data_flat["next", "terminated"].squeeze(-1)[done_mask]
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
    last_done_idx = int(done_mask.nonzero(as_tuple=False)[-1].item())

    return EpisodeMetrics(
        n_done=n_done,
        mean_terminal_area=mean_terminal_area,
        truncation_rate=truncation_rate,
        mean_ep_length=mean_ep_length,
        mean_total_angle=mean_total_angle,
        mean_total_distance=mean_total_distance,
        mean_area=mean_area,
        last_done_idx=last_done_idx,
    )


def maybe_build_episode_composite(
    data_flat: TensorDictBase,
    env: SofaEnv,
    batch_idx: int,
    image_log_interval: int,
    last_done_idx: int,
) -> np.ndarray | None:
    """Build a composite sofa image when image logging is enabled."""
    # Periodic sofa image (only for grid modes — boundary obs is 1D)
    if batch_idx % image_log_interval != 0 or env.cfg.observation_type == "boundary":
        return None

    sofa_img = data_flat["next", "observation"][last_done_idx, 0].float().cpu().numpy()
    # Reconstruct corridor mask from pose for visualization
    pose_td = data_flat["next", "pose"][last_done_idx].unsqueeze(0)
    corridor_full = env.rasterizer.corridor_mask(pose_td)
    mask_img = (
        env._downscale_obs(env._crop(corridor_full).to(torch.uint8))[0, 0]
        .float()
        .cpu()
        .numpy()
    )
    return build_composite(sofa_img, mask_img)
