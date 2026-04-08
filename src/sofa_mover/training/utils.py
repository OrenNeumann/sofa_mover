"""Standalone training helpers."""

from dataclasses import dataclass

import numpy as np
import torch
from tensordict import TensorDictBase
from torchrl.objectives import ClipPPOLoss

from sofa_mover.env import SofaEnv
from sofa_mover.networks import SofaCriticNet
from sofa_mover.training.normalizer import Normalizer
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
    area_at_goal: float
    goal_rate: float
    truncation_rate: float
    mean_ep_length: float
    mean_total_angle: float
    mean_total_distance: float
    last_done_idx: int


def compute_gae_direct(
    data: TensorDictBase,
    loss_module: ClipPPOLoss,
    critic_net: SofaCriticNet,
    gamma: float,
    gae_lambda: float,
) -> None:
    """Compute GAE advantages by calling the critic directly.

    Avoids TorchRL's vmap-based value estimation.
    Batches current + next state value predictions into a single forward pass.
    """
    B, T = data.shape
    adv_key = loss_module.tensor_keys.advantage
    vt_key = loss_module.tensor_keys.value_target

    with torch.no_grad():
        # Extract raw observation tensors (B, T, ...)
        obs = data["observation"]
        next_obs = data["next", "observation"]

        # Concatenate current and next obs for a single critic call (2*B*T)
        all_sv = torch.cat(
            [obs["sofa_view"].flatten(0, 1), next_obs["sofa_view"].flatten(0, 1)]
        )
        all_p = torch.cat([obs["pose"].flatten(0, 1), next_obs["pose"].flatten(0, 1)])
        all_pr = torch.cat(
            [obs["progress"].flatten(0, 1), next_obs["progress"].flatten(0, 1)]
        )

        all_values = critic_net(all_sv, all_p, all_pr)  # (2*B*T, 1)
        values, next_values = all_values.reshape(2, B, T, 1).unbind(0)

        reward = data["next", "reward"]  # (B, T, 1)
        done = data["next", "done"].float()  # (B, T, 1)
        terminated = data["next", "terminated"].float()  # (B, T, 1)

        # TD-error: delta = r + gamma * (1 - terminated) * V_next - V
        delta = reward + gamma * (1.0 - terminated) * next_values - values

        # Backward GAE scan
        advantages = torch.empty_like(delta)
        last_gae = torch.zeros(B, 1, device=delta.device, dtype=delta.dtype)
        for t in reversed(range(T)):
            not_done = 1.0 - done[:, t]
            last_gae = delta[:, t] + gamma * gae_lambda * not_done * last_gae
            advantages[:, t] = last_gae

    data.set(adv_key, advantages)
    data.set(vt_key, advantages + values)


# TODO: hacky, clean up
def normalize_rewards_inplace(
    data: TensorDictBase, normalizer: Normalizer
) -> torch.Tensor:
    """Normalize rollout rewards in-place and return the normalized tensor."""
    normalized_reward = normalizer.normalize_rewards(
        data["next", "reward"],
        data["next", "done"],
    )
    data["next"].set("reward", normalized_reward)
    return normalized_reward


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
        # Shuffle once and slice contiguously (much faster than random indexing
        # into TensorDicts, which gathers every nested tensor individually).
        perm = torch.randperm(total_samples, device=device)
        shuffled = data_flat[perm]
        for mb_start in range(0, total_samples, minibatch_size):
            mb_end = min(mb_start + minibatch_size, total_samples)
            minibatch = shuffled[mb_start:mb_end]

            loss_td = loss_module(minibatch)
            loss = (
                loss_td["loss_objective"]
                + loss_td["loss_critic"]
                + loss_td["loss_entropy"]
            )

            optimizer.zero_grad(set_to_none=True)
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
    # truncated = done AND NOT terminated
    ep_terminated = data_flat["next", "terminated"].squeeze(-1)[done_mask]
    truncated = ~ep_terminated

    # area_at_goal: average terminal area over ALL done episodes.
    # Truncated/area_dead episodes have terminal_area=0, so this naturally
    # combines success rate with area quality.
    area_at_goal = ep_terminal_area.mean().item()
    goal_rate = (ep_terminal_area > 0).float().mean().item()
    truncation_rate = truncated.float().mean().item()
    mean_ep_length = ep_length.mean().item()
    mean_total_angle = ep_total_angle.mean().item()
    mean_total_distance = ep_total_distance.mean().item()
    last_done_idx = int(done_mask.nonzero(as_tuple=False)[-1].item())

    return EpisodeMetrics(
        n_done=n_done,
        area_at_goal=area_at_goal,
        goal_rate=goal_rate,
        truncation_rate=truncation_rate,
        mean_ep_length=mean_ep_length,
        mean_total_angle=mean_total_angle,
        mean_total_distance=mean_total_distance,
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

    sofa_img = (
        data_flat["next", "observation", "sofa_view"][last_done_idx, 0]
        .float()
        .clamp(0, 1)
        .cpu()
        .numpy()
    )

    # Reconstruct corridor mask from pose for visualization
    pose_td = data_flat["next", "observation", "pose"][last_done_idx].unsqueeze(0)
    corridor_full = env.rasterizer.corridor_mask(pose_td)
    mask_img = (
        env._downscale_obs(env._crop(corridor_full).to(torch.uint8))[0, 0]
        .float()
        .cpu()
        .numpy()
    )
    return build_composite(sofa_img, mask_img)
