"""Standalone training helpers."""

import itertools
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDictBase
from torchrl.modules import OneHotCategorical

from sofa_mover.env import SofaEnv
from sofa_mover.networks import SofaActorNet, SofaCriticNet
from sofa_mover.visualization.render import build_composite

# TensorDict keys set by TorchRL's ProbabilisticActor / used by GAE computation
_ADV_KEY = "advantage"
_VT_KEY = "value_target"
_SAMPLE_LOG_PROB_KEY = "action_log_prob"


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
    best_area_at_goal: float
    goal_rate: float
    truncation_rate: float
    mean_ep_length: float
    mean_total_angle: float
    mean_total_distance: float
    last_done_idx: int


def compute_gae_direct(
    data: TensorDictBase,
    critic_net: SofaCriticNet,
    gamma: float,
    gae_lambda: float,
) -> None:
    """Compute GAE advantages by calling the critic directly.

    Avoids TorchRL's vmap-based value estimation.
    Batches current + next state value predictions into a single forward pass.
    """
    B, T = data.shape

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

    data.set(_ADV_KEY, advantages)
    data.set(_VT_KEY, advantages + values)


def optimize_ppo_epochs(
    data_flat: TensorDictBase,
    actor_net: SofaActorNet,
    critic_net: SofaCriticNet,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    minibatch_size: int,
    max_grad_norm: float,
    device: torch.device,
    clip_epsilon: float,
    entropy_coeff: float,
    critic_coeff: float,
) -> OptimizationStats:
    """Run PPO minibatch optimization with direct tensor ops.

    Calls the shared encoder once per minibatch (not twice as separate
    actor/critic forward passes would).
    """
    N = data_flat.shape[0]

    # Extract all relevant tensors from the TensorDict once (outside the epoch
    # loop) to avoid repeated dict lookups per minibatch.
    sv = data_flat["observation", "sofa_view"]
    pose = data_flat["observation", "pose"]
    prog = data_flat["observation", "progress"]
    action = data_flat["action"]
    old_log_prob = data_flat[_SAMPLE_LOG_PROB_KEY]
    advantage = data_flat[_ADV_KEY]
    value_target = data_flat[_VT_KEY]

    encoder = actor_net.encoder
    actor_head = actor_net.head
    critic_head = critic_net.head

    last_stats: OptimizationStats | None = None

    for _epoch in range(num_epochs):
        # Shuffle all tensors together with a single permutation.
        perm = torch.randperm(N, device=device)
        sv_s = sv[perm]
        pose_s = pose[perm]
        prog_s = prog[perm]
        act_s = action[perm]
        olp_s = old_log_prob[perm]
        adv_s = advantage[perm]
        vt_s = value_target[perm]

        for mb_start in range(0, N, minibatch_size):
            mb_end = min(mb_start + minibatch_size, N)

            # Encoder called ONCE — shared features for actor and critic heads.
            features = encoder(
                sv_s[mb_start:mb_end],
                pose_s[mb_start:mb_end],
                prog_s[mb_start:mb_end],
            )

            logits = actor_head(features)
            dist = OneHotCategorical(logits=logits)
            new_log_prob = dist.log_prob(act_s[mb_start:mb_end])
            entropy = dist.entropy()

            values = critic_head(features).squeeze(-1)

            ratio = torch.exp(new_log_prob - olp_s[mb_start:mb_end].squeeze(-1))
            adv_mb = adv_s[mb_start:mb_end].squeeze(-1)
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv_mb
            loss_pol = -torch.min(surr1, surr2).mean()
            loss_crit = critic_coeff * F.mse_loss(
                values, vt_s[mb_start:mb_end].squeeze(-1)
            )
            loss_ent = -entropy_coeff * entropy.mean()
            loss = loss_pol + loss_crit + loss_ent

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                itertools.chain(actor_net.parameters(), critic_net.head.parameters()),
                max_grad_norm,
            ).item()
            optimizer.step()
            last_stats = OptimizationStats(
                loss_policy=loss_pol.item(),
                loss_critic=loss_crit.item(),
                loss_entropy=loss_ent.item(),
                grad_norm=grad_norm,
            )

    if last_stats is None:
        raise RuntimeError("PPO optimization ran with zero minibatches.")

    return last_stats


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
    best_area_at_goal = ep_terminal_area.max().item()
    goal_rate = (ep_terminal_area > 0).float().mean().item()
    truncation_rate = truncated.float().mean().item()
    mean_ep_length = ep_length.mean().item()
    mean_total_angle = ep_total_angle.mean().item()
    mean_total_distance = ep_total_distance.mean().item()
    last_done_idx = int(done_mask.nonzero(as_tuple=False)[-1].item())

    return EpisodeMetrics(
        n_done=n_done,
        area_at_goal=area_at_goal,
        best_area_at_goal=best_area_at_goal,
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
    corridor_mask = env.rasterizer.corridor_mask(pose_td)  # already cropped bool
    mask_img = (
        env._downscale_obs(corridor_mask.to(torch.uint8))[0, 0].float().cpu().numpy()
    )
    return build_composite(sofa_img, mask_img)
