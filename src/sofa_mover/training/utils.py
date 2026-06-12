"""Standalone training helpers."""

import itertools
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDictBase

from sofa_mover.env import SofaEnv
from sofa_mover.networks import SofaActorNet, SofaCriticNet
from sofa_mover.visualization.render import build_composite

# TensorDict keys set by the collector / used by GAE computation
_ADV_KEY = "advantage"
_VT_KEY = "value_target"
_SAMPLE_LOG_PROB_KEY = "action_log_prob"

# Rows per critic call in compute_gae_direct.
_GAE_CHUNK = 8192


@dataclass(frozen=True)
class OptimizationStats:
    """Scalar summaries from PPO optimization."""

    loss_policy: float
    loss_critic: float
    loss_entropy: float
    grad_norm: float
    approx_kl: float
    clip_fraction: float
    epochs_completed: int


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

        all_values = torch.cat(
            [
                critic_net(view, pose, progress)
                for view, pose, progress in zip(
                    all_sv.split(_GAE_CHUNK),
                    all_p.split(_GAE_CHUNK),
                    all_pr.split(_GAE_CHUNK),
                )
            ]
        )  # (2*B*T, 1)
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


def _minibatch_losses(
    actor_net: SofaActorNet,
    critic_head: torch.nn.Module,
    sofa_view: torch.Tensor,
    pose: torch.Tensor,
    progress: torch.Tensor,
    action: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantage: torch.Tensor,
    value_target: torch.Tensor,
    clip_epsilon: float,
    entropy_coeff: float,
    critic_coeff: float,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """One PPO minibatch forward: all losses plus KL/clip diagnostics.

    Runs under torch.compile, so the log_prob/entropy math is written out
    here instead of calling MultiDiscreteCategorical. The two must stay
    equivalent (enforced by a test).

    Returns ``(loss, loss_policy, loss_critic, loss_entropy, approx_kl,
    clip_fraction)``.
    """
    n_axes = len(actor_net.nvec)
    n_bins = actor_net.nvec[0]
    features = actor_net.encoder(sofa_view, pose, progress)
    logits = actor_net.head(features).view(-1, n_axes, n_bins)
    log_probs = F.log_softmax(logits, dim=-1)
    # action is concatenated one-hot, so a dot product extracts the per-axis
    # chosen log-prob (equivalent to argmax + gather).
    action_one_hot = action.view(-1, n_axes, n_bins)
    new_log_prob = (log_probs * action_one_hot).sum(dim=(-2, -1))
    entropy = -(log_probs.exp() * log_probs).sum(dim=(-2, -1))

    values = critic_head(features).squeeze(-1)

    log_ratio = new_log_prob - old_log_prob.squeeze(-1)
    ratio = torch.exp(log_ratio)
    # Schulman's k3 approx-KL estimator: E[(r-1) - log r], unbiased + non-negative.
    with torch.no_grad():
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = (ratio - 1.0).abs().gt(clip_epsilon).float().mean()
    adv = advantage.squeeze(-1)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv
    loss_policy = -torch.min(surr1, surr2).mean()
    loss_critic = critic_coeff * F.mse_loss(values, value_target.squeeze(-1))
    loss_entropy = -entropy_coeff * entropy.mean()
    loss = loss_policy + loss_critic + loss_entropy
    return loss, loss_policy, loss_critic, loss_entropy, approx_kl, clip_fraction


# Default mode only: "reduce-overhead" (CUDA graphs) allocates private memory
# pools that OOM the 6 GB card under host VRAM contention.
_compiled_minibatch_losses = torch.compile(_minibatch_losses)


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
    target_kl: float | None = None,
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
    # Per-batch advantage normalization — standard PPO recipe. Reduces gradient
    # magnitude on outlier batches where a few transitions have huge advantages
    # (e.g. first goal-reaches with the cubic terminal reward).
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    value_target = data_flat[_VT_KEY]

    # Accumulate scalar stats on-GPU across all minibatches and materialize once
    loss_pol_sum = torch.zeros((), device=device)
    loss_crit_sum = torch.zeros((), device=device)
    loss_ent_sum = torch.zeros((), device=device)
    grad_norm_sum = torch.zeros((), device=device)
    approx_kl_sum = torch.zeros((), device=device)
    clip_frac_sum = torch.zeros((), device=device)
    num_minibatches = (N + minibatch_size - 1) // minibatch_size
    mbs_done = 0
    epochs_done = 0
    # Per-epoch KL accumulator on GPU; used for early stopping check (one .item() per epoch).
    epoch_kl_sum = torch.zeros((), device=device)

    break_outer = False
    for _epoch in range(num_epochs):
        epoch_kl_sum.zero_()
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

            (
                loss,
                loss_pol,
                loss_crit,
                loss_ent,
                approx_kl_mb,
                clip_frac_mb,
            ) = _compiled_minibatch_losses(
                actor_net,
                critic_net.head,
                sv_s[mb_start:mb_end],
                pose_s[mb_start:mb_end],
                prog_s[mb_start:mb_end],
                act_s[mb_start:mb_end],
                olp_s[mb_start:mb_end],
                adv_s[mb_start:mb_end],
                vt_s[mb_start:mb_end],
                clip_epsilon,
                entropy_coeff,
                critic_coeff,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                itertools.chain(actor_net.parameters(), critic_net.head.parameters()),
                max_grad_norm,
            )
            optimizer.step()

            loss_pol_sum += loss_pol.detach()
            loss_crit_sum += loss_crit.detach()
            loss_ent_sum += loss_ent.detach()
            grad_norm_sum += grad_norm  # no detach, already no grad_fn
            approx_kl_sum += approx_kl_mb
            clip_frac_sum += clip_frac_mb
            epoch_kl_sum += approx_kl_mb
            mbs_done += 1

            # Per-minibatch hard KL early stop: a single batch with extreme
            # advantage spikes can blow up approx_kl > 1 within one epoch
            # (cubic terminal reward in this env). Catch that immediately.
            # Threshold is 4× target_kl: tolerates normal drift, kills outliers.
            # The .item() sync is free here — the loop is dominated by the
            # backward/optimizer kernels (benchmarked vs an async pinned-memory
            # check, which was strictly slower).
            if target_kl is not None and approx_kl_mb.item() > 4.0 * target_kl:
                break_outer = True
                break

        epochs_done += 1
        if break_outer:
            break
        # KL early stopping: break out of the epoch loop if mean KL this epoch
        # exceeds target. One device->host sync per epoch (cheap relative to the
        # backward passes inside the loop).
        if target_kl is not None:
            epoch_kl = (epoch_kl_sum / num_minibatches).item()
            if epoch_kl > target_kl:
                break

    return OptimizationStats(
        loss_policy=(loss_pol_sum / mbs_done).item(),
        loss_critic=(loss_crit_sum / mbs_done).item(),
        loss_entropy=(loss_ent_sum / mbs_done).item(),
        grad_norm=(grad_norm_sum / mbs_done).item(),
        approx_kl=(approx_kl_sum / mbs_done).item(),
        clip_fraction=(clip_frac_sum / mbs_done).item(),
        epochs_completed=epochs_done,
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
