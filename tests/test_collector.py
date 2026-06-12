"""Tests for the RolloutCollector's batch layout contract."""

import torch

import torch.nn.functional as F

from sofa_mover.env import make_sofa_env
from sofa_mover.networks import BoundaryEncoder, MultiDiscreteCategorical, SofaActorNet
from sofa_mover.training.collector import RolloutCollector, _sample_actions
from sofa_mover.training.config import SofaEnvConfig

NUM_ENVS = 4
ROLLOUT = 8
N_RAYS = 8


def _collect_one_batch():
    torch.manual_seed(0)
    cfg = SofaEnvConfig(
        observation_type="boundary",
        boundary_rays=N_RAYS,
        max_steps=3,  # force several resets inside one rollout
    )
    env = make_sofa_env(
        total_frames=NUM_ENVS * ROLLOUT,
        num_envs=NUM_ENVS,
        cfg=cfg,
        device=torch.device("cpu"),
    )
    encoder = BoundaryEncoder(n_rays=2 * N_RAYS, feature_dim=16, width=16, depth=1)
    actor_net = SofaActorNet(nvec=cfg.nvec, encoder=encoder, width=16, depth=1)
    collector = RolloutCollector(
        env=env,
        actor_net=actor_net,
        rollout_length=ROLLOUT,
        total_frames=NUM_ENVS * ROLLOUT,
    )
    return actor_net, next(iter(collector))


def test_auto_reset_layout() -> None:
    """At a done step, `next.observation` keeps the terminal state while the
    next root observation is the freshly reset one; at non-done steps the two
    are the same state."""
    _, data = _collect_one_batch()
    assert data.shape == (NUM_ENVS, ROLLOUT)

    done = data["next", "done"].squeeze(-1)
    assert done[:, :-1].any(), "expected at least one mid-batch reset"

    obs, next_obs = data["observation"], data["next", "observation"]
    for b, t in done[:, :-1].nonzero(as_tuple=False).tolist():
        # terminal state is preserved under `next`
        assert next_obs["pose"][b, t].abs().sum() > 0
        # the following root obs starts a new episode
        assert torch.all(obs["pose"][b, t + 1] == 0.0)
        assert torch.all(obs["progress"][b, t + 1] == 0.0)
    for b, t in (~done[:, :-1]).nonzero(as_tuple=False).tolist():
        for key in ("sofa_view", "pose", "progress"):
            torch.testing.assert_close(obs[key][b, t + 1], next_obs[key][b, t])


def test_gumbel_sampler_matches_categorical_distribution() -> None:
    """The Gumbel-max sampler draws each axis's bin with the probabilities
    given by the policy's softmax (same distribution as Categorical)."""
    torch.manual_seed(0)
    n_bins, n_samples = 5, 200_000
    encoder = BoundaryEncoder(n_rays=2 * N_RAYS, feature_dim=16, width=16, depth=1)
    actor_net = SofaActorNet(nvec=[n_bins] * 3, encoder=encoder, width=16, depth=1)

    # One observation repeated, so every row samples from the same policy.
    obs = [
        torch.rand(1, 2 * N_RAYS).expand(n_samples, -1),
        torch.randn(1, 3).expand(n_samples, -1),
        torch.rand(1, 1).expand(n_samples, -1),
    ]
    with torch.no_grad():
        actions, _ = _sample_actions(actor_net, *obs)
        logits = actor_net(obs[0][:1], obs[1][:1], obs[2][:1]).view(3, n_bins)
        probs = F.softmax(logits, dim=-1).flatten()
    empirical = actions.mean(dim=0)
    torch.testing.assert_close(empirical, probs, atol=0.005, rtol=0.0)


def test_stored_log_prob_matches_policy() -> None:
    """`action_log_prob` must equal the policy's log-prob of the stored action
    at the stored observation (PPO ratios start at 1)."""
    actor_net, data = _collect_one_batch()
    flat = data.reshape(-1)
    obs = flat["observation"]
    with torch.no_grad():
        logits = actor_net(obs["sofa_view"], obs["pose"], obs["progress"])
    dist = MultiDiscreteCategorical(logits=logits, nvec=actor_net.nvec)
    torch.testing.assert_close(dist.log_prob(flat["action"]), flat["action_log_prob"])
