"""BestEpisodeTracker must reconstruct the exact action sequence of the
highest-terminal-area episode, including episodes spanning batch boundaries."""

import pytest
import torch
import torch.nn.functional as F

from sofa_mover.training.best_tracker import BestEpisodeTracker

N_BINS = 3


def _actions(ts: range | list[int]) -> torch.Tensor:
    """Distinct, recognizable one-hot actions for steps ts: (T, 3*N_BINS)."""
    idx = torch.tensor([[t % N_BINS, (t + 1) % N_BINS, (t + 2) % N_BINS] for t in ts])
    return torch.cat(
        [F.one_hot(idx[:, axis], N_BINS).float() for axis in range(3)], dim=-1
    )


def _tracker(num_envs: int = 1) -> BestEpisodeTracker:
    return BestEpisodeTracker(
        num_envs=num_envs, max_steps=10, n_bins=N_BINS, device=torch.device("cpu")
    )


def test_global_best_selected_across_envs_and_kept() -> None:
    tracker = _tracker(num_envs=2)
    actions = torch.stack([_actions([0, 1]), _actions([3, 4])])
    terminal_area = torch.tensor([[0.0, 0.4], [0.0, 0.6]])
    done = torch.tensor([[False, True], [False, True]])
    assert tracker.update(actions, terminal_area, done)
    assert tracker.best_area == pytest.approx(0.6)
    assert tracker.best_actions is not None
    torch.testing.assert_close(tracker.best_actions, _actions([3, 4]))

    # A later, worse episode does not displace the best.
    worse = torch.stack([_actions([5]), _actions([6])])
    worse_area = torch.tensor([[0.5], [0.0]])
    worse_done = torch.tensor([[True], [False]])
    assert not tracker.update(worse, worse_area, worse_done)
    assert tracker.best_area == pytest.approx(0.6)
    torch.testing.assert_close(tracker.best_actions, _actions([3, 4]))


def test_episode_spanning_two_batches_is_stitched() -> None:
    tracker = _tracker()
    first = _actions(range(3)).unsqueeze(0)
    no_done = torch.zeros(1, 3, dtype=torch.bool)
    assert not tracker.update(first, torch.zeros(1, 3), no_done)

    second = _actions([3, 4, 5]).unsqueeze(0)
    terminal_area = torch.tensor([[0.0, 0.9, 0.0]])
    done = torch.tensor([[False, True, False]])
    assert tracker.update(second, terminal_area, done)
    # Episode = all of batch 1 plus the first two steps of batch 2.
    assert tracker.best_actions is not None
    torch.testing.assert_close(tracker.best_actions, _actions(range(5)))


def test_multiple_dones_leave_only_the_tail_in_carryover() -> None:
    tracker = _tracker()
    first = _actions(range(6)).unsqueeze(0)
    # Episodes end at t=1 and t=3; steps 4-5 start a third, in-progress episode.
    terminal_area = torch.tensor([[0.0, 0.2, 0.0, 0.1, 0.0, 0.0]])
    done = torch.tensor([[False, True, False, True, False, False]])
    tracker.update(first, terminal_area, done)

    second = _actions([6]).unsqueeze(0)
    assert tracker.update(second, torch.tensor([[0.9]]), torch.tensor([[True]]))
    # Best episode = tail of batch 1 (steps 4, 5) + step 6.
    assert tracker.best_actions is not None
    torch.testing.assert_close(tracker.best_actions, _actions([4, 5, 6]))
