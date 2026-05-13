"""Capture the highest-terminal-area episode of a training run.

`update(actions, terminal_area, done)` is called for each rollout batch
from the vectorized env. After it returns, `best_actions` holds the action
sequence of the best episode seen so far across all envs and batches, and
`best_area` its terminal area.

Episodes can cross batch boundaries, so the tracker keeps a per-env buffer
of the in-progress action sequence and stitches it onto the current batch
on done.
"""

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor


_N_AXES = 3  # (dx, dy, dtheta)


def _batch_kernel(
    batch_actions: Tensor,
    terminal_area: Tensor,
    done: Tensor,
    carryover: Tensor,
    carryover_len: Tensor,
    env_idx: Tensor,
    arange_t: Tensor,
    neg_one: Tensor,
    n_bins: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """All vectorized per-batch work fused into one graph:
    one-hot→indices, best detection (per env contract `terminal_area > 0`
    implies `done`), and out-of-place scatter into a fresh carryover.

    Returns `(batch_indices, max_area, max_flat, new_carryover, new_carryover_len)`.
    Functional (no mutation) so `mode="reduce-overhead"` can use CUDA graphs.
    """
    B, T, _ = batch_actions.shape
    batch_indices = (
        batch_actions.view(B, T, _N_AXES, n_bins).argmax(dim=-1).to(torch.int8)
    )
    max_area, max_flat = terminal_area.flatten().max(dim=0)

    done_pos = torch.where(done, arange_t, neg_one)
    latest_done_cm, _ = done_pos.cummax(dim=1)
    last_done = latest_done_cm[:, -1]
    latest_done_prior = F.pad(latest_done_cm[:, :-1], (1, 0), value=-1)
    write_pos = torch.where(
        latest_done_prior == -1,
        carryover_len.unsqueeze(1) + arange_t,
        arange_t - latest_done_prior - 1,
    )
    new_carryover = carryover.index_put(
        (env_idx.expand(-1, T), write_pos), batch_indices
    )
    new_carryover_len = torch.where(
        last_done == -1, carryover_len + T, T - last_done - 1
    )
    return batch_indices, max_area, max_flat, new_carryover, new_carryover_len


_compiled_kernel = torch.compile(_batch_kernel)


class BestEpisodeTracker:
    """Tracks the trajectory with the highest terminal area across training.

    Contract: `done[b, t] = True` must coincide with the env's auto-reset of
    env b. Under that contract `carryover_len[b]` is bounded by `max_steps`.
    """

    carryover: Int[Tensor, "B max_steps n_axes"]
    carryover_len: Int[Tensor, "B"]  # noqa: F821 (jaxtyping dim name)
    best_area: float
    best_actions: Tensor | None

    def __init__(
        self,
        num_envs: int,
        max_steps: int,
        n_bins: int,
        device: torch.device,
    ) -> None:
        if n_bins > 128:
            raise ValueError(f"n_bins={n_bins} doesn't fit in int8; bump dtype.")
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.n_bins = n_bins
        self.device = device

        self.carryover = torch.zeros(
            num_envs, max_steps, _N_AXES, dtype=torch.int8, device=device
        )
        self.carryover_len = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._env_idx = torch.arange(num_envs, device=device).unsqueeze(1)  # (B, 1)
        self._arange_t: Tensor | None = None
        self._neg_one: Tensor | None = None

        self.best_area = 0.0
        self.best_actions = None

    def update(
        self,
        batch_actions: Float[Tensor, "B T action_dim"],
        terminal_area: Float[Tensor, "B T"],
        done: Bool[Tensor, "B T"],
    ) -> bool:
        """Process a rollout batch. Returns True iff a new global best was found.

        Best detection + reconstruction reads the carryover BEFORE we update
        it, so no separate pre-batch snapshot is needed.
        """
        T = done.shape[1]
        if self._arange_t is None or self._arange_t.shape[1] != T:
            self._arange_t = torch.arange(T, device=self.device).unsqueeze(0)
            self._neg_one = torch.full((1, T), -1, dtype=torch.long, device=self.device)

        (
            batch_indices,
            max_area,
            max_flat,
            new_carryover,
            new_carryover_len,
        ) = _compiled_kernel(
            batch_actions,
            terminal_area,
            done,
            self.carryover,
            self.carryover_len,
            self._env_idx,
            self._arange_t,
            self._neg_one,  # type: ignore[arg-type]  # set in tandem with _arange_t above
            self.n_bins,
        )

        max_area_val = max_area.item()
        new_best = max_area_val > self.best_area
        if new_best:
            # Reconstruct using the PRE-update carryover (still in self.carryover).
            max_b, max_t = divmod(int(max_flat.item()), T)
            prev_done_in_b = done[max_b, :max_t]
            if bool(prev_done_in_b.any().item()):
                prev_done_t = int(prev_done_in_b.nonzero(as_tuple=True)[0][-1].item())
                ep_indices = batch_indices[max_b, prev_done_t + 1 : max_t + 1]
            else:
                co_len = int(self.carryover_len[max_b].item())
                ep_indices = torch.cat(
                    [self.carryover[max_b, :co_len], batch_indices[max_b, : max_t + 1]],
                    dim=0,
                )
            self.best_actions = self._indices_to_onehot(ep_indices).cpu()
            self.best_area = max_area_val

        self.carryover = new_carryover
        self.carryover_len = new_carryover_len
        return new_best

    def _indices_to_onehot(
        self, ep_indices: Int[Tensor, "T n_axes"]
    ) -> Float[Tensor, "T action_dim"]:
        per_axis = [
            F.one_hot(ep_indices[:, axis].long(), num_classes=self.n_bins).float()
            for axis in range(_N_AXES)
        ]
        return torch.cat(per_axis, dim=-1)
