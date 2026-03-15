import math

import torch

from sofa_mover.rasterize import Rasterizer


def test_one_substep_matches_single_mask(rasterizer: Rasterizer) -> None:
    """With num_substeps=1, swept mask == corridor_mask at next pose."""
    d = rasterizer.template.device
    pose_prev = torch.tensor([[0.0, 0.0, 0.0]], device=d)
    pose_next = torch.tensor([[0.3, -0.2, math.pi / 6]], device=d)

    swept = rasterizer.swept_mask(pose_prev, pose_next, num_substeps=1)
    single = rasterizer.corridor_mask(pose_next)
    assert torch.equal(swept, single)


def test_swept_mask_subset_of_endpoint(rasterizer: Rasterizer) -> None:
    """Swept mask area <= endpoint-only mask area (it's a subset)."""
    d = rasterizer.template.device
    pose_prev = torch.tensor([[0.0, 0.0, 0.0]], device=d)
    pose_next = torch.tensor([[0.2, -0.3, math.pi / 4]], device=d)

    swept = rasterizer.swept_mask(pose_prev, pose_next, num_substeps=4)
    endpoint = rasterizer.corridor_mask(pose_next)
    assert swept.sum().item() <= endpoint.sum().item()


def test_more_erosion_than_endpoint_only(rasterizer: Rasterizer) -> None:
    """With nontrivial rotation, swept mask has strictly less passable area than endpoint."""
    d = rasterizer.template.device
    pose_prev = torch.tensor([[0.0, 0.0, 0.0]], device=d)
    pose_next = torch.tensor([[0.0, 0.0, math.pi / 4]], device=d)

    swept = rasterizer.swept_mask(pose_prev, pose_next, num_substeps=8)
    endpoint = rasterizer.corridor_mask(pose_next)
    assert swept.sum().item() < endpoint.sum().item()


def test_more_substeps_more_erosion(rasterizer: Rasterizer) -> None:
    """Increasing num_substeps monotonically decreases (or maintains) swept mask area."""
    d = rasterizer.template.device
    pose_prev = torch.tensor([[0.0, 0.0, 0.0]], device=d)
    pose_next = torch.tensor([[0.2, -0.3, math.pi / 4]], device=d)

    areas = []
    for k in [1, 2, 4, 8, 16]:
        swept = rasterizer.swept_mask(pose_prev, pose_next, num_substeps=k)
        areas.append(swept.sum().item())

    for i in range(1, len(areas)):
        assert areas[i] <= areas[i - 1]


def test_no_motion_is_idempotent(rasterizer: Rasterizer) -> None:
    """If prev == next, swept mask equals single mask regardless of num_substeps."""
    d = rasterizer.template.device
    pose = torch.tensor([[0.2, -0.1, 0.3]], device=d)
    single = rasterizer.corridor_mask(pose)

    for k in [1, 4, 8]:
        swept = rasterizer.swept_mask(pose, pose, num_substeps=k)
        assert torch.equal(swept, single)


def test_batched_matches_individual(rasterizer: Rasterizer) -> None:
    """Batched swept masks match individually computed ones."""
    d = rasterizer.template.device
    pose_prev = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.1],
            [-0.1, 0.1, -0.1],
            [0.0, -0.2, 0.0],
        ],
        device=d,
    )
    pose_next = torch.tensor(
        [
            [0.2, -0.3, 0.4],
            [0.3, -0.1, 0.0],
            [0.0, -0.2, 0.3],
            [-0.3, 0.0, -0.2],
        ],
        device=d,
    )

    batched = rasterizer.swept_mask(pose_prev, pose_next, num_substeps=4)

    for i in range(4):
        individual = rasterizer.swept_mask(
            pose_prev[i : i + 1], pose_next[i : i + 1], num_substeps=4
        )
        assert torch.equal(batched[i : i + 1], individual)


def test_output_is_binary(rasterizer: Rasterizer) -> None:
    """Swept mask output contains only 0.0 and 1.0."""
    d = rasterizer.template.device
    pose_prev = torch.tensor([[0.0, 0.0, 0.0]], device=d)
    pose_next = torch.tensor([[0.2, -0.3, math.pi / 6]], device=d)

    swept = rasterizer.swept_mask(pose_prev, pose_next, num_substeps=4)
    unique_vals = torch.unique(swept)
    for v in unique_vals:
        assert v.item() in (0.0, 1.0)


def test_fused_matches_loop(rasterizer: Rasterizer) -> None:
    """Fused swept_mask and loop-based _swept_mask_loop produce identical results."""
    d = rasterizer.template.device
    pose_prev = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, -0.1, 0.2],
        ],
        device=d,
    )
    pose_next = torch.tensor(
        [
            [0.3, -0.2, math.pi / 6],
            [-0.2, 0.1, -0.3],
        ],
        device=d,
    )

    for k in [1, 2, 4, 8]:
        fused = rasterizer.swept_mask(pose_prev, pose_next, num_substeps=k)
        loop = rasterizer._swept_mask_loop(pose_prev, pose_next, num_substeps=k)
        assert torch.equal(fused, loop), f"Mismatch at num_substeps={k}"
