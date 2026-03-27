"""Radial boundary extraction from sofa grids via ray-casting."""

import math

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


class BoundaryExtractor:
    """Extracts radial boundary profiles from sofa grids via ray-casting.

    Precomputes a grid of sample points along evenly-spaced rays from the
    initial sofa's COM.  At each step, samples the sofa along these rays
    and returns the normalized distance to the first gap per ray.
    """

    def __init__(
        self, n_rays: int, init_sofa: Float[Tensor, "H W"], device: torch.device
    ) -> None:
        crop_h, crop_w = init_sofa.shape
        angles = torch.linspace(0, 2 * math.pi, n_rays + 1, device=device)[:-1]

        # Ray origin = COM of initial sofa in pixel coords
        nz = init_sofa.nonzero(as_tuple=True)
        center_r, center_c = nz[0].float().mean(), nz[1].float().mean()

        # Sample points along each ray in pixel coords
        max_r = math.sqrt(crop_h**2 + crop_w**2) / 2
        self._n_samples = int(max_r) + 1
        t = torch.linspace(0, max_r, self._n_samples, device=device)  # (M,)

        # (N, M) pixel positions → grid_sample [-1, 1] coords
        rows = center_r - t.unsqueeze(0) * torch.sin(angles).unsqueeze(1)
        cols = center_c + t.unsqueeze(0) * torch.cos(angles).unsqueeze(1)
        grid_x = 2.0 * cols / (crop_w - 1) - 1.0
        grid_y = 2.0 * rows / (crop_h - 1) - 1.0
        self._ray_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

    def __call__(self, sofa: Float[Tensor, "B 1 H W"]) -> Float[Tensor, "B N"]:
        """Extract radial boundary profile, normalized to [0, 1]."""
        grid = self._ray_grid.expand(sofa.shape[0], -1, -1, -1)
        # fmt: off
        samples = F.grid_sample(
            sofa, grid, mode="nearest", padding_mode="zeros", align_corners=True
        )[:, 0]  # (B, N, M)
        # fmt: on
        # Count consecutive 1s from center along each ray
        return samples.cumprod(dim=-1).sum(dim=-1) / self._n_samples
