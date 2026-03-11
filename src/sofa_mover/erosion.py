from jaxtyping import Float
from torch import Tensor


def erode(
    sofa: Float[Tensor, "B 1 H W"],
    corridor_mask: Float[Tensor, "B 1 H W"],
) -> Float[Tensor, "B 1 H W"]:
    """Erode the sofa by intersecting with the corridor mask.

    Any sofa pixel outside the corridor (mask=0) is permanently removed.
    This is the core operation: sofa = sofa ∩ corridor_pose(t).
    """
    return sofa * corridor_mask
