"""Rendering utilities for sofa erosion trajectory videos."""

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray
from PIL import GifImagePlugin, Image, ImageFilter
from torch import Tensor
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm


warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
matplotlib.use("Agg")

ScalarImage: TypeAlias = NDArray[np.float32]
ColorImage: TypeAlias = NDArray[np.float32]
Extent = tuple[float, float, float, float]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DARK_GRAY = np.array([0.2, 0.2, 0.2], dtype=np.float32)
STEEL_BLUE = np.array([0.27, 0.47, 0.73], dtype=np.float32)
GLOW_ORANGE = np.array([1.0, 0.76, 0.28], dtype=np.float32)

LEFT_PAD_FRACTION = 0.08
TITLE_PAD = 12.0
PANEL_TARGET_SHORT_SIDE: int = 384

# World width spanned by one full repeat of the floor texture.
FLOOR_TILE_WORLD_SIZE_X: float = 4.5
_FLOOR_TEXTURE_SHORT_SIDE: int = 192


def _load_floor_texture() -> NDArray[np.float32]:
    """Load the parquet texture at moderate resolution, preserving aspect."""
    src = Image.open(
        Path(__file__).resolve().parents[3] / "assets" / "parquet_3.png"
    ).convert("RGB")
    if src.width >= src.height:
        tex_h = _FLOOR_TEXTURE_SHORT_SIDE
        tex_w = round(_FLOOR_TEXTURE_SHORT_SIDE * src.width / src.height)
    else:
        tex_w = _FLOOR_TEXTURE_SHORT_SIDE
        tex_h = round(_FLOOR_TEXTURE_SHORT_SIDE * src.height / src.width)
    return (
        np.asarray(src.resize((tex_w, tex_h), Image.LANCZOS), dtype=np.float32) / 255.0
    )


FLOOR_TEXTURE = _load_floor_texture()
FLOOR_TILE_WORLD_SIZE_Y: float = (
    FLOOR_TILE_WORLD_SIZE_X * FLOOR_TEXTURE.shape[0] / FLOOR_TEXTURE.shape[1]
)


# ---------------------------------------------------------------------------
# Frame data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameData:
    """All data needed to render a single animation frame."""

    step: int
    pose: tuple[float, float, float]
    sofa: ScalarImage
    corridor_mask: ScalarImage
    area: float


def compute_frame_data(
    step: int,
    pose: tuple[float, float, float],
    sofa: Float[Tensor, "1 1 H W"],
    corridor_mask: Float[Tensor, "1 1 H W"],
    cell_area: float,
) -> FrameData:
    """Package tensor data into a FrameData for rendering."""
    return FrameData(
        step=step,
        pose=pose,
        sofa=sofa[0, 0].cpu().numpy(),
        corridor_mask=corridor_mask[0, 0].cpu().numpy(),
        area=sofa.sum().item() * cell_area,
    )


# ---------------------------------------------------------------------------
# Coordinate grids and sampling
# ---------------------------------------------------------------------------


def _make_grid(
    extent: Extent, shape: tuple[int, int]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Build a meshgrid of world coordinates for the given extent and shape."""
    height, width = shape
    x_min, x_max, y_min, y_max = extent
    xs = np.linspace(x_min, x_max, width, dtype=np.float32)
    ys = np.linspace(y_min, y_max, height, dtype=np.float32)
    return np.meshgrid(xs, ys, indexing="xy")


def _world_to_corridor_coords(
    world_x: NDArray[np.float32],
    world_y: NDArray[np.float32],
    pose: tuple[float, float, float],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Transform world coordinates to corridor-local coordinates."""
    tx, ty, theta = pose
    cos_t = np.float32(np.cos(theta))
    sin_t = np.float32(np.sin(theta))
    dx = world_x - tx
    dy = world_y - ty
    return cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy


def _is_in_l_corridor(
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    half_width: float,
) -> NDArray[np.bool_]:
    """Test whether corridor-local coordinates fall inside the L-corridor."""
    horizontal = (x >= -half_width) & (np.abs(y) <= half_width)
    vertical = (x >= -half_width) & (x <= half_width) & (y <= half_width)
    return horizontal | vertical


def _sample_image_at_world_coords(
    image: ScalarImage,
    source_extent: Extent,
    world_x: NDArray[np.float32],
    world_y: NDArray[np.float32],
) -> ScalarImage:
    """Sample an image defined on a world-coordinate extent with nearest neighbors."""
    sx_min, sx_max, sy_min, sy_max = source_extent
    src_h, src_w = image.shape
    x_scale = (src_w - 1) / (sx_max - sx_min)
    y_scale = (src_h - 1) / (sy_max - sy_min)

    col_idx = np.rint((world_x - sx_min) * x_scale).astype(np.int64)
    row_idx = np.rint((world_y - sy_min) * y_scale).astype(np.int64)
    valid = (col_idx >= 0) & (col_idx < src_w) & (row_idx >= 0) & (row_idx < src_h)

    sampled = np.zeros(world_x.shape, dtype=np.float32)
    sampled[valid] = image[row_idx[valid], col_idx[valid]]
    return sampled


def sample_image_in_extent(
    image: ScalarImage,
    source_extent: Extent,
    target_extent: Extent,
    output_shape: tuple[int, int],
) -> ScalarImage:
    """Resample an image from one world-coordinate extent onto another."""
    world_x, world_y = _make_grid(target_extent, output_shape)
    return _sample_image_at_world_coords(image, source_extent, world_x, world_y)


def sample_sofa_in_corridor_frame(
    sofa: ScalarImage,
    pose: tuple[float, float, float],
    sofa_extent: Extent,
    corridor_extent: Extent,
    output_shape: tuple[int, int],
) -> ScalarImage:
    """Rasterize a sofa mask into corridor-local coordinates for one pose."""
    cx_grid, cy_grid = _make_grid(corridor_extent, output_shape)
    tx, ty, theta = pose
    cos_t = np.float32(np.cos(theta))
    sin_t = np.float32(np.sin(theta))
    world_x = cos_t * cx_grid - sin_t * cy_grid + tx
    world_y = sin_t * cx_grid + cos_t * cy_grid + ty
    return _sample_image_at_world_coords(sofa, sofa_extent, world_x, world_y)


# ---------------------------------------------------------------------------
# Corridor mask builders
# ---------------------------------------------------------------------------


def build_corridor_mask(
    extent: Extent,
    output_shape: tuple[int, int],
    corridor_width: float,
) -> ScalarImage:
    """Build an L-corridor mask directly in corridor-local coordinates."""
    x_grid, y_grid = _make_grid(extent, output_shape)
    return _is_in_l_corridor(x_grid, y_grid, corridor_width / 2).astype(np.float32)


def build_world_corridor_mask(
    pose: tuple[float, float, float],
    extent: Extent,
    output_shape: tuple[int, int],
    corridor_width: float,
) -> ScalarImage:
    """Build an L-corridor mask in world coordinates for a given pose."""
    world_x, world_y = _make_grid(extent, output_shape)
    cx, cy = _world_to_corridor_coords(world_x, world_y, pose)
    return _is_in_l_corridor(cx, cy, corridor_width / 2).astype(np.float32)


# ---------------------------------------------------------------------------
# Extent helpers
# ---------------------------------------------------------------------------


def expand_extent(extent: Extent, shape: tuple[int, int], pad_pixels: int) -> Extent:
    """Expand a world-coordinate extent to match pixel padding."""
    height, width = shape
    x_min, x_max, y_min, y_max = extent
    x_pad = (x_max - x_min) * pad_pixels / width
    y_pad = (y_max - y_min) * pad_pixels / height
    return (x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)


def _aspect_matched_shape(extent: Extent, target_short_side: int) -> tuple[int, int]:
    """Return (rows, cols) at target_short_side px matching the extent's aspect."""
    x_min, x_max, y_min, y_max = extent
    width_world = x_max - x_min
    height_world = y_max - y_min
    if width_world <= height_world:
        cols = target_short_side
        rows = max(1, round(target_short_side * height_world / width_world))
    else:
        rows = target_short_side
        cols = max(1, round(target_short_side * width_world / height_world))
    return (rows, cols)


def _mask_bbox(mask: ScalarImage, extent: Extent) -> Extent:
    """World-coordinate bounding box of the nonzero region of a mask."""
    nonzero = np.argwhere(mask > 0)
    if nonzero.size == 0:
        raise RuntimeError("Expected sofa mask to contain at least one nonzero pixel.")

    y_min_idx, x_min_idx = nonzero.min(axis=0)
    y_max_idx, x_max_idx = nonzero.max(axis=0)
    height, width = mask.shape
    x_min, x_max, y_min, y_max = extent
    x_step = (x_max - x_min) / (width - 1)
    y_step = (y_max - y_min) / (height - 1)
    return (
        float(x_min + x_step * x_min_idx),
        float(x_min + x_step * x_max_idx),
        float(y_min + y_step * y_min_idx),
        float(y_min + y_step * y_max_idx),
    )


def compute_corridor_replay_extent(
    frames: Sequence[FrameData],
    final_sofa: ScalarImage,
    sofa_extent: Extent,
    corridor_width: float,
) -> Extent:
    """Compute a zoomed-out corridor-local extent that fits the full replay."""
    x0, x1, y0, y1 = _mask_bbox(final_sofa, sofa_extent)
    corners = np.array([[x0, y0], [x0, y1], [x1, y0], [x1, y1]], dtype=np.float32)

    all_points = [
        np.column_stack(_world_to_corridor_coords(corners[:, 0], corners[:, 1], f.pose))
        for f in frames
    ]
    half = corridor_width / 2
    all_points.append(
        np.array(
            [[-half, -half], [-half, half], [half, -half], [half, half]],
            dtype=np.float32,
        )
    )
    stacked = np.concatenate(all_points, axis=0)

    margin = corridor_width + 0.5 * max(x1 - x0, y1 - y0)
    return (
        float(stacked[:, 0].min()) - margin,
        float(stacked[:, 0].max()) + margin,
        float(stacked[:, 1].min()) - margin,
        float(stacked[:, 1].max()) + margin,
    )


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------


def rotate_image_counterclockwise(image: NDArray[np.float32]) -> NDArray[np.float32]:
    """Rotate an image 90 degrees counterclockwise."""
    return np.rot90(image, k=1).copy()


def rotate_extent_counterclockwise(extent: Extent) -> Extent:
    """Rotate display coordinates 90 degrees counterclockwise."""
    x_min, x_max, y_min, y_max = extent
    return (-y_max, -y_min, x_min, x_max)


# ---------------------------------------------------------------------------
# Composite rendering
# ---------------------------------------------------------------------------


def _floor_from_world_coords(
    world_x: NDArray[np.float32],
    world_y: NDArray[np.float32],
) -> ColorImage:
    """Bilinear-sample the floor texture at world-space coordinates.

    Tiles X and Y independently so the texture's aspect ratio is preserved.
    Bilinear blending smooths the aliasing that nearest-neighbour lookup would
    introduce when render pixel size differs from texel size.
    """
    tex_h, tex_w = FLOOR_TEXTURE.shape[:2]
    fx = (world_x / FLOOR_TILE_WORLD_SIZE_X) % 1.0 * tex_w
    fy = (world_y / FLOOR_TILE_WORLD_SIZE_Y) % 1.0 * tex_h
    col0 = np.floor(fx).astype(np.int32) % tex_w
    row0 = np.floor(fy).astype(np.int32) % tex_h
    col1 = (col0 + 1) % tex_w
    row1 = (row0 + 1) % tex_h
    dx = (fx - np.floor(fx))[..., None].astype(np.float32)
    dy = (fy - np.floor(fy))[..., None].astype(np.float32)
    top = FLOOR_TEXTURE[row0, col0] * (1.0 - dx) + FLOOR_TEXTURE[row0, col1] * dx
    bot = FLOOR_TEXTURE[row1, col0] * (1.0 - dx) + FLOOR_TEXTURE[row1, col1] * dx
    return (top * (1.0 - dy) + bot * dy).astype(np.float32)


def _tiled_floor(shape: tuple[int, int]) -> ColorImage:
    """Pixel-space tiling fallback used when no world grid is available."""
    h, w = shape
    th, tw = FLOOR_TEXTURE.shape[:2]
    rh = (h + th - 1) // th
    rw = (w + tw - 1) // tw
    return np.tile(FLOOR_TEXTURE, (rh, rw, 1))[:h, :w].copy()


def _build_background(
    shape: tuple[int, int],
    corridor_mask: ScalarImage,
    world_x: NDArray[np.float32] | None,
    world_y: NDArray[np.float32] | None,
    glow: ScalarImage | None,
) -> ColorImage:
    """Floor + walls (with optional glow) — everything that isn't the sofa."""
    if world_x is not None and world_y is not None:
        rgb = _floor_from_world_coords(world_x, world_y)
    else:
        rgb = _tiled_floor(shape)
    rgb = 0.78 * rgb + 0.08

    wall = corridor_mask == 0
    rgb[wall] = DARK_GRAY

    if glow is not None:
        glow_u8 = (np.minimum(glow, 1.0) * 255).astype(np.uint8)
        glow_img = (
            Image.fromarray(glow_u8)
            .filter(ImageFilter.MaxFilter(5))
            .filter(ImageFilter.GaussianBlur(min(shape) / 40))
        )
        alpha = np.asarray(glow_img, dtype=np.float32) / 255.0
        alpha_wall = alpha[wall][:, None]
        rgb[wall] = rgb[wall] * (1.0 - alpha_wall) + GLOW_ORANGE * alpha_wall

    return rgb


def _overlay_sofa(
    background: ColorImage,
    sofa: ScalarImage,
    corridor_mask: ScalarImage,
) -> ColorImage:
    """Paint the sofa (with shading and dark boundary) over a prepared background."""
    sofa_present = (corridor_mask == 1) & (sofa == 1)
    sofa_u8 = sofa_present.astype(np.uint8) * 255
    sofa_img = Image.fromarray(sofa_u8)

    sofa_eroded = np.asarray(sofa_img.filter(ImageFilter.MinFilter(3))).astype(bool)
    sofa_blur = sofa_img.filter(ImageFilter.GaussianBlur(min(sofa.shape) / 20))
    sofa_shade = 0.55 + 0.45 * np.asarray(sofa_blur, dtype=np.float32) / 255.0

    rgb = background.copy()
    rgb[sofa_present] = STEEL_BLUE * sofa_shade[sofa_present][:, None]
    rgb[sofa_present & ~sofa_eroded] = STEEL_BLUE * 0.45
    return rgb


def build_composite(
    sofa: ScalarImage,
    corridor_mask: ScalarImage,
    world_x: NDArray[np.float32] | None = None,
    world_y: NDArray[np.float32] | None = None,
    glow: ScalarImage | None = None,
) -> ColorImage:
    """Build an RGB image from sofa and corridor mask arrays.

    When world_x/world_y are provided the floor texture is sampled at fixed
    world-space coordinates so tiles remain stationary as the view moves.
    Without them the texture is tiled in pixel space (wandb image logging).
    """
    background = _build_background(sofa.shape, corridor_mask, world_x, world_y, glow)
    return _overlay_sofa(background, sofa, corridor_mask)


def _build_erosion_glow(
    erosion_masks: Sequence[ScalarImage],
    frame_idx: int,
    erosion_peak: float,
    sofa_extent: Extent,
    target_extent: Extent,
    output_shape: tuple[int, int],
) -> ScalarImage:
    """Composite recent erosion masks into a decaying world-space glow."""
    glow = np.zeros(output_shape, dtype=np.float32)
    if erosion_peak <= 0.0:
        return glow
    for lag, decay in enumerate((1.0, 0.65, 0.38, 0.18)):
        if frame_idx < lag:
            break
        erosion = erosion_masks[frame_idx - lag]
        glow += sample_image_in_extent(
            erosion, sofa_extent, target_extent, output_shape
        ) * np.float32(decay * erosion.sum() / erosion_peak)
    return glow


# ---------------------------------------------------------------------------
# GIF encoding
# ---------------------------------------------------------------------------


def _render_palette_frame(fig: matplotlib.figure.Figure) -> Image.Image:
    """Rasterize the current figure into a paletted image for GIF encoding."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)  # type: ignore[attr-defined]
    return Image.fromarray(rgba, mode="RGBA").convert("P", palette=Image.ADAPTIVE)  # type: ignore[attr-defined]


def _write_streaming_gif(
    fig: matplotlib.figure.Figure,
    output_path: Path,
    fps: int,
    num_frames: int,
    draw_frame: Callable[[int], None],
) -> None:
    """Write a GIF incrementally so long trajectories do not fill memory."""
    frame_duration_ms = max(1, round(1000 / fps))
    with output_path.open("wb") as file_obj:
        for frame_idx in range(num_frames):
            draw_frame(frame_idx)
            frame = _render_palette_frame(fig)
            encoder_info: dict[str, int | bool] = {
                "duration": frame_duration_ms,
                "disposal": 2,
            }
            if frame_idx == 0:
                for block in GifImagePlugin._get_global_header(frame, {"loop": 0}):
                    file_obj.write(block)
            else:
                encoder_info["include_color_table"] = True
            GifImagePlugin._write_frame_data(file_obj, frame, (0, 0), encoder_info)
        file_obj.write(b";")
        file_obj.flush()


# ---------------------------------------------------------------------------
# Main render entry point
# ---------------------------------------------------------------------------


def _shrink_extent(extent: Extent, factor: float) -> Extent:
    """Shrink an extent symmetrically about its centre by `factor` (0..1)."""
    x_min, x_max, y_min, y_max = extent
    cx, cy = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
    hx, hy = 0.5 * (x_max - x_min) * factor, 0.5 * (y_max - y_min) * factor
    return (cx - hx, cx + hx, cy - hy, cy + hy)


def _square_extent(extent: Extent) -> Extent:
    """Expand an extent symmetrically so its width equals its height."""
    x_min, x_max, y_min, y_max = extent
    cx, cy = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
    half = 0.5 * max(x_max - x_min, y_max - y_min)
    return (cx - half, cx + half, cy - half, cy + half)


def render_trajectory(
    frames: Sequence[FrameData],
    output_path: Path,
    sofa_extent: Extent,
    fps: int = 10,
    corridor_width: float = 1.0,
) -> Path:
    """Render a sequence of frames into an animated gif."""
    if len(frames) == 0:
        raise ValueError("Cannot render an empty trajectory.")

    # Extents -------------------------------------------------------------
    sofa_shape = frames[0].sofa.shape
    left_pad = max(4, int(round(min(sofa_shape) * LEFT_PAD_FRACTION)))
    left_extent = _square_extent(expand_extent(sofa_extent, sofa_shape, left_pad))

    corridor_extent = _shrink_extent(
        compute_corridor_replay_extent(
            frames, frames[-1].sofa, sofa_extent, corridor_width
        ),
        factor=0.6,
    )
    rotated_corridor_extent = rotate_extent_counterclockwise(corridor_extent)

    # Per-panel render shapes chosen to match each extent's aspect ratio so
    # world-space sampling (including the floor texture) is isotropic.
    left_shape = _aspect_matched_shape(left_extent, PANEL_TARGET_SHORT_SIDE)
    right_shape = _aspect_matched_shape(corridor_extent, PANEL_TARGET_SHORT_SIDE)

    # Static right-panel data: corridor mask, floor+walls background, pose-independent.
    right_corridor = build_corridor_mask(corridor_extent, right_shape, corridor_width)
    right_wx, right_wy = _make_grid(corridor_extent, right_shape)
    right_background = _build_background(
        right_shape, right_corridor, right_wx, right_wy, glow=None
    )

    # Left-panel fixed pixel grid (transformed to corridor-local per frame).
    left_sx, left_sy = _make_grid(left_extent, left_shape)

    # Erosion sequence for the left-panel glow.
    final_sofa = frames[-1].sofa
    erosion_masks = [np.zeros_like(final_sofa)] + [
        np.maximum(frames[idx - 1].sofa - frames[idx].sofa, 0.0).astype(np.float32)
        for idx in range(1, len(frames))
    ]
    erosion_peak = max(float(mask.sum()) for mask in erosion_masks)

    # Figure setup --------------------------------------------------------
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=120)

    left_dummy = np.zeros((*left_shape, 3), dtype=np.float32)
    right_dummy = np.zeros((right_shape[1], right_shape[0], 3), dtype=np.float32)

    left_artist = ax_left.imshow(
        left_dummy, origin="lower", extent=left_extent, interpolation="bilinear"
    )
    ax_left.set_axis_off()
    ax_left.set_aspect("equal")
    left_title = ax_left.set_title(f"Sofa frame - step {frames[0].step}", pad=TITLE_PAD)

    right_artist = ax_right.imshow(
        right_dummy,
        origin="lower",
        extent=rotated_corridor_extent,
        interpolation="bilinear",
    )
    ax_right.set_axis_off()
    ax_right.set_aspect("equal")
    ax_right.set_title(f"Corridor frame - area {frames[-1].area:.3f}", pad=TITLE_PAD)

    fig.tight_layout()

    # Per-frame drawing ---------------------------------------------------
    def draw_frame(frame_idx: int) -> None:
        frame = frames[frame_idx]

        left_sofa = sample_image_in_extent(
            frame.sofa, sofa_extent, left_extent, left_shape
        )
        left_corridor = build_world_corridor_mask(
            frame.pose, left_extent, left_shape, corridor_width
        )
        left_cx, left_cy = _world_to_corridor_coords(left_sx, left_sy, frame.pose)
        left_glow = _build_erosion_glow(
            erosion_masks,
            frame_idx,
            erosion_peak,
            sofa_extent,
            left_extent,
            left_shape,
        )
        left_artist.set_data(
            build_composite(
                left_sofa,
                left_corridor,
                world_x=left_cx,
                world_y=left_cy,
                glow=left_glow,
            )
        )
        left_title.set_text(f"Sofa frame - step {frame.step}")

        right_sofa = sample_sofa_in_corridor_frame(
            final_sofa,
            frame.pose,
            sofa_extent,
            corridor_extent,
            right_shape,
        )
        right_img = _overlay_sofa(right_background, right_sofa, right_corridor)
        right_artist.set_data(rotate_image_counterclockwise(right_img))

    # Encode --------------------------------------------------------------
    gif_path = output_path.with_suffix(".gif")
    with tqdm(total=len(frames), desc="Rendering", unit="frame") as pbar:

        def render_and_update(frame_idx: int) -> None:
            draw_frame(frame_idx)
            pbar.update(1)

        _write_streaming_gif(fig, gif_path, fps, len(frames), render_and_update)

    plt.close(fig)
    return gif_path
