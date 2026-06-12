"""Rendering utilities for sofa erosion trajectory videos.

The trajectory video is a single corridor-frame view: the corridor stays
fixed while the sofa travels through it. The final (surviving) sofa shape is
drawn solid; the material that will eventually be eroded surrounds it as a
translucent halo that disappears exactly where and when the simulation
removes it, leaving a brief warm trace at the walls.
"""

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float
from matplotlib.patches import Circle, Rectangle
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

CANVAS = np.array([0.105, 0.11, 0.12], dtype=np.float32)
STEEL_BLUE = np.array([0.27, 0.47, 0.73], dtype=np.float32)
GLOW_ORANGE = np.array([1.0, 0.76, 0.28], dtype=np.float32)
HALO_BLUE = np.array([0.38, 0.55, 0.78], dtype=np.float32)

PANEL_TARGET_SHORT_SIDE: int = 480
# Masks are composited at this multiple of display resolution, then box-
# downsampled — cheap anti-aliasing for every edge in the frame.
SUPERSAMPLE = 2
# Rendered frames per simulation step (poses interpolated in between).
SMOOTH_STEPS = 2

# Area of Gerver's sofa, the best known solution for a width-1 L-corridor.
GERVER_AREA = 2.2074

# Doomed-material halo opacity.
HALO_ALPHA = 0.5
# Drop shadow cast by the block (light from the top-left).
SHADOW_ALPHA = 0.35
# Erosion trace: per-rendered-frame decay and peak opacity of the embers.
TRACE_DECAY = 0.88
TRACE_ALPHA = 0.5
# Frame durations (ms): linger on the intact block and on the solved sofa.
HOLD_FIRST_MS = 1200
HOLD_LAST_MS = 3000

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
    area: float


def compute_frame_data(
    step: int,
    pose: tuple[float, float, float],
    sofa: Float[Tensor, "1 1 H W"],
    cell_area: float,
) -> FrameData:
    """Package tensor data into a FrameData for rendering."""
    return FrameData(
        step=step,
        pose=pose,
        sofa=sofa[0, 0].cpu().numpy(),
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
    """Transform world (sofa-frame) coordinates to corridor-local coordinates."""
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


def build_corridor_mask(
    extent: Extent,
    output_shape: tuple[int, int],
    corridor_width: float,
) -> ScalarImage:
    """Build an L-corridor mask directly in corridor-local coordinates."""
    x_grid, y_grid = _make_grid(extent, output_shape)
    return _is_in_l_corridor(x_grid, y_grid, corridor_width / 2).astype(np.float32)


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


def _trajectory_extent(
    frames: Sequence[FrameData],
    sofa_extent: Extent,
    corridor_width: float,
    margin: float = 0.45,
) -> Extent:
    """Corridor-frame extent containing the original block at every pose."""
    x0, x1, y0, y1 = sofa_extent
    corners = np.array([[x0, y0], [x0, y1], [x1, y0], [x1, y1]], dtype=np.float32)
    points = [
        np.column_stack(_world_to_corridor_coords(corners[:, 0], corners[:, 1], f.pose))
        for f in frames
    ]
    half = corridor_width / 2
    points.append(np.array([[-half, -half], [half, half]], dtype=np.float32))
    stacked = np.concatenate(points, axis=0)
    return (
        float(stacked[:, 0].min()) - margin,
        float(stacked[:, 0].max()) + margin,
        float(stacked[:, 1].min()) - margin,
        float(stacked[:, 1].max()) + margin,
    )


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


def _soft_mask(mask: ScalarImage, radius: float) -> ScalarImage:
    """Gaussian-soften a 0..1 mask via PIL."""
    img = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8))
    return (
        np.asarray(img.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
        / 255.0
    )


def _build_background(
    shape: tuple[int, int],
    corridor_mask: ScalarImage,
    world_x: NDArray[np.float32] | None,
    world_y: NDArray[np.float32] | None,
) -> ColorImage:
    """Parquet floor sunk into the dark canvas.

    Walls share the canvas color; an ambient-occlusion rim darkens the floor
    where it meets them, which is what makes the corridor read as recessed.
    """
    if world_x is not None and world_y is not None:
        rgb = _floor_from_world_coords(world_x, world_y)
    else:
        rgb = _tiled_floor(shape)
    rgb = 0.82 * rgb + 0.06

    wall = corridor_mask == 0
    occlusion = _soft_mask(wall.astype(np.float32), min(shape) / 70)
    rgb *= 1.0 - 0.45 * occlusion[..., None]
    rgb[wall] = CANVAS
    return rgb


def _overlay_sofa(
    background: ColorImage,
    sofa: ScalarImage,
    corridor_mask: ScalarImage,
) -> ColorImage:
    """Paint the sofa over a prepared background.

    Shading combines a soft "puff" from the blurred mask with a directional
    light from the top-left (gradient of the puff), plus a darker boundary rim.
    """
    present = (corridor_mask == 1) & (sofa == 1)
    sofa_img = Image.fromarray(present.astype(np.uint8) * 255)

    interior = np.asarray(sofa_img.filter(ImageFilter.MinFilter(3))).astype(bool)
    puff = (
        np.asarray(
            sofa_img.filter(ImageFilter.GaussianBlur(min(sofa.shape) / 20)),
            dtype=np.float32,
        )
        / 255.0
    )
    grad_y, grad_x = np.gradient(puff)
    ndotl = grad_x - grad_y  # light from the top-left (origin="lower")
    peak = np.abs(ndotl).max()
    if peak > 0:
        ndotl = ndotl / peak
    shade = (0.62 + 0.38 * puff) * (1.0 + 0.22 * ndotl)

    rgb = background.copy()
    rgb[present] = np.clip(STEEL_BLUE * shade[present][:, None], 0.0, 1.0)
    rgb[present & ~interior] = STEEL_BLUE * 0.45
    return rgb


def build_composite(
    sofa: ScalarImage,
    corridor_mask: ScalarImage,
    world_x: NDArray[np.float32] | None = None,
    world_y: NDArray[np.float32] | None = None,
) -> ColorImage:
    """Build an RGB image from sofa and corridor mask arrays.

    When world_x/world_y are provided the floor texture is sampled at fixed
    world-space coordinates so tiles remain stationary as the view moves.
    Without them the texture is tiled in pixel space (wandb image logging).
    """
    background = _build_background(sofa.shape, corridor_mask, world_x, world_y)
    return _overlay_sofa(background, sofa, corridor_mask)


def _blend(rgb: ColorImage, color: NDArray[np.float32], alpha: ScalarImage) -> None:
    """In-place alpha blend of a flat color into rgb."""
    a = alpha[..., None]
    rgb *= 1.0 - a
    rgb += color * a


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
    durations_ms: Sequence[int],
    draw_frame: Callable[[int], None],
) -> None:
    """Write a GIF incrementally so long trajectories do not fill memory.

    `draw_frame` is called once per frame, in order — the renderer's erosion
    trace relies on that sequencing.
    """
    with output_path.open("wb") as file_obj:
        for frame_idx, duration in enumerate(durations_ms):
            draw_frame(frame_idx)
            frame = _render_palette_frame(fig)
            encoder_info: dict[str, int | bool] = {
                "duration": duration,
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


def _downsample(rgb: ColorImage, factor: int) -> ColorImage:
    """Box-downsample a supersampled composite to display resolution."""
    h, w, _ = rgb.shape
    binned = rgb.reshape(h // factor, factor, w // factor, factor, 3)
    return binned.mean(axis=(1, 3), dtype=np.float32)


def _render_plan(
    frames: Sequence[FrameData],
) -> list[tuple[int, tuple[float, float, float]]]:
    """(source frame index, pose) per rendered frame.

    Inserts SMOOTH_STEPS - 1 pose-interpolated frames before each step's
    arrival; interpolated frames reuse the previous step's sofa mask (erosion
    lands when the step completes).
    """
    plan = [(0, frames[0].pose)]
    for i in range(1, len(frames)):
        prev = np.array(frames[i - 1].pose)
        cur = np.array(frames[i].pose)
        for k in range(1, SMOOTH_STEPS):
            t = k / SMOOTH_STEPS
            x, y, theta = prev + t * (cur - prev)
            plan.append((i - 1, (float(x), float(y), float(theta))))
        plan.append((i, frames[i].pose))
    return plan


def render_trajectory(
    frames: Sequence[FrameData],
    output_path: Path,
    sofa_extent: Extent,
    fps: int = 10,
    corridor_width: float = 1.0,
    goal_point: tuple[float, float] | None = None,
    goal_radius: float = 0.3,
) -> Path:
    """Render a trajectory into a single corridor-frame animated gif.

    The final sofa shape is drawn solid; the rest of the original block is a
    translucent halo that decays away as the corridor walls erode it. An area
    gauge tracks the remaining area against Gerver's optimum.
    """
    if len(frames) == 0:
        raise ValueError("Cannot render an empty trajectory.")

    extent = _trajectory_extent(frames, sofa_extent, corridor_width)
    shape = _aspect_matched_shape(extent, PANEL_TARGET_SHORT_SIDE)
    shape_hi = (shape[0] * SUPERSAMPLE, shape[1] * SUPERSAMPLE)
    corridor = build_corridor_mask(extent, shape_hi, corridor_width)
    on_floor = corridor == 1
    world_x, world_y = _make_grid(extent, shape_hi)
    background = _build_background(shape_hi, corridor, world_x, world_y)

    final_sofa = frames[-1].sofa
    trace = np.zeros(shape_hi, dtype=np.float32)
    px = min(shape_hi)
    trace_blur = px / 160
    shadow_blur = px / 110
    shadow_off = round(px / 120)

    plan = _render_plan(frames)
    max_area = frames[0].area
    show_gerver = abs(corridor_width - 1.0) < 1e-9 and GERVER_AREA < max_area

    # Figure: dark full-bleed canvas, manual layout ------------------------
    x_min, x_max, y_min, y_max = extent
    aspect = (x_max - x_min) / (y_max - y_min)
    main_box = (0.04, 0.115, 0.92, 0.80)  # (left, bottom, width, height)
    fig_height = 6.6
    fig_width = fig_height * aspect * main_box[3] / main_box[2]
    canvas_color = tuple(float(c) for c in CANVAS)
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=100, facecolor=canvas_color)
    ax = fig.add_axes(main_box)
    ax.set_facecolor(canvas_color)
    ax.set_axis_off()
    ax.set_aspect("equal")
    artist = ax.imshow(
        np.zeros((*shape, 3), dtype=np.float32),
        origin="lower",
        extent=extent,
        interpolation="bilinear",
    )
    if goal_point is not None:
        ax.add_patch(
            Circle(
                goal_point,
                goal_radius,
                fill=False,
                edgecolor=(1.0, 1.0, 1.0, 0.5),
                linewidth=1.2,
            )
        )

    fig.text(0.05, 0.94, " ".join("THE MOVING SOFA"), fontsize=11, color="0.8")
    readout = fig.text(
        0.95,
        0.94,
        "",
        ha="right",
        family="DejaVu Sans Mono",
        fontsize=10,
        color="0.62",
    )

    # Area gauge: remaining area, with a tick at Gerver's optimum.
    gauge = fig.add_axes((0.32, 0.052, 0.36, 0.012))
    gauge.set_xlim(0, max_area)
    gauge.set_ylim(0, 1)
    gauge.set_axis_off()
    gauge.add_patch(Rectangle((0, 0), max_area, 1, color="0.22"))
    gauge_fill = Rectangle((0, 0), max_area, 1, color=tuple(STEEL_BLUE))
    gauge.add_patch(gauge_fill)
    gauge.text(
        -0.04,
        0.5,
        "area",
        transform=gauge.transAxes,
        ha="right",
        va="center",
        fontsize=8,
        color="0.55",
    )
    if show_gerver:
        gauge.plot(
            [GERVER_AREA, GERVER_AREA], [-0.7, 1.7], color="0.85", lw=1.0, clip_on=False
        )
        gauge.text(
            GERVER_AREA,
            2.6,
            f"Gerver {GERVER_AREA:.3f}",
            ha="center",
            fontsize=7.5,
            color="0.55",
        )

    def in_corridor_frame(
        mask: ScalarImage, pose: tuple[float, float, float]
    ) -> ScalarImage:
        return sample_sofa_in_corridor_frame(mask, pose, sofa_extent, extent, shape_hi)

    # Per-frame drawing ---------------------------------------------------
    prev_src = 0

    def draw_frame(plan_idx: int) -> None:
        nonlocal trace, prev_src
        src_idx, pose = plan[plan_idx]
        frame = frames[src_idx]
        current = in_corridor_frame(frame.sofa, pose)
        final = in_corridor_frame(final_sofa, pose)

        # Material lost this step stays where it was shaved off and fades.
        trace = trace * TRACE_DECAY
        if src_idx != prev_src:
            lost = np.maximum(frames[src_idx - 1].sofa - frame.sofa, 0.0)
            trace = np.maximum(trace, in_corridor_frame(lost, pose))
        prev_src = src_idx

        rgb = background.copy()

        # Drop shadow cast by the remaining block (light from the top-left).
        shadow = _soft_mask(
            np.roll(current, (-shadow_off, shadow_off), axis=(0, 1)), shadow_blur
        )
        rgb *= 1.0 - SHADOW_ALPHA * (shadow * on_floor)[..., None]

        # Embers: screen blend so the erosion trace glows instead of staining.
        ember = (_soft_mask(trace, trace_blur) * TRACE_ALPHA)[..., None]
        rgb = 1.0 - (1.0 - rgb) * (1.0 - ember * GLOW_ORANGE)

        halo = ((current == 1) & (final == 0) & on_floor).astype(np.float32)
        _blend(rgb, HALO_BLUE, halo * HALO_ALPHA)

        artist.set_data(_downsample(_overlay_sofa(rgb, final, corridor), SUPERSAMPLE))
        if plan_idx == len(plan) - 1 and show_gerver:
            readout.set_text(
                f"final area {frame.area:.3f} · {frame.area / GERVER_AREA:.1%}"
                " of optimal"
            )
        else:
            readout.set_text(f"step {frame.step:>3} · area {frame.area:.3f}")
        gauge_fill.set_width(frame.area)

    # Encode --------------------------------------------------------------
    durations = [max(1, round(1000 / (fps * SMOOTH_STEPS)))] * len(plan)
    durations[0] = HOLD_FIRST_MS
    durations[-1] = HOLD_LAST_MS

    gif_path = output_path.with_suffix(".gif")
    with tqdm(total=len(plan), desc="Rendering", unit="frame") as pbar:

        def render_and_update(plan_idx: int) -> None:
            draw_frame(plan_idx)
            pbar.update(1)

        _write_streaming_gif(fig, gif_path, durations, render_and_update)

    plt.close(fig)
    return gif_path
