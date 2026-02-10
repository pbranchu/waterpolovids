"""Single-lens crop/pan render: crop a window from a large source frame and
smoothly follow the tracked ball position.  Zooms out when the ball is lost."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from wpv.tracking.tracker import TrackResult


# ---------------------------------------------------------------------------
# PoolBounds – pool geometry extracted from the pool mask
# ---------------------------------------------------------------------------


@dataclass
class FisheyeIntrinsics:
    """Fisheye lens intrinsic parameters (single Insta360 X5 lens)."""

    focal_length_px: float
    center_x: float
    center_y: float


def fisheye_hfov_for_zoom(base_hfov_deg: float, zoom: float) -> float:
    """Scale the output HFOV by the zoom factor."""
    return base_hfov_deg * zoom


def build_fisheye_to_perspective_maps(
    cx_fish: float,
    cy_fish: float,
    out_w: int,
    out_h: int,
    hfov_deg: float,
    intrinsics: FisheyeIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (map_x, map_y) remap arrays: perspective output ← fisheye source.

    For each output pixel (u, v), computes the corresponding source pixel in
    the fisheye frame using equidistant projection.

    Parameters
    ----------
    cx_fish, cy_fish : float
        Look-direction in fisheye pixel coordinates (where the virtual camera
        is pointed).
    out_w, out_h : int
        Output perspective image dimensions.
    hfov_deg : float
        Horizontal field of view of the output perspective image.
    intrinsics : FisheyeIntrinsics
        Fisheye lens intrinsic parameters.

    Returns
    -------
    map_x, map_y : np.ndarray
        Float32 arrays of shape (out_h, out_w) for use with cv2.remap().
    """
    f_fish = intrinsics.focal_length_px
    ocx = intrinsics.center_x
    ocy = intrinsics.center_y

    hfov_rad = np.radians(hfov_deg)
    f_out = (out_w / 2.0) / np.tan(hfov_rad / 2.0)

    # Look direction: vector from fisheye optical centre to (cx_fish, cy_fish)
    dx = cx_fish - ocx
    dy = cy_fish - ocy
    r_pix = np.sqrt(dx * dx + dy * dy)

    if r_pix < 1e-6:
        # Looking straight along optical axis — forward is (0, 0, 1)
        forward = np.array([0.0, 0.0, 1.0])
    else:
        phi = np.arctan2(dy, dx)
        theta = r_pix / f_fish
        forward = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ])

    # Build rotation matrix: columns are right, down, forward
    up_hint = np.array([0.0, -1.0, 0.0])
    if abs(np.dot(forward, up_hint)) > 0.999:
        up_hint = np.array([0.0, 0.0, 1.0])

    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)

    # Rotation matrix: world = R @ camera_ray
    R = np.column_stack([right, down, forward])  # (3, 3)

    # Generate output pixel grid
    u = np.arange(out_w, dtype=np.float64) - out_w / 2.0
    v = np.arange(out_h, dtype=np.float64) - out_h / 2.0
    uu, vv = np.meshgrid(u, v)  # (out_h, out_w)

    # Camera-frame rays: (u/f, v/f, 1), then normalize
    rays_cam = np.stack([uu / f_out, vv / f_out, np.ones_like(uu)], axis=-1)
    norms = np.linalg.norm(rays_cam, axis=-1, keepdims=True)
    rays_cam /= norms

    # Rotate to world frame
    rays_world = np.einsum("ij,hwj->hwi", R, rays_cam)

    # Project world rays to fisheye pixels (equidistant model)
    wx = rays_world[..., 0]
    wy = rays_world[..., 1]
    wz = rays_world[..., 2]

    # theta = angle from optical axis
    wz_clamped = np.clip(wz, -1.0, 1.0)
    theta_world = np.arccos(wz_clamped)
    phi_world = np.arctan2(wy, wx)

    r_fish = f_fish * theta_world
    map_x = (ocx + r_fish * np.cos(phi_world)).astype(np.float32)
    map_y = (ocy + r_fish * np.sin(phi_world)).astype(np.float32)

    return map_x, map_y


@dataclass
class PoolBounds:
    """Bounding box of the pool in pixel coordinates.

    *x_min* / *x_max* include goal padding (extend beyond the water edge
    so the crop can show the goals).  *y_min* includes camera-side padding
    for residual fisheye distortion.  The crop must stay within these
    bounds vertically.
    """

    x_min: int
    x_max: int
    y_min: int
    y_max: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def cx(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def cy(self) -> float:
        """Vertical centre of the pool — also the goal y-position."""
        return (self.y_min + self.y_max) / 2.0

    @classmethod
    def from_polygon(
        cls,
        points: list[list[int]],
        goal_pad_px: int = 150,
        camera_pad_px: int = 50,
        frame_w: int = 4608,
        frame_h: int = 4608,
    ) -> "PoolBounds":
        """Build pool bounds from a manually-set 4-point polygon.

        The polygon comes from the game-area mask (``game_masks.json``).
        *goal_pad_px* extends the x-axis beyond the goal lines so that
        the crop window can include the goals.  *camera_pad_px* extends
        the y-axis toward the camera (smaller y) to account for residual
        fisheye distortion at the pool edge.
        """
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return cls(
            x_min=max(0, min(xs) - goal_pad_px),
            x_max=min(frame_w, max(xs) + goal_pad_px),
            y_min=max(0, min(ys) - camera_pad_px),
            y_max=max(ys),
        )


def load_game_mask_bounds(
    game_masks_path: str | Path,
    clip_name: str,
    goal_pad_px: int = 150,
    camera_pad_px: int = 50,
    frame_w: int = 4608,
    frame_h: int = 4608,
) -> PoolBounds:
    """Load pool bounds from a game_masks.json file for a given clip.

    Parameters
    ----------
    game_masks_path : path to game_masks.json
    clip_name : key in the JSON, e.g. "clip_000"
    goal_pad_px : extend x-axis beyond goal lines (each side)
    camera_pad_px : extend y-axis toward camera (smaller y)
    """
    import json

    with open(game_masks_path) as f:
        masks = json.load(f)
    if clip_name not in masks:
        raise KeyError(f"Clip {clip_name!r} not found in {game_masks_path}")
    return PoolBounds.from_polygon(
        masks[clip_name],
        goal_pad_px=goal_pad_px,
        camera_pad_px=camera_pad_px,
        frame_w=frame_w,
        frame_h=frame_h,
    )


def detect_pool_bounds(
    video_path: str | Path,
    goal_pad_px: int = 150,
    camera_pad_px: int = 50,
) -> PoolBounds:
    """Auto-detect pool boundaries from the first video frame.

    Fallback when no manual game mask is available.  Uses
    :func:`~wpv.tracking.detector.detect_pool_mask` (blue-water
    thresholding) to find the water region, then applies padding.
    """
    import cv2

    from wpv.tracking.detector import detect_pool_mask

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read first frame from {video_path}")

    mask = detect_pool_mask(frame)
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        h, w = frame.shape[:2]
        return PoolBounds(0, w, 0, h)

    fh, fw = frame.shape[:2]
    return PoolBounds(
        x_min=max(0, int(xs.min()) - goal_pad_px),
        x_max=min(fw, int(xs.max()) + goal_pad_px),
        y_min=max(0, int(ys.min()) - camera_pad_px),
        y_max=int(ys.max()),
    )


# ---------------------------------------------------------------------------
# CropPath – per-frame crop centres, zoom, and output dimensions
# ---------------------------------------------------------------------------


@dataclass
class CropPath:
    """Per-frame crop centres and zoom factors.

    At each frame the crop rectangle is::

        w = base_w * zooms[i]
        h = base_h * zooms[i]
        x = cxs[i] - w/2   (clamped)
        y = cys[i] - h/2   (clamped)

    The output is always scaled to *base_w* x *base_h*.
    """

    cxs: np.ndarray    # shape (N,) float – crop centre x
    cys: np.ndarray    # shape (N,) float – crop centre y
    zooms: np.ndarray  # shape (N,) float – zoom factor (1.0 = tight)
    base_w: int        # output width (e.g. 1280)
    base_h: int        # output height (e.g. 720)

    # Backward compat aliases
    @property
    def crop_w(self) -> int:
        return self.base_w

    @property
    def crop_h(self) -> int:
        return self.base_h


def compute_crop_path(
    track: TrackResult,
    crop_w: int = 1280,
    crop_h: int = 720,
    alpha: float = 0.08,
    dead_zone: float = 20.0,
    max_vel: float = 120.0,
    end_zone_frac: float = 0.25,
    pool_bounds: PoolBounds | None = None,
    max_zoom: float = 0.0,
    zoom_out_delay_s: float = 0.5,
    zoom_out_alpha: float = 0.06,
    zoom_in_alpha: float = 0.04,
) -> CropPath:
    """Build a smoothed crop path with dynamic zoom from tracker output.

    Four passes:
    1. **Fill** – accept only high-confidence ("tracking" state) points,
       remove aberrations, fill gaps with adaptive interpolation.
    2. **Zoom** – compute per-frame zoom factor.  When the ball is lost
       for more than *zoom_out_delay_s*, rapidly zoom out.  When the ball
       is re-acquired, zoom back in.
    3. **Smooth** – zone-aware target selection (lock at goals, follow
       ball in transit) with uniform smoothing speed everywhere.
    4. **Clamp** – keep the crop window inside the frame, respecting
       pool bounds for vertical positioning.

    Parameters
    ----------
    track : TrackResult
        Tracker output.
    crop_w, crop_h : int
        Base output crop size (at zoom 1.0).
    alpha : float
        EMA smoothing factor for camera position (uniform everywhere).
    dead_zone : float
        Minimum displacement (px) before the camera starts moving.
    max_vel : float
        Max camera displacement per frame (px).
    end_zone_frac : float
        Fraction of pool width for each end zone.  0 disables zones.
    pool_bounds : PoolBounds | None
        Pool geometry.
    max_zoom : float
        Maximum zoom-out factor.  0 = auto-compute from pool bounds
        (enough to show the full pool width).
    zoom_out_delay_s : float
        Seconds of ball loss before zoom-out starts.
    zoom_out_alpha : float
        EMA speed for zooming out (higher = faster).
    zoom_in_alpha : float
        EMA speed for zooming back in (higher = faster).
    """
    n = track.frame_count
    src_w = track.width
    src_h = track.height
    fps = track.fps or 25.0

    # --- Pass 1: fill raw centres from high-confidence points ---------------
    raw = np.full((n, 2), np.nan, dtype=np.float64)
    for pt in track.points:
        if 0 <= pt.frame < n and 0 <= pt.x <= src_w and 0 <= pt.y <= src_h:
            if getattr(pt, "state", "tracking") == "tracking":
                raw[pt.frame] = [pt.x, pt.y]

    _remove_aberrations(raw, max_jump_px=500, max_burst_frames=10)

    # Remember which frames have real data BEFORE gap fill (for zoom)
    has_ball = np.isfinite(raw[:, 0]).copy()

    # Adaptive gap fill
    _fill_gaps_hold_snap(raw, fps=fps)

    # Default remaining NaNs to pool centre (or frame centre)
    if pool_bounds is not None:
        fallback = np.array([pool_bounds.cx, pool_bounds.cy])
    else:
        fallback = np.array([src_w / 2, src_h / 2])
    nans = np.isnan(raw[:, 0])
    raw[nans] = fallback

    # --- Pass 2: zoom -------------------------------------------------------
    if max_zoom <= 0:
        if pool_bounds is not None:
            max_zoom = max(1.0, pool_bounds.width / crop_w)
        else:
            max_zoom = max(1.0, src_w / crop_w)
    # Cap zoom so crop doesn't exceed frame
    max_zoom = min(max_zoom, src_w / crop_w, src_h / crop_h)

    zooms = _compute_zoom(
        has_ball, n, fps,
        delay_s=zoom_out_delay_s,
        max_zoom=max_zoom,
        zoom_out_alpha=zoom_out_alpha,
        zoom_in_alpha=zoom_in_alpha,
    )

    # --- Pass 3: smooth (uniform speed, zone-aware targets) -----------------
    if end_zone_frac > 0 and src_w > crop_w:
        smooth = _smooth_zone_aware(
            raw, src_w, src_h, crop_w, crop_h,
            end_zone_frac, dead_zone, max_vel, alpha,
            pool_bounds=pool_bounds,
        )
    else:
        smooth = _smooth_simple(raw, alpha, dead_zone, max_vel)

    # --- Pass 4: clamp centres so the crop stays within bounds --------------
    cxs = smooth[:, 0].copy()
    cys = smooth[:, 1].copy()

    for i in range(n):
        z = zooms[i]
        cw = crop_w * z
        ch = crop_h * z

        # X: clamp so crop fits in frame
        half_w = cw / 2.0
        cxs[i] = np.clip(cxs[i], half_w, src_w - half_w)

        # Y: clamp relative to pool or frame
        half_h = ch / 2.0
        if pool_bounds is not None:
            if pool_bounds.height >= ch:
                cys[i] = np.clip(cys[i],
                                 pool_bounds.y_min + half_h,
                                 pool_bounds.y_max - half_h)
            else:
                cys[i] = pool_bounds.cy
        else:
            cys[i] = np.clip(cys[i], half_h, src_h - half_h)

    return CropPath(
        cxs=cxs,
        cys=cys,
        zooms=zooms,
        base_w=crop_w,
        base_h=crop_h,
    )


def _compute_zoom(
    has_ball: np.ndarray,
    n: int,
    fps: float,
    delay_s: float = 0.5,
    max_zoom: float = 2.5,
    zoom_out_alpha: float = 0.06,
    zoom_in_alpha: float = 0.04,
) -> np.ndarray:
    """Compute per-frame zoom factor based on ball tracking status.

    Zoom stays at 1.0 while the ball is tracked.  After *delay_s* of
    continuous ball loss, zoom ramps toward *max_zoom*.  When the ball
    is re-acquired, zoom eases back to 1.0.
    """
    delay_frames = int(delay_s * fps)
    zooms = np.ones(n, dtype=np.float64)

    frames_lost = 0
    z = 1.0

    for i in range(n):
        if has_ball[i]:
            frames_lost = 0
            target = 1.0
        else:
            frames_lost += 1
            target = max_zoom if frames_lost > delay_frames else 1.0

        a = zoom_out_alpha if target > z else zoom_in_alpha
        z += a * (target - z)
        zooms[i] = z

    return zooms


def _smooth_simple(
    raw: np.ndarray, alpha: float, dead_zone: float, max_vel: float,
) -> np.ndarray:
    """Uniform EMA smoothing (used when end_zone_frac == 0)."""
    n = len(raw)
    smooth = np.empty_like(raw)
    smooth[0] = raw[0]
    for i in range(1, n):
        delta = raw[i] - smooth[i - 1]
        dist = np.linalg.norm(delta)
        if dist < dead_zone:
            smooth[i] = smooth[i - 1]
        else:
            candidate = smooth[i - 1] + alpha * delta
            step = candidate - smooth[i - 1]
            step_len = np.linalg.norm(step)
            if step_len > max_vel:
                step = step * (max_vel / step_len)
            smooth[i] = smooth[i - 1] + step
    return smooth


def _smooth_zone_aware(
    raw: np.ndarray,
    src_w: int, src_h: int,
    crop_w: int, crop_h: int,
    end_zone_frac: float,
    dead_zone: float,
    max_vel: float,
    alpha: float,
    pool_bounds: PoolBounds | None = None,
) -> np.ndarray:
    """Zone-aware smoothing with uniform camera speed.

    Zones affect WHERE the camera targets (lock at goals vs follow ball)
    but NOT the smoothing speed — alpha and max_vel are the same everywhere.

    A 200 px transition band blends targets so there is no discontinuity
    at zone boundaries.
    """
    n = len(raw)

    # --- derive zone geometry from pool bounds or frame --------------------
    if pool_bounds is not None:
        pool_left = pool_bounds.x_min
        pool_right = pool_bounds.x_max
        pool_w = pool_right - pool_left
        end_left = pool_left + pool_w * end_zone_frac
        end_right = pool_right - pool_w * end_zone_frac
        cam_left = pool_left + crop_w / 2.0
        cam_right = pool_right - crop_w / 2.0
        goal_y = pool_bounds.cy
    else:
        end_left = src_w * end_zone_frac
        end_right = src_w * (1 - end_zone_frac)
        cam_left = crop_w / 2.0
        cam_right = src_w - crop_w / 2.0
        goal_y = src_h / 2.0

    margin = 100.0
    half_vis = crop_w / 2.0 - margin
    band = 200.0

    # --- per-frame target (zone blending, uniform speed) -------------------
    target = np.empty((n, 2))

    for i in range(n):
        bx = raw[i, 0]
        by = raw[i, 1]

        if bx < end_left + band:
            preferred = cam_left
            lo = bx - half_vis
            hi = bx + half_vis
            locked = np.clip(preferred, lo, hi)
            t = np.clip((bx - (end_left - band)) / (2 * band), 0, 1)
            target[i, 0] = locked * (1 - t) + bx * t
            target[i, 1] = goal_y * (1 - t) + by * t
        elif bx > end_right - band:
            preferred = cam_right
            lo = bx - half_vis
            hi = bx + half_vis
            locked = np.clip(preferred, lo, hi)
            t = np.clip(((end_right + band) - bx) / (2 * band), 0, 1)
            target[i, 0] = locked * (1 - t) + bx * t
            target[i, 1] = goal_y * (1 - t) + by * t
        else:
            target[i, 0] = bx
            target[i, 1] = by

    # --- smooth with uniform alpha/max_vel --------------------------------
    smooth = np.empty((n, 2))
    smooth[0] = target[0]
    for i in range(1, n):
        delta = target[i] - smooth[i - 1]
        dist = np.linalg.norm(delta)
        if dist < dead_zone:
            smooth[i] = smooth[i - 1]
        else:
            step = alpha * delta
            step_len = np.linalg.norm(step)
            if step_len > max_vel:
                step = step * (max_vel / step_len)
            smooth[i] = smooth[i - 1] + step

    return smooth


def _fill_gaps_hold_snap(
    raw: np.ndarray,
    fps: float = 25.0,
    long_gap_s: float = 2.0,
    long_hold_frac: float = 0.3,
) -> None:
    """Fill NaN gaps in-place with adaptive hold-then-interpolate.

    Short gaps (< *long_gap_s*): pure linear interpolation.
    Long gaps (≥ *long_gap_s*): hold *long_hold_frac* then interpolate.
    """
    n = len(raw)
    long_gap_frames = int(long_gap_s * fps)

    for axis in range(2):
        col = raw[:, axis]
        valid = np.where(np.isfinite(col))[0]
        if len(valid) == 0:
            continue
        if len(valid) == 1:
            col[:] = col[valid[0]]
            continue

        result = np.full(n, np.nan)
        for v in valid:
            result[v] = col[v]

        result[: valid[0]] = col[valid[0]]
        result[valid[-1] + 1 :] = col[valid[-1]]

        for j in range(len(valid) - 1):
            start = valid[j]
            end = valid[j + 1]
            gap_len = end - start
            if gap_len <= 1:
                continue

            if gap_len >= long_gap_frames:
                hold_end = start + max(1, int(gap_len * long_hold_frac))
                result[start + 1 : hold_end] = col[start]
                interp_len = end - hold_end
                if interp_len > 0:
                    t = np.linspace(0, 1, interp_len + 1)[1:]
                    result[hold_end:end] = col[start] + t * (col[end] - col[start])
            else:
                t = np.linspace(0, 1, gap_len + 1)[1:-1]
                result[start + 1 : end] = col[start] + t * (col[end] - col[start])

        col[:] = result


def _remove_aberrations(
    raw: np.ndarray,
    max_jump_px: float = 500.0,
    max_burst_frames: int = 10,
) -> None:
    """Remove physically impossible tracking bursts in-place.

    Segments the valid points into contiguous runs, splitting wherever
    two consecutive valid frames are more than *max_jump_px* apart.
    Any resulting short run (≤ *max_burst_frames*) that is far from
    both the nearest long run before it and the nearest long run after
    it is set to NaN.
    """
    n = len(raw)
    valid = np.isfinite(raw[:, 0])

    segments: list[tuple[int, int]] = []
    start: int | None = None
    for i in range(n):
        if valid[i]:
            if start is None:
                start = i
            elif (
                valid[i - 1]
                and np.linalg.norm(raw[i] - raw[i - 1]) > max_jump_px
            ):
                segments.append((start, i))
                start = i
        else:
            if start is not None:
                segments.append((start, i))
                start = None
    if start is not None:
        segments.append((start, n))

    if len(segments) < 2:
        return

    for idx, (s, e) in enumerate(segments):
        if e - s > max_burst_frames:
            continue

        seg_mean = np.nanmean(raw[s:e], axis=0)

        prev_pos = None
        for j in range(idx - 1, -1, -1):
            ps, pe = segments[j]
            if pe - ps > max_burst_frames:
                prev_pos = raw[pe - 1]
                break

        next_pos = None
        for j in range(idx + 1, len(segments)):
            ns, ne = segments[j]
            if ne - ns > max_burst_frames:
                next_pos = raw[ns]
                break

        has_prev = prev_pos is not None
        has_next = next_pos is not None

        if not has_prev and not has_next:
            continue

        far_prev = (
            has_prev
            and np.linalg.norm(seg_mean - prev_pos) > max_jump_px
        )
        far_next = (
            has_next
            and np.linalg.norm(seg_mean - next_pos) > max_jump_px
        )

        if (far_prev or not has_prev) and (far_next or not has_next):
            raw[s:e] = np.nan


# ---------------------------------------------------------------------------
# FFmpegWriter – pipe raw frames to ffmpeg
# ---------------------------------------------------------------------------


class FFmpegWriter:
    """Context-managed subprocess that accepts raw BGR24 frames on *stdin*
    and writes an H.264 MP4 to *output_path*.

    Audio is copied from the *source_path* file (if present).
    """

    def __init__(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        fps: float,
        source_path: str | Path | None = None,
        codec: str = "libx264",
        crf: int = 20,
        preset: str = "medium",
    ) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH")

        self._output_path = str(output_path)
        self._width = width
        self._height = height
        self._fps = fps
        self._source_path = str(source_path) if source_path else None
        self._codec = codec
        self._crf = crf
        self._preset = preset
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> "FFmpegWriter":
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self._width}x{self._height}",
            "-r", str(self._fps),
            "-i", "pipe:0",
        ]
        if self._source_path is not None:
            cmd += ["-i", self._source_path]
            cmd += ["-map", "0:v", "-map", "1:a?"]
        is_nvenc = "nvenc" in self._codec
        cmd += ["-c:v", self._codec]
        if is_nvenc:
            cmd += ["-qp", str(self._crf), "-preset", self._preset]
        else:
            cmd += ["-crf", str(self._crf), "-preset", self._preset]
        cmd += ["-pix_fmt", "yuv420p"]
        if self._source_path is not None:
            cmd += ["-c:a", "copy"]
        cmd += ["-movflags", "+faststart", self._output_path]

        # Redirect stderr to a temp file to avoid pipe buffer deadlock.
        # ffmpeg writes continuous progress to stderr; a PIPE buffer (64KB)
        # fills up during long encodes and deadlocks both processes.
        self._stderr_file = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr_file,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        if self._proc is None:
            return
        self._proc.stdin.close()
        self._proc.wait()
        if self._proc.returncode != 0:
            self._stderr_file.seek(0)
            stderr = self._stderr_file.read().decode(errors="replace")
            self._stderr_file.close()
            raise RuntimeError(
                f"ffmpeg exited with code {self._proc.returncode}:\n{stderr[-2000:]}"
            )
        self._stderr_file.close()

    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single BGR24 uint8 frame (H×W×3) to the pipe."""
        assert self._proc is not None, "FFmpegWriter not entered as context manager"
        self._proc.stdin.write(frame.tobytes())


# ---------------------------------------------------------------------------
# CropRenderer – orchestrator
# ---------------------------------------------------------------------------


class CropRenderer:
    """Crop-and-pan renderer with dynamic zoom."""

    def __init__(
        self,
        video_path: str | Path,
        track: TrackResult,
        output_path: str | Path,
        crop_w: int = 1280,
        crop_h: int = 720,
        alpha: float = 0.08,
        dead_zone: float = 20.0,
        max_vel: float = 120.0,
        codec: str = "libx264",
        crf: int = 20,
        preset: str = "medium",
        preview: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
        pool_bounds: PoolBounds | None = None,
        start_frame: int = 0,
        end_frame: int = 0,
        undistort: bool = True,
        fisheye_intrinsics: FisheyeIntrinsics | None = None,
        hfov_deg: float = 70.0,
    ) -> None:
        self.video_path = Path(video_path)
        self.track = track
        self.output_path = Path(output_path)
        self.crop_w = crop_w
        self.crop_h = crop_h
        self.alpha = alpha
        self.dead_zone = dead_zone
        self.max_vel = max_vel
        self.codec = codec
        self.crf = crf
        self.preset = preset
        self.preview = preview
        self.progress_callback = progress_callback
        self.pool_bounds = pool_bounds
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.undistort = undistort
        self.hfov_deg = hfov_deg

        if undistort and fisheye_intrinsics is None:
            from wpv.config import settings
            fisheye_intrinsics = FisheyeIntrinsics(
                focal_length_px=settings.fisheye_focal_length_px,
                center_x=settings.fisheye_center_x,
                center_y=settings.fisheye_center_y,
            )
        self.fisheye_intrinsics = fisheye_intrinsics

    def run(self) -> Path:
        """Execute the crop-and-pan render, returning the output path.

        Always uses the Python frame-by-frame path (required for dynamic
        zoom).  Falls back to ffmpeg native crop when zoom is constant.
        """
        pool_bounds = self.pool_bounds
        if pool_bounds is None:
            pool_bounds = detect_pool_bounds(self.video_path)

        crop_path = compute_crop_path(
            self.track,
            crop_w=self.crop_w,
            crop_h=self.crop_h,
            alpha=self.alpha,
            dead_zone=self.dead_zone,
            max_vel=self.max_vel,
            pool_bounds=pool_bounds,
        )

        has_zoom = not np.allclose(crop_path.zooms, 1.0)
        if (
            has_zoom
            or self.undistort
            or self.preview
            or self.start_frame > 0
            or self.end_frame > 0
        ):
            return self._run_python(crop_path)
        return self._run_ffmpeg_native(crop_path)

    def _run_ffmpeg_native(self, crop_path: CropPath) -> Path:
        """Render using ffmpeg's crop filter (zoom == 1.0 throughout)."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        src_w = self.track.width
        src_h = self.track.height
        max_x = src_w - self.crop_w
        max_y = src_h - self.crop_h

        # Convert centres to top-left
        xs = np.clip(crop_path.cxs - self.crop_w / 2, 0, max_x).astype(int)
        ys = np.clip(crop_path.cys - self.crop_h / 2, 0, max_y).astype(int)

        x_expr = _build_crop_expr(xs, max_x, keyframe_step=25)
        y_expr = _build_crop_expr(ys, max_y, keyframe_step=25)
        vf = f"crop={self.crop_w}:{self.crop_h}:{x_expr}:{y_expr}"

        is_nvenc = "nvenc" in self.codec
        cmd = [
            "ffmpeg", "-y",
            "-i", str(self.video_path),
            "-vf", vf,
            "-c:v", self.codec,
        ]
        if is_nvenc:
            cmd += ["-qp", str(self.crf), "-preset", self.preset]
        else:
            cmd += ["-crf", str(self.crf), "-preset", self.preset]
        cmd += [
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(self.output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg crop render failed (rc={result.returncode}):\n"
                + result.stderr[-1000:]
            )
        return self.output_path

    def _run_python(self, crop_path: CropPath) -> Path:
        """Frame-by-frame Python render with dynamic zoom support."""
        import cv2

        from wpv.tracking.video_reader import VideoReader

        out_w = self.crop_w
        out_h = self.crop_h
        crf = self.crf
        preset = self.preset

        if self.preview:
            out_w = self.crop_w // 2
            out_h = self.crop_h // 2
            crf = 28
            preset = "ultrafast"

        src_w = self.track.width
        src_h = self.track.height

        # Frame range
        start = self.start_frame
        end = self.end_frame if self.end_frame > 0 else self.track.frame_count

        reader = VideoReader(self.video_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with FFmpegWriter(
            output_path=self.output_path,
            width=out_w,
            height=out_h,
            fps=reader.fps,
            source_path=self.video_path,
            codec=self.codec,
            crf=crf,
            preset=preset,
        ) as writer:
            for frame_idx, frame in reader.sequential_frames():
                if frame_idx < start:
                    continue
                if frame_idx >= end:
                    break
                if frame_idx >= len(crop_path.cxs):
                    break

                z = crop_path.zooms[frame_idx]
                cx = crop_path.cxs[frame_idx]
                cy = crop_path.cys[frame_idx]

                if self.undistort and self.fisheye_intrinsics is not None:
                    hfov = fisheye_hfov_for_zoom(self.hfov_deg, z)
                    map_x, map_y = build_fisheye_to_perspective_maps(
                        cx, cy, out_w, out_h, hfov, self.fisheye_intrinsics,
                    )
                    remapped = cv2.remap(
                        frame, map_x, map_y,
                        cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0),
                    )
                    writer.write_frame(remapped)
                else:
                    cw = int(crop_path.base_w * z)
                    ch = int(crop_path.base_h * z)
                    x1 = int(np.clip(cx - cw / 2, 0, src_w - cw))
                    y1 = int(np.clip(cy - ch / 2, 0, src_h - ch))
                    crop = frame[y1 : y1 + ch, x1 : x1 + cw]
                    resized = cv2.resize(crop, (out_w, out_h),
                                         interpolation=cv2.INTER_AREA)
                    writer.write_frame(resized)

                if self.progress_callback is not None:
                    self.progress_callback(
                        frame_idx - start + 1, end - start,
                    )

        return self.output_path


def _build_crop_expr(
    values: np.ndarray, max_val: int, keyframe_step: int = 25,
) -> str:
    """Encode per-frame crop positions as an ffmpeg expression.

    Uses a balanced binary tree of ``if(lt(n,...), ...)`` so the nesting
    depth is O(log N) — well within ffmpeg's limits.  Keyframes are
    sampled every *keyframe_step* frames with linear interpolation
    in between.
    """
    n = len(values)
    kf_i = list(range(0, n, keyframe_step))
    if kf_i[-1] != n - 1:
        kf_i.append(n - 1)
    kf_v = [int(values[i]) for i in kf_i]

    def _tree(frames, vals, lo, hi):
        if hi - lo <= 1:
            return str(vals[lo])
        if hi - lo == 2:
            n0, n1 = frames[lo], frames[lo + 1]
            v0, v1 = vals[lo], vals[lo + 1]
            if v0 == v1:
                return str(v0)
            dn = n1 - n0
            return f"clip({v0}+({v1}-{v0})*(n-{n0})/{dn}\\,0\\,{max_val})"
        mid = (lo + hi) // 2
        left = _tree(frames, vals, lo, mid + 1)
        right = _tree(frames, vals, mid, hi)
        return f"if(lt(n\\,{frames[mid]})\\,{left}\\,{right})"

    return _tree(kf_i, kf_v, 0, len(kf_i))
