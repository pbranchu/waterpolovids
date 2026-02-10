"""Ball tracker: 5-state machine consuming per-frame detections."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from wpv.tracking.detector import (
    BallDetector,
    Detection,
    detect_pool_mask,
)
from wpv.tracking.kalman import BallKalmanFilter
from wpv.tracking.video_reader import VideoReader


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class TrackState(enum.Enum):
    """Tracker state machine states."""

    INIT = "init"
    TRACKING = "tracking"
    SEARCH_FORWARD = "search_forward"
    REWIND_BACKWARD = "rewind_backward"
    GAP_BRIDGE = "gap_bridge"
    ACTION_PROXY = "action_proxy"


@dataclass
class TrackPoint:
    """A single tracked position in a clip."""

    frame: int
    x: float  # equirect pixel x (full-res space)
    y: float  # equirect pixel y (full-res space)
    confidence: float  # 0.0-1.0
    state: str  # TrackState value


@dataclass
class TrackGap:
    """A gap in tracking (ball lost)."""

    start_frame: int
    end_frame: int
    gap_type: str  # "bridged" or "proxy"


@dataclass
class TrackResult:
    """Complete tracking result for a clip."""

    clip_name: str
    video_path: str
    fps: float
    frame_count: int
    width: int  # full-res width
    height: int  # full-res height
    detection_scale: float
    points: list[TrackPoint] = field(default_factory=list)
    gaps: list[TrackGap] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# BallTracker
# ---------------------------------------------------------------------------


def _smoothstep(t: float) -> float:
    """Hermite smoothstep: 3t^2 - 2t^3, clamped to [0, 1]."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


class BallTracker:
    """Track a ball through a video clip using a 5-state machine.

    States:
        INIT            - read sequentially until first high-confidence detection
        TRACKING        - detect every frame, Kalman update/predict
        SEARCH_FORWARD  - jump ahead to find ball after loss
        REWIND_BACKWARD - binary search back to find exact reappearance
        GAP_BRIDGE      - smoothstep interpolation over gap
        ACTION_PROXY    - hold last position when ball is lost for extended period
    """

    def __init__(
        self,
        detector: BallDetector,
        detection_scale: float = 0.5,
        loss_frames: int = 19,
        search_step_s: float = 5.0,
        search_max_gap_s: float = 45.0,
        rewind_step_s: float = 0.25,
        reacquire_persistence: int = 5,
        gate_distance: float = 5.0,
        confidence_threshold: float = 0.6,
    ):
        self._detector = detector
        self._scale = detection_scale
        self._loss_frames = loss_frames
        self._search_step_s = search_step_s
        self._search_max_gap_s = search_max_gap_s
        self._rewind_step_s = rewind_step_s
        self._reacquire_persistence = reacquire_persistence
        self._gate_distance = gate_distance
        self._confidence_threshold = confidence_threshold

    def track_clip(
        self,
        video_path: str | Path,
        clip_name: str | None = None,
        pool_mask: np.ndarray | None = None,
        game_mask: np.ndarray | None = None,
        hsv_bands: list[tuple[tuple[int, int, int], tuple[int, int, int]]] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> TrackResult:
        """Track the ball through a video clip.

        Parameters
        ----------
        video_path : path to the video file
        clip_name : optional name for the clip (defaults to filename stem)
        pool_mask : binary mask (255 = pool) at full resolution; computed if None
        game_mask : binary mask (255 = game area) at full resolution; optional
        hsv_bands : custom HSV bands for detection
        progress_callback : called as callback(frame_idx, total_frames, state_name)

        Returns
        -------
        TrackResult with per-frame points and gap info.
        """
        video_path = Path(video_path)
        reader = VideoReader(video_path)
        if clip_name is None:
            clip_name = video_path.stem

        result = TrackResult(
            clip_name=clip_name,
            video_path=str(video_path),
            fps=reader.fps,
            frame_count=reader.frame_count,
            width=reader.width,
            height=reader.height,
            detection_scale=self._scale,
        )

        t0 = time.monotonic()
        self._last_progress_t = 0.0
        kalman = BallKalmanFilter(fps=reader.fps)
        state = TrackState.INIT
        consecutive_misses = 0
        loss_x, loss_y = 0.0, 0.0

        state_counts: dict[str, int] = {s.value: 0 for s in TrackState}

        # Detection-scale masks (computed once)
        det_pool_mask: np.ndarray | None = None
        det_game_mask: np.ndarray | None = None

        def _scale_mask(mask: np.ndarray) -> np.ndarray:
            h = int(mask.shape[0] * self._scale)
            w = int(mask.shape[1] * self._scale)
            return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        def _detect(frame_bgr: np.ndarray) -> list[Detection]:
            if self._scale != 1.0:
                h = int(frame_bgr.shape[0] * self._scale)
                w = int(frame_bgr.shape[1] * self._scale)
                small = cv2.resize(frame_bgr, (w, h))
            else:
                small = frame_bgr
            return self._detector.detect(small, pool_mask=det_pool_mask, hsv_bands=hsv_bands)

        def _best(detections: list[Detection]) -> Detection | None:
            above = [d for d in detections if d.confidence >= self._confidence_threshold]
            if not above:
                return None
            if kalman.initialized:
                gated = []
                for d in above:
                    cx, cy = d.candidate.centroid
                    fx, fy = cx / self._scale, cy / self._scale
                    dist = kalman.mahalanobis_distance(fx, fy)
                    if dist < self._gate_distance:
                        gated.append((d, dist))
                if gated:
                    gated.sort(key=lambda x: x[1])
                    return gated[0][0]
                return None
            above.sort(key=lambda d: -d.confidence)
            return above[0]

        def _to_full(det: Detection) -> tuple[float, float]:
            cx, cy = det.candidate.centroid
            return cx / self._scale, cy / self._scale

        # --- Main tracking loop ---
        # We use a frame queue approach: sequential read, with possible
        # interruptions for SEARCH_FORWARD that use seek.
        frame_iter = reader.sequential_frames()
        resume_from: int | None = None

        while True:
            # Get next frame
            try:
                if resume_from is not None:
                    frame_iter = reader.sequential_frames(start=resume_from)
                    resume_from = None
                frame_idx, frame_bgr = next(frame_iter)
            except StopIteration:
                break

            if progress_callback:
                now = time.monotonic()
                if now - self._last_progress_t >= 2.0:
                    progress_callback(frame_idx, reader.frame_count, state.value)
                    self._last_progress_t = now

            state_counts[state.value] += 1

            # Compute pool mask from first frame if not provided
            if pool_mask is None and frame_idx == 0:
                pool_mask = detect_pool_mask(frame_bgr)
            if det_pool_mask is None and pool_mask is not None:
                det_pool_mask = _scale_mask(pool_mask)
            if det_game_mask is None and game_mask is not None:
                det_game_mask = _scale_mask(game_mask)

            # --- INIT ---
            if state == TrackState.INIT:
                detections = _detect(frame_bgr)
                best = _best(detections)
                if best is not None:
                    fx, fy = _to_full(best)
                    kalman.init_state(fx, fy)
                    state = TrackState.TRACKING
                    consecutive_misses = 0
                    result.points.append(
                        TrackPoint(frame_idx, fx, fy, best.confidence, TrackState.TRACKING.value)
                    )
                continue

            # --- TRACKING ---
            if state == TrackState.TRACKING:
                kalman.predict()
                detections = _detect(frame_bgr)
                best = _best(detections)

                if best is not None:
                    fx, fy = _to_full(best)
                    kx, ky, _, _ = kalman.update(fx, fy, best.confidence)
                    consecutive_misses = 0
                    result.points.append(
                        TrackPoint(frame_idx, kx, ky, best.confidence, TrackState.TRACKING.value)
                    )
                else:
                    px, py = kalman.state[0], kalman.state[1]
                    consecutive_misses += 1
                    result.points.append(
                        TrackPoint(frame_idx, px, py, 0.1, TrackState.TRACKING.value)
                    )

                    if consecutive_misses >= self._loss_frames:
                        loss_frame = frame_idx
                        loss_x, loss_y = px, py

                        resume = self._handle_search(
                            reader, kalman, result, loss_frame, loss_x, loss_y,
                            det_pool_mask, hsv_bands, state_counts,
                            progress_callback,
                        )
                        if resume is not None:
                            state = TrackState.TRACKING
                            consecutive_misses = 0
                            resume_from = resume + 1
                        else:
                            state = TrackState.ACTION_PROXY
                            self._fill_action_proxy(
                                reader, result, frame_idx + 1, loss_x, loss_y,
                                det_pool_mask, hsv_bands, kalman, state_counts,
                                progress_callback,
                            )
                            break
                continue

            # --- ACTION_PROXY (shouldn't reach here normally) ---
            if state == TrackState.ACTION_PROXY:
                break

        reader.close()
        result.elapsed_s = time.monotonic() - t0
        result.stats = {
            "state_distribution": state_counts,
            "num_points": len(result.points),
            "num_gaps": len(result.gaps),
            "mean_confidence": (
                sum(p.confidence for p in result.points) / len(result.points)
                if result.points
                else 0.0
            ),
        }
        return result

    def _handle_search(
        self,
        reader: VideoReader,
        kalman: BallKalmanFilter,
        result: TrackResult,
        loss_frame: int,
        loss_x: float,
        loss_y: float,
        det_pool_mask: np.ndarray | None,
        hsv_bands,
        state_counts: dict[str, int],
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> int | None:
        """SEARCH_FORWARD -> GAP_BRIDGE.

        Uses forward scanning instead of random seeks — on H.265, sequential
        decode is ~50x faster than seek (8.6fps vs 6.9s per seek).

        Returns the frame index to resume TRACKING from, or None for ACTION_PROXY.
        """
        fps = reader.fps
        max_gap_frames = int(self._search_max_gap_s * fps)
        # Check every N frames during scan (1 second intervals)
        scan_interval = max(1, int(fps))
        end_frame = min(loss_frame + max_gap_frames, reader.frame_count)

        consecutive = 0
        last_pos: tuple[float, float] | None = None
        found_frame: int | None = None
        found_x, found_y = loss_x, loss_y

        for frame_idx, frame_bgr in reader.sequential_frames(start=loss_frame + 1, end=end_frame):
            state_counts[TrackState.SEARCH_FORWARD.value] += 1

            # Only run detection every scan_interval frames
            if (frame_idx - loss_frame) % scan_interval != 0:
                continue

            if progress_callback:
                progress_callback(frame_idx, reader.frame_count, TrackState.SEARCH_FORWARD.value)

            if self._scale != 1.0:
                h = int(frame_bgr.shape[0] * self._scale)
                w = int(frame_bgr.shape[1] * self._scale)
                small = cv2.resize(frame_bgr, (w, h))
            else:
                small = frame_bgr

            dets = self._detector.detect(small, pool_mask=det_pool_mask, hsv_bands=hsv_bands)
            above = [d for d in dets if d.confidence >= self._confidence_threshold]

            if above:
                best = max(above, key=lambda d: d.confidence)
                cx, cy = best.candidate.centroid
                pos = (cx / self._scale, cy / self._scale)

                if last_pos is not None:
                    dt_frames = max(1, frame_idx - (found_frame or frame_idx))
                    dt_s = dt_frames / fps
                    dist = np.sqrt(
                        (pos[0] - last_pos[0]) ** 2 + (pos[1] - last_pos[1]) ** 2
                    )
                    if dist / max(dt_s, 0.001) > 2000:
                        consecutive = 0
                        last_pos = None
                        found_frame = None
                        continue

                consecutive += 1
                last_pos = pos
                if found_frame is None:
                    found_frame = frame_idx
                    found_x, found_y = pos

                if consecutive >= self._reacquire_persistence:
                    break
            else:
                consecutive = 0
                last_pos = None
                found_frame = None

        if found_frame is None or consecutive < self._reacquire_persistence:
            return None

        # GAP_BRIDGE: smoothstep interpolation from loss to reacquire
        state_counts[TrackState.GAP_BRIDGE.value] += 1
        gap_len = found_frame - loss_frame
        if gap_len > 0:
            for f in range(loss_frame + 1, found_frame):
                t = (f - loss_frame) / gap_len
                s = _smoothstep(t)
                ix = loss_x + s * (found_x - loss_x)
                iy = loss_y + s * (found_y - loss_y)
                result.points.append(
                    TrackPoint(f, ix, iy, 0.3, TrackState.GAP_BRIDGE.value)
                )

        result.gaps.append(TrackGap(loss_frame, found_frame, "bridged"))

        # Re-init Kalman at reacquire point
        kalman.init_state(found_x, found_y)
        result.points.append(
            TrackPoint(found_frame, found_x, found_y, 0.5, TrackState.TRACKING.value)
        )

        return found_frame

    def _fill_action_proxy(
        self,
        reader: VideoReader,
        result: TrackResult,
        start_frame: int,
        hold_x: float,
        hold_y: float,
        det_pool_mask: np.ndarray | None,
        hsv_bands,
        kalman: BallKalmanFilter,
        state_counts: dict[str, int],
        progress_callback,
    ) -> None:
        """ACTION_PROXY: hold last position, keep detecting, resume if found."""
        center_x = reader.width / 2.0
        center_y = reader.height / 2.0
        drift_rate = 0.001

        proxy_start = start_frame
        x, y = hold_x, hold_y

        for frame_idx, frame_bgr in reader.sequential_frames(start=start_frame):
            if progress_callback:
                now = time.monotonic()
                if now - self._last_progress_t >= 2.0:
                    progress_callback(frame_idx, reader.frame_count, TrackState.ACTION_PROXY.value)
                    self._last_progress_t = now

            state_counts[TrackState.ACTION_PROXY.value] += 1

            x += (center_x - x) * drift_rate
            y += (center_y - y) * drift_rate

            if self._scale != 1.0:
                h = int(frame_bgr.shape[0] * self._scale)
                w = int(frame_bgr.shape[1] * self._scale)
                small = cv2.resize(frame_bgr, (w, h))
            else:
                small = frame_bgr

            dets = self._detector.detect(small, pool_mask=det_pool_mask, hsv_bands=hsv_bands)
            above = [d for d in dets if d.confidence >= self._confidence_threshold]

            if above:
                best = max(above, key=lambda d: d.confidence)
                cx, cy = best.candidate.centroid
                fx, fy = cx / self._scale, cy / self._scale

                result.gaps.append(TrackGap(proxy_start, frame_idx, "proxy"))
                kalman.init_state(fx, fy)
                result.points.append(
                    TrackPoint(frame_idx, fx, fy, best.confidence, TrackState.TRACKING.value)
                )

                # Continue tracking remaining frames sequentially
                for frame_idx2, frame_bgr2 in reader.sequential_frames(start=frame_idx + 1):
                    if progress_callback:
                        now = time.monotonic()
                        if now - self._last_progress_t >= 2.0:
                            progress_callback(frame_idx2, reader.frame_count, TrackState.TRACKING.value)
                            self._last_progress_t = now
                    state_counts[TrackState.TRACKING.value] += 1
                    kalman.predict()

                    if self._scale != 1.0:
                        h2 = int(frame_bgr2.shape[0] * self._scale)
                        w2 = int(frame_bgr2.shape[1] * self._scale)
                        small2 = cv2.resize(frame_bgr2, (w2, h2))
                    else:
                        small2 = frame_bgr2

                    dets2 = self._detector.detect(
                        small2, pool_mask=det_pool_mask, hsv_bands=hsv_bands
                    )
                    above2 = [d for d in dets2 if d.confidence >= self._confidence_threshold]
                    best2 = None
                    if above2 and kalman.initialized:
                        gated = []
                        for d in above2:
                            dcx, dcy = d.candidate.centroid
                            dfx, dfy = dcx / self._scale, dcy / self._scale
                            dist = kalman.mahalanobis_distance(dfx, dfy)
                            if dist < self._gate_distance:
                                gated.append((d, dist))
                        if gated:
                            gated.sort(key=lambda g: g[1])
                            best2 = gated[0][0]

                    if best2 is not None:
                        dcx, dcy = best2.candidate.centroid
                        dfx, dfy = dcx / self._scale, dcy / self._scale
                        kx, ky, _, _ = kalman.update(dfx, dfy, best2.confidence)
                        result.points.append(
                            TrackPoint(frame_idx2, kx, ky, best2.confidence, TrackState.TRACKING.value)
                        )
                    else:
                        px, py = kalman.state[0], kalman.state[1]
                        result.points.append(
                            TrackPoint(frame_idx2, px, py, 0.1, TrackState.TRACKING.value)
                        )
                return

            result.points.append(
                TrackPoint(frame_idx, x, y, 0.05, TrackState.ACTION_PROXY.value)
            )

        # Ran out of frames in proxy
        if result.points:
            last_frame = result.points[-1].frame
            if last_frame >= proxy_start:
                result.gaps.append(TrackGap(proxy_start, last_frame, "proxy"))
