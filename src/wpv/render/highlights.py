"""Highlight extraction: score track data, select segments, build montage."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from wpv.tracking.tracker import TrackResult


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ScoredMoment:
    """Per-frame highlight score with component breakdown."""

    frame: int
    score: float  # composite 0-1
    speed: float  # normalized speed component
    direction_change: float  # normalized direction-change component
    state_transition: float  # bonus for ball reappearance


@dataclass
class HighlightSegment:
    """A contiguous segment selected for the highlight reel."""

    start_frame: int
    end_frame: int  # exclusive
    start_s: float
    end_s: float
    peak_score: float
    mean_score: float


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_track(
    track: TrackResult,
    speed_sigma_threshold: float = 2.0,
    direction_window_s: float = 0.5,
    gap_reappear_bonus: float = 0.3,
) -> list[ScoredMoment]:
    """Score each frame based on speed, direction change, and state transitions.

    Pure numpy — no video I/O required.
    """
    n = track.frame_count
    fps = track.fps
    if n == 0:
        return []

    # Build per-frame position arrays (NaN where no data)
    xs = np.full(n, np.nan, dtype=np.float64)
    ys = np.full(n, np.nan, dtype=np.float64)
    states = [""] * n

    for pt in track.points:
        if 0 <= pt.frame < n:
            xs[pt.frame] = pt.x
            ys[pt.frame] = pt.y
            states[pt.frame] = pt.state

    # --- Speed component ---
    # Finite differences on positions
    dx = np.diff(xs, prepend=xs[0])
    dy = np.diff(ys, prepend=ys[0])
    speed = np.sqrt(dx**2 + dy**2)
    # NaN where either endpoint was NaN
    speed[0] = 0.0

    valid_speed = speed[np.isfinite(speed)]
    if len(valid_speed) > 0:
        mean_spd = np.nanmean(valid_speed)
        std_spd = np.nanstd(valid_speed)
        threshold = mean_spd + speed_sigma_threshold * std_spd
        if std_spd > 0 and threshold > mean_spd:
            # Linear ramp from mean to threshold → 0..1
            speed_norm = np.clip((speed - mean_spd) / (threshold - mean_spd), 0, 1)
        else:
            speed_norm = np.zeros(n)
    else:
        speed_norm = np.zeros(n)

    # Replace NaN with 0
    speed_norm = np.where(np.isfinite(speed_norm), speed_norm, 0.0)

    # --- Direction change component ---
    window = max(1, int(round(direction_window_s * fps)))
    angles = np.arctan2(dy, dx)

    dir_change = np.zeros(n)
    for i in range(window, n):
        a1 = angles[i - window]
        a2 = angles[i]
        if np.isfinite(a1) and np.isfinite(a2):
            diff = abs(a2 - a1)
            diff = min(diff, 2 * np.pi - diff)
            dir_change[i] = diff

    max_dir = np.max(dir_change) if np.any(dir_change > 0) else 1.0
    dir_norm = dir_change / max_dir if max_dir > 0 else dir_change

    # --- State transition bonus ---
    state_bonus = np.zeros(n)
    non_tracking = {"", "init", "search_forward", "rewind_backward", "gap_bridge"}
    for i in range(1, n):
        prev = states[i - 1]
        curr = states[i]
        if prev in non_tracking and curr == "tracking":
            state_bonus[i] = gap_reappear_bonus

    # --- Composite ---
    composite = (speed_norm + dir_norm) / 2 + state_bonus
    composite = np.clip(composite, 0, 1)

    return [
        ScoredMoment(
            frame=i,
            score=float(composite[i]),
            speed=float(speed_norm[i]),
            direction_change=float(dir_norm[i]),
            state_transition=float(state_bonus[i]),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Segment selection
# ---------------------------------------------------------------------------


def select_segments(
    scores: list[ScoredMoment],
    fps: float,
    threshold: float = 0.4,
    context_s: float = 3.0,
    min_segment_s: float = 3.0,
    target_duration_s: float = 300.0,
    max_segments: int = 50,
) -> list[HighlightSegment]:
    """Select highlight segments from scored moments.

    1. Threshold → binary mask
    2. Label contiguous runs
    3. Expand by ±context_s
    4. Merge overlapping
    5. Drop short segments
    6. Rank by peak_score, greedily fill to target_duration_s
    7. Return chronologically sorted
    """
    if not scores:
        return []

    n = len(scores)
    score_arr = np.array([s.score for s in scores])

    # 1. Binary mask
    mask = score_arr >= threshold

    if not np.any(mask):
        return []

    # 2. Label contiguous runs
    diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]  # exclusive

    # 3. Expand by context
    context_frames = int(round(context_s * fps))
    starts = np.clip(starts - context_frames, 0, n)
    ends = np.clip(ends + context_frames, 0, n)

    # 4. Merge overlapping
    merged_starts = []
    merged_ends = []
    for s, e in zip(starts, ends):
        if merged_starts and s <= merged_ends[-1]:
            merged_ends[-1] = max(merged_ends[-1], e)
        else:
            merged_starts.append(s)
            merged_ends.append(e)

    # 5. Build segments, drop short ones
    min_frames = int(round(min_segment_s * fps))
    segments = []
    for s, e in zip(merged_starts, merged_ends):
        if e - s < min_frames:
            continue
        seg_scores = score_arr[s:e]
        segments.append(
            HighlightSegment(
                start_frame=int(s),
                end_frame=int(e),
                start_s=s / fps,
                end_s=e / fps,
                peak_score=float(np.max(seg_scores)),
                mean_score=float(np.mean(seg_scores)),
            )
        )

    # 6. Rank by peak_score, greedily fill target duration
    segments.sort(key=lambda seg: seg.peak_score, reverse=True)
    selected = []
    total_dur = 0.0
    for seg in segments:
        if len(selected) >= max_segments:
            break
        dur = seg.end_s - seg.start_s
        if total_dur + dur > target_duration_s and selected:
            continue
        selected.append(seg)
        total_dur += dur

    # 7. Sort chronologically
    selected.sort(key=lambda seg: seg.start_frame)
    return selected


# ---------------------------------------------------------------------------
# Segment I/O
# ---------------------------------------------------------------------------


def save_segments(segments: list[HighlightSegment], path: str | Path) -> None:
    """Write highlight segments to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(s) for s in segments]
    path.write_text(json.dumps(data, indent=2))


def load_segments(path: str | Path) -> list[HighlightSegment]:
    """Read highlight segments from JSON."""
    data = json.loads(Path(path).read_text())
    return [HighlightSegment(**d) for d in data]


# ---------------------------------------------------------------------------
# Montage builder
# ---------------------------------------------------------------------------


def build_montage(
    segments: list[HighlightSegment],
    source_video: str | Path,
    output_path: str | Path,
    fps: float = 30.0,
    crossfade_s: float = 0.5,
    codec: str = "libx264",
    crf: int = 20,
    preset: str = "medium",
) -> Path:
    """Build a highlight montage by extracting + concatenating segments via ffmpeg.

    Two-phase approach (no frame-level Python I/O):
    1. Extract each segment with stream copy (fast)
    2. Concatenate with xfade transitions (single re-encode pass)

    Single segment → just copy, no xfade.
    """
    source_video = Path(source_video)
    output_path = Path(output_path)

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")

    if not segments:
        raise ValueError("No segments to build montage from")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="wpv_montage_") as tmpdir:
        tmp = Path(tmpdir)
        seg_paths = []

        # Phase 1: extract segments (stream copy — fast)
        for i, seg in enumerate(segments):
            seg_path = tmp / f"seg_{i:04d}.mp4"
            seg_paths.append(seg_path)
            duration = seg.end_s - seg.start_s
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{seg.start_s:.3f}",
                "-t", f"{duration:.3f}",
                "-i", str(source_video),
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                str(seg_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True)

        if len(seg_paths) == 1:
            # Single segment — just copy to output
            _copy_single(seg_paths[0], output_path, codec, crf, preset)
        else:
            # Multiple segments — concatenate with xfade
            _concat_with_xfade(
                seg_paths, output_path, crossfade_s, codec, crf, preset,
            )

    return output_path


def _copy_single(
    src: Path, dst: Path, codec: str, crf: int, preset: str,
) -> None:
    """Re-encode a single segment to the output (ensures consistent encoding)."""
    is_nvenc = "nvenc" in codec
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-c:v", codec,
    ]
    if is_nvenc:
        cmd += ["-qp", str(crf), "-preset", preset]
    else:
        cmd += ["-crf", str(crf), "-preset", preset]
    cmd += ["-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _concat_with_xfade(
    seg_paths: list[Path],
    output_path: Path,
    crossfade_s: float,
    codec: str,
    crf: int,
    preset: str,
) -> None:
    """Concatenate segments with xfade + acrossfade transitions."""
    n = len(seg_paths)

    # Probe durations for xfade offset calculation
    durations = []
    for p in seg_paths:
        dur = _probe_duration(p)
        durations.append(dur)

    # Build filter_complex for video xfade chain
    inputs = []
    for p in seg_paths:
        inputs += ["-i", str(p)]

    # Video xfade chain
    # [0:v][1:v] xfade=...[v01]; [v01][2:v] xfade=...[v012]; ...
    v_filters = []
    offset = durations[0] - crossfade_s
    prev_label = "[0:v]"
    for i in range(1, n):
        out_label = f"[v{i}]" if i < n - 1 else "[vout]"
        v_filters.append(
            f"{prev_label}[{i}:v]xfade=transition=fade:duration={crossfade_s}:offset={offset:.3f}{out_label}"
        )
        prev_label = out_label
        if i < n - 1:
            offset += durations[i] - crossfade_s

    # Audio acrossfade chain
    a_filters = []
    a_prev = "[0:a]"
    for i in range(1, n):
        a_out = f"[a{i}]" if i < n - 1 else "[aout]"
        a_filters.append(
            f"{a_prev}[{i}:a]acrossfade=d={crossfade_s}:c1=tri:c2=tri{a_out}"
        )
        a_prev = a_out

    filter_complex = ";".join(v_filters + a_filters)
    is_nvenc = "nvenc" in codec
    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", codec,
    ]
    if is_nvenc:
        cmd += ["-qp", str(crf), "-preset", preset]
    else:
        cmd += ["-crf", str(crf), "-preset", preset]
    cmd += [
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        # Fallback: concat without audio crossfade (source may lack audio)
        _concat_video_only(
            seg_paths, output_path, durations, crossfade_s, codec, crf, preset,
        )


def _concat_video_only(
    seg_paths: list[Path],
    output_path: Path,
    durations: list[float],
    crossfade_s: float,
    codec: str,
    crf: int,
    preset: str,
) -> None:
    """Fallback: xfade on video only, no audio crossfade."""
    n = len(seg_paths)
    inputs = []
    for p in seg_paths:
        inputs += ["-i", str(p)]

    v_filters = []
    offset = durations[0] - crossfade_s
    prev_label = "[0:v]"
    for i in range(1, n):
        out_label = f"[v{i}]" if i < n - 1 else "[vout]"
        v_filters.append(
            f"{prev_label}[{i}:v]xfade=transition=fade:duration={crossfade_s}:offset={offset:.3f}{out_label}"
        )
        prev_label = out_label
        if i < n - 1:
            offset += durations[i] - crossfade_s

    filter_complex = ";".join(v_filters)
    is_nvenc = "nvenc" in codec
    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-c:v", codec,
    ]
    if is_nvenc:
        cmd += ["-qp", str(crf), "-preset", preset]
    else:
        cmd += ["-crf", str(crf), "-preset", preset]
    cmd += [
        "-pix_fmt", "yuv420p",
        "-an",
        "-movflags", "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _probe_duration(path: Path) -> float:
    """Get video duration in seconds via ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())
