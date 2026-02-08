"""Pre-publish quality gates for track and highlight data."""

from __future__ import annotations

from dataclasses import dataclass, field

from wpv.render.highlights import HighlightSegment
from wpv.tracking.tracker import TrackResult


@dataclass
class QualityReport:
    """Result of running quality gates."""

    passed: bool
    track_coverage_pct: float
    track_coverage_ok: bool
    highlight_duration_s: float
    highlight_duration_ok: bool
    details: dict = field(default_factory=dict)


def run_quality_gates(
    track: TrackResult,
    segments: list[HighlightSegment] | None = None,
    min_track_coverage_pct: float = 60.0,
    min_highlight_duration_s: float = 180.0,
) -> QualityReport:
    """Run quality gates and return a report.

    Gates:
    - Track coverage: % of frames with state == "tracking" vs threshold.
    - Highlight duration: total segment duration vs minimum (skipped if no segments).
    """
    # --- Track coverage gate ---
    n = track.frame_count
    if n > 0:
        tracking_frames = sum(
            1 for pt in track.points if pt.state == "tracking"
        )
        coverage_pct = (tracking_frames / n) * 100.0
    else:
        coverage_pct = 0.0

    track_ok = coverage_pct >= min_track_coverage_pct

    # --- Highlight duration gate ---
    if segments is not None:
        hl_dur = sum(seg.end_s - seg.start_s for seg in segments)
        hl_ok = hl_dur >= min_highlight_duration_s
    else:
        hl_dur = 0.0
        hl_ok = True  # skip gate when no segments provided

    passed = track_ok and hl_ok

    return QualityReport(
        passed=passed,
        track_coverage_pct=round(coverage_pct, 2),
        track_coverage_ok=track_ok,
        highlight_duration_s=round(hl_dur, 2),
        highlight_duration_ok=hl_ok,
        details={
            "frame_count": n,
            "tracking_frames": sum(
                1 for pt in track.points if pt.state == "tracking"
            ) if n > 0 else 0,
            "min_track_coverage_pct": min_track_coverage_pct,
            "min_highlight_duration_s": min_highlight_duration_s,
            "segments_provided": segments is not None,
            "num_segments": len(segments) if segments else 0,
        },
    )
