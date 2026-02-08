"""Tests for wpv.quality — quality gate logic."""

from __future__ import annotations

from wpv.quality import run_quality_gates
from wpv.render.highlights import HighlightSegment
from wpv.tracking.tracker import TrackPoint, TrackResult


def _make_track(
    n: int = 100,
    tracking_pct: float = 80.0,
    fps: float = 25.0,
) -> TrackResult:
    """Build a TrackResult with a given % of tracking frames."""
    n_tracking = int(n * tracking_pct / 100.0)
    points = []
    for i in range(n):
        state = "tracking" if i < n_tracking else "search_forward"
        points.append(
            TrackPoint(frame=i, x=100.0, y=100.0, confidence=0.9, state=state)
        )
    return TrackResult(
        clip_name="test",
        video_path="test.mp4",
        fps=fps,
        frame_count=n,
        width=1920,
        height=1080,
        detection_scale=0.5,
        points=points,
        gaps=[],
        stats={},
        elapsed_s=0.0,
    )


class TestQualityGates:
    def test_all_gates_pass(self):
        track = _make_track(n=1000, tracking_pct=80.0)
        segments = [
            HighlightSegment(start_frame=0, end_frame=5000,
                             start_s=0.0, end_s=200.0,
                             peak_score=0.9, mean_score=0.6),
        ]
        report = run_quality_gates(track, segments=segments,
                                   min_track_coverage_pct=60.0,
                                   min_highlight_duration_s=180.0)
        assert report.passed is True
        assert report.track_coverage_ok is True
        assert report.highlight_duration_ok is True
        assert report.track_coverage_pct == 80.0

    def test_low_track_coverage_fails(self):
        track = _make_track(n=1000, tracking_pct=30.0)
        report = run_quality_gates(track, min_track_coverage_pct=60.0)
        assert report.passed is False
        assert report.track_coverage_ok is False
        assert report.track_coverage_pct == 30.0

    def test_low_highlight_duration_fails(self):
        track = _make_track(n=1000, tracking_pct=80.0)
        segments = [
            HighlightSegment(start_frame=0, end_frame=100,
                             start_s=0.0, end_s=10.0,
                             peak_score=0.9, mean_score=0.6),
        ]
        report = run_quality_gates(track, segments=segments,
                                   min_highlight_duration_s=180.0)
        assert report.passed is False
        assert report.highlight_duration_ok is False

    def test_no_segments_skips_highlight_gate(self):
        track = _make_track(n=1000, tracking_pct=80.0)
        report = run_quality_gates(track, segments=None,
                                   min_track_coverage_pct=60.0,
                                   min_highlight_duration_s=180.0)
        assert report.passed is True
        assert report.highlight_duration_ok is True
        assert report.highlight_duration_s == 0.0

    def test_empty_track(self):
        track = _make_track(n=0, tracking_pct=0)
        report = run_quality_gates(track, min_track_coverage_pct=60.0)
        assert report.passed is False
        assert report.track_coverage_pct == 0.0
