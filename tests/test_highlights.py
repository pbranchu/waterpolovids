"""Tests for wpv.render.highlights — scoring, segment selection, I/O, montage."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from wpv.render.highlights import (
    HighlightSegment,
    ScoredMoment,
    build_montage,
    load_segments,
    save_segments,
    score_track,
    select_segments,
)
from wpv.tracking.tracker import TrackPoint, TrackResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_track(
    points: list[TrackPoint] | None = None,
    frame_count: int = 100,
    fps: float = 25.0,
) -> TrackResult:
    """Build a minimal TrackResult for testing."""
    return TrackResult(
        clip_name="test",
        video_path="test.mp4",
        fps=fps,
        frame_count=frame_count,
        width=1920,
        height=1080,
        detection_scale=0.5,
        points=points or [],
        gaps=[],
        stats={},
        elapsed_s=0.0,
    )


def _tracking_points(n: int, xs: np.ndarray, ys: np.ndarray) -> list[TrackPoint]:
    """Build TrackPoints with state='tracking'."""
    return [
        TrackPoint(frame=i, x=float(xs[i]), y=float(ys[i]),
                   confidence=0.9, state="tracking")
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# score_track tests
# ---------------------------------------------------------------------------


class TestScoreTrack:
    def test_empty_track(self):
        track = _make_track(frame_count=0)
        assert score_track(track) == []

    def test_returns_one_per_frame(self):
        n = 50
        xs = np.linspace(0, 100, n)
        ys = np.full(n, 50.0)
        track = _make_track(points=_tracking_points(n, xs, ys), frame_count=n)
        scores = score_track(track)
        assert len(scores) == n
        assert all(isinstance(s, ScoredMoment) for s in scores)

    def test_scores_bounded_0_1(self):
        n = 200
        rng = np.random.RandomState(42)
        xs = np.cumsum(rng.randn(n) * 10) + 500
        ys = np.cumsum(rng.randn(n) * 10) + 500
        track = _make_track(points=_tracking_points(n, xs, ys), frame_count=n)
        scores = score_track(track)
        for s in scores:
            assert 0.0 <= s.score <= 1.0, f"Score {s.score} out of bounds at frame {s.frame}"

    def test_speed_spike_detected(self):
        """A sudden position jump should produce a high speed score."""
        n = 100
        xs = np.full(n, 100.0)
        ys = np.full(n, 100.0)
        # Insert a spike at frame 50
        xs[50] = 500.0
        track = _make_track(points=_tracking_points(n, xs, ys), frame_count=n)
        scores = score_track(track)
        # Frame 50 or 51 should have an elevated speed component
        spike_speed = max(scores[50].speed, scores[51].speed)
        median_speed = np.median([s.speed for s in scores])
        assert spike_speed > median_speed

    def test_state_transition_bonus(self):
        """Frames where state goes from non-tracking → tracking get a bonus."""
        n = 20
        points = []
        for i in range(n):
            state = "search_forward" if i < 10 else "tracking"
            points.append(
                TrackPoint(frame=i, x=100.0, y=100.0, confidence=0.9, state=state)
            )
        track = _make_track(points=points, frame_count=n)
        scores = score_track(track, gap_reappear_bonus=0.5)
        assert scores[10].state_transition == 0.5
        assert scores[9].state_transition == 0.0
        assert scores[11].state_transition == 0.0

    def test_direction_change(self):
        """A sharp reversal should have a higher direction_change score."""
        n = 100
        xs = np.zeros(n)
        ys = np.zeros(n)
        # Move right for first half, then left
        for i in range(n):
            if i < 50:
                xs[i] = float(i * 5)
            else:
                xs[i] = float((99 - i) * 5)
            ys[i] = 100.0
        track = _make_track(points=_tracking_points(n, xs, ys), frame_count=n, fps=25.0)
        scores = score_track(track, direction_window_s=0.5)
        # Around frame 50, direction_change should spike
        reversal_scores = [s.direction_change for s in scores[48:55]]
        edge_scores = [s.direction_change for s in scores[10:20]]
        assert max(reversal_scores) > max(edge_scores) if edge_scores else True

    def test_no_tracking_points_gives_zero_scores(self):
        """Track with frame_count > 0 but no points → all zeros."""
        track = _make_track(frame_count=50)
        scores = score_track(track)
        assert len(scores) == 50
        assert all(s.score == 0.0 for s in scores)


# ---------------------------------------------------------------------------
# select_segments tests
# ---------------------------------------------------------------------------


class TestSelectSegments:
    def test_empty_scores(self):
        assert select_segments([], fps=25.0) == []

    def test_no_scores_above_threshold(self):
        scores = [ScoredMoment(frame=i, score=0.1, speed=0.1,
                               direction_change=0.0, state_transition=0.0)
                  for i in range(100)]
        assert select_segments(scores, fps=25.0, threshold=0.5) == []

    def test_single_high_region(self):
        """A contiguous block of high scores → one segment."""
        scores = []
        for i in range(200):
            s = 0.8 if 50 <= i < 100 else 0.1
            scores.append(ScoredMoment(frame=i, score=s, speed=s,
                                       direction_change=0.0, state_transition=0.0))
        segs = select_segments(scores, fps=25.0, threshold=0.4,
                               context_s=1.0, min_segment_s=1.0,
                               target_duration_s=300.0)
        assert len(segs) == 1
        seg = segs[0]
        # Should contain the high-score frames plus context
        assert seg.start_frame <= 50
        assert seg.end_frame >= 100
        assert seg.peak_score == pytest.approx(0.8)

    def test_target_duration_limits_segments(self):
        """Greedy fill should stop once target_duration_s is reached."""
        # Create 10 separate high-score regions, each ~4s at 25fps (100 frames)
        scores = []
        for i in range(2500):
            region = i // 250  # 0..9
            offset = i % 250
            s = 0.9 - region * 0.01 if 100 <= offset < 200 else 0.1
            scores.append(ScoredMoment(frame=i, score=s, speed=s,
                                       direction_change=0.0, state_transition=0.0))
        segs = select_segments(scores, fps=25.0, threshold=0.4,
                               context_s=0.5, min_segment_s=1.0,
                               target_duration_s=20.0)
        total_dur = sum(s.end_s - s.start_s for s in segs)
        # Should not vastly exceed 20s
        assert total_dur <= 30.0

    def test_segments_sorted_chronologically(self):
        """Returned segments should be in start_frame order."""
        scores = []
        for i in range(500):
            s = 0.8 if (i % 100) >= 40 and (i % 100) < 60 else 0.1
            scores.append(ScoredMoment(frame=i, score=s, speed=s,
                                       direction_change=0.0, state_transition=0.0))
        segs = select_segments(scores, fps=25.0, threshold=0.4,
                               context_s=0.5, min_segment_s=0.5,
                               target_duration_s=300.0)
        for i in range(len(segs) - 1):
            assert segs[i].start_frame <= segs[i + 1].start_frame


# ---------------------------------------------------------------------------
# Segment I/O tests
# ---------------------------------------------------------------------------


class TestSegmentIO:
    def test_save_load_roundtrip(self, tmp_path: Path):
        segs = [
            HighlightSegment(start_frame=0, end_frame=100,
                             start_s=0.0, end_s=4.0,
                             peak_score=0.9, mean_score=0.6),
            HighlightSegment(start_frame=200, end_frame=350,
                             start_s=8.0, end_s=14.0,
                             peak_score=0.85, mean_score=0.55),
        ]
        path = tmp_path / "segs.json"
        save_segments(segs, path)
        loaded = load_segments(path)
        assert len(loaded) == 2
        assert loaded[0].start_frame == 0
        assert loaded[1].peak_score == pytest.approx(0.85)

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "sub" / "dir" / "segs.json"
        save_segments([], path)
        assert path.exists()
        assert json.loads(path.read_text()) == []


# ---------------------------------------------------------------------------
# build_montage tests (mocked ffmpeg)
# ---------------------------------------------------------------------------


class TestBuildMontage:
    def test_no_segments_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="No segments"):
            build_montage([], source_video="src.mp4",
                          output_path=tmp_path / "out.mp4")

    def test_no_ffmpeg_raises(self, tmp_path: Path):
        seg = HighlightSegment(start_frame=0, end_frame=100,
                               start_s=0.0, end_s=4.0,
                               peak_score=0.9, mean_score=0.6)
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="ffmpeg not found"):
                build_montage([seg], source_video="src.mp4",
                              output_path=tmp_path / "out.mp4")
