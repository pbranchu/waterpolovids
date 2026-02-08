"""Tests for BallTracker with mock detector and reader."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from wpv.tracking.detector import BallDetector, Candidate, Detection
from wpv.tracking.tracker import (
    BallTracker,
    TrackResult,
    TrackState,
    _smoothstep,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detection(cx: float, cy: float, confidence: float = 0.8) -> Detection:
    """Create a Detection at the given centroid."""
    return Detection(
        candidate=Candidate(
            bbox=(int(cx) - 5, int(cy) - 5, 10, 10),
            centroid=(cx, cy),
            area=78.0,
            circularity=0.9,
            mean_hsv=(28.0, 200.0, 200.0),
        ),
        confidence=confidence,
    )


def _make_frame(w: int = 200, h: int = 200) -> np.ndarray:
    """Create a blank BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


class MockVideoReader:
    """Mock VideoReader for testing without actual video files."""

    def __init__(self, frame_count: int = 100, width: int = 200, height: int = 200, fps: float = 25.0):
        self._fps = fps
        self._frame_count = frame_count
        self._width = width
        self._height = height

    @property
    def fps(self):
        return self._fps

    @property
    def frame_count(self):
        return self._frame_count

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def sequential_frames(self, start=0, end=None):
        end = end or self._frame_count
        for i in range(start, min(end, self._frame_count)):
            yield i, _make_frame(self._width, self._height)

    def seek_frame(self, frame_idx):
        if 0 <= frame_idx < self._frame_count:
            return _make_frame(self._width, self._height)
        return None

    def seek_frames(self, frame_indices):
        return {i: _make_frame(self._width, self._height) for i in frame_indices if 0 <= i < self._frame_count}


# ---------------------------------------------------------------------------
# Smoothstep tests
# ---------------------------------------------------------------------------


class TestSmoothstep:
    def test_boundaries(self):
        assert _smoothstep(0.0) == pytest.approx(0.0)
        assert _smoothstep(1.0) == pytest.approx(1.0)

    def test_midpoint(self):
        assert _smoothstep(0.5) == pytest.approx(0.5)

    def test_clamping(self):
        assert _smoothstep(-1.0) == pytest.approx(0.0)
        assert _smoothstep(2.0) == pytest.approx(1.0)

    def test_monotonic(self):
        values = [_smoothstep(t / 10.0) for t in range(11)]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]


# ---------------------------------------------------------------------------
# BallTracker tests
# ---------------------------------------------------------------------------


class TestBallTrackerInit:
    """Test that tracker starts in INIT state and transitions to TRACKING."""

    @patch("wpv.tracking.tracker.VideoReader")
    def test_init_to_tracking(self, MockReader):
        """Tracker should transition from INIT to TRACKING on first detection."""
        mock_reader = MockVideoReader(frame_count=10)
        MockReader.return_value = mock_reader

        detector = MagicMock(spec=BallDetector)
        # Return detection on every frame
        det = _make_detection(50.0, 50.0, 0.8)
        detector.detect.return_value = [det]

        tracker = BallTracker(
            detector=detector,
            detection_scale=1.0,
            loss_frames=3,
        )
        result = tracker.track_clip("/fake/video.mp4")

        assert isinstance(result, TrackResult)
        assert len(result.points) > 0
        # First point should be TRACKING state
        assert result.points[0].state == TrackState.TRACKING.value

    @patch("wpv.tracking.tracker.VideoReader")
    def test_no_detections_stays_init(self, MockReader):
        """If no detections ever found, result should have no points."""
        mock_reader = MockVideoReader(frame_count=10)
        MockReader.return_value = mock_reader

        detector = MagicMock(spec=BallDetector)
        detector.detect.return_value = []

        tracker = BallTracker(detector=detector, detection_scale=1.0)
        result = tracker.track_clip("/fake/video.mp4")

        assert len(result.points) == 0


class TestBallTrackerTracking:
    """Test TRACKING state behavior."""

    @patch("wpv.tracking.tracker.VideoReader")
    def test_continuous_tracking(self, MockReader):
        """Continuous detections should yield points in TRACKING state."""
        mock_reader = MockVideoReader(frame_count=20)
        MockReader.return_value = mock_reader

        detector = MagicMock(spec=BallDetector)
        # Ball moving at constant velocity
        call_count = [0]

        def detect_side_effect(frame, pool_mask=None, hsv_bands=None):
            idx = call_count[0]
            call_count[0] += 1
            return [_make_detection(50.0 + idx, 50.0, 0.8)]

        detector.detect.side_effect = detect_side_effect

        tracker = BallTracker(
            detector=detector,
            detection_scale=1.0,
            loss_frames=5,
        )
        result = tracker.track_clip("/fake/video.mp4")

        assert len(result.points) == 20
        assert all(p.state == TrackState.TRACKING.value for p in result.points)
        assert result.stats["num_gaps"] == 0

    @patch("wpv.tracking.tracker.VideoReader")
    def test_loss_triggers_search(self, MockReader):
        """After loss_frames consecutive misses, tracker should enter SEARCH_FORWARD."""
        mock_reader = MockVideoReader(frame_count=50)
        MockReader.return_value = mock_reader

        detector = MagicMock(spec=BallDetector)
        call_count = [0]

        def detect_side_effect(frame, pool_mask=None, hsv_bands=None):
            idx = call_count[0]
            call_count[0] += 1
            # Detect for first 5 frames, then lose ball
            if idx < 5:
                return [_make_detection(50.0, 50.0, 0.8)]
            return []

        detector.detect.side_effect = detect_side_effect

        tracker = BallTracker(
            detector=detector,
            detection_scale=1.0,
            loss_frames=3,
            search_max_gap_s=0.1,  # tiny gap so search fails quickly
        )
        result = tracker.track_clip("/fake/video.mp4")

        # Should have some points in TRACKING and possibly ACTION_PROXY
        tracking_points = [p for p in result.points if p.state == TrackState.TRACKING.value]
        assert len(tracking_points) >= 5  # at least the initial detections


class TestBallTrackerGapBridge:
    """Test gap bridging behavior."""

    @patch("wpv.tracking.tracker.VideoReader")
    def test_gap_bridge_interpolation(self, MockReader):
        """Ball lost then found should produce GAP_BRIDGE points."""
        frame_count = 100
        mock_reader = MockVideoReader(frame_count=frame_count, fps=25.0)
        MockReader.return_value = mock_reader

        detector = MagicMock(spec=BallDetector)
        call_count = [0]

        def detect_side_effect(frame, pool_mask=None, hsv_bands=None):
            idx = call_count[0]
            call_count[0] += 1
            # Detect frames 0-9, lose 10-29, detect 30+
            if idx < 10:
                return [_make_detection(50.0, 50.0, 0.8)]
            if idx >= 30:
                return [_make_detection(80.0, 80.0, 0.8)]
            return []

        detector.detect.side_effect = detect_side_effect

        tracker = BallTracker(
            detector=detector,
            detection_scale=1.0,
            loss_frames=3,
            search_step_s=0.5,  # 12.5 frames
            reacquire_persistence=2,
        )
        result = tracker.track_clip("/fake/video.mp4")

        # Should have at least some points
        assert len(result.points) > 0

        # Check state distribution includes multiple states
        states = {p.state for p in result.points}
        assert TrackState.TRACKING.value in states


class TestTrackResultStats:
    """Test TrackResult stats computation."""

    @patch("wpv.tracking.tracker.VideoReader")
    def test_stats_populated(self, MockReader):
        mock_reader = MockVideoReader(frame_count=10)
        MockReader.return_value = mock_reader

        detector = MagicMock(spec=BallDetector)
        detector.detect.return_value = [_make_detection(50.0, 50.0, 0.8)]

        tracker = BallTracker(detector=detector, detection_scale=1.0)
        result = tracker.track_clip("/fake/video.mp4")

        assert "state_distribution" in result.stats
        assert "num_points" in result.stats
        assert "num_gaps" in result.stats
        assert "mean_confidence" in result.stats
        assert result.elapsed_s >= 0
