"""Tests for ball detection (HSV candidates + BallDetector)."""

import math

import cv2
import numpy as np
import pytest

from wpv.tracking.detector import (
    BallDetector,
    Candidate,
    Detection,
    detect_hsv_candidates,
)


def _make_frame(width: int = 200, height: int = 200, bg_color_bgr=(200, 50, 50)):
    """Create a solid-color BGR frame."""
    frame = np.full((height, width, 3), bg_color_bgr, dtype=np.uint8)
    return frame


def _draw_yellow_circle(frame, center=(100, 100), radius=15):
    """Draw a filled yellow circle (BGR) on the frame."""
    # Yellow in BGR = (0, 255, 255)
    cv2.circle(frame, center, radius, (0, 255, 255), -1)
    return frame


class TestDetectHSVCandidates:
    def test_finds_yellow_circle(self):
        """A bright yellow circle on a blue background should yield 1 candidate."""
        frame = _make_frame(bg_color_bgr=(200, 50, 50))  # blue-ish
        frame = _draw_yellow_circle(frame, center=(100, 100), radius=15)
        candidates = detect_hsv_candidates(frame, min_area=10, max_area=200)
        assert len(candidates) >= 1
        # Check the candidate is near the drawn circle
        c = candidates[0]
        assert abs(c.centroid[0] - 100) < 10
        assert abs(c.centroid[1] - 100) < 10
        assert c.area > 10
        assert c.circularity > 0.7

    def test_no_yellow_returns_empty(self):
        """A frame with no yellow should produce no candidates."""
        frame = _make_frame(bg_color_bgr=(200, 50, 50))  # solid blue
        candidates = detect_hsv_candidates(frame, min_area=4, max_area=200)
        assert candidates == []

    def test_circularity_rejects_elongated(self):
        """An elongated yellow rectangle should be rejected by circularity filter."""
        frame = _make_frame(bg_color_bgr=(200, 50, 50))
        # Draw a thin yellow rectangle (very low circularity)
        cv2.rectangle(frame, (50, 95), (150, 105), (0, 255, 255), -1)
        candidates = detect_hsv_candidates(frame, min_area=10, max_area=200, min_circularity=0.7)
        assert candidates == []

    def test_area_filter_rejects_tiny(self):
        """A very small blob below min_area should be rejected."""
        frame = _make_frame(bg_color_bgr=(200, 50, 50))
        # Draw a tiny yellow dot (radius=1 → area ~3px)
        cv2.circle(frame, (100, 100), 1, (0, 255, 255), -1)
        candidates = detect_hsv_candidates(frame, min_area=50, max_area=200)
        assert candidates == []

    def test_area_filter_rejects_large(self):
        """A blob exceeding max_area diameter should be rejected."""
        frame = _make_frame(width=400, height=400, bg_color_bgr=(200, 50, 50))
        # Draw a huge yellow circle (radius=100 → diameter=200)
        cv2.circle(frame, (200, 200), 100, (0, 255, 255), -1)
        candidates = detect_hsv_candidates(frame, min_area=4, max_area=20)
        assert candidates == []

    def test_multiple_blobs(self):
        """Two separate yellow circles should yield 2 candidates."""
        frame = _make_frame(width=400, height=200, bg_color_bgr=(200, 50, 50))
        _draw_yellow_circle(frame, center=(80, 100), radius=12)
        _draw_yellow_circle(frame, center=(320, 100), radius=12)
        candidates = detect_hsv_candidates(frame, min_area=10, max_area=200)
        assert len(candidates) == 2


class TestBallDetector:
    def test_passthrough_mode(self):
        """Without a model file, BallDetector returns HSV candidates with confidence=0.5."""
        frame = _make_frame(bg_color_bgr=(200, 50, 50))
        _draw_yellow_circle(frame, center=(100, 100), radius=15)

        detector = BallDetector(model_path=None, min_area=10, max_area=200)
        detections = detector.detect(frame)
        assert len(detections) >= 1
        for det in detections:
            assert isinstance(det, Detection)
            assert det.confidence == 0.5

    def test_no_detections_on_empty_frame(self):
        """No yellow blobs → no detections."""
        frame = _make_frame(bg_color_bgr=(200, 50, 50))
        detector = BallDetector(model_path=None, min_area=4, max_area=200)
        detections = detector.detect(frame)
        assert detections == []

    def test_nonexistent_model_path_uses_passthrough(self):
        """A model path that doesn't exist should fall back to passthrough."""
        frame = _make_frame(bg_color_bgr=(200, 50, 50))
        _draw_yellow_circle(frame, center=(100, 100), radius=15)

        detector = BallDetector(model_path="/tmp/nonexistent_model.pth", min_area=10, max_area=200)
        detections = detector.detect(frame)
        assert len(detections) >= 1
        assert all(d.confidence == 0.5 for d in detections)
