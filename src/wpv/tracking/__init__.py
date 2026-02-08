"""Ball tracking: HSV candidate generation + CNN verification + Kalman filter."""

from wpv.tracking.detector import (
    BallClassifier,
    BallDetector,
    Candidate,
    Detection,
    HSV_BANDS,
    detect_hsv_candidates,
)
from wpv.tracking.kalman import BallKalmanFilter
from wpv.tracking.track_io import load_track, save_track
from wpv.tracking.tracker import (
    BallTracker,
    TrackGap,
    TrackPoint,
    TrackResult,
    TrackState,
)
from wpv.tracking.video_reader import VideoReader

__all__ = [
    "BallClassifier",
    "BallDetector",
    "BallKalmanFilter",
    "BallTracker",
    "Candidate",
    "Detection",
    "HSV_BANDS",
    "TrackGap",
    "TrackPoint",
    "TrackResult",
    "TrackState",
    "VideoReader",
    "detect_hsv_candidates",
    "load_track",
    "save_track",
]
