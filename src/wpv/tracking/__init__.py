"""Ball tracking: HSV candidate generation + CNN verification + Kalman filter."""

from wpv.tracking.detector import (
    BallClassifier,
    BallDetector,
    Candidate,
    Detection,
    HSV_BANDS,
    detect_hsv_candidates,
)

__all__ = [
    "BallClassifier",
    "BallDetector",
    "Candidate",
    "Detection",
    "HSV_BANDS",
    "detect_hsv_candidates",
]
