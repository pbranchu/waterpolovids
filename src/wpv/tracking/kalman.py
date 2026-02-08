"""Kalman filter wrapper for ball tracking: state [x, y, vx, vy]."""

from __future__ import annotations

import cv2
import numpy as np


class BallKalmanFilter:
    """Kalman filter for ball position tracking.

    State vector: [x, y, vx, vy]
    Measurement vector: [x, y]

    Uses high process noise (ball accelerates during shots/passes)
    and measurement noise scaled by 1/confidence.
    """

    def __init__(self, fps: float = 25.0):
        self._fps = fps
        dt = 1.0 / fps

        self._kf = cv2.KalmanFilter(4, 2, 0)

        # Transition matrix: constant velocity model
        # [x]   [1 0 dt  0] [x]
        # [y] = [0 1  0 dt] [y]
        # [vx]  [0 0  1  0] [vx]
        # [vy]  [0 0  0  1] [vy]
        self._kf.transitionMatrix = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Measurement matrix: observe x, y only
        self._kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )

        # Process noise: high — ball can accelerate rapidly (shots, passes)
        # Use discrete noise model: Q = G * G^T * sigma_a^2
        # where G = [0.5*dt^2, 0.5*dt^2, dt, dt]^T and sigma_a ~ 500 px/s^2
        sigma_a = 500.0  # acceleration noise in pixels/s^2
        g = np.array([[0.5 * dt**2], [0.5 * dt**2], [dt], [dt]], dtype=np.float32)
        self._kf.processNoiseCov = (g @ g.T) * sigma_a**2

        # Default measurement noise (will be scaled by 1/confidence)
        self._base_meas_noise = 10.0  # pixels std dev
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self._base_meas_noise**2

        # Error covariance: start large
        self._kf.errorCovPost = np.eye(4, dtype=np.float32) * 1000.0

        self._initialized = False

    def init_state(self, x: float, y: float) -> None:
        """Initialize filter state with a known position (zero velocity)."""
        self._kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self._kf.errorCovPost = np.eye(4, dtype=np.float32) * 100.0
        self._initialized = True

    def predict(self) -> tuple[float, float, float, float]:
        """Predict next state. Returns (x, y, vx, vy)."""
        state = self._kf.predict()
        return float(state[0, 0]), float(state[1, 0]), float(state[2, 0]), float(state[3, 0])

    def update(
        self, x: float, y: float, confidence: float = 1.0
    ) -> tuple[float, float, float, float]:
        """Update with measurement. Returns corrected (x, y, vx, vy).

        Measurement noise is scaled by 1/confidence — lower confidence
        means the filter trusts the prediction more.
        """
        conf = max(confidence, 0.05)  # clamp to avoid division by zero
        noise_scale = 1.0 / conf
        self._kf.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * (self._base_meas_noise * noise_scale) ** 2
        )

        measurement = np.array([[x], [y]], dtype=np.float32)
        state = self._kf.correct(measurement)
        return float(state[0, 0]), float(state[1, 0]), float(state[2, 0]), float(state[3, 0])

    def mahalanobis_distance(self, x: float, y: float) -> float:
        """Mahalanobis distance of measurement from predicted position.

        Uses the innovation covariance (S = H * P * H^T + R) for gating.
        Lower values mean the measurement is closer to the prediction.
        """
        H = self._kf.measurementMatrix
        P = self._kf.errorCovPre if self._kf.errorCovPre is not None else self._kf.errorCovPost
        R = self._kf.measurementNoiseCov

        # Innovation covariance: S = H * P * H^T + R
        S = H @ P @ H.T + R

        # Innovation (residual)
        predicted = H @ self._kf.statePre if self._kf.statePre is not None else H @ self._kf.statePost
        z = np.array([[x], [y]], dtype=np.float32)
        y_innov = z - predicted

        # Mahalanobis: sqrt(y^T * S^-1 * y)
        S_inv = np.linalg.inv(S)
        d2 = float((y_innov.T @ S_inv @ y_innov)[0, 0])
        return float(np.sqrt(max(d2, 0.0)))

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def state(self) -> tuple[float, float, float, float]:
        """Current state (x, y, vx, vy)."""
        s = self._kf.statePost
        return float(s[0, 0]), float(s[1, 0]), float(s[2, 0]), float(s[3, 0])
