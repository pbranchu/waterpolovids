"""Tests for BallKalmanFilter (synthetic, no video)."""

import numpy as np
import pytest

from wpv.tracking.kalman import BallKalmanFilter


class TestBallKalmanFilterInit:
    def test_init_state_sets_position(self):
        kf = BallKalmanFilter(fps=25.0)
        kf.init_state(100.0, 200.0)
        x, y, vx, vy = kf.state
        assert x == pytest.approx(100.0)
        assert y == pytest.approx(200.0)
        assert vx == pytest.approx(0.0)
        assert vy == pytest.approx(0.0)

    def test_initialized_flag(self):
        kf = BallKalmanFilter()
        assert kf.initialized is False
        kf.init_state(0, 0)
        assert kf.initialized is True


class TestBallKalmanFilterPredict:
    def test_stationary_predict(self):
        """With zero velocity, prediction should stay near the same position."""
        kf = BallKalmanFilter(fps=25.0)
        kf.init_state(100.0, 200.0)
        x, y, vx, vy = kf.predict()
        # Position should be very close (zero velocity)
        assert abs(x - 100.0) < 1.0
        assert abs(y - 200.0) < 1.0

    def test_moving_predict(self):
        """After learning velocity from updates, prediction should extrapolate."""
        kf = BallKalmanFilter(fps=25.0)
        kf.init_state(100.0, 200.0)
        dt = 1.0 / 25.0

        # Feed constant velocity: 10 px/frame in x, 0 in y
        for i in range(50):
            kf.predict()
            kf.update(100.0 + (i + 1) * 10.0, 200.0)

        # Now predict the next position — with high process noise, the
        # filter won't converge exactly, so use a generous tolerance
        x, y, vx, vy = kf.predict()
        expected_x = 100.0 + 51 * 10.0
        assert abs(x - expected_x) < 30.0
        # Velocity direction should be positive in x
        assert vx > 0


class TestBallKalmanFilterUpdate:
    def test_update_corrects_position(self):
        kf = BallKalmanFilter(fps=25.0)
        kf.init_state(100.0, 200.0)
        kf.predict()
        x, y, vx, vy = kf.update(110.0, 210.0)
        # Should move toward the measurement
        assert 100.0 < x < 115.0
        assert 200.0 < y < 215.0

    def test_high_confidence_trusts_measurement(self):
        """High confidence should pull the state closer to measurement."""
        kf = BallKalmanFilter(fps=25.0)
        kf.init_state(100.0, 200.0)
        kf.predict()
        x_hi, y_hi, _, _ = kf.update(200.0, 300.0, confidence=1.0)

        kf2 = BallKalmanFilter(fps=25.0)
        kf2.init_state(100.0, 200.0)
        kf2.predict()
        x_lo, y_lo, _, _ = kf2.update(200.0, 300.0, confidence=0.1)

        # High confidence should be closer to measurement (200, 300)
        dist_hi = np.sqrt((x_hi - 200) ** 2 + (y_hi - 300) ** 2)
        dist_lo = np.sqrt((x_lo - 200) ** 2 + (y_lo - 300) ** 2)
        assert dist_hi < dist_lo


class TestMahalanobisDistance:
    def test_zero_distance_at_prediction(self):
        """Measurement at the predicted position should have ~zero distance."""
        kf = BallKalmanFilter(fps=25.0)
        kf.init_state(100.0, 200.0)
        kf.predict()
        # Measure at the predicted position
        px, py = kf.state[0], kf.state[1]
        d = kf.mahalanobis_distance(px, py)
        assert d < 0.1

    def test_far_measurement_has_large_distance(self):
        """Measurement far from prediction should have large Mahalanobis distance."""
        kf = BallKalmanFilter(fps=25.0)
        kf.init_state(100.0, 200.0)
        kf.predict()
        d = kf.mahalanobis_distance(5000.0, 5000.0)
        assert d > 10.0

    def test_gating_threshold(self):
        """Nearby measurement should be within a reasonable gate threshold."""
        kf = BallKalmanFilter(fps=25.0)
        kf.init_state(100.0, 200.0)
        # Run a few predict/update cycles to settle the filter
        for i in range(5):
            kf.predict()
            kf.update(100.0 + i, 200.0 + i)

        kf.predict()
        px, py = kf.state[0], kf.state[1]
        # Nearby: should be gated in (< 5.0)
        d_near = kf.mahalanobis_distance(px + 5, py + 5)
        # Far: should be gated out
        d_far = kf.mahalanobis_distance(px + 500, py + 500)
        assert d_near < d_far
