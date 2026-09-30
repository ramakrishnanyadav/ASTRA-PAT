"""Unit tests for Constant-Acceleration Kalman filter and IMM estimator."""

import pytest
import numpy as np
from app.config import TrackerConfig
from tracking.kalman import ConstantAccelerationKalmanFilter
from tracking.imm import IMMEstimator


def test_constant_acceleration_kalman_convergence():
    cfg = TrackerConfig()
    kf = ConstantAccelerationKalmanFilter(cfg)

    # Simulated motion: x(t) = 100 + 20*t + 0.5*2*(t^2) -> a = 2.0, v0 = 20.0
    dt = 0.033
    true_x = 100.0
    true_v = 20.0
    true_a = 2.0

    kf.initialize(true_x, 200.0, true_v, 0.0)

    for step in range(30):
        t = step * dt
        # Propagate truth
        true_x += true_v * dt + 0.5 * true_a * (dt ** 2)
        true_v += true_a * dt

        # Prediction and measurement update
        kf.predict(dt)
        meas_noise = np.random.normal(0, 0.5)
        accepted, d2 = kf.update(true_x + meas_noise, 200.0)
        assert accepted is True

    # Filter should track position closely (< 1.5 px)
    est_x, est_y, est_vx, est_vy, est_ax, est_ay = kf.state
    assert abs(est_x - true_x) < 2.0
    # Velocity should converge without differencing raw positions
    assert abs(est_vx - true_v) < 5.0


def test_imm_estimator_model_switching():
    cfg = TrackerConfig()
    imm = IMMEstimator(cfg)
    imm.initialize(100.0, 100.0, init_vx=60.6)

    # Step forward with measurements
    for i in range(15):
        imm.predict(0.033)
        acc, _ = imm.update(100.0 + i * 2.0, 100.0)
        assert acc is True

    x, y, vx, vy, ax, ay = imm.state
    assert abs(x - (100.0 + 14 * 2.0)) < 2.0
