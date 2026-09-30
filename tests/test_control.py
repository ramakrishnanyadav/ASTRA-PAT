"""Unit tests for rate limiter and predictive tracking controller."""

import pytest
from app.config import ControlConfig, CameraConfig
from control.rate_limiter import ActuatorRateLimiter
from control.controller import PredictiveTrackingController


def test_rate_limiter_clamping():
    limiter = ActuatorRateLimiter(max_pan_rate_dps=5.0, max_tilt_rate_dps=5.0)

    # 1. Below limit
    pan, tilt, telem = limiter.limit(3.0, -2.5)
    assert pan == 3.0
    assert tilt == -2.5
    assert telem.is_saturated is False

    # 2. Exceeding limit
    pan_sat, tilt_sat, telem_sat = limiter.limit(12.0, -8.0)
    assert pan_sat == 5.0
    assert tilt_sat == -5.0
    assert telem_sat.is_saturated is True
    assert limiter.total_saturation_count == 1


def test_predictive_controller_feed_forward():
    ctrl_cfg = ControlConfig(feed_forward_weight=1.0)
    cam_cfg = CameraConfig(sensor_width=640, sensor_height=480, fov_pan_deg=4.0, fov_tilt_deg=3.0)
    controller = PredictiveTrackingController(ctrl_cfg, cam_cfg)

    # Target centered at (319.5, 239.5) but moving with vx = 160 px/s (1 deg/s)
    pan_rate, tilt_rate, telem = controller.compute_control_command(
        target_x=319.5,
        target_y=239.5,
        target_vx=160.0,
        target_vy=0.0,
        dt=0.033,
        is_tracking=True
    )

    # With zero position error and vx=160 px/s (1 deg/s), feed-forward should command ~1.0 deg/s pan
    assert pytest.approx(pan_rate, abs=0.1) == 1.0
    assert abs(tilt_rate) < 0.1
