"""Unit tests for the simulator core, disturbances, and ground truth logging."""

import pytest
import numpy as np
from app.config import AppConfig, TargetConfig, DisturbanceConfig
from simulator.simulation_input import SimulationInput


def test_simulation_stream_and_ground_truth():
    cfg = AppConfig()
    cfg.target.motion_type = "figure8"
    cfg.target.initial_x = 1000.0
    cfg.target.initial_y = 1000.0
    sim = SimulationInput(cfg)

    frame, ts, gt = sim.get_next_frame()
    assert frame is not None
    assert frame.shape == (480, 640)
    assert frame.dtype == np.uint8
    assert ts == 0.0
    assert gt is not None
    assert gt.frame_id == 0
    assert gt.target_visible is True
    # At center (1000, 1000), target should be near sensor center (319.5, 239.5)
    assert abs(gt.target_sensor_x - 319.5) < 2.0
    assert abs(gt.target_sensor_y - 239.5) < 2.0


def test_disturbances_salt_pepper_and_noise():
    cfg = AppConfig()
    cfg.disturbances.salt_pepper_ratio = 0.10
    cfg.disturbances.gaussian_sigma_px = 10.0
    sim = SimulationInput(cfg)

    frame, ts, gt = sim.get_next_frame()
    # Check that salt and pepper altered pixels
    zeros = np.sum(frame == 0)
    fulls = np.sum(frame == 255)
    total = frame.size
    # ~10% should be salt and pepper
    assert (zeros + fulls) / total > 0.05


def test_camera_slew_rate_limit():
    cfg = AppConfig()
    cfg.camera.max_pan_rate_dps = 5.0
    sim = SimulationInput(cfg)

    # Command an excessive pan rate of 25.0 deg/s
    sim.send_control_command(pan_rate_dps=25.0, tilt_rate_dps=0.0)
    _, _, gt = sim.get_next_frame()
    # Slew should saturate at exactly 5.0 deg/s
    assert gt.slew_saturated is True
    assert pytest.approx(gt.camera_pan_rate_dps, 1e-4) == 5.0


def test_target_dropout_reacquisition():
    cfg = AppConfig()
    cfg.target.initial_x = 1000.0
    cfg.target.initial_y = 1000.0
    cfg.target.drop_out_start_s = 0.05
    cfg.target.drop_out_duration_s = 0.10
    sim = SimulationInput(cfg)

    # At t=0.0 -> visible
    _, _, gt0 = sim.get_next_frame()
    assert gt0.target_visible is True

    # Advance until dropout
    frames = [sim.get_next_frame()[2] for _ in range(5)]
    # At least one frame should be invisible
    dropouts = [f for f in frames if not f.target_visible]
    assert len(dropouts) > 0
