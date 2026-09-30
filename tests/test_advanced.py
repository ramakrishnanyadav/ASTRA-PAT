"""Unit tests for advanced modules: EgoMotion, FlightRecorder, and Ablation runner."""

import os
import shutil
import tempfile
import numpy as np
import pytest

from perception.ego_motion import EgoMotionEstimator
from benchmark.flight_recorder import FlightRecorder
from benchmark.ablation import AblationStudyRunner


def test_ego_motion_featureless_fallback():
    """Verifies honest fallback when scene has no background texture."""
    estimator = EgoMotionEstimator(min_structure_std=3.0)
    
    # Pure dark background with a lone beacon
    f1 = np.full((480, 640), 15, dtype=np.uint8)
    f1[235:245, 315:325] = 240
    
    f2 = np.full((480, 640), 15, dtype=np.uint8)
    f2[238:248, 318:328] = 240
    
    # Feed frame 1
    res1 = estimator.estimate(f1, beacon_xy=(320.0, 240.0))
    assert res1.valid is False
    
    # Feed frame 2
    res2 = estimator.estimate(f2, beacon_xy=(323.0, 243.0))
    # Must reject and honestly report absence of background structure
    assert res2.has_background_structure is False
    assert res2.valid is False
    assert res2.dx_px == 0.0
    assert res2.dy_px == 0.0


def test_ego_motion_with_background_texture():
    """Verifies phase correlation locks when background structure is present."""
    estimator = EgoMotionEstimator(min_structure_std=3.0, min_correlation_peak=0.1)
    
    rng = np.random.default_rng(42)
    # Background texture (e.g. starfield / lunar terrain)
    bg = rng.integers(10, 80, size=(480, 640), dtype=np.uint8)
    
    # Shift background by dx=4, dy=3
    dx_true, dy_true = 4, 3
    shifted_bg = np.roll(bg, (dy_true, dx_true), axis=(0, 1))
    
    res1 = estimator.estimate(bg)
    assert res1.valid is False  # Initial reference frame
    
    res2 = estimator.estimate(shifted_bg)
    assert res2.has_background_structure is True
    assert res2.valid is True
    assert abs(res2.dx_px - dx_true) < 0.5
    assert abs(res2.dy_px - dy_true) < 0.5


def test_flight_recorder_nonblocking_io():
    """Verifies that FlightRecorder buffers and asynchronously writes telemetry."""
    tmp_dir = tempfile.mkdtemp(prefix="flight_rec_test_")
    try:
        recorder = FlightRecorder(output_dir=tmp_dir, queue_size=100)
        recorder.start()
        
        # Enqueue 50 records
        for i in range(50):
            recorder.record_centroid({
                "frame_id": i,
                "timestamp_s": i * 0.033,
                "estimated_x": 320.0 + i,
                "estimated_y": 240.0,
                "confidence": 0.95,
                "area": 100.0,
                "peak": 240.0,
                "processing_ms": 1.2,
                "is_valid": 1,
            })
            recorder.record_tracking({
                "frame_id": i,
                "timestamp_s": i * 0.033,
                "state": "TRACK",
                "est_x": 320.0 + i,
                "est_y": 240.0,
                "est_vx": 30.0,
                "est_vy": 0.0,
                "est_ax": 0.0,
                "est_ay": 0.0,
                "gt_x": 320.0 + i,
                "gt_y": 240.0,
                "error_px": 0.05,
                "camera_pan_deg": 0.0,
                "camera_tilt_deg": 0.0,
                "cmd_pan_dps": 0.0,
                "cmd_tilt_dps": 0.0,
                "slew_saturated": 0,
            })
        
        recorder.flush_and_close()
        
        c_path = os.path.join(tmp_dir, "centroid.csv")
        t_path = os.path.join(tmp_dir, "tracking.csv")
        
        assert os.path.exists(c_path)
        assert os.path.exists(t_path)
        assert os.path.getsize(c_path) > 500
        assert os.path.getsize(t_path) > 500
        assert recorder.records_dropped == 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_ablation_study_execution():
    """Verifies that AblationStudyRunner executes all 4 configurations without error."""
    tmp_dir = tempfile.mkdtemp(prefix="ablation_test_")
    try:
        runner = AblationStudyRunner(output_dir=tmp_dir)
        results = runner.run_all(duration_s=1.0)
        
        assert len(results) == 4
        config_ids = [r.config_id for r in results]
        assert "Config-A" in config_ids
        assert "Config-B" in config_ids
        assert "Config-C" in config_ids
        assert "Config-D" in config_ids
        
        for r in results:
            assert r.mean_fps >= 20.0
            assert r.error_rmse_px >= 0.0
            assert 0.0 <= r.lock_retention_pct <= 100.0
            
        assert os.path.exists(os.path.join(tmp_dir, "ablation_summary.json"))
        assert os.path.exists(os.path.join(tmp_dir, "ablation_report.md"))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
