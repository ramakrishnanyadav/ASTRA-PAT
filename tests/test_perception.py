"""Unit tests for perception: sub-pixel centroiding, candidate validation, 10% noise rejection, and speed."""

import time
import pytest
import numpy as np
from app.config import PerceptionConfig
from perception.detector import ClassicalDetector
from perception.candidate_filter import CandidateValidator, CandidateBlob
from perception.learned_scorer import LearnedConfidenceScorer


def test_subpixel_centroid_accuracy():
    cfg = PerceptionConfig()
    detector = ClassicalDetector(cfg, target_expected_size=10.0)

    # Synthetic image 480x640 with known square beacon at (320.0, 240.0)
    frame = np.full((480, 640), 15, dtype=np.uint8)
    cx_true = 320.0
    cy_true = 240.0
    size = 10
    pad = size // 2
    frame[int(cy_true)-pad : int(cy_true)+pad+1, int(cx_true)-pad : int(cx_true)+pad+1] = 240

    result = detector.detect(frame, timestamp_s=0.0)
    assert result.valid is True
    assert result.confidence > 0.7
    # Centroid accuracy should be within 0.1 px
    assert abs(result.x - cx_true) < 0.1
    assert abs(result.y - cy_true) < 0.1


def test_rejection_of_ten_percent_salt_and_pepper_no_false_lock():
    cfg = PerceptionConfig()
    detector = ClassicalDetector(cfg, target_expected_size=10.0)

    # Frame with 10% salt-and-pepper noise and NO target
    rng = np.random.default_rng(42)
    frame = np.full((480, 640), 15, dtype=np.uint8)
    num_sp = int(frame.size * 0.10)
    indices = rng.choice(frame.size, size=num_sp, replace=False)
    frame.ravel()[indices[:num_sp//2]] = 0
    frame.ravel()[indices[num_sp//2:]] = 255

    result = detector.detect(frame, timestamp_s=0.0)
    # A single bright pixel or isolated salt pixels must NEVER become a valid detection
    assert result.valid is False


def test_detector_with_roi():
    cfg = PerceptionConfig()
    detector = ClassicalDetector(cfg, target_expected_size=10.0)

    frame = np.full((480, 640), 15, dtype=np.uint8)
    cx_true = 450.0
    cy_true = 350.0
    frame[int(cy_true)-5 : int(cy_true)+6, int(cx_true)-5 : int(cx_true)+6] = 240

    # Narrow ROI around (450, 350)
    roi = (400, 300, 100, 100)
    result = detector.detect(frame, timestamp_s=0.0, roi=roi)

    assert result.valid is True
    assert abs(result.x - cx_true) < 0.1
    assert abs(result.y - cy_true) < 0.1
    assert result.roi_used == roi


def test_detector_processing_speed():
    cfg = PerceptionConfig()
    detector = ClassicalDetector(cfg, target_expected_size=10.0)

    frame = np.full((480, 640), 15, dtype=np.uint8)
    frame[235:246, 315:326] = 240

    # Warmup
    detector.detect(frame, timestamp_s=0.0)

    # Measure 50 full-frame iterations
    times = []
    for i in range(50):
        t0 = time.perf_counter()
        detector.detect(frame, timestamp_s=float(i))
        times.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = np.mean(times)
    fps = 1000.0 / mean_ms
    print(f"\nMeasured Full-frame Detector: {mean_ms:.2f} ms ({fps:.1f} FPS)")

    # SPEC requires >=20 FPS (<=50 ms)
    # TARGET margin is <35 ms
    assert mean_ms < 35.0, f"Detector must run under 35 ms, measured {mean_ms:.2f} ms"


def test_learned_scorer_inference():
    scorer = LearnedConfidenceScorer(target_expected_size=10.0)
    valid_blob = CandidateBlob(
        bbox_x=100, bbox_y=100, bbox_w=10, bbox_h=10,
        area_px=100.0, peak_val=240.0, peak_x=105, peak_y=105,
        center_x=104.5, center_y=104.5,
        mean_val=220.0, aspect_ratio=1.0, snr=15.0
    )
    score = scorer.score(valid_blob)
    assert 0.0 <= score <= 1.0
    assert score > 0.5
