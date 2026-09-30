"""Unit tests for TrackerCore strict API and execution."""

import pytest
import numpy as np
from app.config import AppConfig
from tracking.tracker_core import TrackerCore, TrackerOutput, TrackerSummaryState
from tracking.state_machine import TrackingState


def test_tracker_core_lifecycle_and_contract():
    cfg = AppConfig()
    tracker = TrackerCore(is_passive_mp4_mode=False)
    tracker.initialize(cfg)

    # Frame with target at (320, 240)
    frame = np.full((480, 640), 15, dtype=np.uint8)
    frame[235:246, 315:326] = 240

    out1 = tracker.process_frame(frame, timestamp=0.0)
    assert isinstance(out1, TrackerOutput)
    assert out1.frame_id == 0
    assert out1.detection.valid is True
    assert out1.tracking_state == TrackingState.ACQUIRE

    # Feed consecutive frames to transition to TRACK
    out2 = tracker.process_frame(frame, timestamp=0.033)
    out3 = tracker.process_frame(frame, timestamp=0.066)
    assert out3.tracking_state == TrackingState.TRACK
    assert out3.camera_command is not None

    summary = tracker.get_state()
    assert isinstance(summary, TrackerSummaryState)
    assert summary.locked_frames >= 1
    assert summary.mean_processing_ms > 0.0

    # Reset
    tracker.reset()
    assert tracker.frame_id == 0
    assert tracker.state_machine.state == TrackingState.SEARCH


def test_tracker_core_passive_mp4_mode():
    cfg = AppConfig()
    tracker = TrackerCore(is_passive_mp4_mode=True)
    tracker.initialize(cfg)

    frame = np.full((480, 640), 15, dtype=np.uint8)
    frame[235:246, 315:326] = 240

    out = tracker.process_frame(frame, timestamp=0.0)
    # In MP4 mode, camera command MUST be None
    assert out.camera_command is None
