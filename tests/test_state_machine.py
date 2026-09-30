"""Unit tests for tracking state transitions, acquisition, and reacquisition."""

import pytest
from tracking.state_machine import TrackingStateMachine, TrackingState


def test_state_machine_acquisition_flow():
    sm = TrackingStateMachine(hits_to_track=3, misses_to_reacquire=2, misses_to_failsafe=10)

    # Initial state is SEARCH
    assert sm.state == TrackingState.SEARCH

    # 1 hit -> ACQUIRE
    s1 = sm.update(detection_valid=True, confidence=0.8, frame_id=0, timestamp_s=0.0)
    assert s1 == TrackingState.ACQUIRE

    # 2nd hit -> still ACQUIRE
    s2 = sm.update(detection_valid=True, confidence=0.8, frame_id=1, timestamp_s=0.033)
    assert s2 == TrackingState.ACQUIRE

    # 3rd hit -> TRACK
    s3 = sm.update(detection_valid=True, confidence=0.8, frame_id=2, timestamp_s=0.066)
    assert s3 == TrackingState.TRACK
    assert sm.acquisition_duration is not None


def test_state_machine_loss_and_reacquisition():
    sm = TrackingStateMachine(hits_to_track=3, misses_to_reacquire=2, misses_to_failsafe=5)

    # Reach TRACK
    for i in range(3):
        sm.update(detection_valid=True, confidence=0.9, frame_id=i, timestamp_s=i*0.033)
    assert sm.state == TrackingState.TRACK

    # 1 miss -> PREDICT
    s_miss1 = sm.update(detection_valid=False, confidence=0.0, frame_id=3, timestamp_s=0.099)
    assert s_miss1 == TrackingState.PREDICT

    # 2nd miss -> REACQUIRE
    s_miss2 = sm.update(detection_valid=False, confidence=0.0, frame_id=4, timestamp_s=0.132)
    assert s_miss2 == TrackingState.REACQUIRE

    # Re-acquire target (hits: 1 -> ACQUIRE, 2 -> ACQUIRE, 3 -> TRACK)
    sm.update(detection_valid=True, confidence=0.85, frame_id=5, timestamp_s=0.165)
    sm.update(detection_valid=True, confidence=0.85, frame_id=6, timestamp_s=0.198)
    s_reacq = sm.update(detection_valid=True, confidence=0.85, frame_id=7, timestamp_s=0.231)
    assert s_reacq == TrackingState.TRACK
    assert sm.reacquisition_duration is not None
