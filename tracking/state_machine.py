"""Tracking state machine with explicit logged transitions."""

from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import List, Optional, Tuple, Dict, Any


class TrackingState(Enum):
    SEARCH = "SEARCH"
    ACQUIRE = "ACQUIRE"
    TRACK = "TRACK"
    DEGRADED = "DEGRADED"
    PREDICT = "PREDICT"
    REACQUIRE = "REACQUIRE"
    FAILSAFE = "FAILSAFE"


@dataclass
class StateTransitionRecord:
    """Record of a discrete state machine transition."""
    frame_id: int
    timestamp_s: float
    from_state: str
    to_state: str
    reason: str


class TrackingStateMachine:
    """Finite State Machine controlling coarse-PAT tracking states and acquisition timing."""

    def __init__(
        self,
        hits_to_track: int = 3,
        misses_to_reacquire: int = 2,
        misses_to_failsafe: int = 25,
        degraded_confidence_threshold: float = 0.45,
    ) -> None:
        self.hits_to_track = hits_to_track
        self.misses_to_reacquire = misses_to_reacquire
        self.misses_to_failsafe = misses_to_failsafe
        self.degraded_conf_threshold = degraded_confidence_threshold

        self.state: TrackingState = TrackingState.SEARCH
        self.consecutive_hits: int = 0
        self.consecutive_misses: int = 0
        self.total_frames_in_state: int = 0

        # Timing bookmarks
        self.acquisition_start_time: Optional[float] = None
        self.acquisition_duration: Optional[float] = None
        self.loss_start_time: Optional[float] = None
        self.reacquisition_duration: Optional[float] = None

        self.transitions: List[StateTransitionRecord] = []

    def _transition(self, new_state: TrackingState, frame_id: int, timestamp_s: float, reason: str) -> None:
        if new_state == self.state:
            return

        rec = StateTransitionRecord(
            frame_id=frame_id,
            timestamp_s=timestamp_s,
            from_state=self.state.value,
            to_state=new_state.value,
            reason=reason,
        )
        self.transitions.append(rec)
        self.state = new_state
        self.total_frames_in_state = 0

    def update(
        self,
        detection_valid: bool,
        confidence: float,
        frame_id: int,
        timestamp_s: float,
    ) -> TrackingState:
        """Evaluates state transitions based on detection validity and confidence."""
        self.total_frames_in_state += 1

        if detection_valid:
            self.consecutive_hits += 1
            self.consecutive_misses = 0

            # Timing: if we were recovering from target loss
            if self.loss_start_time is not None and self.state in (TrackingState.PREDICT, TrackingState.REACQUIRE):
                self.reacquisition_duration = timestamp_s - self.loss_start_time
                self.loss_start_time = None

            if self.state == TrackingState.SEARCH:
                self.acquisition_start_time = timestamp_s
                self._transition(
                    TrackingState.ACQUIRE, frame_id, timestamp_s,
                    f"Candidate detected (hit 1/{self.hits_to_track})"
                )

            elif self.state == TrackingState.ACQUIRE:
                if self.consecutive_hits >= self.hits_to_track:
                    if self.acquisition_start_time is not None:
                        self.acquisition_duration = timestamp_s - self.acquisition_start_time
                    self._transition(
                        TrackingState.TRACK, frame_id, timestamp_s,
                        f"Acquisition confirmed ({self.consecutive_hits} consecutive hits)"
                    )

            elif self.state == TrackingState.TRACK:
                if confidence < self.degraded_conf_threshold:
                    self._transition(
                        TrackingState.DEGRADED, frame_id, timestamp_s,
                        f"Confidence dropped below threshold ({confidence:.2f} < {self.degraded_conf_threshold})"
                    )

            elif self.state == TrackingState.DEGRADED:
                if confidence >= self.degraded_conf_threshold:
                    self._transition(
                        TrackingState.TRACK, frame_id, timestamp_s,
                        f"Confidence restored ({confidence:.2f} >= {self.degraded_conf_threshold})"
                    )

            elif self.state in (TrackingState.PREDICT, TrackingState.REACQUIRE):
                if self.consecutive_hits >= 2:
                    self._transition(
                        TrackingState.TRACK, frame_id, timestamp_s,
                        "Target reacquired with confirmation"
                    )
                else:
                    self._transition(
                        TrackingState.ACQUIRE, frame_id, timestamp_s,
                        "Candidate detected during reacquisition"
                    )

            elif self.state == TrackingState.FAILSAFE:
                self._transition(
                    TrackingState.ACQUIRE, frame_id, timestamp_s,
                    "Target detected after failsafe reset"
                )

        else:
            # Detection missed
            self.consecutive_hits = 0
            self.consecutive_misses += 1

            if self.loss_start_time is None and self.state in (TrackingState.TRACK, TrackingState.DEGRADED):
                self.loss_start_time = timestamp_s

            if self.state in (TrackingState.TRACK, TrackingState.DEGRADED):
                self._transition(
                    TrackingState.PREDICT, frame_id, timestamp_s,
                    "Missed measurement; entering pure kinematic prediction"
                )

            elif self.state == TrackingState.PREDICT:
                if self.consecutive_misses >= self.misses_to_reacquire:
                    self._transition(
                        TrackingState.REACQUIRE, frame_id, timestamp_s,
                        f"Persistent miss ({self.consecutive_misses} frames); expanding search window"
                    )

            elif self.state == TrackingState.REACQUIRE:
                if self.consecutive_misses >= self.misses_to_failsafe:
                    self._transition(
                        TrackingState.FAILSAFE, frame_id, timestamp_s,
                        f"Target lost past reacquisition timeout ({self.consecutive_misses} frames)"
                    )

            elif self.state == TrackingState.ACQUIRE:
                if self.consecutive_misses >= 2:
                    self._transition(
                        TrackingState.SEARCH, frame_id, timestamp_s,
                        "Candidate failed persistence confirmation"
                    )

            elif self.state == TrackingState.FAILSAFE:
                if self.total_frames_in_state > 5:
                    self._transition(
                        TrackingState.SEARCH, frame_id, timestamp_s,
                        "Failsafe cycle complete; resetting to full-frame search"
                    )

        return self.state

    def reset(self) -> None:
        self.state = TrackingState.SEARCH
        self.consecutive_hits = 0
        self.consecutive_misses = 0
        self.total_frames_in_state = 0
        self.acquisition_start_time = None
        self.acquisition_duration = None
        self.loss_start_time = None
        self.reacquisition_duration = None
        self.transitions.clear()
