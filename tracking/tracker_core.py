"""TrackerCore: Strict single-entry-point engine for optical beacon tracking."""

from __future__ import annotations
from dataclasses import dataclass, field
import time
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from app.config import AppConfig
from perception.detector import BaseDetector, ClassicalDetector, DetectionResult
from perception.learned_scorer import LearnedConfidenceScorer
from tracking.kalman import ConstantAccelerationKalmanFilter
from tracking.imm import IMMEstimator
from tracking.state_machine import TrackingStateMachine, TrackingState
from tracking.prediction import AdaptiveROIManager
from control.controller import PredictiveTrackingController, RateCommandTelemetry
from reacquisition.reacquisition_manager import ReacquisitionManager


@dataclass
class TrackerOutput:
    """Standardized output produced per frame by TrackerCore."""
    frame_id: int
    timestamp_s: float
    detection: DetectionResult
    state_estimate: Tuple[float, float, float, float, float, float]  # (x, y, vx, vy, ax, ay)
    tracking_state: TrackingState
    confidence: float
    predicted_position: Tuple[float, float]  # (pred_x, pred_y)
    next_roi: Optional[Tuple[int, int, int, int]]
    camera_command: Optional[Tuple[float, float]]  # (pan_rate_dps, tilt_rate_dps) or None
    rate_telemetry: Optional[RateCommandTelemetry]
    total_processing_ms: float


@dataclass
class TrackerSummaryState:
    """High-level summary state for reporting and telemetry."""
    current_state: TrackingState
    total_frames: int
    locked_frames: int
    lost_frames: int
    reacquisition_events: int
    mean_processing_ms: float
    fps: float


class TrackerCore:
    """Core perception and tracking engine, decoupled from GUI, simulator, and file I/O."""

    def __init__(self, is_passive_mp4_mode: bool = False) -> None:
        self.is_passive_mp4_mode = is_passive_mp4_mode
        self.cfg: Optional[AppConfig] = None

        self.detector: Optional[BaseDetector] = None
        self.estimator: Optional[Any] = None
        self.state_machine: Optional[TrackingStateMachine] = None
        self.roi_manager: Optional[AdaptiveROIManager] = None
        self.reacquisition_mgr: Optional[ReacquisitionManager] = None
        self.controller: Optional[PredictiveTrackingController] = None

        self.frame_id: int = 0
        self.prev_timestamp: Optional[float] = None
        self.next_roi: Optional[Tuple[int, int, int, int]] = None
        self._last_camera_cmd: Optional[Tuple[float, float]] = None

        # Telemetry statistics
        self._processing_times_ms: List[float] = []
        self._locked_frames: int = 0
        self._lost_frames: int = 0
        self._reacquisitions: int = 0

    def initialize(self, config: AppConfig) -> None:
        """Initializes all sub-modules based on unified config."""
        self.cfg = config

        # 1. Detector
        self.detector = ClassicalDetector(
            self.cfg.perception,
            target_expected_size=self.cfg.target.size_px
        )

        # 2. Kinematic Estimator (CA or IMM)
        if self.cfg.tracker.estimation_model == "imm":
            self.estimator = IMMEstimator(self.cfg.tracker)
        else:
            self.estimator = ConstantAccelerationKalmanFilter(self.cfg.tracker)

        # 3. State Machine
        self.state_machine = TrackingStateMachine(
            hits_to_track=self.cfg.tracker.consecutive_hits_to_track,
            misses_to_reacquire=self.cfg.tracker.consecutive_misses_to_lost,
        )

        # 4. Adaptive ROI & Reacquisition
        self.roi_manager = AdaptiveROIManager(
            frame_width=self.cfg.camera.sensor_width,
            frame_height=self.cfg.camera.sensor_height,
            base_margin=self.cfg.tracker.adaptive_roi_base_margin,
            max_margin=self.cfg.tracker.adaptive_roi_max_margin,
        )
        self.reacquisition_mgr = ReacquisitionManager(
            frame_width=self.cfg.camera.sensor_width,
            frame_height=self.cfg.camera.sensor_height,
        )

        # 5. Controller (only active in active/simulation mode)
        self.controller = PredictiveTrackingController(
            self.cfg.control,
            self.cfg.camera
        )

        self.reset()

    def reset(self) -> None:
        """Resets tracking state to initial conditions."""
        self.frame_id = 0
        self.prev_timestamp = None
        self.next_roi = None
        self._last_camera_cmd = None
        self._processing_times_ms.clear()
        self._locked_frames = 0
        self._lost_frames = 0
        self._reacquisitions = 0

        if self.estimator is not None:
            if hasattr(self.estimator, "is_initialized"):
                self.estimator.is_initialized = False
        if self.state_machine is not None:
            self.state_machine.reset()
        if self.reacquisition_mgr is not None:
            self.reacquisition_mgr.reset_lock()
        if self.controller is not None:
            self.controller.reset()

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> TrackerOutput:
        """Processes a single video frame. Strict entry point for both simulation and MP4 modes."""
        t_start = time.perf_counter()

        if self.cfg is None:
            raise RuntimeError("TrackerCore must be initialized with AppConfig prior to processing.")

        # Compute dt
        if self.prev_timestamp is not None:
            dt = max(timestamp - self.prev_timestamp, 1e-4)
        else:
            dt = 1.0 / max(self.cfg.camera.update_hz, 1.0)
        self.prev_timestamp = timestamp

        # Determine active ROI for this frame
        active_roi = roi if roi is not None else self.next_roi

        # Spatial prior from estimator if available
        prior_xy = None
        if self.estimator and getattr(self.estimator, "is_initialized", False):
            prior_xy = (self.estimator.state[0], self.estimator.state[1])

        # 1. Perception Detection
        det = self.detector.detect(
            frame,
            timestamp_s=timestamp,
            roi=active_roi,
            prior_xy=prior_xy
        )

        # 2. Camera gimbal ego-motion compensation for sensor coordinates
        if self._last_camera_cmd is not None and not self.is_passive_mp4_mode:
            k_pan = self.cfg.camera.sensor_width / self.cfg.camera.fov_pan_deg
            k_tilt = self.cfg.camera.sensor_height / self.cfg.camera.fov_tilt_deg
            shift_x = -self._last_camera_cmd[0] * dt * k_pan
            shift_y = self._last_camera_cmd[1] * dt * k_tilt
            # Shift estimator position by camera displacement
            if self.estimator and getattr(self.estimator, "is_initialized", False):
                if hasattr(self.estimator, "x"):
                    self.estimator.x[0, 0] += shift_x
                    self.estimator.x[1, 0] += shift_y
                elif hasattr(self.estimator, "ca"):
                    self.estimator.ca.x[0, 0] += shift_x
                    self.estimator.ca.x[1, 0] += shift_y
                    self.estimator.cv.x[0, 0] += shift_x
                    self.estimator.cv.x[1, 0] += shift_y

        # 3. Kalman / IMM Prediction
        pred_x, pred_y = self.estimator.predict(dt)

        # 4. Kalman / IMM Measurement Update
        meas_accepted = False
        if det.valid:
            meas_accepted, gate_dist = self.estimator.update(det.x, det.y)
            # In established TRACK state, gate out spurious clutter outliers
            if not meas_accepted and self.state_machine.state in (TrackingState.TRACK, TrackingState.DEGRADED):
                det.valid = False

        # 4. State Machine Update
        old_state = self.state_machine.state
        current_state = self.state_machine.update(
            detection_valid=det.valid,
            confidence=det.confidence,
            frame_id=self.frame_id,
            timestamp_s=timestamp
        )

        # Track statistics
        if current_state in (TrackingState.TRACK, TrackingState.DEGRADED):
            self._locked_frames += 1
        elif current_state in (TrackingState.PREDICT, TrackingState.REACQUIRE, TrackingState.FAILSAFE):
            self._lost_frames += 1

        if old_state in (TrackingState.PREDICT, TrackingState.REACQUIRE) and current_state == TrackingState.TRACK:
            self._reacquisitions += 1

        # Current state estimate
        est_state = self.estimator.state

        # 5. Lookahead prediction for NEXT frame
        next_pred_x = est_state[0] + est_state[2] * dt + 0.5 * est_state[4] * (dt ** 2)
        next_pred_y = est_state[1] + est_state[3] * dt + 0.5 * est_state[5] * (dt ** 2)

        # 6. Reacquisition / Next ROI Generation
        pos_std_x, pos_std_y = self.estimator.position_std
        if current_state in (TrackingState.PREDICT, TrackingState.REACQUIRE):
            self.reacquisition_mgr.step_miss()
            self.next_roi = self.reacquisition_mgr.get_search_roi(next_pred_x, next_pred_y)
        elif current_state in (TrackingState.TRACK, TrackingState.DEGRADED):
            self.reacquisition_mgr.reset_lock()
            self.next_roi = self.roi_manager.compute_next_roi(
                current_state, next_pred_x, next_pred_y, pos_std_x, pos_std_y
            )
        else:
            self.next_roi = None

        # 7. Control Command (Bypassed in MP4 mode)
        if not self.is_passive_mp4_mode and self.controller is not None:
            is_active_track = current_state in (TrackingState.TRACK, TrackingState.DEGRADED, TrackingState.PREDICT)
            pan_rate, tilt_rate, telem = self.controller.compute_control_command(
                target_x=est_state[0],
                target_y=est_state[1],
                target_vx=est_state[2],
                target_vy=est_state[3],
                dt=dt,
                is_tracking=is_active_track
            )
            camera_cmd = (pan_rate, tilt_rate)
            rate_telem = telem
        else:
            camera_cmd = None
            rate_telem = None

        self._last_camera_cmd = camera_cmd

        total_ms = (time.perf_counter() - t_start) * 1000.0
        self._processing_times_ms.append(total_ms)

        output = TrackerOutput(
            frame_id=self.frame_id,
            timestamp_s=timestamp,
            detection=det,
            state_estimate=est_state,
            tracking_state=current_state,
            confidence=det.confidence if det.valid else 0.0,
            predicted_position=(next_pred_x, next_pred_y),
            next_roi=self.next_roi,
            camera_command=camera_cmd,
            rate_telemetry=rate_telem,
            total_processing_ms=total_ms,
        )

        self.frame_id += 1
        return output

    def get_state(self) -> TrackerSummaryState:
        """Returns summary state metrics."""
        total = max(self.frame_id, 1)
        mean_ms = float(np.mean(self._processing_times_ms)) if self._processing_times_ms else 0.0
        fps = (1000.0 / mean_ms) if mean_ms > 0 else 0.0

        current_s = self.state_machine.state if self.state_machine else TrackingState.SEARCH

        return TrackerSummaryState(
            current_state=current_s,
            total_frames=self.frame_id,
            locked_frames=self._locked_frames,
            lost_frames=self._lost_frames,
            reacquisition_events=self._reacquisitions,
            mean_processing_ms=mean_ms,
            fps=fps,
        )
