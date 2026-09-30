"""Benchmark execution engine running scenarios end-to-end and saving canonical telemetry."""

from __future__ import annotations
import csv
import json
import os
import time
from typing import Optional, Dict, Any, List
import numpy as np
from app.config import AppConfig
from simulator.simulation_input import SimulationInput, BaseInputAdapter
from benchmark.mp4_input import MP4Input
from tracking.tracker_core import TrackerCore, TrackerOutput
from tracking.state_machine import TrackingState
from benchmark.metrics import MetricsEngine, PerformanceSummary
from benchmark.flight_recorder import FlightRecorder


class BenchmarkEvaluator:
    """Executes a benchmark run, feeds the unified pipeline, and records full telemetry."""

    def __init__(self, config: AppConfig, output_dir: str = "results") -> None:
        self.cfg = config
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.tracker = TrackerCore(is_passive_mp4_mode=False)
        self.metrics = MetricsEngine(self.cfg.acceptance)
        self.flight_recorder: Optional[FlightRecorder] = None

    def run_simulation(self, duration_s: float = 10.0) -> PerformanceSummary:
        """Runs closed-loop simulation benchmark with asynchronous flight recording."""
        self.tracker.initialize(self.cfg)
        sim = SimulationInput(self.cfg)

        self.flight_recorder = FlightRecorder(self.output_dir)
        self.flight_recorder.start()

        max_frames = int(duration_s * self.cfg.camera.update_hz)

        for _ in range(max_frames):
            frame_data = sim.get_next_frame()
            if frame_data is None:
                break
            frame, timestamp, gt = frame_data

            # 1. TrackerCore process frame
            out: TrackerOutput = self.tracker.process_frame(frame, timestamp)

            # 2. Feed camera command back to simulator
            if out.camera_command is not None:
                sim.send_control_command(out.camera_command[0], out.camera_command[1])

            # 3. Ground truth extraction
            gt_xy = (gt.target_sensor_x, gt.target_sensor_y) if (gt and gt.target_visible) else None
            is_locked = (out.tracking_state in (TrackingState.TRACK, TrackingState.DEGRADED))

            # Record metrics
            err = self.metrics.record_frame(
                estimated_xy=(out.state_estimate[0], out.state_estimate[1]),
                ground_truth_xy=gt_xy,
                target_visible=(gt.target_visible if gt else False),
                is_locked=is_locked,
                processing_ms=out.total_processing_ms,
                slew_saturated=(gt.slew_saturated if gt else False),
            )

            # Check acquisition times from tracker state machine
            if self.tracker.state_machine.acquisition_duration is not None:
                self.metrics.acquisition_time_s = self.tracker.state_machine.acquisition_duration
            if self.tracker.state_machine.reacquisition_duration is not None:
                self.metrics.reacquisition_time_s = self.tracker.state_machine.reacquisition_duration

            # Canonical centroid record (asynchronous flight recorder)
            self.flight_recorder.record_centroid({
                "frame_id": out.frame_id,
                "timestamp_s": out.timestamp_s,
                "estimated_x": out.detection.x if out.detection.valid else out.state_estimate[0],
                "estimated_y": out.detection.y if out.detection.valid else out.state_estimate[1],
                "confidence": out.confidence,
                "area": out.detection.area,
                "peak": out.detection.peak,
                "processing_ms": out.total_processing_ms,
                "is_valid": int(out.detection.valid),
            })

            # Full tracking record
            self.flight_recorder.record_tracking({
                "frame_id": out.frame_id,
                "timestamp_s": out.timestamp_s,
                "state": out.tracking_state.value,
                "est_x": out.state_estimate[0],
                "est_y": out.state_estimate[1],
                "est_vx": out.state_estimate[2],
                "est_vy": out.state_estimate[3],
                "est_ax": out.state_estimate[4],
                "est_ay": out.state_estimate[5],
                "gt_x": gt.target_sensor_x if gt else -1.0,
                "gt_y": gt.target_sensor_y if gt else -1.0,
                "error_px": err if err is not None else -1.0,
                "camera_pan_deg": gt.camera_pan_deg if gt else 0.0,
                "camera_tilt_deg": gt.camera_tilt_deg if gt else 0.0,
                "cmd_pan_dps": out.camera_command[0] if out.camera_command else 0.0,
                "cmd_tilt_dps": out.camera_command[1] if out.camera_command else 0.0,
                "slew_saturated": int(gt.slew_saturated if gt else False),
            })

        # Cleanly flush flight recorder queue to disk
        self.flight_recorder.flush_and_close()

        summary = self.metrics.compute_summary()
        self._export_metrics_json(summary)
        return summary

    def run_mp4_benchmark(self, video_path: str) -> PerformanceSummary:
        """Runs Benchmark-2 on an external .mp4 video file, bypassing PTZ."""
        tracker = TrackerCore(is_passive_mp4_mode=True)
        tracker.initialize(self.cfg)
        metrics = MetricsEngine(self.cfg.acceptance)

        flight_rec = FlightRecorder(self.output_dir)
        flight_rec.start()

        reader = MP4Input(video_path)

        while True:
            item = reader.get_next_frame()
            if item is None:
                break
            frame, timestamp, _ = item

            out = tracker.process_frame(frame, timestamp)
            metrics.record_frame(
                estimated_xy=(out.state_estimate[0], out.state_estimate[1]),
                ground_truth_xy=None,
                target_visible=out.detection.valid,
                is_locked=(out.tracking_state == TrackingState.TRACK),
                processing_ms=out.total_processing_ms,
                slew_saturated=False,
            )

            flight_rec.record_centroid({
                "frame_id": out.frame_id,
                "timestamp_s": out.timestamp_s,
                "estimated_x": out.detection.x if out.detection.valid else out.state_estimate[0],
                "estimated_y": out.detection.y if out.detection.valid else out.state_estimate[1],
                "confidence": out.confidence,
                "area": out.detection.area,
                "peak": out.detection.peak,
                "processing_ms": out.total_processing_ms,
                "is_valid": int(out.detection.valid),
            })

        reader.close()
        flight_rec.flush_and_close()

        summary = metrics.compute_summary()
        self._export_metrics_json(summary)
        return summary

    def _export_metrics_json(self, summary: PerformanceSummary) -> None:
        """Exports metrics summary to JSON."""
        m_path = os.path.join(self.output_dir, "metrics.json")
        with open(m_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)

