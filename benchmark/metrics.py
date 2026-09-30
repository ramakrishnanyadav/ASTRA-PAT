"""Performance metrics computation and mission-readiness verification."""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from app.config import AcceptanceThresholds


@dataclass
class MetricThresholdStatus:
    name: str
    spec_threshold: str
    target_threshold: str
    measured_val: str
    status: str  # PASS / FAIL
    target_met: bool


@dataclass
class PerformanceSummary:
    """Comprehensive performance report statistics."""
    total_frames: int
    eligible_frames: int
    locked_frames: int
    lost_frames: int
    lock_retention_pct: float
    target_loss_pct: float
    
    # Centroid errors (px)
    error_mean_px: float
    error_rmse_px: float
    error_median_px: float
    error_p95_px: float
    error_max_px: float
    
    # Timing (s)
    acquisition_time_s: Optional[float]
    reacquisition_time_s: Optional[float]
    
    # Throughput
    mean_processing_ms: float
    p95_processing_ms: float
    mean_fps: float
    
    # Actuator
    slew_limit_saturations: int
    
    # Mission readiness
    checklist: List[MetricThresholdStatus]
    all_spec_passed: bool

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["checklist"] = [asdict(item) for item in self.checklist]
        return d


class MetricsEngine:
    """Computes, logs, and evaluates tracking benchmarks against spec and engineering targets."""

    def __init__(self, thresholds: Optional[AcceptanceThresholds] = None) -> None:
        self.thresholds = thresholds or AcceptanceThresholds()

        self.errors_px: List[float] = []
        self.processing_times_ms: List[float] = []
        self.total_frames: int = 0
        self.eligible_frames: int = 0
        self.locked_frames: int = 0
        self.lost_frames: int = 0
        self.slew_saturations: int = 0

        self.acquisition_time_s: Optional[float] = None
        self.reacquisition_time_s: Optional[float] = None

    def record_frame(
        self,
        estimated_xy: Tuple[float, float],
        ground_truth_xy: Optional[Tuple[float, float]],
        target_visible: bool,
        is_locked: bool,
        processing_ms: float,
        slew_saturated: bool = False,
    ) -> Optional[float]:
        """Records per-frame metrics. Returns current frame error in px (or None)."""
        self.total_frames += 1
        self.processing_times_ms.append(processing_ms)

        if slew_saturated:
            self.slew_saturations += 1

        err = None
        if target_visible:
            self.eligible_frames += 1
            if is_locked:
                self.locked_frames += 1
            else:
                self.lost_frames += 1

            if ground_truth_xy is not None:
                err = float(np.hypot(estimated_xy[0] - ground_truth_xy[0], estimated_xy[1] - ground_truth_xy[1]))
                self.errors_px.append(err)

        return err

    def compute_summary(self) -> PerformanceSummary:
        """Computes statistical summary and verifies against SPEC and TARGET thresholds."""
        eligible = max(self.eligible_frames, 1)
        lock_retention = (self.locked_frames / eligible) * 100.0
        target_loss = (self.lost_frames / eligible) * 100.0

        if self.errors_px:
            err_arr = np.array(self.errors_px)
            err_mean = float(np.mean(err_arr))
            err_rmse = float(np.sqrt(np.mean(err_arr ** 2)))
            err_median = float(np.median(err_arr))
            err_p95 = float(np.percentile(err_arr, 95))
            err_max = float(np.max(err_arr))
        else:
            err_mean = err_rmse = err_median = err_p95 = err_max = 0.0

        if self.processing_times_ms:
            t_arr = np.array(self.processing_times_ms)
            mean_ms = float(np.mean(t_arr))
            p95_ms = float(np.percentile(t_arr, 95))
            mean_fps = float(1000.0 / mean_ms) if mean_ms > 0 else 0.0
        else:
            mean_ms = p95_ms = mean_fps = 0.0

        # Construct Mission-Readiness Checklist
        checklist: List[MetricThresholdStatus] = []
        all_passed = True

        # 1. Centroid Tracking Error
        pass_err = (err_rmse <= self.thresholds.max_tracking_error_px)
        all_passed = all_passed and pass_err
        checklist.append(MetricThresholdStatus(
            name="Tracking Centroid Error (RMSE)",
            spec_threshold=f"<= {self.thresholds.max_tracking_error_px:.1f} px",
            target_threshold="< 3.0 px",
            measured_val=f"{err_rmse:.2f} px",
            status="PASS" if pass_err else "FAIL",
            target_met=(err_rmse < 3.0),
        ))

        # 2. Target Loss Percentage
        pass_loss = (target_loss < self.thresholds.max_target_loss_pct)
        all_passed = all_passed and pass_loss
        checklist.append(MetricThresholdStatus(
            name="Target Loss Rate",
            spec_threshold=f"< {self.thresholds.max_target_loss_pct:.1f}%",
            target_threshold="< 1.0%",
            measured_val=f"{target_loss:.2f}%",
            status="PASS" if pass_loss else "FAIL",
            target_met=(target_loss < 1.0),
        ))

        # 3. Processing Speed / FPS
        pass_fps = (mean_fps >= self.thresholds.min_fps)
        all_passed = all_passed and pass_fps
        checklist.append(MetricThresholdStatus(
            name="Processing Frame Rate",
            spec_threshold=f">= {self.thresholds.min_fps:.1f} FPS (<= 50.0 ms)",
            target_threshold=">= 28.6 FPS (< 35.0 ms)",
            measured_val=f"{mean_fps:.1f} FPS ({mean_ms:.2f} ms)",
            status="PASS" if pass_fps else "FAIL",
            target_met=(mean_fps >= self.thresholds.target_fps_margin),
        ))

        # 4. Acquisition Time
        acq_val = self.acquisition_time_s if self.acquisition_time_s is not None else 0.0
        pass_acq = (acq_val <= self.thresholds.max_acquisition_sec)
        all_passed = all_passed and pass_acq
        checklist.append(MetricThresholdStatus(
            name="Initial Acquisition Time",
            spec_threshold=f"<= {self.thresholds.max_acquisition_sec:.1f} s",
            target_threshold="<= 0.5 s",
            measured_val=f"{acq_val:.3f} s" if self.acquisition_time_s is not None else "N/A",
            status="PASS" if pass_acq else "FAIL",
            target_met=(acq_val <= 0.5),
        ))

        # 5. Re-acquisition Time
        if self.reacquisition_time_s is not None:
            pass_reacq = (self.reacquisition_time_s <= self.thresholds.max_reacquisition_sec)
            all_passed = all_passed and pass_reacq
            checklist.append(MetricThresholdStatus(
                name="Re-acquisition Time",
                spec_threshold=f"<= {self.thresholds.max_reacquisition_sec:.1f} s",
                target_threshold="<= 0.5 s",
                measured_val=f"{self.reacquisition_time_s:.3f} s",
                status="PASS" if pass_reacq else "FAIL",
                target_met=(self.reacquisition_time_s <= 0.5),
            ))

        return PerformanceSummary(
            total_frames=self.total_frames,
            eligible_frames=self.eligible_frames,
            locked_frames=self.locked_frames,
            lost_frames=self.lost_frames,
            lock_retention_pct=lock_retention,
            target_loss_pct=target_loss,
            error_mean_px=err_mean,
            error_rmse_px=err_rmse,
            error_median_px=err_median,
            error_p95_px=err_p95,
            error_max_px=err_max,
            acquisition_time_s=self.acquisition_time_s,
            reacquisition_time_s=self.reacquisition_time_s,
            mean_processing_ms=mean_ms,
            p95_processing_ms=p95_ms,
            mean_fps=mean_fps,
            slew_limit_saturations=self.slew_saturations,
            checklist=checklist,
            all_spec_passed=all_passed,
        )
