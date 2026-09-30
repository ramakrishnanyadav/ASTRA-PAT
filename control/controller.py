"""Predictive tracking controller (Feed-Forward + Proportional-Derivative feedback)."""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
from app.config import ControlConfig, CameraConfig
from coordinates.transforms import CoordinateTransformer, CameraOptics
from control.rate_limiter import ActuatorRateLimiter, RateCommandTelemetry


class PredictiveTrackingController:
    """Computes camera pan/tilt rate commands using state prediction, feed-forward, and rate limiting."""

    def __init__(
        self,
        ctrl_cfg: ControlConfig,
        cam_cfg: CameraConfig,
        transformer: Optional[CoordinateTransformer] = None,
    ) -> None:
        self.cfg = ctrl_cfg
        self.cam_cfg = cam_cfg
        self.transformer = transformer or CoordinateTransformer(
            optics=CameraOptics(
                sensor_width=cam_cfg.sensor_width,
                sensor_height=cam_cfg.sensor_height,
                fov_pan_deg=cam_cfg.fov_pan_deg,
                fov_tilt_deg=cam_cfg.fov_tilt_deg,
            )
        )
        self.rate_limiter = ActuatorRateLimiter(
            max_pan_rate_dps=cam_cfg.max_pan_rate_dps,
            max_tilt_rate_dps=cam_cfg.max_tilt_rate_dps,
        )

        self._prev_err_pan = 0.0
        self._prev_err_tilt = 0.0

    def compute_control_command(
        self,
        target_x: float,
        target_y: float,
        target_vx: float,
        target_vy: float,
        dt: float,
        is_tracking: bool = True,
    ) -> Tuple[float, float, RateCommandTelemetry]:
        """Calculates commanded pan and tilt rates.
        
        Args:
            target_x, target_y: Target position in sensor frame (px)
            target_vx, target_vy: Target velocity in sensor frame (px/s)
            dt: Control update interval (s)
            is_tracking: True if in active closed loop, False if searching
            
        Returns:
            (achieved_pan_rate_dps, achieved_tilt_rate_dps, telemetry)
        """
        if not is_tracking:
            achieved_pan, achieved_tilt, telem = self.rate_limiter.limit(0.0, 0.0)
            return achieved_pan, achieved_tilt, telem

        # 1. Pixel error relative to sensor center
        err_x_px = target_x - self.transformer.optics.center_x
        err_y_px = target_y - self.transformer.optics.center_y

        # 2. Convert pixel error to angular error (deg)
        err_pan_deg, err_tilt_deg = self.transformer.sensor_error_to_angular_error(err_x_px, err_y_px)

        # 3. Feed-forward angular rate from target velocity (deg/s)
        # Target velocity in sensor px/s -> angular rate
        ff_pan_dps = (target_vx / self.transformer.optics.px_per_deg_pan) * self.cfg.feed_forward_weight
        ff_tilt_dps = (-target_vy / self.transformer.optics.px_per_deg_tilt) * self.cfg.feed_forward_weight

        # 4. Derivative of error
        d_err_pan = (err_pan_deg - self._prev_err_pan) / max(dt, 1e-4)
        d_err_tilt = (err_tilt_deg - self._prev_err_tilt) / max(dt, 1e-4)

        # 5. Combined control law: u = FF + Kp * e + Kd * de/dt
        cmd_pan_dps = ff_pan_dps + self.cfg.kp_pan * err_pan_deg + self.cfg.kd_pan * d_err_pan
        cmd_tilt_dps = ff_tilt_dps + self.cfg.kp_tilt * err_tilt_deg + self.cfg.kd_tilt * d_err_tilt

        self._prev_err_pan = err_pan_deg
        self._prev_err_tilt = err_tilt_deg

        # 6. Apply physical slew limits
        achieved_pan, achieved_tilt, telem = self.rate_limiter.limit(cmd_pan_dps, cmd_tilt_dps)
        return achieved_pan, achieved_tilt, telem

    def reset(self) -> None:
        self.rate_limiter.reset()
        self._prev_err_pan = 0.0
        self._prev_err_tilt = 0.0
