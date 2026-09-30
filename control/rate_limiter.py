"""Physical gimbal rate limiter and actuator telemetry."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class RateCommandTelemetry:
    """Telemetry record for a single control command."""
    commanded_pan_dps: float
    commanded_tilt_dps: float
    achieved_pan_dps: float
    achieved_tilt_dps: float
    is_saturated: bool


class ActuatorRateLimiter:
    """Enforces slew rate saturation and logs saturation telemetry."""

    def __init__(self, max_pan_rate_dps: float = 5.0, max_tilt_rate_dps: float = 5.0) -> None:
        self.max_pan_rate = float(max_pan_rate_dps)
        self.max_tilt_rate = float(max_tilt_rate_dps)
        self.total_saturation_count: int = 0
        self.total_commands: int = 0

    def limit(self, cmd_pan_dps: float, cmd_tilt_dps: float) -> Tuple[float, float, RateCommandTelemetry]:
        """Clamps commanded rates to allowable physical slew limits."""
        self.total_commands += 1

        sat_pan = abs(cmd_pan_dps) > self.max_pan_rate
        sat_tilt = abs(cmd_tilt_dps) > self.max_tilt_rate
        is_sat = sat_pan or sat_tilt

        if is_sat:
            self.total_saturation_count += 1

        achieved_pan = float(np.clip(cmd_pan_dps, -self.max_pan_rate, self.max_pan_rate))
        achieved_tilt = float(np.clip(cmd_tilt_dps, -self.max_tilt_rate, self.max_tilt_rate))

        telem = RateCommandTelemetry(
            commanded_pan_dps=cmd_pan_dps,
            commanded_tilt_dps=cmd_tilt_dps,
            achieved_pan_dps=achieved_pan,
            achieved_tilt_dps=achieved_tilt,
            is_saturated=is_sat,
        )

        return achieved_pan, achieved_tilt, telem

    def reset(self) -> None:
        self.total_saturation_count = 0
        self.total_commands = 0
