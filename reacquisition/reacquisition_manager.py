"""Hierarchical beacon re-acquisition manager."""

from __future__ import annotations
from enum import Enum, auto
from typing import Optional, Tuple
import numpy as np


class ReacquisitionLevel(Enum):
    PREDICTION_LOCAL = auto()    # Level 1: Tight ROI around propagated Kalman prediction
    MEDIUM_EXPANSION = auto()    # Level 2: Medium ROI
    LARGE_EXPANSION = auto()     # Level 3: Large ROI covering sensor quadrants
    FULL_FRAME = auto()          # Level 4: Full sensor frame search


class ReacquisitionManager:
    """Orchestrates hierarchical multi-stage reacquisition following target loss."""

    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.current_level = ReacquisitionLevel.PREDICTION_LOCAL
        self.missed_frame_count = 0

    def step_miss(self) -> ReacquisitionLevel:
        """Advances reacquisition hierarchy upon consecutive misses."""
        self.missed_frame_count += 1

        if self.missed_frame_count <= 4:
            self.current_level = ReacquisitionLevel.PREDICTION_LOCAL
        elif self.missed_frame_count <= 10:
            self.current_level = ReacquisitionLevel.MEDIUM_EXPANSION
        elif self.missed_frame_count <= 20:
            self.current_level = ReacquisitionLevel.LARGE_EXPANSION
        else:
            self.current_level = ReacquisitionLevel.FULL_FRAME

        return self.current_level

    def reset_lock(self) -> None:
        """Resets hierarchy when lock is regained."""
        self.current_level = ReacquisitionLevel.PREDICTION_LOCAL
        self.missed_frame_count = 0

    def get_search_roi(self, pred_x: float, pred_y: float) -> Optional[Tuple[int, int, int, int]]:
        """Computes bounding ROI for current reacquisition level."""
        if self.current_level == ReacquisitionLevel.FULL_FRAME:
            return None

        if self.current_level == ReacquisitionLevel.PREDICTION_LOCAL:
            margin = 40.0
        elif self.current_level == ReacquisitionLevel.MEDIUM_EXPANSION:
            margin = 90.0
        elif self.current_level == ReacquisitionLevel.LARGE_EXPANSION:
            margin = 160.0
        else:
            return None

        rx0 = max(0, int(round(pred_x - margin)))
        ry0 = max(0, int(round(pred_y - margin)))
        rx1 = min(self.frame_width, int(round(pred_x + margin)))
        ry1 = min(self.frame_height, int(round(pred_y + margin)))

        rw = rx1 - rx0
        rh = ry1 - ry0
        if rw < 10 or rh < 10:
            return None

        return (rx0, ry0, rw, rh)
