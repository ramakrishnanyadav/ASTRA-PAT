"""Kinematic lookahead prediction and adaptive Region of Interest (ROI) generation."""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
from app.config import TrackerConfig
from tracking.state_machine import TrackingState


class AdaptiveROIManager:
    """Computes lookahead predicted position and dynamically scales search ROI based on filter uncertainty."""

    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        base_margin: float = 30.0,
        max_margin: float = 180.0,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.base_margin = base_margin
        self.max_margin = max_margin

    def compute_next_roi(
        self,
        state: TrackingState,
        pred_x: float,
        pred_y: float,
        pos_std_x: float,
        pos_std_y: float,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Calculates next frame's ROI bounding box (x, y, w, h) or None for full-frame search."""
        if state in (TrackingState.SEARCH, TrackingState.FAILSAFE):
            return None

        # Margin scales with state and filter positional covariance
        if state == TrackingState.TRACK:
            margin = self.base_margin + 3.0 * max(pos_std_x, pos_std_y)
        elif state == TrackingState.DEGRADED:
            margin = self.base_margin * 1.5 + 4.0 * max(pos_std_x, pos_std_y)
        elif state == TrackingState.PREDICT:
            margin = self.base_margin * 2.0 + 5.0 * max(pos_std_x, pos_std_y)
        elif state == TrackingState.REACQUIRE:
            # Hierarchical expansion
            margin = self.max_margin
        else:
            return None

        margin = min(margin, self.max_margin)

        # ROI centered at predicted position
        rx0 = int(round(pred_x - margin))
        ry0 = int(round(pred_y - margin))
        rx1 = int(round(pred_x + margin))
        ry1 = int(round(pred_y + margin))

        # Clamp to frame dimensions
        rx0 = max(0, min(rx0, self.frame_width - 10))
        ry0 = max(0, min(ry0, self.frame_height - 10))
        rx1 = max(rx0 + 10, min(rx1, self.frame_width))
        ry1 = max(ry0 + 10, min(ry1, self.frame_height))

        rw = rx1 - rx0
        rh = ry1 - ry0

        # If ROI is larger than 75% of screen, default to full-frame
        if rw >= self.frame_width * 0.85 and rh >= self.frame_height * 0.85:
            return None

        return (rx0, ry0, rw, rh)
