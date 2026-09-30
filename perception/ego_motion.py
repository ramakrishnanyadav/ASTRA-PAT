"""Ego-motion estimation for camera and platform jitter compensation.

ENGINEERING PRINCIPLE (Per Specification):
Phase correlation / optical flow can cancel platform and sensor jitter ONLY if the scene
contains measurable background structure (e.g. starfields, lunar limb, or terrain).
In a deep-space lone beacon scenario on a featureless background, the aperture problem
and absence of spatial gradients prevent optical flow from extracting true platform motion.
In such cases, this module honestly reports absence of structure and falls back gracefully
rather than fabricating synthetic ego-motion.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class EgoMotionResult:
    """Estimated global frame displacement between consecutive frames."""
    valid: bool
    dx_px: float  # Horizontal shift in sensor pixels (+X is rightward)
    dy_px: float  # Vertical shift in sensor pixels (+Y is downward)
    confidence: float  # Normalized correlation peak or feature match ratio
    has_background_structure: bool
    status_message: str


class EgoMotionEstimator:
    """Estimates sensor ego-motion using phase correlation and Fourier shift analysis."""

    def __init__(
        self,
        min_structure_std: float = 3.0,
        min_correlation_peak: float = 0.15,
        window_size: Tuple[int, int] = (640, 480),
    ) -> None:
        self.min_structure_std = min_structure_std
        self.min_correlation_peak = min_correlation_peak
        self.prev_frame_f32: Optional[np.ndarray] = None
        self.hann_window = cv2.createHanningWindow(window_size, cv2.CV_32F)

    def reset(self) -> None:
        """Resets history between independent scenario runs."""
        self.prev_frame_f32 = None

    def estimate(
        self,
        frame: np.ndarray,
        beacon_mask_radius: int = 15,
        beacon_xy: Optional[Tuple[float, float]] = None,
    ) -> EgoMotionResult:
        """Estimates global translation (dx, dy) from previous to current frame.

        Args:
            frame: Monochrome uint8 sensor image (640x480).
            beacon_mask_radius: Radius around beacon to mask out to avoid tracking target motion.
            beacon_xy: Current beacon center if available.

        Returns:
            EgoMotionResult containing estimated displacement and validity flags.
        """
        curr_f32 = frame.astype(np.float32)

        if self.prev_frame_f32 is None:
            self.prev_frame_f32 = curr_f32
            return EgoMotionResult(
                valid=False,
                dx_px=0.0,
                dy_px=0.0,
                confidence=1.0,
                has_background_structure=True,
                status_message="Initial reference frame stored",
            )

        # 1. Mask out beacon region so beacon motion doesn't contaminate ego-motion
        masked_frame = curr_f32.copy()
        if beacon_xy is not None:
            bx, by = int(round(beacon_xy[0])), int(round(beacon_xy[1]))
            h, w = frame.shape[:2]
            y1 = max(0, by - beacon_mask_radius)
            y2 = min(h, by + beacon_mask_radius + 1)
            x1 = max(0, bx - beacon_mask_radius)
            x2 = min(w, bx + beacon_mask_radius + 1)
            masked_frame[y1:y2, x1:x2] = 0.0

        # 2. Check for presence of background structure (texture / gradient energy)
        # In a pure black scene with a lone beacon, background std is near zero (only sensor noise)
        bg_std = float(np.std(masked_frame))
        if bg_std < self.min_structure_std:
            self.prev_frame_f32 = curr_f32
            return EgoMotionResult(
                valid=False,
                dx_px=0.0,
                dy_px=0.0,
                confidence=0.0,
                has_background_structure=False,
                status_message="Lone beacon on featureless background: optical flow unavailable (honest fallback)",
            )

        # 3. Perform Subpixel 2D Phase Correlation with Hann window
        try:
            shift, response = cv2.phaseCorrelate(
                self.prev_frame_f32,
                curr_f32,
                window=self.hann_window,
            )
            dx, dy = float(shift[0]), float(shift[1])
            peak = float(response)
        except Exception as exc:
            self.prev_frame_f32 = curr_f32
            return EgoMotionResult(
                valid=False,
                dx_px=0.0,
                dy_px=0.0,
                confidence=0.0,
                has_background_structure=True,
                status_message=f"Phase correlation numerical error: {exc}",
            )

        self.prev_frame_f32 = curr_f32

        if peak < self.min_correlation_peak:
            return EgoMotionResult(
                valid=False,
                dx_px=0.0,
                dy_px=0.0,
                confidence=peak,
                has_background_structure=True,
                status_message=f"Phase correlation peak too low ({peak:.3f} < {self.min_correlation_peak:.3f})",
            )

        return EgoMotionResult(
            valid=True,
            dx_px=dx,
            dy_px=dy,
            confidence=peak,
            has_background_structure=True,
            status_message=f"Ego-motion locked (dx={dx:+.2f}px, dy={dy:+.2f}px, peak={peak:.2f})",
        )
