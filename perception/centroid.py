"""Sub-pixel intensity-weighted centroid calculation."""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np


class SubpixelCentroidExtractor:
    """Calculates intensity-weighted sub-pixel centroids with background offset correction."""

    def __init__(self, window_radius: int = 5) -> None:
        self.window_radius = max(2, window_radius)

    def compute_centroid(
        self,
        image: np.ndarray,
        center_x: float,
        center_y: float,
        radius: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """Calculates subpixel centroid around (center_x, center_y).
        
        Args:
            image: uint8 or float32 2D array
            center_x: Estimated center x
            center_y: Estimated center y
            radius: Optional search window radius
            
        Returns:
            (subpixel_x, subpixel_y, integrated_flux)
        """
        h, w = image.shape[:2]
        r = radius if radius is not None else self.window_radius

        cx_int = int(round(center_x))
        cy_int = int(round(center_y))

        x0 = max(0, cx_int - r)
        y0 = max(0, cy_int - r)
        x1 = min(w, cx_int + r + 1)
        y1 = min(h, cy_int + r + 1)

        patch = image[y0:y1, x0:x1].astype(np.float32)
        if patch.size == 0:
            return float(center_x), float(center_y), 0.0

        # Estimate local background from patch perimeter
        perimeter = np.concatenate([
            patch[0, :], patch[-1, :], patch[:, 0], patch[:, -1]
        ])
        bg_level = float(np.median(perimeter)) if len(perimeter) > 0 else float(np.min(patch))

        # Weighting: max(0, I - bg)
        weights = np.maximum(0.0, patch - bg_level)
        total_flux = float(np.sum(weights))

        if total_flux < 1e-5:
            return float(peak_x), float(peak_y), 0.0

        # Coordinate grids
        grid_y, grid_x = np.mgrid[y0:y1, x0:x1]

        cx = float(np.sum(grid_x * weights) / total_flux)
        cy = float(np.sum(grid_y * weights) / total_flux)

        return cx, cy, total_flux
