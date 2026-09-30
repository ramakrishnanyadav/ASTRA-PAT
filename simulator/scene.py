"""Virtual 2000x2000 pixel space scene canvas."""

from __future__ import annotations
from typing import Optional, List, Tuple
import numpy as np
from app.config import SceneConfig


class VirtualScene:
    """Manages the 2000x2000 pixel virtual space scene with background stars."""

    def __init__(
        self,
        cfg: SceneConfig,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(42)

        self.width = self.cfg.width
        self.height = self.cfg.height

        # Pre-generate static background stars
        self.background = np.full((self.height, self.width), self.cfg.background_level, dtype=np.uint8)
        self._init_background_stars()

    def _init_background_stars(self) -> None:
        """Seeds faint background stars across the scene."""
        num_stars = self.cfg.num_background_stars
        if num_stars <= 0:
            return

        xs = self.rng.integers(0, self.width, size=num_stars)
        ys = self.rng.integers(0, self.height, size=num_stars)
        intensities = self.rng.integers(
            self.cfg.background_level + 10,
            max(self.cfg.background_level + 11, self.cfg.star_max_intensity),
            size=num_stars,
        )

        for x, y, val in zip(xs, ys, intensities):
            self.background[y, x] = int(val)
            # Add 3x3 faint halo to 20% of stars
            if self.rng.random() < 0.2:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < self.height and 0 <= nx < self.width:
                            self.background[ny, nx] = max(self.background[ny, nx], int(val * 0.5))

    def render_canvas(
        self,
        target_x: float,
        target_y: float,
        target_stamp: np.ndarray,
        target_visible: bool = True,
    ) -> np.ndarray:
        """Copies the static background and blends the optical beacon stamp at (target_x, target_y)."""
        canvas = self.background.copy()

        if not target_visible:
            return canvas

        stamp_h, stamp_w = target_stamp.shape
        pad_x = stamp_w // 2
        pad_y = stamp_h // 2

        ix = int(round(target_x))
        iy = int(round(target_y))

        # Canvas bounds intersection
        x0 = max(0, ix - pad_x)
        y0 = max(0, iy - pad_y)
        x1 = min(self.width, ix + pad_x + 1)
        y1 = min(self.height, iy + pad_y + 1)

        # Stamp bounds intersection
        sx0 = x0 - (ix - pad_x)
        sy0 = y0 - (iy - pad_y)
        sx1 = sx0 + (x1 - x0)
        sy1 = sy0 + (y1 - y0)

        if x1 > x0 and y1 > y0:
            target_slice = target_stamp[sy0:sy1, sx0:sx1]
            canvas_slice = canvas[y0:y1, x0:x1].astype(np.float32)
            blended = np.maximum(canvas_slice, target_slice)
            canvas[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)

        return canvas
