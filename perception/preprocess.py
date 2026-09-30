"""Image preprocessing pipeline for background suppression and noise rejection."""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import cv2
from app.config import PerceptionConfig


class FramePreprocessor:
    """Preprocesses camera frames to suppress background clutter, hot pixels, and salt-and-pepper."""

    def __init__(self, cfg: PerceptionConfig) -> None:
        self.cfg = cfg
        # Reusable morphological kernel for background estimation (top-hat)
        # Sized slightly larger than beacon diameter (e.g. 21x21 for 10-20 px beacon)
        self._tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

    def process(self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocesses frame or ROI.
        
        Args:
            frame: uint8 grayscale image
            roi: Optional (x, y, w, h)
            
        Returns:
            (filtered_image, bg_subtracted_image)
        """
        if roi is not None:
            rx, ry, rw, rh = roi
            img_crop = frame[ry:ry+rh, rx:rx+rw]
        else:
            img_crop = frame

        # 1. Median filter (3x3) eliminates 10% salt-and-pepper single-pixel spikes while preserving spot edges
        ksize = self.cfg.preprocess_median_ksize
        if ksize > 1:
            denoised = cv2.medianBlur(img_crop, ksize)
        else:
            denoised = img_crop

        # 2. Background suppression
        if self.cfg.preprocess_enable_bg_sub:
            # Morphological White Top-Hat extracts bright elements smaller than the structuring element
            # while removing smooth non-uniform illumination and large background structures
            bg_sub = cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, self._tophat_kernel)
        else:
            bg_sub = denoised

        return denoised, bg_sub
