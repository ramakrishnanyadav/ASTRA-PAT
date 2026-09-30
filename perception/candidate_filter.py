"""Candidate beacon extraction and validation filtering."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2
from app.config import PerceptionConfig


@dataclass
class CandidateBlob:
    """A detected blob candidate."""
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    area_px: float
    peak_val: float
    peak_x: int
    peak_y: int
    center_x: float
    center_y: float
    mean_val: float
    aspect_ratio: float
    snr: float


class CandidateValidator:
    """Filters candidate connected components using geometric, radiometric, and SNR rules."""

    def __init__(self, cfg: PerceptionConfig) -> None:
        self.cfg = cfg

    def extract_candidates(self, img_filtered: np.ndarray, img_bgsub: np.ndarray) -> List[CandidateBlob]:
        """Performs adaptive thresholding, connected component analysis, and feature extraction."""
        # 1. Compute robust noise stats on background-subtracted frame
        # In a top-hat image, background is concentrated at lower values
        p95 = float(np.percentile(img_bgsub, 95))
        bg_mask = (img_bgsub <= max(p95, 1.0))
        bg_pixels = img_bgsub[bg_mask]
        
        bg_mean = float(np.mean(bg_pixels)) if bg_pixels.size > 0 else 0.0
        bg_std = float(np.std(bg_pixels)) if bg_pixels.size > 0 else 1.0

        threshold_val = max(
            float(self.cfg.min_peak_intensity) * 0.4,
            bg_mean + self.cfg.adaptive_threshold_sigma * max(bg_std, 2.0)
        )

        # 2. Binary mask
        _, binary = cv2.threshold(img_bgsub, int(threshold_val), 255, cv2.THRESH_BINARY)

        # 3. Connected components with statistics
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        candidates: List[CandidateBlob] = []

        for i in range(1, num_labels):
            area = float(stats[i, cv2.CC_STAT_AREA])
            bx = int(stats[i, cv2.CC_STAT_LEFT])
            by = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])

            # Immediate rejection: single pixel hot/salt noise
            if area < self.cfg.min_area_px or area > self.cfg.max_area_px:
                continue

            # Aspect ratio check
            short_side = min(bw, bh)
            long_side = max(bw, bh)
            aspect_ratio = short_side / max(long_side, 1)
            if aspect_ratio < 0.35:
                # Elongated streaks or line noise
                continue

            # Extract component mask and peak
            blob_patch = img_bgsub[by:by+bh, bx:bx+bw]
            label_patch = labels[by:by+bh, bx:bx+bw]
            mask = (label_patch == i)

            if not np.any(mask):
                continue

            masked_vals = blob_patch[mask]
            peak_val = float(np.max(masked_vals))
            mean_val = float(np.mean(masked_vals))

            if peak_val < self.cfg.min_peak_intensity:
                continue

            # Find peak coordinate within patch
            py_rel, px_rel = np.unravel_index(np.argmax(blob_patch * mask), blob_patch.shape)
            peak_x = bx + int(px_rel)
            peak_y = by + int(py_rel)

            # Centroid from connected components
            cc_cx = float(centroids[i, 0])
            cc_cy = float(centroids[i, 1])

            # Local SNR estimation
            snr = (peak_val - bg_mean) / max(bg_std, 1.0)

            candidates.append(CandidateBlob(
                bbox_x=bx,
                bbox_y=by,
                bbox_w=bw,
                bbox_h=bh,
                area_px=area,
                peak_val=peak_val,
                peak_x=peak_x,
                peak_y=peak_y,
                center_x=cc_cx,
                center_y=cc_cy,
                mean_val=mean_val,
                aspect_ratio=aspect_ratio,
                snr=snr,
            ))

        return candidates
