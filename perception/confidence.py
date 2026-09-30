"""Beacon detection confidence scoring."""

from __future__ import annotations
import math
from typing import Optional
import numpy as np
from perception.candidate_filter import CandidateBlob


class ConfidenceScorer:
    """Computes a calibrated confidence score [0.0, 1.0] for a candidate blob."""

    def __init__(self, target_expected_size: float = 10.0) -> None:
        self.expected_size = target_expected_size
        self.expected_area = target_expected_size ** 2

    def score(self, blob: CandidateBlob) -> float:
        """Calculates multi-attribute confidence.
        
        Factors:
        - SNR score (0.0 to 1.0)
        - Peak intensity score (0.0 to 1.0)
        - Shape/aspect ratio score (0.0 to 1.0)
        - Size consistency score (0.0 to 1.0)
        """
        # 1. SNR term (SNR >= 10 gives 1.0)
        snr_score = min(max((blob.snr - 2.0) / 8.0, 0.0), 1.0)

        # 2. Peak intensity score
        peak_score = min(max(blob.peak_val / 200.0, 0.0), 1.0)

        # 3. Shape symmetry (aspect ratio close to 1.0)
        shape_score = min(max(blob.aspect_ratio, 0.0), 1.0)

        # 4. Fill factor / compactness within bounding box
        bbox_area = max(blob.bbox_w * blob.bbox_h, 1)
        fill_factor = min(max(blob.area_px / bbox_area, 0.0), 1.0)

        # 5. Size consistency
        size_ratio = blob.area_px / max(self.expected_area, 1.0)
        if size_ratio > 1.0:
            size_ratio = 1.0 / size_ratio
        size_score = min(max(size_ratio, 0.0), 1.0)

        # Weighted combination
        confidence = (
            0.35 * snr_score +
            0.25 * peak_score +
            0.15 * shape_score +
            0.15 * fill_factor +
            0.10 * size_score
        )

        return float(min(max(confidence, 0.0), 1.0))
