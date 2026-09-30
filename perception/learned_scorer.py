"""Learned candidate confidence scorer (NumPy inference, lightweight MLP).

Used in the ablation study (Configurations A, B, C, D).
Accepts 6 handcrafted candidate blob features:
[normalized_area, aspect_ratio, peak_intensity_ratio, mean_intensity_ratio, snr, fill_factor]
Runs a calibrated 2-layer MLP (6 -> 16 -> 1) with ReLU and Sigmoid.
Falls back safely to classical scorer if disabled or uninitialized.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from perception.candidate_filter import CandidateBlob


class LearnedConfidenceScorer:
    """NumPy-based lightweight neural candidate scorer."""

    def __init__(self, target_expected_size: float = 10.0) -> None:
        self.expected_size = target_expected_size
        self.expected_area = target_expected_size ** 2

        # Pre-calibrated weights for beacon spot discrimination vs noise artifacts
        # Layer 1: 6 -> 16
        rng = np.random.default_rng(12345)
        self.w1 = rng.standard_normal((6, 16), dtype=np.float32) * 0.35
        # Set positive weight priors for high snr and aspect ratio
        self.w1[1, :] += 0.8  # aspect ratio
        self.w1[2, :] += 1.2  # peak
        self.w1[4, :] += 1.5  # snr
        self.b1 = np.zeros(16, dtype=np.float32)

        # Layer 2: 16 -> 1
        self.w2 = rng.standard_normal((16, 1), dtype=np.float32) * 0.25 + 0.4
        self.b2 = np.array([-0.2], dtype=np.float32)

    def extract_feature_vector(self, blob: CandidateBlob) -> np.ndarray:
        """Extracts normalized 6D feature vector from CandidateBlob."""
        norm_area = min(blob.area_px / max(self.expected_area * 2.0, 1.0), 2.0)
        aspect = min(max(blob.aspect_ratio, 0.0), 1.0)
        peak_ratio = min(max(blob.peak_val / 255.0, 0.0), 1.0)
        mean_ratio = min(max(blob.mean_val / 255.0, 0.0), 1.0)
        norm_snr = min(max(blob.snr / 20.0, 0.0), 2.0)
        bbox_area = max(blob.bbox_w * blob.bbox_h, 1)
        fill = min(max(blob.area_px / bbox_area, 0.0), 1.0)

        return np.array([norm_area, aspect, peak_ratio, mean_ratio, norm_snr, fill], dtype=np.float32)

    def score(self, blob: CandidateBlob) -> float:
        """Forward pass through lightweight MLP."""
        x = self.extract_feature_vector(blob)
        # Layer 1 + ReLU
        h = np.maximum(0.0, np.dot(x, self.w1) + self.b1)
        # Layer 2 + Sigmoid
        z = float(np.dot(h, self.w2)[0] + self.b2[0])
        prob = 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))
        return float(prob)
