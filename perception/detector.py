"""Detector contract and implementations (classical and learned-assisted)."""

from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Optional, Tuple, List
import numpy as np
from app.config import PerceptionConfig
from perception.preprocess import FramePreprocessor
from perception.candidate_filter import CandidateValidator, CandidateBlob
from perception.centroid import SubpixelCentroidExtractor
from perception.confidence import ConfidenceScorer
from perception.learned_scorer import LearnedConfidenceScorer


@dataclass
class DetectionResult:
    """Output contract for any optical beacon detector."""
    valid: bool
    x: float
    y: float
    confidence: float
    area: float
    peak: float
    candidate_count: int
    processing_ms: float
    timestamp_s: float
    roi_used: Optional[Tuple[int, int, int, int]] = None


class BaseDetector:
    """Abstract detector contract."""

    def detect(
        self,
        frame: np.ndarray,
        timestamp_s: float,
        roi: Optional[Tuple[int, int, int, int]] = None,
        prior_xy: Optional[Tuple[float, float]] = None,
    ) -> DetectionResult:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class ClassicalDetector(BaseDetector):
    """Production-grade classical detector using morphological top-hat, adaptive thresholding,
    connected-component validation, and sub-pixel intensity centroiding.
    """

    def __init__(self, cfg: PerceptionConfig, target_expected_size: float = 10.0) -> None:
        self.cfg = cfg
        self.preprocessor = FramePreprocessor(cfg)
        self.validator = CandidateValidator(cfg)
        self.centroid_extractor = SubpixelCentroidExtractor(cfg.subpixel_window_radius)
        self.confidence_scorer = ConfidenceScorer(target_expected_size)
        self.learned_scorer = (
            LearnedConfidenceScorer(target_expected_size) if cfg.enable_learned_scorer else None
        )

    def detect(
        self,
        frame: np.ndarray,
        timestamp_s: float,
        roi: Optional[Tuple[int, int, int, int]] = None,
        prior_xy: Optional[Tuple[float, float]] = None,
    ) -> DetectionResult:
        t0 = time.perf_counter()

        # Handle ROI bounds
        rx, ry, rw, rh = 0, 0, frame.shape[1], frame.shape[0]
        if roi is not None:
            rx = max(0, min(roi[0], frame.shape[1] - 1))
            ry = max(0, min(roi[1], frame.shape[0] - 1))
            rw = max(1, min(roi[2], frame.shape[1] - rx))
            rh = max(1, min(roi[3], frame.shape[0] - ry))
            crop = frame[ry:ry+rh, rx:rx+rw]
            actual_roi = (rx, ry, rw, rh)
        else:
            crop = frame
            actual_roi = None

        # 1. Preprocess (denoise + background suppression)
        denoised_crop, bgsub_crop = self.preprocessor.process(crop)

        # 2. Extract and validate candidate blobs
        candidates = self.validator.extract_candidates(denoised_crop, bgsub_crop)

        if not candidates:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            return DetectionResult(
                valid=False,
                x=0.0,
                y=0.0,
                confidence=0.0,
                area=0.0,
                peak=0.0,
                candidate_count=0,
                processing_ms=dt_ms,
                timestamp_s=timestamp_s,
                roi_used=actual_roi,
            )

        # 3. Score candidates and rank
        scored_candidates = []
        for cand in candidates:
            conf = self.confidence_scorer.score(cand)
            if self.learned_scorer is not None:
                learned_p = self.learned_scorer.score(cand)
                # Combine classical radiometric confidence with learned discriminator
                conf = 0.5 * conf + 0.5 * learned_p

            # If prior position is given, incorporate spatial proximity prior
            if prior_xy is not None:
                glob_cx = cand.peak_x + rx
                glob_cy = cand.peak_y + ry
                dist = np.hypot(glob_cx - prior_xy[0], glob_cy - prior_xy[1])
                proximity_weight = np.exp(-0.5 * (dist / 50.0) ** 2)
                combined_score = 0.7 * conf + 0.3 * proximity_weight
            else:
                combined_score = conf
            scored_candidates.append((combined_score, conf, cand))


        # Select highest ranking candidate
        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_raw_conf, best_blob = scored_candidates[0]

        # 4. Extract intensity-weighted subpixel centroid on denoised crop
        # Window size matches candidate dimension plus background margin
        rad = max(self.cfg.subpixel_window_radius, int(max(best_blob.bbox_w, best_blob.bbox_h) // 2 + 2))
        sub_x, sub_y, flux = self.centroid_extractor.compute_centroid(
            denoised_crop,
            best_blob.center_x,
            best_blob.center_y,
            radius=rad,
        )

        # 5. Transform back to full frame coordinates
        full_x = sub_x + rx
        full_y = sub_y + ry

        dt_ms = (time.perf_counter() - t0) * 1000.0

        return DetectionResult(
            valid=True,
            x=float(full_x),
            y=float(full_y),
            confidence=float(best_raw_conf),
            area=float(best_blob.area_px),
            peak=float(best_blob.peak_val),
            candidate_count=len(candidates),
            processing_ms=dt_ms,
            timestamp_s=timestamp_s,
            roi_used=actual_roi,
        )
