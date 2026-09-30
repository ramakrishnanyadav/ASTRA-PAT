"""MP4 video input adapter for Benchmark-2 evaluation."""

from __future__ import annotations
from typing import Optional, Tuple
import cv2
import numpy as np
from simulator.simulation_input import BaseInputAdapter
from simulator.ground_truth import GroundTruthFrame


class MP4Input(BaseInputAdapter):
    """Decodes .mp4 benchmark videos and feeds full-frame grayscale frames directly to TrackerCore.
    
    Bypasses the virtual PTZ camera actuator entirely.
    """

    def __init__(self, video_path: str, fps: float = 30.0) -> None:
        self.video_path = video_path
        self.default_fps = fps
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise FileNotFoundError(f"Failed to open MP4 video at {video_path}")

        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = video_fps if video_fps > 0 else self.default_fps
        self.dt = 1.0 / self.fps

        self.frame_id = 0
        self.current_time = 0.0

    def get_next_frame(self) -> Optional[Tuple[np.ndarray, float, Optional[GroundTruthFrame]]]:
        if not self.cap.isOpened():
            return None

        ret, frame_bgr = self.cap.read()
        if not ret or frame_bgr is None:
            return None

        # Convert to monochrome / grayscale
        if len(frame_bgr.shape) == 3:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_bgr

        timestamp = self.current_time
        self.current_time += self.dt
        self.frame_id += 1

        # In pure MP4 benchmark, ground truth is either parsed externally or None
        return gray, timestamp, None

    def send_control_command(self, pan_rate_dps: float, tilt_rate_dps: float) -> None:
        # Passive mode: camera commands are intentionally ignored/bypassed
        pass

    def reset(self) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_id = 0
        self.current_time = 0.0

    def close(self) -> None:
        if self.cap.isOpened():
            self.cap.release()
