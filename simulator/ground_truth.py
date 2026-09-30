"""Ground truth tracking definitions and frame records for ASTRA-PAT."""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
import csv


@dataclass
class GroundTruthFrame:
    """Ground truth state for a single simulation frame."""
    frame_id: int
    timestamp_s: float
    # Target state in Scene frame (2000x2000 px)
    target_scene_x: float
    target_scene_y: float
    target_scene_vx: float
    target_scene_vy: float
    target_scene_ax: float
    target_scene_ay: float
    # Target state in Camera Sensor frame (640x480 px)
    target_sensor_x: float
    target_sensor_y: float
    target_visible: bool  # True if inside FOV and not occluded/dropped-out
    # Camera state
    camera_pan_deg: float
    camera_tilt_deg: float
    camera_pan_rate_dps: float
    camera_tilt_rate_dps: float
    slew_saturated: bool
    # Disturbances active
    jitter_x_px: float
    jitter_y_px: float
    platform_x_px: float
    platform_y_px: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GroundTruthRecorder:
    """Thread-safe in-memory and CSV recorder for ground truth telemetry."""

    CSV_HEADER = [
        "frame_id", "timestamp_s",
        "target_scene_x", "target_scene_y", "target_scene_vx", "target_scene_vy",
        "target_scene_ax", "target_scene_ay",
        "target_sensor_x", "target_sensor_y", "target_visible",
        "camera_pan_deg", "camera_tilt_deg",
        "camera_pan_rate_dps", "camera_tilt_rate_dps", "slew_saturated",
        "jitter_x_px", "jitter_y_px", "platform_x_px", "platform_y_px"
    ]

    def __init__(self) -> None:
        self.records: List[GroundTruthFrame] = []

    def record(self, frame: GroundTruthFrame) -> None:
        self.records.append(frame)

    def clear(self) -> None:
        self.records.clear()

    def get_frame(self, frame_id: int) -> Optional[GroundTruthFrame]:
        if 0 <= frame_id < len(self.records):
            return self.records[frame_id]
        return None

    def export_csv(self, filepath: str) -> None:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADER)
            writer.writeheader()
            for rec in self.records:
                writer.writerow(rec.to_dict())
