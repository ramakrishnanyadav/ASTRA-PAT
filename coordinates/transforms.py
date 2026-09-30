"""ASTRA-PAT Coordinate Systems and Transformations.

This module is the SINGLE SOURCE OF TRUTH for all coordinate transformations:
1. Scene Coordinates (X_s, Y_s): Canvas pixel space (e.g. 2000x2000 px).
2. Camera Angular Coordinates (pan, tilt): Physical pointing angles in degrees.
   - Pan: positive is rightward (+X_s).
   - Tilt: positive is upward (-Y_s in image raster convention).
3. Sensor Pixel Coordinates (x_p, y_p): 2D focal plane array coordinates (e.g. 640x480).
   - Origin (0,0) is top-left of sensor.
   - Sensor center is ((width-1)/2, (height-1)/2).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass(frozen=True)
class CameraOptics:
    """Optical parameters of the focal plane array."""
    sensor_width: int = 640
    sensor_height: int = 480
    fov_pan_deg: float = 4.0
    fov_tilt_deg: float = 3.0

    @property
    def center_x(self) -> float:
        return (self.sensor_width - 1.0) / 2.0

    @property
    def center_y(self) -> float:
        return (self.sensor_height - 1.0) / 2.0

    @property
    def px_per_deg_pan(self) -> float:
        return self.sensor_width / self.fov_pan_deg

    @property
    def px_per_deg_tilt(self) -> float:
        return self.sensor_height / self.fov_tilt_deg


@dataclass(frozen=True)
class SceneDimensions:
    """Dimensions of the virtual 2D scene canvas."""
    width: int = 2000
    height: int = 2000

    @property
    def center_x(self) -> float:
        return (self.width - 1.0) / 2.0

    @property
    def center_y(self) -> float:
        return (self.height - 1.0) / 2.0


class CoordinateTransformer:
    """Transforms coordinates between Scene, Angular, and Sensor frames."""

    def __init__(
        self,
        optics: Optional[CameraOptics] = None,
        scene: Optional[SceneDimensions] = None,
    ) -> None:
        self.optics = optics or CameraOptics()
        self.scene = scene or SceneDimensions()

    def camera_boresight_scene(self, pan_deg: float, tilt_deg: float) -> Tuple[float, float]:
        """Calculates scene coordinates (X_s, Y_s) of the camera boresight."""
        bs_x = self.scene.center_x + pan_deg * self.optics.px_per_deg_pan
        # Tilt up decreases Y in raster coordinate convention
        bs_y = self.scene.center_y - tilt_deg * self.optics.px_per_deg_tilt
        return float(bs_x), float(bs_y)

    def scene_to_angular(self, scene_x: float, scene_y: float) -> Tuple[float, float]:
        """Calculates camera pan and tilt angles required to center on scene (X_s, Y_s)."""
        pan_deg = (scene_x - self.scene.center_x) / self.optics.px_per_deg_pan
        tilt_deg = -(scene_y - self.scene.center_y) / self.optics.px_per_deg_tilt
        return float(pan_deg), float(tilt_deg)

    def scene_to_sensor(
        self,
        scene_x: float,
        scene_y: float,
        pan_deg: float,
        tilt_deg: float,
    ) -> Tuple[float, float]:
        """Maps scene point (X_s, Y_s) to sensor pixel (x_p, y_p) given camera pointing."""
        bs_x, bs_y = self.camera_boresight_scene(pan_deg, tilt_deg)
        x_p = self.optics.center_x + (scene_x - bs_x)
        y_p = self.optics.center_y + (scene_y - bs_y)
        return float(x_p), float(y_p)

    def sensor_to_scene(
        self,
        sensor_x: float,
        sensor_y: float,
        pan_deg: float,
        tilt_deg: float,
    ) -> Tuple[float, float]:
        """Maps sensor pixel (x_p, y_p) to scene point (X_s, Y_s) given camera pointing."""
        bs_x, bs_y = self.camera_boresight_scene(pan_deg, tilt_deg)
        scene_x = bs_x + (sensor_x - self.optics.center_x)
        scene_y = bs_y + (sensor_y - self.optics.center_y)
        return float(scene_x), float(scene_y)

    def sensor_error_to_angular_error(
        self,
        err_x_px: float,
        err_y_px: float,
    ) -> Tuple[float, float]:
        """Converts sensor pixel offset (target - center) to angular correction in degrees.
        
        Args:
            err_x_px: target_x - center_x (positive when target is to the right)
            err_y_px: target_y - center_y (positive when target is below center)
            
        Returns:
            (d_pan_deg, d_tilt_deg):
                d_pan_deg is positive to slew right.
                d_tilt_deg is negative to slew down (tilt down = negative tilt).
        """
        d_pan_deg = err_x_px / self.optics.px_per_deg_pan
        # A target with positive err_y is lower on the sensor, so camera must tilt down (negative tilt)
        d_tilt_deg = -err_y_px / self.optics.px_per_deg_tilt
        return float(d_pan_deg), float(d_tilt_deg)

    def is_in_sensor_bounds(self, sensor_x: float, sensor_y: float, margin: float = 0.0) -> bool:
        """Checks if a sensor pixel coordinate lies within the sensor viewable area."""
        return (
            -margin <= sensor_x <= (self.optics.sensor_width - 1.0 + margin)
            and -margin <= sensor_y <= (self.optics.sensor_height - 1.0 + margin)
        )
