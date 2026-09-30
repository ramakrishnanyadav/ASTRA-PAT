"""Virtual Pan-Tilt-Zoom (PTZ) camera sensor and actuator model."""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import cv2
from app.config import CameraConfig, SceneConfig
from coordinates.transforms import CoordinateTransformer, CameraOptics, SceneDimensions


class VirtualCamera:
    """Models a 640x480 monochrome FPA on a 2-DOF pan/tilt gimbal with slew rate limits."""

    def __init__(
        self,
        camera_cfg: CameraConfig,
        scene_cfg: SceneConfig,
        transformer: Optional[CoordinateTransformer] = None,
    ) -> None:
        self.cfg = camera_cfg
        self.scene_cfg = scene_cfg

        optics = CameraOptics(
            sensor_width=self.cfg.sensor_width,
            sensor_height=self.cfg.sensor_height,
            fov_pan_deg=self.cfg.fov_pan_deg,
            fov_tilt_deg=self.cfg.fov_tilt_deg,
        )
        scene_dims = SceneDimensions(width=self.scene_cfg.width, height=self.scene_cfg.height)
        self.transformer = transformer or CoordinateTransformer(optics=optics, scene=scene_dims)

        # Camera state
        self.pan_deg = float(self.cfg.initial_pan_deg)
        self.tilt_deg = float(self.cfg.initial_tilt_deg)
        self.pan_rate_dps = 0.0
        self.tilt_rate_dps = 0.0
        self.slew_saturated = False

        # Actuator limits
        self.max_pan_rate = float(self.cfg.max_pan_rate_dps)
        self.max_tilt_rate = float(self.cfg.max_tilt_rate_dps)

    def apply_rate_command(
        self,
        pan_rate_cmd: float,
        tilt_rate_cmd: float,
        dt: float,
    ) -> Tuple[float, float, bool]:
        """Applies commanded angular rates, enforcing physical slew rate saturation.
        
        Returns:
            (achieved_pan_rate, achieved_tilt_rate, was_saturated)
        """
        # Slew rate saturation
        sat_pan = False
        sat_tilt = False

        if abs(pan_rate_cmd) > self.max_pan_rate:
            achieved_pan_rate = np.sign(pan_rate_cmd) * self.max_pan_rate
            sat_pan = True
        else:
            achieved_pan_rate = pan_rate_cmd

        if abs(tilt_rate_cmd) > self.max_tilt_rate:
            achieved_tilt_rate = np.sign(tilt_rate_cmd) * self.max_tilt_rate
            sat_tilt = True
        else:
            achieved_tilt_rate = tilt_rate_cmd

        self.pan_rate_dps = float(achieved_pan_rate)
        self.tilt_rate_dps = float(achieved_tilt_rate)
        self.slew_saturated = sat_pan or sat_tilt

        # Integrate angles
        self.pan_deg += self.pan_rate_dps * dt
        self.tilt_deg += self.tilt_rate_dps * dt

        return self.pan_rate_dps, self.tilt_rate_dps, self.slew_saturated

    def capture_frame(
        self,
        scene_canvas: np.ndarray,
        jitter_px: Tuple[float, float] = (0.0, 0.0),
        platform_px: Tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        """Extracts the 640x480 sensor viewport from the 2000x2000 scene canvas.
        
        Incorporates camera pointing angle, high-frequency jitter, and low-frequency platform drift.
        """
        bs_x, bs_y = self.transformer.camera_boresight_scene(self.pan_deg, self.tilt_deg)
        effective_center_x = bs_x + jitter_px[0] + platform_px[0]
        effective_center_y = bs_y + jitter_px[1] + platform_px[1]

        # Top-left corner of the sensor viewport on the scene canvas
        top_left_x = effective_center_x - self.transformer.optics.center_x
        top_left_y = effective_center_y - self.transformer.optics.center_y

        # Fast sub-pixel affine translation
        # [x', y'] = [x - top_left_x, y - top_left_y]
        M = np.array([
            [1.0, 0.0, -top_left_x],
            [0.0, 1.0, -top_left_y]
        ], dtype=np.float32)

        sensor_frame = cv2.warpAffine(
            scene_canvas,
            M,
            (self.transformer.optics.sensor_width, self.transformer.optics.sensor_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=int(self.scene_cfg.background_level),
        )

        return sensor_frame
