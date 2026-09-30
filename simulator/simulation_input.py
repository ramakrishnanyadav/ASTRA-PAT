"""Simulation input adapter feeding synthetic frames and ground truth to TrackerCore."""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from app.config import AppConfig
from coordinates.transforms import CoordinateTransformer
from simulator.scene import VirtualScene
from simulator.target import OpticalTarget
from simulator.camera import VirtualCamera
from simulator.disturbances import DisturbanceEngine
from simulator.ground_truth import GroundTruthFrame, GroundTruthRecorder


class BaseInputAdapter:
    """Abstract interface for video input adapters."""

    def get_next_frame(self) -> Optional[Tuple[np.ndarray, float, Optional[GroundTruthFrame]]]:
        raise NotImplementedError

    def send_control_command(self, pan_rate_dps: float, tilt_rate_dps: float) -> None:
        """Sends rate commands to camera actuator (no-op in passive/MP4 mode)."""
        pass

    def reset(self) -> None:
        raise NotImplementedError


class SimulationInput(BaseInputAdapter):
    """Generates synthetic video stream with virtual camera, disturbances, and ground truth."""

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(self.cfg.seed)

        self.transformer = CoordinateTransformer()
        self.scene = VirtualScene(self.cfg.scene, rng=self.rng)
        self.target = OpticalTarget(self.cfg.target, self.cfg.scene, rng=self.rng)
        self.camera = VirtualCamera(self.cfg.camera, self.cfg.scene, transformer=self.transformer)
        self.disturbances = DisturbanceEngine(self.cfg.disturbances, rng=self.rng)
        self.recorder = GroundTruthRecorder()

        self.dt = 1.0 / max(self.cfg.camera.update_hz, 1.0)
        self.current_time = 0.0
        self.frame_id = 0

        # Command buffer from controller
        self._cmd_pan_rate_dps = 0.0
        self._cmd_tilt_rate_dps = 0.0

    def send_control_command(self, pan_rate_dps: float, tilt_rate_dps: float) -> None:
        self._cmd_pan_rate_dps = pan_rate_dps
        self._cmd_tilt_rate_dps = tilt_rate_dps

    def get_next_frame(self) -> Optional[Tuple[np.ndarray, float, Optional[GroundTruthFrame]]]:
        # 1. Apply camera actuator commands from previous step
        achieved_pan_rate, achieved_tilt_rate, saturated = self.camera.apply_rate_command(
            self._cmd_pan_rate_dps,
            self._cmd_tilt_rate_dps,
            self.dt
        )

        # 2. Update target motion
        tx, ty, tvx, tvy, tax, tay, visible = self.target.update(self.current_time, self.dt)

        # 3. Compute disturbances
        jitter_x, jitter_y = self.disturbances.compute_camera_jitter()
        plat_x, plat_y = self.disturbances.compute_platform_motion(self.current_time, self.dt)

        # 4. Render scene canvas
        stamp = self.target.render_stamp()
        canvas = self.scene.render_canvas(tx, ty, stamp, target_visible=visible)

        # 5. Extract camera viewport
        sensor_frame = self.camera.capture_frame(
            canvas,
            jitter_px=(jitter_x, jitter_y),
            platform_px=(plat_x, plat_y)
        )

        # 6. Apply sensor and atmospheric disturbances
        atm_frame = self.disturbances.apply_atmosphere(sensor_frame)
        noisy_frame = self.disturbances.apply_sensor_noise(atm_frame)

        # 7. Compute ground truth sensor position
        # True sensor position includes jitter and platform offset
        true_sensor_x, true_sensor_y = self.transformer.scene_to_sensor(
            tx, ty, self.camera.pan_deg, self.camera.tilt_deg
        )
        # Shift true sensor coordinates by jitter and platform
        true_sensor_x -= (jitter_x + plat_x)
        true_sensor_y -= (jitter_y + plat_y)

        # Check visibility on sensor
        in_bounds = self.transformer.is_in_sensor_bounds(true_sensor_x, true_sensor_y, margin=self.cfg.target.size_px)
        is_truly_visible = visible and in_bounds

        # Ground truth frame
        gt_frame = GroundTruthFrame(
            frame_id=self.frame_id,
            timestamp_s=self.current_time,
            target_scene_x=tx,
            target_scene_y=ty,
            target_scene_vx=tvx,
            target_scene_vy=tvy,
            target_scene_ax=tax,
            target_scene_ay=tay,
            target_sensor_x=float(true_sensor_x) if is_truly_visible else -1.0,
            target_sensor_y=float(true_sensor_y) if is_truly_visible else -1.0,
            target_visible=is_truly_visible,
            camera_pan_deg=self.camera.pan_deg,
            camera_tilt_deg=self.camera.tilt_deg,
            camera_pan_rate_dps=achieved_pan_rate,
            camera_tilt_rate_dps=achieved_tilt_rate,
            slew_saturated=saturated,
            jitter_x_px=jitter_x,
            jitter_y_px=jitter_y,
            platform_x_px=plat_x,
            platform_y_px=plat_y,
        )
        self.recorder.record(gt_frame)

        timestamp = self.current_time
        self.current_time += self.dt
        self.frame_id += 1

        return noisy_frame, timestamp, gt_frame

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.cfg.seed)
        self.scene = VirtualScene(self.cfg.scene, rng=self.rng)
        self.target = OpticalTarget(self.cfg.target, self.cfg.scene, rng=self.rng)
        self.camera = VirtualCamera(self.cfg.camera, self.cfg.scene, transformer=self.transformer)
        self.disturbances = DisturbanceEngine(self.cfg.disturbances, rng=self.rng)
        self.recorder.clear()
        self.current_time = 0.0
        self.frame_id = 0
        self._cmd_pan_rate_dps = 0.0
        self._cmd_tilt_rate_dps = 0.0
