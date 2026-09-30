"""Optical beacon target kinematic models and raster rendering."""

from __future__ import annotations
import math
from typing import Tuple, Optional
import numpy as np
from app.config import TargetConfig, SceneConfig


class OpticalTarget:
    """Simulates the physical motion and optical footprint of the beacon spot."""

    def __init__(
        self,
        target_cfg: TargetConfig,
        scene_cfg: SceneConfig,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.cfg = target_cfg
        self.scene = scene_cfg
        self.rng = rng or np.random.default_rng(42)

        # Initial position
        if self.cfg.initial_x is not None and self.cfg.initial_y is not None:
            self.x = float(self.cfg.initial_x)
            self.y = float(self.cfg.initial_y)
        else:
            # Safe margin inside 2000x2000
            margin = 300.0
            self.x = float(self.rng.uniform(margin, self.scene.width - margin))
            self.y = float(self.rng.uniform(margin, self.scene.height - margin))

        self.initial_x = self.x
        self.initial_y = self.y
        self.vx = 0.0
        self.vy = 0.0
        self.ax = 0.0
        self.ay = 0.0

        # Motion trajectory state
        self._heading = float(self.rng.uniform(0, 2.0 * math.pi))
        if self.cfg.motion_type == "straight":
            self.vx = self.cfg.speed_px_s * math.cos(self._heading)
            self.vy = self.cfg.speed_px_s * math.sin(self._heading)

    def update(self, t: float, dt: float) -> Tuple[float, float, float, float, float, float, bool]:
        """Updates kinematics for timestamp t and time step dt.
        
        Returns:
            (x, y, vx, vy, ax, ay, is_visible)
        """
        # Check drop-out
        if self.cfg.drop_out_start_s is not None:
            if self.cfg.drop_out_start_s <= t <= (self.cfg.drop_out_start_s + self.cfg.drop_out_duration_s):
                visible = False
            else:
                visible = True
        else:
            visible = True

        motion = self.cfg.motion_type
        if motion == "straight":
            self._update_straight(dt)
        elif motion == "circular":
            self._update_circular(t)
        elif motion == "figure8":
            self._update_figure8(t)
        elif motion == "random":
            self._update_random(dt)
        elif motion == "sinusoidal":
            self._update_sinusoidal(t)
        elif motion == "spiral":
            self._update_spiral(t)
        else:
            self._update_figure8(t)

        return self.x, self.y, self.vx, self.vy, self.ax, self.ay, visible

    def _update_straight(self, dt: float) -> None:
        margin = 100.0
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.ax = 0.0
        self.ay = 0.0

        # Boundary rebound
        if self.x <= margin:
            self.x = margin
            self.vx = abs(self.vx)
        elif self.x >= self.scene.width - margin:
            self.x = self.scene.width - margin
            self.vx = -abs(self.vx)

        if self.y <= margin:
            self.y = margin
            self.vy = abs(self.vy)
        elif self.y >= self.scene.height - margin:
            self.y = self.scene.height - margin
            self.vy = -abs(self.vy)

    def _update_circular(self, t: float) -> None:
        R = self.cfg.circle_radius_px
        v = self.cfg.speed_px_s
        omega = v / max(R, 1.0)
        theta = omega * t

        cx = self.initial_x
        cy = self.initial_y

        self.x = cx + R * math.cos(theta)
        self.y = cy + R * math.sin(theta)
        self.vx = -R * omega * math.sin(theta)
        self.vy = R * omega * math.cos(theta)
        self.ax = -R * (omega ** 2) * math.cos(theta)
        self.ay = -R * (omega ** 2) * math.sin(theta)

    def _update_figure8(self, t: float) -> None:
        A = self.cfg.figure8_scale_px
        period = max(self.cfg.figure8_period_s, 1.0)
        omega = 2.0 * math.pi / period

        cx = self.initial_x
        cy = self.initial_y

        self.x = cx + A * math.sin(omega * t)
        self.y = cy + 0.5 * A * math.sin(2.0 * omega * t)
        self.vx = A * omega * math.cos(omega * t)
        self.vy = A * omega * math.cos(2.0 * omega * t)
        self.ax = -A * (omega ** 2) * math.sin(omega * t)
        self.ay = -2.0 * A * (omega ** 2) * math.sin(2.0 * omega * t)

    def _update_random(self, dt: float) -> None:
        # Heading random walk with mean-reverting heading
        d_theta = float(self.rng.normal(0, self.cfg.random_turn_rate_rad_s * math.sqrt(dt)))
        
        # Soft boundary turning
        margin = 250.0
        repulse_x = 0.0
        repulse_y = 0.0
        if self.x < margin:
            repulse_x = (margin - self.x) / margin
        elif self.x > self.scene.width - margin:
            repulse_x = -(self.x - (self.scene.width - margin)) / margin

        if self.y < margin:
            repulse_y = (margin - self.y) / margin
        elif self.y > self.scene.height - margin:
            repulse_y = -(self.y - (self.scene.height - margin)) / margin

        target_heading = math.atan2(repulse_y, repulse_x) if (repulse_x != 0 or repulse_y != 0) else self._heading
        if repulse_x != 0 or repulse_y != 0:
            self._heading += 3.0 * dt * math.sin(target_heading - self._heading)
        else:
            self._heading += d_theta

        old_vx, old_vy = self.vx, self.vy
        self.vx = self.cfg.speed_px_s * math.cos(self._heading)
        self.vy = self.cfg.speed_px_s * math.sin(self._heading)
        self.ax = (self.vx - old_vx) / max(dt, 1e-4)
        self.ay = (self.vy - old_vy) / max(dt, 1e-4)

        self.x += self.vx * dt
        self.y += self.vy * dt
        self.x = np.clip(self.x, 50.0, self.scene.width - 50.0)
        self.y = np.clip(self.y, 50.0, self.scene.height - 50.0)

    def _update_sinusoidal(self, t: float) -> None:
        v = self.cfg.speed_px_s
        A = self.cfg.sinusoid_amplitude_px
        omega = 2.0 * math.pi * self.cfg.sinusoid_freq_hz

        cx = self.initial_x
        cy = self.initial_y

        self.x = cx + v * (t % 20.0) - (v * 10.0)
        self.y = cy + A * math.sin(omega * t)
        self.vx = v
        self.vy = A * omega * math.cos(omega * t)
        self.ax = 0.0
        self.ay = -A * (omega ** 2) * math.sin(omega * t)

    def _update_spiral(self, t: float) -> None:
        R0 = 50.0
        v_r = 15.0
        omega = 0.5
        r = R0 + v_r * (t % 25.0)
        theta = omega * t

        cx = self.initial_x
        cy = self.initial_y

        self.x = cx + r * math.cos(theta)
        self.y = cy + r * math.sin(theta)
        self.vx = v_r * math.cos(theta) - r * omega * math.sin(theta)
        self.vy = v_r * math.sin(theta) + r * omega * math.cos(theta)
        self.ax = -2.0 * v_r * omega * math.sin(theta) - r * (omega ** 2) * math.cos(theta)
        self.ay = 2.0 * v_r * omega * math.cos(theta) - r * (omega ** 2) * math.sin(theta)

    def render_stamp(self) -> np.ndarray:
        """Generates the optical beacon footprint patch (float32 array 0..intensity)."""
        s = int(math.ceil(self.cfg.size_px))
        # Ensure odd kernel for symmetric center
        if s % 2 == 0:
            s += 1
        pad = s // 2
        stamp_size = 2 * pad + 1
        patch = np.zeros((stamp_size, stamp_size), dtype=np.float32)

        peak = float(self.cfg.intensity)
        shape = self.cfg.shape
        r = self.cfg.size_px / 2.0

        y_idx, x_idx = np.mgrid[-pad:pad+1, -pad:pad+1]

        if shape == "square":
            mask = (np.abs(x_idx) <= r) & (np.abs(y_idx) <= r)
            patch[mask] = peak
        elif shape == "circle":
            dist_sq = x_idx ** 2 + y_idx ** 2
            patch[dist_sq <= (r ** 2)] = peak
        elif shape == "gaussian":
            sigma = max(r / 2.0, 0.8)
            patch = peak * np.exp(-(x_idx ** 2 + y_idx ** 2) / (2.0 * (sigma ** 2)))
        else:
            mask = (np.abs(x_idx) <= r) & (np.abs(y_idx) <= r)
            patch[mask] = peak

        return patch
