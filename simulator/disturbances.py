"""Image-space disturbance models and platform perturbation engines.

NOTE ON ATMOSPHERIC MODELLING:
As required by specification, the atmospheric effects implemented here are
IMAGE-SPACE APPROXIMATIONS (contrast attenuation, blur, synthetic rain streaks,
brightness reduction), NOT physical radiometric or radiative transfer propagation models.
"""

from __future__ import annotations
import math
from typing import Tuple, Optional
import numpy as np
import cv2
from app.config import DisturbanceConfig


class DisturbanceEngine:
    """Applies noise, atmospheric effects, camera jitter, and platform motion."""

    def __init__(
        self,
        cfg: DisturbanceConfig,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(42)

        # Persistent platform motion state
        self._plat_time = 0.0
        self._rain_streaks: Optional[np.ndarray] = None

    def compute_camera_jitter(self) -> Tuple[float, float]:
        """Calculates instantaneous camera body jitter in pixels (up to +-20 px/frame)."""
        amp = float(self.cfg.camera_jitter_px)
        if amp <= 0.0:
            return 0.0, 0.0
        # High frequency uncorrelated vibration
        jx = float(self.rng.uniform(-amp, amp))
        jy = float(self.rng.uniform(-amp, amp))
        return jx, jy

    def compute_platform_motion(self, t: float, dt: float) -> Tuple[float, float]:
        """Calculates platform drift offset in pixels (up to +-20 px/frame)."""
        amp = float(self.cfg.platform_motion_amp_px)
        m_type = self.cfg.platform_motion_type
        if amp <= 0.0 or m_type == "none":
            return 0.0, 0.0

        freq = max(self.cfg.platform_motion_freq_hz, 0.01)
        omega = 2.0 * math.pi * freq

        if m_type == "linear":
            # Triangular back-and-forth slew
            phase = (t * freq) % 1.0
            triangle = 4.0 * abs(phase - 0.5) - 1.0
            px = amp * triangle
            py = 0.5 * amp * triangle
        elif m_type == "circular":
            px = amp * math.cos(omega * t)
            py = amp * math.sin(omega * t)
        elif m_type == "figure8":
            px = amp * math.sin(omega * t)
            py = 0.5 * amp * math.sin(2.0 * omega * t)
        elif m_type == "random":
            # Smooth low-pass filtered random drift
            d_x = float(self.rng.normal(0, amp * 0.2))
            d_y = float(self.rng.normal(0, amp * 0.2))
            px = np.clip(d_x, -amp, amp)
            py = np.clip(d_y, -amp, amp)
        else:
            px = amp * math.sin(omega * t)
            py = amp * math.cos(omega * t)

        return float(px), float(py)

    def apply_atmosphere(self, image: np.ndarray) -> np.ndarray:
        """Applies image-space atmospheric approximation.
        
        Clear: no modification
        Haze: contrast attenuation + atmospheric airlight veil
        Fog: contrast attenuation + spatial Gaussian blur + haze veil
        Rain: transient falling streaks + added noise
        Low light: severe brightness and contrast reduction
        """
        mode = self.cfg.atmosphere
        if mode == "clear":
            return image

        img = image.astype(np.float32)

        if mode == "haze":
            # Contrast attenuation (transmission factor ~0.55) + airlight offset
            transmission = 0.55
            airlight = 55.0
            out = img * transmission + airlight
            return np.clip(out, 0, 255).astype(np.uint8)

        elif mode == "fog":
            # Contrast attenuation + airlight + Gaussian blur
            transmission = 0.40
            airlight = 75.0
            out = img * transmission + airlight
            out = cv2.GaussianBlur(out, (7, 7), sigmaX=2.0)
            return np.clip(out, 0, 255).astype(np.uint8)

        elif mode == "rain":
            # Dynamic rain streak generation
            h, w = img.shape[:2]
            rain_layer = np.zeros((h, w), dtype=np.float32)
            # Create transient slanted streaks
            num_streaks = int(w * h * 0.0015)
            streak_xs = self.rng.integers(0, w, size=num_streaks)
            streak_ys = self.rng.integers(0, h, size=num_streaks)
            lengths = self.rng.integers(8, 22, size=num_streaks)

            for sx, sy, slen in zip(streak_xs, streak_ys, lengths):
                ex = int(sx + slen * 0.3)
                ey = int(sy + slen)
                cv2.line(rain_layer, (sx, sy), (ex, ey), color=float(self.rng.uniform(70, 160)), thickness=1)

            out = cv2.addWeighted(img, 0.85, rain_layer, 0.45, 0.0)
            return np.clip(out, 0, 255).astype(np.uint8)

        elif mode == "low_light":
            # Reduced exposure / contrast and photon starvation
            out = img * 0.30
            # Photon noise in low light
            noise = self.rng.normal(0, 4.0, size=img.shape)
            out = out + noise
            return np.clip(out, 0, 255).astype(np.uint8)

        return image

    def apply_sensor_noise(self, image: np.ndarray) -> np.ndarray:
        """Applies Gaussian, Poisson, and Salt-and-Pepper noise."""
        img = image.astype(np.float32)

        # 1. Gaussian noise
        if self.cfg.gaussian_sigma_px > 0:
            sigma = float(self.cfg.gaussian_sigma_px)
            noise = self.rng.normal(0, sigma, size=img.shape).astype(np.float32)
            img = img + noise

        # 2. Poisson noise
        if self.cfg.poisson_scaling > 0:
            scale = float(self.cfg.poisson_scaling)
            # Normalize to photons, sample Poisson, scale back
            norm_img = np.maximum(img / 255.0, 0.0)
            peak_photons = max(20.0 / scale, 1.0)
            noisy_photons = self.rng.poisson(norm_img * peak_photons).astype(np.float32)
            img = (noisy_photons / peak_photons) * 255.0

        # 3. Salt and Pepper noise (up to 10% coverage)
        if self.cfg.salt_pepper_ratio > 0:
            ratio = min(max(float(self.cfg.salt_pepper_ratio), 0.0), 0.25)
            num_pixels = img.size
            num_sp = int(num_pixels * ratio)
            
            # Salt (255) and Pepper (0)
            rand_indices = self.rng.choice(num_pixels, size=num_sp, replace=False)
            salt_count = num_sp // 2
            
            flat = img.ravel()
            # Pepper
            flat[rand_indices[:salt_count]] = 0.0
            # Salt
            flat[rand_indices[salt_count:]] = 255.0
            img = flat.reshape(img.shape)

        return np.clip(img, 0, 255).astype(np.uint8)
