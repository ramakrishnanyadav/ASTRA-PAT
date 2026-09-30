"""Configuration models and serialization for ASTRA-PAT."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Literal
import yaml


@dataclass
class SceneConfig:
    width: int = 2000
    height: int = 2000
    background_level: int = 15  # Low dark-space ambient level (0-255)
    num_background_stars: int = 150  # Fixed faint cosmic background stars for ego-motion / realism
    star_max_intensity: int = 70


@dataclass
class CameraConfig:
    sensor_width: int = 640
    sensor_height: int = 480
    fov_pan_deg: float = 4.0
    fov_tilt_deg: float = 3.0
    update_hz: float = 30.0  # Camera frame rate (>=30 Hz)
    control_hz: float = 30.0  # Control update rate (>=20 Hz)
    max_pan_rate_dps: float = 5.0  # Slew rate limit in deg/s (default 5.0, user-selectable 5-10)
    max_tilt_rate_dps: float = 5.0
    initial_pan_deg: float = 0.0
    initial_tilt_deg: float = 0.0


@dataclass
class TargetConfig:
    shape: Literal["square", "circle", "gaussian"] = "square"
    size_px: float = 10.0  # 5-20 px (default 10)
    intensity: float = 245.0  # Peak intensity (0-255)
    initial_x: Optional[float] = None  # None = random
    initial_y: Optional[float] = None  # None = random
    motion_type: Literal[
        "straight", "circular", "figure8", "random", "sinusoidal", "spiral"
    ] = "figure8"
    speed_px_s: float = 90.0  # Tangential / linear speed in scene px/s
    # Motion specific parameters
    circle_radius_px: float = 180.0
    figure8_scale_px: float = 220.0
    figure8_period_s: float = 12.0
    random_turn_rate_rad_s: float = 0.6
    sinusoid_amplitude_px: float = 200.0
    sinusoid_freq_hz: float = 0.1
    # Multi-target / vanishing support
    drop_out_start_s: Optional[float] = None  # Target vanishing time for reacquisition test
    drop_out_duration_s: float = 2.0


@dataclass
class DisturbanceConfig:
    noise_type: Literal["none", "gaussian", "poisson", "salt_pepper", "combined"] = "none"
    gaussian_sigma_px: float = 0.0  # Noise std up to 20 px-equivalent
    poisson_scaling: float = 0.0  # Poisson noise factor
    salt_pepper_ratio: float = 0.0  # Up to 0.10 (~10% salt and pepper)
    camera_jitter_px: float = 0.0  # Camera vibration up to +-20 px/frame
    atmosphere: Literal["clear", "haze", "fog", "rain", "low_light"] = "clear"
    platform_motion_type: Literal[
        "none", "linear", "circular", "random", "spiral", "figure8"
    ] = "none"
    platform_motion_amp_px: float = 0.0  # Platform drift up to +-20 px/frame
    platform_motion_freq_hz: float = 0.5


@dataclass
class PerceptionConfig:
    detector_backend: Literal["classical", "learned_assisted"] = "classical"
    preprocess_median_ksize: int = 3
    preprocess_enable_bg_sub: bool = True
    adaptive_threshold_sigma: float = 3.5  # Number of std above local noise floor
    min_area_px: float = 8.0
    max_area_px: float = 500.0
    min_peak_intensity: float = 60.0
    subpixel_window_radius: int = 5
    enable_learned_scorer: bool = False
    learned_scorer_model_path: Optional[str] = None


@dataclass
class TrackerConfig:
    estimation_model: Literal["constant_velocity", "constant_acceleration", "imm"] = (
        "constant_acceleration"
    )
    process_noise_pos: float = 10.0
    process_noise_vel: float = 100.0
    process_noise_acc: float = 200.0
    measurement_noise_r: float = 6.0  # Subpixel centroid variance in px^2 under noise
    gate_mahalanobis_threshold: float = 64.0  # 8-sigma gate accommodating up to +-20px jitter
    consecutive_hits_to_track: int = 3
    consecutive_misses_to_lost: int = 6
    adaptive_roi_base_margin: float = 65.0
    adaptive_roi_max_margin: float = 180.0


@dataclass
class ControlConfig:
    kp_pan: float = 0.8
    kp_tilt: float = 0.8
    kd_pan: float = 0.05
    kd_tilt: float = 0.05
    feed_forward_weight: float = 1.0  # Use estimated target angular velocity in feed-forward
    enable_rate_limiter: bool = True


@dataclass
class AcceptanceThresholds:
    max_acquisition_sec: float = 2.0
    max_tracking_error_px: float = 10.0
    max_target_loss_pct: float = 5.0
    max_reacquisition_sec: float = 1.0
    min_fps: float = 20.0
    target_fps_margin: float = 28.57  # <35 ms/frame target margin


@dataclass
class AppConfig:
    seed: int = 42
    scene: SceneConfig = field(default_factory=SceneConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    disturbances: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    acceptance: AcceptanceThresholds = field(default_factory=AcceptanceThresholds)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        return cls(
            seed=data.get("seed", 42),
            scene=SceneConfig(**data.get("scene", {})),
            camera=CameraConfig(**data.get("camera", {})),
            target=TargetConfig(**data.get("target", {})),
            disturbances=DisturbanceConfig(**data.get("disturbances", {})),
            perception=PerceptionConfig(**data.get("perception", {})),
            tracker=TrackerConfig(**data.get("tracker", {})),
            control=ControlConfig(**data.get("control", {})),
            acceptance=AcceptanceThresholds(**data.get("acceptance", {})),
        )

    def save_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)

    @classmethod
    def load_yaml(cls, path: str) -> "AppConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)
