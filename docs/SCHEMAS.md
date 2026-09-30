# ASTRA-PAT Data Schemas & Telemetry Specification

This document defines the canonical schemas for all configuration files, streaming data logs, and performance metrics emitted by the **ASTRA-PAT** coarse-pointing simulator.

---

## 1. Scenario Configuration Schema (`scenario.yaml`)

Scenario definitions are fully deterministic and formatted in YAML. Every run initializes random number generators with the top-level `seed`.

```yaml
seed: 42                           # Integer seed for deterministic reproduction

scene:
  width: 2000                      # Scene canvas width in pixels (min 2000)
  height: 2000                     # Scene canvas height in pixels (min 2000)
  background_level: 15             # Ambient background black level (0-255)
  num_background_stars: 150        # Ambient celestial background points
  star_max_intensity: 70           # Maximum intensity of cosmic background stars

camera:
  sensor_width: 640                # Focal plane array width in pixels
  sensor_height: 480               # Focal plane array height in pixels
  fov_pan_deg: 4.0                 # Horizontal field of view in degrees (160 px/deg)
  fov_tilt_deg: 3.0                # Vertical field of view in degrees (160 px/deg)
  update_hz: 30.0                  # Camera sensor sampling rate (>= 30 Hz)
  control_hz: 30.0                 # Control loop update rate (>= 20 Hz)
  max_pan_rate_dps: 5.0            # Pan slew rate limit in degrees/sec (5-10 dps)
  max_tilt_rate_dps: 5.0           # Tilt slew rate limit in degrees/sec (5-10 dps)
  initial_pan_deg: 0.0             # Initial camera boresight pan angle
  initial_tilt_deg: 0.0            # Initial camera boresight tilt angle

target:
  shape: "square"                  # "square", "circle", or "gaussian"
  size_px: 10.0                    # Characteristic dimension in pixels (5-20 px)
  intensity: 245.0                 # Peak target intensity (0-255)
  initial_x: 1000.0                # Initial X in scene coordinates (or null for random)
  initial_y: 1000.0                # Initial Y in scene coordinates (or null for random)
  motion_type: "figure8"           # "straight", "circular", "figure8", "random", "sinusoidal", "spiral"
  speed_px_s: 90.0                 # Tangential / linear velocity (scene px/s)
  circle_radius_px: 180.0          # Radius for circular trajectories
  figure8_scale_px: 220.0          # Spatial span for Lissajous figure-8
  figure8_period_s: 12.0           # Orbit period for figure-8 in seconds
  random_turn_rate_rad_s: 0.6      # Angular wandering rate for random motion
  drop_out_start_s: null           # Forced target disappearance timestamp (seconds)
  drop_out_duration_s: 2.0         # Disappearance duration in seconds

disturbances:
  noise_type: "none"               # "none", "gaussian", "poisson", "salt_pepper", "combined"
  gaussian_sigma_px: 0.0           # Gaussian noise standard deviation
  poisson_scaling: 0.0             # Poisson shot noise multiplier
  salt_pepper_ratio: 0.0           # Salt-and-pepper fraction (e.g. 0.10 for 10% coverage)
  camera_jitter_px: 0.0            # High-frequency camera jitter amplitude (+- px/frame)
  atmosphere: "clear"              # "clear", "haze", "fog", "rain", "low_light"
  platform_motion_type: "none"     # "none", "linear", "circular", "random", "spiral", "figure8"
  platform_motion_amp_px: 0.0      # Low-frequency platform drift amplitude (+- px/frame)
  platform_motion_freq_hz: 0.5     # Platform drift frequency in Hz

perception:
  detector_backend: "classical"    # "classical" or "learned_assisted"
  preprocess_median_ksize: 3       # Kernel size for non-linear rank filtering
  preprocess_enable_bg_sub: true   # Enable morphological top-hat background suppression
  adaptive_threshold_sigma: 3.5    # Threshold sigma multiplier above local noise floor
  min_area_px: 8.0                 # Minimum candidate component area (filters isolated salt pixels)
  max_area_px: 500.0               # Maximum candidate component area
  min_peak_intensity: 60.0         # Minimum peak brightness threshold
  subpixel_window_radius: 5        # Radius for intensity-weighted subpixel centroiding
  enable_learned_scorer: false     # Enable 6-feature MLP confidence discriminator

tracker:
  estimation_model: "constant_acceleration"  # "constant_velocity", "constant_acceleration", "imm"
  process_noise_pos: 10.0          # Discrete process noise covariance Q_pos
  process_noise_vel: 100.0         # Process noise Q_vel
  process_noise_acc: 200.0         # Process noise Q_acc
  measurement_noise_r: 6.0         # Sensor measurement noise covariance R (px^2)
  gate_mahalanobis_threshold: 64.0 # Statistical gating distance (8-sigma ellipsoid)
  consecutive_hits_to_track: 3     # Frame confirmations required to transition ACQUIRE -> TRACK
  consecutive_misses_to_lost: 6    # Frame misses required to transition TRACK -> REACQUIRE
  adaptive_roi_base_margin: 65.0   # Tightly-bounded ROI margin in pixels during TRACK
  adaptive_roi_max_margin: 180.0   # Expanded search ROI margin during DEGRADED/PREDICT

control:
  kp_pan: 0.8                      # Proportional gain for pan axis
  kp_tilt: 0.8                     # Proportional gain for tilt axis
  kd_pan: 0.05                     # Derivative gain for pan axis
  kd_tilt: 0.05                    # Derivative gain for tilt axis
  feed_forward_weight: 1.0         # Estimated target velocity feed-forward weight (0.0 to 1.0)
  enable_rate_limiter: true        # Strict enforcement of actuator max slew rates
```

---

## 2. Canonical Centroid Stream (`centroid.csv`)

Emitted per-frame by the **Flight Recorder** during Benchmark-1 and Benchmark-2 evaluation.

| Column | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `frame_id` | Integer | - | Monotonically increasing 0-indexed frame identifier |
| `timestamp_s` | Float | seconds | Timestamp relative to scenario origin ($t_0 = 0.0$) |
| `estimated_x` | Float | pixels | Subpixel X coordinate in sensor coordinates ($0.0 \dots 639.0$) |
| `estimated_y` | Float | pixels | Subpixel Y coordinate in sensor coordinates ($0.0 \dots 479.0$) |
| `confidence` | Float | normalized | Target detection confidence score ($0.0 \dots 1.0$) |
| `area` | Float | pixels$^2$ | Measured connected component pixel area |
| `peak` | Float | ADU | Peak pixel intensity value within candidate blob ($0 \dots 255$) |
| `processing_ms`| Float | milliseconds| Total wall-clock processing time for this frame |
| `is_valid` | Integer | boolean | Binary flag indicating whether detection was validated ($1 = \text{valid}, 0 = \text{invalid}$) |

---

## 3. Kinematic Tracking Telemetry (`tracking.csv`)

Detailed closed-loop state, ground truth, and actuator rate telemetry.

| Column | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `frame_id` | Integer | - | Frame sequence number |
| `timestamp_s` | Float | seconds | Frame timestamp |
| `state` | String | - | Current finite-state machine mode (`SEARCH`, `ACQUIRE`, `TRACK`, `DEGRADED`, `PREDICT`, `REACQUIRE`, `FAILSAFE`) |
| `est_x` | Float | pixels | Filtered target position estimate $\hat{x}$ on sensor |
| `est_y` | Float | pixels | Filtered target position estimate $\hat{y}$ on sensor |
| `est_vx` | Float | px/s | Filtered target velocity estimate $\hat{v}_x$ |
| `est_vy` | Float | px/s | Filtered target velocity estimate $\hat{v}_y$ |
| `est_ax` | Float | px/s$^2$| Filtered target acceleration estimate $\hat{a}_x$ |
| `est_ay` | Float | px/s$^2$| Filtered target acceleration estimate $\hat{a}_y$ |
| `gt_x` | Float | pixels | Ground-truth target sensor X position (-1.0 if not visible) |
| `gt_y` | Float | pixels | Ground-truth target sensor Y position (-1.0 if not visible) |
| `error_px` | Float | pixels | Euclidean centroid error $\| \hat{\mathbf{p}} - \mathbf{p}_{\text{GT}} \|$ |
| `camera_pan_deg` | Float | degrees | Virtual PTZ achieved boresight pan angle |
| `camera_tilt_deg`| Float | degrees | Virtual PTZ achieved boresight tilt angle |
| `cmd_pan_dps` | Float | deg/s | Rate limiter commanded pan rate |
| `cmd_tilt_dps` | Float | deg/s | Rate limiter commanded tilt rate |
| `slew_saturated`| Integer| boolean | 1 if requested rate was clamped by actuator limit, 0 otherwise |

---

## 4. Performance Metrics Summary (`metrics.json`)

Standard JSON format containing final benchmark aggregates and mission-readiness compliance:

```json
{
  "total_frames": 300,
  "eligible_frames": 298,
  "locked_frames": 295,
  "lost_frames": 3,
  "lock_retention_pct": 98.99,
  "target_loss_pct": 1.01,
  "error_mean_px": 2.05,
  "error_rmse_px": 2.12,
  "error_median_px": 1.98,
  "error_p95_px": 3.82,
  "error_max_px": 6.14,
  "acquisition_time_s": 0.067,
  "reacquisition_time_s": null,
  "mean_processing_ms": 1.15,
  "p95_processing_ms": 1.48,
  "mean_fps": 869.5,
  "slew_limit_saturations": 0,
  "checklist": [
    {
      "name": "Acquisition Time",
      "spec_threshold": "<= 2.0 s",
      "target_threshold": "<= 1.0 s",
      "measured_val": "0.067 s",
      "status": "PASS",
      "target_met": true
    },
    {
      "name": "Tracking Error RMSE",
      "spec_threshold": "<= 10.0 px",
      "target_threshold": "<= 5.0 px",
      "measured_val": "2.12 px",
      "status": "PASS",
      "target_met": true
    },
    {
      "name": "Target Loss Rate",
      "spec_threshold": "< 5.0%",
      "target_threshold": "< 2.0%",
      "measured_val": "1.01%",
      "status": "PASS",
      "target_met": true
    },
    {
      "name": "Processing Throughput",
      "spec_threshold": ">= 20.0 FPS",
      "target_threshold": ">= 28.6 FPS (<35ms)",
      "measured_val": "869.5 FPS (1.15 ms)",
      "status": "PASS",
      "target_met": true
    }
  ],
  "all_spec_passed": true
}
```
