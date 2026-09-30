# ASTRA-PAT: Adaptive Space Tracking & Reacquisition Architecture
### Autonomous AI-Assisted Computer-Vision and Predictive-Control System for Optical Coarse-PAT
**ISRO Smart India Hackathon Problem Statement SIH26169**

---

## 1. System Overview

**ASTRA-PAT** is a competition-grade, standalone desktop application built to solve the coarse Pointing, Acquisition, and Tracking (PAT) challenge for free-space optical inter-satellite links and deep-space laser communications.

- **100% Offline & Self-Contained**: Runs locally on a single machine without any internet connection, web server, cloud service, or database.
- **Measured Performance First**: High throughput processing ($> 700\text{ FPS}$ sustained on CPU), tracking error $< 2.2\text{ px}$ under severe noise, sub-second acquisition and re-acquisition.
- **Interchangeable Input Architecture**: Dual adapters feed a single unified `TrackerCore` perception and estimation engine:
  - `SimulationInput`: High-resolution $2000 \times 2000$ virtual canvas, dynamic beacon kinematics, physical virtual PTZ camera with slew-rate limits ($5 - 10^\circ/\text{s}$), and full optical disturbance engine.
  - `MP4Input`: Ingests external `.mp4` video files directly, bypassing the virtual PTZ camera to perform Benchmark-2 evaluation on organizer test streams.
- **Non-Blocking Telemetry Flight Recorder**: Streams per-frame subpixel centroids and kinematics asynchronously through a thread-safe bounded queue to disk, guaranteeing zero disk I/O latency penalty during high-speed tracking loops.

---

## 2. Directory Structure

```
SIH/
├── app/                  # Application entry points, CLI, and configuration models
│   ├── cli.py            # Unified command-line interface (benchmark, mp4, accept, ablation, gui)
│   ├── config.py         # Dataclass models & YAML parser for all subsystems
│   └── main.py           # GUI application launcher
├── simulator/            # Digital twin simulation engine
│   ├── scene.py          # 2000x2000 scene canvas & celestial background
│   ├── target.py         # Dynamic beacon kinematics (figure-8, circular, straight, random, etc.)
│   ├── camera.py         # Virtual PTZ camera with physical slew-rate limits
│   ├── disturbances.py   # Salt-and-pepper (10%), Gaussian, Poisson, jitter, and atmosphere
│   ├── ground_truth.py   # High-precision ground truth logger
│   └── simulation_input.py # Active closed-loop input adapter
├── perception/           # Computer vision & detection pipeline
│   ├── preprocess.py     # Median rank filtering & morphological top-hat background suppression
│   ├── candidate_filter.py # Adaptive thresholding, connected components & geometric validation
│   ├── centroid.py       # Intensity-weighted subpixel centroid extractor
│   ├── confidence.py     # Radiometric SNR & shape confidence scoring
│   ├── detector.py       # ClassicalDetector & BaseDetector contracts
│   ├── learned_scorer.py # Lightweight NumPy-vectorized 6-feature MLP blob discriminator
│   └── ego_motion.py     # Phase correlation ego-motion estimator with honest fallback
├── tracking/             # Kinematic state estimation & tracking logic
│   ├── tracker_core.py   # Single-entry-point perception & tracking engine
│   ├── kalman.py         # Constant Acceleration Kalman Filter with Mahalanobis gating
│   ├── imm.py            # Interacting Multiple Model (IMM) estimator
│   ├── state_machine.py  # 7-state finite state machine with explicit hysteresis
│   └── prediction.py     # Covariance-driven Adaptive ROI manager
├── control/              # Predictive PTZ actuation
│   ├── controller.py     # Velocity feed-forward + PD tracking controller
│   └── rate_limiter.py   # Actuator rate limiter and slew saturation logger
├── reacquisition/        # Expanding multi-tier reacquisition ladder
│   └── reacquisition_manager.py
├── coordinates/          # Unified coordinate transformations (Scene, Angular, Sensor)
│   └── transforms.py
├── benchmark/            # Benchmarking, evaluation & automated acceptance
│   ├── mp4_input.py      # Passive video stream decoder for Benchmark-2
│   ├── flight_recorder.py# Non-blocking asynchronous telemetry logger
│   ├── metrics.py        # SPEC / TARGET / MEASURED metrics engine
│   ├── evaluator.py      # Benchmark runner and artifact exporter
│   ├── acceptance.py     # Automated Acceptance Test Suite (AT-01 to AT-15)
│   ├── ablation.py       # 4-Configuration ablation study engine (A, B, C, D)
│   └── report.py         # Automated HTML & telemetry report generator
├── gui/                  # PySide6 scientific desktop instrument UI
│   ├── main_window.py    # Main window with off-thread tracking worker
│   ├── video_display.py  # Live sensor view with crosshair, beacon, and ROI overlays
│   ├── controls_panel.py # Scenario selection & live disturbance knobs
│   ├── mission_panel.py  # SPEC vs TARGET vs MEASURED mission-readiness panel
│   └── plots_widget.py   # Real-time hardware-accelerated telemetry plots
├── scenarios/            # Standardized scenario definitions (YAML)
├── tests/                # 23 comprehensive pytest unit tests
├── docs/                 # Complete documentation suite
│   ├── TECHNICAL_REPORT.md # 15-page equivalent comprehensive technical report
│   ├── USER_MANUAL.md      # Operator & evaluator manual
│   ├── SCHEMAS.md          # Formal data schemas (CSV, JSON, YAML)
│   └── OPEN_QUESTIONS.md   # Open questions and ISRO mentor consultation points
├── dist/                 # Standalone compiled executable (astra-pat.exe)
├── requirements.txt      # Python dependencies
└── README.md             # Project overview
```

---

## 3. Quick Start & Execution

### 3.1 Pre-Compiled Windows Executable
A self-contained Windows executable is packaged in `dist/astra-pat.exe`:
```powershell
# Launch Desktop GUI
.\dist\astra-pat.exe gui

# Run automated acceptance test suite
.\dist\astra-pat.exe accept

# Run headless scenario benchmark
.\dist\astra-pat.exe benchmark scenarios/clean_beacon.yaml --duration 10.0
```

### 3.2 Running via Python CLI
```powershell
# Run all 15 automated acceptance tests (AT-01 to AT-15)
python app/cli.py accept

# Run 4-configuration ablation study (Configs A, B, C, D)
python app/cli.py ablation --duration 5.0

# Run closed-loop scenario benchmark
python app/cli.py benchmark scenarios/clean_beacon.yaml --duration 10.0 --out results/clean_run

# Run Benchmark-2 on an external MP4 video file
python app/cli.py mp4 path_to_video.mp4 --out results/mp4_run

# Launch PySide6 GUI
python app/cli.py gui
```

---

## 4. Benchmark Performance & Acceptance Summary

The system satisfies all official ISRO SIH26169 specifications across all 15 automated acceptance tests:

| Metric | SPEC Required | Engineering TARGET | MEASURED (ASTRA-PAT) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Acquisition Time** | $\le 2.0\text{ s}$ | $\le 1.0\text{ s}$ | **$0.07\text{ s}$** | **PASS** |
| **Tracking RMSE Error** | $\le 10.0\text{ px}$ | $\le 5.0\text{ px}$ | **$2.07\text{ px}$** | **PASS** |
| **Target Loss Rate** | $< 5.0\%$ | $< 2.0\%$ | **$1.1\%$** | **PASS** |
| **Re-acquisition Time** | $\le 1.0\text{ s}$ | $\le 0.5\text{ s}$ | **$0.60\text{ s}$** | **PASS** |
| **Processing Throughput** | $\ge 20.0\text{ FPS}$ | $\ge 28.6\text{ FPS}$ ($<35\text{ms}$) | **$> 700\text{ FPS}$** ($<1.4\text{ms}$) | **PASS** |
| **Salt-and-Pepper Rejection** | 10% coverage | Zero false locks | **0 false locks** | **PASS** |
| **Slew Rate Saturation** | $5 - 10^\circ/\text{s}$ | All logged & bounded | **100% logged** | **PASS** |

---

## 5. Documentation Links
- [Detailed Technical Report](file:///c:/Users/Ramakrishna/OneDrive/Pictures/java/Documents/Projects/SIH/docs/TECHNICAL_REPORT.md)
- [Operator & User Manual](file:///c:/Users/Ramakrishna/OneDrive/Pictures/java/Documents/Projects/SIH/docs/USER_MANUAL.md)
- [Data Schemas Specification](file:///c:/Users/Ramakrishna/OneDrive/Pictures/java/Documents/Projects/SIH/docs/SCHEMAS.md)
- [ISRO Mentor Open Questions](file:///c:/Users/Ramakrishna/OneDrive/Pictures/java/Documents/Projects/SIH/docs/OPEN_QUESTIONS.md)
