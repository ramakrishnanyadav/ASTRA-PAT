# ASTRA-PAT User Manual & Operator Guide

**Adaptive Space Tracking & Reacquisition Architecture (ASTRA-PAT)**  
*Autonomous AI-Assisted Computer-Vision and Predictive-Control System for Optical Coarse-PAT (ISRO Smart India Hackathon SIH26169)*

---

## 1. Overview & Operational Principles

ASTRA-PAT is a competition-grade, standalone desktop simulator and evaluation suite designed to solve the Pointing, Acquisition, and Tracking (PAT) challenge for optical inter-satellite links and deep-space optical terminals.

### Core Architecture Highlights:
- **100% Offline & Self-Contained**: Operates strictly on a single local machine without any cloud services, internet access, databases, or web servers.
- **Dual Interchangeable Input Adapters**:
  - `SimulationInput`: Generates a high-resolution 2000x2000 virtual space environment with dynamic beacon kinematics, physical camera slew limits, and realistic optical disturbances.
  - `MP4Input`: Ingests external `.mp4` video files directly, bypassing the virtual PTZ camera to evaluate coarse-pointing centroiding accuracy on pre-recorded benchmark streams.
- **Unified Perception & Predictive Estimation**: Both input modes pass through the identical `TrackerCore` pipeline (morphological background suppression, subpixel intensity centroiding, Constant Acceleration / IMM Kalman estimation, and 7-state finite state machine).
- **Non-Blocking Telemetry Flight Recorder**: Streams tracking data asynchronously through a thread-safe bounded queue to disk, guaranteeing zero disk I/O latency penalty during high-speed tracking loops.

---

## 2. Installation & Quick Start

### 2.1 Using the Pre-Compiled Standalone Windows Executable
No Python installation or external dependencies are required when running the pre-compiled binary:
```powershell
# Run the PySide6 Desktop GUI
.\dist\astra-pat.exe gui

# Or run headless automated acceptance test suite
.\dist\astra-pat.exe accept

# Or execute a headless scenario benchmark
.\dist\astra-pat.exe benchmark scenarios/clean_beacon.yaml --duration 10.0
```

### 2.2 Running from Source (Python 3.11+)
1. Clone or extract the repository onto the host machine.
2. Install dependencies listed in `requirements.txt`:
   ```powershell
   pip install -r requirements.txt
   ```
3. Run test suite to verify installation:
   ```powershell
   pytest -v
   ```

---

## 3. Command Line Interface (CLI) Reference

ASTRA-PAT provides a full headless CLI for automated grading, continuous integration, and batch benchmarking.

### 3.1 `astra-pat benchmark <scenario.yaml>`
Runs closed-loop simulation on a specified YAML scenario file, generates canonical telemetry, and exports an HTML performance report.
```powershell
python app/cli.py benchmark scenarios/clean_beacon.yaml --duration 10.0 --out results/clean_run
```
**Options:**
- `scenario` *(required)*: Path to the YAML scenario configuration file.
- `--duration` *(default: 10.0)*: Simulation duration in seconds.
- `--out` *(default: "results")*: Destination folder for `centroid.csv`, `tracking.csv`, `metrics.json`, and `report.html`.

### 3.2 `astra-pat mp4 <video.mp4>`
Executes **Benchmark-2** on an external `.mp4` video file, bypassing virtual PTZ camera actuation and feeding the full-frame stream to `TrackerCore`.
```powershell
python app/cli.py mp4 path/to/organizer_benchmark.mp4 --out results/benchmark2
```
**Artifacts Generated:**
- `centroid.csv`: Canonical timestamped subpixel centroid stream.
- `metrics.json`: Summary processing statistics (FPS, mean ms/frame).

### 3.3 `astra-pat accept`
Executes the comprehensive 15-test automated acceptance suite (`AT-01` through `AT-15`) and verifies all performance criteria against specification thresholds.
```powershell
python app/cli.py accept --endurance 10.0
```
**Console Output:**
Displays per-test `PASS`/`FAIL` indicators, measured performance metrics, and writes `acceptance_summary.json` and `acceptance_summary.txt`.

### 3.4 `astra-pat ablation`
Runs the rigorous 4-configuration ablation study to empirically evaluate and justify the system architecture:
- **Config A**: Classical baseline (raw detection, direct feedback, no Kalman filter)
- **Config B**: Classical + Kalman/IMM estimation with velocity feed-forward
- **Config C**: + Learned confidence scorer (MLP blob discriminator)
- **Config D**: Full architecture (Classical + Learned + IMM + Predictive Control)
```powershell
python app/cli.py ablation --duration 5.0 --out results/ablation
```
**Artifacts Generated:**
- `ablation_summary.json`: Raw metric numbers across all 4 configurations.
- `ablation_report.md`: Markdown summary table and scientific justification.

### 3.5 `astra-pat gui`
Launches the interactive PySide6 scientific desktop application.
```powershell
python app/cli.py gui
```

---

## 4. Interactive Desktop GUI Guide

The desktop GUI is built using PySide6 and styled with a clean dark-mode scientific instrument palette.

```
+---------------------------------------------------------------------------------------+
|  ASTRA-PAT | Coarse-PAT Optical Tracking Simulator                                   |
+------------------------------------+--------------------------------------------------+
|                                    |  CONTROLS & DISTURBANCE INJECTION                |
|         LIVE SENSOR VIEW           |  [ Mode: Simulation / MP4 ]                      |
|         (640 x 480 FPA)            |  [ Scenario: figure8 / circular / straight ]     |
|                                    |  [ Noise: Gaussian / Poisson / Salt & Pepper ]   |
|   +----------------------------+   |  [ Atmosphere: Clear / Haze / Fog / Rain ]       |
|   |          [+]               |   |  [ Platform Motion & Jitter Sliders ]            |
|   |         Beacon             |   |  [ Slew Rate Limit: 5 - 10 deg/s ]               |
|   |     [ Adaptive ROI ]       |   |                                                  |
|   +----------------------------+   |  [ Start / Pause / Reset / Generate Report ]     |
+------------------------------------+--------------------------------------------------+
|  MISSION READINESS CHECKLIST       |  REAL-TIME TELEMETRY PLOTS                       |
|  - Acquisition Time : PASS (0.07s) |  - Pixel Tracking Error vs Time (px)             |
|  - Error RMSE       : PASS (2.1px) |  - Detection Confidence vs Time (0.0 - 1.0)      |
|  - Target Loss Rate : PASS (1.1%)  |  - Throughput FPS vs Time (Hz)                   |
|  - Throughput       : PASS (>700)  |                                                  |
+------------------------------------+--------------------------------------------------+
```

### Key GUI Features:
1. **Video Display**:
   - Crosshair overlay at sensor optical boresight $(319.5, 239.5)$.
   - Green bounding box and centroid marker on the detected optical beacon.
   - Yellow bounding box indicating the active **Adaptive ROI**.
   - Red indicator if the target enters `PREDICT` or `REACQUIRE` states.
2. **Controls Panel**:
   - **Scenario Selection**: Quick loading of presets (`clean_beacon.yaml`, `salt_pepper.yaml`, etc.).
   - **Disturbance Knobs**: Live adjustment of Gaussian noise ($\sigma$), Salt-and-Pepper ratio (up to 10%), Camera Jitter ($\pm 20$ px), and Platform Motion drift ($\pm 20$ px).
   - **Atmospheric Conditions**: Immediate switching between Clear, Haze, Fog, Rain, and Low-Light approximations.
   - **Mode Switch**: Seamless transition between closed-loop Simulation and external MP4 benchmark analysis.
3. **Mission-Readiness Checklist**:
   - Displays real-time compliance across all 4 key mission thresholds:
     - Acquisition Time ($\le 2.0\text{ s}$)
     - Tracking Error RMSE ($\le 10.0\text{ px}$)
     - Target Loss Rate ($< 5.0\%$)
     - Processing Speed ($\ge 20.0\text{ FPS}$)
4. **Telemetry Graphs**:
   - High-speed hardware-accelerated scrolling plots rendered using PyQtGraph off the tracking thread.

---

## 5. Judge & Evaluator Protocol

### Benchmark-1 Verification:
1. Open terminal in project root.
2. Run automated test suite:
   ```powershell
   python app/cli.py accept
   ```
3. Verify that all 15 automated acceptance tests output `PASS`.
4. Run a 10-second scenario run:
   ```powershell
   python app/cli.py benchmark scenarios/clean_beacon.yaml --duration 10.0 --out results/bench1
   ```
5. Inspect generated `results/bench1/report.html` and verify `centroid.csv` records.

### Benchmark-2 Verification (MP4 Ingestion):
1. Place organizer's `.mp4` video in the workspace directory.
2. Execute MP4 benchmark runner:
   ```powershell
   python app/cli.py mp4 path_to_video.mp4 --out results/bench2
   ```
3. Open `results/bench2/centroid.csv` to verify per-frame centroid coordinates, confidence scores, and that processing throughput exceeds the $\ge 20\text{ FPS}$ specification.
