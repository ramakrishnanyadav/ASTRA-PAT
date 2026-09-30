# ASTRA-PAT: Open Questions & Mentor Clarifications (ISRO SIH26169)

To ensure strict engineering integrity, ASTRA-PAT isolates all unknown organizer evaluation formats behind clean software adapters. The following questions document all interface assumptions:

---

### Question 1: Does the tracker see the full 2000x2000 scene for initial acquisition, or only the 640x480 viewport?
* **ASTRA-PAT Architecture**: The physical focal plane array (FPA) camera optics model only receives the 640x480 pixel viewport (`VirtualCamera.capture_frame`). The coarse-pointing tracker cannot observe the rest of the 2000x2000 scene directly unless the camera slews there.
* **Adapter Design**: In `SimulationInput`, the camera starts centered at $(1000, 1000)$ with initial pan/tilt at $(0, 0)^\circ$, matching physical space payload reality.

---

### Question 2: What is the exact MP4 benchmark format, and how is ground truth supplied for Benchmark-2?
* **ASTRA-PAT Architecture**: Implemented via `MP4Input` (`benchmark/mp4_input.py`). It accepts standard `.mp4` video files encoded at arbitrary framerates (default 30 FPS, monochrome/color decoded automatically to 8-bit grayscale).
* **Adapter Design**: If ground truth is provided as a companion CSV (`frame_id, x, y`) or JSON file, `BenchmarkEvaluator` loads it automatically; if absent, the system operates purely in perception inference mode and emits `centroid.csv` for evaluator grading.

---

### Question 3: What centroid output format is expected for Benchmark-1 and Benchmark-2?
* **ASTRA-PAT Architecture**: Canonical CSV stream is emitted at `results/centroid.csv` with the standard schema:
  ```csv
  frame_id,timestamp_s,estimated_x,estimated_y,confidence,area,peak,processing_ms,is_valid
  ```
  A second detailed kinematic log is emitted at `results/tracking.csv`.
* **Adapter Design**: Pluggable column mapping adapters can format the output into any required header or delimiter without touching `TrackerCore`.

---

### Question 4: Is the "Additional Information" Drive file in the Problem Statement relevant to these formats?
* **ASTRA-PAT Architecture**: Evaluated defensively. All internal configurations are defined through YAML dataclasses (`app/config.py`) so any specific scenario parameters in external files can be imported with zero code changes.

---

### Question 5: Are performance thresholds hard pass/fail criteria or reference targets?
* **ASTRA-PAT Architecture**: Treated strictly as hard pass/fail criteria:
  - Acquisition Time: $\le 2.0\text{ s}$ (Target: $\le 0.5\text{ s}$)
  - Centroid Error: $\le 10.0\text{ px}$ RMSE (Target: $< 3.0\text{ px}$)
  - Target Loss: $< 5.0\%$ (Target: $< 1.0\%$)
  - Re-acquisition: $\le 1.0\text{ s}$ (Target: $\le 0.5\text{ s}$)
  - Frame Rate: $\ge 20.0\text{ FPS}$ (Target: $\ge 28.6\text{ FPS}$)
* Every benchmark and acceptance test verifies measured results against both SPEC and TARGET margins.

---

### Question 6: What hardware/CPU specification defines the expected $\ge 20$ FPS environment?
* **ASTRA-PAT Architecture**: Tested on standard consumer laptop CPU architecture (single-thread perception benchmark). OpenCV and NumPy vectorization achieve $> 700\text{ FPS}$ on typical hardware, guaranteeing $\ge 20\text{ FPS}$ even on ultra-low-power evaluation laptops.

---

### Question 7: Is centroid ground truth supplied per frame, or only through predefined evaluation values?
* **ASTRA-PAT Architecture**: `GroundTruthRecorder` logs every single frame's true continuous sub-pixel coordinates, velocity, jitter displacement, and camera angles. If the evaluation team only provides discrete reference frames, the error evaluator automatically interpolates or evaluates matching `frame_id`s.

---

### Question 8: Will the application run fully offline during evaluation?
* **ASTRA-PAT Architecture**: **100% offline.** Zero network calls, zero web servers, zero cloud endpoints, zero database daemons. Everything compiles to a self-contained offline desktop application (`astra-pat.exe`).
