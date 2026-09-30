"""Main application window for ASTRA-PAT PySide6 desktop GUI."""

from __future__ import annotations
import os
import sys
import time
import webbrowser
from typing import Optional
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QIcon, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QMenuBar, QMenu, QFileDialog, QMessageBox, QStatusBar
)

from app.config import AppConfig
from simulator.simulation_input import SimulationInput, BaseInputAdapter
from benchmark.mp4_input import MP4Input
from tracking.tracker_core import TrackerCore, TrackerOutput
from tracking.state_machine import TrackingState
from benchmark.metrics import MetricsEngine, PerformanceSummary
from benchmark.report import ReportGenerator
from benchmark.acceptance import AcceptanceTestSuite
from gui.video_display import VideoDisplayWidget
from gui.plots_widget import TelemetryPlotsWidget
from gui.controls_panel import ControlsPanel
from gui.mission_panel import MissionReadinessPanel


class TrackingWorker(QThread):
    """Off-thread execution loop processing frames and streaming telemetry to UI."""

    frame_ready = Signal(np.ndarray, TrackerOutput, object)  # (frame, output, gt)
    summary_ready = Signal(PerformanceSummary)

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.cfg = config
        self.is_running = False
        self.is_paused = False

        self.input_adapter: Optional[BaseInputAdapter] = None
        self.tracker = TrackerCore(is_passive_mp4_mode=False)
        self.metrics = MetricsEngine(self.cfg.acceptance)

        self.mp4_file: Optional[str] = None
        self.output_dir = "results"
        os.makedirs(self.output_dir, exist_ok=True)

    def set_config(self, config: AppConfig) -> None:
        self.cfg = config
        self.metrics = MetricsEngine(self.cfg.acceptance)
        if self.tracker:
            self.tracker.initialize(self.cfg)

    def set_mp4_mode(self, video_path: str) -> None:
        self.mp4_file = video_path
        self.tracker = TrackerCore(is_passive_mp4_mode=True)
        self.tracker.initialize(self.cfg)
        self.input_adapter = MP4Input(video_path)

    def set_simulation_mode(self) -> None:
        self.mp4_file = None
        self.tracker = TrackerCore(is_passive_mp4_mode=False)
        self.tracker.initialize(self.cfg)
        self.input_adapter = SimulationInput(self.cfg)

    def run(self) -> None:
        if self.input_adapter is None:
            self.set_simulation_mode()

        self.is_running = True
        interval = 1.0 / max(self.cfg.camera.update_hz, 1.0)

        while self.is_running:
            if self.is_paused:
                self.msleep(30)
                continue

            t_loop_start = time.perf_counter()

            # 1. Fetch next frame
            item = self.input_adapter.get_next_frame()
            if item is None:
                # Video ended or stream finished
                if self.mp4_file:
                    self.input_adapter.reset()
                    continue
                else:
                    break

            frame, timestamp, gt = item

            # 2. Perception & Tracking
            out = self.tracker.process_frame(frame, timestamp)

            # 3. Actuator feedback
            if out.camera_command is not None:
                self.input_adapter.send_control_command(out.camera_command[0], out.camera_command[1])

            # 4. Metrics
            gt_xy = (gt.target_sensor_x, gt.target_sensor_y) if (gt and gt.target_visible) else None
            is_locked = (out.tracking_state in (TrackingState.TRACK, TrackingState.DEGRADED))

            self.metrics.record_frame(
                estimated_xy=(out.state_estimate[0], out.state_estimate[1]),
                ground_truth_xy=gt_xy,
                target_visible=(gt.target_visible if gt else False),
                is_locked=is_locked,
                processing_ms=out.total_processing_ms,
                slew_saturated=(gt.slew_saturated if gt else False),
            )

            # 5. Emit to UI
            self.frame_ready.emit(frame, out, gt)

            # Emit summary every 10 frames
            if out.frame_id % 10 == 0:
                summary = self.metrics.compute_summary()
                self.summary_ready.emit(summary)

            # Sleep to match camera frame rate
            elapsed = time.perf_counter() - t_loop_start
            sleep_time = max(0.0, interval - elapsed)
            self.msleep(int(sleep_time * 1000.0))

    def stop(self) -> None:
        self.is_running = False
        self.wait()

    def reset(self) -> None:
        if self.input_adapter:
            self.input_adapter.reset()
        if self.tracker:
            self.tracker.reset()
        self.metrics = MetricsEngine(self.cfg.acceptance)


class MainWindow(QMainWindow):
    """Primary desktop interface for ASTRA-PAT."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        super().__init__()
        self.cfg = config or AppConfig()
        self.setWindowTitle("ASTRA-PAT: Coarse-PAT Simulator & Tracking Benchmark Suite [ISRO SIH26169]")
        self.resize(1380, 850)

        # Style sheet
        self.setStyleSheet("""
            QMainWindow { background-color: #0b0f19; }
            QSplitter::handle { background-color: #1e293b; }
            QStatusBar { background: #0f172a; color: #94a3b8; font-family: monospace; }
        """)

        # Worker
        self.worker = TrackingWorker(self.cfg)
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.summary_ready.connect(self._on_summary_ready)

        self._build_ui()
        self._build_menu()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)

        # Left: Controls
        self.controls = ControlsPanel(self.cfg)
        self.controls.config_changed.connect(self._on_config_changed)
        self.controls.start_clicked.connect(self._start_tracking)
        self.controls.pause_clicked.connect(self._pause_tracking)
        self.controls.reset_clicked.connect(self._reset_tracking)
        self.controls.report_clicked.connect(self._generate_report)
        self.controls.mp4_selected.connect(self._load_mp4)
        splitter.addWidget(self.controls)
        splitter.setStretchFactor(0, 0)

        # Center: Video Display
        v_box = QVBoxLayout()
        self.video_widget = VideoDisplayWidget()
        v_box.addWidget(self.video_widget)
        video_container = QWidget()
        video_container.setLayout(v_box)
        splitter.addWidget(video_container)
        splitter.setStretchFactor(1, 2)

        # Right: Telemetry Plots and Mission Panel
        right_box = QVBoxLayout()
        self.plots_widget = TelemetryPlotsWidget(max_points=120)
        right_box.addWidget(self.plots_widget, 2)

        self.mission_panel = MissionReadinessPanel()
        right_box.addWidget(self.mission_panel, 1)

        right_container = QWidget()
        right_container.setLayout(right_box)
        splitter.addWidget(right_container)
        splitter.setStretchFactor(2, 2)

        main_layout.addWidget(splitter)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("System Ready. Click START to begin closed-loop tracking.")

    def _build_menu(self) -> None:
        bar = self.menuBar()
        bar.setStyleSheet("background: #0f172a; color: #cbd5e1;")

        file_menu = bar.addMenu("Scenario")
        act_load = file_menu.addAction("Load Scenario YAML...")
        act_load.triggered.connect(self._load_scenario_dialog)
        act_save = file_menu.addAction("Save Current Scenario YAML...")
        act_save.triggered.connect(self._save_scenario_dialog)
        file_menu.addSeparator()
        act_exit = file_menu.addAction("Exit")
        act_exit.triggered.connect(self.close)

        test_menu = bar.addMenu("Benchmark")
        act_run_accept = test_menu.addAction("Run All Acceptance Tests (AT-01 to AT-15)...")
        act_run_accept.triggered.connect(self._run_acceptance_dialog)

    def _start_tracking(self) -> None:
        if not self.worker.isRunning():
            self.worker.start()
        self.worker.is_paused = False
        self.status.showMessage("Tracking Loop Active. Closed-loop PTZ operational.")

    def _pause_tracking(self) -> None:
        self.worker.is_paused = True
        self.status.showMessage("Tracking Paused.")

    def _reset_tracking(self) -> None:
        self.worker.reset()
        self.plots_widget.clear()
        self.status.showMessage("Tracking State Reset to Initial.")

    def _on_config_changed(self, new_cfg: AppConfig) -> None:
        self.worker.set_config(new_cfg)

    def _load_mp4(self, path: str) -> None:
        self.worker.stop()
        self.worker.set_mp4_mode(path)
        self.plots_widget.clear()
        self.worker.start()
        self.status.showMessage(f"Benchmark-2 Active: Decoding {os.path.basename(path)}")

    def _on_frame_ready(self, frame: np.ndarray, out: TrackerOutput, gt: Optional[Any]) -> None:
        self.video_widget.update_frame(frame, out)

        # Plot telemetry
        err = -1.0
        if gt and gt.target_visible:
            err = np.hypot(out.state_estimate[0] - gt.target_sensor_x, out.state_estimate[1] - gt.target_sensor_y)
        slew_dps = out.camera_command[0] if out.camera_command else 0.0
        self.plots_widget.add_telemetry(out.timestamp_s, err, out.confidence, slew_dps)

        # Status text
        self.status.showMessage(
            f"Frame: {out.frame_id:04d} | State: {out.tracking_state.value:<9} | "
            f"FPS: {1000.0/max(out.total_processing_ms, 0.1):.1f} | Error: {f'{err:.2f}px' if err >= 0 else 'N/A'}"
        )

    def _on_summary_ready(self, summary: PerformanceSummary) -> None:
        self.mission_panel.update_checklist(summary)

    def _generate_report(self) -> None:
        summary = self.worker.metrics.compute_summary()
        reporter = ReportGenerator(output_dir="results")
        html_path = reporter.build_html_report(summary, scenario_name="Live Session", tracking_csv_path="results/tracking.csv")
        webbrowser.open(os.path.abspath(html_path))
        QMessageBox.information(self, "Report Generated", f"Performance Report successfully exported to:\n{html_path}")

    def _run_acceptance_dialog(self) -> None:
        ans = QMessageBox.question(self, "Run Acceptance Tests", "Run full automated acceptance suite AT-01 through AT-15 now?")
        if ans == QMessageBox.Yes:
            self._pause_tracking()
            suite = AcceptanceTestSuite()
            results = suite.run_all(endurance_seconds=5.0)
            summary_txt = os.path.join(suite.output_dir, "acceptance_summary.txt")
            if os.path.exists(summary_txt):
                with open(summary_txt, "r") as f:
                    txt = f.read()
                QMessageBox.information(self, "Acceptance Test Results", txt)

    def _load_scenario_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Scenario YAML", "scenarios", "YAML Files (*.yaml *.yml)")
        if path:
            cfg = AppConfig.load_yaml(path)
            self.cfg = cfg
            self.worker.set_config(cfg)
            self._reset_tracking()
            QMessageBox.information(self, "Loaded", f"Loaded scenario: {os.path.basename(path)}")

    def _save_scenario_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Scenario YAML", "scenarios/my_scenario.yaml", "YAML Files (*.yaml)")
        if path:
            self.cfg.save_yaml(path)
            QMessageBox.information(self, "Saved", f"Saved scenario to: {os.path.basename(path)}")

    def closeEvent(self, event) -> None:
        self.worker.stop()
        event.accept()


def run_gui() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
