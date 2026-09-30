"""Interactive controls and disturbance injection panel."""

from __future__ import annotations
from typing import Optional, Callable
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QComboBox,
    QSlider, QDoubleSpinBox, QCheckBox, QPushButton, QFileDialog
)
from app.config import AppConfig


class ControlsPanel(QWidget):
    """Scientific controls panel for configuring simulation and injecting disturbances in real time."""

    config_changed = Signal(AppConfig)
    start_clicked = Signal()
    pause_clicked = Signal()
    reset_clicked = Signal()
    report_clicked = Signal()
    mp4_selected = Signal(str)

    def __init__(self, config: AppConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.cfg = config

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.setSpacing(10)

        # Style
        self.setStyleSheet("""
            QGroupBox {
                border: 1px solid #334155;
                border-radius: 6px;
                margin-top: 12px;
                font-weight: bold;
                color: #38bdf8;
                padding-top: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QLabel { color: #cbd5e1; font-size: 12px; }
            QComboBox, QDoubleSpinBox {
                background: #1e293b;
                color: white;
                border: 1px solid #475569;
                border-radius: 4px;
                padding: 3px 6px;
                font-size: 12px;
            }
            QPushButton {
                background: #0284c7;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background: #0369a1; }
            QPushButton#btn_report { background: #16a34a; }
            QPushButton#btn_report:hover { background: #15803d; }
            QPushButton#btn_reset { background: #dc2626; }
            QPushButton#btn_reset:hover { background: #b91c1c; }
        """)

        self._build_mode_section()
        self._build_target_section()
        self._build_disturbance_section()
        self._build_action_buttons()

    def _build_mode_section(self) -> None:
        grp = QGroupBox("Operating Mode")
        vbox = QVBoxLayout(grp)

        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Closed-Loop Simulation (Virtual PTZ)", "Benchmark-2 (MP4 Full-Frame Decoder)"])
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        vbox.addWidget(self.combo_mode)

        self.btn_select_mp4 = QPushButton("Open MP4 Video File...")
        self.btn_select_mp4.setEnabled(False)
        self.btn_select_mp4.clicked.connect(self._select_mp4_file)
        vbox.addWidget(self.btn_select_mp4)

        self.layout.addWidget(grp)

    def _build_target_section(self) -> None:
        grp = QGroupBox("Target Kinematics")
        vbox = QVBoxLayout(grp)

        # Motion Type
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Trajectory:"))
        self.combo_motion = QComboBox()
        self.combo_motion.addItems(["figure8", "circular", "straight", "random", "sinusoidal", "spiral"])
        self.combo_motion.setCurrentText(self.cfg.target.motion_type)
        self.combo_motion.currentTextChanged.connect(self._update_config)
        h1.addWidget(self.combo_motion)
        vbox.addLayout(h1)

        # Speed
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Speed (px/s):"))
        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(20.0, 300.0)
        self.spin_speed.setValue(self.cfg.target.speed_px_s)
        self.spin_speed.valueChanged.connect(self._update_config)
        h2.addWidget(self.spin_speed)
        vbox.addLayout(h2)

        # Target Size
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Size (px):"))
        self.spin_size = QDoubleSpinBox()
        self.spin_size.setRange(5.0, 20.0)
        self.spin_size.setValue(self.cfg.target.size_px)
        self.spin_size.valueChanged.connect(self._update_config)
        h3.addWidget(self.spin_size)
        vbox.addLayout(h3)

        self.layout.addWidget(grp)

    def _build_disturbance_section(self) -> None:
        grp = QGroupBox("Disturbances & Environment")
        vbox = QVBoxLayout(grp)

        # Atmosphere
        h_atm = QHBoxLayout()
        h_atm.addWidget(QLabel("Atmosphere:"))
        self.combo_atm = QComboBox()
        self.combo_atm.addItems(["clear", "haze", "fog", "rain", "low_light"])
        self.combo_atm.setCurrentText(self.cfg.disturbances.atmosphere)
        self.combo_atm.currentTextChanged.connect(self._update_config)
        h_atm.addWidget(self.combo_atm)
        vbox.addLayout(h_atm)

        # Gaussian Noise
        h_noise = QHBoxLayout()
        h_noise.addWidget(QLabel("Gaussian Sigma:"))
        self.slider_noise = QSlider(Qt.Horizontal)
        self.slider_noise.setRange(0, 20)
        self.slider_noise.setValue(int(self.cfg.disturbances.gaussian_sigma_px))
        self.slider_noise.valueChanged.connect(self._update_config)
        h_noise.addWidget(self.slider_noise)
        vbox.addLayout(h_noise)

        # 10% Salt & Pepper toggle
        self.chk_salt_pepper = QCheckBox("Inject 10% Salt & Pepper Noise")
        self.chk_salt_pepper.setChecked(self.cfg.disturbances.salt_pepper_ratio > 0.05)
        self.chk_salt_pepper.toggled.connect(self._update_config)
        vbox.addWidget(self.chk_salt_pepper)

        # Camera Jitter
        h_jit = QHBoxLayout()
        h_jit.addWidget(QLabel("Camera Jitter (px):"))
        self.slider_jitter = QSlider(Qt.Horizontal)
        self.slider_jitter.setRange(0, 20)
        self.slider_jitter.setValue(int(self.cfg.disturbances.camera_jitter_px))
        self.slider_jitter.valueChanged.connect(self._update_config)
        h_jit.addWidget(self.slider_jitter)
        vbox.addLayout(h_jit)

        # Platform Motion
        h_plat = QHBoxLayout()
        h_plat.addWidget(QLabel("Platform Drift (px):"))
        self.slider_plat = QSlider(Qt.Horizontal)
        self.slider_plat.setRange(0, 20)
        self.slider_plat.setValue(int(self.cfg.disturbances.platform_motion_amp_px))
        self.slider_plat.valueChanged.connect(self._update_config)
        h_plat.addWidget(self.slider_plat)
        vbox.addLayout(h_plat)

        self.layout.addWidget(grp)

    def _build_action_buttons(self) -> None:
        btn_box = QVBoxLayout()

        h_ctrl = QHBoxLayout()
        self.btn_run = QPushButton("START")
        self.btn_run.clicked.connect(self.start_clicked.emit)
        h_ctrl.addWidget(self.btn_run)

        self.btn_pause = QPushButton("PAUSE")
        self.btn_pause.clicked.connect(self.pause_clicked.emit)
        h_ctrl.addWidget(self.btn_pause)

        self.btn_reset = QPushButton("RESET")
        self.btn_reset.setObjectName("btn_reset")
        self.btn_reset.clicked.connect(self.reset_clicked.emit)
        h_ctrl.addWidget(self.btn_reset)
        btn_box.addLayout(h_ctrl)

        self.btn_report = QPushButton("GENERATE PERFORMANCE REPORT")
        self.btn_report.setObjectName("btn_report")
        self.btn_report.clicked.connect(self.report_clicked.emit)
        btn_box.addWidget(self.btn_report)

        self.layout.addLayout(btn_box)

    def _on_mode_changed(self, index: int) -> None:
        is_mp4 = (index == 1)
        self.btn_select_mp4.setEnabled(is_mp4)

    def _select_mp4_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Open MP4 Benchmark Video", "", "Video Files (*.mp4)")
        if file_path:
            self.mp4_selected.emit(file_path)

    def _update_config(self) -> None:
        self.cfg.target.motion_type = self.combo_motion.currentText()  # type: ignore
        self.cfg.target.speed_px_s = float(self.spin_speed.value())
        self.cfg.target.size_px = float(self.spin_size.value())

        self.cfg.disturbances.atmosphere = self.combo_atm.currentText()  # type: ignore
        self.cfg.disturbances.gaussian_sigma_px = float(self.slider_noise.value())
        self.cfg.disturbances.salt_pepper_ratio = 0.10 if self.chk_salt_pepper.isChecked() else 0.0
        self.cfg.disturbances.camera_jitter_px = float(self.slider_jitter.value())
        self.cfg.disturbances.platform_motion_amp_px = float(self.slider_plat.value())
        if self.slider_plat.value() > 0:
            self.cfg.disturbances.platform_motion_type = "linear"
        else:
            self.cfg.disturbances.platform_motion_type = "none"

        self.config_changed.emit(self.cfg)
