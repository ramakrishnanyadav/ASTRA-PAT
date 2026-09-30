"""Real-time scrolling telemetry plots using pyqtgraph."""

from __future__ import annotations
from typing import Optional, List
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout
import numpy as np


class TelemetryPlotsWidget(QWidget):
    """Scientific scrolling telemetry graphs for error, confidence, and slew rates."""

    def __init__(self, parent: Optional[QWidget] = None, max_points: int = 150) -> None:
        super().__init__(parent)
        self.max_points = max_points

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(4, 4, 4, 4)

        pg.setConfigOption("background", "#0f172a")
        pg.setConfigOption("foreground", "#94a3b8")

        self.win = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.win)

        # Buffers
        self.t_data: List[float] = []
        self.err_data: List[float] = []
        self.conf_data: List[float] = []
        self.slew_data: List[float] = []

        # 1. Error Plot
        self.p_err = self.win.addPlot(title="Tracking Error (px) [SPEC <= 10.0]")
        self.p_err.showGrid(x=True, y=True, alpha=0.3)
        self.curve_err = self.p_err.plot(pen=pg.mkPen("#00e5ff", width=1.5))
        # Spec limit line
        line_spec = pg.InfiniteLine(pos=10.0, angle=0, pen=pg.mkPen("#ef4444", width=1, style=pg.QtCore.Qt.DashLine))
        line_target = pg.InfiniteLine(pos=3.0, angle=0, pen=pg.mkPen("#22c55e", width=1, style=pg.QtCore.Qt.DotLine))
        self.p_err.addItem(line_spec)
        self.p_err.addItem(line_target)
        self.p_err.setYRange(0, 15)

        self.win.nextRow()

        # 2. Confidence Plot
        self.p_conf = self.win.addPlot(title="Beacon Detection Confidence")
        self.p_conf.showGrid(x=True, y=True, alpha=0.3)
        self.curve_conf = self.p_conf.plot(pen=pg.mkPen("#10b981", width=1.5))
        self.p_conf.setYRange(0, 1.0)

        self.win.nextRow()

        # 3. Commanded Slew Rate Plot
        self.p_slew = self.win.addPlot(title="Camera Pan/Tilt Rate Demand (deg/s) [Limit +-5.0]")
        self.p_slew.showGrid(x=True, y=True, alpha=0.3)
        self.curve_slew = self.p_slew.plot(pen=pg.mkPen("#f59e0b", width=1.5))
        self.p_slew.addItem(pg.InfiniteLine(pos=5.0, angle=0, pen=pg.mkPen("#e11d48", width=1, style=pg.QtCore.Qt.DashLine)))
        self.p_slew.addItem(pg.InfiniteLine(pos=-5.0, angle=0, pen=pg.mkPen("#e11d48", width=1, style=pg.QtCore.Qt.DashLine)))
        self.p_slew.setYRange(-6, 6)

    def add_telemetry(self, timestamp: float, error_px: float, confidence: float, slew_dps: float) -> None:
        """Appends new sample and redraws curves."""
        self.t_data.append(timestamp)
        self.err_data.append(error_px if error_px >= 0 else 0.0)
        self.conf_data.append(confidence)
        self.slew_data.append(slew_dps)

        if len(self.t_data) > self.max_points:
            self.t_data.pop(0)
            self.err_data.pop(0)
            self.conf_data.pop(0)
            self.slew_data.pop(0)

        self.curve_err.setData(self.t_data, self.err_data)
        self.curve_conf.setData(self.t_data, self.conf_data)
        self.curve_slew.setData(self.t_data, self.slew_data)

    def clear(self) -> None:
        self.t_data.clear()
        self.err_data.clear()
        self.conf_data.clear()
        self.slew_data.clear()
        self.curve_err.clear()
        self.curve_conf.clear()
        self.curve_slew.clear()
