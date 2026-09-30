"""High-performance scientific video display widget with HUD overlays."""

from __future__ import annotations
from typing import Optional, Tuple
import cv2
import numpy as np
from PySide6.QtCore import Qt, QRect, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout
from tracking.tracker_core import TrackerOutput
from tracking.state_machine import TrackingState


class VideoDisplayWidget(QWidget):
    """Renders monochrome FPA video stream with tracking reticles, state badges, and ROI overlays."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.current_frame: Optional[np.ndarray] = None
        self.latest_output: Optional[TrackerOutput] = None

    def update_frame(self, frame: np.ndarray, output: TrackerOutput) -> None:
        """Updates internal frame buffer and triggers repaint."""
        self.current_frame = frame
        self.latest_output = output
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # 1. Draw black background
        painter.fillRect(0, 0, w, h, QColor(10, 15, 25))

        if self.current_frame is None:
            painter.setPen(QColor(100, 120, 150))
            painter.setFont(QFont("Segoe UI", 14))
            painter.drawText(QRect(0, 0, w, h), Qt.AlignCenter, "ASTRA-PAT: Optical Sensor Inactive")
            return

        # 2. Convert grayscale frame to QPixmap
        fh, fw = self.current_frame.shape[:2]
        bytes_per_line = fw
        qimg = QImage(self.current_frame.data, fw, fh, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg).scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Center in widget
        px_x = (w - pixmap.width()) // 2
        px_y = (h - pixmap.height()) // 2
        painter.drawPixmap(px_x, px_y, pixmap)

        scale_x = pixmap.width() / float(fw)
        scale_y = pixmap.height() / float(fh)

        # 3. Draw Boresight Center Reticle
        cx = px_x + (fw / 2.0) * scale_x
        cy = px_y + (fh / 2.0) * scale_y
        painter.setPen(QPen(QColor(80, 140, 200, 120), 1, Qt.DashLine))
        painter.drawLine(int(cx - 20), int(cy), int(cx + 20), int(cy))
        painter.drawLine(int(cx), int(cy - 20), int(cx), int(cy + 20))

        if self.latest_output is None:
            return

        out = self.latest_output

        # 4. Draw Adaptive ROI
        if out.next_roi is not None:
            rx, ry, rw, rh = out.next_roi
            draw_rx = int(px_x + rx * scale_x)
            draw_ry = int(px_y + ry * scale_y)
            draw_rw = int(rw * scale_x)
            draw_rh = int(rh * scale_y)

            roi_color = QColor(255, 200, 0, 180) if out.tracking_state == TrackingState.TRACK else QColor(255, 80, 80, 180)
            painter.setPen(QPen(roi_color, 1, Qt.DotLine))
            painter.drawRect(draw_rx, draw_ry, draw_rw, draw_rh)

        # 5. Draw Beacon Marker
        if out.detection.valid:
            bx = int(px_x + out.detection.x * scale_x)
            by = int(px_y + out.detection.y * scale_y)

            # Cyan target tracking bracket
            painter.setPen(QPen(QColor(0, 230, 255), 2))
            s = 10
            # Four corner brackets
            painter.drawLine(bx - s, by - s, bx - s + 4, by - s)
            painter.drawLine(bx - s, by - s, bx - s, by - s + 4)

            painter.drawLine(bx + s, by - s, bx + s - 4, by - s)
            painter.drawLine(bx + s, by - s, bx + s, by - s + 4)

            painter.drawLine(bx - s, by + s, bx - s + 4, by + s)
            painter.drawLine(bx - s, by + s, bx - s, by + s - 4)

            painter.drawLine(bx + s, by + s, bx + s - 4, by + s)
            painter.drawLine(bx + s, by + s, bx + s, by + s - 4)

            # Sub-pixel center dot
            painter.setBrush(QColor(0, 255, 180))
            painter.drawEllipse(QPoint(bx, by), 2, 2)

        # 6. HUD State Badge
        state_str = out.tracking_state.value
        state_colors = {
            "TRACK": QColor(40, 180, 80),
            "ACQUIRE": QColor(220, 160, 20),
            "DEGRADED": QColor(200, 100, 20),
            "PREDICT": QColor(140, 80, 220),
            "REACQUIRE": QColor(220, 40, 60),
            "SEARCH": QColor(80, 120, 160),
            "FAILSAFE": QColor(200, 20, 40),
        }
        badge_col = state_colors.get(state_str, QColor(100, 100, 100))

        painter.setBrush(badge_col)
        painter.setPen(Qt.NoPen)
        badge_w, badge_h = 100, 26
        painter.drawRoundedRect(px_x + 12, px_y + 12, badge_w, badge_h, 4, 4)

        painter.setPen(Qt.white)
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.drawText(QRect(px_x + 12, px_y + 12, badge_w, badge_h), Qt.AlignCenter, state_str)

        # 7. Telemetry Overlay
        painter.setFont(QFont("Consolas", 9))
        painter.setPen(QColor(220, 230, 240))
        fps_text = f"{1000.0 / max(out.total_processing_ms, 0.1):.1f} FPS ({out.total_processing_ms:.1f}ms)"
        conf_text = f"Conf: {out.confidence:.2f}"
        pos_text = f"Pos: ({out.state_estimate[0]:.1f}, {out.state_estimate[1]:.1f})"
        
        painter.drawText(px_x + 12, px_y + 60, fps_text)
        painter.drawText(px_x + 12, px_y + 75, conf_text)
        painter.drawText(px_x + 12, px_y + 90, pos_text)
