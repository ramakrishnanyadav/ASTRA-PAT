"""Mission Readiness Dashboard Widget displaying live spec compliance."""

from __future__ import annotations
from typing import Optional, List
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView, QLabel, QFrame
)
from benchmark.metrics import PerformanceSummary, MetricThresholdStatus


class MissionReadinessPanel(QWidget):
    """Real-time mission-readiness verification panel."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("MISSION-READINESS VERIFICATION PANEL")
        title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        title.setStyleSheet("color: #38bdf8; letter-spacing: 1px; margin-bottom: 4px;")
        layout.addWidget(title)

        self.table = QTableWidget(5, 5)
        self.table.setHorizontalHeaderLabels([
            "Parameter", "Spec Limit", "Target Margin", "Measured Value", "Status"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #1e293b;
                color: #e2e8f0;
                gridline-color: #334155;
                border: 1px solid #334155;
                border-radius: 4px;
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #0f172a;
                color: #94a3b8;
                padding: 6px;
                font-weight: bold;
                border: none;
                border-bottom: 1px solid #334155;
            }
        """)
        layout.addWidget(self.table)

    def update_checklist(self, summary: PerformanceSummary) -> None:
        """Updates table rows from latest performance summary."""
        self.table.setRowCount(len(summary.checklist))
        for r_idx, item in enumerate(summary.checklist):
            p_item = QTableWidgetItem(item.name)
            s_item = QTableWidgetItem(item.spec_threshold)
            t_item = QTableWidgetItem(item.target_threshold)
            m_item = QTableWidgetItem(item.measured_val)
            st_item = QTableWidgetItem(item.status)

            st_item.setTextAlignment(Qt.AlignCenter)
            if item.status == "PASS":
                st_item.setForeground(QColor("#4ade80"))
                st_item.setFont(QFont("Segoe UI", 11, QFont.Bold))
            else:
                st_item.setForeground(QColor("#f87171"))
                st_item.setFont(QFont("Segoe UI", 11, QFont.Bold))

            self.table.setItem(r_idx, 0, p_item)
            self.table.setItem(r_idx, 1, s_item)
            self.table.setItem(r_idx, 2, t_item)
            self.table.setItem(r_idx, 3, m_item)
            self.table.setItem(r_idx, 4, st_item)
