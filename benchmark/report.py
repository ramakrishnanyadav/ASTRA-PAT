"""Automated performance report generator producing HTML and telemetry plots."""

from __future__ import annotations
import base64
import io
import os
from typing import Dict, Any, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from benchmark.metrics import PerformanceSummary


class ReportGenerator:
    """Generates standalone visual HTML reports and analytical plots for ASTRA-PAT runs."""

    def __init__(self, output_dir: str = "results") -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_plots(self, tracking_csv_path: str) -> Dict[str, str]:
        """Reads tracking.csv and produces PNG plots, returning base64 encoded strings."""
        if not os.path.exists(tracking_csv_path):
            return {}

        df = pd.read_csv(tracking_csv_path)
        plot_b64: Dict[str, str] = {}

        # 1. Error vs Time Plot
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        valid_errs = df[df["error_px"] >= 0]
        if not valid_errs.empty:
            ax.plot(valid_errs["timestamp_s"], valid_errs["error_px"], label="Centroid Error (px)", color="#00bcd4", lw=1.5)
            ax.axhline(10.0, color="#f44336", linestyle="--", lw=1.5, label="SPEC Limit (10.0 px)")
            ax.axhline(3.0, color="#4caf50", linestyle=":", lw=1.5, label="Target Margin (3.0 px)")
        ax.set_title("Centroid Tracking Error vs Time", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Error (pixels)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        plt.tight_layout()

        err_plot_path = os.path.join(self.output_dir, "error_vs_time.png")
        fig.savefig(err_plot_path)

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        plot_b64["error_plot"] = base64.b64encode(buf.read()).decode("utf-8")

        # 2. Camera Rates & Slew Saturation
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        ax.plot(df["timestamp_s"], df["cmd_pan_dps"], label="Cmd Pan (deg/s)", color="#2196f3", lw=1.2)
        ax.plot(df["timestamp_s"], df["cmd_tilt_dps"], label="Cmd Tilt (deg/s)", color="#ff9800", lw=1.2)
        ax.axhline(5.0, color="#e91e63", linestyle="--", alpha=0.7, label="Max Slew (+-5 deg/s)")
        ax.axhline(-5.0, color="#e91e63", linestyle="--", alpha=0.7)
        ax.set_title("Actuator Slew Rate Demands vs Limits", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Rate (deg/s)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        plt.tight_layout()

        slew_plot_path = os.path.join(self.output_dir, "rates_vs_time.png")
        fig.savefig(slew_plot_path)

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        plot_b64["rates_plot"] = base64.b64encode(buf.read()).decode("utf-8")

        return plot_b64

    def build_html_report(
        self,
        summary: PerformanceSummary,
        scenario_name: str = "Coarse-PAT Mission Run",
        tracking_csv_path: Optional[str] = None,
    ) -> str:
        """Generates self-contained HTML report."""
        plots = {}
        if tracking_csv_path:
            plots = self.generate_plots(tracking_csv_path)

        status_badge = (
            '<span style="background:#4caf50;color:white;padding:4px 12px;border-radius:4px;font-weight:bold;">MISSION READY (ALL SPECS PASSED)</span>'
            if summary.all_spec_passed else
            '<span style="background:#f44336;color:white;padding:4px 12px;border-radius:4px;font-weight:bold;">SPEC VIOLATION DETECTED</span>'
        )

        rows = ""
        for item in summary.checklist:
            badge_color = "#4caf50" if item.status == "PASS" else "#f44336"
            target_badge = '<span style="color:#4caf50;font-weight:bold;">YES</span>' if item.target_met else '<span style="color:#ff9800;">MARGINAL</span>'
            rows += f"""
            <tr>
                <td style="padding:10px;border-bottom:1px solid #ddd;font-weight:bold;">{item.name}</td>
                <td style="padding:10px;border-bottom:1px solid #ddd;">{item.spec_threshold}</td>
                <td style="padding:10px;border-bottom:1px solid #ddd;">{item.target_threshold}</td>
                <td style="padding:10px;border-bottom:1px solid #ddd;font-family:monospace;">{item.measured_val}</td>
                <td style="padding:10px;border-bottom:1px solid #ddd;"><span style="background:{badge_color};color:white;padding:2px 8px;border-radius:3px;font-size:12px;">{item.status}</span></td>
                <td style="padding:10px;border-bottom:1px solid #ddd;">{target_badge}</td>
            </tr>
            """

        err_img_tag = f'<img src="data:image/png;base64,{plots.get("error_plot")}" style="width:100%;border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,0.1);margin-top:15px;"/>' if "error_plot" in plots else ""
        rates_img_tag = f'<img src="data:image/png;base64,{plots.get("rates_plot")}" style="width:100%;border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,0.1);margin-top:15px;"/>' if "rates_plot" in plots else ""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ASTRA-PAT Benchmark Performance Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 25px; background: #f8fafc; color: #1e293b; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}
        h1 {{ margin-top: 0; color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 15px; font-size: 24px; }}
        h2 {{ color: #334155; margin-top: 30px; font-size: 18px; border-left: 4px solid #0ea5e9; padding-left: 10px; }}
        .header-meta {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 14px; }}
        th {{ background: #f1f5f9; padding: 12px 10px; text-align: left; font-weight: 600; color: #475569; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 20px; }}
        .card {{ background: #f8fafc; border: 1px solid #e2e8f0; padding: 15px; border-radius: 6px; text-align: center; }}
        .card-val {{ font-size: 22px; font-weight: bold; color: #0284c7; margin-top: 5px; }}
        .card-label {{ font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }}
        .disclaimer {{ background: #fffbeb; border: 1px solid #fef3c7; color: #92400e; padding: 12px; border-radius: 6px; font-size: 13px; margin-top: 30px; }}
    </style>
</head>
<body>
<div class="container">
    <div class="header-meta">
        <div>
            <h1>ASTRA-PAT: Coarse-PAT Performance Report</h1>
            <div style="color:#64748b;font-size:14px;">Scenario: <strong>{scenario_name}</strong> | Total Frames: {summary.total_frames}</div>
        </div>
        <div>{status_badge}</div>
    </div>

    <div class="summary-grid">
        <div class="card">
            <div class="card-label">RMSE Tracking Error</div>
            <div class="card-val">{summary.error_rmse_px:.2f} px</div>
        </div>
        <div class="card">
            <div class="card-label">Mean Frame Rate</div>
            <div class="card-val">{summary.mean_fps:.1f} FPS</div>
        </div>
        <div class="card">
            <div class="card-label">Lock Retention</div>
            <div class="card-val">{summary.lock_retention_pct:.1f}%</div>
        </div>
        <div class="card">
            <div class="card-label">Acquisition Time</div>
            <div class="card-val">{f"{summary.acquisition_time_s:.2f} s" if summary.acquisition_time_s else "N/A"}</div>
        </div>
    </div>

    <h2>Mission Readiness Verification</h2>
    <table>
        <thead>
            <tr>
                <th>Benchmark Parameter</th>
                <th>Spec Threshold</th>
                <th>Target Margin</th>
                <th>Measured Value</th>
                <th>Spec Status</th>
                <th>Target Margin</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>

    <h2>Centroid Telemetry & Actuator Slew</h2>
    {err_img_tag}
    {rates_img_tag}

    <div class="disclaimer">
        <strong>Atmospheric Model Transparency Notice:</strong> Atmospheric effects shown in simulation (haze, fog, rain, low light) are image-space approximations (contrast attenuation, blur, synthetic streaks, brightness attenuation), NOT physical radiative transfer or radiometric propagation simulations.
    </div>
</div>
</body>
</html>"""

        report_path = os.path.join(self.output_dir, "report.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)

        return report_path
