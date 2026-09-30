"""Command Line Interface (CLI) entry point for ASTRA-PAT."""

from __future__ import annotations
import argparse
import os
import sys

# Ensure repository root is on sys.path
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from app.config import AppConfig
from benchmark.evaluator import BenchmarkEvaluator
from benchmark.report import ReportGenerator
from benchmark.acceptance import AcceptanceTestSuite


def run_benchmark_cli(scenario_file: str, duration: float, output_dir: str) -> int:
    """Executes a scenario headless and generates reports."""
    print(f"\n[ASTRA-PAT] Loading scenario from: {scenario_file}")
    if not os.path.exists(scenario_file):
        print(f"Error: Scenario file '{scenario_file}' not found.")
        return 1

    cfg = AppConfig.load_yaml(scenario_file)
    evaluator = BenchmarkEvaluator(cfg, output_dir=output_dir)
    print(f"[ASTRA-PAT] Running closed-loop tracking benchmark for {duration:.1f}s...")
    summary = evaluator.run_simulation(duration_s=duration)

    print(f"\n==================== BENCHMARK RESULTS ====================")
    print(f"Total Frames Processed : {summary.total_frames}")
    print(f"Tracking RMSE Error    : {summary.error_rmse_px:.3f} px (SPEC: <= 10.0 px)")
    print(f"Lock Retention Rate    : {summary.lock_retention_pct:.1f}%")
    print(f"Target Loss Rate       : {summary.target_loss_pct:.1f}% (SPEC: < 5.0%)")
    print(f"Throughput             : {summary.mean_fps:.1f} FPS (SPEC: >= 20.0 FPS)")
    if summary.acquisition_time_s:
        print(f"Acquisition Time       : {summary.acquisition_time_s:.3f} s (SPEC: <= 2.0 s)")
    if summary.reacquisition_time_s:
        print(f"Reacquisition Time     : {summary.reacquisition_time_s:.3f} s (SPEC: <= 1.0 s)")
    print(f"Slew Limit Violations  : {summary.slew_limit_saturations}")
    print(f"Mission Status         : {'PASS (ALL SPECS MET)' if summary.all_spec_passed else 'FAIL'}")
    print(f"===========================================================\n")

    # Generate HTML report and telemetry graphs
    tracking_csv = os.path.join(output_dir, "tracking.csv")
    reporter = ReportGenerator(output_dir=output_dir)
    html_path = reporter.build_html_report(
        summary=summary,
        scenario_name=os.path.basename(scenario_file),
        tracking_csv_path=tracking_csv
    )
    print(f"[ASTRA-PAT] Report generated at: {html_path}")
    print(f"[ASTRA-PAT] Artifacts saved in : {output_dir}\n")
    return 0 if summary.all_spec_passed else 2


def run_mp4_cli(video_file: str, output_dir: str) -> int:
    """Executes Benchmark-2 on an external MP4 video file."""
    print(f"\n[ASTRA-PAT] Running Benchmark-2 on: {video_file}")
    if not os.path.exists(video_file):
        print(f"Error: Video file '{video_file}' not found.")
        return 1

    cfg = AppConfig()
    evaluator = BenchmarkEvaluator(cfg, output_dir=output_dir)
    summary = evaluator.run_mp4_benchmark(video_file)

    print(f"\n==================== MP4 BENCHMARK RESULTS ====================")
    print(f"Total Frames Processed : {summary.total_frames}")
    print(f"Throughput             : {summary.mean_fps:.1f} FPS (SPEC: >= 20.0 FPS)")
    print(f"Mean Processing Time   : {summary.mean_processing_ms:.2f} ms")
    print(f"Centroid Stream Output : {os.path.join(output_dir, 'centroid.csv')}")
    print(f"===============================================================\n")
    return 0


def run_accept_cli(endurance_s: float) -> int:
    """Executes all 15 automated acceptance tests."""
    suite = AcceptanceTestSuite()
    results = suite.run_all(endurance_seconds=endurance_s)
    all_ok = all(r.passed for r in results)
    return 0 if all_ok else 1


def run_ablation_cli(duration_s: float, output_dir: str) -> int:
    """Executes the 4-configuration ablation study (Configs A, B, C, D)."""
    from benchmark.ablation import AblationStudyRunner
    print(f"\n[ASTRA-PAT] Initiating 4-Configuration Ablation Study...")
    runner = AblationStudyRunner(output_dir=output_dir)
    results = runner.run_all(duration_s=duration_s)

    print("\n" + "=" * 95)
    print("                    ASTRA-PAT ABLATION STUDY RESULTS                    ")
    print("=" * 95)
    print(f"{'Config':<12} {'Description':<42} {'RMSE (px)':<12} {'Lock (%)':<10} {'FPS'}")
    print("-" * 95)
    for r in results:
        print(f"{r.config_id:<12} {r.name:<42} {r.error_rmse_px:<12.2f} {r.lock_retention_pct:<10.1f} {r.mean_fps:.1f}")
    print("=" * 95)
    print(f"[ASTRA-PAT] Ablation summary JSON saved: {os.path.join(output_dir, 'ablation_summary.json')}")
    print(f"[ASTRA-PAT] Ablation report Markdown saved: {os.path.join(output_dir, 'ablation_report.md')}\n")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="astra-pat",
        description="ASTRA-PAT: Coarse-PAT Simulator and Benchmark Suite for ISRO SIH26169",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run scenario headless benchmark")
    p_bench.add_argument("scenario", type=str, help="Path to scenario YAML config")
    p_bench.add_argument("--duration", type=float, default=10.0, help="Simulation duration in seconds")
    p_bench.add_argument("--out", type=str, default="results", help="Output directory for artifacts")

    # mp4
    p_mp4 = subparsers.add_parser("mp4", help="Run Benchmark-2 on MP4 video")
    p_mp4.add_argument("video", type=str, help="Path to .mp4 video file")
    p_mp4.add_argument("--out", type=str, default="results/mp4", help="Output directory")

    # accept
    p_acc = subparsers.add_parser("accept", help="Run automated acceptance test suite AT-01 to AT-15")
    p_acc.add_argument("--endurance", type=float, default=10.0, help="Duration for AT-15 endurance test")

    # ablation
    p_abl = subparsers.add_parser("ablation", help="Run ablation study across Configurations A, B, C, D")
    p_abl.add_argument("--duration", type=float, default=5.0, help="Duration per configuration in seconds")
    p_abl.add_argument("--out", type=str, default="results/ablation", help="Output directory for reports")

    # gui
    subparsers.add_parser("gui", help="Launch interactive PySide6 desktop GUI")

    args = parser.parse_args()

    if args.command == "benchmark":
        sys.exit(run_benchmark_cli(args.scenario, args.duration, args.out))
    elif args.command == "mp4":
        sys.exit(run_mp4_cli(args.video, args.out))
    elif args.command == "accept":
        sys.exit(run_accept_cli(args.endurance))
    elif args.command == "ablation":
        sys.exit(run_ablation_cli(args.duration, args.out))
    elif args.command == "gui" or args.command is None:
        from gui.main_window import run_gui
        run_gui()


if __name__ == "__main__":
    main()

