"""Asynchronous Flight Recorder for ASTRA-PAT telemetry.

ENGINEERING PRINCIPLE (Per Specification):
"The tracking loop must never block on disk I/O: bounded telemetry queue plus
a separate logger thread. Benchmark the pipeline, not the filesystem."

This module implements a decoupled, high-throughput flight recorder that drains
telemetry from a thread-safe bounded queue to disk using a background daemon thread,
ensuring strict zero-latency impact on real-time computer vision processing.
"""

from __future__ import annotations
import csv
import os
import queue
import threading
import time
from typing import Dict, Any, Optional, List


class FlightRecorder:
    """Non-blocking telemetry recorder using a separate background writer thread."""

    def __init__(
        self,
        output_dir: str = "results",
        queue_size: int = 5000,
        batch_size: int = 50,
        flush_interval_s: float = 0.5,
    ) -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._batch_size = batch_size
        self._flush_interval_s = flush_interval_s

        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        self._centroid_file = None
        self._centroid_writer: Optional[csv.DictWriter] = None
        self._tracking_file = None
        self._tracking_writer: Optional[csv.DictWriter] = None

        self.records_logged: int = 0
        self.records_dropped: int = 0

    def start(self) -> None:
        """Starts the background disk writer thread."""
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._writer_loop,
            name="FlightRecorderThread",
            daemon=True,
        )
        self._worker_thread.start()

    def record_centroid(self, record: Dict[str, Any]) -> None:
        """Enqueues a canonical centroid telemetry record non-blockingly."""
        try:
            self._queue.put_nowait(("centroid", record))
        except queue.Full:
            self.records_dropped += 1

    def record_tracking(self, record: Dict[str, Any]) -> None:
        """Enqueues a full tracking kinematic record non-blockingly."""
        try:
            self._queue.put_nowait(("tracking", record))
        except queue.Full:
            self.records_dropped += 1

    def flush_and_close(self, timeout_s: float = 5.0) -> None:
        """Flushes all queued telemetry to disk and cleanly closes file handles."""
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout_s)

        # Drain any remaining elements directly on caller thread
        self._drain_remaining()

        # Close open file descriptors
        if self._centroid_file:
            try:
                self._centroid_file.flush()
                self._centroid_file.close()
            except Exception:
                pass
            self._centroid_file = None

        if self._tracking_file:
            try:
                self._tracking_file.flush()
                self._tracking_file.close()
            except Exception:
                pass
            self._tracking_file = None

    def _writer_loop(self) -> None:
        """Background thread loop draining telemetry records to CSV."""
        last_flush = time.perf_counter()

        while not self._stop_event.is_set():
            try:
                rec_type, data = self._queue.get(timeout=0.05)
                self._write_single_record(rec_type, data)
                self.records_logged += 1
                self._queue.task_done()
            except queue.Empty:
                pass

            now = time.perf_counter()
            if now - last_flush >= self._flush_interval_s:
                self._flush_files()
                last_flush = now

    def _write_single_record(self, rec_type: str, data: Dict[str, Any]) -> None:
        if rec_type == "centroid":
            if self._centroid_writer is None:
                c_path = os.path.join(self.output_dir, "centroid.csv")
                self._centroid_file = open(c_path, "w", newline="", encoding="utf-8")
                self._centroid_writer = csv.DictWriter(self._centroid_file, fieldnames=list(data.keys()))
                self._centroid_writer.writeheader()
            self._centroid_writer.writerow(data)

        elif rec_type == "tracking":
            if self._tracking_writer is None:
                t_path = os.path.join(self.output_dir, "tracking.csv")
                self._tracking_file = open(t_path, "w", newline="", encoding="utf-8")
                self._tracking_writer = csv.DictWriter(self._tracking_file, fieldnames=list(data.keys()))
                self._tracking_writer.writeheader()
            self._tracking_writer.writerow(data)

    def _flush_files(self) -> None:
        if self._centroid_file:
            self._centroid_file.flush()
        if self._tracking_file:
            self._tracking_file.flush()

    def _drain_remaining(self) -> None:
        while not self._queue.empty():
            try:
                rec_type, data = self._queue.get_nowait()
                self._write_single_record(rec_type, data)
                self.records_logged += 1
                self._queue.task_done()
            except queue.Empty:
                break
        self._flush_files()
