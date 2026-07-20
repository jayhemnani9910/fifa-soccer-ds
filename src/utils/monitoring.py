"""Monitoring and observability utilities for FIFA Soccer DS.

This module provides:
- Prometheus metrics collection for pipeline performance
- GPU memory and utilization monitoring
- Structured JSON logging for log aggregation
- Performance timing decorators
"""

from __future__ import annotations

import importlib
import json
import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

try:
    torch: Any = importlib.import_module("torch")
except ImportError:
    torch = None

try:
    psutil: Any = importlib.import_module("psutil")
except ImportError:
    psutil = None

try:
    prometheus: Any = importlib.import_module("prometheus_client")
except ImportError:
    prometheus = None

LOGGER = logging.getLogger(__name__)


class _NoopMetric:
    """Metric-compatible sink used when Prometheus is not installed."""

    def inc(self, amount: float = 1.0) -> None:
        del amount

    def set(self, value: float) -> None:
        del value

    def observe(self, value: float) -> None:
        del value


# Prometheus metrics. These names always exist so optional observability can
# never prevent the core pipeline from importing.
if prometheus is not None:
    # Counters
    FRAMES_PROCESSED = prometheus.Counter("fifa_frames_processed_total", "Total frames processed")
    DETECTIONS_MADE = prometheus.Counter("fifa_detections_total", "Total detections made")
    TRACKS_CREATED = prometheus.Counter("fifa_tracks_created_total", "Total tracks created")

    # Gauges
    ACTIVE_TRACKS = prometheus.Gauge("fifa_active_tracks", "Number of currently active tracks")
    GPU_MEMORY_USED = prometheus.Gauge("fifa_gpu_memory_bytes", "GPU memory usage in bytes")
    CPU_USAGE = prometheus.Gauge("fifa_cpu_usage_percent", "CPU usage percentage")

    # Histograms
    PROCESSING_TIME = prometheus.Histogram(
        "fifa_frame_processing_seconds", "Time spent processing each frame"
    )
    DETECTION_TIME = prometheus.Histogram("fifa_detection_seconds", "Time spent on detection")
    TRACKING_TIME = prometheus.Histogram("fifa_tracking_seconds", "Time spent on tracking")
else:
    FRAMES_PROCESSED = _NoopMetric()
    DETECTIONS_MADE = _NoopMetric()
    TRACKS_CREATED = _NoopMetric()
    ACTIVE_TRACKS = _NoopMetric()
    GPU_MEMORY_USED = _NoopMetric()
    CPU_USAGE = _NoopMetric()
    PROCESSING_TIME = _NoopMetric()
    DETECTION_TIME = _NoopMetric()
    TRACKING_TIME = _NoopMetric()


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "track_id"):
            log_entry["track_id"] = record.track_id
        if hasattr(record, "frame_id"):
            log_entry["frame_id"] = record.frame_id
        if hasattr(record, "detection_count"):
            log_entry["detection_count"] = record.detection_count

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def setup_json_logging(level: str = "INFO") -> None:
    """Configure structured JSON logging."""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


def get_gpu_memory() -> int:
    """Get current GPU memory usage in bytes."""
    if torch is None or not torch.cuda.is_available():
        return 0

    return torch.cuda.memory_allocated()


def get_cpu_usage() -> float:
    """Get current CPU usage percentage."""
    if psutil is None:
        return 0.0

    return psutil.cpu_percent()


def get_system_metrics() -> dict:
    """Get system metrics for health checks.

    Returns:
        Dict containing CPU usage, memory info, and GPU memory.
    """
    metrics = {
        "cpu_usage_percent": get_cpu_usage(),
        "gpu_memory_bytes": get_gpu_memory(),
    }

    if psutil is not None:
        mem = psutil.virtual_memory()
        metrics["memory_total_bytes"] = mem.total
        metrics["memory_used_bytes"] = mem.used
        metrics["memory_percent"] = mem.percent

    return metrics


def get_gpu_memory_usage() -> int:
    """Get GPU memory usage in bytes.

    Returns:
        GPU memory usage in bytes, or 0 if GPU unavailable.
    """
    return get_gpu_memory()


def update_system_metrics() -> None:
    """Update system-level metrics."""
    GPU_MEMORY_USED.set(get_gpu_memory())
    CPU_USAGE.set(get_cpu_usage())


def timed(metric_histogram: Any | None = None):
    """Decorator to time function execution and record to Prometheus."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                if metric_histogram:
                    metric_histogram.observe(duration)

                # Log timing with structured logging
                LOGGER.info(
                    "Function %s completed",
                    func.__name__,
                    extra={
                        "func_name": func.__name__,
                        "duration": duration,
                        "func_module": func.__module__,
                    },
                )

        return wrapper

    return decorator


def start_metrics_server(port: int = 8000) -> None:
    """Start Prometheus metrics HTTP server."""
    if prometheus is not None:
        LOGGER.info("Starting Prometheus metrics server on port %d", port)
        prometheus.start_http_server(port)
    else:
        LOGGER.warning("prometheus_client not available, metrics server not started")


# Convenience decorators remain functional even when metrics are disabled.
timed_processing = timed(PROCESSING_TIME)
timed_detection = timed(DETECTION_TIME)
timed_tracking = timed(TRACKING_TIME)
