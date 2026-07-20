"""Threaded RTSP capture utility with automatic reconnection."""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
from collections.abc import Iterator
from typing import Any

import cv2

from src.utils.network_security import redact_url_credentials

LOGGER = logging.getLogger(__name__)


class RTSPCapture:
    """Background RTSP frame reader with resilience to connection drops."""

    def __init__(
        self,
        url: str,
        reconnect_interval: float = 5.0,
        max_queue_size: int = 32,
    ) -> None:
        self.url = url
        self.reconnect_interval = reconnect_interval
        self.max_queue_size = max_queue_size
        if reconnect_interval <= 0 or max_queue_size < 1:
            raise ValueError("reconnect_interval and max_queue_size must be positive")

        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: Any | None = None
        self._capture_lock = threading.RLock()

    def start(self) -> bool:
        if self._thread and self._thread.is_alive():
            return True
        self._stop_event.clear()
        if not self._ensure_capture():
            return False

        self._thread = threading.Thread(target=self._run_loop, name="rtsp-capture", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        self._release_capture()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=max(2.0, self.reconnect_interval + 1.0))
            if self._thread.is_alive():
                LOGGER.warning("RTSP capture thread did not stop before timeout")
        self._thread = None
        self._drain_queue()

    def iter_frames(self, timeout: float = 0.5) -> Iterator[Any]:
        while not self._stop_event.is_set():
            try:
                yield self._queue.get(timeout=timeout)
            except queue.Empty:
                continue

    # ------------------------------------------------------------------ internals
    def _run_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                if not self._ensure_capture():
                    self._stop_event.wait(self.reconnect_interval)
                    continue

                with self._capture_lock:
                    capture = self._capture
                if capture is None:  # defensive against a concurrent stop()
                    continue
                ret, frame = capture.read()
                if not ret or frame is None:
                    LOGGER.warning("RTSP read failed; attempting reconnect.")
                    self._release_capture()
                    self._stop_event.wait(self.reconnect_interval)
                    continue

                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    with contextlib.suppress(queue.Empty):
                        self._queue.get_nowait()
                    with contextlib.suppress(queue.Full):
                        self._queue.put_nowait(frame)
                    LOGGER.debug("Frame queue full; dropped oldest frame.")
        finally:
            self._release_capture()
            self._stop_event.set()

    def _ensure_capture(self) -> bool:
        with self._capture_lock:
            if self._capture is not None:
                return True
            try:
                capture = cv2.VideoCapture(self.url)
                if not capture.isOpened():
                    capture.release()
                    LOGGER.error("Unable to open RTSP stream: %s", redact_url_credentials(self.url))
                    return False
                self._capture = capture
                return True
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.error("Exception while opening RTSP stream: %s", exc)
                self._capture = None
                return False

    def _release_capture(self) -> None:
        with self._capture_lock:
            capture, self._capture = self._capture, None
        if capture is not None:
            with contextlib.suppress(Exception):
                capture.release()

    def _drain_queue(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:  # pragma: no cover - queue already empty
                break


__all__ = ["RTSPCapture"]
