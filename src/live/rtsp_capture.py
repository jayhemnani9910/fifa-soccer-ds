"""Threaded RTSP capture utility with automatic reconnection."""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from collections.abc import Iterator
from typing import Any

try:  # pragma: no cover - optional dependency in tests
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

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

        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: Any | None = None

    def start(self) -> bool:
        if self._thread and self._thread.is_alive():
            return True
        if cv2 is None:
            LOGGER.error("OpenCV is required for RTSP capture.")
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="rtsp-capture", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._release_capture()
        self._drain_queue()

    def iter_frames(self, timeout: float = 0.5) -> Iterator[Any]:
        while not self._stop_event.is_set():
            try:
                yield self._queue.get(timeout=timeout)
            except queue.Empty:
                continue

    # ------------------------------------------------------------------ internals
    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self._ensure_capture():
                time.sleep(self.reconnect_interval)
                continue

            ret, frame = self._capture.read()
            if not ret or frame is None:
                LOGGER.warning("RTSP read failed; attempting reconnect.")
                self._release_capture()
                time.sleep(self.reconnect_interval)
                continue

            try:
                self._queue.put(frame, timeout=0.1)
            except queue.Full:
                LOGGER.debug("Frame queue full; dropping frame.")

    def _ensure_capture(self) -> bool:
        if self._capture is not None:
            return True
        try:
            self._capture = cv2.VideoCapture(self.url)
            if not self._capture.isOpened():
                LOGGER.error("Unable to open RTSP stream: %s", self.url)
                self._release_capture()
                return False
            return True
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error("Exception while opening RTSP stream: %s", exc)
            self._release_capture()
            return False

    def _release_capture(self) -> None:
        if self._capture is not None:
            with contextlib.suppress(Exception):
                self._capture.release()
            self._capture = None

    def _drain_queue(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:  # pragma: no cover - queue already empty
                break


__all__ = ["RTSPCapture"]
