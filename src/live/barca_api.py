"""FastAPI service orchestrating live Barca match inference."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.detect.infer import extract_detections
from src.detect.yolo_lora_adapter import YOLOLoRAAdapter
from src.live.rtsp_capture import RTSPCapture

LOGGER = logging.getLogger(__name__)


class StreamRequest(BaseModel):
    rtsp_url: str


@dataclass(slots=True)
class BarcaAPIServer:
    detector_factory: Callable[[], Any] = YOLOLoRAAdapter
    capture_factory: Callable[[str], RTSPCapture] = RTSPCapture
    batch_size: int = 5
    confidence: float = 0.5
    max_queue_size: int = 32
    checkpoint_metadata: dict[str, Any] = field(default_factory=dict)
    app: FastAPI = field(init=False)
    _capture: RTSPCapture | None = field(init=False, default=None)
    _detector: Any | None = field(init=False, default=None)
    _inference_thread: threading.Thread | None = field(init=False, default=None)
    _stop_event: threading.Event = field(init=False, default_factory=threading.Event)
    _latest_detections: list[dict[str, Any]] = field(init=False, default_factory=list)
    _latest_latency_ms: float = field(init=False, default=0.0)
    _latest_timestamp: float = field(init=False, default=0.0)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.app = FastAPI(title="Barca Live Inference", version="0.1.0")

        self._register_routes()

    # ------------------------------------------------------------------ routing
    def _register_routes(self) -> None:
        @self.app.post("/stream/start")
        async def start_stream(request: StreamRequest) -> JSONResponse:
            if self._capture is not None:
                raise HTTPException(status_code=409, detail="Stream already running")

            capture = self.capture_factory(request.rtsp_url)
            if not capture.start():
                raise HTTPException(status_code=500, detail="Unable to start RTSP capture")

            self._capture = capture
            self._detector = self.detector_factory()
            self._stop_event.clear()
            self._inference_thread = threading.Thread(
                target=self._inference_loop, name="barca-inference", daemon=True
            )
            self._inference_thread.start()

            return JSONResponse({"status": "started", "url": request.rtsp_url})

        @self.app.post("/stream/stop")
        async def stop_stream() -> JSONResponse:
            self._shutdown_stream()
            return JSONResponse({"status": "stopped"})

        @self.app.get("/detections")
        async def get_detections() -> JSONResponse:
            with self._lock:
                payload = {
                    "detections": self._latest_detections,
                    "latency_ms": self._latest_latency_ms,
                    "timestamp": self._latest_timestamp,
                }
            return JSONResponse(payload)

        @self.app.get("/checkpoint")
        async def get_checkpoint() -> JSONResponse:
            return JSONResponse(self.checkpoint_metadata or {"status": "unavailable"})

        @self.app.get("/health")
        async def health() -> JSONResponse:
            return JSONResponse({"status": "ok", "streaming": self._capture is not None})

    # ---------------------------------------------------------------- inference
    def _inference_loop(self) -> None:
        assert self._capture is not None
        detector = self._detector
        if detector is None:
            detector = self.detector_factory()
            self._detector = detector

        batch: list[np.ndarray] = []

        for frame in self._capture.iter_frames():
            if self._stop_event.is_set():
                break

            if not isinstance(frame, np.ndarray):
                frame = np.asarray(frame)

            batch.append(frame)
            if len(batch) < self.batch_size:
                continue

            start_ts = time.time()
            detections = self._run_detector(detector, batch)
            latency = (time.time() - start_ts) * 1000.0

            with self._lock:
                self._latest_detections = detections
                self._latest_latency_ms = latency
                self._latest_timestamp = time.time()

            batch.clear()

        LOGGER.info("Inference loop terminated.")

    def _run_detector(self, detector: Any, frames: list[np.ndarray]) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        for index, frame in enumerate(frames):
            frame_id = int(time.time() * 1000) + index
            if hasattr(detector, "predict"):
                try:
                    results = detector.predict(frame, conf=self.confidence, verbose=False)
                    parsed = extract_detections(results[0]) if results else []
                except Exception as exc:  # pragma: no cover - defensive fallback
                    LOGGER.error("Detector inference error: %s", exc)
                    parsed = []
            else:
                height, width = frame.shape[:2]
                parsed = [
                    {
                        "bbox": [0, 0, width, height],
                        "confidence": 1.0,
                        "class_id": 0,
                        "class_name": "player",
                    }
                ]

            detections.append({"frame_id": frame_id, "detections": parsed})
        return detections

    def _shutdown_stream(self) -> None:
        self._stop_event.set()
        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=2.0)
        self._inference_thread = None

        if self._capture is not None:
            self._capture.stop()
        self._capture = None
        self._detector = None

    def shutdown(self) -> None:
        """Public shutdown method for cleanup in tests."""
        self._shutdown_stream()

    # ---------------------------------------------------------------- metadata
    def load_checkpoint_metadata(self, path: Path) -> None:
        if not path.exists():
            LOGGER.warning("Checkpoint metadata file not found: %s", path)
            return
        try:
            payload = json.loads(path.read_text())
            self.checkpoint_metadata = payload
        except json.JSONDecodeError as exc:  # pragma: no cover - invalid json
            LOGGER.error("Failed to parse checkpoint metadata: %s", exc)


__all__ = ["BarcaAPIServer"]
