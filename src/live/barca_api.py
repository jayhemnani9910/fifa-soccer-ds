"""FastAPI service orchestrating live Barca match inference."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.detect.infer import InferenceConfig, extract_detections, first_prediction, load_model
from src.live.rtsp_capture import RTSPCapture
from src.pipeline_orchestrator import WeightsNotConfiguredError, describe_weights_problem
from src.utils.network_security import redact_url_credentials, validate_rtsp_target

LOGGER = logging.getLogger(__name__)


def _default_detector_factory() -> Any:
    weights = os.getenv("YOLO_WEIGHTS") or ""
    problem = describe_weights_problem(weights)
    if problem is not None:
        raise WeightsNotConfiguredError(problem)
    return load_model(InferenceConfig(weights=weights))


class StreamRequest(BaseModel):
    rtsp_url: str = Field(min_length=1, max_length=2048)

    @field_validator("rtsp_url")
    @classmethod
    def validate_scheme(cls, value: str) -> str:
        if not value.lower().startswith(("rtsp://", "rtsps://")):
            raise ValueError("Only rtsp:// and rtsps:// sources are supported")
        return value


@dataclass(slots=True)
class BarcaAPIServer:
    detector_factory: Callable[[], Any] = _default_detector_factory
    capture_factory: Callable[[str], RTSPCapture] = RTSPCapture
    target_validator: Callable[[str], None] = validate_rtsp_target
    batch_size: int = 5
    confidence: float = 0.5
    checkpoint_metadata: dict[str, Any] = field(default_factory=dict)
    app: FastAPI = field(init=False)
    _capture: RTSPCapture | None = field(init=False, default=None)
    _detector: Any | None = field(init=False, default=None)
    _inference_thread: threading.Thread | None = field(init=False, default=None)
    _stop_event: threading.Event = field(init=False, default_factory=threading.Event)
    _latest_detections: list[dict[str, Any]] = field(init=False, default_factory=list)
    _latest_latency_ms: float = field(init=False, default=0.0)
    _latest_timestamp: float = field(init=False, default=0.0)
    _next_frame_id: int = field(init=False, default=0)
    _last_error: str | None = field(init=False, default=None)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    _lifecycle_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be within [0, 1]")

        @asynccontextmanager
        async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
            try:
                yield
            finally:
                await asyncio.to_thread(self._shutdown_stream)

        self.app = FastAPI(title="Barca Live Inference", version="0.1.0", lifespan=lifespan)

        self._register_routes()

    # ------------------------------------------------------------------ routing
    def _register_routes(self) -> None:
        @self.app.post("/stream/start")
        async def start_stream(request: StreamRequest) -> JSONResponse:
            async with self._lifecycle_lock:
                if self._capture is not None:
                    raise HTTPException(status_code=409, detail="Stream already running")

                try:
                    await asyncio.to_thread(self.target_validator, request.rtsp_url)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc

                capture = self.capture_factory(request.rtsp_url)
                if not await asyncio.to_thread(capture.start):
                    raise HTTPException(status_code=500, detail="Unable to start RTSP capture")

                try:
                    detector = await asyncio.to_thread(self.detector_factory)
                    if not callable(getattr(detector, "predict", None)):
                        raise TypeError("Configured detector does not implement predict()")
                except Exception as exc:
                    await asyncio.to_thread(capture.stop)
                    LOGGER.exception("Detector initialization failed")
                    raise HTTPException(
                        status_code=503, detail="Detector initialization failed"
                    ) from exc

                self._capture = capture
                self._detector = detector
                self._last_error = None
                self._stop_event.clear()
                self._inference_thread = threading.Thread(
                    target=self._inference_loop, name="barca-inference", daemon=True
                )
                self._inference_thread.start()

            return JSONResponse(
                {"status": "started", "source": redact_url_credentials(request.rtsp_url)}
            )

        @self.app.post("/stream/stop")
        async def stop_stream() -> JSONResponse:
            async with self._lifecycle_lock:
                await asyncio.to_thread(self._shutdown_stream)
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
            with self._lock:
                last_error = self._last_error
            return JSONResponse(
                {
                    "status": "degraded" if last_error else "ok",
                    "streaming": bool(
                        self._capture is not None
                        and self._inference_thread is not None
                        and self._inference_thread.is_alive()
                    ),
                    "last_error": last_error,
                }
            )

    # ---------------------------------------------------------------- inference
    def _inference_loop(self) -> None:
        capture = self._capture
        if capture is None:
            message = "Inference loop started without an active capture"
            with self._lock:
                self._last_error = message
            LOGGER.error(message)
            return
        detector = self._detector
        if detector is None:
            detector = self.detector_factory()
            self._detector = detector

        batch: list[np.ndarray] = []

        try:
            for frame in capture.iter_frames():
                if self._stop_event.is_set():
                    break

                if not isinstance(frame, np.ndarray):
                    frame = np.asarray(frame)

                batch.append(frame)
                if len(batch) < self.batch_size:
                    continue

                self._publish_batch(detector, batch)
                batch.clear()

            if batch and not self._stop_event.is_set():
                self._publish_batch(detector, batch)
        except Exception:
            LOGGER.exception("Inference loop terminated unexpectedly")
            with self._lock:
                self._last_error = "Inference loop terminated unexpectedly"
        finally:
            capture.stop()
            with self._lock:
                if self._capture is capture:
                    self._capture = None
                    self._detector = None
            LOGGER.info("Inference loop terminated.")

    def _publish_batch(self, detector: Any, batch: list[np.ndarray]) -> None:
        start_ts = time.monotonic()
        detections = self._run_detector(detector, batch)
        latency = (time.monotonic() - start_ts) * 1000.0
        with self._lock:
            self._latest_detections = detections
            self._latest_latency_ms = latency
            self._latest_timestamp = time.time()

    def _run_detector(self, detector: Any, frames: list[np.ndarray]) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        for frame in frames:
            with self._lock:
                frame_id = self._next_frame_id
                self._next_frame_id += 1
            try:
                results = detector.predict(frame, conf=self.confidence, verbose=False)
                prediction = first_prediction(results)
                parsed = extract_detections(prediction) if prediction is not None else []
                with self._lock:
                    self._last_error = None
            except Exception:  # pragma: no cover - defensive runtime boundary
                LOGGER.exception("Detector inference failed")
                with self._lock:
                    self._last_error = "Detector inference failed"
                detections.append(
                    {"frame_id": frame_id, "detections": [], "error": "inference_failed"}
                )
                continue

            detections.append({"frame_id": frame_id, "detections": parsed})
        return detections

    def _shutdown_stream(self) -> None:
        self._stop_event.set()
        capture = self._capture
        if capture is not None:
            capture.stop()
        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=2.0)
        self._inference_thread = None

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
            if path.stat().st_size > 1024 * 1024:
                raise ValueError("checkpoint metadata exceeds 1 MiB")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("checkpoint metadata must be a JSON object")
            self.checkpoint_metadata = dict(payload)
        except (OSError, ValueError) as exc:  # pragma: no cover - invalid file
            LOGGER.error("Failed to parse checkpoint metadata: %s", exc)


__all__ = ["BarcaAPIServer"]
