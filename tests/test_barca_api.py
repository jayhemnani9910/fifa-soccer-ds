from __future__ import annotations

import asyncio
import queue
import threading
import time

import httpx
import numpy as np
import pytest

from src.live.barca_api import BarcaAPIServer


@pytest.fixture(autouse=True)
def _run_thread_offloads_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep API tests deterministic in runtimes where asyncio workers are unavailable."""

    async def run_inline(function, /, *args, **kwargs):  # type: ignore[no-untyped-def]
        return function(*args, **kwargs)

    monkeypatch.setattr("src.live.barca_api.asyncio.to_thread", run_inline)


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


class DummyBoxes:
    def __init__(self):
        self.xyxy = np.array([[0.0, 0.0, 5.0, 5.0]])
        self.conf = np.array([0.9])
        self.cls = np.array([0])


class DummyResult:
    def __init__(self):
        self.boxes = DummyBoxes()
        self.names = {0: "player"}


class FakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, frame, conf: float = 0.5, verbose: bool = False):
        self.calls += 1
        return [DummyResult()]


class FakeCapture:
    def __init__(self, url: str) -> None:
        self.url = url
        self._active = False
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        self._active = True
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()
        return True

    def _produce(self) -> None:
        for _ in range(12):
            if not self._active:
                break
            frame = np.ones((8, 8, 3), dtype=np.uint8)
            self._queue.put(frame)
            time.sleep(0.01)
        self._active = False

    def iter_frames(self, timeout: float = 0.1):
        while self._active or not self._queue.empty():
            try:
                yield self._queue.get(timeout=timeout)
            except queue.Empty:
                if not self._active:
                    break

    def stop(self) -> None:
        self._active = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)


@pytest.fixture()
def api_server():  # type: ignore[no-untyped-def]
    """Fixture providing BarcaAPIServer with mock detector and capture."""

    def fake_capture_factory(url: str) -> FakeCapture:  # type: ignore[name-defined]
        return FakeCapture(url)  # type: ignore[name-defined]

    def fake_detector_factory():  # type: ignore[no-untyped-def]
        return FakeDetector()  # type: ignore[name-defined]

    server = BarcaAPIServer(
        detector_factory=fake_detector_factory,
        capture_factory=fake_capture_factory,
        target_validator=lambda _url: None,
    )  # type: ignore[arg-type]
    server.checkpoint_metadata = {"commit": "abc123", "mAP@0.5": 0.52, "data_version": 3}
    yield server
    # Cleanup: ensure stream is fully stopped and threads joined
    if hasattr(server, "_stop_event"):
        server._stop_event.set()
    if hasattr(server, "_capture") and server._capture:
        server._capture.stop()
        server._capture = None
    if (
        hasattr(server, "_inference_thread")
        and server._inference_thread
        and server._inference_thread.is_alive()
    ):
        server._inference_thread.join(timeout=1.0)
    time.sleep(0.1)  # Allow threads to fully terminate


@pytest.mark.anyio
async def test_api_start_stop(api_server: BarcaAPIServer) -> None:  # type: ignore[name-defined]
    transport = httpx.ASGITransport(app=api_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/stream/start", json={"rtsp_url": "rtsp://example"})
        assert response.status_code == 200
        await asyncio.sleep(0.15)

        detections = (await client.get("/detections")).json()
        assert "detections" in detections

        checkpoint = (await client.get("/checkpoint")).json()
        assert checkpoint["commit"] == "abc123"

        stop = await client.post("/stream/stop")
        assert stop.status_code == 200

        health = (await client.get("/health")).json()
        assert health["streaming"] is False


def test_inference_loop_without_capture_records_error(api_server: BarcaAPIServer) -> None:
    api_server._capture = None

    api_server._inference_loop()

    assert api_server._last_error == "Inference loop started without an active capture"


@pytest.mark.anyio
async def test_api_detection_latency(api_server: BarcaAPIServer) -> None:  # type: ignore[name-defined]
    transport = httpx.ASGITransport(app=api_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/stream/start", json={"rtsp_url": "rtsp://example"})
        assert response.status_code == 200

        latency_ms = None
        for _ in range(30):
            payload = (await client.get("/detections")).json()
            if payload.get("detections"):
                latency_ms = payload.get("latency_ms")
                break
            await asyncio.sleep(0.05)

        assert latency_ms is not None
        assert latency_ms >= 0.0

        stop = await client.post("/stream/stop")
        assert stop.status_code == 200


def test_frame_ids_are_monotonic_across_batches(api_server: BarcaAPIServer) -> None:
    detector = FakeDetector()
    frame = np.ones((8, 8, 3), dtype=np.uint8)

    first = api_server._run_detector(detector, [frame, frame])
    second = api_server._run_detector(detector, [frame])

    assert [item["frame_id"] for item in first + second] == [0, 1, 2]


def test_default_detector_factory_requires_configured_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: unset YOLO_WEIGHTS must not silently fall back to a bundled default."""
    from src.live.barca_api import _default_detector_factory
    from src.pipeline_orchestrator import WeightsNotConfiguredError

    monkeypatch.delenv("YOLO_WEIGHTS", raising=False)

    with pytest.raises(WeightsNotConfiguredError, match="YOLO_WEIGHTS"):
        _default_detector_factory()
