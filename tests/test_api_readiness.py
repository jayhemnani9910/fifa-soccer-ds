"""Contract tests for liveness (/health) vs readiness (/ready) and the weights guard.

/health must stay a pure liveness probe: it reports 200 whenever the process is
running, regardless of whether a model checkpoint is configured. /ready is the
serving-capability probe: it must fail with a clear reason when no usable YOLO
weights file is configured, without ever loading the checkpoint.
"""

import asyncio

import httpx
import pytest

from src.api import main as api

pytestmark = pytest.mark.anyio


class FakeOrchestrator:
    """Stands in for PipelineOrchestrator; process_youtube_video must never be
    reached once the weights guard rejects the request."""

    def __init__(self) -> None:
        self.config = {"pipeline": {"version": "test"}, "youtube": {"max_duration": 600}}

    async def process_youtube_video(self, _request):  # type: ignore[no-untyped-def]
        await asyncio.sleep(0)
        raise AssertionError("process_youtube_video must not run when weights are unconfigured")


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def isolated_api_state(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    api.task_storage.clear()
    api.running_tasks.clear()
    monkeypatch.setattr(api, "orchestrator", FakeOrchestrator())
    monkeypatch.setattr(api, "API_KEY", None)
    monkeypatch.delenv("YOLO_WEIGHTS", raising=False)
    yield
    for task in tuple(api.running_tasks.values()):
        task.cancel()
    api.running_tasks.clear()
    api.task_storage.clear()


def _transport() -> httpx.ASGITransport:
    return httpx.ASGITransport(app=api.app, raise_app_exceptions=False)


async def test_readiness_fails_when_no_weights_configured() -> None:
    async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
        response = await client.get("/ready")

    assert response.status_code == 503
    assert "YOLO_WEIGHTS" in response.json()["error"]["message"]


async def test_readiness_fails_when_configured_weights_file_is_absent(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "does-not-exist.pt"
    monkeypatch.setenv("YOLO_WEIGHTS", str(missing))

    async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
        response = await client.get("/ready")

    assert response.status_code == 503
    assert str(missing) not in response.text


async def test_readiness_succeeds_when_weights_file_is_present(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = tmp_path / "yolov8n.pt"
    weights.write_bytes(b"dummy-checkpoint-bytes")
    monkeypatch.setenv("YOLO_WEIGHTS", str(weights))

    async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
        response = await client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


@pytest.mark.parametrize(
    "weights_env",
    [None, "does-not-matter.pt"],
    ids=["unset", "configured-but-nonexistent"],
)
async def test_liveness_stays_200_regardless_of_weights_state(
    weights_env: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    if weights_env is None:
        monkeypatch.delenv("YOLO_WEIGHTS", raising=False)
    else:
        monkeypatch.setenv("YOLO_WEIGHTS", weights_env)

    async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200


async def test_analysis_fails_clearly_when_no_weights_configured() -> None:
    async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
        response = await client.post(
            "/analyze",
            json={"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        )

    assert response.status_code == 503
    assert "YOLO_WEIGHTS" in response.json()["error"]["message"]
    assert not api.task_storage


async def test_ready_endpoint_is_public_when_api_key_required(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    weights = tmp_path / "yolov8n.pt"
    weights.write_bytes(b"dummy-checkpoint-bytes")
    monkeypatch.setenv("YOLO_WEIGHTS", str(weights))
    monkeypatch.setattr(api, "API_KEY", "test-secret")

    async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
        response = await client.get("/ready")

    assert response.status_code == 200
