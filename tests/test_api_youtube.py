"""Contract tests for the canonical YouTube analysis API."""

import asyncio
from datetime import datetime
from unittest.mock import patch

import httpx
import pytest

from src.api import main as api

pytestmark = pytest.mark.anyio


class FakeResult:
    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {"pipeline_version": "test", "errors": [], "warnings": []}


class FakeOrchestrator:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.config = {"pipeline": {"version": "test"}, "youtube": {"max_duration": 600}}

    async def process_youtube_video(self, _request):  # type: ignore[no-untyped-def]
        await asyncio.sleep(0)
        if self.error is not None:
            raise self.error
        return FakeResult()


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def isolated_api_state(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    api.task_storage.clear()
    api.running_tasks.clear()
    fake = FakeOrchestrator()
    monkeypatch.setattr(api, "orchestrator", fake)
    monkeypatch.setattr(api, "API_KEY", None)
    yield fake
    for task in tuple(api.running_tasks.values()):
        task.cancel()
    api.running_tasks.clear()
    api.task_storage.clear()


def _transport() -> httpx.ASGITransport:
    return httpx.ASGITransport(app=api.app, raise_app_exceptions=False)


async def test_root_and_health_endpoints() -> None:
    with (
        patch.object(api, "get_system_metrics", return_value={"cpu": 1.0}),
        patch.object(api, "get_gpu_memory_usage", return_value=0),
    ):
        async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
            root = await client.get("/")
            health = await client.get("/health")

    assert root.status_code == 200
    assert root.json()["service"] == "FIFA Soccer DS YouTube Analysis API"
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"


async def test_configured_api_key_protects_non_public_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "API_KEY", "test-secret")
    with (
        patch.object(api, "get_system_metrics", return_value={"cpu": 1.0}),
        patch.object(api, "get_gpu_memory_usage", return_value=0),
    ):
        async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
            public_health = await client.get("/health")
            missing = await client.get("/metrics")
            accepted = await client.get("/metrics", headers={"X-API-Key": "test-secret"})

    assert public_health.status_code == 200
    assert missing.status_code == 401
    assert accepted.status_code == 200


async def test_tactical_compute_validates_teams_positions_and_grid_limits() -> None:
    async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
        one_team = await client.post(
            "/tactical/compute",
            json={
                "players": [
                    {"player_id": 1, "team_id": 0, "position": [0.2, 0.5]},
                    {"player_id": 2, "team_id": 0, "position": [0.8, 0.5]},
                ]
            },
        )
        outside_pitch = await client.post(
            "/tactical/compute",
            json={
                "players": [
                    {"player_id": 1, "team_id": 0, "position": [1.2, 0.5]},
                    {"player_id": 2, "team_id": 1, "position": [0.8, 0.5]},
                ]
            },
        )
        oversized_grid = await client.post(
            "/tactical/compute",
            json={
                "players": [
                    {"player_id": 1, "team_id": 0, "position": [0.2, 0.5]},
                    {"player_id": 2, "team_id": 1, "position": [0.8, 0.5]},
                ],
                "grid_shape": [129, 129],
            },
        )

    assert one_team.status_code == 422
    assert outside_pitch.status_code == 422
    assert oversized_grid.status_code == 422


async def test_tactical_compute_returns_measured_grid_for_both_teams() -> None:
    async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
        response = await client.post(
            "/tactical/compute",
            json={
                "players": [
                    {"player_id": 1, "team_id": 0, "position": [0.2, 0.5]},
                    {"player_id": 2, "team_id": 1, "position": [0.8, 0.5]},
                ],
                "grid_shape": [4, 6],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["pitch_control"]["grid"]) == 4
    assert len(payload["pitch_control"]["grid"][0]) == 6


async def test_analysis_task_completes_and_returns_results() -> None:
    async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
        response = await client.post(
            "/analyze",
            json={"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        )
        assert response.status_code == 200
        task_id = response.json()["task_id"]

        for _ in range(10):
            status = await client.get(f"/tasks/{task_id}")
            if status.json()["status"] == "completed":
                break
            await asyncio.sleep(0)

    assert status.status_code == 200
    payload = status.json()
    assert payload["status"] == "completed"
    assert payload["results"]["pipeline_version"] == "test"


@pytest.mark.parametrize(
    "url",
    [
        "invalid-url",
        "http://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://example.com/watch?v=dQw4w9WgXcQ",
        "https://user:secret@youtube.com/watch?v=dQw4w9WgXcQ",
    ],
)
async def test_analysis_rejects_untrusted_urls(url: str) -> None:
    async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
        response = await client.post("/analyze", json={"youtube_url": url})

    assert response.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"youtube_url": "https://youtu.be/dQw4w9WgXcQ", "confidence_threshold": 1.1},
        {"youtube_url": "https://youtu.be/dQw4w9WgXcQ", "sample_duration": 0},
        {"youtube_url": "https://youtu.be/dQw4w9WgXcQ", "output_dir": "../escape"},
    ],
)
async def test_analysis_request_bounds(payload: dict[str, object]) -> None:
    async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
        response = await client.post("/analyze", json=payload)

    assert response.status_code == 422


async def test_missing_tasks_return_sanitized_404() -> None:
    async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
        get_response = await client.get("/tasks/not-present")
        delete_response = await client.delete("/tasks/not-present")

    assert get_response.status_code == 404
    assert get_response.json()["error"]["message"] == "Task not found"
    assert delete_response.status_code == 404


async def test_active_task_limit_is_enforced() -> None:
    for index in range(api.MAX_ACTIVE_TASKS):
        task_id = f"existing-{index}"
        api.task_storage[task_id] = api.TaskStatus(
            task_id=task_id,
            status="processing",
            progress=0.5,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
        response = await client.post(
            "/analyze",
            json={"youtube_url": "https://youtu.be/dQw4w9WgXcQ"},
        )

    assert response.status_code == 429


async def test_processing_errors_do_not_disclose_exception_details(
    isolated_api_state: FakeOrchestrator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert isolated_api_state is not None
    monkeypatch.setattr(api, "orchestrator", FakeOrchestrator(RuntimeError("secret /srv/path")))

    async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
        response = await client.post(
            "/analyze",
            json={"youtube_url": "https://youtu.be/dQw4w9WgXcQ"},
        )
        task_id = response.json()["task_id"]
        for _ in range(10):
            status = await client.get(f"/tasks/{task_id}")
            if status.json()["status"] == "error":
                break
            await asyncio.sleep(0)

    payload = status.json()
    assert payload["status"] == "error"
    assert payload["error_details"] == "Processing failed; consult server logs"
    assert "secret" not in str(payload)


async def test_pending_task_can_be_cancelled() -> None:
    task_id = "pending-task"
    api.task_storage[task_id] = api.TaskStatus(
        task_id=task_id,
        status="pending",
        progress=0.0,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    background = asyncio.create_task(asyncio.sleep(60))
    api.running_tasks[task_id] = background

    async with httpx.AsyncClient(transport=_transport(), base_url="http://test") as client:
        response = await client.delete(f"/tasks/{task_id}")

    assert response.status_code == 200
    assert api.task_storage[task_id].status == "cancelled"
    assert background.cancelled() or background.cancelling()


def test_legacy_module_exports_canonical_app() -> None:
    from src.api.youtube_endpoints import app

    assert app is api.app
