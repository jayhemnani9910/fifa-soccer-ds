"""FastAPI service layer for FIFA Soccer DS YouTube Pipeline.

This module provides REST API endpoints for external access to the
YouTube soccer analysis pipeline.
"""

import asyncio
import logging
import math
import os
import secrets
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src import __version__
from src.pipeline_orchestrator import PipelineOrchestrator, create_pipeline_orchestrator
from src.schemas import TaskStatus, YouTubeAnalysisRequest, validate_youtube_url
from src.utils.monitoring import get_gpu_memory_usage, get_system_metrics
from src.utils.output_paths import validate_output_name

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting configuration
# Default: 10 requests per minute for analysis, 60 requests per minute for reads
limiter = Limiter(key_func=get_remote_address)


# In-process task state. Multi-worker deployments must replace this with a
# durable shared queue/store before enabling more than one worker.
task_storage: dict[str, TaskStatus] = {}
running_tasks: dict[str, asyncio.Task[None]] = {}
orchestrator: PipelineOrchestrator | None = None
MAX_ACTIVE_TASKS = int(os.getenv("MAX_ACTIVE_ANALYSES", "3"))
MAX_TASK_HISTORY = int(os.getenv("MAX_TASK_HISTORY", "200"))
API_KEY = os.getenv("API_KEY")
API_REQUIRE_KEY = os.getenv("API_REQUIRE_KEY", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}
if MAX_ACTIVE_TASKS < 1 or MAX_TASK_HISTORY < 1:
    raise ValueError("MAX_ACTIVE_ANALYSES and MAX_TASK_HISTORY must be positive integers")
if API_REQUIRE_KEY and not API_KEY:
    raise ValueError("API_REQUIRE_KEY is enabled but API_KEY is not configured")


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _prune_task_history() -> None:
    completed = sorted(
        (task for task in task_storage.values() if task.status not in {"pending", "processing"}),
        key=lambda task: task.updated_at,
    )
    for task in completed[: max(0, len(task_storage) - MAX_TASK_HISTORY)]:
        task_storage.pop(task.task_id, None)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Initialize pipeline state and cancel pending coroutines on shutdown."""
    global orchestrator
    logger.info("Starting FIFA Soccer DS API service")
    orchestrator = await create_pipeline_orchestrator()
    try:
        yield
    finally:
        for task in list(running_tasks.values()):
            task.cancel()
        if running_tasks:
            await asyncio.gather(*running_tasks.values(), return_exceptions=True)
        running_tasks.clear()
        orchestrator = None


# Create FastAPI app
app = FastAPI(
    title="FIFA Soccer DS YouTube Analysis API",
    description="REST API for analyzing YouTube videos for soccer content",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Attach limiter to app
app.state.limiter = limiter


def _handle_rate_limit(request: Request, exc: Exception) -> Response:
    if not isinstance(exc, RateLimitExceeded):  # pragma: no cover - framework contract
        raise exc
    return _rate_limit_exceeded_handler(request, exc)


app.add_exception_handler(RateLimitExceeded, _handle_rate_limit)

# CORS Configuration
# Define allowed origins - update this list for your deployment
ALLOWED_ORIGINS = [
    "http://localhost:3000",  # Local frontend development
    "http://localhost:8080",  # Local alternative
    "http://127.0.0.1:3000",  # Local frontend
    "http://127.0.0.1:8080",  # Local alternative
]


# Add production origins from environment variable (comma-separated)
def _validate_cors_origin(origin: str) -> str:
    try:
        parsed = urlsplit(origin)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"Invalid CORS origin: {origin!r}") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or origin == "*"
        or port is not None
        and not 1 <= port <= 65535
    ):
        raise ValueError(f"Invalid CORS origin: {origin!r}")
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    authority = f"{host}:{port}" if port is not None else host
    return f"{parsed.scheme}://{authority}"


_extra_origins = os.getenv("CORS_ALLOWED_ORIGINS", "")
if _extra_origins:
    ALLOWED_ORIGINS.extend(
        _validate_cors_origin(origin.strip())
        for origin in _extra_origins.split(",")
        if origin.strip()
    )
ALLOWED_ORIGINS = list(dict.fromkeys(ALLOWED_ORIGINS))

# Add CORS middleware with restricted origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
)


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Require an API key for non-public routes when one is configured."""
    public_paths = {"/", "/health", "/docs", "/redoc", "/openapi.json"}
    if API_KEY and request.url.path not in public_paths:
        supplied_key = request.headers.get("X-API-Key", "")
        if not secrets.compare_digest(supplied_key, API_KEY):
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
    return await call_next(request)


class AnalyzeRequest(BaseModel):
    """Request model for video analysis."""

    youtube_url: HttpUrl
    output_dir: str | None = None
    frame_rate: float = Field(default=1.0, gt=0, le=30)
    max_duration: int | None = Field(default=300, gt=0, le=7200)
    include_audio: bool = False
    confidence_threshold: float = Field(default=0.75, ge=0, le=1)
    force_full_analysis: bool = False
    sample_duration: float = Field(default=60.0, gt=0, le=300)

    @field_validator("youtube_url")
    @classmethod
    def validate_url(cls, value: HttpUrl) -> HttpUrl:
        if not validate_youtube_url(str(value)):
            raise ValueError("youtube_url must identify a video on an approved HTTPS YouTube host")
        return value

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, value: str | None) -> str | None:
        return validate_output_name(value) if value is not None else None


class AnalyzeResponse(BaseModel):
    """Response model for video analysis."""

    task_id: str
    status: str
    message: str
    estimated_duration: float | None = None


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    timestamp: datetime
    version: str
    system_metrics: dict[str, Any]
    gpu_memory: int | None = None
    active_tasks: int


class TaskResponse(BaseModel):
    """Response model for task status."""

    task_id: str
    status: str
    progress: float
    message: str | None = None
    created_at: datetime
    updated_at: datetime
    results: dict[str, Any] | None = None
    error_details: str | None = None


@app.get("/", response_model=dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "FIFA Soccer DS YouTube Analysis API",
        "version": __version__,
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_endpoint() -> HealthResponse:
    """Health check endpoint."""
    try:
        # Get system metrics
        system_metrics = get_system_metrics()
        gpu_memory = get_gpu_memory_usage()

        # Count active tasks
        active_tasks = sum(
            1 for task in task_storage.values() if task.status in ["pending", "processing"]
        )

        return HealthResponse(
            status="healthy" if orchestrator is not None else "degraded",
            timestamp=_utcnow(),
            version=__version__,
            system_metrics=system_metrics,
            gpu_memory=gpu_memory,
            active_tasks=active_tasks,
        )

    except Exception as exc:
        logger.exception("Health check failed")
        raise HTTPException(status_code=503, detail="Service unhealthy") from exc


@app.post("/analyze", response_model=AnalyzeResponse)
@limiter.limit("10/minute")  # Limit analysis requests to prevent abuse
async def analyze_video(request: Request, req: AnalyzeRequest):
    """Start video analysis task."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    active_tasks = sum(
        1 for task in task_storage.values() if task.status in {"pending", "processing"}
    )
    if active_tasks >= MAX_ACTIVE_TASKS:
        raise HTTPException(status_code=429, detail="Maximum concurrent analyses reached")

    try:
        _prune_task_history()
        task_id = f"task_{uuid.uuid4().hex}"

        # Create analysis request
        analysis_request = YouTubeAnalysisRequest.model_validate(
            {
                "url": str(req.youtube_url),
                "output_dir": req.output_dir,
                "frame_rate": req.frame_rate,
                "max_duration": req.max_duration,
                "include_audio": req.include_audio,
                "confidence_threshold": req.confidence_threshold,
                "force_full_analysis": req.force_full_analysis,
                "sample_duration": req.sample_duration,
            }
        )

        # Create task status
        task_status = TaskStatus(
            task_id=task_id,
            status="pending",
            message="Task created",
            progress=0.0,
            created_at=_utcnow(),
            updated_at=_utcnow(),
        )

        # Store task
        task_storage[task_id] = task_status

        background_task = asyncio.create_task(
            process_video_analysis(task_id, analysis_request),
            name=f"analysis-{task_id}",
        )
        running_tasks[task_id] = background_task
        background_task.add_done_callback(lambda _task: running_tasks.pop(task_id, None))

        # Estimate processing duration (rough estimate)
        estimated_duration = 30.0 + (req.max_duration or 300) * 0.1

        return AnalyzeResponse(
            task_id=task_id,
            status="pending",
            message="Analysis task started",
            estimated_duration=estimated_duration,
        )

    except ValueError as exc:
        logger.info("Rejected analysis request: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to start analysis")
        raise HTTPException(status_code=500, detail="Failed to start analysis") from exc


@app.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task_status(task_id: str):
    """Get status of analysis task."""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")

    task = task_storage[task_id]

    return TaskResponse(
        task_id=task.task_id,
        status=task.status,
        progress=task.progress,
        message=task.message,
        created_at=task.created_at,
        updated_at=task.updated_at,
        results=task.results.model_dump(mode="json") if task.results else None,
        error_details=task.error_details,
    )


@app.get("/tasks", response_model=list[TaskResponse])
async def list_tasks(
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of tasks to return"),
):
    """List analysis tasks."""
    tasks = list(task_storage.values())

    # Filter by status if specified
    if status:
        tasks = [task for task in tasks if task.status == status]

    # Sort by creation time (newest first)
    tasks.sort(key=lambda x: x.created_at, reverse=True)

    # Limit results
    tasks = tasks[:limit]

    return [
        TaskResponse(
            task_id=task.task_id,
            status=task.status,
            progress=task.progress,
            message=task.message,
            created_at=task.created_at,
            updated_at=task.updated_at,
            results=task.results.model_dump(mode="json") if task.results else None,
            error_details=task.error_details,
        )
        for task in tasks
    ]


@app.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a running analysis task."""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")

    task = task_storage[task_id]

    if task.status in ["completed", "error", "cancelled"]:
        raise HTTPException(status_code=400, detail=f"Cannot cancel task in status: {task.status}")

    background_task = running_tasks.get(task_id)
    if task.status != "pending" or background_task is None:
        raise HTTPException(
            status_code=409,
            detail="Processing has started and cannot be cancelled safely",
        )

    background_task.cancel()
    task.status = "cancelled"
    task.message = "Task cancelled before processing started"
    task.updated_at = _utcnow()

    return {"message": f"Task {task_id} cancelled"}


@app.get("/metrics", response_model=dict[str, Any])
async def get_metrics():
    """Get system and pipeline metrics."""
    try:
        system_metrics = get_system_metrics()
        gpu_memory = get_gpu_memory_usage()

        # Count tasks by status
        task_counts: dict[str, int] = {}
        for task in task_storage.values():
            status = task.status
            task_counts[status] = task_counts.get(status, 0) + 1

        return {
            "system_metrics": system_metrics,
            "gpu_memory_bytes": gpu_memory,
            "task_counts": task_counts,
            "total_tasks": len(task_storage),
            "timestamp": _utcnow().isoformat(),
        }

    except Exception as exc:
        logger.exception("Failed to get metrics")
        raise HTTPException(status_code=500, detail="Failed to collect metrics") from exc


@app.get("/pipeline/info", response_model=dict[str, Any])
async def get_pipeline_info():
    """Get pipeline configuration and information."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    return {
        "pipeline_version": orchestrator.config.get("pipeline", {}).get("version", __version__),
        "supported_formats": ["YouTube URLs"],
        "max_video_duration": orchestrator.config.get("youtube", {}).get("max_duration", 7200),
        "supported_languages": ["en", "es", "fr", "de", "it"],
        "confidence_threshold_default": 0.75,
        "features": [
            "Video metadata analysis",
            "Heuristic thumbnail analysis",
            "Optional audio transcription and heuristics",
            "YOLO object detection and ByteTrack-style tracking",
            "Spatial-temporal graph construction",
            "Optional tactical analytics when both teams can be classified",
        ],
    }


# ===== TACTICAL ANALYTICS ENDPOINTS =====


class TacticalPlayer(BaseModel):
    """Validated normalized player state for tactical computation."""

    player_id: int = Field(ge=0)
    team_id: Literal[0, 1]
    position: tuple[float, float]
    velocity: tuple[float, float] | None = None

    @field_validator("position")
    @classmethod
    def validate_position(cls, value: tuple[float, float]) -> tuple[float, float]:
        if not all(math.isfinite(coordinate) and 0.0 <= coordinate <= 1.0 for coordinate in value):
            raise ValueError("position coordinates must be normalized to [0, 1]")
        return value

    @field_validator("velocity")
    @classmethod
    def validate_velocity(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        if value is not None and not all(
            math.isfinite(component) and abs(component) <= 20 for component in value
        ):
            raise ValueError("velocity components must be finite and within ±20 m/s")
        return value


class TacticalComputeRequest(BaseModel):
    """Request model for computing tactical metrics."""

    players: list[TacticalPlayer] = Field(min_length=2, max_length=100)
    frame_id: int = Field(default=0, ge=0)
    grid_shape: tuple[int, int] = (12, 16)

    @field_validator("grid_shape")
    @classmethod
    def validate_grid_shape(cls, value: tuple[int, int]) -> tuple[int, int]:
        if any(dimension < 2 or dimension > 128 for dimension in value):
            raise ValueError("grid dimensions must each be between 2 and 128")
        return value

    @model_validator(mode="after")
    def validate_teams(self) -> "TacticalComputeRequest":
        if {player.team_id for player in self.players} != {0, 1}:
            raise ValueError("players must include both team_id 0 and team_id 1")
        player_ids = [player.player_id for player in self.players]
        if len(player_ids) != len(set(player_ids)):
            raise ValueError("player_id values must be unique within a frame")
        return self


class TacticalComputeResponse(BaseModel):
    """Response model for tactical computation."""

    frame_id: int
    pitch_control: dict[str, Any]
    obso: dict[str, Any]


@app.post("/tactical/compute", response_model=TacticalComputeResponse)
@limiter.limit("30/minute")
async def compute_tactical_metrics(request: Request, req: TacticalComputeRequest):
    """Compute tactical metrics (pitch control, OBSO) for given player positions.

    This endpoint allows direct computation of tactical metrics without
    running the full video analysis pipeline.

    Example request:
    ```json
    {
        "players": [
            {"player_id": 1, "team_id": 0, "position": [0.3, 0.5]},
            {"player_id": 2, "team_id": 0, "position": [0.4, 0.3]},
            {"player_id": 3, "team_id": 1, "position": [0.6, 0.5]},
            {"player_id": 4, "team_id": 1, "position": [0.7, 0.4]}
        ],
        "frame_id": 100,
        "grid_shape": [12, 16]
    }
    ```
    """
    try:
        # Import tactical modules
        import numpy as np

        from src.analytics.tactical import PlayerState, TacticalAnalyzer, TacticalConfig

        # Initialize analyzer
        config = TacticalConfig(grid_shape=req.grid_shape)
        analyzer = TacticalAnalyzer(config=config)

        # Convert request to PlayerState objects
        players: list[PlayerState] = []
        for p in req.players:
            players.append(
                PlayerState(
                    player_id=p.player_id,
                    team_id=p.team_id,
                    position=np.asarray(p.position, dtype=np.float64),
                    velocity=(
                        np.asarray(p.velocity, dtype=np.float64) if p.velocity is not None else None
                    ),
                )
            )

        # Compute tactical metrics
        result = analyzer.compute(req.frame_id, players)

        return TacticalComputeResponse(
            frame_id=result.frame_id,
            pitch_control={
                "grid": result.pitch_control.grid.tolist(),
                "home_control_pct": round(result.pitch_control.home_control_pct, 2),
                "away_control_pct": round(result.pitch_control.away_control_pct, 2),
            },
            obso={
                "grid": result.obso_grid.tolist() if result.obso_grid is not None else None,
                "home_total": round(result.home_obso_total, 4),
                "away_total": round(result.away_obso_total, 4),
            },
        )

    except ImportError as exc:
        logger.exception("Tactical analytics not available")
        raise HTTPException(
            status_code=503, detail="Tactical analytics module not available"
        ) from exc
    except ValueError as exc:
        logger.info("Rejected tactical computation: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Tactical computation failed")
        raise HTTPException(status_code=500, detail="Tactical computation failed") from exc


@app.get("/tactical/info", response_model=dict[str, Any])
async def get_tactical_info():
    """Get information about tactical analytics capabilities."""
    return {
        "available": True,
        "metrics": [
            {
                "name": "pitch_control",
                "description": "Probability that each team controls each area of the pitch",
                "output_range": [0, 1],
                "interpretation": "0 = away team controls, 1 = home team controls",
            },
            {
                "name": "expected_threat",
                "description": "Static heuristic attacking-value surface; not calibrated xT",
                "output_range": [0, 0.5],
                "interpretation": "Higher values near opponent goal",
            },
            {
                "name": "obso",
                "description": "Experimental opportunity surface (pitch control * heuristic value)",
                "output_range": [0, 0.5],
                "interpretation": "Where should attacking players position themselves",
            },
        ],
        "parameters": {
            "grid_shape": {
                "default": [12, 16],
                "description": "Output grid dimensions (rows, columns)",
            },
            "max_speed": {
                "default": 5.0,
                "unit": "m/s",
                "description": "Maximum player running speed",
            },
            "reaction_time": {
                "default": 0.7,
                "unit": "seconds",
                "description": "Time before player reacts",
            },
            "pitch_dimensions": {"length": 105.0, "width": 68.0, "unit": "meters"},
        },
    }


@app.get("/tactical/results/{task_id}", response_model=dict[str, Any])
async def get_tactical_results(task_id: str):
    """Get tactical analytics results for a completed analysis task."""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")

    task = task_storage[task_id]

    if task.status != "completed":
        raise HTTPException(status_code=400, detail=f"Task not completed (status: {task.status})")

    # Check if tactical results exist
    if not task.results:
        raise HTTPException(status_code=404, detail="No results available for this task")

    tactical_data = task.results.tactical_analytics

    if not tactical_data:
        raise HTTPException(status_code=404, detail="No tactical analytics data in task results")

    return tactical_data


async def process_video_analysis(task_id: str, request: YouTubeAnalysisRequest):
    """Background task to process video analysis."""
    task = task_storage.get(task_id)
    if not task:
        return

    try:
        # Update task status
        task.status = "processing"
        task.message = "Starting video analysis"
        task.progress = 0.1
        task.updated_at = _utcnow()

        logger.info(f"Starting analysis for task {task_id}")

        # Process video
        active_orchestrator = orchestrator
        if active_orchestrator is None:
            raise RuntimeError("Pipeline is shutting down")
        result = await active_orchestrator.process_youtube_video(request)

        # Update task status
        task.status = "completed"
        task.message = "Analysis completed successfully"
        task.progress = 1.0
        task.results = result
        task.updated_at = _utcnow()

        logger.info(f"Analysis completed for task {task_id}")

    except asyncio.CancelledError:
        task.status = "cancelled"
        task.message = "Task cancelled before processing started"
        task.updated_at = _utcnow()
        raise
    except Exception:
        # Update task with error
        task.status = "error"
        task.message = "Analysis failed"
        task.error_details = "Processing failed; consult server logs"
        task.updated_at = _utcnow()

        logger.exception("Analysis failed for task %s", task_id)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "http_exception",
                "message": exc.detail,
                "status_code": exc.status_code,
            }
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.exception("Unhandled API exception")
    return JSONResponse(
        status_code=500,
        content={"error": {"type": "internal_error", "message": "Internal server error"}},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_RELOAD", "false").lower() == "true",
        log_level="info",
    )
