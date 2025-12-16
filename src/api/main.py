"""FastAPI service layer for FIFA Soccer DS YouTube Pipeline.

This module provides REST API endpoints for external access to the
YouTube soccer analysis pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.pipeline_orchestrator import PipelineOrchestrator, create_pipeline_orchestrator
from src.schemas import YouTubeAnalysisRequest, TaskStatus, PipelineOutput
from src.utils.health_checks import health_check
from src.utils.monitoring import get_system_metrics, get_gpu_memory_usage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting configuration
# Default: 10 requests per minute for analysis, 60 requests per minute for reads
limiter = Limiter(key_func=get_remote_address)

# Create FastAPI app
app = FastAPI(
    title="FIFA Soccer DS YouTube Analysis API",
    description="REST API for analyzing YouTube videos for soccer content",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Attach limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS Configuration
# Define allowed origins - update this list for your deployment
ALLOWED_ORIGINS = [
    "http://localhost:3000",      # Local frontend development
    "http://localhost:8080",      # Local alternative
    "http://127.0.0.1:3000",      # Local frontend
    "http://127.0.0.1:8080",      # Local alternative
]

# Add production origins from environment variable (comma-separated)
import os
_extra_origins = os.getenv("CORS_ALLOWED_ORIGINS", "")
if _extra_origins:
    ALLOWED_ORIGINS.extend([origin.strip() for origin in _extra_origins.split(",") if origin.strip()])

# Add CORS middleware with restricted origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

# Global storage for task status (in production, use Redis or database)
task_storage: Dict[str, TaskStatus] = {}

# Global orchestrator instance
orchestrator: Optional[PipelineOrchestrator] = None


class AnalyzeRequest(BaseModel):
    """Request model for video analysis."""
    youtube_url: HttpUrl
    output_dir: Optional[str] = None
    frame_rate: float = 1.0
    max_duration: Optional[int] = 300
    include_audio: bool = True
    confidence_threshold: float = 0.75
    force_full_analysis: bool = False


class AnalyzeResponse(BaseModel):
    """Response model for video analysis."""
    task_id: str
    status: str
    message: str
    estimated_duration: Optional[float] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: datetime
    version: str
    system_metrics: Dict[str, Any]
    gpu_memory: Optional[int] = None
    active_tasks: int


class TaskResponse(BaseModel):
    """Response model for task status."""
    task_id: str
    status: str
    progress: float
    message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    results: Optional[Dict[str, Any]] = None
    error_details: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the API service."""
    global orchestrator
    
    logger.info("Starting FIFA Soccer DS API service...")
    
    try:
        # Initialize pipeline orchestrator
        orchestrator = await create_pipeline_orchestrator()
        logger.info("Pipeline orchestrator initialized successfully")
        
        # Log startup completion
        logger.info("API service startup completed")
        
    except Exception as e:
        logger.error(f"Failed to initialize API service: {e}")
        raise


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "FIFA Soccer DS YouTube Analysis API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_endpoint():
    """Health check endpoint."""
    try:
        # Get system metrics
        system_metrics = get_system_metrics()
        gpu_memory = get_gpu_memory_usage()
        
        # Check pipeline health
        pipeline_health = health_check()
        
        # Count active tasks
        active_tasks = sum(1 for task in task_storage.values() 
                          if task.status in ["pending", "processing"])
        
        return HealthResponse(
            status="healthy" if pipeline_health['status'] == 'healthy' else 'degraded',
            timestamp=datetime.now(),
            version="1.0.0",
            system_metrics=system_metrics,
            gpu_memory=gpu_memory,
            active_tasks=active_tasks
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/analyze", response_model=AnalyzeResponse)
@limiter.limit("10/minute")  # Limit analysis requests to prevent abuse
async def analyze_video(request: Request, req: AnalyzeRequest, background_tasks: BackgroundTasks):
    """Start video analysis task."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Generate task ID
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(req.youtube_url)) % 10000}"
        
        # Create analysis request
        analysis_request = YouTubeAnalysisRequest(
            url=str(req.youtube_url),
            output_dir=req.output_dir,
            frame_rate=req.frame_rate,
            max_duration=req.max_duration,
            include_audio=req.include_audio,
            confidence_threshold=req.confidence_threshold,
            force_full_analysis=req.force_full_analysis
        )
        
        # Create task status
        task_status = TaskStatus(
            task_id=task_id,
            status="pending",
            message="Task created",
            progress=0.0,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Store task
        task_storage[task_id] = task_status
        
        # Add background task
        background_tasks.add_task(
            process_video_analysis, task_id, analysis_request
        )
        
        # Estimate processing duration (rough estimate)
        estimated_duration = 30.0 + (req.max_duration or 300) * 0.1
        
        return AnalyzeResponse(
            task_id=task_id,
            status="pending",
            message="Analysis task started",
            estimated_duration=estimated_duration
        )
        
    except Exception as e:
        logger.error(f"Failed to start analysis: {e}")
        raise HTTPException(status_code=400, detail=str(e))


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
        results=task.results.dict() if task.results else None,
        error_details=task.error_details
    )


@app.get("/tasks", response_model=List[TaskResponse])
async def list_tasks(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of tasks to return")
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
            results=task.results.dict() if task.results else None,
            error_details=task.error_details
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
    
    # Update task status
    task.status = "cancelled"
    task.message = "Task cancelled by user"
    task.updated_at = datetime.now()
    
    return {"message": f"Task {task_id} cancelled"}


@app.get("/metrics", response_model=Dict[str, Any])
async def get_metrics():
    """Get system and pipeline metrics."""
    try:
        system_metrics = get_system_metrics()
        gpu_memory = get_gpu_memory_usage()
        
        # Count tasks by status
        task_counts = {}
        for task in task_storage.values():
            status = task.status
            task_counts[status] = task_counts.get(status, 0) + 1
        
        return {
            "system_metrics": system_metrics,
            "gpu_memory_bytes": gpu_memory,
            "task_counts": task_counts,
            "total_tasks": len(task_storage),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pipeline/info", response_model=Dict[str, Any])
async def get_pipeline_info():
    """Get pipeline configuration and information."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return {
        "pipeline_version": orchestrator.config.get("pipeline", {}).get("version", "1.0.0"),
        "supported_formats": ["YouTube URLs"],
        "max_video_duration": orchestrator.config.get("youtube", {}).get("max_duration", 7200),
        "supported_languages": ["en", "es", "fr", "de", "it"],
        "confidence_threshold_default": 0.75,
        "features": [
            "Video metadata analysis",
            "Thumbnail visual analysis",
            "Audio content analysis",
            "Soccer event detection",
            "Player tracking",
            "Graph analysis",
            "Tactical analytics (pitch control, xT, OBSO)"
        ]
    }


# ===== TACTICAL ANALYTICS ENDPOINTS =====

class TacticalComputeRequest(BaseModel):
    """Request model for computing tactical metrics."""
    players: List[Dict[str, Any]]  # List of {player_id, team_id, position: [x, y]}
    frame_id: int = 0
    grid_shape: List[int] = [12, 16]


class TacticalComputeResponse(BaseModel):
    """Response model for tactical computation."""
    frame_id: int
    pitch_control: Dict[str, Any]
    obso: Dict[str, Any]


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
        from src.analytics.tactical import TacticalAnalyzer, TacticalConfig, PlayerState
        import numpy as np

        # Initialize analyzer
        config = TacticalConfig(grid_shape=tuple(req.grid_shape))
        analyzer = TacticalAnalyzer(config=config)

        # Convert request to PlayerState objects
        players = []
        for p in req.players:
            players.append(PlayerState(
                player_id=p["player_id"],
                team_id=p["team_id"],
                position=np.array(p["position"]),
                velocity=np.array(p.get("velocity")) if p.get("velocity") else None
            ))

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
            }
        )

    except ImportError as e:
        logger.error(f"Tactical analytics not available: {e}")
        raise HTTPException(status_code=503, detail="Tactical analytics module not available")
    except Exception as e:
        logger.error(f"Tactical computation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tactical/info", response_model=Dict[str, Any])
async def get_tactical_info():
    """Get information about tactical analytics capabilities."""
    return {
        "available": True,
        "metrics": [
            {
                "name": "pitch_control",
                "description": "Probability that each team controls each area of the pitch",
                "output_range": [0, 1],
                "interpretation": "0 = away team controls, 1 = home team controls"
            },
            {
                "name": "expected_threat",
                "description": "Expected goal probability from each pitch zone",
                "output_range": [0, 0.5],
                "interpretation": "Higher values near opponent goal"
            },
            {
                "name": "obso",
                "description": "Off-Ball Scoring Opportunity (pitch_control * xT)",
                "output_range": [0, 0.5],
                "interpretation": "Where should attacking players position themselves"
            }
        ],
        "parameters": {
            "grid_shape": {
                "default": [12, 16],
                "description": "Output grid dimensions (rows, columns)"
            },
            "max_speed": {
                "default": 5.0,
                "unit": "m/s",
                "description": "Maximum player running speed"
            },
            "reaction_time": {
                "default": 0.7,
                "unit": "seconds",
                "description": "Time before player reacts"
            },
            "pitch_dimensions": {
                "length": 105.0,
                "width": 68.0,
                "unit": "meters"
            }
        }
    }


@app.get("/tactical/results/{task_id}", response_model=Dict[str, Any])
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

    results_dict = task.results.dict() if hasattr(task.results, 'dict') else task.results
    tactical_data = results_dict.get("tactical_analytics")

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
        task.updated_at = datetime.now()
        
        logger.info(f"Starting analysis for task {task_id}")
        
        # Process video
        result = await orchestrator.process_youtube_video(request)
        
        # Update task status
        task.status = "completed"
        task.message = "Analysis completed successfully"
        task.progress = 1.0
        task.results = result
        task.updated_at = datetime.now()
        
        logger.info(f"Analysis completed for task {task_id}")
        
    except Exception as e:
        # Update task with error
        task.status = "error"
        task.message = f"Analysis failed: {str(e)}"
        task.error_details = str(e)
        task.updated_at = datetime.now()
        
        logger.error(f"Analysis failed for task {task_id}: {e}")


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
                "status_code": exc.status_code
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "internal_error",
                "message": "Internal server error"
            }
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )