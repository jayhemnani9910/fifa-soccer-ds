"""
FastAPI endpoints for YouTube video processing and analysis.
Provides REST API for the Smart YouTube Analyzer functionality.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any
import os
import json
import asyncio
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FIFA Soccer DS - YouTube Analyzer API",
    description="API for processing and analyzing soccer videos from YouTube",
    version="1.0.0"
)

# Pydantic models for request/response
class YouTubeAnalysisRequest(BaseModel):
    """Request model for YouTube video analysis."""
    url: HttpUrl
    output_dir: Optional[str] = None
    frame_rate: Optional[int] = 1  # Extract 1 frame per second by default
    max_duration: Optional[int] = None  # Maximum duration to process (seconds)
    include_audio: Optional[bool] = True
    confidence_threshold: Optional[float] = 0.75

class AnalysisStatusResponse(BaseModel):
    """Response model for analysis status."""
    task_id: str
    status: str  # 'pending', 'processing', 'completed', 'error'
    message: Optional[str] = None
    progress: Optional[float] = None
    results_url: Optional[str] = None

class YouTubeMetadataResponse(BaseModel):
    """Response model for YouTube video metadata."""
    title: str
    description: str
    duration: int
    view_count: int
    like_count: Optional[int]
    channel_title: str
    publish_date: str
    tags: List[str]
    thumbnail_url: str

class AnalysisResultResponse(BaseModel):
    """Response model for analysis results."""
    task_id: str
    video_info: YouTubeMetadataResponse
    analysis_summary: Dict[str, Any]
    events_detected: List[Dict[str, Any]]
    player_analysis: Dict[str, Any]
    visualizations: List[str]
    output_directory: str

# In-memory storage for task status (use Redis/database in production)
task_status = {}

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "FIFA Soccer DS - YouTube Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze": "POST - Start YouTube video analysis",
            "/status/{task_id}": "GET - Check analysis status",
            "/results/{task_id}": "GET - Get analysis results",
            "/metadata/{video_id}": "GET - Get YouTube video metadata"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "youtube-analyzer"}

@app.post("/analyze", response_model=AnalysisStatusResponse)
async def start_youtube_analysis(request: YouTubeAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Start YouTube video analysis in the background.
    
    Args:
        request: YouTube analysis request with URL and options
        
    Returns:
        AnalysisStatusResponse with task ID and initial status
    """
    try:
        # Generate unique task ID
        task_id = f"task_{len(task_status) + 1}_{hash(str(request.url))}"
        
        # Set initial status
        task_status[task_id] = {
            "status": "pending",
            "message": "Analysis queued",
            "progress": 0.0,
            "request": request.dict()
        }
        
        # Add background task
        background_tasks.add_task(process_youtube_video_async, task_id, request)
        
        return AnalysisStatusResponse(
            task_id=task_id,
            status="pending",
            message="Analysis started"
        )
        
    except Exception as e:
        logger.error(f"Failed to start analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{task_id}", response_model=AnalysisStatusResponse)
async def get_analysis_status(task_id: str):
    """
    Get the status of a YouTube analysis task.
    
    Args:
        task_id: Unique task identifier
        
    Returns:
        AnalysisStatusResponse with current status
    """
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    status_info = task_status[task_id]
    return AnalysisStatusResponse(
        task_id=task_id,
        status=status_info["status"],
        message=status_info.get("message"),
        progress=status_info.get("progress"),
        results_url=status_info.get("results_url")
    )

@app.get("/results/{task_id}", response_model=AnalysisResultResponse)
async def get_analysis_results(task_id: str):
    """
    Get the results of a completed YouTube analysis.
    
    Args:
        task_id: Unique task identifier
        
    Returns:
        AnalysisResultResponse with complete analysis results
    """
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    status_info = task_status[task_id]
    
    if status_info["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed yet")
    
    if "results" not in status_info:
        raise HTTPException(status_code=500, detail="Results not found")
    
    return status_info["results"]

@app.get("/metadata/{video_id}")
async def get_youtube_metadata(video_id: str):
    """
    Get metadata for a YouTube video without analysis.
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        YouTubeMetadataResponse with video information
    """
    try:
        # Import with fallback for missing dependencies
        try:
            from ..youtube.metadata_parser import YouTubeMetadataParser
        except ImportError:
            raise HTTPException(status_code=503, detail="YouTube dependencies not installed")
        
        parser = YouTubeMetadataParser()
        # Extract metadata from video ID (need to construct full URL)
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        metadata = parser.extract_metadata(youtube_url)
        
        if not metadata:
            raise HTTPException(status_code=404, detail="Video not found")
        
        return YouTubeMetadataResponse(**metadata)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{task_id}")
async def download_results(task_id: str):
    """
    Download analysis results as a ZIP file.
    
    Args:
        task_id: Unique task identifier
        
    Returns:
        FileResponse with ZIP file containing results
    """
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    status_info = task_status[task_id]
    
    if status_info["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed yet")
    
    output_dir = status_info.get("output_dir")
    if not output_dir or not os.path.exists(output_dir):
        raise HTTPException(status_code=404, detail="Results directory not found")
    
    # Create ZIP file (implement ZIP creation logic here)
    zip_path = f"{output_dir}/results.zip"
    
    if not os.path.exists(zip_path):
        # Create ZIP file from results directory
        import zipfile
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file != "results.zip":
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, output_dir)
                        zipf.write(file_path, arcname)
    
    return FileResponse(
        zip_path,
        media_type='application/zip',
        filename=f"youtube_analysis_{task_id}.zip"
    )

@app.delete("/task/{task_id}")
async def delete_task(task_id: str):
    """
    Delete a task and its results.
    
    Args:
        task_id: Unique task identifier
        
    Returns:
        JSONResponse with deletion status
    """
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    try:
        # Clean up files
        status_info = task_status[task_id]
        output_dir = status_info.get("output_dir")
        if output_dir and os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
        
        # Remove from status
        del task_status[task_id]
        
        return {"message": f"Task {task_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Failed to delete task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_youtube_video_async(task_id: str, request: YouTubeAnalysisRequest):
    """
    Background task to process YouTube video analysis.
    
    Args:
        task_id: Unique task identifier
        request: YouTube analysis request
    """
    try:
        # Update status to processing
        task_status[task_id]["status"] = "processing"
        task_status[task_id]["message"] = "Downloading video..."
        task_status[task_id]["progress"] = 0.1
        
        # Set output directory
        output_dir = request.output_dir or f"./outputs/youtube_analysis_{task_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Import pipeline functions with fallback
        try:
            from ..pipeline_full import process_youtube_video
        except ImportError:
            # Fallback: implement basic YouTube processing here
            task_status[task_id]["status"] = "error"
            task_status[task_id]["message"] = "Pipeline module not available"
            return
        
        # Update progress
        task_status[task_id]["progress"] = 0.3
        task_status[task_id]["message"] = "Processing video..."
        
        # Process the video
        results = process_youtube_video(
            youtube_url=str(request.url),
            output_dir=Path(output_dir),
            sample_duration=60.0,  # Default 60 seconds
            force_full_analysis=False  # Default conservative analysis
        )
        
        # Update progress
        task_status[task_id]["progress"] = 0.9
        task_status[task_id]["message"] = "Finalizing results..."
        
        # Format results for API response
        api_results = format_analysis_results(results, task_id, output_dir)
        
        # Mark as completed
        task_status[task_id]["status"] = "completed"
        task_status[task_id]["message"] = "Analysis completed successfully"
        task_status[task_id]["progress"] = 1.0
        task_status[task_id]["results"] = api_results
        task_status[task_id]["output_dir"] = output_dir
        task_status[task_id]["results_url"] = f"/results/{task_id}"
        
    except Exception as e:
        logger.error(f"Analysis failed for task {task_id}: {e}")
        task_status[task_id]["status"] = "error"
        task_status[task_id]["message"] = f"Analysis failed: {str(e)}"

def format_analysis_results(results: Dict[str, Any], task_id: str, output_dir: str) -> AnalysisResultResponse:
    """
    Format analysis results for API response.
    
    Args:
        results: Raw analysis results from pipeline
        task_id: Task identifier
        output_dir: Output directory path
        
    Returns:
        AnalysisResultResponse formatted results
    """
    # Extract video info (implement based on your results structure)
    video_info = results.get("video_info", {})
    
    # Extract analysis summary
    analysis_summary = results.get("summary", {})
    
    # Extract events
    events_detected = results.get("events", [])
    
    # Extract player analysis
    player_analysis = results.get("player_analysis", {})
    
    # Find visualization files
    visualizations = []
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                visualizations.append(f"/outputs/{os.path.basename(output_dir)}/{file}")
    
    return AnalysisResultResponse(
        task_id=task_id,
        video_info=YouTubeMetadataResponse(**video_info),
        analysis_summary=analysis_summary,
        events_detected=events_detected,
        player_analysis=player_analysis,
        visualizations=visualizations,
        output_directory=output_dir
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)