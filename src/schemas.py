"""
Schema validation for FIFA Soccer DS YouTube Pipeline

This module provides Pydantic models for input/output validation
to ensure data integrity across the entire pipeline.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


class DetectionBox(BaseModel):
    """Validation schema for object detection bounding boxes."""

    x1: float = Field(ge=0, le=1, description="Normalized top-left x coordinate")
    y1: float = Field(ge=0, le=1, description="Normalized top-left y coordinate")
    x2: float = Field(ge=0, le=1, description="Normalized bottom-right x coordinate")
    y2: float = Field(ge=0, le=1, description="Normalized bottom-right y coordinate")
    confidence: float = Field(ge=0, le=1, description="Detection confidence score")
    class_id: int = Field(ge=0, description="Object class identifier")

    @field_validator("x2")
    @classmethod
    def validate_x2(cls, v, info):
        x1 = info.data.get("x1")
        if x1 is not None and v <= x1:
            raise ValueError("x2 must be greater than x1")
        return v

    @field_validator("y2")
    @classmethod
    def validate_y2(cls, v, info):
        y1 = info.data.get("y1")
        if y1 is not None and v <= y1:
            raise ValueError("y2 must be greater than y1")
        return v


class PlayerTrack(BaseModel):
    """Validation schema for player tracking data."""

    track_id: int = Field(ge=0, description="Unique track identifier")
    frame_number: int = Field(ge=0, description="Frame timestamp")
    bbox: DetectionBox = Field(description="Bounding box coordinates")
    class_id: int = Field(ge=0, description="Player class identifier")
    confidence: float = Field(ge=0, le=1, description="Detection confidence")

    # Optional tracking features
    velocity_x: float | None = Field(default=None, description="X velocity")
    velocity_y: float | None = Field(default=None, description="Y velocity")
    appearance_features: list[float] | None = Field(
        default=None, description="Appearance embeddings"
    )


class SoccerEvent(BaseModel):
    """Validation schema for detected soccer events."""

    timestamp: float = Field(ge=0, description="Event timestamp in seconds")
    event_type: Literal[
        "goal", "foul", "card", "substitution", "corner", "free_kick", "offside", "penalty"
    ]
    confidence: float = Field(ge=0, le=1, description="Event detection confidence")
    description: str | None = Field(default=None, description="Event description")

    # Event details
    players_involved: list[int] | None = Field(
        default=None, description="Track IDs of involved players"
    )
    location: DetectionBox | None = Field(default=None, description="Event location")
    team: str | None = Field(default=None, description="Team involved")


class VideoMetadata(BaseModel):
    """Validation schema for YouTube video metadata."""

    video_id: str = Field(min_length=11, max_length=11, description="YouTube video ID")
    title: str = Field(min_length=1, max_length=200, description="Video title")
    description: str = Field(default="", description="Video description")
    duration_seconds: int = Field(ge=1, le=28800, description="Video duration (max 8 hours)")

    # Channel info
    channel_title: str = Field(min_length=1, max_length=100, description="Channel name")
    channel_id: str = Field(min_length=1, description="Channel identifier")

    # Engagement metrics
    view_count: int = Field(ge=0, description="View count")
    like_count: int | None = Field(default=None, ge=0, description="Like count")
    comment_count: int | None = Field(default=None, ge=0, description="Comment count")

    # Content info
    tags: list[str] = Field(default_factory=list, description="Video tags")
    categories: list[str] = Field(default_factory=list, description="Content categories")
    publish_date: datetime = Field(description="Publication date")

    # Quality metrics
    resolution: str = Field(description="Video resolution")
    fps: float = Field(gt=0, le=120, description="Frames per second")
    audio_codec: str = Field(description="Audio codec")
    video_codec: str = Field(description="Video codec")


class AudioTranscription(BaseModel):
    """Validation schema for audio transcription results."""

    text: str = Field(description="Transcribed text")
    language: str = Field(min_length=2, max_length=10, description="Detected language")
    confidence: float = Field(ge=0, le=1, description="Transcription confidence")

    # Timing information
    segments: list[dict[str, Any]] = Field(default_factory=list, description="Timestamped segments")
    duration_seconds: float = Field(gt=0, description="Audio duration")

    # Soccer-related content
    soccer_keywords_found: list[str] = Field(
        default_factory=list, description="Soccer terms detected"
    )
    commentary_confidence: float = Field(ge=0, le=1, description="Sports commentary confidence")


class PlayerAnalysis(BaseModel):
    """Validation schema for player tracking and analysis."""

    total_players_detected: int = Field(ge=0, description="Number of unique players")
    player_tracks: list[PlayerTrack] = Field(
        default_factory=list, description="Individual player tracks"
    )

    # Team analysis
    teams_detected: int = Field(ge=0, le=2, description="Number of teams identified")
    team_colors: list[str] = Field(default_factory=list, description="Detected team colors")

    # Performance metrics
    avg_track_length: float = Field(ge=0, description="Average track duration")
    player_positions: dict[str, list[tuple]] = Field(
        default_factory=dict, description="Player position history"
    )


class SoccerClassification(BaseModel):
    """Validation schema for soccer content classification."""

    is_soccer: bool = Field(description="Whether content is soccer-related")
    soccer_confidence: float = Field(ge=0, le=1, description="Soccer classification confidence")

    # Content analysis
    content_type: Literal["match", "highlight", "training", "interview", "analysis", "other"]
    match_phase: (
        Literal["pre_match", "first_half", "half_time", "second_half", "full_time"] | None
    ) = Field(default=None)

    # Event statistics
    events_detected: list[SoccerEvent] = Field(default_factory=list, description="Detected events")
    total_events: int = Field(ge=0, description="Total number of events")

    # Quality metrics
    detection_quality: float = Field(ge=0, le=1, description="Overall detection quality")
    processing_success_rate: float = Field(ge=0, le=1, description="Successful processing rate")


class PipelineOutput(BaseModel):
    """Validation schema for complete pipeline output."""

    # Processing info
    pipeline_version: str = Field(description="Pipeline version used")
    processing_timestamp: datetime = Field(description="Processing completion time")
    processing_duration_seconds: float = Field(gt=0, description="Total processing time")

    # Input information
    input_source: Literal["youtube", "frame_directory", "video_file"]
    input_url: HttpUrl | None = Field(default=None, description="Input source URL")
    input_metadata: VideoMetadata = Field(description="Input video metadata")

    # Analysis results
    soccer_classification: SoccerClassification = Field(description="Soccer content classification")
    player_analysis: PlayerAnalysis = Field(description="Player tracking results")
    audio_analysis: AudioTranscription | None = Field(
        default=None, description="Audio transcription"
    )

    # Output files
    output_files: dict[str, str] = Field(default_factory=dict, description="Generated output files")
    summary_report: str | None = Field(default=None, description="Summary report path")

    # Error information (if any)
    errors: list[str] = Field(default_factory=list, description="Processing errors")
    warnings: list[str] = Field(default_factory=list, description="Processing warnings")


class YouTubeAnalysisRequest(BaseModel):
    """Validation schema for YouTube analysis API requests."""

    url: HttpUrl = Field(description="YouTube video URL")
    output_dir: str | None = Field(default=None, description="Output directory")

    # Processing options
    frame_rate: float = Field(default=1.0, gt=0, le=30, description="Frame extraction rate")
    max_duration: int | None = Field(
        default=None, gt=0, le=7200, description="Max processing duration"
    )
    include_audio: bool = Field(default=True, description="Include audio analysis")
    confidence_threshold: float = Field(
        default=0.75, ge=0, le=1, description="Detection confidence threshold"
    )

    # Advanced options
    force_full_analysis: bool = Field(
        default=False, description="Force analysis even with low confidence"
    )
    sample_duration: float = Field(default=60.0, gt=0, le=300, description="Audio sample duration")

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v):
        if v is not None:
            # Sanitize directory path
            v = re.sub(r"[^\w\-_\.]", "_", v)
            if len(v) > 100:
                raise ValueError("Output directory name too long")
        return v


class TaskStatus(BaseModel):
    """Validation schema for task status tracking."""

    task_id: str = Field(min_length=1, description="Unique task identifier")
    status: Literal["pending", "processing", "completed", "error", "cancelled"]
    message: str | None = Field(default=None, description="Status message")
    progress: float = Field(ge=0, le=1, description="Progress percentage")

    # Timing information
    created_at: datetime = Field(description="Task creation time")
    updated_at: datetime = Field(description="Last update time")

    # Results (when completed)
    results: PipelineOutput | None = Field(default=None, description="Analysis results")
    error_details: str | None = Field(default=None, description="Error details if failed")


# Configuration validation schemas
class PipelineConfig(BaseModel):
    """Validation schema for pipeline configuration."""

    pipeline: dict[str, Any] = Field(description="Pipeline settings")
    youtube: dict[str, Any] = Field(description="YouTube settings")
    audio: dict[str, Any] = Field(description="Audio processing settings")
    soccer_analysis: dict[str, Any] = Field(description="Soccer analysis settings")
    output: dict[str, Any] = Field(description="Output settings")
    error_handling: dict[str, Any] = Field(description="Error handling settings")
    monitoring: dict[str, Any] = Field(description="Monitoring settings")
    api: dict[str, Any] = Field(description="API settings")
    validation: dict[str, Any] = Field(description="Validation settings")


# Utility functions for schema validation
def validate_youtube_url(url: str) -> bool:
    """Validate YouTube URL format."""
    youtube_pattern = r"^(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})"
    return bool(re.match(youtube_pattern, url))


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to be safe for all filesystems."""
    # Remove or replace unsafe characters
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    # Remove control characters
    filename = "".join(char for char in filename if ord(char) >= 32)
    # Limit length
    if len(filename) > 200:
        name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
        filename = name[: 200 - len(ext) - 1] + "." + ext if ext else name[:200]
    return filename


def validate_video_duration(duration_seconds: int, max_duration: int = 28800) -> bool:
    """Validate video duration is within acceptable limits."""
    return 1 <= duration_seconds <= max_duration


def validate_video_codec(codec: str, supported_codecs: list[str] = None) -> bool:
    """Validate video codec is supported.

    Args:
        codec: Video codec string
        supported_codecs: List of supported codecs (defaults to common ones)

    Returns:
        True if codec is supported, False otherwise
    """
    if supported_codecs is None:
        supported_codecs = ["h264", "h265", "hevc", "vp9", "vp8", "av1", "mpeg4", "wmv", "avi"]

    codec_lower = codec.lower() if codec else ""
    return any(supported in codec_lower for supported in supported_codecs)


def validate_audio_codec(codec: str, supported_codecs: list[str] = None) -> bool:
    """Validate audio codec is supported.

    Args:
        codec: Audio codec string
        supported_codecs: List of supported codecs (defaults to common ones)

    Returns:
        True if codec is supported, False otherwise
    """
    if supported_codecs is None:
        supported_codecs = ["aac", "mp3", "opus", "vorbis", "flac", "pcm", "wav", "ac3", "dts"]

    codec_lower = codec.lower() if codec else ""
    return any(supported in codec_lower for supported in supported_codecs)


def validate_fps(fps: float, min_fps: float = 1.0, max_fps: float = 120.0) -> bool:
    """Validate frames per second is within acceptable range.

    Args:
        fps: Frames per second
        min_fps: Minimum acceptable FPS
        max_fps: Maximum acceptable FPS

    Returns:
        True if FPS is within range, False otherwise
    """
    return min_fps <= fps <= max_fps


def validate_resolution(resolution: str, min_width: int = 320, min_height: int = 240) -> bool:
    """Validate video resolution meets minimum requirements.

    Args:
        resolution: Resolution string (e.g., "1920x1080")
        min_width: Minimum width in pixels
        min_height: Minimum height in pixels

    Returns:
        True if resolution meets requirements, False otherwise
    """
    try:
        if "x" in resolution:
            width, height = resolution.lower().split("x")
            width = int(width)
            height = int(height)
            return width >= min_width and height >= min_height
        return False
    except (ValueError, AttributeError):
        return False


def validate_video_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Comprehensive video metadata validation.

    Args:
        metadata: Video metadata dictionary

    Returns:
        Dictionary with validation results and issues
    """
    issues = []
    warnings = []

    # Validate duration
    duration = metadata.get("duration_seconds", 0)
    if not validate_video_duration(duration):
        issues.append(f"Invalid duration: {duration} seconds")

    # Validate video codec
    video_codec = metadata.get("video_codec", "")
    if video_codec and not validate_video_codec(video_codec):
        warnings.append(f"Potentially unsupported video codec: {video_codec}")

    # Validate audio codec
    audio_codec = metadata.get("audio_codec", "")
    if audio_codec and not validate_audio_codec(audio_codec):
        warnings.append(f"Potentially unsupported audio codec: {audio_codec}")

    # Validate FPS
    fps = metadata.get("fps", 0)
    if fps and not validate_fps(fps):
        issues.append(f"Invalid FPS: {fps}")

    # Validate resolution
    resolution = metadata.get("resolution", "")
    if resolution and not validate_resolution(resolution):
        warnings.append(f"Low resolution: {resolution}")

    # Check for missing critical fields
    critical_fields = ["video_id", "title", "channel_title", "duration_seconds"]
    for field in critical_fields:
        if field not in metadata or not metadata[field]:
            issues.append(f"Missing critical field: {field}")

    # Determine overall validation status
    if issues:
        status = "invalid"
    elif warnings:
        status = "valid_with_warnings"
    else:
        status = "valid"

    return {
        "status": status,
        "issues": issues,
        "warnings": warnings,
        "validation_details": {
            "duration_valid": validate_video_duration(duration),
            "video_codec_supported": validate_video_codec(video_codec) if video_codec else True,
            "audio_codec_supported": validate_audio_codec(audio_codec) if audio_codec else True,
            "fps_valid": validate_fps(fps) if fps else True,
            "resolution_acceptable": validate_resolution(resolution) if resolution else True,
        },
    }


# Export all schemas
__all__ = [
    "DetectionBox",
    "PlayerTrack",
    "SoccerEvent",
    "VideoMetadata",
    "AudioTranscription",
    "PlayerAnalysis",
    "SoccerClassification",
    "PipelineOutput",
    "YouTubeAnalysisRequest",
    "TaskStatus",
    "PipelineConfig",
    "validate_youtube_url",
    "sanitize_filename",
    "validate_video_duration",
    "validate_video_codec",
    "validate_audio_codec",
    "validate_fps",
    "validate_resolution",
    "validate_video_metadata",
]
