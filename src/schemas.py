"""
Schema validation for FIFA Soccer DS YouTube Pipeline

This module provides Pydantic models for input/output validation
to ensure data integrity across the entire pipeline.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal
from urllib.parse import parse_qs, urlsplit

from pydantic import BaseModel, Field, HttpUrl, field_validator

from src.utils.output_paths import validate_output_name

_YOUTUBE_HOSTS = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
        "youtube-nocookie.com",
        "www.youtube-nocookie.com",
    }
)
_YOUTUBE_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}\Z")


def extract_youtube_video_id(url: str) -> str:
    """Validate a YouTube video URL and return its canonical 11-character ID."""
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        if (
            parsed.scheme != "https"
            or host not in _YOUTUBE_HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in {None, 443}
        ):
            raise ValueError("URL must use HTTPS on an approved YouTube host")

        if host == "youtu.be":
            video_id = parsed.path.strip("/").split("/", 1)[0]
        elif parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            path_parts = [part for part in parsed.path.split("/") if part]
            if len(path_parts) != 2 or path_parts[0] not in {"embed", "live", "shorts", "v"}:
                raise ValueError("URL does not identify a single YouTube video")
            video_id = path_parts[1]

        if _YOUTUBE_VIDEO_ID.fullmatch(video_id) is None:
            raise ValueError("YouTube video ID must contain exactly 11 safe characters")
        return video_id
    except ValueError as exc:
        raise ValueError("Invalid YouTube video URL") from exc


def validate_youtube_url(url: str) -> bool:
    """Return whether *url* is an HTTPS URL for a single YouTube video."""
    try:
        extract_youtube_video_id(url)
    except ValueError:
        return False
    return True


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
    duration_seconds: int | None = Field(
        default=None,
        ge=0,
        le=28800,
        description="Video duration in seconds when supplied by upstream metadata",
    )

    # Channel info
    channel_title: str | None = Field(
        default=None, min_length=1, max_length=100, description="Channel name"
    )
    channel_id: str | None = Field(default=None, min_length=1, description="Channel identifier")

    # Engagement metrics
    view_count: int | None = Field(default=None, ge=0, description="View count when available")
    like_count: int | None = Field(default=None, ge=0, description="Like count")
    comment_count: int | None = Field(default=None, ge=0, description="Comment count")

    # Content info
    tags: list[str] = Field(default_factory=list, description="Video tags")
    categories: list[str] = Field(default_factory=list, description="Content categories")
    publish_date: datetime | None = Field(
        default=None, description="Publication date when supplied by upstream metadata"
    )

    # Quality metrics
    resolution: str | None = Field(default=None, description="Video resolution when available")
    fps: float | None = Field(default=None, gt=0, le=120, description="Frames per second")
    audio_codec: str | None = Field(default=None, description="Audio codec")
    video_codec: str | None = Field(default=None, description="Video codec")


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
    avg_track_length: float | None = Field(
        default=None, ge=0, description="Average track duration when measured"
    )
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
    total_events: int | None = Field(
        default=None, ge=0, description="Total number of events when event detection was run"
    )

    # Quality metrics
    detection_quality: float | None = Field(
        default=None, ge=0, le=1, description="Overall detection quality when evaluated"
    )
    processing_success_rate: float | None = Field(
        default=None, ge=0, le=1, description="Successful processing rate when measured"
    )


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
    tactical_analytics: dict[str, Any] | None = Field(
        default=None, description="Measured tactical aggregates when team classification succeeds"
    )
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
    include_audio: bool = Field(
        default=False, description="Include optional Whisper/librosa audio analysis"
    )
    confidence_threshold: float = Field(
        default=0.75, ge=0, le=1, description="Detection confidence threshold"
    )

    # Advanced options
    force_full_analysis: bool = Field(
        default=False, description="Force analysis even with low confidence"
    )
    sample_duration: float = Field(default=60.0, gt=0, le=300, description="Audio sample duration")

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: HttpUrl) -> HttpUrl:
        if not validate_youtube_url(str(value)):
            raise ValueError("url must identify a video on an approved HTTPS YouTube host")
        return value

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, value: str | None) -> str | None:
        return validate_output_name(value) if value is not None else None


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
    return 0 <= duration_seconds <= max_duration


def validate_video_codec(codec: str, supported_codecs: list[str] | None = None) -> bool:
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


def validate_audio_codec(codec: str, supported_codecs: list[str] | None = None) -> bool:
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
            width_text, height_text = resolution.lower().split("x")
            width = int(width_text)
            height = int(height_text)
            return width >= min_width and height >= min_height
        return False
    except (ValueError, AttributeError):
        return False


def validate_video_metadata(metadata: BaseModel | dict[str, Any]) -> dict[str, Any]:
    """Comprehensive video metadata validation.

    Args:
        metadata: Video metadata dictionary

    Returns:
        Dictionary with validation results and issues
    """
    values = metadata.model_dump() if isinstance(metadata, BaseModel) else metadata
    issues = []
    warnings = []

    # Validate duration
    duration = values.get("duration_seconds")
    duration_valid = duration is None or validate_video_duration(duration)
    if not duration_valid:
        issues.append(f"Invalid duration: {duration} seconds")
    elif duration is None:
        warnings.append("Video duration is unavailable")

    # Validate video codec
    video_codec = values.get("video_codec", "")
    if video_codec and not validate_video_codec(video_codec):
        warnings.append(f"Potentially unsupported video codec: {video_codec}")

    # Validate audio codec
    audio_codec = values.get("audio_codec", "")
    if audio_codec and not validate_audio_codec(audio_codec):
        warnings.append(f"Potentially unsupported audio codec: {audio_codec}")

    # Validate FPS
    fps = values.get("fps", 0)
    if fps and not validate_fps(fps):
        issues.append(f"Invalid FPS: {fps}")

    # Validate resolution
    resolution = values.get("resolution", "")
    if resolution and not validate_resolution(resolution):
        warnings.append(f"Low resolution: {resolution}")

    # Check for missing critical fields
    critical_fields = ["video_id", "title"]
    for field in critical_fields:
        if field not in values or not values[field]:
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
            "duration_valid": duration_valid,
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
    "extract_youtube_video_id",
    "validate_youtube_url",
    "sanitize_filename",
    "validate_video_duration",
    "validate_video_codec",
    "validate_audio_codec",
    "validate_fps",
    "validate_resolution",
    "validate_video_metadata",
]
