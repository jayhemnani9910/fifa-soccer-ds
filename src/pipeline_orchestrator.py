"""Async orchestration for the verified YouTube processing pipeline."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml

from src.pipeline_full import PipelineConfig
from src.pipeline_full import process_youtube_video as run_youtube_pipeline
from src.schemas import (
    PipelineOutput,
    PlayerAnalysis,
    SoccerClassification,
    VideoMetadata,
    YouTubeAnalysisRequest,
    extract_youtube_video_id,
)
from src.utils.health_checks import health_check
from src.utils.output_paths import create_analysis_output_dir

LOGGER = logging.getLogger(__name__)

PipelineProcessor = Callable[..., dict[str, Any]]

_DEFAULT_CONFIG: dict[str, Any] = {
    "pipeline": {"name": "YouTube Soccer Analyzer", "version": "0.1.0"},
    "youtube": {"video": {"max_duration_seconds": 1800}},
    "output": {"base_dir": "outputs"},
    "processing": {"max_frames": 300},
    "analytics": {"enable_tactical": False},
}


class PipelineError(RuntimeError):
    """Raised when the end-to-end pipeline cannot produce a valid result."""


class WeightsNotConfiguredError(PipelineError):
    """Raised when no usable YOLO weights checkpoint is configured."""


def describe_weights_problem(weights_path: str | None) -> str | None:
    """Return None if `weights_path` is a usable checkpoint file, else why not.

    Only checks configuration and filesystem state; never loads the checkpoint,
    so this is cheap enough to call on every readiness probe.
    """
    if not weights_path:
        return "YOLO_WEIGHTS is not set. Configure it to point at a trusted YOLO checkpoint file."
    path = Path(weights_path)
    try:
        if not path.is_file():
            return "The configured YOLO_WEIGHTS file was not found."
        if not os.access(path, os.R_OK):
            return "The configured YOLO_WEIGHTS file is not readable."
        if path.stat().st_size == 0:
            return "The configured YOLO_WEIGHTS file is empty."
    except OSError:
        return "The configured YOLO_WEIGHTS file could not be checked."
    return None


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_publish_date(raw: Any) -> datetime | None:
    if isinstance(raw, datetime):
        return raw if raw.tzinfo is not None else raw.replace(tzinfo=UTC)
    if isinstance(raw, str):
        for parser in (datetime.fromisoformat, lambda value: datetime.strptime(value, "%Y%m%d")):
            try:
                parsed = parser(raw)
                return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
            except ValueError:
                continue
    return None


def _content_type(
    classification: Mapping[str, Any],
) -> Literal["match", "highlight", "training", "interview", "analysis", "other"]:
    raw = (
        classification.get("analysis_breakdown", {})
        .get("metadata_analysis", {})
        .get("content_type", "other")
    )
    aliases: dict[
        str, Literal["match", "highlight", "training", "interview", "analysis", "other"]
    ] = {
        "highlights": "highlight",
        "game": "match",
        "match": "match",
        "highlight": "highlight",
        "training": "training",
        "interview": "interview",
        "analysis": "analysis",
        "other": "other",
    }
    normalized = aliases.get(str(raw).lower(), str(raw).lower())
    return aliases.get(normalized, "other")


class PipelineOrchestrator:
    """Run the synchronous CV pipeline without blocking the API event loop."""

    def __init__(
        self,
        config_path: str | Path = "configs/youtube_pipeline.yaml",
        *,
        processor: PipelineProcessor = run_youtube_pipeline,
    ) -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.processor = processor

    def _load_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            LOGGER.warning("Pipeline config not found; using safe defaults: %s", self.config_path)
            return _deep_merge({}, _DEFAULT_CONFIG)

        try:
            raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"Could not load pipeline config {self.config_path}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise ValueError(f"Pipeline config must be a mapping: {self.config_path}")
        return _deep_merge(_DEFAULT_CONFIG, raw)

    def _pipeline_config(self, request: YouTubeAnalysisRequest) -> PipelineConfig:
        configured_limit = int(self.config.get("processing", {}).get("max_frames", 300))
        requested_frames = round((request.max_duration or 300) * request.frame_rate)
        max_frames = max(1, min(configured_limit, requested_frames))
        weights = os.getenv("YOLO_WEIGHTS") or ""
        problem = describe_weights_problem(weights)
        if problem is not None:
            raise WeightsNotConfiguredError(problem)
        return PipelineConfig(
            weights=weights,
            confidence=request.confidence_threshold,
            min_confidence=request.confidence_threshold,
            gnn_weights=os.getenv("GNN_WEIGHTS", ""),
            max_frames=max_frames,
            enable_tactical_analytics=bool(
                self.config.get("analytics", {}).get("enable_tactical", False)
            ),
            calibration_path=str(self.config.get("analytics", {}).get("calibration_path", "")),
        )

    @health_check
    async def process_youtube_video(self, request: YouTubeAnalysisRequest) -> PipelineOutput:
        """Run classification, download, detection, tracking, and graph generation."""
        started = time.monotonic()
        output_dir = create_analysis_output_dir(request.output_dir, uuid.uuid4().hex)
        configured_duration = int(
            self.config.get("youtube", {}).get("video", {}).get("max_duration_seconds", 1800)
        )
        max_duration = min(request.max_duration or configured_duration, configured_duration)
        pipeline_config = self._pipeline_config(request)

        try:
            raw_result = await asyncio.to_thread(
                self.processor,
                str(request.url),
                output_dir,
                pipeline_config,
                request.sample_duration,
                request.force_full_analysis,
                include_audio=request.include_audio,
                max_duration=max_duration,
            )
            return self._to_pipeline_output(
                request=request,
                raw_result=raw_result,
                output_dir=output_dir,
                elapsed=max(time.monotonic() - started, 1e-9),
            )
        except asyncio.CancelledError:
            LOGGER.info("Pipeline task cancelled for %s", request.url)
            raise
        except Exception as exc:
            LOGGER.exception("YouTube pipeline failed")
            raise PipelineError(f"Pipeline execution failed: {exc}") from exc

    def _to_pipeline_output(
        self,
        *,
        request: YouTubeAnalysisRequest,
        raw_result: Mapping[str, Any],
        output_dir: Path,
        elapsed: float,
    ) -> PipelineOutput:
        classification = raw_result.get("classification")
        if not isinstance(classification, Mapping):
            raise PipelineError("Pipeline result omitted classification evidence")

        video_info = classification.get("processing_info", {}).get("video_info", {})
        if not isinstance(video_info, Mapping):
            video_info = {}
        metadata = VideoMetadata(
            video_id=str(video_info.get("id") or extract_youtube_video_id(str(request.url))),
            title=str(video_info.get("title") or "")[:200],
            description=str(video_info.get("description") or ""),
            duration_seconds=(
                int(video_info["duration"]) if video_info.get("duration") is not None else None
            ),
            channel_title=(
                str(video_info["uploader"])[:100] if video_info.get("uploader") else None
            ),
            channel_id=(str(video_info["channel_id"]) if video_info.get("channel_id") else None),
            view_count=(
                int(video_info["view_count"]) if video_info.get("view_count") is not None else None
            ),
            like_count=video_info.get("like_count"),
            comment_count=video_info.get("comment_count"),
            tags=list(video_info.get("tags") or []),
            categories=list(video_info.get("categories") or []),
            publish_date=_parse_publish_date(video_info.get("upload_date")),
            resolution=(str(video_info["resolution"]) if video_info.get("resolution") else None),
            fps=(float(video_info["fps"]) if video_info.get("fps") is not None else None),
            audio_codec=(str(video_info["acodec"]) if video_info.get("acodec") else None),
            video_codec=(str(video_info["vcodec"]) if video_info.get("vcodec") else None),
        )

        summary = raw_result.get("pipeline_summary", {})
        if not isinstance(summary, Mapping):
            summary = {}
        track_ids = [int(item) for item in summary.get("unique_track_ids", [])]
        is_soccer = bool(classification.get("is_soccer"))

        warnings = [
            "Event detection and per-player match statistics are not implemented.",
            "Tracker IDs are not verified player identities and may switch or fragment.",
        ]
        if not is_soccer:
            warnings = [
                "Full CV analysis was skipped because the classifier marked the video non-soccer."
            ]
        failures = summary.get("failures", {})
        if isinstance(failures, Mapping) and any(
            isinstance(value, int | float) and value > 0 for value in failures.values()
        ):
            warnings.append(f"Pipeline completed with partial frame failures: {dict(failures)}")

        success_rate = summary.get("processing_success_rate")
        if success_rate is None:
            measured_success_rate = None
        elif (
            isinstance(success_rate, bool)
            or not isinstance(success_rate, int | float)
            or not math.isfinite(float(success_rate))
            or not 0.0 <= float(success_rate) <= 1.0
        ):
            raise ValueError("processing_success_rate must be finite and between 0 and 1")
        else:
            measured_success_rate = float(success_rate)

        output_files: dict[str, str] = {}
        candidates = {
            "analysis": output_dir / "youtube_analysis.json",
            "pipeline_summary": output_dir / "soccer_analysis" / "pipeline_summary.json",
            "graph": output_dir / "soccer_analysis" / "graphs" / "final_graph.json",
            "report": output_dir / "analysis_report.md",
        }
        for name, path in candidates.items():
            if path.exists():
                output_files[name] = str(path)

        return PipelineOutput(
            pipeline_version=str(self.config.get("pipeline", {}).get("version", "0.1.0")),
            processing_timestamp=datetime.now(UTC),
            processing_duration_seconds=elapsed,
            input_source="youtube",
            input_url=request.url,
            input_metadata=metadata,
            soccer_classification=SoccerClassification(
                is_soccer=is_soccer,
                soccer_confidence=float(classification.get("confidence") or 0.0),
                content_type=_content_type(classification),
                events_detected=[],
                total_events=None,
                detection_quality=None,
                processing_success_rate=measured_success_rate,
            ),
            player_analysis=PlayerAnalysis(
                total_players_detected=len(track_ids),
                player_tracks=[],
                teams_detected=0,
                team_colors=[],
                avg_track_length=None,
                player_positions={},
            ),
            tactical_analytics=(
                dict(raw_result["tactical_analytics"])
                if isinstance(raw_result.get("tactical_analytics"), Mapping)
                else None
            ),
            output_files=output_files,
            summary_report=output_files.get("report"),
            errors=[],
            warnings=warnings,
        )


async def create_pipeline_orchestrator(
    config_path: str | Path = "configs/youtube_pipeline.yaml",
) -> PipelineOrchestrator:
    """Create the API orchestrator."""
    return PipelineOrchestrator(config_path)


async def process_youtube_url(
    url: str, output_dir: str | None = None, **kwargs: Any
) -> PipelineOutput:
    """Validate and process one YouTube URL."""
    request = YouTubeAnalysisRequest.model_validate(
        {"url": url, "output_dir": output_dir, **kwargs}
    )
    orchestrator = await create_pipeline_orchestrator()
    return await orchestrator.process_youtube_video(request)


__all__ = [
    "PipelineError",
    "PipelineOrchestrator",
    "create_pipeline_orchestrator",
    "process_youtube_url",
]
