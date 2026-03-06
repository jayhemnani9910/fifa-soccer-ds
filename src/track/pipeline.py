"""End-to-end tracking pipeline that runs detection followed by association."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from src.detect.infer import InferenceConfig, extract_detections, load_model
from src.detect.video_streaming import stream_video_detections
from src.track.bytetrack_runtime import ByteTrackRuntime, Tracklets


@dataclass(slots=True)
class TrackingPipelineConfig:
    """Configuration bundle for the tracking pipeline."""

    detector: InferenceConfig = field(default_factory=InferenceConfig)
    min_confidence: float = 0.25
    distance_threshold: float = 80.0
    max_age: int = 15
    max_frames: int = 120


def filter_detections(detections: Iterable[dict], min_confidence: float) -> list[dict]:
    filtered: list[dict] = []
    for det in detections:
        bbox = det.get("bbox")
        confidence = det.get("confidence")
        if confidence is None and "score" in det:
            confidence = det["score"]
        if not bbox:
            continue
        if confidence is not None and confidence < min_confidence:
            continue
        filtered.append(
            {
                "bbox": bbox,
                "score": float(confidence or 0.0),
                "class_id": det.get("class_id"),
                "class_name": det.get("class_name"),
            }
        )
    return filtered


def run_tracking_pipeline(
    video_path: str | Path,
    config: TrackingPipelineConfig | None = None,
) -> list[Tracklets]:
    """Execute detection and tracking over a short clip."""

    cfg = config or TrackingPipelineConfig()
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    detector = load_model(cfg.detector)
    tracker = ByteTrackRuntime(
        min_confidence=cfg.min_confidence,
        distance_threshold=cfg.distance_threshold,
        max_age=cfg.max_age,
    )

    track_history: list[Tracklets] = []

    for frame_idx, result in stream_video_detections(
        detector, video, conf=cfg.detector.confidence, max_frames=cfg.max_frames
    ):
        detections = extract_detections(result)
        filtered = filter_detections(detections, min_confidence=cfg.min_confidence)
        tracklets = tracker.update(frame_id=frame_idx, detections=filtered)
        track_history.append(tracklets)

    return track_history
