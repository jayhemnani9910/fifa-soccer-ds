"""Shared video streaming utilities for detection and tracking pipelines."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None  # type: ignore[assignment]


def stream_video_detections(
    model: YOLO,
    video_path: str | Path,
    conf: float = 0.25,
    max_frames: int | None = None,
) -> Iterator[tuple[int, object]]:
    """Stream detection results from a video file using YOLO model.

    Args:
        model: YOLO detector instance
        video_path: Path to input video file
        conf: Confidence threshold for detections
        max_frames: Maximum number of frames to process (None = all)

    Yields:
        (frame_index, detection_result) tuples
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    predictions = model.predict(
        source=video_path.as_posix(),
        stream=True,
        conf=conf,
        verbose=False,
    )

    for frame_idx, result in enumerate(predictions):
        if max_frames is not None and frame_idx >= max_frames:
            break
        yield frame_idx, result
