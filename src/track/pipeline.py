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


def filter_detections(
    detections: Iterable[dict],
    min_confidence: float,
    allowed_classes: set[str] | None = None,
    max_area_ratio: float = 1.0,
    frame_shape: tuple[int, int] | None = None,
    nms_iou: float = 1.0,
) -> list[dict]:
    """Filter raw detections by confidence, class, bbox area, and per-class NMS.

    Args:
        detections: iterable of dicts with ``bbox``, ``confidence`` (or ``score``),
            ``class_id``, and optionally ``class_name``.
        min_confidence: drop any detection below this score.
        allowed_classes: if non-empty, keep only detections whose ``class_name``
            is in this set. ``None`` disables class filtering.
        max_area_ratio: drop bboxes whose area / frame area exceeds this.
            Requires ``frame_shape``; ignored if frame_shape is None or ratio is 1.0.
        frame_shape: ``(H, W)`` of the source frame for area-ratio calc.
        nms_iou: per-class IoU threshold for NMS. ``1.0`` disables.
    """
    filtered: list[dict] = []
    frame_area = float(frame_shape[0]) * float(frame_shape[1]) if frame_shape else 0.0
    for det in detections:
        bbox = det.get("bbox")
        confidence = det.get("confidence")
        if confidence is None and "score" in det:
            confidence = det["score"]
        if not bbox:
            continue
        if confidence is not None and confidence < min_confidence:
            continue
        class_name = det.get("class_name")
        if allowed_classes is not None and class_name not in allowed_classes:
            continue
        if frame_area > 0 and max_area_ratio < 1.0:
            x1, y1, x2, y2 = bbox[:4]
            area = max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))
            if area / frame_area > max_area_ratio:
                continue
        filtered.append(
            {
                "bbox": bbox,
                "score": float(confidence or 0.0),
                "class_id": det.get("class_id"),
                "class_name": class_name,
            }
        )

    if nms_iou < 1.0 and len(filtered) > 1:
        filtered = _nms_per_class(filtered, iou_threshold=nms_iou)
    return filtered


def _nms_per_class(detections: list[dict], iou_threshold: float) -> list[dict]:
    """Per-class IoU-NMS using torchvision.ops.nms with a pure-Python fallback."""
    by_class: dict = {}
    for det in detections:
        by_class.setdefault(det.get("class_name"), []).append(det)

    try:
        import torch
        from torchvision.ops import nms as tv_nms
    except ImportError:
        torch = None
        tv_nms = None

    kept: list[dict] = []
    for group in by_class.values():
        if len(group) <= 1:
            kept.extend(group)
            continue
        if tv_nms is not None:
            boxes = torch.tensor([d["bbox"][:4] for d in group], dtype=torch.float32)
            scores = torch.tensor([d["score"] for d in group], dtype=torch.float32)
            idxs = tv_nms(boxes, scores, iou_threshold).tolist()
            kept.extend(group[i] for i in idxs)
        else:
            kept.extend(_nms_numpy(group, iou_threshold))
    return kept


def _nms_numpy(group: list[dict], iou_threshold: float) -> list[dict]:
    order = sorted(range(len(group)), key=lambda i: group[i]["score"], reverse=True)
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        ax1, ay1, ax2, ay2 = group[i]["bbox"][:4]
        a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        remaining = []
        for j in order:
            bx1, by1, bx2, by2 = group[j]["bbox"][:4]
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = a_area + b_area - inter
            iou = inter / union if union > 0 else 0.0
            if iou <= iou_threshold:
                remaining.append(j)
        order = remaining
    return [group[i] for i in keep]


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
