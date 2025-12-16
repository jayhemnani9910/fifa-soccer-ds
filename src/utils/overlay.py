"""Overlay and annotation helpers for live pipelines."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore[assignment]


def draw_boxes(
    frame: np.ndarray,
    boxes: Iterable[Sequence[float]],
    colors: Iterable[tuple[int, int, int]] | None = None,
    labels: Iterable[str] | None = None,
    thickness: int = 2,
) -> np.ndarray:
    if cv2 is None:
        raise ImportError("OpenCV is required for overlay drawing.")

    output = frame.copy()
    colors = list(colors) if colors is not None else []
    labels = list(labels) if labels is not None else []

    for idx, box in enumerate(boxes):
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = map(int, box[:4])
        color = colors[idx % len(colors)] if colors else (0, 255, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        if labels and idx < len(labels):
            cv2.putText(
                output,
                labels[idx],
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    return output


def draw_track_ids(
    frame: np.ndarray, boxes: Iterable[Sequence[float]], track_ids: Iterable[int]
) -> np.ndarray:
    if cv2 is None:
        raise ImportError("OpenCV is required for overlay drawing.")

    output = frame.copy()
    for box, track_id in zip(boxes, track_ids, strict=False):
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = map(int, box[:4])
        label = f"ID {track_id}"
        cv2.putText(
            output,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return output
