"""Visualization helpers for overlaying detections on frames."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore[assignment]


def draw_bounding_boxes(
    frame: np.ndarray,
    boxes: Iterable[Sequence[float]],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Overlay axis-aligned bounding boxes on the frame."""

    if cv2 is None:
        raise ImportError("OpenCV is required for drawing functions.")

    output = frame.copy()
    for box in boxes:
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
    return output


def annotate_frame(
    frame: np.ndarray,
    texts: Iterable[tuple[str, tuple[int, int]]],
    font_scale: float = 0.5,
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Write small labels on the frame at provided coordinates."""

    if cv2 is None:
        raise ImportError("OpenCV is required for annotation functions.")

    output = frame.copy()
    for text, position in texts:
        cv2.putText(
            output,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )
    return output
