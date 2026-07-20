"""Frame subsampling and resizing for FIFA DS clips."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency check
    raise ImportError("OpenCV (cv2) is required for preprocessing.") from exc

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subsample frames and resize to target resolution."
    )
    parser.add_argument("--input", required=True, help="Path to the input video clip.")
    parser.add_argument("--output-dir", required=True, help="Directory to store processed frames.")
    parser.add_argument("--stride", type=int, default=5, help="Sample every N-th frame.")
    parser.add_argument("--width", type=int, default=1280, help="Output frame width.")
    parser.add_argument("--height", type=int, default=720, help="Output frame height.")
    parser.add_argument(
        "--image-format",
        default="jpg",
        choices=("jpg", "png"),
        help="File extension for the exported frames.",
    )
    return parser.parse_args()


def preprocess(
    input_path: Path,
    output_dir: Path,
    stride: int,
    width: int,
    height: int,
    image_format: str,
) -> int:
    if stride <= 0:
        raise ValueError("Stride must be a positive integer.")
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive integers.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input clip not found: {input_path}")

    capture = cv2.VideoCapture(input_path.as_posix())
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open input clip: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_id = 0
    saved = 0
    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            if frame_id % stride != 0:
                frame_id += 1
                continue

            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            output_path = output_dir / f"frame_{saved:06d}.{image_format}"
            success = cv2.imwrite(output_path.as_posix(), resized)
            if not success:
                raise RuntimeError(f"Failed to write frame to {output_path}")
            saved += 1
            frame_id += 1
    finally:
        capture.release()
    log.info("Saved %s frames to %s", saved, output_dir)
    return saved


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    preprocess(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        stride=args.stride,
        width=args.width,
        height=args.height,
        image_format=args.image_format,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
