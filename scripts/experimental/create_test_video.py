"""Create a deterministic synthetic video for local pipeline smoke tests."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


def create_synthetic_video(
    output_path: Path,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    duration_seconds: int = 3,
    num_objects: int = 5,
) -> None:
    """Generate a video with moving rectangles that exercise media I/O paths."""
    if width < 64 or height < 64:
        raise ValueError("width and height must both be at least 64 pixels")
    if fps <= 0 or duration_seconds <= 0:
        raise ValueError("fps and duration_seconds must be positive")
    if num_objects <= 0:
        raise ValueError("num_objects must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path.as_posix(), fourcc, fps, (width, height))
    if not out.isOpened():
        out.release()
        raise RuntimeError(f"OpenCV could not create the video: {output_path}")

    total_frames = fps * duration_seconds

    rng = np.random.default_rng(42)
    objects = []
    for _ in range(num_objects):
        obj = {
            "x": int(rng.integers(0, max(1, width - 40))),
            "y": int(rng.integers(0, max(1, height - 40))),
            "vx": int(rng.choice([-5, -4, -3, -2, 2, 3, 4, 5])),
            "vy": int(rng.choice([-5, -4, -3, -2, 2, 3, 4, 5])),
            "color": tuple(int(value) for value in rng.integers(0, 256, 3)),
            "size": int(rng.integers(20, min(80, width, height))),
        }
        objects.append(obj)

    log.info("Generating %d frames at %d FPS", total_frames, fps)
    try:
        for _ in range(total_frames):
            frame = np.full((height, width, 3), (34, 139, 34), dtype=np.uint8)
            cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 255, 255), 2)
            cv2.circle(
                frame,
                (width // 2, height // 2),
                min(100, width // 4, height // 4),
                (255, 255, 255),
                2,
            )
            cv2.rectangle(frame, (20, 20), (width - 20, height - 20), (255, 255, 255), 2)

            for obj in objects:
                obj["x"] += obj["vx"]
                obj["y"] += obj["vy"]
                if obj["x"] < 0 or obj["x"] > width - obj["size"]:
                    obj["vx"] *= -1
                if obj["y"] < 0 or obj["y"] > height - obj["size"]:
                    obj["vy"] *= -1
                obj["x"] = max(0, min(width - obj["size"], obj["x"]))
                obj["y"] = max(0, min(height - obj["size"], obj["y"]))

                top_left = (obj["x"], obj["y"])
                bottom_right = (obj["x"] + obj["size"], obj["y"] + obj["size"])
                cv2.rectangle(frame, top_left, bottom_right, obj["color"], -1)
                cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), 2)

            out.write(frame)
    finally:
        out.release()
    log.info("Synthetic video created at %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create synthetic test video")
    parser.add_argument("--output", default="data/raw/sample.mp4", help="Output video path")
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--height", type=int, default=720, help="Video height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--duration", type=int, default=3, help="Duration in seconds")
    parser.add_argument("--objects", type=int, default=5, help="Number of moving objects")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    create_synthetic_video(
        output_path=Path(args.output),
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration_seconds=args.duration,
        num_objects=args.objects,
    )


if __name__ == "__main__":
    main()
