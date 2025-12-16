"""Live inference pipeline for real-time detection and tracking.

This module provides:
- Real-time YOLO inference on camera/RTSP streams
- ByteTrack multi-object tracking with temporal consistency
- Optional video output and GUI preview
- Graceful shutdown handling with resource cleanup
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

from src.detect.infer import InferenceConfig, extract_detections, load_model
from src.track.bytetrack_runtime import ByteTrackRuntime, Tracklets
from src.track.pipeline import filter_detections
from src.utils.overlay import draw_boxes, draw_track_ids

LOGGER = logging.getLogger(__name__)

shutdown_requested = False


def _configure_logging(level: str = "INFO") -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("live_pipeline.log", encoding="utf-8"),
        ],
    )


def _register_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""

    del frame
    global shutdown_requested
    LOGGER.info("Received signal %s, initiating graceful shutdown...", signum)
    shutdown_requested = True

@dataclass(slots=True)
class TrackerRuntimeConfig:
    min_confidence: float = 0.25
    distance_threshold: float = 80.0
    max_age: int = 15


@dataclass(slots=True)
class LiveCaptureConfig:
    source: str = "0"
    rtsp: bool = False
    width: int | None = None
    height: int | None = None
    fps: int | None = None
    fps_limit: int | None = 24


@dataclass(slots=True)
class LiveOutputConfig:
    save_video: bool = True
    output_path: str = "outputs/live/output.mp4"
    show_preview: bool = True


@dataclass(slots=True)
class LivePipelineConfig:
    detector: InferenceConfig = field(default_factory=InferenceConfig)
    tracker: TrackerRuntimeConfig = field(default_factory=TrackerRuntimeConfig)
    capture: LiveCaptureConfig = field(default_factory=LiveCaptureConfig)
    output: LiveOutputConfig = field(default_factory=LiveOutputConfig)


def _normalise_source(source: str, rtsp: bool) -> str | int:
    if rtsp:
        return source
    if source.isdigit():
        return int(source)
    return source


def _initialise_writer(
    output_path: Path, fps: float, frame_shape: tuple[int, int]
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    height, width = frame_shape
    writer = cv2.VideoWriter(output_path.as_posix(), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer at {output_path}")
    return writer


def _apply_overlays(frame: np.ndarray, tracklets) -> np.ndarray:
    if not tracklets.items:
        return frame

    boxes = [track.bbox for track in tracklets.items]
    labels = [str(track.extras.get("class_name", "")) for track in tracklets.items]
    overlaid = draw_boxes(frame, boxes=boxes, labels=labels)
    ids = [track.track_id for track in tracklets.items]
    overlaid = draw_track_ids(overlaid, boxes=boxes, track_ids=ids)
    return overlaid


def run_live_pipeline(config: LivePipelineConfig) -> None:
    capture_source = _normalise_source(config.capture.source, config.capture.rtsp)
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open capture source: {config.capture.source}")

    if config.capture.width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.capture.width)
    if config.capture.height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.capture.height)
    if config.capture.fps:
        capture.set(cv2.CAP_PROP_FPS, config.capture.fps)

    detector = load_model(config.detector)
    tracker = ByteTrackRuntime(
        min_confidence=config.tracker.min_confidence,
        distance_threshold=config.tracker.distance_threshold,
        max_age=config.tracker.max_age,
    )

    writer: cv2.VideoWriter | None = None
    fps_limit = config.capture.fps_limit or 0
    desired_fps = config.capture.fps or capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0

    try:
        while True:
            if shutdown_requested:
                LOGGER.info("Graceful shutdown requested, stopping pipeline...")
                break
                
            loop_start = time.time()
            ret, frame = capture.read()
            if not ret or frame is None:
                break

            results = detector.predict(frame, conf=config.detector.confidence, verbose=False)
            detections = extract_detections(results[0]) if results else []
            filtered = filter_detections(detections, min_confidence=config.tracker.min_confidence)
            tracklets = tracker.update(frame_idx, detections=filtered)

            frame_with_overlay = _apply_overlays(frame, tracklets)

            if config.output.save_video:
                if writer is None:
                    writer = _initialise_writer(
                        Path(config.output.output_path),
                        desired_fps,
                        (frame.shape[0], frame.shape[1]),
                    )
                writer.write(frame_with_overlay)

            if config.output.show_preview:
                cv2.imshow("FIFA DS Live", frame_with_overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    LOGGER.info("Live preview interrupted by user.")
                    break

            frame_idx += 1

            if fps_limit > 0:
                elapsed = time.time() - loop_start
                target = 1.0 / float(fps_limit)
                if elapsed < target:
                    time.sleep(target - elapsed)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if config.output.show_preview:
            cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the live detection + tracking pipeline.")
    parser.add_argument("--source", default="0", help="Camera index, file path, or RTSP URL.")
    parser.add_argument("--rtsp", action="store_true", help="Treat source as RTSP URL.")
    parser.add_argument("--weights", default="yolov8n.pt", help="YOLO checkpoint for detection.")
    parser.add_argument("--device", default="cuda_if_available", help="Torch device string.")
    parser.add_argument(
        "--confidence", type=float, default=0.25, help="Detection confidence threshold."
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.25, help="Tracker minimum confidence."
    )
    parser.add_argument(
        "--distance-threshold", type=float, default=80.0, help="Distance threshold for association."
    )
    parser.add_argument(
        "--max-age", type=int, default=15, help="Tracker max age before recycling IDs."
    )
    parser.add_argument("--width", type=int, default=None, help="Force capture width.")
    parser.add_argument("--height", type=int, default=None, help="Force capture height.")
    parser.add_argument("--fps", type=float, default=None, help="Hint capture FPS.")
    parser.add_argument(
        "--fps-limit", type=float, default=24, help="Limit processing FPS (0 disables)."
    )
    parser.add_argument(
        "--output", default="outputs/live/output.mp4", help="Where to store the preview recording."
    )
    parser.add_argument("--no-save", action="store_true", help="Disable saving the output video.")
    parser.add_argument("--no-preview", action="store_true", help="Disable on-screen preview.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def main() -> None:  # pragma: no cover - CLI entrypoint
    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(args.log_level)
    _register_signal_handlers()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    rtsp_flag = args.rtsp or args.source.startswith("rtsp://")

    pipeline_config = LivePipelineConfig(
        detector=InferenceConfig(
            weights=args.weights, device=args.device, confidence=args.confidence
        ),
        tracker=TrackerRuntimeConfig(
            min_confidence=args.min_confidence,
            distance_threshold=args.distance_threshold,
            max_age=args.max_age,
        ),
        capture=LiveCaptureConfig(
            source=args.source,
            rtsp=rtsp_flag,
            width=args.width,
            height=args.height,
            fps=args.fps,
            fps_limit=args.fps_limit,
        ),
        output=LiveOutputConfig(
            save_video=not args.no_save,
            output_path=args.output,
            show_preview=not args.no_preview,
        ),
    )

    run_live_pipeline(pipeline_config)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
