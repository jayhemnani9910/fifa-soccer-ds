"""YOLOv8 detection helpers with runnable sample pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore[assignment]

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - handled gracefully for environments without ultralytics
    YOLO = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

DEFAULT_IMAGE = Path("data/processed/sample/frame_000000.jpg")
DEFAULT_VIDEO = Path("data/raw/sample.mp4")
DEFAULT_OUTPUT = Path("outputs/detect")


@dataclass(slots=True)
class InferenceConfig:
    """Configuration for quick detection passes."""

    weights: str = "yolov8n.pt"
    device: str = "cuda_if_available"
    confidence: float = 0.25
    max_frames: int = 30


def _resolve_device(device: str) -> str:
    if device == "cuda_if_available":
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_model(config: InferenceConfig) -> YOLO:
    """Load a YOLOv8 model given the configuration."""

    if YOLO is None:
        raise ImportError("ultralytics must be installed to run detection inference.")
    device = _resolve_device(config.device)
    model = YOLO(config.weights)
    model.to(device)
    return model


def run_inference(
    image_path: str | Path,
    config: InferenceConfig | None = None,
    model: YOLO | None = None,
):
    """Execute a one-off inference run to verify the model pipeline."""

    cfg = config or InferenceConfig()
    image = Path(image_path)
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")

    detector = model or load_model(cfg)
    results = detector.predict(image.as_posix(), conf=cfg.confidence, verbose=False)
    return results[0]


def _tensor_to_list(values: Any) -> list[float]:
    """Convert tensor-like objects to Python lists robustly.

    Handles PyTorch tensors, numpy arrays, and standard Python iterables.
    Ensures consistent output type regardless of input backend.
    """
    if values is None:
        return []

    # Move to CPU and detach if needed
    try:
        if hasattr(values, "cpu"):
            values = values.cpu()
        if hasattr(values, "detach"):
            values = values.detach()
    except Exception as e:
        LOGGER.debug(f"Failed to move tensor to CPU: {e}")

    # Primary conversion path: tensor -> numpy -> list
    try:
        if hasattr(values, "numpy"):
            return values.numpy().tolist()
    except (AttributeError, RuntimeError, TypeError) as e:
        LOGGER.debug(f"Numpy conversion failed: {e}")

    # Fallback: direct tolist() for compatible types
    try:
        if hasattr(values, "tolist"):
            return values.tolist()
    except (AttributeError, TypeError) as e:
        LOGGER.debug(f"Direct tolist() failed: {e}")

    # Final fallback: iterate or wrap scalar
    try:
        result = list(values)
        if result and not isinstance(result[0], (int, float)):
            return [float(x) for x in result]
        return result
    except TypeError:
        # Scalar fallback
        try:
            return [float(values)]
        except (TypeError, ValueError) as e:
            LOGGER.warning(f"Unable to convert value to list: {values}, error: {e}")
            return []


def extract_detections(result) -> list[dict[str, Any]]:
    """Convert a YOLO result object into serialisable detection dictionaries."""

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    xyxy = _tensor_to_list(getattr(boxes, "xyxy", None))
    confidences = _tensor_to_list(getattr(boxes, "conf", None))
    classes = _tensor_to_list(getattr(boxes, "cls", None))

    names = getattr(result, "names", {})

    detections: list[dict[str, Any]] = []
    for idx, bbox in enumerate(xyxy):
        conf = confidences[idx] if idx < len(confidences) else None
        cls_id = classes[idx] if idx < len(classes) else None
        class_name = None
        if cls_id is not None:
            try:
                class_name = names[int(cls_id)]
            except (KeyError, ValueError, TypeError):
                class_name = str(cls_id)

        detections.append(
            {
                "bbox": bbox,
                "confidence": conf,
                "class_id": cls_id,
                "class_name": class_name,
            }
        )
    return detections


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def _write_overlay(image_array: Any, path: Path) -> Path | None:
    if image_array is None:
        return None
    if cv2 is None:
        LOGGER.debug("Skipping overlay write to %s because OpenCV is unavailable.", path)
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(path.as_posix(), image_array)
    if not success:
        LOGGER.warning("Unable to write overlay to %s", path)
        return None
    return path


def run_image_detection(
    model: YOLO,
    image_path: Path,
    output_dir: Path,
    config: InferenceConfig,
) -> dict[str, Any]:
    result = run_inference(image_path, config=config, model=model)
    detections = extract_detections(result)

    stem = image_path.stem
    json_path = output_dir / f"{stem}_detections.json"
    overlay_path = output_dir / f"{stem}_overlay.jpg"

    _write_json(
        json_path,
        {
            "image": image_path.as_posix(),
            "detections": detections,
        },
    )
    overlay = result.plot(boxes=True) if hasattr(result, "plot") else None
    overlay_out = _write_overlay(overlay, overlay_path) if overlay is not None else None

    return {
        "json": json_path,
        "overlay": overlay_out,
        "detections": detections,
    }


def run_video_detection(
    model: YOLO,
    video_path: Path,
    output_dir: Path,
    config: InferenceConfig,
    max_frames: int,
) -> list[dict[str, Any]]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    results_metadata: list[dict[str, Any]] = []
    stream = model.predict(
        source=video_path.as_posix(),
        stream=True,
        conf=config.confidence,
        verbose=False,
    )

    for frame_idx, result in enumerate(stream):
        if frame_idx >= max_frames:
            break

        detections = extract_detections(result)
        frame_name = f"{video_path.stem}_frame_{frame_idx:04d}"
        json_path = output_dir / f"{frame_name}_detections.json"
        overlay_path = output_dir / f"{frame_name}_overlay.jpg"

        _write_json(
            json_path,
            {
                "video": video_path.as_posix(),
                "frame_index": frame_idx,
                "detections": detections,
            },
        )
        overlay = result.plot(boxes=True) if hasattr(result, "plot") else None
        overlay_out = _write_overlay(overlay, overlay_path) if overlay is not None else None

        results_metadata.append(
            {
                "frame": frame_idx,
                "json": json_path,
                "overlay": overlay_out,
                "detections": detections,
            }
        )

    return results_metadata


def run(
    image_path: str | Path | None = DEFAULT_IMAGE,
    video_path: str | Path | None = DEFAULT_VIDEO,
    output_dir: str | Path = DEFAULT_OUTPUT,
    config: InferenceConfig | None = None,
    max_frames: int | None = None,
) -> dict[str, Any]:
    cfg = config or InferenceConfig()
    if max_frames is None:
        max_frames = cfg.max_frames

    detector = load_model(cfg)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"output_dir": output_path}

    if image_path:
        image = Path(image_path)
        if image.exists():
            summary["image"] = run_image_detection(detector, image, output_path, cfg)
        else:
            LOGGER.warning("Skipping image inference; file not found: %s", image)

    if video_path:
        video = Path(video_path)
        if video.exists():
            summary["video"] = run_video_detection(detector, video, output_path, cfg, max_frames)
        else:
            LOGGER.warning("Skipping video inference; file not found: %s", video)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLOv8 detections on sample assets.")
    parser.add_argument(
        "--image", default=DEFAULT_IMAGE.as_posix(), help="Path to the sample image."
    )
    parser.add_argument(
        "--video", default=DEFAULT_VIDEO.as_posix(), help="Path to the sample video clip."
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT.as_posix(),
        help="Directory to store detection artefacts.",
    )
    parser.add_argument("--weights", default="yolov8n.pt", help="Path to YOLOv8 weights (.pt).")
    parser.add_argument(
        "--device",
        default="cuda_if_available",
        help="Device string (cpu, cuda, cuda:0, etc.). Use cuda_if_available to auto-select.",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.25, help="Detection confidence threshold."
    )
    parser.add_argument(
        "--max-frames", type=int, default=30, help="Number of video frames to process."
    )
    return parser


def main() -> None:  # pragma: no cover - CLI entrypoint
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = InferenceConfig(
        weights=args.weights,
        device=args.device,
        confidence=args.confidence,
        max_frames=args.max_frames,
    )

    run(
        image_path=args.image if args.image else None,
        video_path=args.video if args.video else None,
        output_dir=args.output,
        config=config,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
