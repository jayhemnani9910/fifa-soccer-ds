"""Export a YOLOv8 model to ONNX for downstream runtimes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional dependency
    YOLO = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


@dataclass(slots=True)
class OnnxExportConfig:
    """Configuration for exporting YOLO weights to ONNX."""

    weights: str = "yolov8n.pt"
    output: str = "build/yolov8n.onnx"
    opset: int = 12
    dynamic: bool = True
    simplify: bool = False
    device: str = "cuda_if_available"


def _resolve_device(device: str) -> str:
    """Resolve the device string into either cuda or cpu."""

    if device == "cuda_if_available":
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def export_to_onnx(config: OnnxExportConfig) -> Path:
    """Export YOLO weights to ONNX using the ultralytics API."""

    if YOLO is None:
        raise ImportError("ultralytics must be installed to export models to ONNX.")

    resolved_output = Path(config.output)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(config.device)
    log.info("Loading YOLO weights '%s' on device '%s'", config.weights, device)
    model = YOLO(config.weights)
    model.to(device)

    log.info(
        "Exporting ONNX model to '%s' (opset=%s, dynamic=%s)",
        resolved_output,
        config.opset,
        config.dynamic,
    )
    model.export(
        format="onnx",
        opset=config.opset,
        dynamic=config.dynamic,
        simplify=config.simplify,
        imgsz=None,
        device=device,
        half=False,
        verbose=False,
        outfile=resolved_output.as_posix(),
    )
    return resolved_output


def run(weights: str | None = None, output: str | None = None) -> Path:
    """Convenience entrypoint for CLI scripts."""

    cfg = OnnxExportConfig()
    if weights:
        cfg.weights = weights
    if output:
        cfg.output = output
    return export_to_onnx(cfg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        run()
    except Exception as exc:  # pragma: no cover - CLI convenience
        log.exception("ONNX export failed: %s", exc)
        raise
