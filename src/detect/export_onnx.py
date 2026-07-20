"""Export a YOLOv8 model to ONNX for downstream runtimes."""

from __future__ import annotations

import errno
import importlib
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    YOLO: Any = importlib.import_module("ultralytics").YOLO
except ImportError:  # pragma: no cover - optional dependency
    YOLO = None

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

    def __post_init__(self) -> None:
        if not self.weights.strip():
            raise ValueError("weights must not be empty")
        if Path(self.output).suffix.lower() != ".onnx":
            raise ValueError("output must use the .onnx extension")
        if (
            isinstance(self.opset, bool)
            or not isinstance(self.opset, int)
            or not 11 <= self.opset <= 21
        ):
            raise ValueError("opset must be an integer between 11 and 21")


def _resolve_device(device: str) -> str:
    """Resolve the device string into either cuda or cpu."""

    if device == "cuda_if_available":
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def validate_onnx_artifact(artifact_path: str | Path) -> None:
    """Validate that an exported model is structurally sound and loadable by ORT."""

    artifact = Path(artifact_path)
    if artifact.suffix.lower() != ".onnx" or not artifact.is_file() or artifact.stat().st_size <= 0:
        raise RuntimeError("ONNX validation requires a non-empty .onnx artifact")

    try:
        onnx = importlib.import_module("onnx")
        ort = importlib.import_module("onnxruntime")
    except ImportError as exc:
        raise ImportError(
            "ONNX validation requires the project's export dependencies; "
            'install them with `python -m pip install -e ".[export]"`.'
        ) from exc

    try:
        model = onnx.load(str(artifact), load_external_data=False)
        onnx.checker.check_model(model)
    except Exception as exc:
        raise RuntimeError(f"ONNX structural validation failed: {exc}") from exc

    available_providers = list(ort.get_available_providers())
    if not available_providers:
        raise RuntimeError("ONNX Runtime reported no available execution providers")
    providers = (
        ["CPUExecutionProvider"]
        if "CPUExecutionProvider" in available_providers
        else [available_providers[0]]
    )
    try:
        session = ort.InferenceSession(str(artifact), providers=providers)
        inputs = session.get_inputs()
        outputs = session.get_outputs()
    except Exception as exc:
        raise RuntimeError(f"ONNX Runtime could not load the exported model: {exc}") from exc
    if not inputs or not outputs:
        raise RuntimeError("ONNX Runtime loaded a model without graph inputs or outputs")

    log.info(
        "Validated ONNX artifact '%s' with provider '%s' (%d inputs, %d outputs)",
        artifact,
        providers[0],
        len(inputs),
        len(outputs),
    )


def _publish_exported_artifact(exported_path: Path, output_path: Path) -> None:
    """Publish an export without exposing a partially copied destination file."""

    try:
        os.replace(exported_path, output_path)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise

    temporary_path: Path | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(file_descriptor, "wb") as destination, exported_path.open("rb") as source:
            shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
        try:
            exported_path.unlink()
        except OSError as exc:
            log.warning(
                "Published ONNX artifact but could not remove exporter output '%s': %s",
                exported_path,
                exc,
            )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def export_to_onnx(config: OnnxExportConfig) -> Path:
    """Export YOLO weights to ONNX using the ultralytics API."""

    if YOLO is None:
        raise ImportError("ultralytics must be installed to export models to ONNX.")

    resolved_output = Path(config.output)
    if resolved_output.exists():
        raise FileExistsError(f"Refusing to overwrite existing ONNX artifact: {resolved_output}")
    likely_export_path = Path(config.weights).with_suffix(".onnx")
    if likely_export_path.exists():
        raise FileExistsError(
            "Refusing to let Ultralytics overwrite its existing default ONNX artifact: "
            f"{likely_export_path}"
        )
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
    exported_value = model.export(
        format="onnx",
        opset=config.opset,
        dynamic=config.dynamic,
        simplify=config.simplify,
        device=device,
        verbose=False,
    )
    if not isinstance(exported_value, (str, Path)):
        raise RuntimeError("Ultralytics did not return an ONNX artifact path")
    exported_path = Path(exported_value)
    if (
        exported_path.suffix.lower() != ".onnx"
        or not exported_path.is_file()
        or exported_path.stat().st_size <= 0
    ):
        raise RuntimeError("Ultralytics did not produce a non-empty ONNX artifact")

    validate_onnx_artifact(exported_path)
    if exported_path.resolve() != resolved_output.resolve():
        _publish_exported_artifact(exported_path, resolved_output)
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
