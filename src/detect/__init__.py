"""Detection utilities with lazy public exports.

Keeping executable modules lazy avoids importing Ultralytics and preloading CLI
modules merely because callers imported :mod:`src.detect`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "InferenceConfig": ("src.detect.infer", "InferenceConfig"),
    "load_model": ("src.detect.infer", "load_model"),
    "run_inference": ("src.detect.infer", "run_inference"),
    "run": ("src.detect.infer", "run"),
    "run_image_detection": ("src.detect.infer", "run_image_detection"),
    "run_video_detection": ("src.detect.infer", "run_video_detection"),
    "extract_detections": ("src.detect.infer", "extract_detections"),
    "YOLOLoRAAdapter": ("src.detect.yolo_lora_adapter", "YOLOLoRAAdapter"),
    "FineTuneConfig": ("src.detect.train_yolo", "FineTuneConfig"),
    "CheckpointManager": ("src.detect.train_yolo", "CheckpointManager"),
    "fine_tune_loop": ("src.detect.train_yolo", "fine_tune_loop"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})


__all__ = list(_EXPORTS)
