"""Live ingest and inference helpers with lazy public exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "LiveCaptureConfig": ("src.live.run_live", "LiveCaptureConfig"),
    "LiveOutputConfig": ("src.live.run_live", "LiveOutputConfig"),
    "LivePipelineConfig": ("src.live.run_live", "LivePipelineConfig"),
    "TrackerRuntimeConfig": ("src.live.run_live", "TrackerRuntimeConfig"),
    "run_live_pipeline": ("src.live.run_live", "run_live_pipeline"),
    "BarcaAPIServer": ("src.live.barca_api", "BarcaAPIServer"),
    "RTSPCapture": ("src.live.rtsp_capture", "RTSPCapture"),
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
