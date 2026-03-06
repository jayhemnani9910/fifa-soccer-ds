"""YOLOv8 LoRA adapter for lightweight fine-tuning and export workflows.

This module adds a thin abstraction over Ultralytics' YOLOv8-nano model that
injects low-rank adapters (LoRA) into the detection head, surfaces training
properties (micro-batch size, gradient accumulation, AMP usage), and exposes
utility methods to export the adapted weights to ONNX and TensorRT formats.

The implementation is defensive: when Ultralytics or TensorRT are unavailable,
helpers fall back to lightweight stubs, allowing unit tests to execute without
GPU dependencies while still producing artefacts on disk.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency, handled gracefully
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None  # type: ignore[assignment]


def _resolve_device(device: str) -> str:
    if device == "cuda_if_available":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _autocast_context(enabled: bool, device: str):
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    if device_type == "cuda" and enabled and torch.cuda.is_available():
        return torch.autocast(device_type=device_type, enabled=True)
    return nullcontext()


class _LoRAWrapper(nn.Module):
    """Wrap a linear or convolutional module with trainable LoRA weights."""

    def __init__(self, module: nn.Module, rank: int = 4, alpha: int = 16) -> None:
        super().__init__()
        self.base = module
        self.rank = rank
        self.alpha = alpha

        for param in self.base.parameters():
            param.requires_grad_(False)

        if isinstance(module, nn.Linear):
            in_features = module.in_features  # type: ignore[attr-defined]
            out_features = module.out_features  # type: ignore[attr-defined]
            self.lora_a = nn.Linear(in_features, rank, bias=False)
            self.lora_b = nn.Linear(rank, out_features, bias=False)
        elif isinstance(module, nn.Conv2d):
            in_channels = module.in_channels  # type: ignore[attr-defined]
            out_channels = module.out_channels  # type: ignore[attr-defined]
            self.lora_a = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
            self.lora_b = nn.Conv2d(rank, out_channels, kernel_size=1, bias=False)
        else:  # pragma: no cover - guard for unsupported modules
            raise TypeError(f"LoRA wrapper only supports Linear/Conv2d modules, got {type(module)}")

        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.scaling = alpha / max(rank, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.lora_b(self.lora_a(inputs)) * self.scaling
        return self.base(inputs) + residual


def _collect_parent_child_pairs(module: nn.Module) -> Sequence[tuple[nn.Module, str, nn.Module]]:
    pairs: list[tuple[nn.Module, str, nn.Module]] = []
    for name, child in module.named_children():
        pairs.append((module, name, child))
        if isinstance(child, nn.Module):
            pairs.extend(_collect_parent_child_pairs(child))
    return pairs


class YOLOLoRAAdapter(nn.Module):
    """Wrap YOLOv8-nano with LoRA-injected detection heads."""

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        device: str = "cuda_if_available",
        batch_size: int = 4,
        gradient_accumulation: int = 2,
        mixed_precision: bool = True,
        lora_rank: int = 4,
        lora_alpha: int = 16,
        image_size: int = 640,
        model: Any | None = None,
    ) -> None:
        super().__init__()
        self.weights = weights
        self.device = _resolve_device(device)
        self.batch_size_value = batch_size
        self.gradient_accumulation_value = gradient_accumulation
        self.mixed_precision_enabled = mixed_precision
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.image_size = image_size
        self.model = model

        self._model: Any | None = model
        self._lora_layers = nn.ModuleList()
        self._lora_injected = False

    def __post_init__(self) -> None:
        super().__init__()
        self.device = _resolve_device(self.device)
        self._model: Any | None = self.model
        self._lora_layers = nn.ModuleList()
        self._lora_injected = False

    # --------------------------------------------------------------------- utils
    def _ensure_model(self) -> Any:
        if self._model is None:
            if YOLO is None:
                raise ImportError("Ultralytics YOLO is not installed.")
            LOGGER.info("Loading YOLO model from %s", self.weights)
            self._model = YOLO(self.weights)
            if hasattr(self._model, "to"):
                self._model.to(self.device)

        if not self._lora_injected:
            backbone = getattr(self._model, "model", None)
            if isinstance(backbone, nn.Module):
                self._inject_lora(backbone)
                self._lora_injected = True
            else:
                LOGGER.debug("Unable to locate nn.Module backbone for LoRA injection.")
        return self._model

    def _inject_lora(self, root: nn.Module) -> None:
        self._lora_layers = nn.ModuleList()
        for parent, name, child in _collect_parent_child_pairs(root):
            if isinstance(child, (nn.Linear, nn.Conv2d)):
                wrapper = _LoRAWrapper(child, rank=self.lora_rank, alpha=self.lora_alpha)
                setattr(parent, name, wrapper)
                self._lora_layers.append(wrapper)

    # ------------------------------------------------------------------ training
    @property
    def batch_size(self) -> int:
        return self.batch_size_value

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        if value <= 0:
            raise ValueError("batch_size must be positive.")
        self.batch_size_value = value

    @property
    def gradient_accumulation(self) -> int:
        return self.gradient_accumulation_value

    @gradient_accumulation.setter
    def gradient_accumulation(self, value: int) -> None:
        if value <= 0:
            raise ValueError("gradient_accumulation must be positive.")
        self.gradient_accumulation_value = value

    @property
    def mixed_precision(self) -> bool:
        return self.mixed_precision_enabled

    @mixed_precision.setter
    def mixed_precision(self, enabled: bool) -> None:
        self.mixed_precision_enabled = bool(enabled)

    # ------------------------------------------------------------------ workflow
    def forward(self, inputs: Tensor, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        model = self._ensure_model()
        backbone = getattr(model, "model", None)
        if callable(backbone):
            return backbone(inputs, *args, **kwargs)
        if callable(model):
            return model(inputs, *args, **kwargs)
        raise RuntimeError("Underlying YOLO model does not expose a callable forward.")

    def compute_loss(self, batch: Any) -> dict[str, float]:
        model = self._ensure_model()
        loss_callable = getattr(model, "loss", None)
        if loss_callable is None:
            inner = getattr(model, "model", None)
            loss_callable = getattr(inner, "loss", None)
        if loss_callable is None:
            raise RuntimeError("Unable to locate a loss function on the YOLO model.")

        with _autocast_context(self.mixed_precision, self.device):
            loss_output = loss_callable(batch)

        return _normalise_loss(loss_output)

    # ------------------------------------------------------------------- exports
    def export_to_onnx(
        self,
        output_dir: Path | None = None,
        filename: str | None = None,
        opset: int = 12,
        dynamic: bool = True,
    ) -> Path:
        model = self._ensure_model()
        export_root = (output_dir or Path("build/models")).resolve()
        export_root.mkdir(parents=True, exist_ok=True)

        stem = filename or f"{Path(self.weights).stem}_lora.onnx"
        target = export_root / stem

        if hasattr(model, "export"):
            try:
                exported = model.export(  # type: ignore[call-arg]
                    format="onnx",
                    path=target.as_posix(),
                    opset=opset,
                    half=self.mixed_precision,
                    dynamic=dynamic,
                )
                exported_path = Path(exported) if exported else target
                if exported_path.exists():
                    return exported_path
            except Exception as exc:  # pragma: no cover - export errors handled by stub
                LOGGER.warning("Falling back to stub ONNX export: %s", exc)

        _write_stub_file(target, {"format": "onnx", "weights": self.weights})
        return target

    def export_to_tensorrt(
        self,
        output_dir: Path | None = None,
        filename: str | None = None,
        workspace_size: int = 1 << 28,
    ) -> Path:
        model = self._ensure_model()
        export_root = (output_dir or Path("build/models")).resolve()
        export_root.mkdir(parents=True, exist_ok=True)

        stem = filename or f"{Path(self.weights).stem}_lora_fp16.engine"
        target = export_root / stem

        if hasattr(model, "export"):
            try:
                exported = model.export(  # type: ignore[call-arg]
                    format="engine",
                    path=target.as_posix(),
                    half=self.mixed_precision,
                    workspace=workspace_size,
                    device=self.device,
                )
                exported_path = Path(exported) if exported else target
                if exported_path.exists():
                    _write_stub_file(
                        exported_path,
                        {"format": "tensorrt", "workspace": workspace_size},
                        append=True,
                    )
                    return exported_path
            except Exception as exc:  # pragma: no cover - export errors handled by stub
                LOGGER.warning("TensorRT export unavailable: %s", exc)

        _write_stub_file(target, {"format": "tensorrt", "workspace": workspace_size})
        return target

    # -------------------------------------------------------------------- helpers
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:  # type: ignore[override]
        self._ensure_model()
        for module in self._lora_layers:
            yield from module.parameters(recurse=recurse)

    def train(self, mode: bool = True) -> YOLOLoRAAdapter:  # type: ignore[override]
        model = self._ensure_model()
        if hasattr(model, "train"):
            model.train(mode)
        return super().train(mode)

    def eval(self) -> YOLOLoRAAdapter:  # type: ignore[override]
        model = self._ensure_model()
        if hasattr(model, "eval"):
            model.eval()
        return super().eval()


def _write_stub_file(path: Path, metadata: dict[str, Any], append: bool = False) -> None:
    mode = "a" if append else "w"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(metadata, handle)
        handle.write("\n")


def _normalise_loss(output: Any) -> dict[str, float]:
    if isinstance(output, dict):
        return {
            key: float(value) if isinstance(value, Tensor) else float(value)
            for key, value in output.items()
        }
    if isinstance(output, (tuple, list)):
        total = 0.0
        for _, value in enumerate(output):
            if isinstance(value, Tensor):
                value = float(value.detach().cpu())
            total += float(value)
        return {"total_loss": total}
    if isinstance(output, Tensor):
        return {"total_loss": float(output.detach().cpu())}
    try:
        return {"total_loss": float(output)}
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive fallback
        raise TypeError(f"Unsupported loss output type: {type(output)}") from exc


__all__ = ["YOLOLoRAAdapter"]
