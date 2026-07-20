"""Fine-tuning loop for the YOLOv8 LoRA adapter."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from src.detect.yolo_lora_adapter import YOLOLoRAAdapter
from src.utils.mlflow_helper import (
    log_run_artifacts,
    log_run_metrics,
    log_run_params,
    start_run,
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class FineTuneConfig:
    epochs: int = 10
    learning_rate: float = 1e-4
    batch_size: int = 4
    gradient_accumulation: int = 2
    patience: int = 3
    experiment: str = "yolo-lora-finetune"
    run_name: str | None = None
    checkpoint_dir: Path = Path("build/checkpoints/yolo_lora")
    resume_from: Path | None = None


class CheckpointManager:
    """Track best-performing checkpoints based on validation metrics."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.best_map = float("-inf")
        self.best_loss = float("inf")
        self.best_checkpoint: Path | None = None

    def save(
        self,
        adapter: YOLOLoRAAdapter,
        metrics: dict[str, float | None],
        epoch: int,
        optimizer_state: dict | None = None,
    ) -> Path:
        val_loss = metrics.get("val_loss")
        if val_loss is None or not math.isfinite(val_loss):
            raise ValueError("A finite validation loss is required to save a checkpoint")

        snapshot = {
            "epoch": epoch,
            "state_dict": adapter.state_dict(),
            "metrics": metrics,
        }
        if optimizer_state is not None:
            snapshot["optimizer_state"] = optimizer_state
        path = self.directory / f"epoch_{epoch:02d}.pt"
        temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            torch.save(snapshot, temporary_path)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)

        map_score = metrics.get("mAP@0.5")
        if map_score is not None and map_score > self.best_map:
            self.best_map = map_score

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_checkpoint = path
        return path

    def load_best(self, adapter: YOLOLoRAAdapter) -> Path | None:
        if self.best_checkpoint and self.best_checkpoint.exists():
            LOGGER.info("Restoring best checkpoint from %s", self.best_checkpoint)
            payload = torch.load(self.best_checkpoint, map_location="cpu", weights_only=True)
            adapter.load_state_dict(payload["state_dict"])
        return self.best_checkpoint


def _loss_from_dict(loss_dict: Mapping[str, float | torch.Tensor]) -> float:
    if not loss_dict:
        raise ValueError("The model returned no loss values")
    if "total_loss" in loss_dict:
        value = loss_dict["total_loss"]
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu())
        return float(value)
    total = 0.0
    for value in loss_dict.values():
        if isinstance(value, torch.Tensor):
            total += float(value.detach().cpu())
        else:
            total += float(value)
    return total


def _loss_tensor(loss_dict: Mapping[str, float | torch.Tensor]) -> torch.Tensor:
    total: torch.Tensor | None = None
    device: torch.device | None = None
    for value in loss_dict.values():
        if isinstance(value, torch.Tensor):
            current = value
            device = value.device
        else:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            current = torch.tensor(float(value), device=device, requires_grad=True)
        total = current if total is None else total + current

    if total is None:
        raise ValueError("The model returned no loss values")
    return total


def _evaluate(
    adapter: YOLOLoRAAdapter,
    dataloader: DataLoader | None,
) -> dict[str, float | None]:
    if dataloader is None:
        raise ValueError("val_loader is required; validation metrics cannot be fabricated")

    adapter.eval()
    losses: list[float] = []
    map_scores: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    batch_count = 0

    with torch.no_grad():
        for batch in dataloader:
            batch_count += 1
            batch_dict = batch if isinstance(batch, dict) else {"batch": batch}
            result = adapter.compute_loss(batch_dict)
            loss = _loss_from_dict(result)
            if not math.isfinite(loss):
                raise FloatingPointError("Validation produced a non-finite loss")
            losses.append(loss)

            metrics = batch_dict.get("metrics", {})
            if not isinstance(metrics, Mapping):
                raise TypeError("Batch metrics must be a mapping")
            for name, values in (
                ("mAP@0.5", map_scores),
                ("precision", precisions),
                ("recall", recalls),
            ):
                if name in metrics:
                    value = float(metrics[name])
                    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                        raise ValueError(f"{name} must be finite and between 0 and 1")
                    values.append(value)

    if batch_count == 0:
        raise ValueError("val_loader must contain at least one batch")

    def optional_mean(values: list[float]) -> float | None:
        if not values:
            return None
        if len(values) != batch_count:
            raise ValueError("Optional validation metrics must be present in every batch")
        return float(sum(values) / batch_count)

    val_loss = float(sum(losses) / batch_count)
    metrics = {
        "val_loss": val_loss,
        "mAP@0.5": optional_mean(map_scores),
        "precision": optional_mean(precisions),
        "recall": optional_mean(recalls),
    }
    return metrics


def fine_tune_loop(
    adapter: YOLOLoRAAdapter,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    config: FineTuneConfig | None = None,
) -> dict[str, float | str | None]:
    cfg = config or FineTuneConfig()
    if cfg.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if not math.isfinite(cfg.learning_rate) or cfg.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if cfg.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if cfg.gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be at least 1")
    if cfg.patience < 1:
        raise ValueError("patience must be at least 1")
    adapter.batch_size = cfg.batch_size
    adapter.gradient_accumulation = cfg.gradient_accumulation

    optimizer = torch.optim.Adam(adapter.parameters(), lr=cfg.learning_rate)
    scaler = GradScaler("cuda", enabled=adapter.mixed_precision and torch.cuda.is_available())
    checkpoint_manager = CheckpointManager(cfg.checkpoint_dir)

    start_epoch = 0
    resumed_metrics: dict[str, float | None] | None = None
    if cfg.resume_from is not None and not cfg.resume_from.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {cfg.resume_from}")
    if cfg.resume_from is not None:
        LOGGER.info("Resuming training from checkpoint %s", cfg.resume_from)
        payload = torch.load(cfg.resume_from, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
            raise ValueError("Resume checkpoint has an invalid schema")
        adapter.load_state_dict(payload["state_dict"])
        optimizer_state = payload.get("optimizer_state")
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        checkpoint_manager.best_checkpoint = cfg.resume_from
        metrics = payload.get("metrics")
        if isinstance(metrics, Mapping):
            val_loss = metrics.get("val_loss")
            if not isinstance(val_loss, int | float) or not math.isfinite(float(val_loss)):
                raise ValueError("Resume checkpoint is missing a finite validation loss")
            resumed_metrics = {
                name: float(value) if isinstance(value, int | float) else None
                for name, value in metrics.items()
            }
            checkpoint_manager.best_loss = float(val_loss)
            map_score = resumed_metrics.get("mAP@0.5")
            if map_score is not None and math.isfinite(map_score):
                checkpoint_manager.best_map = map_score
        else:
            raise ValueError("Resume checkpoint is missing validation metrics")
        checkpoint_epoch = payload.get("epoch")
        if not isinstance(checkpoint_epoch, int) or checkpoint_epoch < 0:
            raise ValueError("Resume checkpoint has an invalid epoch")
        start_epoch = checkpoint_epoch + 1

    best_metrics = resumed_metrics
    epochs_without_improvement = 0

    with start_run(experiment=cfg.experiment, run_name=cfg.run_name):
        log_run_params(
            {
                "learning_rate": cfg.learning_rate,
                "batch_size": cfg.batch_size,
                "gradient_accumulation": cfg.gradient_accumulation,
                "epochs": cfg.epochs,
            }
        )

        for epoch in range(start_epoch, cfg.epochs):
            adapter.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            accumulation_count = 0
            train_batch_count = 0

            for step, batch in enumerate(train_loader, start=1):
                train_batch_count += 1
                batch_dict = batch if isinstance(batch, dict) else {"batch": batch}

                autocast_context = torch.autocast(
                    device_type="cuda" if adapter.device.startswith("cuda") else "cpu",
                    enabled=adapter.mixed_precision and torch.cuda.is_available(),
                )
                with autocast_context:
                    loss_dict = adapter.compute_loss(batch_dict)
                    loss_tensor = _loss_tensor(loss_dict) / cfg.gradient_accumulation

                if not torch.isfinite(loss_tensor.detach()).all():
                    raise FloatingPointError(f"Training produced a non-finite loss at step {step}")

                if scaler.is_enabled():
                    scaler.scale(loss_tensor).backward()
                else:
                    loss_tensor.backward()

                accumulation_count += 1
                if accumulation_count % cfg.gradient_accumulation == 0:
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                running_loss += _loss_from_dict(loss_dict)

            if accumulation_count % cfg.gradient_accumulation:
                correction = cfg.gradient_accumulation / (
                    accumulation_count % cfg.gradient_accumulation
                )
                for parameter in adapter.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if train_batch_count == 0:
                raise ValueError("train_loader must contain at least one batch")
            val_metrics = _evaluate(adapter, val_loader)
            val_metrics["train_loss"] = running_loss / train_batch_count
            val_loss = val_metrics["val_loss"]
            if val_loss is None:
                raise RuntimeError("validation completed without a validation loss")
            improved_loss = val_loss < checkpoint_manager.best_loss
            checkpoint_manager.save(adapter, val_metrics, epoch, optimizer.state_dict())

            log_run_metrics(
                {name: value for name, value in val_metrics.items() if value is not None},
                step=epoch,
            )

            if improved_loss:
                best_metrics = val_metrics
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                LOGGER.info(
                    "Validation loss worsened (%.4f > %.4f); patience %s/%s",
                    val_loss,
                    checkpoint_manager.best_loss,
                    epochs_without_improvement,
                    cfg.patience,
                )

            if epochs_without_improvement >= cfg.patience:
                LOGGER.info("Early stopping triggered at epoch %s", epoch)
                break

        best_path = checkpoint_manager.load_best(adapter)
        if best_path:
            log_run_artifacts(best_path.as_posix())
            log_run_params({"best_checkpoint": best_path.as_posix()})
        if best_metrics:
            log_run_metrics(
                {
                    f"best_{name}": value
                    for name, value in best_metrics.items()
                    if name != "train_loss" and value is not None
                }
            )

    if best_metrics is None or best_path is None:
        raise RuntimeError("Training did not produce a validated checkpoint")
    summary: dict[str, float | str | None] = dict(best_metrics)
    summary["best_checkpoint"] = best_path.as_posix() if best_path else None
    return summary


__all__ = ["FineTuneConfig", "CheckpointManager", "fine_tune_loop"]
