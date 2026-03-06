"""Fine-tuning loop for the YOLOv8 LoRA adapter."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from src.detect.yolo_lora_adapter import YOLOLoRAAdapter
from src.utils.mlflow_helper import start_run

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
        metrics: dict[str, float],
        epoch: int,
        optimizer_state: dict | None = None,
    ) -> Path:
        snapshot = {
            "epoch": epoch,
            "state_dict": adapter.state_dict(),
            "metrics": metrics,
        }
        if optimizer_state is not None:
            snapshot["optimizer_state"] = optimizer_state
        path = self.directory / f"epoch_{epoch:02d}.pt"
        torch.save(snapshot, path)

        if metrics.get("mAP@0.5", float("-inf")) >= self.best_map:
            self.best_map = metrics["mAP@0.5"]
            self.best_checkpoint = path

        best_loss = metrics.get("val_loss")
        if best_loss is not None and best_loss <= self.best_loss:
            self.best_loss = best_loss
        return path

    def load_best(self, adapter: YOLOLoRAAdapter) -> Path | None:
        if self.best_checkpoint and self.best_checkpoint.exists():
            LOGGER.info("Restoring best checkpoint from %s", self.best_checkpoint)
            payload = torch.load(self.best_checkpoint, map_location="cpu")
            adapter.load_state_dict(payload["state_dict"])
        return self.best_checkpoint


def _loss_from_dict(loss_dict: dict[str, float | torch.Tensor]) -> float:
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


def _loss_tensor(loss_dict: dict[str, float | torch.Tensor]) -> torch.Tensor:
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
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        total = torch.tensor(0.0, device=device, requires_grad=True)
    return total


def _evaluate(
    adapter: YOLOLoRAAdapter,
    dataloader: DataLoader | None,
) -> dict[str, float]:
    if dataloader is None:
        return {"val_loss": 0.0, "mAP@0.5": 0.0, "precision": 0.0, "recall": 0.0}

    adapter.eval()
    losses: list[float] = []
    map_scores: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            batch_dict = batch if isinstance(batch, dict) else {"batch": batch}
            result = adapter.compute_loss(batch_dict)
            losses.append(_loss_from_dict(result))

            metrics = batch_dict.get("metrics", {})
            map_scores.append(float(metrics.get("mAP@0.5", 0.0)))
            precisions.append(float(metrics.get("precision", 0.0)))
            recalls.append(float(metrics.get("recall", 0.0)))

    val_loss = float(sum(losses) / max(len(losses), 1))
    metrics = {
        "val_loss": val_loss,
        "mAP@0.5": float(sum(map_scores) / max(len(map_scores), 1)),
        "precision": float(sum(precisions) / max(len(precisions), 1)),
        "recall": float(sum(recalls) / max(len(recalls), 1)),
    }
    return metrics


def fine_tune_loop(
    adapter: YOLOLoRAAdapter,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    config: FineTuneConfig | None = None,
) -> dict[str, float]:
    cfg = config or FineTuneConfig()
    adapter.batch_size = cfg.batch_size
    adapter.gradient_accumulation = cfg.gradient_accumulation

    optimizer = torch.optim.Adam(adapter.parameters(), lr=cfg.learning_rate)
    scaler = GradScaler(enabled=adapter.mixed_precision and torch.cuda.is_available())
    checkpoint_manager = CheckpointManager(cfg.checkpoint_dir)

    if cfg.resume_from and cfg.resume_from.exists():
        LOGGER.info("Resuming training from checkpoint %s", cfg.resume_from)
        payload = torch.load(cfg.resume_from, map_location="cpu")
        adapter.load_state_dict(payload["state_dict"])
        optimizer_state = payload.get("optimizer_state")
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        checkpoint_manager.best_checkpoint = cfg.resume_from
        metrics = payload.get("metrics")
        if metrics:
            checkpoint_manager.best_map = metrics.get("mAP@0.5", checkpoint_manager.best_map)
            checkpoint_manager.best_loss = metrics.get("val_loss", checkpoint_manager.best_loss)

    best_metrics: dict[str, float] | None = None
    epochs_without_improvement = 0

    with start_run(experiment=cfg.experiment, run_name=cfg.run_name) as _:
        import mlflow  # local import for easier patching in tests

        mlflow.log_params(
            {
                "learning_rate": cfg.learning_rate,
                "batch_size": cfg.batch_size,
                "gradient_accumulation": cfg.gradient_accumulation,
                "epochs": cfg.epochs,
            }
        )

        for epoch in range(cfg.epochs):
            adapter.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0

            for step, batch in enumerate(train_loader, start=1):
                batch_dict = batch if isinstance(batch, dict) else {"batch": batch}

                autocast_context = torch.autocast(
                    device_type="cuda" if adapter.device.startswith("cuda") else "cpu",
                    enabled=adapter.mixed_precision and torch.cuda.is_available(),
                )
                with autocast_context:
                    loss_dict = adapter.compute_loss(batch_dict)
                    loss_tensor = _loss_tensor(loss_dict) / cfg.gradient_accumulation

                if torch.isnan(loss_tensor.detach()).any():
                    LOGGER.warning("Skipping NaN loss at step %s", step)
                    continue

                if scaler.is_enabled():
                    scaler.scale(loss_tensor).backward()
                else:
                    loss_tensor.backward()

                if step % cfg.gradient_accumulation == 0:
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                running_loss += float(loss_tensor.detach().cpu())

            val_metrics = _evaluate(adapter, val_loader)
            val_metrics["train_loss"] = running_loss / max(len(train_loader), 1)
            checkpoint_manager.save(adapter, val_metrics, epoch, optimizer.state_dict())

            mlflow.log_metrics(
                {
                    "train_loss": val_metrics["train_loss"],
                    "val_loss": val_metrics["val_loss"],
                    "mAP@0.5": val_metrics["mAP@0.5"],
                    "precision": val_metrics["precision"],
                    "recall": val_metrics["recall"],
                },
                step=epoch,
            )

            if val_metrics["val_loss"] <= checkpoint_manager.best_loss:
                best_metrics = val_metrics
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                LOGGER.info(
                    "Validation loss worsened (%.4f > %.4f); patience %s/%s",
                    val_metrics["val_loss"],
                    checkpoint_manager.best_loss,
                    epochs_without_improvement,
                    cfg.patience,
                )

            if epochs_without_improvement >= cfg.patience:
                LOGGER.info("Early stopping triggered at epoch %s", epoch)
                break

        best_path = checkpoint_manager.load_best(adapter)
        if best_path:
            mlflow.log_artifact(best_path.as_posix())
            mlflow.log_param("best_checkpoint", best_path.as_posix())
        if best_metrics:
            mlflow.log_metrics(
                {
                    "best_mAP@0.5": best_metrics["mAP@0.5"],
                    "best_precision": best_metrics["precision"],
                    "best_recall": best_metrics["recall"],
                }
            )

    summary = best_metrics or {
        "mAP@0.5": checkpoint_manager.best_map,
        "val_loss": checkpoint_manager.best_loss,
        "precision": 0.0,
        "recall": 0.0,
    }
    summary = dict(summary)
    summary["best_checkpoint"] = best_path.as_posix() if best_path else None
    return summary


__all__ = ["FineTuneConfig", "CheckpointManager", "fine_tune_loop"]
