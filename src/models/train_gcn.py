"""Training loop skeleton for the GraphSAGE classifier.

This module provides:
- Full training loop with validation and metrics tracking
- Checkpoint saving and loading for best models
- Early stopping to prevent overfitting
- Learning rate scheduling (optional)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil

# Subprocesses use fixed argument vectors and shutil-resolved executables.
import subprocess  # nosec B404
import uuid
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import random_split
from torch_geometric.data import InMemoryDataset

from src.eval.metrics import accuracy, f1_score
from src.models.gcn import GraphSAGENet, create_dataloader, forward_pass
from src.utils.mlflow_helper import log_run_artifacts, log_run_metrics, log_run_params, start_run

LOGGER = logging.getLogger(__name__)


def load_graph_dataset(path: Path, *, trust_pickle: bool = False):
    """Load a legacy pickled PyG dataset only after explicit trust confirmation.

    ``torch_geometric`` datasets saved as whole Python objects require pickle and
    may execute code during deserialization. Callers must establish provenance
    (for example, a reviewed DVC artifact) before opting in.
    """
    if not trust_pickle:
        raise ValueError(
            "Refusing to deserialize a pickled graph dataset without explicit trust confirmation"
        )
    LOGGER.warning("Loading trusted pickled graph dataset: %s", path)
    # The explicit trust_pickle gate above is required because legacy PyG objects use pickle.
    return torch.load(  # nosec B614
        path, map_location="cpu", weights_only=False
    )


def get_git_hash() -> str:
    """Get current git commit hash."""
    git_executable = shutil.which("git")
    if git_executable is None:
        return "unknown"
    try:
        return (
            # The executable is resolved and every argument is constant.
            subprocess.check_output(  # nosec B603
                [git_executable, "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_dataset_version(dataset_path: Path) -> str:
    """Return a content-derived dataset provenance identifier."""
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as dataset_file:
        for chunk in iter(lambda: dataset_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256_{digest.hexdigest()[:16]}"


def generate_experiment_name(
    model_type: str = "graphsage",
    dataset_path: Path | None = None,
    custom_suffix: str | None = None,
) -> str:
    """Generate unique MLflow experiment name to avoid collisions.

    Args:
        model_type: Type of model being trained
        dataset_path: Path to dataset for versioning
        custom_suffix: Additional suffix for uniqueness

    Returns:
        Unique experiment name string
    """
    git_hash = get_git_hash()
    date_str = datetime.now(UTC).strftime("%Y%m%d")

    # Build base name
    parts = ["fifa_gnn", model_type, date_str, git_hash]

    # Add dataset version if available
    if dataset_path:
        dataset_version = get_dataset_version(dataset_path)
        parts.append(dataset_version)

    # Add custom suffix if provided
    if custom_suffix:
        parts.append(custom_suffix)

    # Join and ensure reasonable length
    experiment_name = "_".join(parts)

    # Truncate if too long (MLflow has limits)
    if len(experiment_name) > 100:
        # Keep essential parts only
        experiment_name = f"fifa_gnn_{model_type}_{date_str}_{git_hash}"
        if custom_suffix:
            experiment_name += f"_{custom_suffix[:8]}"

    return experiment_name


class EarlyStopping:
    """Early stopping callback to halt training when validation metric plateaus.

    Monitors a validation metric and stops training if it doesn't improve for
    a specified number of epochs.
    """

    def __init__(self, patience: int = 3, metric_name: str = "val_acc"):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            metric_name: Name of metric to monitor (val_acc, val_f1, etc.)
        """
        self.patience = patience
        self.metric_name = metric_name
        self.best_value = float("-inf")
        self.counter = 0
        self.best_epoch = 0

    def __call__(self, current_value: float, epoch: int) -> bool:
        """Check if training should stop.

        Args:
            current_value: Current metric value
            epoch: Current epoch number

        Returns:
            True if training should stop, False otherwise
        """
        if current_value > self.best_value:
            self.best_value = current_value
            self.counter = 0
            self.best_epoch = epoch
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                LOGGER.info(
                    "Early stopping triggered: no improvement for %d epochs. "
                    "Best value: %.4f at epoch %d",
                    self.patience,
                    self.best_value,
                    self.best_epoch,
                )
                return True
            return False


def train_epoch(model, loader, criterion, optimizer, device: torch.device) -> float:
    """Execute one training epoch.

    Args:
        model: GraphSAGE model
        loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer instance
        device: Torch device (cpu/cuda)

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        outputs = forward_pass(model, batch)
        loss = criterion(outputs.logits, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def evaluate(model, loader, device: torch.device) -> tuple[float, float]:
    """Evaluate model on validation/test set.

    Args:
        model: GraphSAGE model
        loader: Validation data loader
        device: Torch device (cpu/cuda)

    Returns:
        Tuple of (accuracy, f1_score)
    """
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = forward_pass(model, batch)
            probabilities = outputs.probabilities
            predictions = probabilities.argmax(dim=-1)
            preds.extend(predictions.cpu().tolist())
            targets.extend(batch.y.cpu().tolist())
    acc = accuracy(preds, targets)
    f1 = f1_score(preds, targets)
    return acc, f1


def save_checkpoint(model: GraphSAGENet, optimizer, epoch: int, metrics: dict, path: Path):
    """Save model checkpoint with optimizer state and metrics.

    Args:
        model: Model to save
        optimizer: Optimizer with state
        epoch: Current epoch number
        metrics: Dictionary of metrics at this epoch
        path: Path where checkpoint will be saved
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    LOGGER.info(
        "Checkpoint saved: %s (epoch %d, val_acc=%.4f)", path, epoch, metrics.get("val_acc", 0)
    )


def load_checkpoint(model: GraphSAGENet, optimizer, path: Path) -> dict:
    """Load model checkpoint.

    Args:
        model: Model to load state into
        optimizer: Optimizer to restore state
        path: Path to checkpoint file

    Returns:
        Dictionary containing checkpoint metadata
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    LOGGER.info("Checkpoint loaded: %s (epoch %d)", path, checkpoint["epoch"])
    return checkpoint


def _run_training_impl(
    dataset,
    model: GraphSAGENet | None = None,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | None = None,
    early_stopping_patience: int = 3,
    checkpoint_dir: str | Path | None = None,
    tracking_enabled: bool = False,
    seed: int = 42,
) -> tuple[GraphSAGENet, dict]:
    """Execute a validated training run inside an optional active MLflow run.

    Args:
        dataset: torch_geometric InMemoryDataset
        model: Pre-initialized model (or None to create new)
        epochs: Maximum number of epochs
        batch_size: Mini-batch size
        lr: Learning rate
        weight_decay: L2 regularization weight
        device: Device string (cpu/cuda)
        early_stopping_patience: Patience for early stopping
        checkpoint_dir: Directory to save checkpoints (None = don't save)
        tracking_enabled: Whether an active MLflow run is available
        seed: Reproducibility seed used for model initialization and dataset splitting

    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_length = max(1, min(len(dataset) - 1, int(len(dataset) * 0.8)))
    val_length = len(dataset) - train_length
    split_generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        dataset, [train_length, val_length], generator=split_generator
    )

    model = model or GraphSAGENet(
        in_channels=dataset.num_node_features, num_classes=dataset.num_classes
    )
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device_obj)

    train_loader = create_dataloader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_subset, batch_size=batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    early_stopper = EarlyStopping(patience=early_stopping_patience, metric_name="val_acc")
    checkpoint_path = None
    if checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir) / "best_model.pt"

    metrics_history: dict[str, list[float]] = {
        "train_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    # Track final epoch for logging
    final_epoch = 0

    if tracking_enabled:
        # Log training parameters
        training_params = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "early_stopping_patience": early_stopping_patience,
            "model_features": dataset.num_node_features,
            "model_classes": dataset.num_classes,
            "train_samples": len(train_subset),
            "val_samples": len(val_subset),
            "seed": seed,
        }
        log_run_params(training_params, prefix="training")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device_obj)
        val_acc, val_f1 = evaluate(model, val_loader, device_obj)

        metrics_history["train_loss"].append(train_loss)
        metrics_history["val_acc"].append(val_acc)
        metrics_history["val_f1"].append(val_f1)

        # Log metrics to MLflow if enabled
        if tracking_enabled:
            epoch_metrics = {
                "train_loss": train_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "epoch": epoch,
            }
            log_run_metrics(epoch_metrics, step=epoch)

        LOGGER.info(
            "Epoch %d/%d | loss=%.4f | val_acc=%.3f | val_f1=%.3f",
            epoch,
            epochs,
            train_loss,
            val_acc,
            val_f1,
        )

        # Save best checkpoint
        if checkpoint_path and val_acc == max(metrics_history["val_acc"]):
            save_checkpoint(
                model, optimizer, epoch, {"val_acc": val_acc, "val_f1": val_f1}, checkpoint_path
            )

            # Log best model as artifact if MLflow enabled
            if tracking_enabled and checkpoint_path:
                log_run_artifacts(str(checkpoint_path), "best_model")

        # Check early stopping
        if early_stopper(val_acc, epoch):
            LOGGER.info("Training stopped early at epoch %d", epoch)
            if checkpoint_path and checkpoint_path.exists():
                load_checkpoint(model, optimizer, checkpoint_path)
            final_epoch = epoch
            break
        else:
            final_epoch = epoch

    # Log final metrics and artifacts
    if tracking_enabled:
        final_metrics = {
            "final_train_loss": metrics_history["train_loss"][-1]
            if metrics_history["train_loss"]
            else 0,
            "final_val_acc": metrics_history["val_acc"][-1] if metrics_history["val_acc"] else 0,
            "final_val_f1": metrics_history["val_f1"][-1] if metrics_history["val_f1"] else 0,
            "best_val_acc": max(metrics_history["val_acc"]) if metrics_history["val_acc"] else 0,
            "best_val_f1": max(metrics_history["val_f1"]) if metrics_history["val_f1"] else 0,
            "total_epochs": final_epoch,
            "early_stopped": early_stopper.counter >= early_stopping_patience,
        }
        log_run_metrics(final_metrics)

        # Log metrics history as artifact
        if checkpoint_path:
            metrics_path = checkpoint_path.parent / "training_metrics.json"
            with metrics_path.open("w") as f:
                json.dump(metrics_history, f, indent=2)
            log_run_artifacts(str(metrics_path), "metrics")

    return model, metrics_history


def run_training(
    dataset,
    model: GraphSAGENet | None = None,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | None = None,
    early_stopping_patience: int = 3,
    checkpoint_dir: str | Path | None = None,
    dataset_path: Path | None = None,
    experiment_name: str | None = None,
    enable_mlflow: bool = True,
    seed: int = 42,
) -> tuple[GraphSAGENet, dict]:
    """Run reproducible GraphSAGE training with scoped MLflow tracking."""
    if not isinstance(dataset, InMemoryDataset):
        raise TypeError("Dataset must be a torch_geometric InMemoryDataset")
    if len(dataset) < 2:
        raise ValueError("Dataset must contain at least two graphs for train/validation splitting")
    if epochs <= 0 or batch_size <= 0 or early_stopping_patience <= 0:
        raise ValueError("epochs, batch_size, and early_stopping_patience must be positive")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    dataset_version = get_dataset_version(dataset_path) if dataset_path else "not_provided"
    resolved_experiment = experiment_name or generate_experiment_name(
        model_type="graphsage",
        dataset_path=dataset_path,
        custom_suffix=f"epochs_{epochs}",
    )
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    implementation_args = {
        "dataset": dataset,
        "model": model,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "device": str(resolved_device),
        "early_stopping_patience": early_stopping_patience,
        "checkpoint_dir": checkpoint_dir,
        "seed": seed,
    }
    if not enable_mlflow:
        return _run_training_impl(**implementation_args, tracking_enabled=False)

    with start_run(
        experiment=resolved_experiment,
        run_name=f"graphsage_train_{datetime.now(UTC).strftime('%H%M%S')}",
        tags={
            "model_type": "graphsage",
            "git_hash": get_git_hash(),
            "dataset_version": dataset_version,
            "device": resolved_device.type,
            "epochs": str(epochs),
            "batch_size": str(batch_size),
            "learning_rate": str(lr),
            "seed": str(seed),
        },
    ) as active_run:
        return _run_training_impl(
            **implementation_args,
            tracking_enabled=active_run is not None,
        )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Train the GraphSAGE classifier.")
    parser.add_argument(
        "--dataset", required=True, help="Path to a torch_geometric InMemoryDataset (.pt)."
    )
    parser.add_argument(
        "--trust-pickled-dataset",
        action="store_true",
        help="Acknowledge that the dataset is trusted; .pt dataset loading can execute code.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Maximum number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (L2).")
    parser.add_argument("--device", default=None, help="Torch device string (cpu, cuda, etc.).")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed.")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Patience for early stopping.",
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints", help="Directory to save model checkpoints."
    )
    parser.add_argument(
        "--output-metrics", default="metrics.json", help="Path to save training metrics."
    )
    parser.add_argument("--experiment-name", default=None, help="Custom MLflow experiment name.")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking.")
    return parser


def main() -> None:
    """CLI entrypoint."""  # pragma: no cover
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    try:
        dataset = load_graph_dataset(dataset_path, trust_pickle=args.trust_pickled_dataset)
    except ValueError as exc:
        parser.error(str(exc))
    model, metrics = run_training(
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_dir=args.checkpoint_dir,
        dataset_path=dataset_path,
        experiment_name=args.experiment_name,
        enable_mlflow=not args.no_mlflow,  # Enable MLflow unless disabled
        seed=args.seed,
    )

    # Save metrics
    metrics_path = Path(args.output_metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    LOGGER.info("Training metrics saved to %s", metrics_path)


if __name__ == "__main__":  # pragma: no cover
    main()
