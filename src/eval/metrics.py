"""Evaluation helpers for graph-based predictions.

This module provides standard classification metrics for multi-class problems:
- Accuracy: Fraction of correct predictions
- F1 Score: Harmonic mean of precision and recall (macro/micro averaging)
- Confusion Matrix: Per-class prediction breakdown

Metrics are useful for evaluating:
- Graph classification tasks (possession, interception, etc.)
- Track quality assessment
- Model performance across training epochs
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ConfusionMatrix:
    matrix: np.ndarray
    labels: Sequence[int]


def accuracy(predictions: Sequence[int], targets: Sequence[int]) -> float:
    preds = np.asarray(list(predictions))
    targs = np.asarray(list(targets))
    if preds.shape != targs.shape or preds.size == 0:
        return 0.0
    return float(np.mean(preds == targs))


def f1_score(predictions: Sequence[int], targets: Sequence[int], average: str = "macro") -> float:
    """Compute F1 score for a multi-class setting."""

    preds = np.asarray(list(predictions))
    targs = np.asarray(list(targets))
    labels = sorted(set(targs.tolist()) | set(preds.tolist()))
    if not labels:
        return 0.0

    f1_values: list[float] = []

    for label in labels:
        tp = int(np.sum((preds == label) & (targs == label)))
        fp = int(np.sum((preds == label) & (targs != label)))
        fn = int(np.sum((preds != label) & (targs == label)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_values.append(f1)

    if average == "macro":
        return float(np.mean(f1_values))

    if average == "micro":
        tp = sum(int(np.sum((preds == label) & (targs == label))) for label in labels)
        fp = sum(int(np.sum((preds == label) & (targs != label))) for label in labels)
        fn = sum(int(np.sum((preds != label) & (targs == label))) for label in labels)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return float(
            (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        )

    raise ValueError(f"Unsupported average mode: {average}")


def confusion_matrix(predictions: Sequence[int], targets: Sequence[int]) -> ConfusionMatrix:
    preds = np.asarray(list(predictions))
    targs = np.asarray(list(targets))
    labels = sorted(set(targs.tolist()) | set(preds.tolist()))
    if not labels:
        return ConfusionMatrix(matrix=np.zeros((0, 0), dtype=np.int32), labels=[])

    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int32)

    for pred, targ in zip(preds, targs, strict=False):
        matrix[label_to_idx[targ], label_to_idx[pred]] += 1

    return ConfusionMatrix(matrix=matrix, labels=labels)


def log_evaluation_to_mlflow(
    predictions: Sequence[int],
    targets: Sequence[int],
    output_dir: Path | None = None,
) -> dict[str, float]:
    """Compute metrics and log to MLflow if available."""
    try:
        import mlflow
    except ImportError:
        LOGGER.warning("MLflow not available; skipping experiment logging.")
        mlflow = None

    acc = accuracy(predictions, targets)
    f1_macro = f1_score(predictions, targets, average="macro")
    f1_micro = f1_score(predictions, targets, average="micro")
    cm = confusion_matrix(predictions, targets)

    metrics_dict = {"accuracy": acc, "f1_macro": f1_macro, "f1_micro": f1_micro}

    if mlflow:
        mlflow.log_metric("eval_accuracy", acc)
        mlflow.log_metric("eval_f1_macro", f1_macro)
        mlflow.log_metric("eval_f1_micro", f1_micro)

    # Save confusion matrix artifact
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cm_path = output_dir / "confusion_matrix.json"
        cm_data = {
            "matrix": cm.matrix.tolist(),
            "labels": list(cm.labels),
        }
        cm_path.write_text(__import__("json").dumps(cm_data, indent=2))
        if mlflow:
            mlflow.log_artifact(str(cm_path))

    return metrics_dict


__all__ = [
    "accuracy",
    "f1_score",
    "confusion_matrix",
    "ConfusionMatrix",
    "log_evaluation_to_mlflow",
]
