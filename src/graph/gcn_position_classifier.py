"""GraphSAGE-based position classifier for player role labelling."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv

from src.eval.metrics import f1_score


class PositionClassifier(nn.Module):
    """GraphSAGE classifier producing role logits per tracked player."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(
        self, data: Data | torch.Tensor, edge_index: torch.Tensor | None = None
    ) -> torch.Tensor:
        if isinstance(data, Data):
            x = data.x
            edge_index = data.edge_index if edge_index is None else edge_index
        else:
            if edge_index is None:
                raise ValueError("edge_index must be provided when passing raw tensors")
            x = data

        if edge_index is None:
            raise ValueError("edge_index is required for message passing")

        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)

        logits = self.classifier(x)
        return logits


def _evaluate(
    model: PositionClassifier, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            preds = logits.argmax(dim=-1)
            predictions.extend(preds.cpu().tolist())
            targets.extend(batch.y.cpu().tolist())

    score = f1_score(predictions, targets) if targets else 0.0
    return {"f1": score}


def train_loop(
    model: PositionClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    device: torch.device | None = None,
) -> dict[str, float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_metrics = {"f1": 0.0, "epoch": -1}
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())

        eval_loader = val_loader or train_loader
        metrics = _evaluate(model, eval_loader, device)
        metrics["loss"] = running_loss / max(len(train_loader), 1)
        metrics["epoch"] = epoch

        if metrics["f1"] >= best_metrics["f1"]:
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_metrics


def export_checkpoint(
    model: PositionClassifier,
    scaler: Any | None = None,
    output_dir: Path | None = None,
    filename: str = "position_classifier.pt",
) -> Path:
    base_dir = (output_dir or Path("build/models")).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = base_dir / filename

    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "meta": {
            "in_channels": model.in_channels,
            "hidden_channels": model.hidden_channels,
            "num_layers": model.num_layers,
            "num_classes": model.num_classes,
        },
    }

    if scaler is not None:
        payload["scaler"] = scaler

    torch.save(payload, checkpoint_path)
    return checkpoint_path


__all__ = ["PositionClassifier", "train_loop", "export_checkpoint"]
