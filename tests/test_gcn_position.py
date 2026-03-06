from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.graph.gcn_position_classifier import PositionClassifier, export_checkpoint, train_loop


def test_gcn_forward_shape() -> None:
    torch.manual_seed(0)
    x = torch.randn(5, 4)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 0, 2, 4],
            [1, 2, 3, 4, 0, 2, 0, 1],
        ],
        dtype=torch.long,
    )
    data = Data(x=x, edge_index=edge_index)

    model = PositionClassifier(in_channels=4, hidden_channels=8, num_layers=2, num_classes=3)
    logits = model(data)

    assert logits.shape == (5, 3)


def test_position_classification_accuracy(tmp_path) -> None:
    torch.manual_seed(42)

    features = torch.tensor(
        [
            [1.0, 0.1, 0.0],
            [0.9, 0.2, 0.0],
            [0.0, 1.0, 0.1],
            [0.1, 0.9, 0.2],
            [0.0, 0.0, 1.0],
            [0.1, 0.1, 0.9],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0],
            [1, 2, 3, 4, 5, 0, 0, 1, 2, 3, 4, 5],
        ],
        dtype=torch.long,
    )

    graph = Data(x=features, edge_index=edge_index, y=labels)
    loader = DataLoader([graph], batch_size=1)

    model = PositionClassifier(
        in_channels=3, hidden_channels=16, num_layers=2, num_classes=3, dropout=0.0
    )
    metrics = train_loop(model, loader, epochs=50, learning_rate=0.05)

    assert metrics["f1"] >= 0.7

    checkpoint_path = export_checkpoint(model, scaler={"mean": [0.5]}, output_dir=tmp_path)
    assert checkpoint_path.exists()
