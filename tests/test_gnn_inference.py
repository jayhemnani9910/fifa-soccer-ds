"""Smoke test for GNN position-classifier inference wiring."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pyg = pytest.importorskip("torch_geometric")

from torch_geometric.data import Data

from src.graph.gcn_position_classifier import (
    PositionClassifier,
    export_checkpoint,
    predict_positions,
)


@pytest.mark.smoke
def test_predict_positions_roundtrip(tmp_path):
    torch.manual_seed(0)

    num_classes = 3
    model = PositionClassifier(
        in_channels=4, hidden_channels=8, num_layers=2, num_classes=num_classes
    )
    ckpt_path = export_checkpoint(model, output_dir=tmp_path, filename="gnn.pt")
    assert ckpt_path.exists()

    x = torch.randn(5, 4)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long
    )
    meta = [(0, 10), (0, 11), (1, 10), (1, 11), (2, 10)]
    graph = Data(x=x, edge_index=edge_index, meta=meta)

    preds = predict_positions(graph, ckpt_path, device=torch.device("cpu"))

    assert len(preds) == 5
    for i, entry in enumerate(preds):
        assert entry["node_idx"] == i
        assert 0 <= entry["label"] < num_classes
        assert 0.0 <= entry["confidence"] <= 1.0
        assert entry["frame_id"] == meta[i][0]
        assert entry["track_id"] == meta[i][1]
