import pytest
import torch

torch_geometric = pytest.importorskip("torch_geometric")
from torch_geometric.data import Data  # noqa: E402

from src.models.gcn import GraphSAGENet, forward_pass  # noqa: E402


def test_graphsage_forward_shape():
    num_nodes = 10
    in_channels = 4
    num_classes = 3
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    batch = torch.zeros(num_nodes, dtype=torch.long)

    data = Data(
        x=torch.randn(num_nodes, in_channels),
        edge_index=edge_index,
        batch=batch,
        y=torch.zeros(1, dtype=torch.long),
    )
    model = GraphSAGENet(in_channels=in_channels, hidden_channels=16, num_classes=num_classes)
    output = forward_pass(model, data)

    assert output.logits.shape == (1, num_classes)
    assert output.probabilities.shape == (1, num_classes)
    assert torch.allclose(output.probabilities.sum(dim=-1), torch.ones(1), atol=1e-4)
