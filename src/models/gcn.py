"""GraphSAGE building blocks for experimental interaction classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool


class GraphSAGENet(nn.Module):
    """Two-layer GraphSAGE classifier for interaction labels."""

    def __init__(
        self, in_channels: int = 4, hidden_channels: int = 64, num_classes: int = 3
    ) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.head = nn.Linear(hidden_channels, num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        pooled = global_mean_pool(x, batch)
        return self.head(pooled)


@dataclass(slots=True)
class ForwardOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor


def forward_pass(model: GraphSAGENet, data: Data) -> ForwardOutput:
    logits = model(data)
    probabilities = torch.softmax(logits, dim=-1)
    return ForwardOutput(logits=logits, probabilities=probabilities)


def create_dataloader(dataset: Any, batch_size: int = 8, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
