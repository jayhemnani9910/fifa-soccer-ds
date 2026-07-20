from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from src.models.train_gcn import load_graph_dataset


def test_ultralytics_restricted_checkpoint_loading_is_the_default() -> None:
    assert os.environ["ULTRALYTICS_SAFE_LOAD"] == "true"


def test_pickled_graph_dataset_requires_explicit_trust(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.pt"
    torch.save({"data": [1, 2, 3]}, dataset_path)

    with pytest.raises(ValueError, match="explicit trust"):
        load_graph_dataset(dataset_path)


def test_trusted_pickled_graph_dataset_can_be_loaded(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.pt"
    torch.save({"data": [1, 2, 3]}, dataset_path)

    assert load_graph_dataset(dataset_path, trust_pickle=True) == {"data": [1, 2, 3]}
