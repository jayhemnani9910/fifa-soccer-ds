from __future__ import annotations

import hashlib
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from torch_geometric.data import InMemoryDataset

from src.models import train_gcn


def _dataset_mock(length: int = 2) -> MagicMock:
    dataset = MagicMock(spec=InMemoryDataset)
    dataset.__len__.return_value = length
    return dataset


def test_dataset_version_is_content_derived_and_missing_files_fail(tmp_path) -> None:
    dataset_path = tmp_path / "graphs.pt"
    payload = b"reviewed dataset artifact"
    dataset_path.write_bytes(payload)

    assert train_gcn.get_dataset_version(dataset_path) == (
        f"sha256_{hashlib.sha256(payload).hexdigest()[:16]}"
    )
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        train_gcn.get_dataset_version(tmp_path / "missing.pt")


def test_training_enters_and_exits_mlflow_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    @contextmanager
    def fake_start_run(**_kwargs):
        events.append("entered")
        try:
            yield object()
        finally:
            events.append("exited")

    def fail_training(**kwargs):
        assert kwargs["tracking_enabled"] is True
        raise RuntimeError("training failed")

    monkeypatch.setattr(train_gcn, "start_run", fake_start_run)
    monkeypatch.setattr(train_gcn, "_run_training_impl", fail_training)

    with pytest.raises(RuntimeError, match="training failed"):
        train_gcn.run_training(_dataset_mock(), enable_mlflow=True)

    assert events == ["entered", "exited"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"epochs": 0}, "must be positive"),
        ({"lr": float("nan")}, "lr must be finite"),
        ({"weight_decay": -1.0}, "weight_decay"),
        ({"seed": -1}, "seed"),
    ],
)
def test_training_rejects_invalid_hyperparameters(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        train_gcn.run_training(_dataset_mock(), enable_mlflow=False, **kwargs)


def test_training_requires_a_non_empty_validation_split() -> None:
    with pytest.raises(ValueError, match="at least two graphs"):
        train_gcn.run_training(_dataset_mock(length=1), enable_mlflow=False)
