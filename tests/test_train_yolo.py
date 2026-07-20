from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from src.detect import train_yolo


class TinyAdapter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.mixed_precision = False
        self.device = "cpu"
        self.batch_size = 1
        self.gradient_accumulation = 1

    def compute_loss(self, _batch):  # type: ignore[no-untyped-def]
        return {"total_loss": self.weight.square()}


def test_gradient_accumulation_steps_final_partial_batch(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    optimizers: list[torch.optim.Adam] = []
    real_adam = torch.optim.Adam

    class CountingAdam(real_adam):
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            super().__init__(*args, **kwargs)
            self.step_calls = 0
            optimizers.append(self)

        def step(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            self.step_calls += 1
            return super().step(*args, **kwargs)

    monkeypatch.setattr(train_yolo.torch.optim, "Adam", CountingAdam)
    monkeypatch.setattr(train_yolo, "start_run", lambda **_kwargs: nullcontext(None))
    monkeypatch.setattr(train_yolo, "log_run_params", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_yolo, "log_run_metrics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_yolo, "log_run_artifacts", lambda *_args, **_kwargs: None)

    train_yolo.fine_tune_loop(
        TinyAdapter(),  # type: ignore[arg-type]
        train_loader=[{}, {}, {}],  # type: ignore[arg-type]
        val_loader=[{}],  # type: ignore[arg-type]
        config=train_yolo.FineTuneConfig(
            epochs=1,
            gradient_accumulation=2,
            checkpoint_dir=tmp_path,
        ),
    )

    assert optimizers[0].step_calls == 2


def test_validation_metrics_are_unknown_when_not_measured(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(train_yolo, "start_run", lambda **_kwargs: nullcontext(None))
    monkeypatch.setattr(train_yolo, "log_run_params", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_yolo, "log_run_metrics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_yolo, "log_run_artifacts", lambda *_args, **_kwargs: None)

    summary = train_yolo.fine_tune_loop(
        TinyAdapter(),  # type: ignore[arg-type]
        train_loader=[{}],  # type: ignore[arg-type]
        val_loader=[{}],  # type: ignore[arg-type]
        config=train_yolo.FineTuneConfig(epochs=1, checkpoint_dir=tmp_path),
    )

    assert summary["mAP@0.5"] is None
    assert summary["precision"] is None
    assert summary["recall"] is None
    assert isinstance(summary["val_loss"], float)


def test_fine_tuning_requires_validation_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="val_loader"):
        train_yolo.fine_tune_loop(
            TinyAdapter(),  # type: ignore[arg-type]
            train_loader=[{}],  # type: ignore[arg-type]
            config=train_yolo.FineTuneConfig(epochs=1, checkpoint_dir=tmp_path),
        )


def test_checkpoint_selection_uses_lowest_validation_loss(tmp_path: Path) -> None:
    manager = train_yolo.CheckpointManager(tmp_path)
    adapter = TinyAdapter()

    first = manager.save(adapter, {"val_loss": 2.0, "mAP@0.5": 0.9}, 0)
    second = manager.save(adapter, {"val_loss": 1.0, "mAP@0.5": 0.8}, 1)

    assert first != second
    assert manager.best_checkpoint == second
    assert manager.best_loss == 1.0
    assert not list(tmp_path.glob("*.tmp"))
