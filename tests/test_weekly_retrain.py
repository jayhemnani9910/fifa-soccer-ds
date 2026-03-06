from __future__ import annotations

import json
from pathlib import Path

import torch

from src.train.weekly_retrainer import WeeklyRetrainer


class FakeLoader:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.metadata_filename = None
        self._metadata_calls = 0

    def download(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        sample = self.cache_dir / "fixture.csv"
        sample.write_text("match_id,home_team\n1,Barcelona\n", encoding="utf-8")
        return [sample]

    def load_metadata(self):
        self._metadata_calls += 1
        return []


class DummyAdapter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))


class TrainerStub:
    def __init__(self, checkpoint_root: Path, losses: list[float]) -> None:
        self.checkpoint_root = checkpoint_root
        self.losses = losses
        self._call = 0

    def __call__(self, adapter, train_loader, val_loader, config):
        checkpoint = self.checkpoint_root / f"ckpt_{self._call}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": adapter.state_dict()}, checkpoint)

        loss = self.losses[min(self._call, len(self.losses) - 1)]
        self._call += 1
        return {
            "val_loss": loss,
            "mAP@0.5": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "best_checkpoint": checkpoint.as_posix(),
        }


def evaluator_stub(adapter, test_loader):
    return {"mAP@0.5": 0.55, "f1": 0.6}


def test_retrain_checkpoint_rollback(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    loader = FakeLoader(cache_dir)
    checkpoint_root = tmp_path / "checkpoints"
    trainer = TrainerStub(checkpoint_root, losses=[0.3, 0.8])

    retrainer = WeeklyRetrainer(
        data_loader=loader,
        train_loader=lambda: [],
        val_loader=None,
        test_loader=None,
        adapter_factory=DummyAdapter,
        trainer_fn=trainer,
        evaluator_fn=evaluator_stub,
        version_file=tmp_path / "version_rollback.json",
    )

    retrainer.load_new_la_liga_data()
    first_summary = retrainer.run_fine_tune_loop()
    assert first_summary["rolled_back"] is False
    first_checkpoint = Path(first_summary["best_checkpoint"])
    assert first_checkpoint.exists()

    second_summary = retrainer.run_fine_tune_loop()
    assert second_summary["rolled_back"] is True
    assert second_summary["best_checkpoint"] == first_checkpoint.as_posix()
    meta = first_checkpoint.with_suffix(first_checkpoint.suffix + ".meta.json")
    assert meta.exists()
    metadata = json.loads(meta.read_text())
    assert metadata["dataset_version"] == retrainer.current_data_version


def test_data_version_increment(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    loader = FakeLoader(cache_dir)
    checkpoint_root = tmp_path / "checkpoints"
    trainer = TrainerStub(checkpoint_root, losses=[0.4])

    retrainer = WeeklyRetrainer(
        data_loader=loader,
        train_loader=lambda: [],
        val_loader=None,
        test_loader=None,
        adapter_factory=DummyAdapter,
        trainer_fn=trainer,
        version_file=tmp_path / "version.json",
    )

    info_first = retrainer.load_new_la_liga_data()
    assert info_first["version"] == 1
    info_second = retrainer.load_new_la_liga_data()
    assert info_second["version"] == 2


def test_retrain_exit_code(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    loader = FakeLoader(cache_dir)
    checkpoint_root = tmp_path / "checkpoints"

    success_trainer = TrainerStub(checkpoint_root, losses=[0.5])
    retrainer = WeeklyRetrainer(
        data_loader=loader,
        train_loader=lambda: [],
        val_loader=None,
        test_loader=None,
        adapter_factory=DummyAdapter,
        trainer_fn=success_trainer,
        evaluator_fn=evaluator_stub,
        version_file=tmp_path / "version_exit.json",
    )

    exit_code = retrainer.schedule_retrain()
    assert exit_code == 0

    def raises_trainer(*_args, **_kwargs):
        raise RuntimeError("boom")

    failing_retrainer = WeeklyRetrainer(
        data_loader=loader,
        train_loader=lambda: [],
        val_loader=None,
        test_loader=None,
        adapter_factory=DummyAdapter,
        trainer_fn=raises_trainer,
        evaluator_fn=evaluator_stub,
        version_file=tmp_path / "version_retry.json",
    )

    exit_code_failure = failing_retrainer.schedule_retrain()
    assert exit_code_failure == 1
