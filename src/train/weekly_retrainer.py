"""Weekly retraining orchestration for the La Liga detection stack."""

from __future__ import annotations

import fcntl
import json
import logging
import math
import os
import re
import shutil

# Subprocesses use fixed argument vectors and shutil-resolved executables.
import subprocess  # nosec B404
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO, cast

import torch

from src.data.la_liga_loader import KaggleDataLoader, version_data_with_dvc
from src.detect.train_yolo import FineTuneConfig, fine_tune_loop
from src.detect.yolo_lora_adapter import YOLOLoRAAdapter
from src.utils.mlflow_helper import (
    log_run_artifacts,
    log_run_metrics,
    log_run_params,
    start_run,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticVersion:
    """Semantic versioning for model releases."""

    major: int = 1
    minor: int = 0
    patch: int = 0

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def increment_patch(self) -> SemanticVersion:
        """Increment patch version (bug fixes)."""
        return SemanticVersion(self.major, self.minor, self.patch + 1)

    def increment_minor(self) -> SemanticVersion:
        """Increment minor version (new features)."""
        return SemanticVersion(self.major, self.minor + 1, 0)

    def increment_major(self) -> SemanticVersion:
        """Increment major version (breaking changes)."""
        return SemanticVersion(self.major + 1, 0, 0)

    @classmethod
    def from_string(cls, version_str: str) -> SemanticVersion:
        """Parse version string like '1.2.3'."""
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version_str.strip())
        if not match:
            raise ValueError(f"Invalid version format: {version_str}")
        return cls(int(match.group(1)), int(match.group(2)), int(match.group(3)))

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary for JSON serialization."""
        return {"major": self.major, "minor": self.minor, "patch": self.patch}


@dataclass
class ModelVersionConflict:
    """Represents a model version conflict and its resolution."""

    version_a: SemanticVersion
    version_b: SemanticVersion
    metrics_a: dict[str, float]
    metrics_b: dict[str, float]
    resolution: str = "auto"  # "auto", "manual", "rollback"
    resolved_version: SemanticVersion | None = None
    conflict_reason: str = ""

    def compare_metrics(self, primary_metric: str = "mAP@0.5") -> tuple[str, float]:
        """Compare versions based on primary metric."""
        metric_a = self.metrics_a.get(primary_metric, 0.0)
        metric_b = self.metrics_b.get(primary_metric, 0.0)

        if abs(metric_a - metric_b) < 0.001:  # Within tolerance
            return "equal", 0.0
        elif metric_a > metric_b:
            return "a_better", metric_a - metric_b
        else:
            return "b_better", metric_b - metric_a

    def auto_resolve(self, primary_metric: str = "mAP@0.5") -> SemanticVersion:
        """Automatically resolve conflict based on metrics."""
        comparison, diff = self.compare_metrics(primary_metric)

        if comparison == "equal":
            # Equal performance - use newer version
            self.resolution = "auto_equal"
            metric_value = self.metrics_a.get(primary_metric, 0.0)
            self.conflict_reason = f"Equal performance ({primary_metric}: {metric_value:.4f})"
            self.resolved_version = max(
                self.version_a, self.version_b, key=lambda v: (v.major, v.minor, v.patch)
            )
        elif comparison == "a_better":
            self.resolution = "auto_a"
            self.conflict_reason = f"Version A better ({primary_metric}: +{diff:.4f})"
            self.resolved_version = self.version_a
        else:
            self.resolution = "auto_b"
            self.conflict_reason = f"Version B better ({primary_metric}: +{diff:.4f})"
            self.resolved_version = self.version_b

        return self.resolved_version


class ModelVersionManager:
    """Manages semantic versioning and conflict resolution for models."""

    def __init__(self, version_file: Path, primary_metric: str = "mAP@0.5"):
        self.version_file = version_file
        self.primary_metric = primary_metric
        self.versions_file = version_file.parent / "versions_history.json"

    def get_current_version(self) -> SemanticVersion:
        """Get current semantic version."""
        if self.version_file.exists():
            try:
                data = json.loads(self.version_file.read_text())
                version_str = data.get("semantic_version", "1.0.0")
                return SemanticVersion.from_string(version_str)
            except (json.JSONDecodeError, ValueError, KeyError):
                return SemanticVersion(1, 0, 0)
        return SemanticVersion(1, 0, 0)

    def increment_version(
        self, version_type: str = "patch", metrics: dict[str, float] | None = None
    ) -> SemanticVersion:
        """Increment version based on change type."""
        current = self.get_current_version()

        if version_type == "major":
            new_version = current.increment_major()
        elif version_type == "minor":
            new_version = current.increment_minor()
        else:  # patch
            new_version = current.increment_patch()

        # Atomic update
        with AtomicFileWriter(self.version_file) as writer:
            writer.write(
                json.dumps(
                    {
                        "semantic_version": str(new_version),
                        "version_type": version_type,
                        "timestamp": time.time(),
                        "previous_version": str(current),
                    },
                    indent=2,
                )
            )

        # Update history
        self._update_version_history(new_version, version_type, current, metrics)

        return new_version

    def detect_conflicts(self, new_metrics: dict[str, float]) -> ModelVersionConflict | None:
        """Detect version conflicts based on performance comparison."""
        current_version = self.get_current_version()

        # Check if there's a recent version with similar performance
        if self.versions_file.exists():
            try:
                history = json.loads(self.versions_file.read_text())
                recent_versions = history.get("versions", [])[-5:]  # Last 5 versions

                for version_info in recent_versions:
                    if version_info.get("metrics"):
                        old_metrics = version_info["metrics"]
                        old_version = SemanticVersion.from_string(version_info["semantic_version"])

                        # Compare primary metrics
                        old_metric = old_metrics.get(self.primary_metric, 0.0)
                        new_metric = new_metrics.get(self.primary_metric, 0.0)

                        # If performance is very similar, it might be a conflict
                        if math.isclose(
                            old_metric,
                            new_metric,
                            rel_tol=0.0,
                            abs_tol=0.005 + 1e-12,
                        ):  # 0.5 percentage-point tolerance
                            conflict = ModelVersionConflict(
                                version_a=current_version,
                                version_b=old_version,
                                metrics_a=new_metrics,
                                metrics_b=old_metrics,
                            )

                            # Auto-resolve if clear winner
                            resolved = conflict.auto_resolve(self.primary_metric)
                            conflict.resolved_version = resolved

                            return conflict

            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        return None

    def _update_version_history(
        self,
        new_version: SemanticVersion,
        version_type: str,
        previous_version: SemanticVersion,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Update version history with atomic write."""
        history_data: dict[str, Any] = {"versions": []}

        if self.versions_file.exists():
            try:
                history_data = json.loads(self.versions_file.read_text())
            except (json.JSONDecodeError, ValueError):
                history_data = {"versions": []}

        # Add new version entry
        version_entry = {
            "semantic_version": str(new_version),
            "version_type": version_type,
            "timestamp": time.time(),
            "previous_version": str(previous_version),
            "metrics": dict(metrics or {}),
        }

        history_data["versions"].append(version_entry)

        # Keep only last 50 versions
        if len(history_data["versions"]) > 50:
            history_data["versions"] = history_data["versions"][-50:]

        # Atomic write
        with AtomicFileWriter(self.versions_file) as writer:
            writer.write(json.dumps(history_data, indent=2))

    def get_version_info(self) -> dict[str, Any]:
        """Get comprehensive version information."""
        current = self.get_current_version()

        info: dict[str, Any] = {
            "current_version": str(current),
            "version_components": current.to_dict(),
            "version_file": str(self.version_file),
            "history_file": str(self.versions_file),
        }

        if self.versions_file.exists():
            try:
                history = json.loads(self.versions_file.read_text())
                info["total_versions"] = len(history.get("versions", []))
                info["recent_versions"] = history.get("versions", [])[-5:]
            except (json.JSONDecodeError, ValueError):
                info["total_versions"] = 0
                info["recent_versions"] = []

        return info


class RetrainingLock:
    """File-based lock that coordinates retraining across threads and processes."""

    def __init__(
        self,
        lock_file: Path | None = None,
        timeout: float = 7200,
        poll_interval: float = 1.0,
        stale_threshold: float | None = None,
    ) -> None:
        if lock_file is None:
            runtime_root = Path(os.environ.get("XDG_RUNTIME_DIR", tempfile.gettempdir()))
            lock_dir = runtime_root / f"fifa-soccer-ds-{os.getuid()}"
            lock_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            directory_fd = os.open(
                lock_dir,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
            try:
                if os.fstat(directory_fd).st_uid != os.getuid():
                    raise PermissionError(f"Runtime lock directory is not user-owned: {lock_dir}")
                os.fchmod(directory_fd, 0o700)
            finally:
                os.close(directory_fd)
            lock_file = lock_dir / "weekly-retrain.lock"
        self.lock_file = lock_file
        self.timeout = float(timeout)
        self.poll_interval = max(float(poll_interval), 0.01)
        self.stale_threshold = float(stale_threshold or timeout)
        self.lock_fd: TextIO | None = None
        self._locked = False

    def acquire(self) -> bool:
        """Acquire the lock, returning ``False`` when the timeout expires."""
        if self._locked:
            return True

        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = self.lock_file.open("a+", encoding="utf-8")
        start_time = time.monotonic()

        while True:
            try:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                elapsed = time.monotonic() - start_time
                if elapsed >= self.timeout:
                    lock_fd.close()
                    return False
                time.sleep(min(self.poll_interval, self.timeout - elapsed))
                continue

            lock_fd.seek(0)
            lock_fd.truncate()
            lock_fd.write(f"{os.getpid()}\n{time.time()}\n")
            lock_fd.flush()
            os.fsync(lock_fd.fileno())
            self.lock_fd = lock_fd
            self._locked = True
            LOGGER.info("Acquired retraining lock: %s", self.lock_file)
            return True

    def release(self) -> None:
        """Release the lock while retaining the inode used by waiting processes."""
        if self.lock_fd is None:
            return
        try:
            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
        finally:
            self.lock_fd.close()
            self.lock_fd = None
            self._locked = False
        LOGGER.info("Released retraining lock: %s", self.lock_file)

    def is_locked(self) -> bool:
        """Return whether this instance currently owns the lock."""
        return self._locked

    def __enter__(self) -> RetrainingLock:
        if not self.acquire():
            raise TimeoutError(f"Could not acquire lock after {self.timeout} seconds")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def _is_stale_lock(self) -> bool:
        """Inspect lock metadata without breaking an active operating-system lock."""
        if not self.lock_file.exists():
            return False
        try:
            pid_text, timestamp_text = self.lock_file.read_text(encoding="utf-8").split("\n", 1)
            lock_time = float(timestamp_text.strip())
            try:
                os.kill(int(pid_text), 0)
            except OSError:
                return True
            return time.time() - lock_time > self.stale_threshold
        except (ValueError, OSError):
            return True


@contextmanager
def checkpoint_lock(checkpoint_dir: Path):
    """Context manager for atomic checkpoint operations."""
    lock_file = checkpoint_dir / ".checkpoint.lock"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with lock_file.open("a+", encoding="utf-8") as lock_fd:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)


class AtomicFileWriter:
    """Atomic file writing with temp file and rename."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.temp_path: Path | None = None
        self._handle: TextIO | None = None

    def __enter__(self) -> TextIO:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.file_path.parent,
            prefix=f".{self.file_path.name}.",
            suffix=".tmp",
            delete=False,
        )
        self._handle = cast(TextIO, handle)
        self.temp_path = Path(handle.name)
        return self._handle

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._handle is None or self.temp_path is None:
            return

        try:
            if exc_type is None:
                self._handle.flush()
                os.fsync(self._handle.fileno())
            self._handle.close()

            if exc_type is None:
                os.replace(self.temp_path, self.file_path)
            else:
                self.temp_path.unlink(missing_ok=True)
        except Exception:
            self.temp_path.unlink(missing_ok=True)
            raise
        finally:
            self._handle = None
            self.temp_path = None


def _current_git_commit() -> str:
    git_executable = shutil.which("git")
    if git_executable is None:
        return "unknown"
    try:
        # The executable is resolved and every argument is constant.
        result = subprocess.run(  # nosec B603
            [git_executable, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Unable to resolve git commit hash: %s", exc)
        return "unknown"


def _increment_version_safe(version_file: Path, snapshot: dict[str, Any] | None = None) -> int:
    """Atomically increment and persist a dataset version and optional snapshot."""
    version_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file = version_file.with_name(f".{version_file.name}.lock")

    with lock_file.open("a+", encoding="utf-8") as lock_fd:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        try:
            try:
                payload = json.loads(version_file.read_text(encoding="utf-8"))
                version = int(payload.get("version", 0)) + 1
            except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError):
                version = 1

            payload = dict(snapshot or {})
            payload["version"] = version
            with AtomicFileWriter(version_file) as writer:
                writer.write(json.dumps(payload, indent=2))
        finally:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)

    return version


def _mlflow_key(name: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_. /:")
    return "".join(char if char in allowed else "_" for char in name)


@dataclass(slots=True)
class WeeklyRetrainer:
    data_loader: KaggleDataLoader
    train_loader: Any
    val_loader: Any | None
    test_loader: Any | None
    adapter_factory: Callable[[], YOLOLoRAAdapter] = YOLOLoRAAdapter
    trainer_fn: Callable[
        [YOLOLoRAAdapter, Any, Any | None, FineTuneConfig | None], dict[str, Any]
    ] = fine_tune_loop
    evaluator_fn: Callable[[YOLOLoRAAdapter, Any], dict[str, float]] | None = None
    fine_tune_config: FineTuneConfig = field(default_factory=FineTuneConfig)
    experiment: str = "weekly-retrainer"
    version_file: Path = Path("data/processed/la_liga_version.json")
    last_best_metrics: dict[str, float] | None = field(init=False, default=None)
    last_checkpoint: Path | None = field(init=False, default=None)
    current_data_version: int | None = field(init=False, default=None)
    current_dataset_hash: str | None = field(init=False, default=None)
    _adapter: YOLOLoRAAdapter | None = field(init=False, default=None)

    # Version management
    primary_metric: str = "mAP@0.5"
    version_manager: ModelVersionManager | None = field(init=False, default=None)
    semantic_version_file: Path | None = None

    def __post_init__(self) -> None:
        """Initialize version manager after dataclass creation."""
        if self.semantic_version_file is None:
            self.semantic_version_file = self.version_file.parent / "semantic_version.json"
        self.semantic_version_file.parent.mkdir(parents=True, exist_ok=True)
        self.version_manager = ModelVersionManager(self.semantic_version_file, self.primary_metric)

    # ----------------------------------------------------------------- pipeline
    def load_new_la_liga_data(self) -> dict[str, Any]:
        LOGGER.info("Refreshing La Liga dataset via Kaggle API")

        cached_files = (
            self.data_loader.download()
        )  # pragma: no cover (network side effects mocked in tests)

        try:
            metadata = self.data_loader.load_metadata()
        except FileNotFoundError:
            metadata = []

        dvc_info = version_data_with_dvc(self.data_loader.cache_dir)
        self.current_dataset_hash = dvc_info["dataset_hash"]
        serialized_metadata = [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in metadata
        ]
        snapshot = {
            "files": [str(path) for path in cached_files],
            "metadata": serialized_metadata,
            "dataset_hash": self.current_dataset_hash,
        }
        self.current_data_version = _increment_version_safe(self.version_file, snapshot)

        LOGGER.info(
            "Dataset version v%s (hash=%s)",
            self.current_data_version,
            self.current_dataset_hash,
        )

        return {
            "files": [str(path) for path in cached_files],
            "metadata": serialized_metadata,
            "version": self.current_data_version,
            "dataset_hash": self.current_dataset_hash,
        }

    def run_fine_tune_loop(self) -> dict[str, Any]:
        adapter = self.adapter_factory()
        self._adapter = adapter

        train_loader = self._resolve_loader(self.train_loader)
        val_loader = self._resolve_loader(self.val_loader)

        summary = self.trainer_fn(adapter, train_loader, val_loader, self.fine_tune_config)
        raw_val_loss = summary.get("val_loss")
        if (
            isinstance(raw_val_loss, bool)
            or not isinstance(raw_val_loss, int | float)
            or not math.isfinite(float(raw_val_loss))
        ):
            raise RuntimeError("Trainer did not report a finite validation loss")
        current_val_loss = float(raw_val_loss)

        if self.last_best_metrics and current_val_loss > self.last_best_metrics.get(
            "val_loss", float("inf")
        ):
            LOGGER.warning(
                "Validation loss increased; rolling back to checkpoint %s", self.last_checkpoint
            )
            if self.last_checkpoint and self.last_checkpoint.exists():
                payload = torch.load(self.last_checkpoint, map_location="cpu", weights_only=True)
                adapter.load_state_dict(payload["state_dict"])
                summary["best_checkpoint"] = self.last_checkpoint.as_posix()
                summary["rolled_back"] = True
                summary["val_loss"] = self.last_best_metrics["val_loss"]
            else:
                LOGGER.warning("No previous checkpoint available for rollback.")
        else:
            summary["rolled_back"] = False

        effective_checkpoint = summary.get("best_checkpoint")
        if effective_checkpoint:
            checkpoint_path = Path(effective_checkpoint)
            self._tag_checkpoint(checkpoint_path)
            self.last_checkpoint = checkpoint_path
            if self.current_dataset_hash:
                version_data_with_dvc(
                    checkpoint_path.parent,
                    param_key="checkpoint_path",
                    hash_key="checkpoint_hash",
                )

        self.last_best_metrics = {"val_loss": current_val_loss}
        map_score = summary.get("mAP@0.5")
        if isinstance(map_score, int | float) and not isinstance(map_score, bool):
            if not math.isfinite(float(map_score)):
                raise RuntimeError("Trainer reported a non-finite mAP@0.5")
            self.last_best_metrics["mAP@0.5"] = float(map_score)
        return summary

    def evaluate_on_test_set(self) -> dict[str, float]:
        test_loader = self._resolve_loader(self.test_loader)
        if self.evaluator_fn is not None and self._adapter is not None:
            metrics = self.evaluator_fn(self._adapter, test_loader)
        else:
            raise RuntimeError(
                "A trained adapter and evaluator_fn are required; refusing to publish placeholder metrics"
            )

        for name, value in metrics.items():
            if not math.isfinite(float(value)):
                raise RuntimeError(f"Evaluator reported a non-finite {name}")
        if self.current_data_version is not None:
            metrics.setdefault("dataset_version", float(self.current_data_version))
        return metrics

    def _register_best_checkpoint(
        self, checkpoint_path: Path, metrics: dict[str, float]
    ) -> dict[str, Any]:
        """Register best checkpoint with MLflow, DVC, and version management."""
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")
        primary_value = metrics.get(self.primary_metric)
        if primary_value is None or not math.isfinite(float(primary_value)):
            raise ValueError(
                f"A finite {self.primary_metric} evaluation metric is required for registration"
            )
        summary = {
            "checkpoint_path": str(checkpoint_path),
            "metrics": metrics,
            "timestamp": time.time(),
        }

        # Version management and conflict resolution
        if self.version_manager:
            # Detect version conflicts
            conflict = self.version_manager.detect_conflicts(metrics)

            if conflict:
                LOGGER.warning(f"Model version conflict detected: {conflict.conflict_reason}")

                # Auto-resolve conflict
                resolved_version = conflict.auto_resolve(self.primary_metric)
                summary["version_conflict"] = True
                summary["conflict_resolution"] = conflict.resolution
                summary["resolved_version"] = str(resolved_version)

                log_run_params(
                    {
                        "conflict.version_a": str(conflict.version_a),
                        "conflict.version_b": str(conflict.version_b),
                        "conflict.resolution": conflict.resolution,
                        "conflict.reason": conflict.conflict_reason,
                    }
                )
                log_run_metrics(
                    {
                        "conflict.metric_a": conflict.metrics_a.get(self.primary_metric, 0.0),
                        "conflict.metric_b": conflict.metrics_b.get(self.primary_metric, 0.0),
                        "conflict.metric_diff": abs(
                            conflict.metrics_a.get(self.primary_metric, 0.0)
                            - conflict.metrics_b.get(self.primary_metric, 0.0)
                        ),
                    }
                )
            else:
                # No conflict - increment version based on improvement
                version_type = self._determine_version_type(metrics)
                new_version = self.version_manager.increment_version(version_type, metrics)
                summary["semantic_version"] = str(new_version)
                summary["version_type"] = version_type

        if self.last_best_metrics:
            # Compare with previous best
            prev_mAP = self.last_best_metrics.get("mAP@0.5", 0.0)
            curr_mAP = metrics.get("mAP@0.5", 0.0)
            improvement = curr_mAP - prev_mAP

            if improvement > 0.001:  # Significant improvement
                summary["improvement"] = improvement
                summary["status"] = "new_best"
            elif improvement < -0.005:  # Significant degradation
                summary["status"] = "degraded"
                LOGGER.warning(f"Performance degraded by {improvement:.4f}")
            else:
                summary["status"] = "stable"

        self.last_best_metrics = metrics.copy()
        self.last_checkpoint = checkpoint_path
        log_run_artifacts(checkpoint_path.as_posix(), "best_model")

        return summary

    def _determine_version_type(self, metrics: dict[str, float]) -> str:
        """Determine version increment type based on performance changes."""
        if not self.last_best_metrics:
            return "minor"  # First version

        prev_mAP = self.last_best_metrics.get("mAP@0.5", 0.0)
        curr_mAP = metrics.get("mAP@0.5", 0.0)
        improvement = curr_mAP - prev_mAP

        # Check other metrics too
        prev_f1 = self.last_best_metrics.get("F1", 0.0)
        curr_f1 = metrics.get("F1", 0.0)
        f1_improvement = curr_f1 - prev_f1

        # Determine version type
        if improvement > 0.02 or f1_improvement > 0.02:  # >2% improvement
            return "minor"  # New feature/improvement
        elif improvement < -0.01 or f1_improvement < -0.01:  # >1% degradation
            return "major"  # Breaking change
        else:
            return "patch"  # Bug fix/optimization

    def schedule_retrain(self) -> int:
        try:
            lock_path = self.version_file.with_suffix(self.version_file.suffix + ".retrain.lock")
            with RetrainingLock(lock_file=lock_path, timeout=0.1, poll_interval=0.02):
                dataset_info = self.load_new_la_liga_data()
                with start_run(
                    experiment=self.experiment,
                    run_name=f"weekly-v{dataset_info['version']}",
                ):
                    log_run_params(
                        {
                            "dataset_version": dataset_info["version"],
                            "dataset_hash": dataset_info["dataset_hash"],
                        }
                    )

                    train_summary = self.run_fine_tune_loop()
                    log_run_metrics(
                        {
                            _mlflow_key(f"train_{key}"): float(value)
                            for key, value in train_summary.items()
                            if isinstance(value, int | float)
                        }
                    )

                    eval_summary = self.evaluate_on_test_set()
                    log_run_metrics(
                        {
                            _mlflow_key(f"eval_{k}"): float(v)
                            for k, v in eval_summary.items()
                            if isinstance(v, int | float)
                        }
                    )
                    if self.last_checkpoint is None:
                        raise RuntimeError("Trainer did not produce a checkpoint")
                    self._register_best_checkpoint(self.last_checkpoint, eval_summary)

            return 0
        except Exception as exc:  # pragma: no cover - ensures cron-friendly exit codes
            LOGGER.exception("Weekly retraining failed: %s", exc)
            return 1

    # ------------------------------------------------------------------ helpers
    def _tag_checkpoint(self, checkpoint: Path) -> None:
        if not checkpoint.exists():
            return

        metadata = {
            "commit": _current_git_commit(),
            "dataset_version": self.current_data_version,
            "dataset_hash": self.current_dataset_hash,
        }

        # Add semantic version information
        if self.version_manager:
            version_info = self.version_manager.get_version_info()
            metadata.update(
                {
                    "semantic_version": version_info.get("current_version"),
                    "version_components": version_info.get("version_components"),
                    "primary_metric": self.primary_metric,
                }
            )

        meta_path = checkpoint.with_suffix(checkpoint.suffix + ".meta.json")
        with AtomicFileWriter(meta_path) as writer:
            writer.write(json.dumps(metadata, indent=2))

    def _resolve_loader(self, loader: Any) -> Any:
        if callable(loader):
            return loader()
        return loader


def main() -> int:  # pragma: no cover - CLI entrypoint
    LOGGER.error(
        "WeeklyRetrainer requires application-supplied train/validation/test loaders and an "
        "evaluator; construct it from a deployment-specific launcher."
    )
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
