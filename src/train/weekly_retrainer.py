"""Weekly retraining orchestration for the La Liga detection stack."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.data.la_liga_loader import KaggleDataLoader, version_data_with_dvc
from src.detect.train_yolo import FineTuneConfig, fine_tune_loop
from src.detect.yolo_lora_adapter import YOLOLoRAAdapter
from src.utils.mlflow_helper import start_run

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticVersion:
    """Semantic versioning for model releases."""
    major: int = 1
    minor: int = 0
    patch: int = 0
    
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def increment_patch(self) -> "SemanticVersion":
        """Increment patch version (bug fixes)."""
        return SemanticVersion(self.major, self.minor, self.patch + 1)
    
    def increment_minor(self) -> "SemanticVersion":
        """Increment minor version (new features)."""
        return SemanticVersion(self.major, self.minor + 1, 0)
    
    def increment_major(self) -> "SemanticVersion":
        """Increment major version (breaking changes)."""
        return SemanticVersion(self.major + 1, 0, 0)
    
    @classmethod
    def from_string(cls, version_str: str) -> "SemanticVersion":
        """Parse version string like '1.2.3'."""
        match = re.match(r'^(\d+)\.(\d+)\.(\d+)$', version_str.strip())
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
            self.conflict_reason = f"Equal performance ({primary_metric}: {self.metrics_a[primary_metric]:.4f})"
            self.resolved_version = max(self.version_a, self.version_b, key=lambda v: (v.major, v.minor, v.patch))
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
    
    def increment_version(self, version_type: str = "patch") -> SemanticVersion:
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
            writer.write(json.dumps({
                "semantic_version": str(new_version),
                "version_type": version_type,
                "timestamp": time.time(),
                "previous_version": str(current)
            }, indent=2))
        
        # Update history
        self._update_version_history(new_version, version_type, current)
        
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
                        if abs(old_metric - new_metric) < 0.005:  # 0.5% tolerance
                            conflict = ModelVersionConflict(
                                version_a=current_version,
                                version_b=old_version,
                                metrics_a=new_metrics,
                                metrics_b=old_metrics
                            )
                            
                            # Auto-resolve if clear winner
                            resolved = conflict.auto_resolve(self.primary_metric)
                            conflict.resolved_version = resolved
                            
                            return conflict
                            
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
        
        return None
    
    def _update_version_history(self, new_version: SemanticVersion, version_type: str, previous_version: SemanticVersion):
        """Update version history with atomic write."""
        history_data = {"versions": []}
        
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
            "previous_version": str(previous_version)
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
        
        info = {
            "current_version": str(current),
            "version_components": current.to_dict(),
            "version_file": str(self.version_file),
            "history_file": str(self.versions_file)
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
    """File-based distributed lock for retraining coordination."""
    
    def __init__(
        self, 
        lock_file: Path = Path("/tmp/fifa_weekly_retrain.lock"),
        timeout: int = 7200,  # 2 hours
        poll_interval: float = 1.0
    ):
        self.lock_file = lock_file
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.lock_fd = None
        
    def __enter__(self):
        """Acquire exclusive lock with timeout."""
        start_time = time.time()
        
        while True:
            try:
                # Create lock file directory
                self.lock_file.parent.mkdir(parents=True, exist_ok=True)
                self.lock_fd = open(self.lock_file, 'w')
                
                # Try to acquire exclusive lock
                fcntl.flock(
                    self.lock_fd.fileno(), 
                    fcntl.LOCK_EX | fcntl.LOCK_NB
                )
                
                # Lock acquired successfully
                self.lock_fd.write(f"{os.getpid()}\n{time.time()}\n")
                self.lock_fd.flush()
                
                LOGGER.info(f"Acquired retraining lock: {self.lock_file}")
                return self
                
            except BlockingIOError:
                # Lock is held by another process
                if time.time() - start_time > self.timeout:
                    raise TimeoutError(
                        f"Could not acquire lock after {self.timeout} seconds"
                    )
                
                # Check if lock is stale
                if self._is_stale_lock():
                    LOGGER.warning("Breaking stale retraining lock")
                    self._break_stale_lock()
                    continue
                    
                LOGGER.info(
                    f"Retraining lock held, waiting {self.poll_interval}s..."
                )
                time.sleep(self.poll_interval)
                
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release lock."""
        if self.lock_fd:
            try:
                fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
                self.lock_fd.close()
                self.lock_file.unlink(missing_ok=True)
                LOGGER.info(f"Released retraining lock: {self.lock_file}")
            except (OSError, IOError) as e:
                LOGGER.error(f"Error releasing lock: {e}")
                
    def _is_stale_lock(self) -> bool:
        """Check if lock file is stale (process dead)."""
        if not self.lock_file.exists():
            return False
            
        try:
            content = self.lock_file.read_text().strip()
            if '\n' not in content:
                return True
                
            pid_str, timestamp_str = content.split('\n', 1)
            pid = int(pid_str)
            lock_time = float(timestamp_str)
            
            # Check if process still exists
            try:
                os.kill(pid, 0)  # Signal 0 checks if process exists
            except OSError:
                return True  # Process doesn't exist
                
            # Check if lock is older than timeout
            return time.time() - lock_time > self.timeout
            
        except (ValueError, OSError):
            return True
            
    def _break_stale_lock(self):
        """Force remove stale lock."""
        try:
            if self.lock_fd:
                self.lock_fd.close()
            self.lock_file.unlink(missing_ok=True)
            LOGGER.info("Removed stale retraining lock")
        except OSError as e:
            LOGGER.error(f"Failed to remove stale lock: {e}")


@contextmanager
def checkpoint_lock(checkpoint_dir: Path):
    """Context manager for atomic checkpoint operations."""
    lock_file = checkpoint_dir / ".checkpoint.lock"
    
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        lock_fd = open(lock_file, 'w')
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            lock_fd.close()
            lock_file.unlink(missing_ok=True)
        except (OSError, IOError):
            pass


class AtomicFileWriter:
    """Atomic file writing with temp file and rename."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.temp_path = None
        
    def __enter__(self):
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create temp file in same directory
        self.temp_path = self.file_path.with_suffix(
            f".tmp.{os.getpid()}.{int(time.time())}"
        )
        return open(self.temp_path, 'w')
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and self.temp_path and self.temp_path.exists():
            # Atomic rename
            self.temp_path.rename(self.file_path)
        elif self.temp_path and self.temp_path.exists():
            # Cleanup on error
            self.temp_path.unlink(missing_ok=True)


def _current_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Unable to resolve git commit hash: %s", exc)
        return "unknown"


def _increment_version(version_file: Path) -> int:
    version_file.parent.mkdir(parents=True, exist_ok=True)
    if version_file.exists():
        try:
            payload = json.loads(version_file.read_text())
            version = int(payload.get("version", 0)) + 1
        except (json.JSONDecodeError, ValueError, TypeError):
            version = 1
    else:
        version = 1
    version_file.write_text(json.dumps({"version": version}, indent=2))
    return version


def _increment_version_safe(version_file: Path) -> int:
    """Thread-safe version increment with file locking."""
    try:
        with open(version_file, 'r') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                if version_file.stat().st_size > 0:
                    payload = json.loads(f.read())
                    version = int(payload.get("version", 0)) + 1
                else:
                    version = 1
            except (json.JSONDecodeError, ValueError, TypeError):
                version = 1
    except FileNotFoundError:
        version = 1
    
    # Atomic write with lock
    with open(version_file, 'w') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(json.dumps({"version": version}, indent=2))
        
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
    semantic_version_file: Path = field(init=False, default=Path("checkpoints/semantic_version.json"))

    def __post_init__(self) -> None:
        """Initialize version manager after dataclass creation."""
        self.semantic_version_file.parent.mkdir(parents=True, exist_ok=True)
        self.version_manager = ModelVersionManager(
            self.semantic_version_file, 
            self.primary_metric
        )

    # ----------------------------------------------------------------- pipeline
    def load_new_la_liga_data(self) -> dict[str, Any]:
        LOGGER.info("Refreshing La Liga dataset via Kaggle API")
        
        # Use atomic file operations for version file
        with AtomicFileWriter(self.version_file) as f:
            cached_files = (
                self.data_loader.download()
            )  # pragma: no cover (network side effects mocked in tests)

            try:
                metadata = self.data_loader.load_metadata()
            except FileNotFoundError:
                metadata = []

            self.current_data_version = _increment_version_safe(self.version_file)
            dvc_info = version_data_with_dvc(self.data_loader.cache_dir)
            self.current_dataset_hash = dvc_info["dataset_hash"]

            LOGGER.info(
                "Dataset version v%s (hash=%s)", self.current_data_version, self.current_dataset_hash
            )
            
            # Write version info atomically
            version_info = {
                "files": [str(path) for path in cached_files],
                "metadata": metadata,
                "version": self.current_data_version,
                "dataset_hash": self.current_dataset_hash,
            }
            f.write(json.dumps(version_info, indent=2))
            
        return {
            "files": [str(path) for path in cached_files],
            "metadata": metadata,
            "version": self.current_data_version,
            "dataset_hash": self.current_dataset_hash,
        }

    def run_fine_tune_loop(self) -> dict[str, Any]:
        adapter = self.adapter_factory()
        self._adapter = adapter

        train_loader = self._resolve_loader(self.train_loader)
        val_loader = self._resolve_loader(self.val_loader)

        summary = self.trainer_fn(adapter, train_loader, val_loader, self.fine_tune_config)
        current_val_loss = float(summary.get("val_loss", 0.0))

        if self.last_best_metrics and current_val_loss > self.last_best_metrics.get(
            "val_loss", float("inf")
        ):
            LOGGER.warning(
                "Validation loss increased; rolling back to checkpoint %s", self.last_checkpoint
            )
            if self.last_checkpoint and self.last_checkpoint.exists():
                payload = torch.load(self.last_checkpoint, map_location="cpu")
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

        self.last_best_metrics = {
            "val_loss": float(summary.get("val_loss", 0.0)),
            "mAP@0.5": float(summary.get("mAP@0.5", 0.0)),
        }
        return summary

    def evaluate_on_test_set(self) -> dict[str, float]:
        test_loader = self._resolve_loader(self.test_loader)
        if self.evaluator_fn is not None and self._adapter is not None:
            metrics = self.evaluator_fn(self._adapter, test_loader)
        else:
            metrics = {"mAP@0.5": 0.0, "f1": 0.0}

        if self.test_loader and isinstance(metrics, dict):
            metrics.setdefault("dataset_version", self.current_data_version or 0)
        return metrics
    
    def _register_best_checkpoint(self, checkpoint_path: Path, metrics: dict[str, float]) -> dict[str, Any]:
        """Register best checkpoint with MLflow, DVC, and version management."""
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
                
                # Log conflict to MLflow
                with start_run(experiment=f"{self.experiment}-conflicts", run_name=f"conflict-{int(time.time())}"):
                    import mlflow
                    mlflow.log_params({
                        "version_a": str(conflict.version_a),
                        "version_b": str(conflict.version_b),
                        "resolution": conflict.resolution,
                        "reason": conflict.conflict_reason
                    })
                    mlflow.log_metrics({
                        "metric_a": conflict.metrics_a.get(self.primary_metric, 0.0),
                        "metric_b": conflict.metrics_b.get(self.primary_metric, 0.0),
                        "metric_diff": abs(conflict.metrics_a.get(self.primary_metric, 0.0) - 
                                       conflict.metrics_b.get(self.primary_metric, 0.0))
                    })
            else:
                # No conflict - increment version based on improvement
                version_type = self._determine_version_type(metrics)
                new_version = self.version_manager.increment_version(version_type)
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
            with start_run(experiment=self.experiment, run_name=None):
                import mlflow

                dataset_info = self.load_new_la_liga_data()
                mlflow.set_tag("mlflow.runName", f"weekly-v{dataset_info['version']}")
                mlflow.log_param("dataset_version", dataset_info["version"])
                mlflow.log_param("dataset_hash", dataset_info["dataset_hash"])

                train_summary = self.run_fine_tune_loop()
                for key, value in train_summary.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(_mlflow_key(f"train_{key}"), value)

                eval_summary = self.evaluate_on_test_set()
                if isinstance(eval_summary, dict):
                    mlflow.log_metrics(
                        {
                            _mlflow_key(f"eval_{k}"): v
                            for k, v in eval_summary.items()
                            if isinstance(v, (int, float))
                        }
                    )

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
            metadata.update({
                "semantic_version": version_info.get("current_version"),
                "version_components": version_info.get("version_components"),
                "primary_metric": self.primary_metric
            })
        
        meta_path = checkpoint.with_suffix(checkpoint.suffix + ".meta.json")
        meta_path.write_text(json.dumps(metadata, indent=2))

    def _resolve_loader(self, loader: Any) -> Any:
        if callable(loader):
            return loader()
        return loader


def main() -> int:  # pragma: no cover - CLI entrypoint
    from torch.utils.data import DataLoader

    from src.data.la_liga_loader import KaggleDataLoader

    loader = KaggleDataLoader(dataset="laliga/dataset")
    # Placeholder dataloaders; real implementation should hook actual datasets.
    dummy_loader = DataLoader([])

    retrainer = WeeklyRetrainer(
        data_loader=loader,
        train_loader=dummy_loader,
        val_loader=None,
        test_loader=None,
    )
    return retrainer.schedule_retrain()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
