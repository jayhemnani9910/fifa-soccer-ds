"""Enhanced MLflow helper with robust configuration and error handling."""

from __future__ import annotations

import importlib
import logging
import os
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from mlflow.entities import Run

try:
    mlflow: Any = importlib.import_module("mlflow")
except ImportError:
    mlflow = None

LOGGER = logging.getLogger(__name__)

DEFAULT_TRACKING_DIR = Path("mlruns")
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT_NAME = "fifa_soccer_ds"
DEFAULT_ARTIFACT_PATH = "artifacts"


class MLflowConfig:
    """Configuration for MLflow setup."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        artifact_location: str | None = None,
        default_tags: dict[str, str] | None = None,
    ):
        self.tracking_uri = (
            tracking_uri
            if tracking_uri is not None
            else os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
        )
        self.experiment_name = (
            experiment_name
            if experiment_name is not None
            else os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)
        )
        self.artifact_location = (
            artifact_location
            if artifact_location is not None
            else os.getenv("MLFLOW_ARTIFACT_LOCATION")
        )
        self.default_tags = dict(default_tags or {})

    def validate(self) -> bool:
        """Validate MLflow configuration."""
        if mlflow is None:
            LOGGER.warning("MLflow not available")
            return False

        # Validate tracking URI
        if self.tracking_uri:
            try:
                mlflow.set_tracking_uri(self.tracking_uri)
                # Test connection
                client = mlflow.tracking.MlflowClient()
                client.search_experiments(max_results=1)
                return True
            except Exception as e:
                LOGGER.error("MLflow tracking URI validation failed: %s", e)
                return False

        return True


def ensure_local_backend(path: Path = DEFAULT_TRACKING_DIR) -> Path:
    """Configure a local SQLite tracking database inside *path*."""

    path.mkdir(parents=True, exist_ok=True)

    if mlflow is None:
        LOGGER.info("MLflow not available; using filesystem path %s without tracking.", path)
        return path

    try:
        tracking_uri = f"sqlite:///{(path / 'mlflow.db').resolve()}"
        mlflow.set_tracking_uri(tracking_uri)
        LOGGER.info("MLflow tracking configured: %s", tracking_uri)
        return path
    except Exception as e:
        LOGGER.error("Failed to configure local MLflow backend: %s", e)
        raise


def ensure_experiment_exists(experiment_name: str, artifact_location: str | None = None) -> str:
    """Ensure MLflow experiment exists, return experiment ID."""
    if mlflow is None:
        raise ImportError("MLflow not available")

    try:
        # Get or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None:
            LOGGER.info("Creating new MLflow experiment: %s", experiment_name)
            experiment_id = mlflow.create_experiment(
                name=experiment_name, artifact_location=artifact_location
            )
        else:
            experiment_id = experiment.experiment_id
            LOGGER.info("Using existing MLflow experiment: %s", experiment_name)

        return str(experiment_id)

    except Exception as e:
        LOGGER.error("Failed to ensure MLflow experiment exists: %s", e)
        raise


@contextmanager
def start_run(
    experiment: str = DEFAULT_EXPERIMENT_NAME,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
    description: str | None = None,
) -> Iterator[Run | None]:
    """Enhanced context manager with robust error handling."""
    if mlflow is None:
        LOGGER.warning("MLflow not available, skipping logging")
        yield None
        return

    config = MLflowConfig()
    stack = ExitStack()
    try:
        if not config.validate():
            raise RuntimeError("MLflow configuration validation failed")
        experiment_id = ensure_experiment_exists(experiment)
        merged_tags = {**config.default_tags, **(tags or {})}
        run = stack.enter_context(
            mlflow.start_run(
                experiment_id=experiment_id,
                run_name=run_name,
                tags=merged_tags,
                description=description,
            )
        )
    except Exception as e:
        stack.close()
        LOGGER.error("Failed to start MLflow run: %s", e)
        if os.getenv("MLFLOW_REQUIRED", "false").lower() == "true":
            raise
        yield None
        return

    with stack:
        LOGGER.info(
            "Started MLflow run: %s in experiment %s",
            run.info.run_id if run else "unknown",
            experiment,
        )
        yield run


def log_run_metrics(metrics: dict[str, float], step: int | None = None, prefix: str | None = None):
    """Safely log metrics with error handling."""
    if mlflow is None or not metrics:
        return

    try:
        # Add prefix to metric names if specified
        if prefix:
            metrics = {f"{prefix}.{k}": v for k, v in metrics.items()}

        mlflow.log_metrics(metrics, step=step)
        LOGGER.debug("Logged %d metrics to MLflow", len(metrics))

    except Exception as e:
        LOGGER.warning("Failed to log metrics to MLflow: %s", e)


def log_run_params(params: dict[str, Any], prefix: str | None = None):
    """Safely log parameters with error handling."""
    if mlflow is None or not params:
        return

    try:
        # Add prefix to parameter names if specified
        if prefix:
            params = {f"{prefix}.{k}": v for k, v in params.items()}

        mlflow.log_params(params)
        LOGGER.debug("Logged %d parameters to MLflow", len(params))

    except Exception as e:
        LOGGER.warning("Failed to log parameters to MLflow: %s", e)


def log_run_artifacts(artifact_path: str, artifact_dir: str | None = None):
    """Safely log artifacts with error handling."""
    if mlflow is None or not artifact_path:
        return

    try:
        path = Path(artifact_path)
        if not path.exists():
            LOGGER.warning("Artifact path does not exist: %s", artifact_path)
            return

        # Log directory or file
        if path.is_dir():
            mlflow.log_artifacts(str(path), artifact_dir)
        else:
            mlflow.log_artifact(str(path), artifact_dir)

        LOGGER.info("Logged artifacts from: %s", artifact_path)

    except Exception as e:
        LOGGER.warning("Failed to log artifacts to MLflow: %s", e)


def get_active_run() -> Run | None:
    """Get currently active MLflow run safely."""
    if mlflow is None:
        return None

    try:
        return mlflow.active_run()
    except Exception as e:
        LOGGER.warning("Failed to get active MLflow run: %s", e)
        return None


def end_run(status: str = "FINISHED"):
    """End current MLflow run with status."""
    if mlflow is None:
        return

    try:
        active_run = get_active_run()
        if active_run:
            mlflow.end_run(status)
            LOGGER.info("Ended MLflow run: %s with status %s", active_run.info.run_id, status)
    except Exception as e:
        LOGGER.warning("Failed to end MLflow run: %s", e)


class MLflowTimer:
    """Timer for MLflow operations."""

    def __init__(self, name: str):
        self.name = name
        self.start_time: float | None = None

    def __enter__(self) -> Self:
        self.start_time = time.time()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self.start_time is not None and mlflow is not None:
            duration = time.time() - self.start_time
            try:
                mlflow.log_metric(f"{self.name}_duration_seconds", duration)
                LOGGER.info("Timer %s: %.3fs", self.name, duration)
            except Exception as e:
                LOGGER.warning("Failed to log timer metric: %s", e)


def configure_mlflow_for_development() -> bool:
    """Configure MLflow for development environment."""
    dev_config = MLflowConfig(
        tracking_uri=DEFAULT_TRACKING_URI,
        experiment_name="fifa_soccer_ds_dev",
        default_tags={"environment": "development", "version": "0.1.0"},
    )

    if dev_config.validate():
        LOGGER.info("MLflow configured for development")
        return True
    else:
        LOGGER.warning("MLflow development configuration failed")
        return False


def configure_mlflow_for_production(
    tracking_server: str, experiment_name: str = DEFAULT_EXPERIMENT_NAME
) -> bool:
    """Configure MLflow for production environment."""
    prod_config = MLflowConfig(
        tracking_uri=tracking_server,
        experiment_name=experiment_name,
        default_tags={"environment": "production", "version": "0.1.0"},
    )

    if prod_config.validate():
        LOGGER.info("MLflow configured for production: %s", tracking_server)
        return True
    else:
        LOGGER.error("MLflow production configuration failed")
        return False
