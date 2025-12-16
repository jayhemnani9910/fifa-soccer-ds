"""Enhanced MLflow helper with robust configuration and error handling."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import mlflow
    from mlflow.entities import Run
    from mlflow.tracking import MlflowClient
except ImportError:
    mlflow = None
    Run = None
    MlflowClient = None

LOGGER = logging.getLogger(__name__)

DEFAULT_TRACKING_DIR = Path("mlruns")
DEFAULT_EXPERIMENT_NAME = "fifa_soccer_ds"
DEFAULT_ARTIFACT_PATH = "artifacts"


class MLflowConfig:
    """Configuration for MLflow setup."""
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
        artifact_location: Optional[str] = None,
        default_tags: Optional[Dict[str, str]] = None
    ):
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        self.experiment_name = experiment_name
        self.artifact_location = artifact_location
        self.default_tags = default_tags or {}
        
        # Auto-configure from environment
        self._configure_from_env()
    
    def _configure_from_env(self):
        """Configure MLflow from environment variables."""
        env_mappings = {
            "MLFLOW_TRACKING_URI": "tracking_uri",
            "MLFLOW_EXPERIMENT_NAME": "experiment_name", 
            "MLFLOW_ARTIFACT_LOCATION": "artifact_location"
        }
        
        for env_var, attr in env_mappings.items():
            if env_var in os.environ and not getattr(self, attr):
                setattr(self, attr, os.environ[env_var])
    
    def validate(self) -> bool:
        """Validate MLflow configuration."""
        if not mlflow:
            LOGGER.warning("MLflow not available")
            return False
        
        # Validate tracking URI
        if self.tracking_uri:
            try:
                mlflow.set_tracking_uri(self.tracking_uri)
                # Test connection
                client = mlflow.tracking.MlflowClient()
                client.list_experiments()  # Test connectivity
                return True
            except Exception as e:
                LOGGER.error("MLflow tracking URI validation failed: %s", e)
                return False
        
        return True


def ensure_local_backend(path: Path = DEFAULT_TRACKING_DIR) -> Path:
    """Ensure local MLflow tracking directory exists and is configured."""

    path.mkdir(parents=True, exist_ok=True)

    if not mlflow:
        LOGGER.info("MLflow not available; using filesystem path %s without tracking.", path)
        return path

    try:
        tracking_uri = path.resolve().as_uri()
        mlflow.set_tracking_uri(tracking_uri)
        LOGGER.info("MLflow tracking configured: %s", tracking_uri)
        return path
    except Exception as e:
        LOGGER.error("Failed to configure local MLflow backend: %s", e)
        raise


def ensure_experiment_exists(experiment_name: str, artifact_location: Optional[str] = None) -> str:
    """Ensure MLflow experiment exists, return experiment ID."""
    if not mlflow:
        raise ImportError("MLflow not available")
    
    try:
        # Get or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            LOGGER.info("Creating new MLflow experiment: %s", experiment_name)
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=artifact_location
            )
        else:
            experiment_id = experiment.experiment_id
            LOGGER.info("Using existing MLflow experiment: %s", experiment_name)
        
        return experiment_id
        
    except Exception as e:
        LOGGER.error("Failed to ensure MLflow experiment exists: %s", e)
        raise


@contextmanager
def start_run(
    experiment: str = DEFAULT_EXPERIMENT_NAME,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    description: Optional[str] = None
) -> Iterator[Run]:
    """Enhanced context manager with robust error handling."""
    if not mlflow:
        LOGGER.warning("MLflow not available, skipping logging")
        yield None
        return
    
    try:
        # Ensure experiment exists
        experiment_id = ensure_experiment_exists(experiment)
        
        # Merge with default tags
        config = MLflowConfig()
        merged_tags = {**config.default_tags, **(tags or {})}
        
        # Start run with enhanced configuration
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            tags=merged_tags,
            description=description
        ) as run:
            LOGGER.info(
                "Started MLflow run: %s in experiment %s", 
                run.info.run_id if run else "unknown",
                experiment
            )
            yield run
            
    except Exception as e:
        LOGGER.error("Failed to start MLflow run: %s", e)
        # Continue without MLflow rather than failing
        yield None


def log_run_metrics(
    metrics: Dict[str, float], 
    step: Optional[int] = None,
    prefix: Optional[str] = None
):
    """Safely log metrics with error handling."""
    if not mlflow or not metrics:
        return
    
    try:
        # Add prefix to metric names if specified
        if prefix:
            metrics = {f"{prefix}.{k}": v for k, v in metrics.items()}
        
        mlflow.log_metrics(metrics, step=step)
        LOGGER.debug("Logged %d metrics to MLflow", len(metrics))
        
    except Exception as e:
        LOGGER.warning("Failed to log metrics to MLflow: %s", e)


def log_run_params(
    params: Dict[str, Any], 
    prefix: Optional[str] = None
):
    """Safely log parameters with error handling."""
    if not mlflow or not params:
        return
    
    try:
        # Add prefix to parameter names if specified
        if prefix:
            params = {f"{prefix}.{k}": v for k, v in params.items()}
        
        mlflow.log_params(params)
        LOGGER.debug("Logged %d parameters to MLflow", len(params))
        
    except Exception as e:
        LOGGER.warning("Failed to log parameters to MLflow: %s", e)


def log_run_artifacts(
    artifact_path: str, 
    artifact_dir: Optional[str] = None
):
    """Safely log artifacts with error handling."""
    if not mlflow or not artifact_path:
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


def get_active_run() -> Optional[Run]:
    """Get currently active MLflow run safely."""
    if not mlflow:
        return None
    
    try:
        return mlflow.active_run()
    except Exception as e:
        LOGGER.warning("Failed to get active MLflow run: %s", e)
        return None


def end_run(status: str = "FINISHED"):
    """End current MLflow run with status."""
    if not mlflow:
        return
    
    try:
        active_run = get_active_run()
        if active_run:
            mlflow.end_run(status)
            LOGGER.info("Ended MLflow run: %s with status %s", 
                       active_run.info.run_id, status)
    except Exception as e:
        LOGGER.warning("Failed to end MLflow run: %s", e)


class MLflowTimer:
    """Timer for MLflow operations."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time and mlflow:
            duration = time.time() - self.start_time
            try:
                mlflow.log_metric(f"{self.name}_duration_seconds", duration)
                LOGGER.info("Timer %s: %.3fs", self.name, duration)
            except Exception as e:
                LOGGER.warning("Failed to log timer metric: %s", e)


def configure_mlflow_for_development():
    """Configure MLflow for development environment."""
    dev_config = MLflowConfig(
        tracking_uri="file:./mlruns",
        experiment_name="fifa_soccer_ds_dev",
        default_tags={
            "environment": "development",
            "version": "1.0.0"
        }
    )
    
    if dev_config.validate():
        LOGGER.info("MLflow configured for development")
    else:
        LOGGER.warning("MLflow development configuration failed")


def configure_mlflow_for_production(
    tracking_server: str,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME
):
    """Configure MLflow for production environment."""
    prod_config = MLflowConfig(
        tracking_uri=tracking_server,
        experiment_name=experiment_name,
        default_tags={
            "environment": "production",
            "version": "1.0.0"
        }
    )
    
    if prod_config.validate():
        LOGGER.info("MLflow configured for production: %s", tracking_server)
    else:
        LOGGER.error("MLflow production configuration failed")


# Auto-configure based on environment
if os.getenv("ENVIRONMENT", "development").lower() == "development":
    configure_mlflow_for_development()
elif "MLFLOW_TRACKING_URI" in os.environ:
    configure_mlflow_for_production(os.environ["MLFLOW_TRACKING_URI"])
