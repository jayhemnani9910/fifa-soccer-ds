"""Tests for fail-open MLflow telemetry and pipeline integration."""

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from src.pipeline_full import PipelineConfig, process_frames_directory
from src.utils import mlflow_helper


def _write_frames(frames_dir: Path, count: int) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    for index in range(count):
        assert cv2.imwrite(str(frames_dir / f"frame_{index:03d}.jpg"), image)


def _process_frames(frames_dir: Path, output_dir: Path, count: int = 3):
    _write_frames(frames_dir, count)
    detector = Mock(predict=Mock(return_value=[]))
    with patch("src.pipeline_full.load_model", return_value=detector):
        return process_frames_directory(
            frames_dir=frames_dir,
            output_dir=output_dir,
            config=PipelineConfig(max_frames=count, enable_tactical_analytics=False),
        )


def test_pipeline_without_mlflow(tmp_path: Path) -> None:
    with patch("src.pipeline_full.mlflow", None):
        summary = _process_frames(tmp_path / "frames", tmp_path / "output")

    assert summary.total_frames == 3


def test_pipeline_logs_to_mlflow_api(tmp_path: Path) -> None:
    mlflow_api = Mock()
    active_run = Mock()

    with (
        patch("src.pipeline_full.mlflow", mlflow_api),
        patch("src.pipeline_full.start_run", return_value=nullcontext(active_run)) as start_run,
    ):
        summary = _process_frames(tmp_path / "frames", tmp_path / "output")

    start_run.assert_called_once_with(experiment="pipeline_full", run_name="frames")
    logged_params = mlflow_api.log_params.call_args.args[0]
    logged_metrics = mlflow_api.log_metrics.call_args.args[0]
    assert logged_params["confidence"] == 0.25
    assert logged_metrics["total_frames"] == summary.total_frames == 3
    assert logged_metrics["avg_detections_per_frame"] == 0
    assert Path(mlflow_api.log_artifact.call_args.args[0]).name == "pipeline_summary.json"


def test_pipeline_skips_logging_when_run_is_unavailable(tmp_path: Path) -> None:
    mlflow_api = Mock()
    with (
        patch("src.pipeline_full.mlflow", mlflow_api),
        patch("src.pipeline_full.start_run", return_value=nullcontext(None)),
    ):
        summary = _process_frames(tmp_path / "frames", tmp_path / "output")

    assert summary.total_frames == 3
    mlflow_api.log_params.assert_not_called()
    mlflow_api.log_metrics.assert_not_called()
    mlflow_api.log_artifact.assert_not_called()


def test_pipeline_tolerates_mlflow_logging_failure(tmp_path: Path) -> None:
    mlflow_api = Mock()
    mlflow_api.log_params.side_effect = ConnectionError("tracking server unavailable")
    with (
        patch("src.pipeline_full.mlflow", mlflow_api),
        patch("src.pipeline_full.start_run", return_value=nullcontext(Mock())),
    ):
        summary = _process_frames(tmp_path / "frames", tmp_path / "output")

    assert summary.total_frames == 3
    mlflow_api.log_metrics.assert_not_called()


def test_config_reads_environment_without_overriding_explicit_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://tracking.example.test")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "from-environment")
    monkeypatch.setenv("MLFLOW_ARTIFACT_LOCATION", "s3://example/artifacts")

    from_environment = mlflow_helper.MLflowConfig()
    explicit = mlflow_helper.MLflowConfig(
        tracking_uri="sqlite:///explicit.db",
        experiment_name="explicit",
        artifact_location="file:///explicit",
    )

    assert from_environment.tracking_uri == "https://tracking.example.test"
    assert from_environment.experiment_name == "from-environment"
    assert from_environment.artifact_location == "s3://example/artifacts"
    assert explicit.tracking_uri == "sqlite:///explicit.db"
    assert explicit.experiment_name == "explicit"
    assert explicit.artifact_location == "file:///explicit"


def test_config_validate_uses_supported_search_api() -> None:
    mlflow_api = Mock()
    client = mlflow_api.tracking.MlflowClient.return_value
    with patch.object(mlflow_helper, "mlflow", mlflow_api):
        assert mlflow_helper.MLflowConfig("sqlite:///test.db").validate()

    client.search_experiments.assert_called_once_with(max_results=1)


def test_start_run_is_fail_open_when_setup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_REQUIRED", raising=False)
    with (
        patch.object(mlflow_helper, "mlflow", Mock()),
        patch.object(mlflow_helper.MLflowConfig, "validate", return_value=False),
        mlflow_helper.start_run() as active_run,
    ):
        assert active_run is None


def test_start_run_can_require_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_REQUIRED", "true")
    with (
        patch.object(mlflow_helper, "mlflow", Mock()),
        patch.object(mlflow_helper.MLflowConfig, "validate", return_value=False),
        pytest.raises(RuntimeError, match="validation failed"),
        mlflow_helper.start_run(),
    ):
        pytest.fail("unreachable")


def test_start_run_does_not_swallow_body_exceptions() -> None:
    mlflow_api = Mock()
    active_run = Mock()
    mlflow_api.start_run.return_value = nullcontext(active_run)

    with (
        patch.object(mlflow_helper, "mlflow", mlflow_api),
        patch.object(mlflow_helper.MLflowConfig, "validate", return_value=True),
        patch.object(mlflow_helper, "ensure_experiment_exists", return_value="1"),
        pytest.raises(ValueError, match="application failure"),
        mlflow_helper.start_run(),
    ):
        raise ValueError("application failure")
