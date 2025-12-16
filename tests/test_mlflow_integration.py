"""Test MLflow integration and configuration."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.pipeline_full import PipelineConfig, process_frames_directory


class TestMLflowIntegration:
    """Test MLflow integration scenarios."""

    @patch('src.pipeline_full.mlflow', None)
    def test_pipeline_without_mlflow(self):
        """Test pipeline works when MLflow is not available."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir)
            output_dir = Path(tmp_dir) / "output"
            
            # Create frame files
            for i in range(3):
                (frames_dir / f"frame_{i:03d}.jpg").touch()
            
            # Should work without MLflow
            summary = process_frames_directory(
                frames_dir=frames_dir,
                output_dir=output_dir,
                config=PipelineConfig(max_frames=3),
            )
            
            assert summary.total_frames == 3

    @patch('src.pipeline_full.mlflow')
    @patch('src.pipeline_full.start_run')
    def test_mlflow_experiment_logging(self, mock_start_run, mock_mlflow):
        """Test MLflow experiment logging."""
        mock_run = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_run
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir)
            output_dir = Path(tmp_dir) / "output"
            
            # Create frame files
            for i in range(3):
                (frames_dir / f"frame_{i:03d}.jpg").touch()
            
            summary = process_frames_directory(
                frames_dir=frames_dir,
                output_dir=output_dir,
                config=PipelineConfig(
                    max_frames=3,
                    confidence=0.5,
                    distance_threshold=100.0,
                ),
            )
            
            # Verify MLflow was called
            mock_start_run.assert_called_once()
            mock_run.log_params.assert_called_once()
            mock_run.log_metrics.assert_called_once()
            mock_run.log_artifact.assert_called_once()
            
            # Check logged parameters
            logged_params = mock_run.log_params.call_args[0][0]
            assert "confidence" in logged_params
            assert "distance_threshold" in logged_params
            assert logged_params["confidence"] == 0.5
            
            # Check logged metrics
            logged_metrics = mock_run.log_metrics.call_args[0][0]
            assert "total_frames" in logged_metrics
            assert "unique_tracks" in logged_metrics
            assert logged_metrics["total_frames"] == 3

    @patch('src.pipeline_full.mlflow')
    @patch('src.pipeline_full.start_run')
    def test_mlflow_connection_failure(self, mock_start_run, mock_mlflow):
        """Test graceful handling of MLflow connection failure."""
        mock_start_run.side_effect = Exception("Connection failed")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir)
            output_dir = Path(tmp_dir) / "output"
            
            # Create frame files
            for i in range(3):
                (frames_dir / f"frame_{i:03d}.jpg").touch()
            
            # Should continue without MLflow
            summary = process_frames_directory(
                frames_dir=frames_dir,
                output_dir=output_dir,
                config=PipelineConfig(max_frames=3),
            )
            
            assert summary.total_frames == 3

    @patch('src.pipeline_full.mlflow')
    @patch('src.pipeline_full.start_run')
    def test_mlflow_partial_failure(self, mock_start_run, mock_mlflow):
        """Test handling of partial MLflow logging failures."""
        mock_run = MagicMock()
        mock_run.log_params.side_effect = Exception("Param logging failed")
        mock_start_run.return_value.__enter__.return_value = mock_run
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir)
            output_dir = Path(tmp_dir) / "output"
            
            # Create frame files
            for i in range(3):
                (frames_dir / f"frame_{i:03d}.jpg").touch()
            
            # Should continue despite MLflow failure
            summary = process_frames_directory(
                frames_dir=frames_dir,
                output_dir=output_dir,
                config=PipelineConfig(max_frames=3),
            )
            
            assert summary.total_frames == 3

    @patch('src.pipeline_full.mlflow')
    @patch('src.pipeline_full.start_run')
    def test_mlflow_artifact_logging(self, mock_start_run, mock_mlflow):
        """Test MLflow artifact logging."""
        mock_run = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_run
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir)
            output_dir = Path(tmp_dir) / "output"
            
            # Create frame files
            for i in range(3):
                (frames_dir / f"frame_{i:03d}.jpg").touch()
            
            summary = process_frames_directory(
                frames_dir=frames_dir,
                output_dir=output_dir,
                config=PipelineConfig(max_frames=3),
            )
            
            # Verify artifact logging
            mock_run.log_artifact.assert_called_once()
            artifact_path = mock_run.log_artifact.call_args[0][0]
            assert "pipeline_summary.json" in artifact_path

    @patch('src.pipeline_full.mlflow')
    def test_mlflow_experiment_naming(self, mock_mlflow):
        """Test MLflow experiment naming convention."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir) / "test_frames"
            output_dir = Path(tmp_dir) / "output"
            
            # Create frame files
            for i in range(3):
                (frames_dir / f"frame_{i:03d}.jpg").touch()
            
            process_frames_directory(
                frames_dir=frames_dir,
                output_dir=output_dir,
                config=PipelineConfig(max_frames=3),
            )
            
            # Check experiment name
            from src.pipeline_full import start_run
            start_run.assert_called_once_with(
                experiment="pipeline_full",
                run_name="test_frames"
            )

    def test_mlflow_environment_variables(self):
        """Test MLflow environment variable handling."""
        with patch.dict('os.environ', {'MLFLOW_TRACKING_URI': 'http://test-server:5000'}):
            # This test would verify environment variables are properly used
            # Implementation would check MLflow configuration
            pass

    @patch('src.pipeline_full.mlflow')
    def test_mlflow_metrics_calculation(self, mock_mlflow):
        """Test accurate MLflow metrics calculation."""
        mock_run = MagicMock()
        
        with patch('src.pipeline_full.start_run', return_value.__enter__(return_value=mock_run)):
            with tempfile.TemporaryDirectory() as tmp_dir:
                frames_dir = Path(tmp_dir)
                output_dir = Path(tmp_dir) / "output"
                
                # Create frame files
                for i in range(10):
                    (frames_dir / f"frame_{i:03d}.jpg").touch()
                
                summary = process_frames_directory(
                    frames_dir=frames_dir,
                    output_dir=output_dir,
                    config=PipelineConfig(max_frames=10),
                )
                
                # Check metrics calculation
                logged_metrics = mock_run.log_metrics.call_args[0][0]
                
                # Verify average calculations
                if summary.total_frames > 0:
                    expected_avg = summary.total_detections / summary.total_frames
                    assert logged_metrics["avg_detections_per_frame"] == expected_avg
                
                # Verify basic metrics
                assert logged_metrics["total_frames"] == 10
                assert "unique_tracks" in logged_metrics
                assert "graph_nodes" in logged_metrics
                assert "graph_edges" in logged_metrics