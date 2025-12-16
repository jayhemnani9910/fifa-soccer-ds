"""Test error handling and edge cases in pipeline components."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline_full import PipelineConfig, process_frames_directory


class TestPipelineErrorHandling:
    """Test suite for pipeline error handling."""

    def test_invalid_frames_directory(self):
        """Test handling of non-existent frames directory."""
        with pytest.raises(FileNotFoundError, match="Frames directory not found"):
            process_frames_directory(
                frames_dir=Path("/non/existent/path"),
                output_dir=Path("/tmp/test"),
            )

    def test_frames_directory_not_a_directory(self):
        """Test handling when frames path is not a directory."""
        with tempfile.NamedTemporaryFile() as tmp_file:
            with pytest.raises(ValueError, match="Frames path is not a directory"):
                process_frames_directory(
                    frames_dir=Path(tmp_file.name),
                    output_dir=Path("/tmp/test"),
                )

    def test_invalid_confidence_values(self):
        """Test validation of confidence thresholds."""
        # Test confidence > 1
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            process_frames_directory(
                frames_dir=Path("/tmp"),
                output_dir=Path("/tmp/test"),
                config=PipelineConfig(confidence=1.5),
            )

        # Test confidence < 0
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            process_frames_directory(
                frames_dir=Path("/tmp"),
                output_dir=Path("/tmp/test"),
                config=PipelineConfig(confidence=-0.1),
            )

        # Test min_confidence > 1
        with pytest.raises(ValueError, match="Min confidence must be between 0 and 1"):
            process_frames_directory(
                frames_dir=Path("/tmp"),
                output_dir=Path("/tmp/test"),
                config=PipelineConfig(min_confidence=1.2),
            )

    def test_empty_frames_directory(self):
        """Test handling of directory with no frame files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir)
            output_dir = Path(tmp_dir) / "output"
            
            with pytest.raises(ValueError, match="No frame files found"):
                process_frames_directory(
                    frames_dir=frames_dir,
                    output_dir=output_dir,
                )

    @patch('src.pipeline_full.cv2')
    def test_opencv_not_available(self, mock_cv2):
        """Test handling when OpenCV is not available."""
        mock_cv2.imread.return_value = None
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir)
            output_dir = Path(tmp_dir) / "output"
            
            # Create a dummy frame file
            (frames_dir / "frame_001.jpg").touch()
            
            # Should not raise error, but should log warning
            summary = process_frames_directory(
                frames_dir=frames_dir,
                output_dir=output_dir,
                config=PipelineConfig(max_frames=1),
            )
            
            assert summary.total_frames == 1
            assert summary.total_detections == 0

    @patch('src.pipeline_full.load_model')
    def test_model_loading_failure(self, mock_load_model):
        """Test handling of model loading failure."""
        mock_load_model.side_effect = Exception("Model loading failed")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir)
            output_dir = Path(tmp_dir) / "output"
            
            # Create dummy frame files
            for i in range(3):
                (frames_dir / f"frame_{i:03d}.jpg").touch()
            
            with pytest.raises(RuntimeError, match="Model loading failed"):
                process_frames_directory(
                    frames_dir=frames_dir,
                    output_dir=output_dir,
                    config=PipelineConfig(max_frames=3),
                )

    @patch('src.pipeline_full.ByteTrackRuntime')
    def test_tracker_initialization_failure(self, mock_tracker):
        """Test handling of tracker initialization failure."""
        mock_tracker.side_effect = Exception("Tracker init failed")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir)
            output_dir = Path(tmp_dir) / "output"
            
            # Create dummy frame files
            for i in range(3):
                (frames_dir / f"frame_{i:03d}.jpg").touch()
            
            with pytest.raises(RuntimeError, match="Tracker initialization failed"):
                process_frames_directory(
                    frames_dir=frames_dir,
                    output_dir=output_dir,
                    config=PipelineConfig(max_frames=3),
                )

    def test_output_directory_creation_failure(self):
        """Test handling of output directory creation failure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir)
            output_dir = Path("/root/forbidden/output")  # Likely inaccessible
            
            # Create dummy frame files
            for i in range(3):
                (frames_dir / f"frame_{i:03d}.jpg").touch()
            
            with pytest.raises(OSError):
                process_frames_directory(
                    frames_dir=frames_dir,
                    output_dir=output_dir,
                    config=PipelineConfig(max_frames=3),
                )

    @patch('src.pipeline_full.mlflow')
    def test_mlflow_logging_failure(self, mock_mlflow):
        """Test graceful handling of MLflow logging failure."""
        mock_mlflow.start_run.side_effect = Exception("MLflow connection failed")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir)
            output_dir = Path(tmp_dir) / "output"
            
            # Create dummy frame files
            for i in range(3):
                (frames_dir / f"frame_{i:03d}.jpg").touch()
            
            # Should not raise error, but should continue without MLflow
            summary = process_frames_directory(
                frames_dir=frames_dir,
                output_dir=output_dir,
                config=PipelineConfig(max_frames=3),
            )
            
            assert summary.total_frames == 3

    def test_pipeline_summary_serialization(self):
        """Test that pipeline summary can be properly serialized."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir)
            output_dir = Path(tmp_dir) / "output"
            
            # Create dummy frame files
            for i in range(3):
                (frames_dir / f"frame_{i:03d}.jpg").touch()
            
            summary = process_frames_directory(
                frames_dir=frames_dir,
                output_dir=output_dir,
                config=PipelineConfig(max_frames=3),
            )
            
            # Verify summary file exists and is valid JSON
            summary_file = output_dir / "pipeline_summary.json"
            assert summary_file.exists()
            
            with summary_file.open('r') as f:
                data = json.load(f)
            
            # Verify required fields
            required_fields = [
                "total_frames", "total_detections", "unique_track_ids",
                "num_unique_tracks", "graph_nodes", "graph_edges", "config"
            ]
            for field in required_fields:
                assert field in data, f"Missing field: {field}"


class TestConfigurationValidation:
    """Test configuration parameter validation."""

    def test_default_configuration(self):
        """Test that default configuration is valid."""
        config = PipelineConfig()
        
        assert 0 <= config.confidence <= 1
        assert 0 <= config.min_confidence <= 1
        assert config.distance_threshold > 0
        assert config.max_age >= 0
        assert config.graph_window > 0
        assert config.graph_distance_threshold > 0

    def test_edge_case_configurations(self):
        """Test edge case configurations."""
        # Test zero values
        config = PipelineConfig(
            confidence=0.0,
            min_confidence=0.0,
            max_frames=0,
            max_age=0,
        )
        assert config.confidence == 0.0
        assert config.min_confidence == 0.0
        assert config.max_frames == 0
        assert config.max_age == 0
        
        # Test maximum reasonable values
        config = PipelineConfig(
            confidence=1.0,
            min_confidence=1.0,
            distance_threshold=1000.0,
            max_age=1000,
            graph_window=1000,
        )
        assert config.confidence == 1.0
        assert config.min_confidence == 1.0
        assert config.distance_threshold == 1000.0
        assert config.max_age == 1000
        assert config.graph_window == 1000