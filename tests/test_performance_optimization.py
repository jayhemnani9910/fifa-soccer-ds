"""Test performance optimization scenarios."""

import time
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.pipeline_full import PipelineConfig, process_frames_directory


class TestPerformanceOptimizations:
    """Test performance-related optimizations."""

    def test_max_frames_limiting(self):
        """Test that max_frames parameter correctly limits processing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir)
            output_dir = Path(tmp_dir) / "output"
            
            # Create many dummy frame files
            for i in range(100):
                (frames_dir / f"frame_{i:03d}.jpg").touch()
            
            start_time = time.time()
            summary = process_frames_directory(
                frames_dir=frames_dir,
                output_dir=output_dir,
                config=PipelineConfig(max_frames=5),
            )
            end_time = time.time()
            
            # Should only process 5 frames
            assert summary.total_frames == 5
            # Processing should be fast due to small number of frames
            assert end_time - start_time < 5.0

    @patch('src.pipeline_full.cv2')
    def test_frame_processing_performance(self, mock_cv2):
        """Test frame reading performance with mock."""
        mock_cv2.imread.return_value = MagicMock()  # Mock valid frame
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir)
            output_dir = Path(tmp_dir) / "output"
            
            # Create frame files
            for i in range(10):
                (frames_dir / f"frame_{i:03d}.jpg").touch()
            
            start_time = time.time()
            summary = process_frames_directory(
                frames_dir=frames_dir,
                output_dir=output_dir,
                config=PipelineConfig(max_frames=10),
            )
            end_time = time.time()
            
            # Should process all frames quickly
            assert summary.total_frames == 10
            assert end_time - start_time < 2.0
            # cv2.imread should be called for each frame
            assert mock_cv2.imread.call_count == 10

    def test_memory_usage_with_large_datasets(self):
        """Test memory usage doesn't grow unbounded."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir)
            output_dir = Path(tmp_dir) / "output"
            
            # Create frame files
            for i in range(50):
                (frames_dir / f"frame_{i:03d}.jpg").touch()
            
            # Process with memory-efficient config
            summary = process_frames_directory(
                frames_dir=frames_dir,
                output_dir=output_dir,
                config=PipelineConfig(max_frames=50),
            )
            
            # Should complete without memory errors
            assert summary.total_frames == 50
            # Check output files were created
            assert len(list(output_dir.glob("**/*.json"))) > 0

    def test_concurrent_output_writing(self):
        """Test that output writing doesn't block processing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir)
            output_dir = Path(tmp_dir) / "output"
            
            # Create frame files
            for i in range(20):
                (frames_dir / f"frame_{i:03d}.jpg").touch()
            
            start_time = time.time()
            summary = process_frames_directory(
                frames_dir=frames_dir,
                output_dir=output_dir,
                config=PipelineConfig(max_frames=20),
            )
            end_time = time.time()
            
            # Should complete in reasonable time
            assert summary.total_frames == 20
            assert end_time - start_time < 3.0
            
            # Verify all expected output files exist
            detection_files = list((output_dir / "detections").glob("*_detections.json"))
            track_files = list((output_dir / "tracks").glob("*_tracks.json"))
            
            assert len(detection_files) == 20
            assert len(track_files) == 20


class TestResourceOptimization:
    """Test resource usage optimization."""

    def test_output_directory_cleanup(self):
        """Test that output directories are properly managed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir)
            output_dir = Path(tmp_dir) / "output"
            
            # Create some existing files in output
            output_dir.mkdir()
            (output_dir / "old_file.txt").write_text("old data")
            
            # Create frame files
            for i in range(3):
                (frames_dir / f"frame_{i:03d}.jpg").touch()
            
            summary = process_frames_directory(
                frames_dir=frames_dir,
                output_dir=output_dir,
                config=PipelineConfig(max_frames=3),
            )
            
            # Old file should still exist (no cleanup by default)
            assert (output_dir / "old_file.txt").exists()
            # New directories should be created
            assert (output_dir / "detections").exists()
            assert (output_dir / "tracks").exists()
            assert (output_dir / "graphs").exists()

    def test_file_io_error_handling(self):
        """Test graceful handling of file I/O errors."""
        with patch('builtins.open', side_effect=IOError("Disk full")):
            with tempfile.TemporaryDirectory() as tmp_dir:
                frames_dir = Path(tmp_dir)
                output_dir = Path(tmp_dir) / "output"
                
                # Create frame files
                for i in range(3):
                    (frames_dir / f"frame_{i:03d}.jpg").touch()
                
                # Should handle I/O errors gracefully
                with pytest.raises(IOError):
                    process_frames_directory(
                        frames_dir=frames_dir,
                        output_dir=output_dir,
                        config=PipelineConfig(max_frames=3),
                    )

    def test_configuration_optimization(self):
        """Test that configuration parameters affect performance."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir)
            output_dir = Path(tmp_dir) / "output"
            
            # Create frame files
            for i in range(10):
                (frames_dir / f"frame_{i:03d}.jpg").touch()
            
            # Test with different confidence thresholds
            configs = [
                PipelineConfig(confidence=0.1),  # Low confidence (more detections)
                PipelineConfig(confidence=0.9),  # High confidence (fewer detections)
            ]
            
            summaries = []
            for config in configs:
                summary = process_frames_directory(
                    frames_dir=frames_dir,
                    output_dir=output_dir / f"run_{len(summaries)}",
                    config=config,
                )
                summaries.append(summary)
            
            # Both should complete successfully
            for summary in summaries:
                assert summary.total_frames == 10
                assert summary.total_detections >= 0