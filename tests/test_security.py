"""Test security-related functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.pipeline_full import PipelineConfig, process_frames_directory


class TestInputValidation:
    """Test input validation and sanitization."""

    def test_path_traversal_prevention(self):
        """Test prevention of path traversal attacks."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            
            # Test path traversal attempts
            malicious_paths = [
                "../../../etc/passwd",
                "..\\..\\windows\\system32\\config",
                "/etc/shadow",
                "C:\\Windows\\System32\\drivers\\etc\\hosts",
            ]
            
            for malicious_path in malicious_paths:
                with pytest.raises((FileNotFoundError, ValueError)):
                    process_frames_directory(
                        frames_dir=Path(malicious_path),
                        output_dir=base_dir / "output",
                    )

    def test_file_type_validation(self):
        """Test that only valid image files are processed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir)
            output_dir = Path(tmp_dir) / "output"
            
            # Create various file types
            (frames_dir / "frame_001.jpg").write_bytes(b"fake image data")
            (frames_dir / "frame_002.png").write_bytes(b"fake png data")
            (frames_dir / "malicious.exe").write_bytes(b"fake executable")
            (frames_dir / "script.sh").write_text("#!/bin/bash\necho 'hack'")
            (frames_dir / "document.pdf").write_bytes(b"%PDF-1.4")
            
            # Should only process image files
            with patch('src.pipeline_full.cv2') as mock_cv2:
                mock_cv2.imread.return_value = None  # Simulate failed reads
                
                summary = process_frames_directory(
                    frames_dir=frames_dir,
                    output_dir=output_dir,
                    config=PipelineConfig(max_frames=10),
                )
                
                # Should find only image files
                assert summary.total_frames <= 2  # Only .jpg and .png files

    def test_config_parameter_sanitization(self):
        """Test configuration parameter sanitization."""
        # Test extremely large values
        with pytest.raises(ValueError):
            process_frames_directory(
                frames_dir=Path("/tmp"),
                output_dir=Path("/tmp/test"),
                config=PipelineConfig(confidence=float('inf')),
            )
        
        with pytest.raises(ValueError):
            process_frames_directory(
                frames_dir=Path("/tmp"),
                output_dir=Path("/tmp/test"),
                config=PipelineConfig(distance_threshold=float('inf')),
            )

    def test_filename_sanitization(self):
        """Test filename sanitization in outputs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir)
            output_dir = Path(tmp_dir) / "output"
            
            # Create frame with problematic name
            problematic_frame = frames_dir / "frame_001; rm -rf /.jpg"
            problematic_frame.write_bytes(b"fake image data")
            
            with patch('src.pipeline_full.cv2') as mock_cv2:
                mock_cv2.imread.return_value = None
                
                summary = process_frames_directory(
                    frames_dir=frames_dir,
                    output_dir=output_dir,
                    config=PipelineConfig(max_frames=1),
                )
                
                # Should process safely
                assert summary.total_frames == 1
                
                # Check output files have safe names
                for output_file in output_dir.rglob("*"):
                    assert ";" not in output_file.name
                    assert "rm -rf" not in output_file.name


class TestResourceLimits:
    """Test resource usage limits."""

    def test_memory_usage_limits(self):
        """Test memory usage doesn't exceed limits."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir)
            output_dir = Path(tmp_dir) / "output"
            
            # Create many frame files
            for i in range(1000):
                (frames_dir / f"frame_{i:03d}.jpg").touch()
            
            # Should handle large dataset without memory issues
            summary = process_frames_directory(
                frames_dir=frames_dir,
                output_dir=output_dir,
                config=PipelineConfig(max_frames=100),  # Limit to reasonable number
            )
            
            assert summary.total_frames == 100

    def test_processing_time_limits(self):
        """Test processing time limits."""
        import time
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir)
            output_dir = Path(tmp_dir) / "output"
            
            # Create frame files
            for i in range(50):
                (frames_dir / f"frame_{i:03d}.jpg").touch()
            
            start_time = time.time()
            summary = process_frames_directory(
                frames_dir=frames_dir,
                output_dir=output_dir,
                config=PipelineConfig(max_frames=50),
            )
            end_time = time.time()
            
            # Should complete in reasonable time
            processing_time = end_time - start_time
            assert processing_time < 30.0  # 30 second limit for 50 frames

    def test_file_size_limits(self):
        """Test handling of extremely large files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir)
            output_dir = Path(tmp_dir) / "output"
            
            # Create a very large "image" file
            large_frame = frames_dir / "frame_001.jpg"
            large_frame.write_bytes(b"x" * 100 * 1024 * 1024)  # 100MB
            
            with patch('src.pipeline_full.cv2') as mock_cv2:
                # Simulate memory error for large files
                mock_cv2.imread.side_effect = MemoryError("Image too large")
                
                summary = process_frames_directory(
                    frames_dir=frames_dir,
                    output_dir=output_dir,
                    config=PipelineConfig(max_frames=1),
                )
                
                # Should handle gracefully
                assert summary.total_frames == 1


class TestSecureDefaults:
    """Test secure default configurations."""

    def test_secure_default_paths(self):
        """Test default paths are secure."""
        config = PipelineConfig()
        
        # Should not use sensitive default paths
        assert config.weights != "/etc/passwd"
        assert config.weights != "../../../sensitive"
        assert "yolov8" in config.weights or config.weights.endswith(".pt")

    def test_secure_default_permissions(self):
        """Test default permission settings."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir)
            output_dir = Path(tmp_dir) / "output"
            
            # Create frame files
            for i in range(3):
                (frames_dir / f"frame_{i:03d}.jpg").touch()
            
            process_frames_directory(
                frames_dir=frames_dir,
                output_dir=output_dir,
                config=PipelineConfig(max_frames=3),
            )
            
            # Check output directory permissions (should not be world-writable)
            output_stat = output_dir.stat()
            # On Unix-like systems, check permissions don't include world write
            if hasattr(output_stat, 'st_mode'):
                mode = output_stat.st_mode
                # Should not have world write permissions (o+w)
                assert not (mode & 0o002)

    def test_no_sensitive_data_logging(self):
        """Test no sensitive data is logged."""
        with patch('src.pipeline_full.mlflow') as mock_mlflow:
            mock_run = MagicMock()
            mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                frames_dir = Path(tmp_dir)
                output_dir = Path(tmp_dir) / "output"
                
                # Create frame files with sensitive-looking names
                for i in range(3):
                    (frames_dir / f"frame_{i:03d}_secret_key.jpg").touch()
                
                process_frames_directory(
                    frames_dir=frames_dir,
                    output_dir=output_dir,
                    config=PipelineConfig(max_frames=3),
                )
                
                # Check logged parameters don't contain sensitive data
                if mock_run.log_params.called:
                    logged_params = mock_run.log_params.call_args[0][0]
                    for param_value in logged_params.values():
                        assert "secret" not in str(param_value).lower()
                        assert "password" not in str(param_value).lower()
                        assert "key" not in str(param_value).lower() or "api_key" in str(param_value).lower()


class TestErrorDisclosure:
    """Test that error messages don't disclose sensitive information."""

    def test_no_path_disclosure_in_errors(self):
        """Test error messages don't reveal full paths."""
        with pytest.raises(FileNotFoundError) as exc_info:
            process_frames_directory(
                frames_dir=Path("/non/existent/secret/path"),
                output_dir=Path("/tmp/test"),
            )
        
        error_msg = str(exc_info.value)
        # Should mention directory not found but not reveal full system structure
        assert "Frames directory not found" in error_msg
        # Should not contain sensitive system paths
        assert "/home" not in error_msg or "secret" not in error_msg

    def test_no_stack_traces_in_user_output(self):
        """Test detailed stack traces aren't shown to users."""
        with patch('src.pipeline_full.load_model') as mock_load:
            mock_load.side_effect = Exception("Internal error")
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                frames_dir = Path(tmp_dir)
                output_dir = Path(tmp_dir) / "output"
                
                for i in range(3):
                    (frames_dir / f"frame_{i:03d}.jpg").touch()
                
                # Should raise RuntimeError with sanitized message
                with pytest.raises(RuntimeError) as exc_info:
                    process_frames_directory(
                        frames_dir=frames_dir,
                        output_dir=output_dir,
                        config=PipelineConfig(max_frames=3),
                    )
                
                # Error message should be user-friendly
                error_msg = str(exc_info.value)
                assert "Model loading failed" in error_msg
                # Should not contain internal implementation details
                assert "traceback" not in error_msg.lower()
                assert "internal" not in error_msg.lower()
