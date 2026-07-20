"""End-to-end integration tests for FIFA Soccer DS pipeline.

This module tests the complete pipeline flow from YouTube URL input
to final analysis output, ensuring all components work together.
"""

import asyncio
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.classify.soccer_classifier import SoccerClassifier
from src.pipeline_orchestrator import PipelineOrchestrator, WeightsNotConfiguredError
from src.schemas import PipelineOutput, PlayerAnalysis, SoccerClassification, YouTubeAnalysisRequest
from src.youtube.metadata_parser import YouTubeMetadataParser


class TestEndToEndPipeline:
    """Test suite for end-to-end pipeline functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_video_info(self):
        """Mock video information for testing."""
        return {
            "id": "test123abcd",
            "title": "Barcelona vs Real Madrid - El Clasico Highlights",
            "description": "Amazing goals and skills from both teams",
            "duration": 300,
            "view_count": 1000000,
            "like_count": 50000,
            "comment_count": 5000,
            "uploader": "ESPN FC",
            "channel_id": "espnfc123",
            "upload_date": "20231201",
            "tags": ["soccer", "football", "barcelona", "real madrid", "el clasico"],
            "categories": ["Sports"],
            "resolution": "1920x1080",
            "fps": 30.0,
            "vcodec": "h264",
            "acodec": "aac",
        }

    @pytest.fixture
    def test_request(self, temp_dir, monkeypatch):
        """Create test YouTube analysis request."""
        monkeypatch.setenv("ANALYSIS_OUTPUT_ROOT", str(temp_dir))
        weights = temp_dir / "yolov8n.pt"
        weights.write_bytes(b"dummy-checkpoint-bytes")
        monkeypatch.setenv("YOLO_WEIGHTS", str(weights))
        return YouTubeAnalysisRequest(
            url="https://www.youtube.com/watch?v=test123abcd",
            output_dir="e2e",
            frame_rate=1.0,
            max_duration=300,
            include_audio=True,
            confidence_threshold=0.75,
        )

    def test_pipeline_orchestrator_creation(self):
        """Test that pipeline orchestrator can be created."""
        orchestrator = PipelineOrchestrator()
        assert orchestrator is not None
        assert hasattr(orchestrator, "process_youtube_video")
        assert hasattr(orchestrator, "config")

    def test_pipeline_orchestrator_with_mock_data(self, test_request, mock_video_info):
        """Test the async boundary and evidence-to-schema conversion."""

        def fake_processor(_url, output_dir, *_args, **_kwargs):
            (output_dir / "youtube_analysis.json").write_text("{}", encoding="utf-8")
            return {
                "type": "soccer",
                "classification": {
                    "is_soccer": True,
                    "confidence": 0.88,
                    "analysis_breakdown": {"metadata_analysis": {"content_type": "highlights"}},
                    "processing_info": {"video_info": mock_video_info},
                },
                "pipeline_summary": {
                    "total_frames": 3,
                    "attempted_frames": 4,
                    "successful_frames": 3,
                    "processing_success_rate": 0.75,
                    "failures": {
                        "unreadable_frames": 1,
                        "detection_failures": 0,
                        "tracking_failures": 0,
                        "overlay_failures": 0,
                    },
                    "total_detections": 6,
                    "unique_track_ids": [1, 2],
                    "graph_nodes": 6,
                    "graph_edges": 4,
                },
            }

        orchestrator = PipelineOrchestrator(processor=fake_processor)

        async def inline_to_thread(function, *args, **kwargs):
            return function(*args, **kwargs)

        with patch("src.pipeline_orchestrator.asyncio.to_thread", side_effect=inline_to_thread):
            result = asyncio.run(orchestrator.process_youtube_video(test_request))

        assert isinstance(result, PipelineOutput)
        assert result.input_source == "youtube"
        assert result.soccer_classification.is_soccer
        assert result.soccer_classification.soccer_confidence > 0.8
        assert result.soccer_classification.total_events is None
        assert result.soccer_classification.detection_quality is None
        assert result.soccer_classification.processing_success_rate == 0.75
        assert any("partial frame failures" in warning for warning in result.warnings)
        assert result.player_analysis.total_players_detected == 2
        assert result.player_analysis.avg_track_length is None
        assert result.output_files["analysis"].endswith("youtube_analysis.json")

    def test_process_youtube_video_requires_configured_weights(self, test_request, monkeypatch):
        """Regression test: unset YOLO_WEIGHTS must not silently fall back to a bundled default."""
        monkeypatch.delenv("YOLO_WEIGHTS", raising=False)

        def _unreachable_processor(*_args, **_kwargs):
            raise AssertionError("processor must not run when weights are unconfigured")

        orchestrator = PipelineOrchestrator(processor=_unreachable_processor)

        with pytest.raises(WeightsNotConfiguredError, match="YOLO_WEIGHTS"):
            asyncio.run(orchestrator.process_youtube_video(test_request))

    def test_youtube_analysis_request_validation(self):
        """Test YouTube analysis request schema validation."""
        # Valid request
        valid_request = YouTubeAnalysisRequest(
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            output_dir="test_output",
            frame_rate=1.0,
        )
        assert valid_request.url is not None
        assert valid_request.frame_rate == 1.0

        # Invalid URL should raise validation error
        with pytest.raises(Exception):
            YouTubeAnalysisRequest(url="invalid-url", output_dir="test_output")

    def test_metadata_validation(self, mock_video_info):
        """Test metadata validation with schema."""
        parser = YouTubeMetadataParser()

        # Test with valid data
        metadata = parser._create_validated_metadata(mock_video_info)
        assert metadata.video_id == "test123abcd"
        assert metadata.title == "Barcelona vs Real Madrid - El Clasico Highlights"
        assert metadata.duration_seconds == 300
        assert metadata.fps == 30.0

        # Test with missing data (unavailable values remain explicit)
        incomplete_data = {"id": "test456abcd", "title": "Test Video"}
        metadata = parser._create_validated_metadata(incomplete_data)
        assert metadata.video_id == "test456abcd"
        assert metadata.title == "Test Video"
        assert metadata.duration_seconds is None

    def test_soccer_classifier_validation(self):
        """Test soccer classifier input validation."""
        classifier = SoccerClassifier()

        # Valid URL
        with patch.object(
            classifier,
            "_analyze_thumbnail",
            return_value={"visual_score": 0.8, "confidence": 0.8},
        ):
            mock_instance = Mock()
            mock_instance.extract_metadata.return_value = {"title": "Soccer Goals"}
            mock_instance.predict_soccer_content.return_value = {
                "is_soccer": True,
                "confidence": 0.9,
                "relevance_level": "high",
            }
            classifier.metadata_parser = mock_instance

            result = classifier.classify_youtube_content(
                "https://youtube.com/watch?v=test123abcd", include_audio=False
            )
            assert result["is_soccer"]
            assert result["confidence"] > 0.8

        # Invalid URL should raise error
        with pytest.raises(ValueError):
            classifier.classify_youtube_content("invalid-url")

    def test_pipeline_output_schema_roundtrip(self, temp_dir, mock_video_info):
        """Test that a pipeline response survives a JSON schema round trip."""
        output = PipelineOutput(
            pipeline_version="1.0.0",
            processing_timestamp=mock_video_info.get("upload_date"),
            processing_duration_seconds=120.0,
            input_source="youtube",
            input_url="https://youtube.com/watch?v=test123abcd",
            input_metadata=YouTubeMetadataParser()._create_validated_metadata(mock_video_info),
            soccer_classification=SoccerClassification(
                is_soccer=True,
                soccer_confidence=0.85,
                content_type="highlight",
                total_events=None,
                detection_quality=None,
                processing_success_rate=None,
            ),
            player_analysis=PlayerAnalysis(
                total_players_detected=0, teams_detected=0, avg_track_length=None
            ),
            output_files={
                "pipeline_summary": str(temp_dir / "pipeline_summary.json"),
                "graph": str(temp_dir / "graphs" / "final_graph.json"),
            },
        )

        restored = PipelineOutput.model_validate_json(output.model_dump_json())

        assert restored == output
        assert restored.soccer_classification.total_events is None
        assert restored.player_analysis.avg_track_length is None

    def test_error_handling_in_pipeline(self, test_request):
        """Test pipeline error handling and recovery."""
        # Test with invalid URL
        with pytest.raises(Exception):
            YouTubeAnalysisRequest(url="https://invalid-url.com", output_dir="test_output")

    def test_configuration_loading(self):
        """Test pipeline configuration loading."""
        orchestrator = PipelineOrchestrator()
        assert orchestrator.config is not None
        assert "pipeline" in orchestrator.config
        assert "youtube" in orchestrator.config
        assert "output" in orchestrator.config

    def test_health_check_integration(self):
        """Test health check integration."""
        from src.utils.health_checks import health_check

        async def healthy_operation():
            return {"status": "healthy"}

        result = asyncio.run(health_check(healthy_operation)())
        assert result == {"status": "healthy"}


class TestPipelineMetrics:
    def test_metrics_collection(self):
        """Test that metrics are properly collected."""
        from src.utils.monitoring import get_gpu_memory_usage, get_system_metrics

        gpu_memory = get_gpu_memory_usage()
        system_metrics = get_system_metrics()

        assert isinstance(gpu_memory, int | float)
        assert isinstance(system_metrics, dict)
        assert "cpu_usage_percent" in system_metrics
        assert "memory_percent" in system_metrics


if __name__ == "__main__":
    # Run basic smoke tests
    pytest.main([__file__, "-v"])
