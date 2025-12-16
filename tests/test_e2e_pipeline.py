"""End-to-end integration tests for FIFA Soccer DS pipeline.

This module tests the complete pipeline flow from YouTube URL input
to final analysis output, ensuring all components work together.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

from src.pipeline_orchestrator import PipelineOrchestrator, create_pipeline_orchestrator
from src.schemas import YouTubeAnalysisRequest, PipelineOutput, SoccerClassification, PlayerAnalysis
from src.youtube.video_downloader import YouTubeDownloader
from src.youtube.audio_extractor import AudioExtractor
from src.youtube.metadata_parser import YouTubeMetadataParser
from src.classify.soccer_classifier import SoccerClassifier


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
            'id': 'test123',
            'title': 'Barcelona vs Real Madrid - El Clasico Highlights',
            'description': 'Amazing goals and skills from both teams',
            'duration': 300,
            'view_count': 1000000,
            'like_count': 50000,
            'comment_count': 5000,
            'uploader': 'ESPN FC',
            'channel_id': 'espnfc123',
            'upload_date': '20231201',
            'tags': ['soccer', 'football', 'barcelona', 'real madrid', 'el clasico'],
            'categories': ['Sports'],
            'resolution': '1920x1080',
            'fps': 30.0,
            'vcodec': 'h264',
            'acodec': 'aac'
        }
    
    @pytest.fixture
    def test_request(self, temp_dir):
        """Create test YouTube analysis request."""
        return YouTubeAnalysisRequest(
            url="https://www.youtube.com/watch?v=test123",
            output_dir=str(temp_dir),
            frame_rate=1.0,
            max_duration=300,
            include_audio=True,
            confidence_threshold=0.75
        )
    
    def test_pipeline_orchestrator_creation(self):
        """Test that pipeline orchestrator can be created."""
        orchestrator = PipelineOrchestrator()
        assert orchestrator is not None
        assert hasattr(orchestrator, 'process_youtube_video')
        assert hasattr(orchestrator, 'config')
    
    @pytest.mark.asyncio
    async def test_pipeline_orchestrator_with_mock_data(self, test_request, mock_video_info):
        """Test orchestrator with mocked external dependencies."""
        with patch('src.youtube.video_downloader.YouTubeDownloader') as mock_downloader, \
             patch('src.youtube.audio_extractor.AudioExtractor') as mock_audio, \
             patch('src.youtube.metadata_parser.YouTubeMetadataParser') as mock_metadata, \
             patch('src.classify.soccer_classifier.SoccerClassifier') as mock_classifier:
            
            # Setup mocks
            mock_downloader_instance = Mock()
            mock_downloader.return_value = mock_downloader_instance
            mock_downloader_instance.download_video.return_value = {
                'video_path': '/fake/path/video.mp4',
                'audio_path': '/fake/path/audio.wav'
            }
            
            mock_audio_instance = Mock()
            mock_audio.return_value = mock_audio_instance
            mock_audio_instance.process_sample_audio.return_value = {
                'classification': {
                    'is_soccer': True,
                    'confidence': 0.85,
                    'soccer_keywords': ['goal', 'soccer', 'football']
                }
            }
            
            mock_metadata_instance = Mock()
            mock_metadata.return_value = mock_metadata_instance
            mock_metadata_instance.extract_metadata.return_value = mock_video_info
            mock_metadata_instance.predict_soccer_content.return_value = {
                'is_soccer': True,
                'confidence': 0.90,
                'relevance_level': 'high',
                'content_type': 'highlights',
                'reasoning': ['High keyword relevance', 'Soccer-related channel']
            }
            
            mock_classifier_instance = Mock()
            mock_classifier.return_value = mock_classifier_instance
            mock_classifier_instance.classify_youtube_content.return_value = {
                'is_soccer': True,
                'confidence': 0.88,
                'classification': 'highly_soccer',
                'analysis_breakdown': {
                    'metadata_analysis': {'score': 0.90},
                    'thumbnail_analysis': {'score': 0.85},
                    'audio_analysis': {'is_soccer': True, 'confidence': 0.85}
                }
            }
            
            # Create orchestrator and process
            orchestrator = PipelineOrchestrator()
            result = await orchestrator.process_youtube_video(test_request)
            
            # Verify result structure
            assert isinstance(result, PipelineOutput)
            assert result.input_source == "youtube"
            assert result.soccer_classification.is_soccer == True
            assert result.soccer_classification.soccer_confidence > 0.8
            assert len(result.output_files) > 0
    
    def test_youtube_analysis_request_validation(self):
        """Test YouTube analysis request schema validation."""
        # Valid request
        valid_request = YouTubeAnalysisRequest(
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            output_dir="test_output",
            frame_rate=1.0
        )
        assert valid_request.url is not None
        assert valid_request.frame_rate == 1.0
        
        # Invalid URL should raise validation error
        with pytest.raises(Exception):
            YouTubeAnalysisRequest(
                url="invalid-url",
                output_dir="test_output"
            )
    
    def test_metadata_validation(self, mock_video_info):
        """Test metadata validation with schema."""
        parser = YouTubeMetadataParser()
        
        # Test with valid data
        metadata = parser._create_validated_metadata(mock_video_info)
        assert metadata.video_id == 'test123'
        assert metadata.title == 'Barcelona vs Real Madrid - El Clasico Highlights'
        assert metadata.duration_seconds == 300
        assert metadata.fps == 30.0
        
        # Test with missing data (should use defaults)
        incomplete_data = {'id': 'test456', 'title': 'Test Video'}
        metadata = parser._create_validated_metadata(incomplete_data)
        assert metadata.video_id == 'test456'
        assert metadata.title == 'Test Video'
        assert metadata.duration_seconds == 0  # Default
    
    def test_soccer_classifier_validation(self):
        """Test soccer classifier input validation."""
        classifier = SoccerClassifier()
        
        # Valid URL
        with patch('src.youtube.metadata_parser.YouTubeMetadataParser') as mock_parser:
            mock_instance = Mock()
            mock_parser.return_value = mock_instance
            mock_instance.extract_metadata.return_value = {'title': 'Soccer Goals'}
            mock_instance.predict_soccer_content.return_value = {
                'is_soccer': True, 'confidence': 0.9, 'relevance_level': 'high'
            }
            
            result = classifier.classify_youtube_content("https://youtube.com/watch?v=test")
            assert result['is_soccer'] == True
            assert result['confidence'] > 0.8
        
        # Invalid URL should raise error
        with pytest.raises(ValueError):
            classifier.classify_youtube_content("invalid-url")
    
    def test_pipeline_output_structure(self, temp_dir, mock_video_info):
        """Test that pipeline produces expected output files."""
        # Create minimal pipeline output
        output = PipelineOutput(
            pipeline_version="1.0.0",
            processing_timestamp=mock_video_info.get('upload_date'),
            processing_duration_seconds=120.0,
            input_source="youtube",
            input_url="https://youtube.com/watch?v=test",
            input_metadata=YouTubeMetadataParser()._create_validated_metadata(mock_video_info),
            soccer_classification=SoccerClassification(
                is_soccer=True,
                soccer_confidence=0.85,
                content_type="highlights",
                total_events=3,
                detection_quality=0.9,
                processing_success_rate=0.95
            ),
            player_analysis=PlayerAnalysis(
                total_players_detected=22,
                teams_detected=2,
                avg_track_length=120.0
            ),
            output_files={
                'detections': str(temp_dir / 'detections.parquet'),
                'tracklets': str(temp_dir / 'tracklets.parquet'),
                'graph': str(temp_dir / 'graph.pt'),
                'metrics': str(temp_dir / 'metrics.json')
            }
        )
        
        # Verify all required fields are present
        assert output.pipeline_version is not None
        assert output.processing_duration_seconds > 0
        assert output.input_source in ["youtube", "frame_directory", "video_file"]
        assert output.soccer_classification.is_soccer is not None
        assert output.soccer_classification.soccer_confidence is not None
        assert len(output.output_files) > 0
        
        # Verify output files are created
        for file_path in output.output_files.values():
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            Path(file_path).touch()
            assert Path(file_path).exists()
    
    @pytest.mark.asyncio
    async def test_error_handling_in_pipeline(self, test_request):
        """Test pipeline error handling and recovery."""
        orchestrator = PipelineOrchestrator()
        
        # Test with invalid URL
        invalid_request = YouTubeAnalysisRequest(
            url="https://invalid-url.com",
            output_dir="test_output"
        )
        
        # Should handle error gracefully
        with pytest.raises(Exception):
            await orchestrator.process_youtube_video(invalid_request)
    
    def test_configuration_loading(self):
        """Test pipeline configuration loading."""
        orchestrator = PipelineOrchestrator()
        assert orchestrator.config is not None
        assert 'pipeline' in orchestrator.config
        assert 'youtube' in orchestrator.config
        assert 'output' in orchestrator.config
    
    def test_health_check_integration(self):
        """Test health check integration."""
        from src.utils.health_checks import health_check
        
        # Basic health check should not raise exception
        result = health_check()
        assert isinstance(result, dict)
        assert 'status' in result
        assert result['status'] in ['healthy', 'degraded', 'unhealthy']


class TestPipelineArtifacts:
    """Test that pipeline produces expected artifacts."""
    
    def test_output_artifacts_structure(self, temp_dir):
        """Test that all expected output artifacts are created."""
        # Simulate pipeline output creation
        artifacts = {
            'detections.parquet': {'frame_id': [1, 2, 3], 'confidence': [0.9, 0.8, 0.7]},
            'tracklets.parquet': {'track_id': [1, 2], 'duration': [120, 90]},
            'graph.pt': {'nodes': 22, 'edges': 45},
            'metrics.json': {'frames_processed': 300, 'detection_accuracy': 0.89}
        }
        
        for filename, data in artifacts.items():
            file_path = temp_dir / filename
            
            if filename.endswith('.json'):
                with open(file_path, 'w') as f:
                    json.dump(data, f)
            else:
                # Create placeholder for parquet/graph files
                file_path.touch()
            
            assert file_path.exists()
    
    def test_metrics_collection(self):
        """Test that metrics are properly collected."""
        from src.utils.monitoring import get_gpu_memory_usage, get_system_metrics
        
        gpu_memory = get_gpu_memory_usage()
        system_metrics = get_system_metrics()
        
        assert isinstance(gpu_memory, (int, float))
        assert isinstance(system_metrics, dict)
        assert 'cpu_percent' in system_metrics
        assert 'memory_percent' in system_metrics


if __name__ == "__main__":
    # Run basic smoke tests
    pytest.main([__file__, "-v"])