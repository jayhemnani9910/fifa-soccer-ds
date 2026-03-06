"""
Tests for YouTube Analyzer API endpoints.

These tests verify the functionality of the FastAPI endpoints
for YouTube video processing and analysis.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import the API app
from src.api.youtube_endpoints import app

# Create test client
client = TestClient(app)


class TestYouTubeAPI:
    """Test suite for YouTube Analyzer API."""
    
    def test_root_endpoint(self):
        """Test the root endpoint returns API information."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "YouTube Analyzer API" in data["message"]
        assert "endpoints" in data
    
    def test_health_check(self):
        """Test the health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "youtube-analyzer"
    
    def test_start_analysis_with_valid_url(self):
        """Test starting analysis with a valid YouTube URL."""
        test_request = {
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "confidence_threshold": 0.8
        }
        
        response = client.post("/analyze", json=test_request)
        
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "pending"
        assert "Analysis started" in data["message"]
    
    def test_start_analysis_with_invalid_url(self):
        """Test starting analysis with an invalid URL."""
        test_request = {
            "url": "invalid-url",
            "confidence_threshold": 0.8
        }
        
        response = client.post("/analyze", json=test_request)
        
        # Should still accept but may fail during processing
        assert response.status_code == 200
    
    def test_start_analysis_minimal_params(self):
        """Test starting analysis with minimal parameters."""
        test_request = {
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        }
        
        response = client.post("/analyze", json=test_request)
        
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
    
    def test_get_status_nonexistent_task(self):
        """Test getting status for a task that doesn't exist."""
        response = client.get("/status/nonexistent_task_123")
        
        assert response.status_code == 404
        assert "Task not found" in response.json()["detail"]
    
    def test_get_results_nonexistent_task(self):
        """Test getting results for a task that doesn't exist."""
        response = client.get("/results/nonexistent_task_123")
        
        assert response.status_code == 404
        assert "Task not found" in response.json()["detail"]
    
    def test_delete_nonexistent_task(self):
        """Test deleting a task that doesn't exist."""
        response = client.delete("/task/nonexistent_task_123")
        
        assert response.status_code == 404
        assert "Task not found" in response.json()["detail"]
    
    def test_get_metadata_invalid_video_id(self):
        """Test getting metadata for an invalid video ID."""
        response = client.get("/metadata/invalid_video_id_123")
        
        # This might return 500 if dependencies aren't available
        # or 404 if video doesn't exist
        assert response.status_code in [404, 500]
    
    def test_download_nonexistent_results(self):
        """Test downloading results for a task that doesn't exist."""
        response = client.get("/download/nonexistent_task_123")
        
        assert response.status_code == 404
    
    @patch('src.api.youtube_endpoints.task_status')
    def test_get_status_existing_task(self, mock_task_status):
        """Test getting status for an existing task."""
        # Mock task status
        mock_task_status.get.return_value = {
            "status": "processing",
            "message": "Analyzing video...",
            "progress": 0.5
        }
        
        response = client.get("/status/test_task_123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"
        assert data["progress"] == 0.5
    
    def test_api_request_validation(self):
        """Test API request validation for missing required fields."""
        response = client.post("/analyze", json={})
        
        # Should fail validation for missing URL
        assert response.status_code == 422  # Pydantic validation error


class TestAPIIntegration:
    """Integration tests for the YouTube API."""
    
    def test_full_analysis_workflow_simulation(self):
        """Simulate a complete analysis workflow."""
        # Start analysis
        start_response = client.post("/analyze", json={
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        })
        
        assert start_response.status_code == 200
        task_id = start_response.json()["task_id"]
        
        # Check status (should be pending or processing)
        status_response = client.get(f"/status/{task_id}")
        assert status_response.status_code == 200
        
        # Get results (should not be ready yet)
        results_response = client.get(f"/results/{task_id}")
        assert results_response.status_code == 400  # Not ready
    
    def test_multiple_concurrent_requests(self):
        """Test handling multiple concurrent analysis requests."""
        requests = [
            {"url": f"https://www.youtube.com/watch?v=video{i}"}
            for i in range(3)
        ]
        
        responses = []
        for req in requests:
            response = client.post("/analyze", json=req)
            responses.append(response)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "task_id" in data


class TestAPIErrorHandling:
    """Test API error handling and edge cases."""
    
    def test_malformed_json_request(self):
        """Test handling of malformed JSON requests."""
        response = client.post(
            "/analyze",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_unsupported_content_type(self):
        """Test handling of unsupported content types."""
        response = client.post(
            "/analyze",
            data="some text",
            headers={"Content-Type": "text/plain"}
        )
        
        assert response.status_code == 422
    
    def test_very_long_url(self):
        """Test handling of extremely long URLs."""
        long_url = "https://www.youtube.com/watch?v=" + "a" * 1000
        response = client.post("/analyze", json={"url": long_url})
        
        # Should handle gracefully (may succeed or fail with validation error)
        assert response.status_code in [200, 422, 413]  # 413 = Payload too large


if __name__ == "__main__":
    # Run basic smoke tests
    print("Running YouTube API smoke tests...")
    
    # Test basic endpoints
    try:
        response = client.get("/")
        print(f"✅ Root endpoint: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
    
    try:
        response = client.get("/health")
        print(f"✅ Health check: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    try:
        response = client.post("/analyze", json={
            "url": "https://www.youtube.com/watch?v=test"
        })
        print(f"✅ Analysis start: {response.status_code}")
    except Exception as e:
        print(f"❌ Analysis start failed: {e}")
    
    print("Smoke tests completed!")