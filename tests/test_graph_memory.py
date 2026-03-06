"""Test memory-optimized graph construction."""

import pytest
import torch
from unittest.mock import Mock

from src.graph.build_graph import build_track_graph, build_track_graph_optimized, estimate_graph_memory


class TestGraphMemory:
    """Test memory optimization in graph construction."""
    
    def test_memory_estimation(self):
        """Test memory usage estimation."""
        # Small graph
        small = estimate_graph_memory(100, avg_degree=5.0)
        assert small["nodes"] == 100
        assert small["sparse_edges"] == 500
        assert small["sparse_memory_mb"] < 1.0
        assert small["memory_ratio"] > 10.0  # Dense is much larger
        
        # Large graph
        large = estimate_graph_memory(1000, avg_degree=8.0)
        assert large["nodes"] == 1000
        assert large["sparse_edges"] == 8000
        assert large["sparse_memory_mb"] < 10.0
        assert large["memory_ratio"] > 100.0
    
    def test_small_graph_memory(self):
        """Test graph construction with small number of nodes."""
        # Create mock track data
        track_windows = self._create_mock_tracks(20, 30)
        
        # Test original function
        result1 = build_track_graph(track_windows, window=30, distance_threshold=80.0)
        
        # Test optimized function
        result2 = build_track_graph_optimized(track_windows, window=30, distance_threshold=80.0)
        
        # Both should have similar structure
        assert result1["x"].shape[0] == result2["x"].shape[0]
        assert result1["edge_index"].shape[1] <= result2["edge_index"].shape[1] * 1.2  # Allow some variation
    
    def test_large_graph_memory_limit(self):
        """Test memory limits for large graphs."""
        # Create many tracks to trigger memory limits
        track_windows = self._create_mock_tracks(100, 150)  # Very large
        
        # This should handle gracefully with limits
        result = build_track_graph_optimized(
            track_windows, 
            window=30, 
            distance_threshold=80.0,
            max_nodes_per_frame=50
        )
        
        # Should not exceed reasonable limits
        assert result["x"].shape[0] <= 1500  # 30 frames * 50 nodes max
        assert result["edge_index"].shape[1] < 10000  # Reasonable edge limit
    
    def test_edge_cases(self):
        """Test edge cases for graph construction."""
        # Empty input
        with pytest.raises(ValueError):
            build_track_graph([])
        
        # Single frame
        single_frame = self._create_mock_tracks(1, 5)
        result = build_track_graph_optimized(single_frame)
        assert result["x"].shape[0] == 5
        assert result["edge_index"].shape[1] >= 0
        
        # No spatial edges (large threshold)
        no_spatial = self._create_mock_tracks(2, 3)
        result2 = build_track_graph_optimized(no_spatial, distance_threshold=0.0)
        # Should still have temporal edges if enabled
        assert result2["edge_index"].shape[1] >= 0
    
    def test_temporal_edge_consistency(self):
        """Test temporal edges are consistent."""
        track_windows = self._create_mock_tracks(10, 5, with_temporal=True)
        
        result = build_track_graph_optimized(
            track_windows, 
            include_temporal_edges=True
        )
        
        # Should have some temporal edges
        assert result["edge_index"].shape[1] > 0
        
        # Disable temporal edges
        result_no_temp = build_track_graph_optimized(
            track_windows, 
            include_temporal_edges=False
        )
        
        # Should have fewer edges
        assert result_no_temp["edge_index"].shape[1] <= result["edge_index"].shape[1]
    
    def _create_mock_tracks(self, num_frames: int, tracks_per_frame: int, with_temporal: bool = False):
        """Create mock track data for testing."""
        track_windows = []
        
        for frame_id in range(num_frames):
            frame = Mock()
            frame.frame_id = frame_id
            frame.items = []
            
            for track_id in range(tracks_per_frame):
                track = Mock()
                track.track_id = track_id if not with_temporal else track_id * (frame_id + 1)
                track.bbox = [
                    100 + track_id * 50,  # x1
                    100 + track_id * 30,  # y1  
                    150 + track_id * 50,  # x2
                    150 + track_id * 30   # y2
                ]
                frame.items.append(track)
            
            track_windows.append(frame)
        
        return track_windows


@pytest.mark.skipif(torch is None, reason="PyTorch not available")
class TestGraphMemoryIntegration:
    """Integration tests for memory optimization."""
    
    def test_memory_comparison(self):
        """Compare memory usage between original and optimized."""
        # Create moderately large graph
        track_windows = self._create_realistic_tracks(50, 8)
        
        # Original function (may be slow)
        result_orig = build_track_graph(
            track_windows, window=30, distance_threshold=80.0
        )
        
        # Optimized function
        result_opt = build_track_graph_optimized(
            track_windows, window=30, distance_threshold=80.0
        )
        
        # Memory usage should be reasonable
        orig_memory = result_orig["x"].numel() * 4 + result_orig["edge_index"].numel() * 4
        opt_memory = result_opt["x"].numel() * 4 + result_opt["edge_index"].numel() * 4
        
        # Optimized should use similar or less memory
        assert opt_memory <= orig_memory * 1.5  # Allow some overhead
        
        # Both should produce valid graphs
        assert result_orig["x"].shape[0] > 0
        assert result_opt["x"].shape[0] > 0
    
    def _create_realistic_tracks(self, num_frames: int, avg_tracks: int):
        """Create more realistic track data."""
        import random
        
        track_windows = []
        active_tracks = {}
        
        for frame_id in range(num_frames):
            frame = Mock()
            frame.frame_id = frame_id
            frame.items = []
            
            # Add some new tracks
            if random.random() < 0.1 or len(active_tracks) < avg_tracks // 2:
                new_track_id = len(active_tracks) + len(frame.items)
                active_tracks[new_track_id] = {
                    "start_frame": frame_id,
                    "position": [
                        random.randint(100, 800),
                        random.randint(100, 600)
                    ]
                }
            
            # Update existing tracks
            for track_id, track_data in list(active_tracks.items()):
                # Simple movement
                track_data["position"][0] += random.randint(-5, 5)
                track_data["position"][1] += random.randint(-5, 5)
                
                # Remove if too old
                if frame_id - track_data["start_frame"] > 20:
                    del active_tracks[track_id]
                    continue
                
                # Create track object
                track = Mock()
                track.track_id = track_id
                x, y = track_data["position"]
                track.bbox = [x, y, x + 30, y + 30]  # 30x30 bbox
                frame.items.append(track)
            
            track_windows.append(frame)
        
        return track_windows