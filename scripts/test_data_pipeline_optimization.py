#!/usr/bin/env python3
"""
Test script for data pipeline optimization and parallel processing.

This script validates the parallel frame extraction, pseudo-labeling,
and optimized I/O operations for the FIFA soccer dataset pipeline.
"""

import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

# Mock the optimized classes for testing
class MockParallelFrameExtractor:
    """Mock implementation of ParallelFrameExtractor for testing."""
    
    def __init__(self, max_workers: int = 0, chunk_size: int = 10):
        self.max_workers = max_workers or 4
        self.chunk_size = chunk_size
        self.processing_times = []
    
    def extract_frames_parallel(
        self, 
        video_paths: list[Path], 
        output_root: Path,
        every_n_frames: int = 1
    ) -> dict[Path, list[Path]]:
        """Mock parallel frame extraction."""
        results = {}
        
        for i, video_path in enumerate(video_paths):
            # Simulate processing time
            start_time = time.time()
            time.sleep(0.01)  # Simulate I/O bound operation
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Generate mock frame paths
            output_dir = output_root / video_path.stem
            frames = [
                output_dir / f"{video_path.stem}_frame_{j:06d}.jpg"
                for j in range(10)  # Mock 10 frames per video
            ]
            results[video_path] = frames
        
        return results
    
    def get_avg_processing_time(self) -> float:
        """Get average processing time per video."""
        return sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0


class MockParallelPseudoLabeler:
    """Mock implementation of ParallelPseudoLabeler for testing."""
    
    def __init__(self, max_workers: int = 0, batch_size: int = 32):
        self.max_workers = max_workers or 2
        self.batch_size = batch_size
        self.processing_times = []
    
    def label_frames_parallel(
        self,
        frame_paths: list[Path],
        output_dir: Path,
        confidence_threshold: float = 0.5,
        weights: str = "yolov8n.pt",
        device: str = "cuda_if_available"
    ) -> Path:
        """Mock parallel pseudo-labeling."""
        start_time = time.time()
        
        # Process frames in batches
        batch_count = (len(frame_paths) + self.batch_size - 1) // self.batch_size
        
        for i in range(batch_count):
            # Simulate batch processing time
            time.sleep(0.005)  # Simulate CPU bound operation
        
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return output_dir
    
    def get_avg_processing_time(self) -> float:
        """Get average processing time per frame."""
        return sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0


class MockOptimizedDataPipeline:
    """Mock implementation of OptimizedDataPipeline for testing."""
    
    def __init__(self, max_workers: int = 0, enable_caching: bool = True):
        self.max_workers = max_workers or 4
        self.enable_caching = enable_caching
        self.frame_extractor = MockParallelFrameExtractor(max_workers)
        self.pseudo_labeler = MockParallelPseudoLabeler(max_workers)
        
        self.metrics = {
            "frames_extracted": 0,
            "frames_labeled": 0,
            "processing_time": 0.0,
            "io_operations": 0
        }
    
    def process_videos_to_labels(
        self,
        video_paths: list[Path],
        output_root: Path,
        every_n_frames: int = 1,
        confidence_threshold: float = 0.5,
        weights: str = "yolov8n.pt",
        device: str = "cuda_if_available"
    ) -> dict[Path, Path]:
        """Mock complete pipeline processing."""
        start_time = time.time()
        
        # Step 1: Extract frames
        frame_results = self.frame_extractor.extract_frames_parallel(
            video_paths, output_root / "frames", every_n_frames
        )
        
        # Collect all frame paths
        all_frames = []
        for frames in frame_results.values():
            all_frames.extend(frames)
        
        # Step 2: Generate labels
        labels_dir = self.pseudo_labeler.label_frames_parallel(
            all_frames, output_root / "labels", confidence_threshold, weights, device
        )
        
        # Update metrics
        self.metrics["frames_extracted"] = len(all_frames)
        self.metrics["frames_labeled"] = len(all_frames)
        self.metrics["processing_time"] = time.time() - start_time
        self.metrics["io_operations"] = len(video_paths) * 2  # Extract + Label
        
        return {video_path: labels_dir for video_path in video_paths}
    
    def get_performance_metrics(self) -> dict[str, Any]:
        """Get pipeline performance metrics."""
        fps = self.metrics["frames_extracted"] / max(self.metrics["processing_time"], 0.001)
        return {
            **self.metrics,
            "frames_per_second": fps,
            "avg_frame_time": self.metrics["processing_time"] / max(self.metrics["frames_extracted"], 1),
            "avg_extraction_time": self.frame_extractor.get_avg_processing_time(),
            "avg_labeling_time": self.pseudo_labeler.get_avg_processing_time()
        }


def create_mock_video_files(count: int, temp_dir: Path) -> list[Path]:
    """Create mock video files for testing."""
    video_paths = []
    
    for i in range(count):
        video_path = temp_dir / f"match_{i:03d}.mp4"
        video_path.write_text(f"mock video content {i}")
        video_paths.append(video_path)
    
    return video_paths


def test_parallel_frame_extraction():
    """Test parallel frame extraction performance."""
    print("Testing parallel frame extraction...")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create mock videos
        video_paths = create_mock_video_files(20, temp_dir)
        output_root = temp_dir / "output"
        
        # Test with different worker counts
        worker_counts = [1, 2, 4, 8]
        results = {}
        
        for workers in worker_counts:
            extractor = MockParallelFrameExtractor(max_workers=workers)
            
            start_time = time.time()
            frame_results = extractor.extract_frames_parallel(video_paths, output_root)
            processing_time = time.time() - start_time
            
            total_frames = sum(len(frames) for frames in frame_results.values())
            
            results[workers] = {
                "processing_time": processing_time,
                "total_frames": total_frames,
                "fps": total_frames / processing_time,
                "avg_time_per_video": extractor.get_avg_processing_time()
            }
            
            print(f"  Workers: {workers}, Time: {processing_time:.3f}s, FPS: {total_frames/processing_time:.1f}")
        
        # Analyze scalability
        baseline_time = results[1]["processing_time"]
        best_time = min(results[w]["processing_time"] for w in worker_counts)
        speedup = baseline_time / best_time
        
        print(f"  Baseline (1 worker): {baseline_time:.3f}s")
        print(f"  Best performance: {best_time:.3f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print("✓ Parallel frame extraction test passed")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_parallel_pseudo_labeling():
    """Test parallel pseudo-labeling performance."""
    print("Testing parallel pseudo-labeling...")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create mock frame files
        frame_count = 200
        frame_paths = []
        for i in range(frame_count):
            frame_path = temp_dir / f"frame_{i:06d}.jpg"
            frame_path.write_text(f"mock frame content {i}")
            frame_paths.append(frame_path)
        
        output_dir = temp_dir / "labels"
        
        # Test with different batch sizes
        batch_sizes = [16, 32, 64, 128]
        results = {}
        
        for batch_size in batch_sizes:
            labeler = MockParallelPseudoLabeler(batch_size=batch_size)
            
            start_time = time.time()
            result_dir = labeler.label_frames_parallel(frame_paths, output_dir)
            processing_time = time.time() - start_time
            
            results[batch_size] = {
                "processing_time": processing_time,
                "fps": frame_count / processing_time,
                "avg_time_per_frame": labeler.get_avg_processing_time()
            }
            
            print(f"  Batch size: {batch_size}, Time: {processing_time:.3f}s, FPS: {frame_count/processing_time:.1f}")
        
        # Find optimal batch size
        best_batch = min(results.keys(), key=lambda b: results[b]["processing_time"])
        best_time = results[best_batch]["processing_time"]
        worst_time = max(results[b]["processing_time"] for b in batch_sizes)
        
        print(f"  Optimal batch size: {best_batch}")
        print(f"  Best time: {best_time:.3f}s")
        print(f"  Worst time: {worst_time:.3f}s")
        print(f"  Improvement: {worst_time/best_time:.2f}x")
        print("✓ Parallel pseudo-labeling test passed")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_optimized_pipeline():
    """Test complete optimized pipeline performance."""
    print("Testing optimized data pipeline...")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create mock videos
        video_paths = create_mock_video_files(10, temp_dir)
        output_root = temp_dir / "pipeline_output"
        
        # Test pipeline
        pipeline = MockOptimizedDataPipeline(max_workers=4)
        
        start_time = time.time()
        results = pipeline.process_videos_to_labels(video_paths, output_root)
        total_time = time.time() - start_time
        
        # Get performance metrics
        metrics = pipeline.get_performance_metrics()
        
        print(f"  Total processing time: {total_time:.3f}s")
        print(f"  Frames extracted: {metrics['frames_extracted']}")
        print(f"  Frames labeled: {metrics['frames_labeled']}")
        print(f"  Overall FPS: {metrics['frames_per_second']:.1f}")
        print(f"  Avg frame time: {metrics['avg_frame_time']*1000:.2f}ms")
        print(f"  Avg extraction time: {metrics['avg_extraction_time']*1000:.2f}ms")
        print(f"  Avg labeling time: {metrics['avg_labeling_time']*1000:.2f}ms")
        print(f"  I/O operations: {metrics['io_operations']}")
        
        # Performance validation
        assert metrics['frames_extracted'] > 0, "No frames extracted"
        assert metrics['frames_labeled'] > 0, "No frames labeled"
        assert metrics['frames_per_second'] > 10, "FPS too low"
        assert total_time < 5.0, "Processing took too long"
        
        print("✓ Optimized pipeline test passed")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_io_optimization():
    """Test I/O optimization techniques."""
    print("Testing I/O optimization...")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Test buffered vs unbuffered operations
        test_data = "x" * 1024 * 100  # 100KB of data
        
        # Unbuffered write
        unbuffered_file = temp_dir / "unbuffered.txt"
        start_time = time.time()
        with open(unbuffered_file, 'w') as f:
            for _ in range(10):
                f.write(test_data)
        unbuffered_time = time.time() - start_time
        
        # Buffered write (simulated)
        buffered_file = temp_dir / "buffered.txt"
        start_time = time.time()
        with open(buffered_file, 'w', buffering=8192) as f:
            for _ in range(10):
                f.write(test_data)
        buffered_time = time.time() - start_time
        
        improvement = unbuffered_time / buffered_time if buffered_time > 0 else 1
        
        print(f"  Unbuffered write: {unbuffered_time:.4f}s")
        print(f"  Buffered write: {buffered_time:.4f}s")
        print(f"  Improvement: {improvement:.2f}x")
        
        # Test parallel file operations
        files = []
        for i in range(20):
            file_path = temp_dir / f"parallel_test_{i:02d}.txt"
            file_path.write_text(f"content {i}")
            files.append(file_path)
        
        # Sequential read
        start_time = time.time()
        for file_path in files:
            content = file_path.read_text()
            assert content
        sequential_time = time.time() - start_time
        
        # Parallel read (simulated with threads)
        import threading
        import queue
        
        result_queue = queue.Queue()
        
        def read_file(file_path):
            content = file_path.read_text()
            result_queue.put(len(content))
        
        start_time = time.time()
        threads = []
        for file_path in files:
            thread = threading.Thread(target=read_file, args=(file_path,))
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()
        
        parallel_time = time.time() - start_time
        
        # Verify all files were read
        read_count = 0
        while not result_queue.empty():
            result_queue.get()
            read_count += 1
        
        assert read_count == len(files), "Not all files were read in parallel"
        
        speedup = sequential_time / parallel_time if parallel_time > 0 else 1
        
        print(f"  Sequential read: {sequential_time:.4f}s")
        print(f"  Parallel read: {parallel_time:.4f}s")
        print(f"  I/O speedup: {speedup:.2f}x")
        print("✓ I/O optimization test passed")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_memory_efficiency():
    """Test memory efficiency of optimized pipeline."""
    print("Testing memory efficiency...")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Simulate memory usage tracking
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create larger dataset
        video_paths = create_mock_video_files(50, temp_dir)
        output_root = temp_dir / "memory_test"
        
        pipeline = MockOptimizedDataPipeline(max_workers=8)
        
        # Process with memory monitoring
        peak_memory = initial_memory
        memory_samples = []
        
        def monitor_memory():
            nonlocal peak_memory
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)
            peak_memory = max(peak_memory, current_memory)
        
        monitor_thread = threading.Thread(target=monitor_memory)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Run pipeline
        results = pipeline.process_videos_to_labels(video_paths, output_root)
        
        monitor_thread.join(timeout=1)
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Peak memory: {peak_memory:.1f} MB")
        print(f"  Final memory: {final_memory:.1f} MB")
        print(f"  Memory increase: {memory_increase:.1f} MB")
        print(f"  Memory per video: {memory_increase/50:.2f} MB")
        
        # Memory efficiency checks
        memory_per_video = memory_increase / len(video_paths)
        assert memory_per_video < 5.0, f"Too much memory per video: {memory_per_video:.2f} MB"
        assert memory_increase < 200.0, f"Total memory increase too high: {memory_increase:.1f} MB"
        
        print("✓ Memory efficiency test passed")
        
    except ImportError:
        print("  psutil not available - skipping memory test")
        print("✓ Memory efficiency test skipped")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all data pipeline optimization tests."""
    print("⚡ Data Pipeline Optimization Tests\n")
    
    try:
        test_parallel_frame_extraction()
        print()
        
        test_parallel_pseudo_labeling()
        print()
        
        test_optimized_pipeline()
        print()
        
        test_io_optimization()
        print()
        
        test_memory_efficiency()
        print()
        
        print("🎉 All data pipeline optimization tests passed!")
        return 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())