"""Performance optimization utilities for pipeline components."""

from __future__ import annotations

import gc
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for pipeline operations."""
    operation: str
    start_time: float
    end_time: float
    duration: float
    memory_before: Optional[int] = None
    memory_after: Optional[int] = None
    items_processed: int = 0
    
    @property
    def items_per_second(self) -> float:
        """Calculate items processed per second."""
        if self.duration > 0:
            return self.items_processed / self.duration
        return 0.0
    
    @property
    def memory_delta(self) -> Optional[int]:
        """Calculate memory usage change."""
        if self.memory_before is not None and self.memory_after is not None:
            return self.memory_after - self.memory_before
        return None


def performance_monitor(operation_name: str) -> Callable:
    """Decorator to monitor function performance.
    
    Args:
        operation_name: Name of the operation being monitored
        
    Returns:
        Decorated function with performance monitoring
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get memory before operation
            try:
                import psutil
                process = psutil.Process()
                memory_before = process.memory_info().rss
            except ImportError:
                memory_before = None
                LOGGER.debug("psutil not available, memory monitoring disabled")
            
            # Time the operation
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            # Get memory after operation
            try:
                memory_after = process.memory_info().rss
            except Exception:
                memory_after = None
            
            duration = end_time - start_time
            
            # Log performance metrics
            metrics = PerformanceMetrics(
                operation=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                memory_before=memory_before,
                memory_after=memory_after,
            )
            
            LOGGER.info(
                "Performance: %s completed in %.3fs (memory: %s)",
                operation_name,
                duration,
                f"{(metrics.memory_delta or 0) / 1024 / 1024:.1f}MB" if metrics.memory_delta else "N/A"
            )
            
            return result
        return wrapper
    return decorator


@contextmanager
def memory_efficient_processing() -> Generator[None, None, None]:
    """Context manager for memory-efficient processing.
    
    Forces garbage collection and manages memory during intensive operations.
    """
    try:
        # Clear any existing garbage
        gc.collect()
        yield
    finally:
        # Force cleanup after processing
        gc.collect()
        
        # Clear OpenCV cache if available
        try:
            # Clear any cached data in OpenCV
            if hasattr(cv2, 'destroyAllWindows'):
                cv2.destroyAllWindows()
        except Exception:
            pass


class BatchProcessor:
    """Efficient batch processing for large datasets."""
    
    def __init__(self, batch_size: int = 32, max_memory_mb: int = 1024):
        self.batch_size = batch_size
        self.max_memory_mb = max_memory_mb
        self.processed_count = 0
        
    def process_in_batches(
        self, 
        items: list, 
        process_func: Callable[[list], Any],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> list:
        """Process items in memory-efficient batches.
        
        Args:
            items: List of items to process
            process_func: Function to process each batch
            progress_callback: Optional progress callback
            
        Returns:
            List of processed results
        """
        results = []
        total_items = len(items)

        with memory_efficient_processing():
            for i in range(0, total_items, self.batch_size):
                batch = items[i : i + self.batch_size]

                batch_result = process_func(batch)
                results.extend(batch_result if isinstance(batch_result, list) else [batch_result])

                self.processed_count += len(batch)

                if progress_callback:
                    progress_callback(self.processed_count, total_items)

                if self._check_memory_usage():
                    LOGGER.warning(
                        "High memory usage detected: %dMB processed",
                        self.processed_count,
                    )
                    gc.collect()
        
        return results
    
    def _check_memory_usage(self) -> bool:
        """Check if memory usage exceeds limits."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb > self.max_memory_mb
        except ImportError:
            return False


class VideoOptimizer:
    """Optimization utilities for video processing."""
    
    @staticmethod
    def optimize_frame_reading(frame_paths: list[Path]) -> list[Path]:
        """Optimize frame reading order for better cache performance.
        
        Args:
            frame_paths: List of frame file paths
            
        Returns:
            Optimized list of frame paths
        """
        # Sort frames to ensure sequential access
        sorted_paths = sorted(frame_paths)
        
        # Group by directory for better locality
        directories = {}
        for path in sorted_paths:
            dir_path = str(path.parent)
            if dir_path not in directories:
                directories[dir_path] = []
            directories[dir_path].append(path)
        
        # Flatten while maintaining directory grouping
        optimized = []
        for dir_path in sorted(directories.keys()):
            optimized.extend(directories[dir_path])
        
        return optimized
    
    @staticmethod
    def preload_frames(
        frame_paths: list[Path], 
        max_preload: int = 10
    ) -> dict[Path, np.ndarray]:
        """Preload frequently accessed frames into memory.
        
        Args:
            frame_paths: List of frame paths
            max_preload: Maximum number of frames to preload
            
        Returns:
            Dictionary mapping paths to loaded frames
        """
        frame_cache = {}
        
        for i, path in enumerate(frame_paths[:max_preload]):
            try:
                frame = cv2.imread(str(path))
                if frame is not None:
                    frame_cache[path] = frame
                else:
                    LOGGER.warning("Failed to preload frame: %s", path)
            except Exception as e:
                LOGGER.error("Error preloading frame %s: %s", path, e)
        
        LOGGER.info("Preloaded %d frames into memory", len(frame_cache))
        return frame_cache
    
    @staticmethod
    def resize_for_performance(
        frame: np.ndarray, 
        max_dimension: int = 1920
    ) -> np.ndarray:
        """Resize frame for better performance while maintaining aspect ratio.
        
        Args:
            frame: Input frame
            max_dimension: Maximum dimension for resized frame
            
        Returns:
            Resized frame
        """
        height, width = frame.shape[:2]
        
        # Skip if already small enough
        if max(height, width) <= max_dimension:
            return frame
        
        # Calculate new dimensions maintaining aspect ratio
        if width > height:
            new_width = max_dimension
            new_height = int(height * max_dimension / width)
        else:
            new_height = max_dimension
            new_width = int(width * max_dimension / height)
        
        # Use high-quality interpolation
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


class DetectionOptimizer:
    """Optimization utilities for object detection."""
    
    @staticmethod
    def batch_detections(
        frames: list[np.ndarray], 
        detector, 
        confidence: float = 0.25,
        batch_size: int = 8
    ) -> list[list]:
        """Process detections in batches for better GPU utilization.
        
        Args:
            frames: List of frames to detect
            detector: YOLO detector instance
            confidence: Detection confidence threshold
            batch_size: Batch size for processing
            
        Returns:
            List of detection results
        """
        all_results = []
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            
            # Process batch
            batch_results = detector.predict(batch, conf=confidence, verbose=False)
            all_results.extend(batch_results)
            
            # Log batch progress
            LOGGER.debug(
                "Processed detection batch %d-%d/%d", 
                i, 
                min(i + batch_size, len(frames)), 
                len(frames)
            )
        
        return all_results
    
    @staticmethod
    def filter_detections_early(
        detections: list, 
        min_confidence: float = 0.25,
        max_detections: int = 50
    ) -> list:
        """Filter detections early to reduce processing load.
        
        Args:
            detections: List of raw detections
            min_confidence: Minimum confidence threshold
            max_detections: Maximum number of detections to keep
            
        Returns:
            Filtered list of detections
        """
        # Filter by confidence
        filtered = [d for d in detections if d.get('confidence', 0) >= min_confidence]
        
        # Limit number of detections if specified
        if max_detections > 0 and len(filtered) > max_detections:
            # Sort by confidence and keep top N
            filtered.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            filtered = filtered[:max_detections]
        
        return filtered


class TrackingOptimizer:
    """Optimization utilities for object tracking."""
    
    @staticmethod
    def optimize_distance_threshold(
        frame_size: tuple[int, int], 
        object_size: tuple[int, int] = (50, 50)
    ) -> float:
        """Calculate optimal distance threshold based on frame and object size.
        
        Args:
            frame_size: Frame dimensions (width, height)
            object_size: Typical object size (width, height)
            
        Returns:
            Optimal distance threshold
        """
        frame_width, frame_height = frame_size
        obj_width, obj_height = object_size
        
        # Calculate relative size
        frame_area = frame_width * frame_height
        obj_area = obj_width * obj_height
        size_ratio = obj_area / frame_area
        
        # Adaptive threshold based on relative size
        base_threshold = 80.0
        if size_ratio > 0.01:  # Large objects
            return base_threshold * 1.5
        elif size_ratio < 0.001:  # Small objects
            return base_threshold * 0.5
        else:
            return base_threshold
    
    @staticmethod
    def early_termination_check(
        tracklets: list, 
        max_lost_frames: int = 30
    ) -> bool:
        """Check if tracking should terminate early.
        
        Args:
            tracklets: Current tracklets
            max_lost_frames: Maximum frames without detections
            
        Returns:
            True if should terminate early
        """
        if not tracklets:
            return False
        
        # Count tracks that haven't been updated recently
        lost_tracks = 0
        for track in tracklets:
            if hasattr(track, 'lost_frames') and track.lost_frames > max_lost_frames:
                lost_tracks += 1
        
        # Terminate if too many lost tracks
        return lost_tracks > len(tracklets) * 0.5


# Performance monitoring context manager
@contextmanager
def monitor_performance(operation: str) -> Generator[PerformanceMetrics, None, None]:
    """Context manager for performance monitoring.
    
    Args:
        operation: Name of the operation being monitored
        
    Yields:
        PerformanceMetrics object
    """
    start_time = time.time()
    
    try:
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss
    except ImportError:
        memory_before = None
    
    try:
        metrics = PerformanceMetrics(
            operation=operation,
            start_time=start_time,
            end_time=0.0,  # Will be set in finally
            duration=0.0,     # Will be set in finally
            memory_before=memory_before,
        )
        
        yield metrics
        
    finally:
        end_time = time.time()
        
        try:
            memory_after = process.memory_info().rss
        except Exception:
            memory_after = None
        
        metrics.end_time = end_time
        metrics.duration = end_time - start_time
        metrics.memory_after = memory_after
        
        LOGGER.info(
            "Performance: %s - %.3fs, %sMB memory",
            operation,
            metrics.duration,
            f"{(metrics.memory_delta or 0) / 1024 / 1024:.1f}" if metrics.memory_delta else "N/A"
        )
