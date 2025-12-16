#!/usr/bin/env python3
"""
Test suite for race condition protection in weekly retraining.

This test validates the thread-safe operations, file locking mechanisms,
and atomic file operations to prevent concurrent access issues.
"""

import json
import pytest
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.train.weekly_retrainer import (
    RetrainingLock, 
    AtomicFileWriter,
    _increment_version_safe
)


class TestRetrainingLock:
    """Test the RetrainingLock class for proper file locking behavior."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.lock_file = self.temp_dir / "test.lock"
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_basic_lock_acquisition(self):
        """Test basic lock acquisition and release."""
        lock = RetrainingLock(self.lock_file, timeout=1.0)
        
        assert lock.acquire() is True
        assert self.lock_file.exists()
        assert lock.is_locked() is True
        
        lock.release()
        assert not self.lock_file.exists()
        assert lock.is_locked() is False
    
    def test_context_manager(self):
        """Test RetrainingLock as context manager."""
        with RetrainingLock(self.lock_file, timeout=1.0) as lock:
            assert lock.is_locked() is True
            assert self.lock_file.exists()
        
        assert not self.lock_file.exists()
    
    def test_timeout_behavior(self):
        """Test timeout when lock cannot be acquired."""
        # First thread acquires lock
        lock1 = RetrainingLock(self.lock_file, timeout=2.0)
        lock1.acquire()
        
        # Second thread should timeout
        lock2 = RetrainingLock(self.lock_file, timeout=0.5)
        start_time = time.time()
        result = lock2.acquire()
        elapsed = time.time() - start_time
        
        assert result is False
        assert elapsed >= 0.5  # Should wait at least timeout
        
        # Cleanup
        lock1.release()
    
    def test_stale_lock_detection(self):
        """Test detection and cleanup of stale locks."""
        # Create a stale lock file (old timestamp)
        with open(self.lock_file, 'w') as f:
            f.write(f"{threading.get_ident()}\n{time.time() - 400}\n")  # 400 seconds old
        
        lock = RetrainingLock(self.lock_file, stale_threshold=300)
        
        # Should detect stale lock and remove it
        assert lock.acquire() is True
        
        # Lock file should be replaced with new one
        assert self.lock_file.exists()
        
        lock.release()
    
    def test_concurrent_access(self):
        """Test concurrent access to the same lock."""
        results = []
        
        def worker(worker_id):
            try:
                with RetrainingLock(self.lock_file, timeout=2.0) as lock:
                    results.append(f"worker_{worker_id}_acquired")
                    time.sleep(0.1)  # Simulate work
                    results.append(f"worker_{worker_id}_released")
            except Exception as e:
                results.append(f"worker_{worker_id}_error: {e}")
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should have sequential access (no interleaving)
        assert len(results) == 6  # Each worker: acquired + released
        assert "_acquired" in results[0]
        assert "_released" in results[1]
        assert "_acquired" in results[2]
        assert "_released" in results[3]
        assert "_acquired" in results[4]
        assert "_released" in results[5]


class TestAtomicFileWriter:
    """Test the AtomicFileWriter class for atomic file operations."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.target_file = self.temp_dir / "test.json"
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_atomic_write(self):
        """Test basic atomic file writing."""
        data = {"version": 5, "timestamp": "2024-01-01"}
        
        with AtomicFileWriter(self.target_file) as writer:
            writer.write(json.dumps(data))
        
        # File should exist and contain correct data
        assert self.target_file.exists()
        with open(self.target_file) as f:
            loaded_data = json.load(f)
        assert loaded_data == data
    
    def test_write_failure_cleanup(self):
        """Test cleanup when write operation fails."""
        # Mock write to raise exception
        with pytest.raises(ValueError), AtomicFileWriter(self.target_file) as writer:
            writer.write("valid data")
            raise ValueError("Simulated write failure")
        
        # Target file should not exist after failure
        assert not self.target_file.exists()
        
        # Temp files should be cleaned up
        temp_files = list(self.temp_dir.glob("*.tmp"))
        assert len(temp_files) == 0
    
    def test_concurrent_atomic_writes(self):
        """Test concurrent atomic writes to same file."""
        def worker(worker_id, data):
            try:
                with AtomicFileWriter(self.target_file) as writer:
                    writer.write(json.dumps({"worker": worker_id, "data": data}))
                    time.sleep(0.05)  # Simulate write time
                return True
            except Exception:
                return False
        
        # Start multiple threads writing different data
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i, f"data_{i}"))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Final file should exist and be valid JSON
        assert self.target_file.exists()
        with open(self.target_file) as f:
            data = json.load(f)
        
        # Should contain exactly one worker's data
        assert "worker" in data
        assert isinstance(data["worker"], int)
        assert 0 <= data["worker"] < 5


class TestVersionIncrementSafe:
    """Test the thread-safe version increment function."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.version_file = self.temp_dir / "version.json"
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_increment_new_file(self):
        """Test version increment on new file."""
        version = _increment_version_safe(self.version_file)
        
        assert version == 1
        assert self.version_file.exists()
        
        with open(self.version_file) as f:
            data = json.load(f)
        assert data["version"] == 1
    
    def test_increment_existing_file(self):
        """Test version increment on existing file."""
        # Create initial version file
        self.version_file.write_text(json.dumps({"version": 3}))
        
        version = _increment_version_safe(self.version_file)
        
        assert version == 4
        
        with open(self.version_file) as f:
            data = json.load(f)
        assert data["version"] == 4
    
    def test_increment_corrupted_file(self):
        """Test version increment on corrupted file."""
        # Create corrupted JSON file
        self.version_file.write_text("invalid json content")
        
        version = _increment_version_safe(self.version_file)
        
        assert version == 1  # Should reset to 1
        
        with open(self.version_file) as f:
            data = json.load(f)
        assert data["version"] == 1
    
    def test_concurrent_version_increments(self):
        """Test concurrent version increments."""
        results = []
        
        def worker():
            try:
                version = _increment_version_safe(self.version_file)
                results.append(version)
            except Exception as e:
                results.append(f"error: {e}")
        
        # Start multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should have 10 unique version numbers
        assert len(results) == 10
        assert all(isinstance(r, int) for r in results)
        assert sorted(results) == list(range(1, 11))  # 1 through 10


class TestIntegrationRaceConditions:
    """Integration tests for race condition protection in retraining."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.train.weekly_retrainer.version_data_with_dvc')
    @patch('src.train.weekly_retrainer.load_new_la_liga_data')
    def test_concurrent_data_loading(self, mock_load_data, mock_version_dvc):
        """Test concurrent data loading with race condition protection."""
        # Mock data loading
        mock_load_data.return_value = Mock()
        mock_version_dvc.return_value = Mock()
        
        results = []
        
        def worker(worker_id):
            try:
                # This would normally be called in the retraining pipeline
                with RetrainingLock(self.temp_dir / "data_load.lock", timeout=2.0):
                    # Simulate data loading
                    time.sleep(0.1)
                    results.append(f"worker_{worker_id}_completed")
            except Exception as e:
                results.append(f"worker_{worker_id}_error: {e}")
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # All workers should complete successfully
        assert len(results) == 3
        assert all("_completed" in r for r in results)
    
    def test_checkpoint_atomic_operations(self):
        """Test atomic checkpoint operations."""
        checkpoint_file = self.temp_dir / "checkpoint.pt"
        version_file = self.temp_dir / "version.json"
        
        def worker(worker_id):
            try:
                # Simulate checkpoint saving with version increment
                with AtomicFileWriter(checkpoint_file) as writer:
                    writer.write(f"checkpoint_data_worker_{worker_id}")
                
                version = _increment_version_safe(version_file)
                return f"worker_{worker_id}_version_{version}"
            except Exception as e:
                return f"worker_{worker_id}_error: {e}"
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Checkpoint file should exist and be valid
        assert checkpoint_file.exists()
        
        # Version file should have incremented properly
        assert version_file.exists()
        with open(version_file) as f:
            version_data = json.load(f)
        assert version_data["version"] == 5  # 5 workers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])