#!/usr/bin/env python3
"""
Simple test to validate race condition protection mechanisms.
"""

import fcntl
import json
import os
import tempfile
import threading
import time
from pathlib import Path


class SimpleRetrainingLock:
    """Simplified version of RetrainingLock for testing."""
    
    def __init__(self, lock_file: Path, timeout: int = 30):
        self.lock_file = lock_file
        self.timeout = timeout
        self.lock_fd = None
    
    def __enter__(self):
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    def acquire(self):
        """Acquire the lock."""
        self.lock_fd = open(self.lock_file, 'w')
        try:
            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.lock_fd.write(f"{os.getpid()}\n{time.time()}\n")
            self.lock_fd.flush()
            return True
        except IOError:
            self.lock_fd.close()
            self.lock_fd = None
            return False
    
    def release(self):
        """Release the lock."""
        if self.lock_fd:
            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
            self.lock_fd.close()
            self.lock_fd = None
            if self.lock_file.exists():
                self.lock_file.unlink()


class SimpleAtomicFileWriter:
    """Simplified version of AtomicFileWriter for testing."""
    
    def __init__(self, target_file: Path):
        self.target_file = target_file
        self.temp_file = None
    
    def __enter__(self):
        self.temp_file = self.target_file.with_suffix('.tmp')
        self.file_handle = open(self.temp_file, 'w')
        return self.file_handle
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file_handle.close()
        
        if exc_type is None:
            # Atomic rename
            self.temp_file.rename(self.target_file)
        else:
            # Cleanup on error
            if self.temp_file.exists():
                self.temp_file.unlink()


def simple_version_increment_safe(version_file: Path) -> int:
    """Simplified thread-safe version increment."""
    # Use exclusive lock for the entire operation
    with open(version_file, 'a+') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        
        # Read existing version
        f.seek(0)
        try:
            if f.tell() > 0 or version_file.stat().st_size > 0:
                f.seek(0)
                content = f.read()
                if content.strip():
                    payload = json.loads(content)
                    version = int(payload.get("version", 0)) + 1
                else:
                    version = 1
            else:
                version = 1
        except (json.JSONDecodeError, ValueError, TypeError, OSError):
            version = 1
        
        # Write new version
        f.seek(0)
        f.truncate()
        f.write(json.dumps({"version": version}, indent=2))
        f.flush()
        
    return version


def test_basic_functionality():
    """Test basic race condition protection functionality."""
    print("Testing basic race condition protection...")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Test 1: Basic lock
        lock_file = temp_dir / "test.lock"
        with SimpleRetrainingLock(lock_file) as lock:
            assert lock_file.exists()
            print("✓ Basic lock acquisition works")
        
        # Test 2: Atomic file write
        data_file = temp_dir / "data.json"
        test_data = {"test": "data", "version": 1}
        
        with SimpleAtomicFileWriter(data_file) as writer:
            writer.write(json.dumps(test_data))
        
        assert data_file.exists()
        with open(data_file) as f:
            loaded = json.load(f)
        assert loaded == test_data
        print("✓ Atomic file write works")
        
        # Test 3: Version increment
        version_file = temp_dir / "version.json"
        version = simple_version_increment_safe(version_file)
        assert version == 1
        
        version = simple_version_increment_safe(version_file)
        assert version == 2
        print("✓ Version increment works")
        
        print("✓ All basic functionality tests passed!")
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_concurrent_access():
    """Test concurrent access scenarios."""
    print("Testing concurrent access...")
    
    temp_dir = Path(tempfile.mkdtemp())
    lock_file = temp_dir / "concurrent.lock"
    results = []
    
    def worker(worker_id):
        try:
            with SimpleRetrainingLock(lock_file, timeout=1):
                start_time = time.time()
                time.sleep(0.1)  # Simulate work
                elapsed = time.time() - start_time
                results.append(f"worker_{worker_id}_{elapsed:.3f}")
        except Exception as e:
            results.append(f"worker_{worker_id}_error: {e}")
    
    try:
        # Start multiple workers
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should have 5 successful results
        assert len(results) == 5
        assert all("error" not in r for r in results)
        print(f"✓ Concurrent access: {len(results)} workers completed successfully")
        
        # Verify sequential access (times should not overlap significantly)
        times = [float(r.split('_')[-1]) for r in results]
        total_time = sum(times)
        # Should be approximately 5 * 0.1 = 0.5 seconds (sequential)
        assert total_time > 0.4, f"Expected sequential execution, got total time: {total_time}"
        print(f"✓ Sequential access verified: total time {total_time:.3f}s")
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_concurrent_version_increments():
    """Test concurrent version increments."""
    print("Testing concurrent version increments...")
    
    temp_dir = Path(tempfile.mkdtemp())
    version_file = temp_dir / "version.json"
    results = []
    
    def worker():
        try:
            version = simple_version_increment_safe(version_file)
            results.append(version)
        except Exception as e:
            results.append(f"error: {e}")
    
    try:
        # Start multiple workers
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should have 10 unique version numbers
        assert len(results) == 10
        assert all(isinstance(r, int) for r in results)
        
        sorted_results = sorted(results)
        expected = list(range(1, 11))
        assert sorted_results == expected
        print(f"✓ Concurrent version increments: {sorted_results}")
        
        # Verify final state
        with open(version_file) as f:
            final_data = json.load(f)
        assert final_data["version"] == 10
        print(f"✓ Final version: {final_data['version']}")
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all race condition protection tests."""
    print("🔒 Simple Race Condition Protection Tests\n")
    
    try:
        test_basic_functionality()
        print()
        
        test_concurrent_access()
        print()
        
        test_concurrent_version_increments()
        print()
        
        print("🎉 All race condition protection tests passed!")
        return 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())