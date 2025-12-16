#!/usr/bin/env python3
"""
Simple test script to validate race condition protection in weekly retraining.
"""

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train.weekly_retrainer import RetrainingLock, AtomicFileWriter, _increment_version_safe


def test_retraining_lock():
    """Test RetrainingLock functionality."""
    print("Testing RetrainingLock...")
    
    temp_dir = Path(tempfile.mkdtemp())
    lock_file = temp_dir / "test.lock"
    
    # Test basic lock acquisition
    lock = RetrainingLock(lock_file, timeout=1)
    
    try:
        with lock:
            print("✓ Lock acquired successfully")
            assert lock_file.exists(), "Lock file should exist"
            
            # Test concurrent access
            results = []
            
            def worker(worker_id):
                try:
                    with RetrainingLock(lock_file, timeout=0.5):
                        results.append(f"worker_{worker_id}_success")
                        time.sleep(0.1)
                except Exception as e:
                    results.append(f"worker_{worker_id}_failed: {e}")
            
            # Start concurrent workers
            threads = []
            for i in range(3):
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Should have one success (the main thread) and failures from others
            success_count = sum(1 for r in results if "success" in r)
            print(f"✓ Concurrent access test: {success_count} succeeded, {len(results) - success_count} failed")
            
    finally:
        # Cleanup
        if lock_file.exists():
            lock_file.unlink()
        temp_dir.rmdir()


def test_atomic_file_writer():
    """Test AtomicFileWriter functionality."""
    print("Testing AtomicFileWriter...")
    
    temp_dir = Path(tempfile.mkdtemp())
    target_file = temp_dir / "test.json"
    
    try:
        # Test successful write
        data = {"version": 5, "timestamp": "2024-01-01"}
        
        with AtomicFileWriter(target_file) as writer:
            writer.write(json.dumps(data))
        
        assert target_file.exists(), "Target file should exist"
        
        with open(target_file) as f:
            loaded_data = json.load(f)
        assert loaded_data == data, "Data should match"
        
        print("✓ Atomic write successful")
        
        # Test concurrent writes
        results = []
        
        def worker(worker_id):
            try:
                with AtomicFileWriter(target_file) as writer:
                    writer.write(json.dumps({"worker": worker_id}))
                    time.sleep(0.05)
                return True
            except Exception as e:
                print(f"Worker {worker_id} failed: {e}")
                return False
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Final file should exist and be valid
        assert target_file.exists(), "Final file should exist"
        with open(target_file) as f:
            final_data = json.load(f)
        
        print(f"✓ Concurrent writes completed: {final_data}")
        
    finally:
        # Cleanup
        if target_file.exists():
            target_file.unlink()
        temp_dir.rmdir()


def test_version_increment_safe():
    """Test thread-safe version increment."""
    print("Testing thread-safe version increment...")
    
    temp_dir = Path(tempfile.mkdtemp())
    version_file = temp_dir / "version.json"
    
    try:
        results = []
        
        def worker():
            try:
                version = _increment_version_safe(version_file)
                results.append(version)
            except Exception as e:
                results.append(f"error: {e}")
        
        # Start multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should have 10 unique version numbers
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"
        assert all(isinstance(r, int) for r in results), "All results should be integers"
        
        sorted_results = sorted(results)
        expected = list(range(1, 11))
        assert sorted_results == expected, f"Expected {expected}, got {sorted_results}"
        
        print(f"✓ Thread-safe version increment: {sorted_results}")
        
        # Verify final version file
        with open(version_file) as f:
            final_data = json.load(f)
        assert final_data["version"] == 10, f"Final version should be 10, got {final_data['version']}"
        
    finally:
        # Cleanup
        if version_file.exists():
            version_file.unlink()
        temp_dir.rmdir()


def test_integration():
    """Test integration of all race condition protection mechanisms."""
    print("Testing integration...")
    
    temp_dir = Path(tempfile.mkdtemp())
    lock_file = temp_dir / "integration.lock"
    version_file = temp_dir / "version.json"
    data_file = temp_dir / "data.json"
    
    try:
        results = []
        
        def worker(worker_id):
            try:
                # Use lock to protect critical section
                with RetrainingLock(lock_file, timeout=1):
                    # Write data atomically
                    with AtomicFileWriter(data_file) as writer:
                        worker_data = {"worker": worker_id, "timestamp": time.time()}
                        writer.write(json.dumps(worker_data))
                    
                    # Increment version safely
                    version = _increment_version_safe(version_file)
                    
                    results.append(f"worker_{worker_id}_version_{version}")
                    
            except Exception as e:
                results.append(f"worker_{worker_id}_error: {e}")
        
        # Start multiple workers
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # All workers should complete successfully
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"
        assert all("error" not in r for r in results), "No errors should have occurred"
        
        print(f"✓ Integration test completed: {results}")
        
        # Verify final state
        assert data_file.exists(), "Data file should exist"
        assert version_file.exists(), "Version file should exist"
        
        with open(data_file) as f:
            final_data = json.load(f)
        with open(version_file) as f:
            version_data = json.load(f)
        
        assert version_data["version"] == 5, f"Final version should be 5, got {version_data['version']}"
        print(f"✓ Final data: {final_data}")
        
    finally:
        # Cleanup
        for file in [lock_file, version_file, data_file]:
            if file.exists():
                file.unlink()
        temp_dir.rmdir()


def main():
    """Run all race condition protection tests."""
    print("🔒 Testing Race Condition Protection\n")
    
    try:
        test_retraining_lock()
        print()
        
        test_atomic_file_writer()
        print()
        
        test_version_increment_safe()
        print()
        
        test_integration()
        print()
        
        print("🎉 All race condition protection tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())