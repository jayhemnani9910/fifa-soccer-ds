#!/usr/bin/env python3
"""
Simple test script for semantic versioning and model conflict resolution.
"""

import json
import re
import tempfile
import time
from pathlib import Path


class SemanticVersion:
    """Simplified semantic versioning for model releases."""
    
    def __init__(self, major: int = 1, minor: int = 0, patch: int = 0):
        self.major = major
        self.minor = minor
        self.patch = patch
    
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def __lt__(self, other):
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
    
    def increment_patch(self) -> "SemanticVersion":
        return SemanticVersion(self.major, self.minor, self.patch + 1)
    
    def increment_minor(self) -> "SemanticVersion":
        return SemanticVersion(self.major, self.minor + 1, 0)
    
    def increment_major(self) -> "SemanticVersion":
        return SemanticVersion(self.major + 1, 0, 0)
    
    @classmethod
    def from_string(cls, version_str: str) -> "SemanticVersion":
        match = re.match(r'^(\d+)\.(\d+)\.(\d+)$', version_str.strip())
        if not match:
            raise ValueError(f"Invalid version format: {version_str}")
        return cls(int(match.group(1)), int(match.group(2)), int(match.group(3)))


class SimpleAtomicFileWriter:
    """Simplified atomic file writer."""
    
    def __init__(self, target_file: Path):
        self.target_file = target_file
        self.temp_file = None
        self.file_handle = None
    
    def __enter__(self):
        self.temp_file = self.target_file.with_suffix('.tmp')
        self.file_handle = open(self.temp_file, 'w')
        return self.file_handle
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file_handle.close()
        
        if exc_type is None and self.temp_file:
            self.temp_file.rename(self.target_file)
        elif self.temp_file and self.temp_file.exists():
            self.temp_file.unlink()


class SimpleModelVersionManager:
    """Simplified model version manager."""
    
    def __init__(self, version_file: Path, primary_metric: str = "mAP@0.5"):
        self.version_file = version_file
        self.primary_metric = primary_metric
        self.versions_file = version_file.parent / "versions_history.json"
    
    def get_current_version(self) -> SemanticVersion:
        if self.version_file.exists():
            try:
                data = json.loads(self.version_file.read_text())
                version_str = data.get("semantic_version", "1.0.0")
                return SemanticVersion.from_string(version_str)
            except (json.JSONDecodeError, ValueError, KeyError):
                return SemanticVersion(1, 0, 0)
        return SemanticVersion(1, 0, 0)
    
    def increment_version(self, version_type: str = "patch") -> SemanticVersion:
        current = self.get_current_version()
        
        if version_type == "major":
            new_version = current.increment_major()
        elif version_type == "minor":
            new_version = current.increment_minor()
        else:
            new_version = current.increment_patch()
        
        # Atomic update
        with SimpleAtomicFileWriter(self.version_file) as writer:
            writer.write(json.dumps({
                "semantic_version": str(new_version),
                "version_type": version_type,
                "timestamp": time.time(),
                "previous_version": str(current)
            }, indent=2))
        
        return new_version
    
    def get_version_info(self) -> dict:
        current = self.get_current_version()
        return {
            "current_version": str(current),
            "version_components": {
                "major": current.major,
                "minor": current.minor,
                "patch": current.patch
            },
            "version_file": str(self.version_file)
        }


def test_semantic_version():
    """Test SemanticVersion functionality."""
    print("Testing SemanticVersion...")
    
    # Test creation
    v = SemanticVersion(1, 2, 3)
    assert str(v) == "1.2.3"
    print("✓ Version creation works")
    
    # Test parsing
    v_parsed = SemanticVersion.from_string("2.5.1")
    assert v_parsed.major == 2 and v_parsed.minor == 5 and v_parsed.patch == 1
    print("✓ Version parsing works")
    
    # Test increments
    base = SemanticVersion(1, 2, 3)
    patch = base.increment_patch()
    minor = base.increment_minor()
    major = base.increment_major()
    
    assert str(patch) == "1.2.4"
    assert str(minor) == "1.3.0"
    assert str(major) == "2.0.0"
    print("✓ Version increments work")
    
    # Test comparison
    v1 = SemanticVersion(1, 0, 0)
    v2 = SemanticVersion(1, 0, 1)
    assert v1 < v2
    print("✓ Version comparison works")


def test_version_manager():
    """Test ModelVersionManager functionality."""
    print("Testing ModelVersionManager...")
    
    temp_dir = Path(tempfile.mkdtemp())
    version_file = temp_dir / "semantic_version.json"
    
    try:
        manager = SimpleModelVersionManager(version_file)
        
        # Test initial version
        current = manager.get_current_version()
        assert str(current) == "1.0.0"
        print("✓ Initial version correct")
        
        # Test patch increment
        v1 = manager.increment_version("patch")
        assert str(v1) == "1.0.1"
        
        # Verify file was updated
        with open(version_file) as f:
            data = json.load(f)
        assert data["semantic_version"] == "1.0.1"
        assert data["version_type"] == "patch"
        print("✓ Patch increment works")
        
        # Test minor increment
        v2 = manager.increment_version("minor")
        assert str(v2) == "1.1.0"
        print("✓ Minor increment works")
        
        # Test major increment
        v3 = manager.increment_version("major")
        assert str(v3) == "2.0.0"
        print("✓ Major increment works")
        
        # Test version info
        info = manager.get_version_info()
        assert info["current_version"] == "2.0.0"
        assert info["version_components"]["major"] == 2
        print("✓ Version info works")
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_version_lifecycle():
    """Test complete version lifecycle."""
    print("Testing version lifecycle...")
    
    temp_dir = Path(tempfile.mkdtemp())
    version_file = temp_dir / "semantic_version.json"
    
    try:
        manager = SimpleModelVersionManager(version_file)
        
        # Simulate development lifecycle
        print("  Starting development...")
        v0 = manager.get_current_version()
        print(f"  Initial version: {v0}")
        
        # Bug fix
        v1 = manager.increment_version("patch")
        print(f"  Bug fix: {v1}")
        
        # New feature
        v2 = manager.increment_version("minor")
        print(f"  New feature: {v2}")
        
        # Another bug fix
        v3 = manager.increment_version("patch")
        print(f"  Bug fix: {v3}")
        
        # Breaking change
        v4 = manager.increment_version("major")
        print(f"  Breaking change: {v4}")
        
        # Verify final state
        final = manager.get_current_version()
        assert str(final) == "2.0.0"
        print(f"  Final version: {final}")
        
        print("✓ Version lifecycle complete")
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_atomic_file_operations():
    """Test atomic file operations."""
    print("Testing atomic file operations...")
    
    temp_dir = Path(tempfile.mkdtemp())
    target_file = temp_dir / "test.json"
    
    try:
        # Test successful write
        test_data = {"version": 1, "data": "test"}
        
        with SimpleAtomicFileWriter(target_file) as writer:
            writer.write(json.dumps(test_data))
        
        assert target_file.exists()
        with open(target_file) as f:
            loaded = json.load(f)
        assert loaded == test_data
        print("✓ Atomic write successful")
        
        # Test failed write (no temp file should remain)
        try:
            with SimpleAtomicFileWriter(target_file) as writer:
                writer.write("invalid json")
                raise ValueError("Simulated error")
        except ValueError:
            pass
        
        # Original file should still exist and be valid
        assert target_file.exists()
        with open(target_file) as f:
            loaded = json.load(f)
        assert loaded == test_data
        print("✓ Failed write cleanup works")
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_version_comparison_logic():
    """Test version comparison scenarios."""
    print("Testing version comparison logic...")
    
    # Test basic comparisons
    v1 = SemanticVersion(1, 0, 0)
    v2 = SemanticVersion(1, 0, 1)
    v3 = SemanticVersion(1, 1, 0)
    v4 = SemanticVersion(2, 0, 0)
    
    assert v1 < v2 < v3 < v4
    print("✓ Version ordering works")
    
    # Test equal versions
    v5 = SemanticVersion(1, 2, 3)
    v6 = SemanticVersion(1, 2, 3)
    assert not (v5 < v6 or v6 < v5)
    print("✓ Version equality works")
    
    # Test string parsing edge cases
    try:
        SemanticVersion.from_string("invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    try:
        SemanticVersion.from_string("1.2")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    print("✓ Edge case handling works")


def main():
    """Run all version management tests."""
    print("🏷️  Version Management Tests\n")
    
    try:
        test_semantic_version()
        print()
        
        test_atomic_file_operations()
        print()
        
        test_version_manager()
        print()
        
        test_version_lifecycle()
        print()
        
        test_version_comparison_logic()
        print()
        
        print("🎉 All version management tests passed!")
        return 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())