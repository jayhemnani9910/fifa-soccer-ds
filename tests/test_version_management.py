#!/usr/bin/env python3
"""
Test suite for semantic versioning and model conflict resolution.

This test validates the SemanticVersion, ModelVersionConflict, and ModelVersionManager
classes to ensure proper version management and conflict resolution.
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import classes for testing
from train.weekly_retrainer import (
    SemanticVersion,
    ModelVersionConflict,
    ModelVersionManager,
    AtomicFileWriter
)


class TestSemanticVersion:
    """Test the SemanticVersion class."""
    
    def test_version_creation(self):
        """Test basic version creation and string representation."""
        v = SemanticVersion(1, 2, 3)
        assert str(v) == "1.2.3"
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3
    
    def test_version_from_string(self):
        """Test parsing version from string."""
        v = SemanticVersion.from_string("2.5.1")
        assert v.major == 2
        assert v.minor == 5
        assert v.patch == 1
    
    def test_version_from_string_invalid(self):
        """Test parsing invalid version strings."""
        with pytest.raises(ValueError):
            SemanticVersion.from_string("invalid")
        with pytest.raises(ValueError):
            SemanticVersion.from_string("1.2")
        with pytest.raises(ValueError):
            SemanticVersion.from_string("1.2.3.4")
    
    def test_version_increments(self):
        """Test version increment methods."""
        v = SemanticVersion(1, 2, 3)
        
        patch_v = v.increment_patch()
        assert str(patch_v) == "1.2.4"
        
        minor_v = v.increment_minor()
        assert str(minor_v) == "1.3.0"
        
        major_v = v.increment_major()
        assert str(major_v) == "2.0.0"
    
    def test_version_to_dict(self):
        """Test conversion to dictionary."""
        v = SemanticVersion(1, 2, 3)
        expected = {"major": 1, "minor": 2, "patch": 3}
        assert v.to_dict() == expected


class TestModelVersionConflict:
    """Test the ModelVersionConflict class."""
    
    def test_conflict_creation(self):
        """Test basic conflict creation."""
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 0, 1)
        metrics_a = {"mAP@0.5": 0.85, "F1": 0.82}
        metrics_b = {"mAP@0.5": 0.83, "F1": 0.80}
        
        conflict = ModelVersionConflict(v1, v2, metrics_a, metrics_b)
        
        assert conflict.version_a == v1
        assert conflict.version_b == v2
        assert conflict.metrics_a == metrics_a
        assert conflict.metrics_b == metrics_b
        assert conflict.resolution == "auto"
    
    def test_metrics_comparison(self):
        """Test metrics comparison logic."""
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 0, 1)
        metrics_a = {"mAP@0.5": 0.85, "F1": 0.82}
        metrics_b = {"mAP@0.5": 0.83, "F1": 0.80}
        
        conflict = ModelVersionConflict(v1, v2, metrics_a, metrics_b)
        
        # Test comparison
        comparison, diff = conflict.compare_metrics()
        assert comparison == "a_better"
        assert abs(diff - 0.02) < 0.001
        
        # Test equal performance
        metrics_c = {"mAP@0.5": 0.85, "F1": 0.82}
        conflict_equal = ModelVersionConflict(v1, v2, metrics_a, metrics_c)
        
        comparison, diff = conflict_equal.compare_metrics()
        assert comparison == "equal"
        assert diff == 0.0
    
    def test_auto_resolution(self):
        """Test automatic conflict resolution."""
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 0, 1)
        metrics_a = {"mAP@0.5": 0.85, "F1": 0.82}
        metrics_b = {"mAP@0.5": 0.87, "F1": 0.84}  # Better metrics
        
        conflict = ModelVersionConflict(v1, v2, metrics_a, metrics_b)
        
        # Auto-resolve
        resolved = conflict.auto_resolve()
        
        assert resolved == v2  # Should choose version B
        assert conflict.resolution == "auto_b"
        assert "better" in conflict.conflict_reason
        assert conflict.resolved_version == v2


class TestModelVersionManager:
    """Test the ModelVersionManager class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.version_file = self.temp_dir / "semantic_version.json"
        self.manager = ModelVersionManager(self.version_file)
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initial_version(self):
        """Test initial version when no file exists."""
        version = self.manager.get_current_version()
        assert str(version) == "1.0.0"
    
    def test_version_increment_patch(self):
        """Test patch version increment."""
        # Create initial version
        with AtomicFileWriter(self.version_file) as writer:
            writer.write(json.dumps({"semantic_version": "1.2.3"}))
        
        # Increment patch
        new_version = self.manager.increment_version("patch")
        assert str(new_version) == "1.2.4"
        
        # Verify file was updated
        with open(self.version_file) as f:
            data = json.load(f)
        assert data["semantic_version"] == "1.2.4"
        assert data["version_type"] == "patch"
    
    def test_version_increment_minor(self):
        """Test minor version increment."""
        # Create initial version
        with AtomicFileWriter(self.version_file) as writer:
            writer.write(json.dumps({"semantic_version": "1.2.3"}))
        
        # Increment minor
        new_version = self.manager.increment_version("minor")
        assert str(new_version) == "1.3.0"
    
    def test_version_increment_major(self):
        """Test major version increment."""
        # Create initial version
        with AtomicFileWriter(self.version_file) as writer:
            writer.write(json.dumps({"semantic_version": "1.2.3"}))
        
        # Increment major
        new_version = self.manager.increment_version("major")
        assert str(new_version) == "2.0.0"
    
    def test_version_history(self):
        """Test version history tracking."""
        # Increment a few versions
        v1 = self.manager.increment_version("patch")
        v2 = self.manager.increment_version("patch")
        v3 = self.manager.increment_version("minor")
        
        # Check history file
        history_file = self.manager.versions_file
        assert history_file.exists()
        
        with open(history_file) as f:
            history = json.load(f)
        
        assert len(history["versions"]) == 3
        assert history["versions"][0]["semantic_version"] == str(v1)
        assert history["versions"][1]["semantic_version"] == str(v2)
        assert history["versions"][2]["semantic_version"] == str(v3)
    
    def test_conflict_detection(self):
        """Test conflict detection with similar performance."""
        # Create version history
        v1 = self.manager.increment_version("patch")
        
        # Add history entry with metrics
        history_file = self.manager.versions_file
        with open(history_file) as f:
            history = json.load(f)
        
        history["versions"][-1]["metrics"] = {"mAP@0.5": 0.85, "F1": 0.82}
        
        with AtomicFileWriter(history_file) as writer:
            writer.write(json.dumps(history))
        
        # Test conflict detection with similar metrics
        new_metrics = {"mAP@0.5": 0.852, "F1": 0.821}  # Very similar
        
        conflict = self.manager.detect_conflicts(new_metrics)
        
        assert conflict is not None
        assert conflict.resolution == "auto_equal" or conflict.resolution in ["auto_a", "auto_b"]
        assert conflict.resolved_version is not None
    
    def test_get_version_info(self):
        """Test getting comprehensive version information."""
        # Increment a version
        self.manager.increment_version("minor")
        
        info = self.manager.get_version_info()
        
        assert "current_version" in info
        assert "version_components" in info
        assert "version_file" in info
        assert "history_file" in info
        assert info["current_version"] == "1.1.0"
        assert info["version_components"] == {"major": 1, "minor": 1, "patch": 0}


class TestVersionManagementIntegration:
    """Integration tests for version management."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.version_file = self.temp_dir / "semantic_version.json"
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_version_lifecycle(self):
        """Test complete version lifecycle."""
        manager = ModelVersionManager(self.version_file)
        
        # Start with initial version
        assert str(manager.get_current_version()) == "1.0.0"
        
        # Bug fix - patch increment
        v1 = manager.increment_version("patch")
        assert str(v1) == "1.0.1"
        
        # New feature - minor increment
        v2 = manager.increment_version("minor")
        assert str(v2) == "1.1.0"
        
        # Breaking change - major increment
        v3 = manager.increment_version("major")
        assert str(v3) == "2.0.0"
        
        # Verify history
        info = manager.get_version_info()
        assert info["total_versions"] == 3
        assert info["current_version"] == "2.0.0"
    
    def test_conflict_resolution_workflow(self):
        """Test complete conflict resolution workflow."""
        manager = ModelVersionManager(self.version_file)
        
        # Create initial version with metrics
        v1 = manager.increment_version("minor")
        
        # Simulate version history with metrics
        history_file = manager.versions_file
        with open(history_file) as f:
            history = json.load(f)
        
        history["versions"][-1]["metrics"] = {"mAP@0.5": 0.80, "F1": 0.78}
        
        with AtomicFileWriter(history_file) as writer:
            writer.write(json.dumps(history))
        
        # New training with slightly better performance
        new_metrics = {"mAP@0.5": 0.805, "F1": 0.785}
        
        # Detect and resolve conflict
        conflict = manager.detect_conflicts(new_metrics)
        assert conflict is not None
        
        resolved_version = conflict.auto_resolve()
        assert resolved_version is not None
        
        # Should auto-resolve to newer version if similar performance
        assert conflict.resolution in ["auto_equal", "auto_a", "auto_b"]
        
        # Log resolution for audit
        print(f"Conflict resolved: {conflict.conflict_reason}")
        print(f"Resolution: {conflict.resolution}")
        print(f"Resolved version: {resolved_version}")


if __name__ == "__main__":
    import pytest
    
    # Run tests
    pytest.main([__file__, "-v"])