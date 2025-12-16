#!/usr/bin/env python3
"""
Security and optimization enhancement tool for FIFA Soccer DS Analytics.

This script applies security hardening, input validation, and performance optimizations.
"""

import hashlib
import json
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class SecurityOptimizer:
    """Security hardening and optimization tool."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.security_report: Dict[str, Any] = {
            "timestamp": time.time(),
            "vulnerabilities": [],
            "fixes_applied": [],
            "optimizations": [],
            "recommendations": []
        }
    
    def scan_security_issues(self) -> List[str]:
        """Scan for potential security issues."""
        print("🔍 Scanning for security issues...")
        issues = []
        
        # Check for hardcoded secrets
        secrets_pattern = re.compile(r'(password|secret|key|token)\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
        
        for file_path in self.project_root.rglob("*.py"):
            try:
                content = file_path.read_text()
                matches = secrets_pattern.findall(content)
                if matches:
                    issues.append(f"Potential hardcoded secrets in {file_path}")
            except Exception:
                continue
        
        # Check for SQL injection patterns
        sql_pattern = re.compile(r'["\'].*%.*["\']\s*%\s*', re.IGNORECASE)
        for file_path in self.project_root.rglob("*.py"):
            try:
                content = file_path.read_text()
                if sql_pattern.search(content):
                    issues.append(f"Potential SQL injection risk in {file_path}")
            except Exception:
                continue
        
        # Check for path traversal
        path_pattern = re.compile(r'open\s*\(\s*[^)]*\.\.[/\\]', re.IGNORECASE)
        for file_path in self.project_root.rglob("*.py"):
            try:
                content = file_path.read_text()
                if path_pattern.search(content):
                    issues.append(f"Potential path traversal in {file_path}")
            except Exception:
                continue
        
        self.security_report["vulnerabilities"] = issues
        return issues
    
    def apply_input_validation(self) -> List[str]:
        """Apply input validation improvements."""
        print("🛡️ Applying input validation...")
        fixes = []
        
        # Enhanced input validation for pipeline_full.py
        pipeline_file = self.project_root / "src" / "pipeline_full.py"
        if pipeline_file.exists():
            content = pipeline_file.read_text()
            
            # Add input sanitization functions
            validation_code = '''
def sanitize_input(value: str, max_length: int = 255) -> str:
    """Sanitize user input to prevent injection attacks."""
    if not isinstance(value, str):
        raise ValueError("Input must be a string")
    
    # Remove null bytes and control characters
    sanitized = ''.join(char for char in value if ord(char) >= 32 or char in '\\t\\n\\r')
    
    # Limit length
    if len(sanitized) > max_length:
        raise ValueError(f"Input exceeds maximum length of {max_length}")
    
    return sanitized

def validate_path(path: str, base_dir: Optional[Path] = None) -> Path:
    """Validate and sanitize file paths."""
    if not isinstance(path, str):
        raise ValueError("Path must be a string")
    
    # Remove null bytes
    path = path.replace('\\0', '')
    
    # Convert to Path object
    path_obj = Path(path)
    
    # Resolve to absolute path
    try:
        path_obj = path_obj.resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid path: {e}")
    
    # Check if path is within base directory if specified
    if base_dir and not str(path_obj).startswith(str(base_dir.resolve())):
        raise ValueError("Path traversal attempt detected")
    
    return path_obj

def validate_confidence(value: float) -> float:
    """Validate confidence threshold."""
    if not isinstance(value, (int, float)):
        raise ValueError("Confidence must be a number")
    
    if not 0.0 <= value <= 1.0:
        raise ValueError("Confidence must be between 0.0 and 1.0")
    
    return float(value)

'''
            
            if "def sanitize_input" not in content:
                # Insert validation functions after imports
                import_end = content.find("\\n\\n")
                if import_end != -1:
                    new_content = content[:import_end] + validation_code + content[import_end:]
                    pipeline_file.write_text(new_content)
                    fixes.append("Added input validation functions")
        
        self.security_report["fixes_applied"].extend(fixes)
        return fixes
    
    def enhance_error_handling(self) -> List[str]:
        """Enhance error handling with security considerations."""
        print("⚠️ Enhancing error handling...")
        fixes = []
        
        # Enhance error handling in pipeline_full.py
        pipeline_file = self.project_root / "src" / "pipeline_full.py"
        if pipeline_file.exists():
            content = pipeline_file.read_text()
            
            # Add secure error handling
            secure_error_handler = '''
def secure_error_handler(func):
    """Decorator for secure error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            # Don't expose file paths in error messages
            LOGGER.error("Required file not found")
            raise RuntimeError("Required file not found") from e
        except PermissionError:
            LOGGER.error("Permission denied accessing required resources")
            raise RuntimeError("Permission denied") from None
        except Exception as e:
            # Log full error for debugging but don't expose to user
            LOGGER.error(f"Processing error: {type(e).__name__}")
            raise RuntimeError("Processing failed") from e
    return wrapper

'''
            
            if "def secure_error_handler" not in content:
                import_end = content.find("\\n\\n")
                if import_end != -1:
                    new_content = content[:import_end] + secure_error_handler + content[import_end:]
                    pipeline_file.write_text(new_content)
                    fixes.append("Added secure error handling")
        
        self.security_report["fixes_applied"].extend(fixes)
        return fixes
    
    def optimize_memory_usage(self) -> List[str]:
        """Optimize memory usage patterns."""
        print("💾 Optimizing memory usage...")
        optimizations = []
        
        # Add memory optimization utilities
        memory_utils = '''
import gc
from contextlib import contextmanager

@contextmanager
def memory_monitor(name: str):
    """Monitor memory usage for a code section."""
    import psutil
    process = psutil.Process()
    start_memory = process.memory_info().rss
    start_time = time.time()
    
    try:
        yield
    finally:
        end_memory = process.memory_info().rss
        end_time = time.time()
        memory_delta = (end_memory - start_memory) / 1024 / 1024
        duration = end_time - start_time
        
        LOGGER.info(f"{name}: {memory_delta:.1f}MB in {duration:.2f}s")
        
        # Force garbage collection if memory usage is high
        if memory_delta > 100:  # More than 100MB
            gc.collect()

def optimize_large_dataset_processing(data_loader, batch_size: int = 32):
    """Process large datasets with memory optimization."""
    for batch_idx in range(0, len(data_loader), batch_size):
        batch = data_loader[batch_idx:batch_idx + batch_size]
        yield batch
        
        # Periodic garbage collection
        if batch_idx % (batch_size * 10) == 0:
            gc.collect()

'''
        
        utils_file = self.project_root / "src" / "utils" / "memory_optimization.py"
        if not utils_file.exists():
            utils_file.parent.mkdir(parents=True, exist_ok=True)
            utils_file.write_text(memory_utils)
            optimizations.append("Added memory optimization utilities")
        
        self.security_report["optimizations"].extend(optimizations)
        return optimizations
    
    def add_caching_strategies(self) -> List[str]:
        """Add intelligent caching strategies."""
        print("🚀 Adding caching strategies...")
        optimizations = []
        
        # Add caching utilities
        caching_code = '''
import hashlib
import json
import pickle
from functools import wraps
from pathlib import Path
from typing import Any, Optional

class SmartCache:
    """Intelligent caching with size limits and expiration."""
    
    def __init__(self, cache_dir: Path, max_size_mb: int = 512):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._current_size = self._calculate_cache_size()
    
    def _calculate_cache_size(self) -> int:
        """Calculate current cache size."""
        total_size = 0
        for file_path in self.cache_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from function arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set cached value with size management."""
        cache_file = self.cache_dir / f"{key}.pkl"
        
        # Check cache size limit
        if self._current_size > self.max_size_bytes:
            self._cleanup_cache()
        
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(value, f)
            self._current_size += cache_file.stat().st_size
        except Exception:
            pass  # Cache write failed, continue without caching
    
    def _cleanup_cache(self) -> None:
        """Clean up old cache files."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        cache_files.sort(key=lambda x: x.stat().st_mtime)  # Sort by modification time
        
        # Remove oldest 25% of cache files
        files_to_remove = cache_files[:len(cache_files) // 4]
        for file_path in files_to_remove:
            try:
                file_size = file_path.stat().st_size
                file_path.unlink()
                self._current_size -= file_size
            except Exception:
                pass

def cached_function(cache_instance: SmartCache, ttl_seconds: int = 3600):
    """Decorator for cached function execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = cache_instance._get_cache_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_instance.set(cache_key, result)
            
            return result
        return wrapper
    return decorator

# Global cache instance
_global_cache = SmartCache(Path("cache/global"), max_size_mb=256)

'''
        
        cache_file = self.project_root / "src" / "utils" / "smart_cache.py"
        if not cache_file.exists():
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(caching_code)
            optimizations.append("Added intelligent caching system")
        
        self.security_report["optimizations"].extend(optimizations)
        return optimizations
    
    def generate_security_report(self) -> None:
        """Generate comprehensive security report."""
        report_path = self.project_root / "outputs" / "security_report.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, "w") as f:
            json.dump(self.security_report, f, indent=2)
        
        print(f"\n{'='*80}")
        print("SECURITY & OPTIMIZATION REPORT")
        print(f"{'='*80}")
        print(f"Vulnerabilities Found: {len(self.security_report['vulnerabilities'])}")
        print(f"Fixes Applied: {len(self.security_report['fixes_applied'])}")
        print(f"Optimizations Added: {len(self.security_report['optimizations'])}")
        
        if self.security_report["vulnerabilities"]:
            print(f"\n⚠️ Security Issues:")
            for issue in self.security_report["vulnerabilities"]:
                print(f"  - {issue}")
        
        if self.security_report["fixes_applied"]:
            print(f"\n✅ Security Fixes:")
            for fix in self.security_report["fixes_applied"]:
                print(f"  - {fix}")
        
        if self.security_report["optimizations"]:
            print(f"\n🚀 Optimizations:")
            for opt in self.security_report["optimizations"]:
                print(f"  - {opt}")
        
        print(f"\nReport saved to: {report_path}")
    
    def run_security_optimization(self) -> None:
        """Run complete security optimization."""
        print("🔐 Starting Security & Optimization Enhancement")
        
        # Run security scans and fixes
        self.scan_security_issues()
        self.apply_input_validation()
        self.enhance_error_handling()
        
        # Apply optimizations
        self.optimize_memory_usage()
        self.add_caching_strategies()
        
        # Generate report
        self.generate_security_report()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Security and optimization enhancement")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory"
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Only scan for security issues"
    )
    parser.add_argument(
        "--optimize-only",
        action="store_true",
        help="Only apply optimizations"
    )
    
    args = parser.parse_args()
    
    optimizer = SecurityOptimizer(args.project_root)
    
    if args.scan_only:
        issues = optimizer.scan_security_issues()
        print(f"Found {len(issues)} security issues")
    elif args.optimize_only:
        optimizer.optimize_memory_usage()
        optimizer.add_caching_strategies()
    else:
        optimizer.run_security_optimization()


if __name__ == "__main__":
    main()