#!/usr/bin/env python3
"""
Comprehensive test runner for FIFA Soccer DS Analytics project.

This script runs the full test suite with performance monitoring,
coverage analysis, and detailed reporting.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


class TestRunner:
    """Comprehensive test runner with performance monitoring."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results: Dict[str, Any] = {
            "timestamp": time.time(),
            "tests": [],
            "summary": {},
            "performance": {},
            "coverage": {}
        }
    
    def run_command(self, cmd: List[str], description: str) -> Dict[str, Any]:
        """Run a command and capture results."""
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            duration = time.time() - start_time
            
            return {
                "command": " ".join(cmd),
                "description": description,
                "success": result.returncode == 0,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "command": " ".join(cmd),
                "description": description,
                "success": False,
                "duration": time.time() - start_time,
                "error": "Test execution timeout"
            }
        except Exception as e:
            return {
                "command": " ".join(cmd),
                "description": description,
                "success": False,
                "duration": time.time() - start_time,
                "error": str(e)
            }
    
    def run_import_tests(self) -> Dict[str, Any]:
        """Test package imports."""
        return self.run_command(
            ["python", "-m", "pytest", "tests/test_imports.py", "-v"],
            "Package Import Tests"
        )
    
    def run_smoke_tests(self) -> Dict[str, Any]:
        """Run smoke tests."""
        return self.run_command(
            ["python", "-m", "pytest", "tests/", "-m", "smoke", "-v"],
            "Smoke Tests"
        )
    
    def run_detection_tests(self) -> Dict[str, Any]:
        """Run detection-related tests."""
        return self.run_command(
            ["python", "-m", "pytest", "tests/test_detect_smoke.py", "-v"],
            "Detection Tests"
        )
    
    def run_tracking_tests(self) -> Dict[str, Any]:
        """Run tracking-related tests."""
        return self.run_command(
            ["python", "-m", "pytest", "tests/test_track_*.py", "-v"],
            "Tracking Tests"
        )
    
    def run_pipeline_tests(self) -> Dict[str, Any]:
        """Run pipeline tests."""
        return self.run_command(
            ["python", "-m", "pytest", "tests/test_pipeline*.py", "-v"],
            "Pipeline Tests"
        )
    
    def run_api_tests(self) -> Dict[str, Any]:
        """Run API tests."""
        return self.run_command(
            ["python", "-m", "pytest", "tests/test_barca_api.py", "-v"],
            "API Tests"
        )
    
    def run_training_tests(self) -> Dict[str, Any]:
        """Run training-related tests."""
        return self.run_command(
            ["python", "-m", "pytest", "tests/test_weekly_retrain.py", "-v"],
            "Training Tests"
        )
    
    def run_graph_tests(self) -> Dict[str, Any]:
        """Run graph-related tests."""
        return self.run_command(
            ["python", "-m", "pytest", "tests/test_gcn*.py", "tests/test_graph*.py", "-v"],
            "Graph Tests"
        )
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests."""
        return self.run_command(
            ["python", "-m", "pytest", "tests/test_performance_optimization.py", "-v"],
            "Performance Tests"
        )
    
    def run_security_tests(self) -> Dict[str, Any]:
        """Run security tests."""
        return self.run_command(
            ["python", "-m", "pytest", "tests/test_security.py", "-v"],
            "Security Tests"
        )
    
    def run_full_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite."""
        return self.run_command(
            ["python", "-m", "pytest", "tests/", "-v", "--cov=src", "--cov-report=json"],
            "Full Test Suite with Coverage"
        )
    
    def run_linting(self) -> Dict[str, Any]:
        """Run code quality checks."""
        results = []
        
        # Ruff linting
        results.append(self.run_command(
            ["python", "-m", "ruff", "check", "src", "tests"],
            "Ruff Linting"
        ))
        
        # Black format check
        results.append(self.run_command(
            ["python", "-m", "black", "--check", "src", "tests"],
            "Black Format Check"
        ))
        
        # isort check
        results.append(self.run_command(
            ["python", "-m", "isort", "--check-only", "src", "tests"],
            "Import Sort Check"
        ))
        
        # MyPy type checking
        results.append(self.run_command(
            ["python", "-m", "mypy", "src"],
            "MyPy Type Checking"
        ))
        
        return {
            "description": "Code Quality Checks",
            "results": results,
            "success": all(r["success"] for r in results)
        }
    
    def run_build_test(self) -> Dict[str, Any]:
        """Test Docker build."""
        return self.run_command(
            ["docker", "build", "-t", "fifa-soccer-ds:test", "."],
            "Docker Build Test"
        )
    
    def generate_report(self) -> None:
        """Generate comprehensive test report."""
        report_path = self.project_root / "outputs" / "test_report.json"
        report_path.parent.mkdir(exist_ok=True)
        
        # Calculate summary statistics
        total_tests = len(self.results["tests"])
        passed_tests = sum(1 for t in self.results["tests"] if t["success"])
        failed_tests = total_tests - passed_tests
        
        total_duration = sum(t["duration"] for t in self.results["tests"])
        
        self.results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": passed_tests / max(total_tests, 1),
            "total_duration": total_duration,
            "average_test_duration": total_duration / max(total_tests, 1)
        }
        
        # Write report
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Print summary
        print(f"\n{'='*80}")
        print("COMPREHENSIVE TEST REPORT")
        print(f"{'='*80}")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {self.results['summary']['success_rate']:.2%}")
        print(f"Total Duration: {total_duration:.2f}s")
        print(f"Average Duration: {self.results['summary']['average_test_duration']:.2f}s")
        print(f"Report saved to: {report_path}")
        
        if failed_tests > 0:
            print(f"\n{'='*80}")
            print("FAILED TESTS:")
            print(f"{'='*80}")
            for test in self.results["tests"]:
                if not test["success"]:
                    print(f"❌ {test['description']}: {test.get('error', 'Unknown error')}")
    
    def run_all_tests(self) -> bool:
        """Run all tests and return overall success."""
        print("🚀 Starting Comprehensive Test Suite")
        print(f"Project Root: {self.project_root}")
        
        # Define test suite
        test_suite = [
            ("Import Tests", self.run_import_tests),
            ("Smoke Tests", self.run_smoke_tests),
            ("Detection Tests", self.run_detection_tests),
            ("Tracking Tests", self.run_tracking_tests),
            ("Pipeline Tests", self.run_pipeline_tests),
            ("API Tests", self.run_api_tests),
            ("Training Tests", self.run_training_tests),
            ("Graph Tests", self.run_graph_tests),
            ("Performance Tests", self.run_performance_tests),
            ("Security Tests", self.run_security_tests),
            ("Code Quality", self.run_linting),
        ]
        
        # Run tests
        for test_name, test_func in test_suite:
            try:
                result = test_func()
                self.results["tests"].append(result)
                
                if not result["success"]:
                    print(f"❌ {test_name} FAILED")
                else:
                    print(f"✅ {test_name} PASSED")
                    
            except Exception as e:
                error_result = {
                    "description": test_name,
                    "success": False,
                    "error": str(e),
                    "duration": 0
                }
                self.results["tests"].append(error_result)
                print(f"❌ {test_name} ERROR: {e}")
        
        # Generate report
        self.generate_report()
        
        # Return overall success
        return self.results["summary"]["success_rate"] >= 0.9  # 90% success rate


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run comprehensive test suite")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests only"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner(args.project_root)
    
    if args.quick:
        # Run only essential tests
        quick_tests = [
            ("Import Tests", runner.run_import_tests),
            ("Smoke Tests", runner.run_smoke_tests),
            ("Code Quality", runner.run_linting),
        ]
        
        for test_name, test_func in quick_tests:
            result = test_func()
            runner.results["tests"].append(result)
    
    else:
        # Run full test suite
        success = runner.run_all_tests()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()