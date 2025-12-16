#!/bin/bash
# Quick runner for API security tests
# Usage: ./tests/run_security_tests.sh

set -e

echo "=========================================="
echo "FIFA Soccer DS - API Security Tests"
echo "=========================================="
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "Error: pytest is not installed"
    echo "Install it with: pip install pytest"
    exit 1
fi

echo "Running CORS configuration tests..."
echo "--------------------------------------------------"
pytest tests/test_api_security.py::TestCORSConfiguration -v

echo ""
echo "Running rate limiting tests..."
echo "--------------------------------------------------"
pytest tests/test_api_security.py::TestRateLimiting -v

echo ""
echo "Running security integration tests..."
echo "--------------------------------------------------"
pytest tests/test_api_security.py::TestSecurityIntegration -v

echo ""
echo "Running security best practices tests..."
echo "--------------------------------------------------"
pytest tests/test_api_security.py::TestAPISecurityBestPractices -v

echo ""
echo "=========================================="
echo "All security tests completed!"
echo "=========================================="
echo ""
echo "For detailed coverage report, run:"
echo "pytest tests/test_api_security.py --cov=src.api.main --cov-report=html"
echo ""
