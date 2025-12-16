# Security Tests Quick Reference

Quick reference for running and understanding the API security tests.

## Quick Commands

```bash
# Run all security tests
pytest tests/test_api_security.py -v

# Run just CORS tests
pytest tests/test_api_security.py::TestCORSConfiguration -v

# Run just rate limiting tests
pytest tests/test_api_security.py::TestRateLimiting -v

# Run with coverage
pytest tests/test_api_security.py --cov=src.api.main

# Use the shell script
./tests/run_security_tests.sh
```

## What's Tested

### CORS (Cross-Origin Resource Sharing)
✅ Allowed origins get CORS headers
✅ Blocked origins don't get access
✅ Preflight requests work
✅ No wildcard (`*`) configuration

### Rate Limiting
✅ `/analyze` endpoint: 10 requests/minute
✅ 11th request returns HTTP 429
✅ Health checks not rate limited
✅ Limits reset after 1 minute

## Configuration in Code

```python
# CORS Configuration (src/api/main.py)
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
]

# Rate Limiting (src/api/main.py)
@app.post("/analyze")
@limiter.limit("10/minute")
async def analyze_video(...):
    ...
```

## Test File Structure

```
test_api_security.py
├── TestCORSConfiguration (10 tests)
│   ├── Allowed origins
│   ├── Disallowed origins
│   ├── Preflight requests
│   └── Environment config
├── TestRateLimiting (6 tests)
│   ├── Request limiting
│   ├── Error responses
│   └── Endpoint protection
├── TestSecurityIntegration (3 tests)
│   └── Combined features
└── TestAPISecurityBestPractices (4 tests)
    └── Configuration validation
```

## Common Issues

**Problem**: Tests fail with import errors
**Solution**: Run from project root: `cd /home/jey/jh-core/projects/fifa-soccer-ds`

**Problem**: Rate limit tests fail intermittently
**Solution**: Wait 60 seconds between test runs

**Problem**: CORS tests fail
**Solution**: Check exact origin strings in headers (case-sensitive)

## Adding Production Origins

Set environment variable before starting API:

```bash
export CORS_ALLOWED_ORIGINS="https://my-app.com,https://staging.my-app.com"
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## Test Pattern Example

```python
def test_cors_allowed_origin_localhost_3000(self):
    """Test CORS headers for allowed origin."""
    client = TestClient(app)

    response = client.get(
        "/",
        headers={"Origin": "http://localhost:3000"}
    )

    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
```

## Files Created

1. `tests/test_api_security.py` - Main test file (350+ lines)
2. `tests/README_API_SECURITY_TESTS.md` - Detailed documentation
3. `tests/SECURITY_TESTS_SUMMARY.md` - Comprehensive summary
4. `tests/run_security_tests.sh` - Shell script runner
5. `tests/SECURITY_QUICK_REFERENCE.md` - This file

## Documentation

For detailed information, see:
- Full documentation: `tests/README_API_SECURITY_TESTS.md`
- Summary: `tests/SECURITY_TESTS_SUMMARY.md`
- Implementation: `src/api/main.py`
