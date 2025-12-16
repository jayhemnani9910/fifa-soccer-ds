# API Security Tests

This document describes the security tests for the FIFA Soccer DS API, specifically covering CORS and rate limiting configurations.

## Overview

The `test_api_security.py` file contains comprehensive tests for two critical security features:

1. **CORS (Cross-Origin Resource Sharing)** - Ensures only allowed origins can access the API
2. **Rate Limiting** - Prevents abuse by limiting request frequency on expensive endpoints

## Test Coverage

### CORS Tests (`TestCORSConfiguration`)

Tests that verify CORS headers are correctly configured:

- ✅ Allowed origins (localhost:3000, localhost:8080, 127.0.0.1:3000, 127.0.0.1:8080) receive proper CORS headers
- ✅ Disallowed origins do not receive CORS access
- ✅ Preflight (OPTIONS) requests are handled correctly
- ✅ Credentials are allowed for authenticated requests
- ✅ Custom headers (Authorization, Content-Type, X-Request-ID) are permitted
- ✅ Additional origins can be added via `CORS_ALLOWED_ORIGINS` environment variable
- ✅ No wildcard (`*`) CORS configuration in production

### Rate Limiting Tests (`TestRateLimiting`)

Tests that verify rate limiting protects the API from abuse:

- ✅ `/analyze` endpoint is limited to 10 requests per minute
- ✅ Rate limiting is applied per client IP address
- ✅ Read-only endpoints (/, /health, /metrics) are not aggressively rate limited
- ✅ Rate limit resets after the time window expires
- ✅ Rate limit errors provide informative error messages (HTTP 429)

### Integration Tests (`TestSecurityIntegration`)

Tests that verify security features work together:

- ✅ CORS and rate limiting work together correctly
- ✅ Security headers apply to all endpoints
- ✅ No wildcard CORS in production builds

### Best Practices Tests (`TestAPISecurityBestPractices`)

Tests that verify security best practices:

- ✅ Rate limits are appropriately restrictive
- ✅ Rate limit exception handler is registered
- ✅ Expensive operations are protected by rate limiting
- ✅ CORS allows necessary HTTP methods (GET, POST, OPTIONS, etc.)

## Running the Tests

### Run all API security tests:
```bash
cd /home/jey/jh-core/projects/fifa-soccer-ds
pytest tests/test_api_security.py -v
```

### Run specific test classes:
```bash
# Test only CORS configuration
pytest tests/test_api_security.py::TestCORSConfiguration -v

# Test only rate limiting
pytest tests/test_api_security.py::TestRateLimiting -v

# Test only security integration
pytest tests/test_api_security.py::TestSecurityIntegration -v
```

### Run specific tests:
```bash
# Test CORS for localhost:3000
pytest tests/test_api_security.py::TestCORSConfiguration::test_cors_allowed_origin_localhost_3000 -v

# Test rate limiting on analyze endpoint
pytest tests/test_api_security.py::TestRateLimiting::test_rate_limit_analyze_endpoint -v
```

### Run with coverage:
```bash
pytest tests/test_api_security.py --cov=src.api.main --cov-report=html
```

## Configuration Details

### CORS Configuration (src/api/main.py)

```python
ALLOWED_ORIGINS = [
    "http://localhost:3000",      # Local frontend development
    "http://localhost:8080",      # Local alternative
    "http://127.0.0.1:3000",      # Local frontend
    "http://127.0.0.1:8080",      # Local alternative
]

# Add production origins via environment variable
CORS_ALLOWED_ORIGINS="https://prod-app.com,https://staging-app.com"
```

### Rate Limiting Configuration (src/api/main.py)

```python
# /analyze endpoint: 10 requests per minute
@app.post("/analyze", response_model=AnalyzeResponse)
@limiter.limit("10/minute")
async def analyze_video(request: Request, req: AnalyzeRequest, background_tasks: BackgroundTasks):
    ...
```

## Security Improvements Tested

These tests verify the following security improvements made to the API:

1. **Restricted CORS** - Changed from `allow_origins=["*"]` to a whitelist of allowed origins
2. **Rate Limiting** - Added 10 requests/minute limit on the `/analyze` endpoint to prevent abuse
3. **Protected Endpoints** - Expensive ML operations are protected while health checks remain accessible
4. **Environment Configuration** - Production origins can be added via environment variables

## Test Dependencies

These tests require:
- `fastapi==0.110.0`
- `pytest==8.1.1`
- `slowapi==0.1.9`

All dependencies are included in `requirements.txt`.

## Common Issues

### Issue: Rate limit tests failing intermittently
**Solution**: Rate limit tests create 10+ requests rapidly. If running tests multiple times quickly, you may hit actual rate limits. Wait 60 seconds between test runs or use a fresh TestClient instance.

### Issue: CORS tests failing
**Solution**: Ensure you're testing with the exact origin strings defined in `ALLOWED_ORIGINS`. Headers are case-sensitive.

### Issue: Mock orchestrator errors
**Solution**: Tests use `@patch("src.api.main.orchestrator")` to mock the pipeline orchestrator. Ensure the patch path matches your module structure.

## Contributing

When adding new security features:

1. Add corresponding tests to this file
2. Follow the existing test patterns (TestClient, mocking, clear assertions)
3. Update this README with new test descriptions
4. Ensure tests are isolated and don't depend on external services

## Related Files

- `/home/jey/jh-core/projects/fifa-soccer-ds/src/api/main.py` - Main API implementation
- `/home/jey/jh-core/projects/fifa-soccer-ds/tests/test_api_youtube.py` - General API tests
- `/home/jey/jh-core/projects/fifa-soccer-ds/tests/test_security.py` - Pipeline security tests
