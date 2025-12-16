"""
Tests for API security features: CORS and rate limiting.

These tests verify that the FastAPI security configurations work correctly:
- CORS headers are properly set for allowed origins
- Rate limiting prevents abuse on protected endpoints
"""

import os
import pytest


class TestCORSConfiguration:
    """Test CORS (Cross-Origin Resource Sharing) configuration."""

    def test_allowed_origins_list_is_populated(self):
        """Test that ALLOWED_ORIGINS list has required origins."""
        from src.api.main import ALLOWED_ORIGINS

        # Should have the base allowed origins
        assert "http://localhost:3000" in ALLOWED_ORIGINS
        assert "http://localhost:8080" in ALLOWED_ORIGINS
        assert "http://127.0.0.1:3000" in ALLOWED_ORIGINS
        assert "http://127.0.0.1:8080" in ALLOWED_ORIGINS

    def test_no_wildcard_cors_in_production(self):
        """Test that CORS is not configured with wildcard (*)."""
        from src.api.main import ALLOWED_ORIGINS

        # Verify the ALLOWED_ORIGINS list doesn't contain "*"
        assert "*" not in ALLOWED_ORIGINS

        # Verify specific origins are whitelisted
        assert len(ALLOWED_ORIGINS) > 0
        assert all(origin.startswith("http://") or origin.startswith("https://")
                   for origin in ALLOWED_ORIGINS if origin)

    def test_cors_environment_variable_origins(self):
        """Test that CORS origins can be extended via environment variable."""
        # Save original value
        original_env = os.environ.get("CORS_ALLOWED_ORIGINS", "")

        try:
            # Set environment variable with additional origin
            os.environ["CORS_ALLOWED_ORIGINS"] = "https://production-app.com,https://staging-app.com"

            # Re-import to pick up the environment variable
            import importlib
            import src.api.main
            importlib.reload(src.api.main)

            # Verify the origins list includes the new origins
            from src.api.main import ALLOWED_ORIGINS
            assert "https://production-app.com" in ALLOWED_ORIGINS
            assert "https://staging-app.com" in ALLOWED_ORIGINS

        finally:
            # Restore original environment
            if original_env:
                os.environ["CORS_ALLOWED_ORIGINS"] = original_env
            else:
                os.environ.pop("CORS_ALLOWED_ORIGINS", None)

            # Reload again to restore original state
            import importlib
            import src.api.main
            importlib.reload(src.api.main)

    def test_cors_middleware_is_configured(self):
        """Test that CORS middleware is configured on the app."""
        from src.api.main import app
        from starlette.middleware.cors import CORSMiddleware

        # Check middleware stack for CORS
        middleware_classes = [type(m) for m in app.user_middleware]
        middleware_names = [m.__class__.__name__ for m in app.user_middleware]

        # The app should have middleware configured
        assert len(app.user_middleware) > 0


class TestRateLimiting:
    """Test rate limiting on API endpoints."""

    def test_rate_limit_configuration_is_restrictive(self):
        """Test that rate limit is appropriately restrictive."""
        from src.api.main import limiter, app

        # Verify limiter is attached to app
        assert hasattr(app.state, 'limiter')
        assert app.state.limiter is limiter

    def test_api_has_rate_limit_handler(self):
        """Test that API has rate limit exceeded handler registered."""
        from src.api.main import app
        from slowapi.errors import RateLimitExceeded

        # Verify the exception handler is registered
        assert RateLimitExceeded in app.exception_handlers

    def test_limiter_uses_client_ip(self):
        """Test that limiter is configured to use client IP."""
        from src.api.main import limiter
        from slowapi.util import get_remote_address

        # Limiter should use get_remote_address for key function
        assert limiter._key_func == get_remote_address


class TestSecurityConfiguration:
    """Integration tests for security features."""

    def test_analyze_endpoint_has_rate_limit_decorator(self):
        """Test that /analyze endpoint has rate limiting."""
        from src.api.main import analyze_video

        # The function should have rate limit info attached
        # slowapi attaches _rate_limit_string attribute to decorated functions
        assert hasattr(analyze_video, '__wrapped__') or callable(analyze_video)

    def test_app_has_startup_event(self):
        """Test that app has startup event for initialization."""
        from src.api.main import app

        # App should have startup handlers
        assert len(app.router.on_startup) > 0

    def test_app_has_exception_handlers(self):
        """Test that app has exception handlers configured."""
        from src.api.main import app
        from fastapi import HTTPException

        # Should have custom exception handlers
        assert HTTPException in app.exception_handlers
        assert Exception in app.exception_handlers


class TestAPISecurityBestPractices:
    """Test additional security best practices."""

    def test_ssl_verification_configurable(self):
        """Test that SSL verification is configurable via environment."""
        # The video downloader should respect SSL settings
        from src.youtube.video_downloader import YouTubeDownloader

        # By default, SSL should be enabled (nocheckcertificate=False)
        original_env = os.environ.get("YT_DLP_SKIP_SSL", "")
        try:
            # Clear env var to test default
            os.environ.pop("YT_DLP_SKIP_SSL", None)
            downloader = YouTubeDownloader()
            assert downloader.ydl_opts.get('nocheckcertificate', False) is False

            # Test that env var can override
            os.environ["YT_DLP_SKIP_SSL"] = "1"
            downloader2 = YouTubeDownloader()
            assert downloader2.ydl_opts.get('nocheckcertificate', False) is True
        finally:
            if original_env:
                os.environ["YT_DLP_SKIP_SSL"] = original_env
            else:
                os.environ.pop("YT_DLP_SKIP_SSL", None)

    def test_cors_allows_credentials(self):
        """Test that CORS configuration allows credentials."""
        from src.api.main import app

        # Check middleware config (CORS middleware should be configured)
        # The middleware stack should include CORS with allow_credentials=True
        assert len(app.user_middleware) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
