"""Backward-compatible import for the canonical YouTube analysis API.

The maintained application lives in :mod:`src.api.main`.  This module remains so
older deployment commands importing ``src.api.youtube_endpoints:app`` continue to
work without running a second, divergent task implementation.
"""

from src.api.main import app

__all__ = ["app"]
