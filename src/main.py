"""ASGI entrypoint for dockerized deployments."""

from __future__ import annotations

from src.live.barca_api import BarcaAPIServer

server = BarcaAPIServer()
app = server.app

__all__ = ["app", "server"]
