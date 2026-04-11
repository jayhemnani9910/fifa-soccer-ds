# Copilot Instructions for FIFA Soccer DS

## Code Style
- Python 3.11+ features (match statements, `type` aliases, `ExceptionGroup`)
- Line length: 100 characters (ruff/black)
- Double quotes for strings
- Type-hint all function signatures and return types
- Use `from __future__ import annotations` for forward references

## Conventions
- Use Pydantic v2 models for API request/response schemas (see `src/schemas.py`)
- Use `rich` for CLI output formatting
- Use `supervision` library for detection visualization
- Prefer `numpy` vectorized operations over Python loops
- Use `logging` module (not print) for all output in `src/`
- Follow Conventional Commits: `feat:`, `fix:`, `chore:`, `docs:`, `test:`

## Architecture
- Detection: `src/detect/` — YOLOv8 inference and export
- Tracking: `src/track/` — ByteTrack with Kalman filtering
- Graph: `src/graph/` — spatial-temporal graph construction
- Models: `src/models/` — GraphSAGE / GCN architectures
- API: `src/api/` — FastAPI endpoints with rate limiting
- Utils: `src/utils/` — monitoring, visualization, health checks

## Testing
- Test files go in `tests/` with `test_` prefix
- Use `pytest` with markers (`@pytest.mark.smoke` for quick tests)
- Mock external services (MLflow, YouTube, GPU) in tests
- Run `make lint` and `make test` before committing

## Dependencies
- Guard optional imports with try/except (torch-geometric, prometheus_client, psutil)
- Core deps are in `requirements.txt`, full list in `pyproject.toml`
