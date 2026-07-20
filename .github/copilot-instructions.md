# Repository coding instructions

- Target CPython `>=3.12,<3.13`; use `from __future__ import annotations` in Python modules.
- Treat `pyproject.toml` as the dependency and tool-configuration source of truth.
- Use Ruff for linting, import sorting, and formatting (100-character line length) and mypy for
  first-party source typing.
- Use Pydantic v2 models for API boundaries and `pathlib.Path` for filesystem paths.
- Use structured `logging` in production paths; do not add `print` outside intentional CLI output.
- Keep optional integrations isolated and fail explicitly when a requested capability is absent.
- Never fabricate metrics, artifacts, successful results, player identities, or tactical events.
- Validate media/model paths, sizes, URLs, dimensions, finite numerical inputs, and output roots.
- Load PyTorch artifacts with restricted loading where possible; require explicit trust for
  unavoidable pickle-backed datasets.
- Add focused pytest regression coverage for behavior and security changes.
- Run `make lint`, `make test`, and the relevant focused checks before proposing a change.
- Do not claim GPU, network, model-quality, or real-media validation from mocked/synthetic tests.

Major areas are `src/api`, `src/analytics`, `src/calib`, `src/data`, `src/detect`, `src/graph`,
`src/live`, `src/models`, `src/track`, and `src/youtube`. Preserve their import direction and avoid
adding new global mutable state or eager heavy imports in package `__init__.py` files.
