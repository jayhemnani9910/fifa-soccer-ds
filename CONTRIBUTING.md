# Contributing

## Development setup

The project supports CPython `>=3.12,<3.13`. `pyproject.toml` is the dependency source of truth;
`requirements.txt` is an editable development-install wrapper.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
make setup
```

Install optional integrations only when the work needs them:

```bash
make setup-all
```

## Quality gates

Ruff provides linting, import sorting, and formatting. Mypy checks first-party source. Pytest and
coverage provide the test gates.

```bash
make fmt       # intentionally rewrite lint/format issues
make lint      # Ruff lint + format check + mypy
make test      # complete test suite
make test-cov  # branch coverage
make lint-all  # pre-commit against all tracked files
make build     # wheel and source distribution
```

Do not mark a network, GPU, external-service, or real-model test as passing unless it ran in that
environment. Keep unavailable checks explicit in the PR description.

## Change guidelines

1. Branch from the repository's current default branch.
2. Keep changes focused and preserve unrelated work in a dirty worktree.
3. Add a focused regression test for each correctness or security fix.
4. Do not weaken validation, typing, or assertions to make a gate green.
5. Treat model checkpoints and PyG/PyTorch datasets as untrusted serialized input.
6. Do not commit credentials, private media, datasets, model weights, experiment stores, or build output.
7. Update documentation when behavior, configuration, dependencies, or compatibility changes.
8. Record migrations for intentional public behavior changes.

Conventional Commit prefixes (`feat:`, `fix:`, `test:`, `docs:`, `chore:`) are encouraged. Open
the pull request against the current default branch and complete `.github/PULL_REQUEST_TEMPLATE.md`.

## Dependency changes

Dependency upgrades require current official release/compatibility evidence. Coordinate tightly
coupled groups—especially Python packaging tools, PyTorch/TorchVision/PyG/CUDA, ONNX Runtime,
NumPy/SciPy/scikit-learn/OpenCV, and FastAPI/Starlette/Pydantic. Update the dependency report when
the recommended compatibility target changes.

## Reporting bugs and vulnerabilities

Public bugs may use GitHub Issues. Security vulnerabilities must follow [SECURITY.md](SECURITY.md)
and must not be disclosed in a public issue.
