# Contributing

Thanks for your interest in contributing to FIFA Soccer DS!

## Development Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make setup    # installs pre-commit hooks
```

## Code Style

This project uses **ruff**, **black**, **isort**, and **mypy** (configured in `pyproject.toml`).

```bash
make fmt       # auto-format code
make lint      # check linting + type errors
make lint-all  # run all pre-commit hooks
```

## Running Tests

```bash
make smoke     # quick sanity tests
make test      # full test suite
```

## Submitting Changes

1. Fork the repo and create a feature branch from `master`.
2. Make your changes — keep them focused on a single issue.
3. Run `make lint` and `make test` before committing.
4. Use [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `chore:`, `docs:`, etc.
5. Open a PR against `master` and fill out the PR template.

## Reporting Bugs

Use the [bug report template](https://github.com/jayhemnani9910/fifa-soccer-ds/issues/new?template=bug_report.yml) on GitHub Issues.
