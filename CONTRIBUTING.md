# Contributing

I try to reply to issues within a few days.

## Setup

You need Python 3.12 (`>=3.12,<3.13`). `pyproject.toml` is the only place dependencies are declared.

```bash
git clone https://github.com/jayhemnani9910/fifa-soccer-ds
cd fifa-soccer-ds
python3.12 -m venv .venv
source .venv/bin/activate
make setup
make setup-all
```

On a bare Debian or Ubuntu system, OpenCV also needs `apt-get install libgl1 libglib2.0-0`. Without it,
pytest stops at collection with `libGL.so.1: cannot open shared object file`.

`make setup` installs the dev tools and the pre-commit hook, which is enough for `make lint`.
`make setup-all` adds the optional extras, such as MLflow, that CI installs.

## Tests and checks

```bash
make test      # all tests
make smoke     # the quick ones
make test-cov  # with branch coverage
make lint      # ruff, format check, mypy
make lint-all  # every pre-commit hook
make build     # wheel and sdist
```

The tests use synthetic data and model doubles. They need no GPU or footage. `tests/test_error_handling.py`
and `tests/test_security.py` download the stock `yolov8n.pt` into the repo root the first time, so those two
need network. Git ignores the file.
Run them as a normal user. As root, `test_output_directory_creation_failure` fails, because it
expects `/root` to be unwritable.
If you checked something on a real GPU or real footage, say so in the PR. If you could not, say that too.

## Style

Ruff handles lint, imports and formatting. `make fmt` rewrites what it can. Lines are 100 characters.

- Add a regression test with every bug or security fix.
- Don't weaken validation, types or assertions to get a check to pass.
- Treat checkpoints and PyG/PyTorch datasets as untrusted, since they can hold pickle payloads.
- Don't commit credentials, private media, datasets, weights or build output.
- Update the docs when behavior, config or dependencies change.
- Upgrade tied packages together: PyTorch, TorchVision, PyG and CUDA; ONNX Runtime; NumPy, SciPy,
  scikit-learn and OpenCV; FastAPI, Starlette and Pydantic. Record it in
  [DEPENDENCY_UPGRADE_REPORT.md](DEPENDENCY_UPGRADE_REPORT.md).

## Branches and PRs

Branch from `master` and open the PR against `master`. Keep it to one change. Commit messages use
Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`). Fill in the PR template.

## Bugs

Open an issue with the command you ran, what you expected, what happened, and your Python version
and OS. Security problems go through [SECURITY.md](SECURITY.md), not a public issue.
