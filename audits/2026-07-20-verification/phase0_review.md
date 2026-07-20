# Phase 0.3 / 0.4 review record (2026-07-20)

Builder and reviewer were separate Sonnet sessions. The reviewer saw only the
brief and the diff, never the builder's transcript. Two rounds.

## Round 1

VERDICT: APPROVE with two non-blocking findings.

1. `Makefile` `fmt` target still called bare `ruff`, while `lint` had been fixed
   to route through `$(PYTHON) -m`. Not hypothetical: PATH ruff on this machine
   is 0.15.10, the pinned `.venv` ruff is 0.15.22. A developer running
   `make fmt` could produce formatting that `make lint` then rejects.
2. `build` used `rm -rf build` before invoking. It worked and was idempotent,
   but `$(PYTHON) -I -m build` fixes the same problem without deleting
   anything, and `requires-python = ">=3.12,<3.13"` guarantees `-I` is
   available.

Also verified in round 1: `PYTHON` expansion under an exported override, under
`VENV=` pointing at both a missing and a real path, and under `make -C` from an
unrelated working directory. The `env_file` / `required: false` Compose syntax
parses on the installed version. `src/` contains no dotenv import, and
`env_file` appears only on the `dev` service, matching the README and
`.env.example` claims. CI never invokes `make`, so none of this changes CI.

## Round 2

Both findings applied: `fmt` now routes through `$(PYTHON) -m ruff`, and
`build` uses `$(PYTHON) -I -m build` with the delete removed entirely.

VERDICT: APPROVE. No remaining findings.

Evidence gathered for the question of whether `-I` breaks anything, since
isolated mode also implies `-E` and `-s`:

- `-P` (no cwd on `sys.path`, implied by `-I` since 3.11) is what actually
  stops the shadowing. `sys.path[0]` resolves away from the working directory
  and `build` loads from `.venv/lib/python3.12/site-packages`.
- `-E`: a repo-wide grep across source, configs, Makefile, CI and compose for
  `PYTHONPATH`, `PYTHONHOME`, `PYTHONSTARTUP`, `PYTHONDONTWRITEBYTECODE` and
  `PYTHONNOUSERSITE` returned no references, and `PYTHONPATH` is unset in the
  shell that runs the build. Nothing exists for `-E` to suppress.
- `-s`: there is no `setup.py`, the build is pure `pyproject.toml` with
  `setuptools.build_meta`, and `python -m build` provisions its own isolated
  environment for build dependencies regardless of the outer interpreter's
  site configuration.

One clarification worth recording: the `build/lib/...` paths in the build log
belong to a temporary directory PyPA build extracts the sdist into. This
repository's own `build/` directory is never written to by `python -m build` in
either the old or the new Makefile. The stale `build/` directory only ever
mattered because it shadowed the import, which is precisely what `-I` prevents.

## Independent verification by the conductor

Run directly, not taken from either worker's report:

| Check | Result |
| --- | --- |
| `make smoke` | 10 passed |
| `make test` | 251 passed |
| `make lint` | exit 0, no issues in 55 source files |
| `make build` twice with a shadowing `build/__init__.py` present | both exit 0 |
| shadow file after both builds | survived, so nothing is deleted |
| `make fmt` then `git status` | no tracked file modified |
| `docker compose config`, base and mlops | exit 0 |
| `docker compose --env-file /dev/null config` | exit 0 |
| `PYTHON=python3 make -n test` | resolves to `python3` |
| `make -n test` | resolves to `.venv/bin/python` |
