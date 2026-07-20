# v2.0 progress log

One entry per session. Read this at session start, update before session end.
Plan lives in [V2_ROADMAP.md](V2_ROADMAP.md); evidence in
`audits/2026-07-20-verification/`.

## 2026-07-20

### Landed

**Phase 0.1, the git rescue.** The repository was on a detached HEAD inside a
rebase abandoned on April 29, with the entire July audit remediation
uncommitted. Nothing local had ever been pushed: `origin/master` was `2aeea89`,
which was simultaneously the rebase base and the parent of the detached HEAD, so
no published history was at risk. Backed up first with a bundle and a worktree
copy at `~/backups/fifa-soccer-ds-2026-07-20`, verified both. Used
`git rebase --quit`, which discards rebase state while leaving HEAD and the
working tree untouched. `git rebase --abort` would have reset to `588435b` and
destroyed the uncommitted work.

Six unapplied April commits were dropped from history because their content was
entirely superseded: torch 2.8 against the current 2.13, mlflow 3.11 against
3.14, an identical numpy pin, a lifespan test already rewritten, and a ruff
consolidation already done. AUD-028 records that the April torch target carried
known advisories. Those commits remain reachable in the bundle and the reflog.

The remediation landed as seven reviewable commits and was tagged
`audit-remediation-2026-07-20`, deliberately non-semver so release-please keeps
control of versioning.

**Phase 0.3 and 0.4, developer entry points.** Every documented `make` target
now works from a clean checkout. `PYTHON` defaults to the repository virtualenv
with a `python3` fallback and stays overridable. `lint` and `fmt` both route
through the project interpreter: `lint` had been invoking a PATH-resolved mypy
that could not see project dependencies and reported 25 errors across 19 files
against code that was already clean, and `fmt` used ruff 0.15.10 from PATH while
`lint` verified against the pinned 0.15.22. `build` uses isolated mode so a
stale `build/` directory cannot shadow the PyPA build tool. `lint-all` puts the
virtualenv on PATH because the mypy hook is a system hook needing a
dependency-carrying `python`. The `dev` Compose service reads an optional env
file, and the README now states that the application reads process environment
variables only.

**Phase 0.5, CI.** The first CI run of the remediation failed two jobs, which is
notable because the July audit only ever ran locally. The pre-commit gate failed
on a ruff reformat of a file under `audits/`, now excluded with `force-exclude`
since pre-commit passes explicit filenames. The dependency scan failed on
setuptools 78.1.0 carrying PYSEC-2025-49 and PYSEC-2026-3447; the install step
upgraded pip but never setuptools, and the `[build-system]` pin applies to
isolated builds rather than the scanned environment.

Clearing those let CI reach the test step for the first time, which exposed a
real bug. `src/data/la_liga_loader.py` resolved the Kaggle API at module scope
guarded by `except ImportError`, but the kaggle package calls `authenticate()`
in its own `__init__` and that calls `exit(1)`, raising `SystemExit`, which
derives from `BaseException` and slips past the guard. Importing `src.data` was
enough to kill the process. It reached `src/train/weekly_retrainer` and the test
suite, so pytest aborted with INTERNALERROR. `_ensure_api` already converted
authentication failures into `KaggleAuthenticationError`, but that handling was
unreachable. The API now resolves on first use and catches `SystemExit`.

This passed on the developer machine only because `~/.kaggle/kaggle.json`
exists. The suite had never actually passed in a clean environment. The
adversarial reviewer had flagged this exact risk in the abstract that morning,
noting that import tests "only ever run where the venv already masks it."

CI is green on master, all six checks.

### Method notes

Builder and reviewer were always separate Sonnet sessions; the reviewer saw only
the brief and the diff. The review caught two things worth keeping: the
`fmt`/`lint` ruff version skew, and that `rm -rf build` was deleting a directory
`python -m build` never writes to, since the stale directory only ever mattered
by shadowing an import. Every acceptance command was re-run by the conductor
rather than taken from a worker report.

Three worker actions were denied as out of scope or unsafe: reading shell
dotfiles outside the repository, moving the real `.env` aside behind a shell
trap, and rendering the `dev` Compose service in a way that would have printed
`.env` values into the transcript.

### Phase 1.6, done (autonomous continuation)

Closed the largest verified coverage gap. The soccer branch of
`process_youtube_video` assembles the result returned by `POST /analyze`,
including the `capabilities` block and the three helpers that return empty
rather than invented analytics, and it had zero coverage. Three tests now stub
only the network and model boundaries so classification, frame processing,
graph building and result assembly run for real, then assert that capabilities
stay marked not implemented, that events stay empty, that players and team
metrics carry no identity or possession keys, and that counters scale with the
synthetic input instead of being constants.

`src/pipeline_full.py` coverage moved from 46 to 54 percent and the assembly
block is no longer unexecuted. Verified load-bearing by three independent
mutations run by the conductor, each caught by a different test: marking
`event_detection` implemented, adding a `goals` count to players, and replacing
the detection total with a constant. The source file was restored byte-exactly
after each.

Deviation worth noting: this diff did not get a separate reviewer session,
because the account's rolling usage was at 98 percent. It is test-only with no
behaviour change, and three independent mutations are a stronger check on test
quality than a read-through would have been, but the usual two-session
discipline was not applied.

Also swept every module under `src/` for the import-time side effect class that
the Kaggle bug belonged to, by importing each one in a subprocess with a
credential-free HOME. All 55 import cleanly, so that bug was the only one of its
kind.

### Not started

Phase 0.2, recreating `.venv` from `pyproject.toml`. Deliberately deferred: the
current environment works despite two orphaned packages failing `pip check`
(pydrive2 and pyopenssl against cryptography 48), and a rebuild is the slowest
and most fragile item. Build it at a separate path, prove it green, then swap.

Phase 1 items 1.1 through 1.9, and all of Phase 2.
