# Adversarial verification report — fifa-soccer-ds exploration claims

Repo: `/home/po/projects/work/fifa-soccer-ds`, detached HEAD (mid-rebase), large uncommitted
worktree. No git state was changed. No repo file was edited. `.venv` was used for all
pytest/coverage runs; no packages were installed. `.env` contents were never printed.

Environment note: `python3` on this machine is not a bare system interpreter — it has `pytest`,
`fastapi`, `uvicorn`, `build`, `ruff`, and `mypy` importable, but is **missing** `torch`,
`ultralytics`, `slowapi`, and `pre_commit`. This matters for claim 7's target-by-target
breakdown below.

---

## Claim 1 — soccer branch of `process_youtube_video` is untested; E2E test bypasses it

**Verdict: CONFIRMED**

- `src/pipeline_full.py:878-1072` is `process_youtube_video`; the soccer branch (download →
  frame extraction → `process_frames_directory` → result dict with `capabilities` at
  `1040-1044`) spans `906-1072` as claimed.
- Helpers: `_extract_events_from_analysis` at `1163-1166`, `_analyze_players_from_tracks` at
  `1169-1175`, `_calculate_team_metrics` at `1178-1185`.
- Ran the full suite with coverage scoped to this module:
  `.venv/bin/python -m pytest -q --cov=src.pipeline_full --cov-report=term-missing:skip-covered`
  → **251 passed**, `src/pipeline_full.py` 46% statement coverage, and the `Missing` line list
  is: `..., 906-1072, 1077, 1088-1137, 1142-1160, 1165-1166, 1171, 1180, 1190-1211, 1216-1416, 1420`.
  `906-1072`, `1165-1166`, `1171`, `1180` are exactly the ranges named in the claim — zero of
  the 251 tests execute any statement in them.
- `tests/test_e2e_pipeline.py:73-122` (`test_pipeline_orchestrator_with_mock_data`) constructs
  `PipelineOrchestrator(processor=fake_processor)` (line 104), where `fake_processor` (lines
  76-102) is a hand-written stand-in that never calls `src/pipeline_full.py` — confirmed via
  `PipelineOrchestrator.__init__` (`src/pipeline_orchestrator.py:98-106`), which accepts an
  injectable `processor` defaulting to `run_youtube_pipeline` (i.e. `pipeline_full.process_youtube_video`,
  imported at line 19) but happily takes the fake instead.
- `src/pipeline_orchestrator.py:262-270` hardcodes `events_detected=[]` and `total_events=None`
  on every call, regardless of what the processor returns — confirmed by grepping the file for
  any read of `raw_result["events"]`, `["players"]`, or `["team_metrics"]`: zero hits. The
  `events`/`players`/`team_metrics` keys that `pipeline_full.py:1027-1029` puts in its result
  dict are simply never consumed by the orchestrator.
- `test_e2e_pipeline.py:116-117` asserts `result.soccer_classification.total_events is None`
  and `detection_quality is None` — i.e. the test asserts the hardcoded values match the
  hardcoded values. It is internally consistent and green, but exercises none of the code the
  claim is about.

## Claim 2 — `src.main:app` (BarcaAPIServer) has no deployment path

**Verdict: CONFIRMED**, with one addition: it's also a documentation gap, not just an
infra gap.

- `BarcaAPIServer` is defined at `src/live/barca_api.py:45`. `src/main.py:1-10` is a 10-line
  ASGI shim: `server = BarcaAPIServer(); app = server.app`.
- `Dockerfile:55`: `CMD ["python", "-m", "uvicorn", "src.api.main:app", ...]` — not `src.main:app`.
- `docker-compose.yml:24-34` (the only `command:` for the `app` service) also launches
  `src.api.main:app`. The `mlflow` and `dev` services don't run either app module.
- `Makefile:45-46` (`run`) launches `src.api.main:app`; `Makefile:51-52` (`run-live`) launches
  `python -m src.live.run_live`, a CLI tool — not an ASGI server, not `src.main:app`.
- Repo-wide grep for `src.main:app` / `src/main.py` outside `src/main.py` itself: only
  `tests/test_barca_api.py` (which imports `BarcaAPIServer` directly and instantiates it in
  unit tests — never through the ASGI app or a server process).
- Addition: `README.md:55` and `:139` both document `src.main:app` as "live-stream control API"
  with real routes, and `AUDIT_REPORT.md:183`'s own entry-points table lists it as
  `Live API | src.main:app | POST /stream/start, /stream/stop; GET /detections, /checkpoint, /health`
  — treating it as a real, implemented, tested surface. Neither document gives an actual run
  command for it (the batch API has `make run` / Docker instructions; the live API does not).
  The only way to start it is to manually type `uvicorn src.main:app ...` yourself. See Section B.1
  below — this specific "documented but unrunnable" gap isn't called out anywhere in the audit's
  Medium findings even though AUD-030 reviews Compose/deployment wiring in detail.

## Claim 3 — `configs/tracking_tuned.yaml` is dead; values hardcoded elsewhere

**Verdict: CONFIRMED**

- `configs/tracking_tuned.yaml` exists (797 bytes) with `min_confidence: 0.35`,
  `distance_threshold: 120.0`, `max_age: 20`.
- `grep -rln "tracking_tuned" .` (excluding `.venv`/`.git`) → **zero matches** anywhere in the
  repo, including no reference from `dvc.yaml`, `pipeline_full.py`'s CLI arg defaults, CI, or docs.
- `Makefile:54-60` (`run-pipeline`): `--distance-threshold 120.0 --min-confidence 0.35 --max-age 20`,
  hardcoded as literal CLI flags.
- `params.yaml:9-11`: `min_confidence: 0.35`, `distance_threshold: 120.0`, `max_age: 20`,
  independently duplicated.
- All three values match exactly across the dead config file, the Makefile, and params.yaml —
  three independent sources of truth for numbers that are never actually sourced from the
  dedicated config file that exists to hold them.

## Claim 4 — nothing loads `.env`; only `os.environ`/`os.getenv` is used

**Verdict: CONFIRMED**

- `grep -rln "dotenv|python-dotenv|load_dotenv" src/` and repo-wide (excluding `.venv`): zero
  matches. `dotenv` does not appear in `requirements.txt` or `pyproject.toml` either.
- No `env_file:` directive in `docker-compose.yml`; no `.env` sourcing in `Makefile`.
- 21 call sites for `os.environ`/`os.getenv` across `src/`.
- `AUDIT_REPORT.md:191` independently states the same thing: "no `.env` file is loaded or
  tracked automatically" — the audit already knew this; it's not a new finding, just verified.

## Claim 5 — `monitoring.py` has no API task/queue metrics despite `GET /metrics`

**Verdict: CONFIRMED**

- `src/utils/monitoring.py:51-79` defines exactly 9 Prometheus objects, all frame-pipeline
  scoped: `FRAMES_PROCESSED`, `DETECTIONS_MADE`, `TRACKS_CREATED` (counters), `ACTIVE_TRACKS`,
  `GPU_MEMORY_USED`, `CPU_USAGE` (gauges), `PROCESSING_TIME`, `DETECTION_TIME`, `TRACKING_TIME`
  (histograms). No task success/failure/cancellation counter, no queue-wait histogram.
- `src/api/main.py:422-445` (`GET /metrics`) returns only `get_system_metrics()` (cpu/gpu/mem
  from `monitoring.py`) — no task-lifecycle data.
- `grep -n "Counter(|Histogram(|Gauge(|\.inc(|\.observe(" src/api/main.py`: zero hits — the API
  module defines no metrics of its own either.
- Related, feeds Section B: `src/api/main.py:42-44` comments "In-process task state. Multi-worker
  deployments must replace this with a durable shared queue/store" directly above
  `task_storage: dict[str, TaskStatus] = {}` (line 44) — an acknowledged but unaddressed gap.

## Claim 6 — two graph builders, only one used in `src/`; are outputs equivalent?

**Verdict: CONFIRMED usage split; PARTIAL on equivalence — they diverge outside a narrow regime**

- `build_track_graph` (`src/graph/build_graph.py:29`) is the only one imported by
  `pipeline_full.py:35`, used at line `703`.
- `build_track_graph_optimized` (`build_graph.py:129`) is used **only** in
  `tests/test_graph_memory.py` and `tests/test_graph_builder.py` — zero references in `src/`
  outside its own definition.
- Wrote a throwaway comparison script (`compare_graph_builders.py` in this scratchpad) using
  synthetic `Tracklets`/`Tracklet` data and ran it under `.venv/bin/python`:
  - Small/sparse case (5 frames × 10 tracks): nodes 50/50 equal, edges 410/410 equal, edge sets
    (by frame/track identity) identical. Equivalent here.
  - Dense-frame case (3 frames × 60 tracks/frame, over the original's hardcoded 50-per-frame
    "skip dense frame" threshold at `build_graph.py:90`): nodes **180 vs 150** (not even equal
    node counts — the two builders truncate differently), edges 240 vs 2270, only 20 edge-pairs
    shared with the original, 1035 pairs only in the optimized version.
  - Multi-frame case (30 frames × 20 tracks/frame, under the 50-per-frame threshold but large
    enough to hit the original's `max_spatial_edges=1000` cap): nodes equal (600/600), but edges
    2160 vs 7820 — 2830 pairs present only in the optimized output.
  - Reproduced the *actual* comparison already shipped in `tests/test_graph_memory.py::test_small_graph_memory`
    (line 46, `_create_mock_tracks(20, 30)`, i.e. 20 frames × 30 tracks/frame) directly against the
    real repo code: nodes 600/600, edges **2140 vs 2300** (~7% divergence) — inside the test's own
    tolerance (`result1 <= result2 * 1.2`, line 57), so it passes, but only because this
    particular data shape happens to land in a regime where the caps roughly agree.
- Conclusion: the two functions are **not general-purpose interchangeable** — they agree closely
  only for small/sparse inputs or by coincidence of cap alignment; once a frame exceeds 50 tracks
  (the original's own dense-frame skip threshold) or edge counts saturate the original's global
  1000-edge cap in a way the optimized function's per-frame cap doesn't mirror, outputs diverge
  by multiples. Nothing in either docstring documents this.

## Claim 7 — `make smoke` fails under default `PYTHON ?= python3`; `.venv` passes 10/10

**Verdict: CONFIRMED**, with a fuller target-by-target breakdown than the claim states

- `Makefile:1`: `PYTHON ?= python3`.
- `make smoke` reproduced live: `ModuleNotFoundError: No module named 'torch'`, 23 collection
  errors (`Interrupted: 23 errors during collection`), `make: *** [Makefile:43: smoke] Error 2`.
- `.venv/bin/python -m pytest -m smoke -q` → **10 passed, 241 deselected**. Matches the claim
  exactly.
- Full breakdown of Makefile targets under the default `PYTHON=python3`:
  - **Broken, torch missing at collection/import time**: `test`, `test-cov`, `smoke` (confirmed:
    23 collection errors); `run-pipeline` (`src.pipeline_full` imports `src.graph.build_graph`,
    which does `import torch` at module top — confirmed via direct `python3 -c "import
    src.pipeline_full"` → `ModuleNotFoundError: No module named 'torch'`).
  - **Broken, different missing dependency**: `run` (`python3 -c "import src.api.main"` →
    `ModuleNotFoundError: No module named 'slowapi'`, not torch); `lint-all` (`pre_commit` not
    importable under system python3).
  - **Broken, unrelated root cause — namespace-package shadowing**: `build`. `python3 -m build
    --help` → `No module named build.__main__; 'build' is a package and cannot be directly
    executed`. Cause: the repo has a gitignored (`.gitignore:7`) `build/` directory at repo root
    (from a prior local build) that, because CWD is on `sys.path` for `python -m X`, gets
    resolved as an empty namespace package named `build` when the real PyPA `build` package
    isn't installed (`python3 -m pip show build` → not found). `.venv/bin/python -m build --help`
    works fine because the venv has the real package installed, which wins package resolution.
    This is a distinct failure mode from "no torch" and would also bite someone who fixed the
    torch problem by installing torch into system python3.
  - **Import fine, but would fail later at actual model-load time** (deferred torch import via
    `importlib.import_module` inside functions, not at module top): `run-detect`
    (`src/detect/infer.py`), `export` (`src/detect/export_onnx.py`), `run-live`
    (`src/live/run_live.py`). All three `import` cleanly under system python3; they'd only break
    once invoked against real args that reach the lazy-imported torch/ultralytics code.
  - **Safe, doesn't route through `$(PYTHON)`**: `lint` — `ruff check`/`ruff format --check`/`mypy`
    are called as bare commands, resolved via `/home/po/.local/bin/{ruff,mypy}` on `PATH`;
    confirmed both run and `ruff check src` reports "All checks passed!" under this environment.
  - **Not exercised** (would mutate environment or is a no-op by inspection): `setup`,
    `setup-all` (pip install — skipped per "do not pip install"), `clean` (prints a refusal
    message, does nothing).

---

## Extra section A — confidence inflation check

1. **`tests/test_e2e_pipeline.py::test_pipeline_orchestrator_creation`** (lines 66-71):
   ```python
   orchestrator = PipelineOrchestrator()
   assert orchestrator is not None
   assert hasattr(orchestrator, "process_youtube_video")
   assert hasattr(orchestrator, "config")
   ```
   `PipelineOrchestrator()` has no custom `__new__`, so a successful constructor call can never
   return `None` — `assert orchestrator is not None` cannot fail short of the constructor itself
   raising, which would abort the test before that assertion runs anyway. The two `hasattr`
   checks only confirm names exist, not that the method works. This test can fail only on a
   total constructor crash or an outright attribute rename; it protects nothing about actual
   orchestration behavior, yet counts as one of the "251 passed."

2. **`tests/test_graph_memory.py::TestGraphMemory::test_small_graph_memory`** (lines 43-58): the
   only shipped test that compares `build_track_graph` against `build_track_graph_optimized`.
   Its one-sided 20%-tolerance assertion (`result1["edge_index"].shape[1] <= result2[...] * 1.2`)
   combined with its specific data shape (20 tracks/frame, well under the original's 50-per-frame
   dense-skip threshold) lands in a regime where the two happen to be close (~7% divergence, as
   reproduced above), so it passes and reads as "these two are interchangeable." As claim 6
   shows directly, the same functions diverge by multiples (up to ~9x more edges, or outright
   different node counts) once per-frame track density crosses 50 or the original's global edge
   cap engages differently — exactly the regime the original function's own code comments
   describe its dense-frame-skip and cap logic as being for. That regime is never tested.

3. **`tests/test_imports.py::test_package_imports`** (lines 5-10): asserts `import src` and its
   submodules succeed, plus two `hasattr` checks (`detect.run_inference`, `track.ByteTrackRuntime`).
   It contributes a green result to the 251-count while only proving the packages are
   syntactically importable in an environment that already has every heavy dependency (torch,
   ultralytics, etc.) installed. Per claim 7, "imports cleanly" is exactly the property that
   silently differs across environments for other modules (`src.api.main` fails to import under
   system python3 for an unrelated missing-dependency reason) — this test gives no signal about
   that risk because it only ever runs where the venv already masks it.

## Extra section B — missed production blockers

1. **`/health` reports "healthy" without checking that the model the deployment needs actually
   exists.** `src/api/main.py:249-272` (`health_endpoint`) reports `status="healthy"` whenever
   `orchestrator is not None`. `PipelineOrchestrator.__init__` (`src/pipeline_orchestrator.py:98-106`)
   only loads a YAML config — it never touches model weights. The repo ships no `models/`
   directory at all (`ls models/` → "No such file or directory" locally), and
   `docker-compose.yml:46` bind-mounts `./models:/app/models:ro` — Docker silently creates an
   empty directory for a missing bind-mount source. So a default `docker compose up app` starts,
   passes its own healthcheck (`docker-compose.yml:52-61`, which just curls `/health`)
   indefinitely, while every real `POST /analyze` request would fail deep in the pipeline the
   first time it tries to load the (absent) YOLO checkpoint.

2. **Single shared `API_KEY` plus a global, unscoped `task_storage` dict means no per-tenant
   isolation, with or without auth enabled.** `src/api/main.py:44` (`task_storage: dict[str,
   TaskStatus] = {}`) has no owner/caller field. `GET /tasks` (`src/api/main.py:363-379`) returns
   every task in the process — including each task's YouTube URL and results — filtered only by
   optional `status`, with no per-caller scoping. With `API_REQUIRE_KEY` left at its documented
   default of `false` (`src/api/main.py:50-52`, confirmed also by `AUDIT_REPORT.md`'s AUD-010),
   this is fully open to any network caller. Even with auth turned on, the design is a single
   shared secret (`src/api/main.py:171-174`, `secrets.compare_digest(supplied_key, API_KEY)`) —
   any holder of that one key can enumerate every other caller's submitted URLs and analysis
   output. This is distinct from what AUD-007/AUD-010 flag (unbounded retention, auth being
   off by default) — it's a data-isolation gap that persists even when auth is correctly enabled.

3. **`YOLO_WEIGHTS` silently defaults to the exact filename of the untracked, unverified weights
   file sitting in the repo root.** `src/pipeline_orchestrator.py:126`: `weights=os.getenv("YOLO_WEIGHTS",
   "yolov8n.pt")` — a bare relative filename, resolved against CWD. The repo root has
   `yolov8n.pt` (6,549,796 bytes, `*.pt` gitignored, not DVC-tracked, dated Jan 6 2026,
   sha256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`). Anyone running the
   pipeline from the repo root — the natural place to run it from — gets this file loaded with
   zero prompt, directly contradicting README's own stated policy ("Supply a trusted YOLO
   checkpoint with known provenance and license," `README.md:89`). `AUDIT_REPORT.md`'s AUD-020
   already flags the file's provenance as unverified, but doesn't note that the default code
   path picks it up silently rather than requiring an explicit operator choice.

4. **The audit's own entry-points table documents `src.main:app` as real and tested but its
   Compose/deployment review never flags that nothing actually runs it.** See claim 2 above:
   `AUDIT_REPORT.md:183` lists it in the same table as the batch API, and AUD-030 reviews Compose
   model-mount/service wiring in detail without noting this specific service has zero path to
   start (no Dockerfile CMD, no compose service, no Makefile target — only manual
   `uvicorn src.main:app ...`, undocumented anywhere in README's actual run instructions).

5. **`make build` is broken under the default `PYTHON=python3` for a reason unrelated to torch.**
   See claim 7 for full reproduction: a gitignored, locally-generated `build/` directory at the
   repo root shadows the real PyPA `build` tool as an empty namespace package once system python3
   (which doesn't have the real `build` package installed) is used. This means "just install
   torch into system python3" would not actually fix `make build` — a different, undocumented
   failure mode a new contributor would hit next.

6. **Rate limiting is keyed by raw remote address with no reverse-proxy trust configuration, in
   a deployment shape that requires a reverse proxy.** `src/api/main.py:25,39`:
   `Limiter(key_func=get_remote_address)` — slowapi's default, which reads `request.client.host`
   directly. There is no `ProxyHeadersMiddleware`/forwarded-header trust configuration anywhere
   in `src/api/main.py`. `docker-compose.yml:43` binds the app service to `127.0.0.1:${APP_PORT:-8000}`
   only — loopback, which only makes sense if a reverse proxy sits in front of it for real
   traffic. In that shape, every real client's rate-limit bucket collapses to the proxy's own
   address, meaning slowapi's per-IP limits (`@limiter.limit("10/minute")` on `/analyze`, etc.)
   effectively rate-limit the whole service to one shared bucket rather than per caller — the
   opposite of the abuse protection the comment at `src/api/main.py:38` claims.

DONE
