# fifa-soccer-ds — ground-truth exploration report

Repository: `/home/po/projects/work/fifa-soccer-ds`, detached HEAD mid-rebase (`.git/rebase-merge`
present), large uncommitted working tree. All findings below are read from that uncommitted
worktree, which is treated as the current state of the project. Verification commands (`pytest`,
`coverage`, `git ls-files`, etc.) were run read-only; no git state was changed.

---

## 1. Architecture as built

Two independent FastAPI applications exist, not one:

- **Batch analysis API** — `src/api/main.py:92-99` (`app = FastAPI(...)`), exposed via console
  script indirectly (uvicorn target `src.api.main:app`). Routes: `GET /` (`main.py:236`),
  `GET /health` (`:248`), `POST /analyze` (`:276`), `GET /tasks/{task_id}` (`:344`),
  `GET /tasks` (`:364`), `DELETE /tasks/{task_id}` (`:397`), `GET /metrics` (`:423`),
  `GET /pipeline/info` (`:449`), `POST /tactical/compute` (`:531`), `GET /tactical/info` (`:608`),
  `GET /tactical/results/{task_id}` (`:653`). This is the app the Dockerfile
  (`Dockerfile:55`) and `docker-compose.yml:24-34` actually launch.
- **Live inference API** — `src/main.py:1-10` builds `BarcaAPIServer().app` from
  `src/live/barca_api.py:44-78`. Routes: `POST /stream/start` (`barca_api.py:84`),
  `POST /stream/stop` (`:123`), `GET /detections` (`:129`), `GET /checkpoint` (`:139`),
  `GET /health` (`:143`). This app is **not** referenced anywhere in `Dockerfile`,
  `docker-compose.yml`, or `Makefile` — see gap 1 in §6.

Data flow, traced through imports rather than the README:

1. **Frame path** (`fifa-pipeline` CLI / `src.pipeline_full:main`, `pipeline_full.py:1214`):
   `_discover_frame_files` (`:224`) → `_frame_is_safe_to_decode` (`:235`, size/format/dimension
   checks) → `load_model`/`detector.predict` from `src/detect/infer.py` (imported at
   `pipeline_full.py:34`) → `filter_detections` from `src/track/pipeline.py` (`:38`) →
   `ByteTrackRuntime.update` from `src/track/bytetrack_runtime.py` (`:37`) → per-frame JSON writes
   (`detections/`, `tracks/`, `overlays/`) → `build_track_graph` from `src/graph/build_graph.py`
   (`:35`) → optional `predict_positions` from `src/graph/gcn_position_classifier.py` (imported
   lazily at `pipeline_full.py:741`, only when `--gnn-weights` is set) → optional tactical stage
   using `src/analytics/team_classifier.py` + `src/calib/pitch_transform.py` +
   `src/analytics/tactical.py` (imported at `pipeline_full.py:51-53`, gated by
   `TACTICAL_AVAILABLE`) → `pipeline_summary.json`.
2. **YouTube path**: `process_youtube_video` (`pipeline_full.py:878`) →
   `SoccerClassifier.classify_youtube_content` (`src/classify/soccer_classifier.py`, imported at
   `pipeline_full.py:61`) → if soccer, `YouTubeDownloader.download_video`
   (`src/youtube/video_downloader.py`, imported at `:64`) → `_extract_frames_from_video`
   (`pipeline_full.py:1080`) → re-enters the frame path via `process_frames_directory` (`:985`).
3. **Batch API path**: `POST /analyze` (`api/main.py:276`) validates via `AnalyzeRequest`
   (`:178-200`, which calls `validate_youtube_url` from `src/schemas.py` and
   `validate_output_name` from `src/utils/output_paths.py`) → creates an `asyncio.Task` running
   `process_video_analysis` (`api/main.py:675`) → `PipelineOrchestrator.process_youtube_video`
   (`src/pipeline_orchestrator.py:138`) → `asyncio.to_thread(self.processor, ...)` where
   `self.processor` defaults to `pipeline_full.process_youtube_video`
   (`pipeline_orchestrator.py:19,102`) → result reshaped into the `PipelineOutput` schema by
   `_to_pipeline_output` (`:171-288`), which independently re-derives fields such as
   `events_detected=[]`/`total_events=None` (`:266-267`) rather than reusing
   `pipeline_full.py`'s own `capabilities` dict (see spot-check #3 in §5 — these are two separate,
   only partially-overlapping "no fabrication" implementations).
4. **Live path**: `fifa-live` CLI (`src/live/run_live.py:257`, `run_live_pipeline` at `:144`) is a
   synchronous OpenCV capture loop, independent of the `BarcaAPIServer` live API. The API's live
   path is `POST /stream/start` (`barca_api.py:84`) → `validate_rtsp_target` from
   `src/utils/network_security.py:44` (host/IP allow-list, DNS resolution, private-address
   rejection) → `RTSPCapture` (`src/live/rtsp_capture.py:19`, background thread + bounded queue
   with reconnect) → a daemon inference thread (`barca_api.py:160` `_inference_loop`) polling
   `capture.iter_frames()` and writing into a lock-protected `_latest_detections` buffer that
   `GET /detections` reads.
5. **Training path** (`fifa-train-gnn` / `src.models.train_gcn:main`): loads a PyG dataset only
   through `load_graph_dataset` (`src/models/train_gcn.py:38-53`), which refuses to deserialize
   pickle unless `--trust-pickled-dataset` is passed. `src/train/weekly_retrainer.py` is a
   separate, library-only orchestrator (`WeeklyRetrainer.schedule_retrain`, `:722`) requiring
   caller-supplied loaders/evaluator; its own `main()` (`:796-801`) refuses to run and returns
   exit code 2, confirming the audit's "no scheduler or deployment launcher" characterization.
6. **DVC path**: `dvc.yaml:2-24` defines one stage, `analyze_frames`, which shells out to
   `python -m src.pipeline_full` using `params.yaml` values — the same frame path as (1), not a
   separate code path.

Configuration precedence, as implemented: CLI args (`argparse` defaults in `pipeline_full.py:1214`
and `run_live.py:219`) > environment variables read ad hoc via `os.getenv` in ~10 different files
(`api/main.py:47-50`, `output_paths.py:25`, `network_security.py:19,25`, `barca_api.py:30`, etc.)
> `configs/youtube_pipeline.yaml` (only consumed by `PipelineOrchestrator._load_config`,
`pipeline_orchestrator.py:108-119`) > `params.yaml` (DVC only). No central config object ties
these together — see gap 2 in §6.

---

## 2. Module inventory

`src/` totals 55 Python files, 14,528 lines (`wc -l src/**/*.py`). Per-package line totals and
notes (coverage % from a live `pytest --cov=src --cov-branch` run, §3):

| Package | Lines | Key files | Coverage notes | Quality notes |
| --- | ---: | --- | --- | --- |
| `api` | 775 | `main.py` (755) | main.py 73% | Single 755-line file mixes routes, Pydantic models, CORS/API-key middleware, and exception handlers. `youtube_endpoints.py` (10 lines) is a pure re-export shim to `api.main:app` (`youtube_endpoints.py:8`) — dead-weight indirection, not dead code. |
| `analytics` | 919 | `tactical.py` (297), `team_classifier.py` (350), `_pitch_control.py` (244) | tactical.py 80%, team_classifier.py 35% | `team_classifier.py` is large (350 lines) but only a third covered; its K-Means clustering path (`extract_jersey_color`, `classify_tracks`) is the least-tested piece of the tactical stack. |
| `calib` | 996 | `homography.py` (577), `pitch_transform.py` (414) | homography.py 44%, pitch_transform.py 55% | `homography.py` is the single largest calib file and one of the least-covered — many of its ~577 lines are singular-matrix/degenerate-input branches (missing ranges 261-305, 433-482 in `coverage.xml`) that are only partially exercised. |
| `classify` | 563 | `soccer_classifier.py` (single file) | 35% | Contains two of the repo's fourteen functions over 80 lines: `classify_youtube_content` (`soccer_classifier.py:55`, ~109 lines) and `_analyze_image_for_soccer` (`:216`, ~92 lines). |
| `data` | 678 | `la_liga_loader.py` (663) | 38% | `ParallelFrameExtractor` (`la_liga_loader.py:47`) uses a `ThreadPoolExecutor`; the Kaggle download/extraction path is explicitly noted as network-mocked in tests (`weekly_retrainer.py:520` comment "network side effects mocked in tests"), consistent with its low coverage. |
| `detect` | 1,727 | `infer.py` (369), `train_yolo.py` (360), `export_trt.py` (386), `yolo_lora_adapter.py` (306), `export_onnx.py` (213) | infer.py 50%, train_yolo.py 65%, export_trt.py 19%, export_onnx.py 72% | Largest package. `export_trt.py`'s low coverage is hardware-gated (TensorRT absent), not a hygiene issue — see spot-check #10. `build_engine` (`export_trt.py:115`, ~123 lines) and `fine_tune_loop` (`train_yolo.py:186`, ~171 lines) are two of the largest functions in the repo. |
| `eval` | 616 | `evaluate_pipeline.py` (471), `metrics.py` (140) | evaluate_pipeline.py 80%, metrics.py 37% | `metrics.py` is small (140 lines) but under-tested relative to its role as the shared accuracy/F1 implementation used by training/retraining gates. |
| `graph` | 516 | `build_graph.py` (286), `gcn_position_classifier.py` (218) | 92% / 90% | Best-covered package in the repo. Two separate graph builders exist — `build_track_graph` (`build_graph.py:29`, ~97 lines) and `build_track_graph_optimized` (`:129`, ~118 lines) — worth checking for logic drift/duplication between them; not flagged in the audit. |
| `live` | 727 | `barca_api.py` (269), `run_live.py` (294), `rtsp_capture.py` (131) | barca_api.py 76%, rtsp_capture.py 16%, run_live.py 0% | Two independent "live" entry points (see §1). `run_live.py` is entirely unexecuted by tests (`coverage.xml`: 0/163 statements) and is not part of any deployment surface (Docker/Compose/Makefile) — see gap 1. |
| `models` | 630 | `train_gcn.py` (578), `gcn.py` (47) | train_gcn.py 35%, gcn.py 97% | `_run_training_impl` (`train_gcn.py:278`, ~153 lines) is one of the largest functions in the repo; the explicit pickle-trust gate (`:38-53`) is well isolated but the surrounding training loop is under-tested. |
| `track` | 555 | `bytetrack_runtime.py` (382), `pipeline.py` (160) | 82% / 49% | `ByteTrackRuntime.update` (`bytetrack_runtime.py:204`, ~119 lines) is a large stateful method; well covered relative to its size. `pipeline.py` (NMS/area/confidence filtering) is comparatively under-tested at 49%. |
| `train` | 810 | `weekly_retrainer.py` (single file, 805 lines) | 81% | Largest single non-`pipeline_full.py` file in the repo; also one of the better-covered given its concurrency/locking responsibilities (flock locks, atomic writers, semantic versioning all in one file — arguably should be split, though the audit already flags this class of file under AUD-015). |
| `utils` | 1,358 | `health_checks.py` (525), `mlflow_helper.py` (308), `monitoring.py` (216) | health_checks.py 17%, mlflow_helper.py 49%, monitoring.py 57%, network_security.py 81%, output_paths.py 87% | `health_checks.py` is simultaneously the largest utils file and one of the three worst-covered files in the entire repo (see §3), despite its `HealthChecker.check_all_components` (`health_checks.py:76`) claiming to check 8 subsystems. |
| `youtube` | 1,373 | `metadata_parser.py` (542), `video_downloader.py` (411), `audio_extractor.py` (405) | 15% / 24% / 18% | All three files sit in the worst-covered band in the repo — the single worst-covered package by line count (1,373 lines at 15-24% each). |

Top-level files outside packages: `pipeline_full.py` (1,420 lines, 46% covered — by far the
largest file in the repo, combining CLI, batch pipeline, and YouTube pipeline logic in one module,
consistent with AUD-015's "roughly 1,400-line pipeline_full.py" characterization),
`pipeline_orchestrator.py` (314 lines, 75%), `schemas.py` (519 lines, 85%), `main.py` (10 lines,
0% — trivial live-API ASGI shim, not a real coverage gap).

No `TODO`/`FIXME`/`XXX`/`HACK` markers exist anywhere under `src/` (`grep -rn` returned nothing) —
no self-flagged dead code or shortcuts left in comments.

---

## 3. Test suite map

36 files under `tests/`, **251 tests** (`pytest --collect-only -q` and a live `pytest -q` run both
report 251; the live run passed all 251 in 26.71s with the two `torch.jit.script` deprecation
warnings the audit describes — see spot-check #1).

Rough thematic breakdown by test count: security-themed (`test_api_security.py` 13,
`test_race_condition_protection.py` 15, `test_security.py` 9, `test_network_security.py` 5,
`test_output_path_security.py` 5, `test_video_downloader_security.py` 3 — 50 tests total),
version/retraining (`test_version_management.py` 17, `test_weekly_retrain.py` 4,
`test_train_yolo.py` 4), pipeline/tracking (`test_pipeline_full.py` 2,
`test_pipeline_enhanced.py` 4, `test_detect_smoke.py` 6, `test_track_stub.py` 4,
`test_track_graph.py` 2, `test_graph_builder.py` 3, `test_graph_memory.py` 6), GNN
(`test_gcn_position.py`, `test_gcn_shapes.py`, `test_gcn_training_integrity.py` 4,
`test_gnn_inference.py`), API/E2E (`test_e2e_pipeline.py` 11, `test_api_youtube.py` 1,
`test_barca_api.py` 2, `test_mlflow_integration.py` 9), calibration/tactical
(`test_calibration_validation.py` 7, `test_tactical_validation.py` 3, `test_team_classifier.py` 2),
export (`test_export_onnx.py` 8), and data/eval (`test_kaggle_loader.py` 5,
`test_evaluation_metrics.py` 5, `test_metadata_integrity.py` 3).

**Worst 10 source files by `coverage.xml`/live-run coverage** (`Cover` column from
`pytest --cov=src --cov-branch --cov-report=term-missing`, run 2026-07-20):

| Rank | File | Coverage | Note |
| --- | --- | ---: | --- |
| 1 | `src/main.py` | 0% | Trivial 5-statement ASGI shim (`server = BarcaAPIServer(); app = server.app`) — not a substantive gap. |
| 2 | `src/live/run_live.py` | 0% | 163 statements, 50 branches, zero test execution. Real gap — see §6 gap 1. |
| 3 | `src/youtube/metadata_parser.py` | 15% | 173 statements; only the narrow `_create_validated_metadata` helper is exercised (`test_e2e_pipeline.py:144-155`, `test_metadata_integrity.py`). |
| 4 | `src/live/rtsp_capture.py` | 16% | Reconnect/queue-overflow logic (`rtsp_capture.py:69-97`) is entirely unexercised. |
| 5 | `src/utils/health_checks.py` | 17% | Largest file in `utils/`; almost none of its 8 component checks (`health_checks.py:89-98`) run under test. |
| 6 | `src/youtube/audio_extractor.py` | 18% | Whisper/librosa integration path untested (optional-dependency heavy). |
| 7 | `src/detect/export_trt.py` | 19% | Hardware-gated (no TensorRT/GPU in test env) — expected, not a hygiene defect. |
| 8 | `src/utils/vis.py` | 22% | Small (22 statements) visualization helper, mostly unused by tests. |
| 9 | `src/youtube/video_downloader.py` | 24% | Core download/size-limit logic (`video_downloader.py:100-152`) only partially covered despite security tests targeting it. |
| 10 | `src/analytics/team_classifier.py` | 35% | Tied with `src/classify/soccer_classifier.py` (35%) and `src/models/train_gcn.py` (35%); K-Means clustering path under-exercised. |

Beyond the top 10: `src/pipeline_full.py` at 46% is a significant weak spot given it is the
largest file in the repo and the shared implementation for both CLI entry points — in particular,
lines 906-1072 (the entire "soccer" branch of `process_youtube_video`, which builds the
API-facing `events`/`players`/`team_metrics`/`capabilities` result) show **zero** coverage; see
spot-check #3 in §5 for the consequence.

---

## 4. CI/CD and packaging

`.github/workflows/ci.yml` defines three jobs, all reproduced by direct inspection:
- **quality** (45 min timeout): installs CPU Torch/Vision from the PyTorch index, then
  `pip install -e ".[audio,data,dev,export,mlops]"` (`ci.yml:29`, i.e. all five optional-dependency
  groups — matches AUD-031's "install all supported extras" fix), runs `ruff check`/`ruff format`,
  `mypy src`, `pre-commit run --all-files`, `pytest --cov=src --cov-branch ...`, then
  `python -m build` and a `--dry-run` wheel install.
- **security** (30 min timeout): `bandit -c pyproject.toml -r src`, then `pip-audit --local`
  followed by a second `pip-audit` run against a resolved public-version closure — matching the
  audit's "two complementary pip-audit scans" claim.
- **configuration** (10 min timeout): `docker compose ... config --quiet` for both base and
  GPU-overlay profiles, then `docker build --check --target production .`.

`.github/workflows/codeql.yml` runs on push/PR to `master` plus `cron: "30 6 * * 1"` (Mondays
06:30 UTC — matches the audit's schedule claim exactly). `.github/workflows/stale.yml` runs
`cron: "0 0 * * *"` (daily 00:00 UTC — also matches). `.github/workflows/release.yml` triggers on
push to `master` and runs `googleapis/release-please-action` with `release-type: python`. All
three action references use `owner/repo@<40-char-sha> # vX.Y.Z` pins, consistent with the audit's
"immutable commit SHAs" claim. `.github/dependabot.yml` covers `pip` (weekly, grouped into
`pytorch`/`api`/`quality`) and `github-actions` (weekly) ecosystems.

**Dockerfile**: multi-stage (`runtime` → `development`/`mlops`/`production`), pinned by digest
(`nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04@sha256:...`, `Dockerfile:5`). Non-root UID 10001
(`:47-52`), `tini` entrypoint, `HEALTHCHECK` hitting `/health` on port 8000. **The default/production
CMD only ever launches the batch API** (`Dockerfile:55`: `uvicorn src.api.main:app`) — the live
`BarcaAPIServer` app (`src.main:app`) has no image target, Compose service, or Makefile target at
all (see §6 gap 1). The `production` stage (`Dockerfile:77`) is a bare alias of `runtime` with no
additional steps.

**docker-compose.yml**: `app` service builds from `production` target, mounts `./models`/`./data`
read-only, binds to `127.0.0.1` only, sets `read_only: true`/`cap_drop: [ALL]`/`pids_limit: 256` at
the top-level `x-runtime-security` anchor (`docker-compose.yml:3-16`) applied to all three
services (`app`, `mlflow`, `dev`). `mlflow` service is `profiles: ["mlops"]`-gated and runs its own
SQLite-backed server on port 5000, separate from `APP_MLFLOW_TRACKING_URI` — matching AUD-030's
"separate app tracking URI" fix. No service in this file runs the live API.

**pyproject.toml**: `pyproject.toml` is the sole packaging source of truth (`requirements.txt`
contains only `-e ".[dev]"` per `requirements.txt`, confirmed). Four console scripts
(`pyproject.toml:72-76`): `fifa-detect`, `fifa-live`, `fifa-pipeline`, `fifa-train-gnn` — matches
the audit exactly, and matches the actual `def main()` entry points found in
`src/detect/infer.py`, `src/live/run_live.py`, `src/pipeline_full.py`, `src/models/train_gcn.py`.
`requires-python = ">=3.12,<3.13"`, `license = "MIT"` with `license-files = ["LICENSE"]` (SPDX
form, not the legacy `{file = ...}` table). Direct dependencies are all exact-pinned
(`torch==2.13.0`, `fastapi==0.139.2`, etc.); five optional-dependency groups (`audio`, `data`,
`dev`, `export`, `mlops`) are each exact-pinned as well.

---

## 5. Audit spot-check

Ten consequential, concrete claims independently verified against the actual code/tests/tools,
not against the audit's own prose.

**1. Executive summary — "251 passed, two [Torch/PyG] deprecation warning emissions, 54% total
[branch-aware coverage]"**
Verdict: **CONFIRMED**. Ran `.venv/bin/python -m pytest -q` directly: `251 passed, 2 warnings in
14.32s` (a second run with coverage: `251 passed, 2 warnings in 26.71s`), both showing exactly two
emissions of `torch.jit.script is deprecated` from `.venv/lib/python3.12/site-packages/torch/jit/_script.py:1488`.
`pytest --cov=src --cov-branch` reports `TOTAL 6853 2905 1924 328 54%` — i.e. 54% exactly.

**2. AUD-016 (Medium, Test coverage) — "live runner 0%, RTSP 16%, TensorRT 19%, health 17%,
YouTube helpers 15–24%"**
Verdict: **CONFIRMED**, with exact numeric matches from the live coverage run above:
`src/live/run_live.py` 0%, `src/live/rtsp_capture.py` 16%, `src/detect/export_trt.py` 19%,
`src/utils/health_checks.py` 17%, `src/youtube/{metadata_parser,audio_extractor,video_downloader}.py`
15%/18%/24% respectively — all inside the claimed 15–24% band.

**3. AUD-002 (High, Functional/ML truthfulness) + Feature-reality row "Event/pass/touch/PPDA
statistics — Fabricated code removed — Tests: E2E tests assert absence"**
Verdict: **PARTIAL**. The code-removal half is real: `pipeline_full.py:1040-1044` marks
`"event_detection"`, `"player_statistics"`, `"team_possession_ppda_field_tilt"` all
`"not_implemented"`; helper functions `_extract_events_from_analysis` (`:1163-1166`, returns `[]`),
`_analyze_players_from_tracks` (`:1169-1175`, returns track IDs plus an explicit
"limitations" string), and `_calculate_team_metrics` (`:1178-1185`, same pattern) genuinely avoid
inventing stats. But the "Tests" half is overstated: the live coverage run shows lines 906-1072 of
`pipeline_full.py` — the entire "soccer" branch of `process_youtube_video`, including the
`capabilities` dict and all three helper functions above — at **0% coverage** (`Missing` column:
`906-1072`). The only test that resembles an E2E check of this path,
`test_e2e_pipeline.py::test_pipeline_orchestrator_with_mock_data` (`:73-122`), substitutes a
`fake_processor` (`:76-102`) that bypasses `pipeline_full.py` entirely, and its assertion
`result.soccer_classification.total_events is None` (`:116`) is checking a field that
`pipeline_orchestrator.py:266-267` **hard-codes** independently (`events_detected=[]`,
`total_events=None`), not a field derived from `pipeline_full.py`'s anti-fabrication code. No test
in the 251-test suite actually exercises the code this audit row credits with the fix.

**4. AUD-003 (High, Path security/data integrity) — "UUID task directories, resolved
containment, constrained cleanup"**
Verdict: **CONFIRMED**. `api/main.py:289`: `task_id = f"task_{uuid.uuid4().hex}"`.
`output_paths.py:41-47` (`create_analysis_output_dir`): resolves the target path and raises
`ValueError` if `target.parent != output_root`. `output_paths.py:50-67`
(`remove_analysis_output_dir`): rejects symlinks, requires `target.is_relative_to(output_root)`,
and requires an ownership marker file (`_OUTPUT_MARKER`, `:10,46,64`) before `shutil.rmtree`.

**5. AUD-004 (High, SSRF/resource exhaustion) — "strict ... public RTSP policy, redirect
revalidation"**
Verdict: **CONFIRMED** for the RTSP half. `network_security.py:44-74` (`validate_rtsp_target`):
rejects non-`rtsp(s)://` schemes, missing hostnames, URL fragments, and URLs over 2048 chars;
resolves the hostname via `socket.getaddrinfo` (`:32`) and rejects the target if **any** resolved
address is non-global (`:71-74`) unless the host is explicitly allow-listed via
`RTSP_ALLOWED_HOSTS` or `RTSP_ALLOW_PRIVATE=true` is set (`:16-25`). Redirect-revalidation and
byte/duration bounds for the YouTube download path were not independently re-verified line-by-line
in this pass; the RTSP mechanism alone is sufficient to confirm the claimed SSRF-hardening pattern
exists in code, not just in the audit's prose.

**6. AUD-005 (High, Unsafe deserialization/supply chain) — "weights_only where possible; explicit
trusted-PyG opt-in"**
Verdict: **CONFIRMED**. `grep -rn weights_only src/` shows `weights_only=True` at
`graph/gcn_position_classifier.py:182`, `train/weekly_retrainer.py:577`,
`models/train_gcn.py:271`, `detect/train_yolo.py:88,216` — five of six `torch.load` call sites.
The sixth, `models/train_gcn.py:38-53` (`load_graph_dataset`), uses `weights_only=False`
(unavoidable for legacy PyG pickled datasets) but is gated: it raises `ValueError` immediately
(`:45-48`) unless called with `trust_pickle=True`, which only happens via the explicit
`--trust-pickled-dataset` CLI flag (`train_gcn.py:513,551`).

**7. AUD-006/AUD-024 (High/Medium, Reliability + subprocess/lock robustness) — "flock, private
runtime lock path" replacing "a shared literal /tmp lock path"**
Verdict: **CONFIRMED**. `weekly_retrainer.py:285-299` (`RetrainingLock.__init__`): when no
explicit lock file is given, it builds `{XDG_RUNTIME_DIR or tempdir}/fifa-soccer-ds-{uid}`, creates
it with `mode=0o700`, opens it `O_NOFOLLOW`, and verifies `os.fstat(...).st_uid == os.getuid()`
before use — not a shared, world-writable `/tmp` path. Locking itself uses `fcntl.flock(...,
LOCK_EX | LOCK_NB)` with a poll/timeout loop (`:317-325`).

**8. AUD-009 (High, Tactical correctness) — "Require valid homography and both teams; return
explicit unavailable/failure" instead of "50/50 or 100/0 shares"**
Verdict: **CONFIRMED**. `team_classifier.py:302-303`: when sample counts are insufficient,
`classify_tracks` logs a warning and returns `self._unknown_assignments(tracklets)`
(`:329-338`, `team_id=-1`) rather than guessing. `pipeline_full.py:616-618`: any track whose
`assignment.team_id not in {0, 1}` is dropped before tactical computation; `:629-634`: any frame
where `{player.team_id for player in players} != {0, 1}` is skipped entirely with a debug log,
rather than computing a fabricated split.

**9. AUD-015 (Medium, Maintainability/performance) — "process_frames_directory cyclomatic
74/cognitive 200 (603 lines)"**
Verdict: **PARTIAL**. The line count is verified essentially exactly: `process_frames_directory`
spans `pipeline_full.py:272` to the line before the next top-level `def` at `:877`, i.e. lines
272-874 inclusive = 603 lines by inclusive counting (an independent AST-based measurement of
`end_lineno - lineno` gives 602, off by the usual one-based/exclusive counting convention — the two
methods agree). The cyclomatic-complexity (74) and cognitive-complexity (200) figures could not be
independently reproduced in this pass: no complexity-analysis tool (radon, lizard, xenon, etc.) is
installed in `.venv` or referenced anywhere in `pyproject.toml`/CI, and this exploration did not
install one. Manual inspection confirms the function is exceptionally branch-heavy (roughly 30+
`try`/`except` blocks and `if` statements across the 603 lines, several nested three deep), which
is directionally consistent with a very high complexity score, but the specific numbers 74/200 are
unverified — the audit cites its own prior "final 1,642-node graph search" as the source, which was
not re-run here.

**10. Feature-reality row "TensorRT export — Real TensorRT path and explicit failure; no fake
engine"**
Verdict: **CONFIRMED**. `export_trt.py:105-112` (`_require_modules`): `import tensorrt` inside a
`try/except ImportError`, re-raised as an explicit `ImportError` — no fallback engine is produced.
`build_engine` (`:115-238`) uses the real `tensorrt.Builder`/`tensorrt.OnnxParser`/
`tensorrt.NetworkDefinitionCreationFlag` API, raises `RuntimeError` on parse failure (`:176-178`)
or an empty serialized engine (`:206-207`), and publishes atomically via a same-directory temp file
plus `os.replace` (`:210-227`) — matching AUD-029's atomicity fix pattern. INT8 export explicitly
raises `ValueError("INT8 export requires a calibration pipeline, which is not implemented")`
(`:97-100,141-142`) rather than silently no-opping.

**Summary**: 8 of 10 claims fully CONFIRMED with direct reproduction (including re-running the
entire test suite and coverage report live); 2 are PARTIAL — one because the audit's own "Tests"
credit for AUD-002/the events-removal feature row does not hold up against the actual coverage
data (a real, previously unflagged gap — see §6 gap 3), and one because a specific complexity
metric (AUD-015) could not be independently recomputed without tooling this pass did not install.

---

## 6. Independent gaps list

1. **The live inference API has no deployment path at all.** `src.main:app` (`BarcaAPIServer`,
   `barca_api.py`) is real, tested-to-76%, security-hardened code (SSRF-checked RTSP, bounded
   queue, reconnect logic) — but it is absent from `Dockerfile` (only `src.api.main:app` is ever
   the CMD, `Dockerfile:55`), absent from `docker-compose.yml` (no service references
   `src.main` or `BarcaAPIServer`), and absent from `Makefile` (`run-live` at `Makefile:47-48`
   launches the unrelated `src.live.run_live` CLI loop, not the FastAPI server). Anyone who wants
   to run the "Live API" row from the audit's own entry-points table has to hand-invoke
   `uvicorn src.main:app` with no documented port, health-check, or resource-limit configuration —
   the opposite of the batch API's carefully hardened Compose profile.

2. **Configuration is scattered across four independent, partially-overlapping surfaces with no
   reconciliation.** `configs/youtube_pipeline.yaml` (consumed only by `PipelineOrchestrator`),
   `params.yaml` (DVC only), `.env.example` (documents ~20 env vars read ad hoc via `os.getenv` in
   at least 8 files), and `PipelineConfig`'s own dataclass defaults in `pipeline_full.py:111-198`.
   `configs/tracking_tuned.yaml` (26 lines, `distance_threshold=120.0`, `min_confidence=0.35`,
   `max_age=20`) is **dead**: `grep -rn tracking_tuned` across the entire repository (`.py`,
   `.yaml`, `.md`) returns zero references — it is neither loaded by any code nor mentioned in any
   doc. Confusingly, its exact values (120.0/0.35/20) are silently duplicated as hardcoded flags in
   `Makefile:41-45` (`run-pipeline` target) and `params.yaml:9-11`, so three of the four
   config surfaces agree by coincidence, not by construction — a future edit to one will silently
   diverge from the others.

3. **The primary advertised YouTube-analysis code path has zero test coverage** (developed in
   detail as spot-check #3, §5). `pipeline_full.py:906-1072` — the branch that builds the
   `events`/`players`/`team_metrics`/`capabilities`/`written_report_md` fields returned by
   `POST /analyze` for soccer-classified videos — is never executed by any of the 251 tests. This
   is the single largest uncovered block of business logic in the repository and sits directly in
   the feature the README leads with. The audit's coverage ledger table (`AUDIT_REPORT.md:95`)
   labels the whole `src/youtube/` + API-test area "Covered statically/unit" without surfacing that
   this specific integration point inside `pipeline_full.py` (not `src/youtube/`) is untested.

4. **`.env` exists at the repo root but nothing loads it.** `grep -rn "load_dotenv|dotenv"
   src/` returns no results — no file in `src/` imports `python-dotenv` or otherwise reads `.env`.
   Every configuration value is read from `os.environ` directly (e.g. `api/main.py:47-50`), which
   only reflects the process's actual shell/container environment. A developer who edits `.env`
   expecting `API_KEY` or `MAX_ACTIVE_ANALYSES` to take effect on the next `python -m uvicorn ...`
   invocation will be silently ignored unless they separately `export` those variables or use a
   tool that sources the file for them. `.env.example` (`.env.example:1-27`) documents the
   intended variables but does not mention that `.env` itself is inert.

5. **API task-lifecycle observability is metrics-poor.** `process_video_analysis`
   (`api/main.py:675-717`) correctly logs the full exception server-side
   (`logger.exception(...)`, `:717`) while returning only `"Processing failed; consult server
   logs"` to callers (`:714`) — a reasonable anti-leak choice. But `src/utils/monitoring.py:54-69`
   defines only frame-pipeline counters/gauges/histograms (`FRAMES_PROCESSED`, `DETECTIONS_MADE`,
   `TRACKS_CREATED`, `ACTIVE_TRACKS`, `GPU_MEMORY_USED`, `CPU_USAGE`, `PROCESSING_TIME`,
   `DETECTION_TIME`, `TRACKING_TIME`) — there is no counter or histogram for API task
   success/failure/cancellation rate or task queue wait time, despite `GET /metrics`
   (`api/main.py:422-445`) claiming to expose "system and pipeline metrics." An operator watching
   Prometheus has no first-class signal for "the analysis API is failing tasks" short of grepping
   logs or polling `GET /tasks`.

6. **`build_track_graph` and `build_track_graph_optimized` are two parallel, undocumented
   implementations** (`graph/build_graph.py:29` and `:129`, 97 and 118 lines respectively) in the
   same file with no obvious dispatch logic checked in this pass explaining when each is used —
   worth an explicit audit for behavioral drift between the two, since only one is likely exercised
   by `pipeline_full.py`'s single `build_track_graph` import (`pipeline_full.py:35`), leaving the
   "optimized" variant's correctness unverified by the main pipeline's own test coverage (92%
   reported for the file overall, but that aggregates both functions).

---

## 7. Feature reality

Independently assessed, not copied from the audit's own table:

| Feature | Genuinely works | Evidence | Caveat |
| --- | --- | --- | --- |
| Frame detection → tracking → graph (`fifa-pipeline`) | **Yes** | Reproduced live: `pytest -q` passes `test_pipeline_processes_single_frame` and 6 related tests against real (mocked-detector) frame I/O; JSON artifacts genuinely written. | No accuracy/throughput benchmark exists on real labeled data (consistent with audit). |
| YouTube classify → download → analyze (`POST /analyze`, `fifa-pipeline --youtube-url`) | **Partially** | Classification/download/URL-validation code exists and has dedicated security tests. | The result-assembly branch that actually produces the API's headline JSON output is 0%-covered (gap 3) — "works" is unverified for this exact code path, only inferred from adjacent unit tests. |
| Live RTSP inference (`BarcaAPIServer`) | **Code exists, not deployable** | Real SSRF-checked capture + threaded inference loop, 76% covered. | No Docker/Compose/Makefile path runs it (gap 1); no soak/reconnect test under load. |
| Live CLI loop (`fifa-live`) | **Untested** | Real OpenCV capture/detect/track/write loop, straightforward code. | 0% test coverage; no automated verification it works at all beyond code review. |
| Tactical pitch-control / OBSO | **Correctly refuses instead of fabricating** | `team_classifier.py` returns `team_id=-1` when unsure; `pipeline_full.py` skips frames without both teams (spot-check #8). | Real calibration/team-classification accuracy is unverified — this only confirms it fails safe, not that it's accurate. |
| GraphSAGE position model (`fifa-train-gnn`) | **Training scaffold works; no trained model** | Seeded splits, content-hashed datasets, atomic checkpoints (`train_gcn.py`), pickle-trust gate confirmed (spot-check #6). | No labeled dataset or trained checkpoint exists in the repo; `train_gcn.py` itself is only 35% covered. |
| ONNX export | **Yes, locally verified by the audit and structurally confirmed here** | `export_onnx.py` mirrors `export_trt.py`'s atomic-publish pattern; not independently re-run in this pass (no GPU/weights needed, but re-running the real export was out of scope). | Checkpoint provenance (`yolov8n.pt`) remains unverified — it's a `.gitignore`-excluded, untracked file (`git ls-files yolov8n.pt` returns nothing). |
| TensorRT export | **Fails honestly** | Confirmed no-fake-engine behavior (spot-check #10). | Cannot run end-to-end without NVIDIA tooling in this environment (as audit states). |
| Weekly retraining orchestration | **Locking/versioning genuinely robust; not deployable as-is** | Confirmed private 0700-lock-dir flock mechanism (spot-check #7); atomic version/checkpoint writers reviewed directly. | `WeeklyRetrainer.main()` explicitly refuses to run standalone (`weekly_retrainer.py:796-801`, returns exit code 2) — by design, not a bug, but there is no scheduler anywhere in the repo (no cron workflow schedules it) that would ever invoke it. |
| Event/player-stat/possession claims | **Correctly absent, not fabricated** | Confirmed structurally (spot-check #3). | The absence is untested by the suite that credits itself with enforcing it (spot-check #3) — a real, if narrow, verification gap in the audit's own testing claim. |
| API authentication | **Works when enabled, off by default** | `secrets.compare_digest` constant-time check (`api/main.py:173`); `API_REQUIRE_KEY` env var gates whether it's mandatory. | Purely a shared-secret header check; no per-client identity, rotation, or rate-limit-per-key — fine for a single-operator deployment, not for multi-tenant production. |

DONE
