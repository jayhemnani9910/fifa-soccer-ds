# Repository audit and remediation report

Audit date: **2026-07-18** (Asia/Kolkata)
Repository: `fifa-soccer-ds`
Audited worktree: `a0e165b`, detached in the pre-existing `rebasing master` state
Dependency research companion: [DEPENDENCY_UPGRADE_REPORT.md](DEPENDENCY_UPGRADE_REPORT.md)

## Evidence vocabulary

Every conclusion in this report uses one of these labels:

- **Verified fact** — directly observed in tracked source, Git state, command output, or a cited upstream source.
- **Evidence-supported inference** — strongly supported by observed structure or behavior but not exercised in the deployment environment.
- **Unverified hypothesis** — plausible, but unavailable evidence prevents confirmation.
- **Environment/external blocker** — verification requires unavailable hardware, artifacts, credentials, services, network access, or tool permission.

Confidence in findings is reported separately as High, Medium, or Low. No commit, push, deployment,
external service mutation, dataset deletion, model deletion, or training run was performed.
Inventory, architecture, feature, and validation table rows are **Verified fact** unless their cell
explicitly identifies an inference, hypothesis, or blocker.

## Executive summary

**Overall health: conditionally healthy for local CPU engineering; not yet production-ready for
trained soccer analytics.** This is a **Verified fact** based on a green local test/build/static-analysis
suite and a successful bounded real-model CPU smoke run, combined with the absence of tracked model,
dataset, calibration, and trained GNN artifacts.

The original repository mixed substantial implementations with dangerous false-success paths. It could
report events, player/team statistics, possession, tactical percentages, exports, retraining versions,
and API success when the corresponding computation or artifact did not exist. It also had path-boundary,
SSRF, download-bounding, unsafe-deserialization, race, task-retention, lifecycle, metric, packaging, and
dependency-coordination defects. Those Critical/High-class candidates were repaired and covered with
focused regression tests. No Critical finding remains open.

The final clean-environment quality result is **251 passed**, two instances of one third-party
`torch.jit.script` deprecation warning emitted inside Torch/PyG, and **54% branch-aware coverage**. Ruff
lint, Ruff formatting, mypy, Bandit, compilation, standard isolated sdist/wheel build, installed-wheel
import/API startup, clean-environment `pip check`, two complementary `pip-audit` scans, aggregate
pre-commit, base/GPU Compose rendering, Dockerfile BuildKit checks, a real ONNX export/CPU parity run,
and a one-frame detection/tracking/graph pipeline smoke all completed successfully. The most important
remaining risks are:

- **Environment/external blocker:** no NVIDIA GPU/driver, deployable model inventory, labeled evaluation
  set, calibration, real RTSP source, YouTube network run, or trained GNN checkpoint was available.
- **Verified fact:** coverage is uneven; live capture, TensorRT, health checks, YouTube media helpers, and
  several classifiers remain below 40% coverage, with `src/live/run_live.py` unexecuted by coverage.
- **Verified fact:** several large functions remain maintainability and performance-risk hotspots.
- **Verified fact:** exact direct dependencies are pinned and a fresh Linux/Python 3.12 all-extras
  resolution was tested, but no reviewed cross-platform transitive lock is committed; system `apt`
  packages in the Dockerfile remain floating.
- **Verified fact:** the pre-existing 8 GB `.venv` is not reusable as-is: orphaned PyDrive2 1.21.3 and
  pyOpenSSL 24.2.1 require Cryptography `<44`, while the secure project target is 48.0.1. The fresh
  disposable environment is authoritative and passes `pip check`; recreate the ignored `.venv` before use.
- **Unverified hypothesis:** licenses and provenance for the local ignored `yolov8n.pt`, 3.7 GB data,
  16 GB outputs, and 20 MB MLflow state cannot be established from Git metadata.

## Scope, exclusions, and repository state

### Git and ownership baseline

- **Verified fact:** `git branch -a` reported `* (no branch, rebasing master)`; the audit did not alter,
  abort, continue, or otherwise interfere with that rebase.
- **Verified fact:** the worktree was already broadly modified. The user explicitly authorized correction
  or replacement of repository files, but unrelated ignored data/model/output artifacts were preserved.
- **Verified fact:** 138 files are tracked: 87 Python, 11 YAML, 10 Markdown, 7 YML, 1 TOML, 1 HTML,
  9 tracked images, and repository metadata/configuration files.

### Exclusions

- **Verified fact:** expensive training was not started.
- **Verified fact:** generated, ignored, binary, model, data, cache, and experiment directories were
  inventoried but not line-reviewed.
- **Environment/external blocker:** no live external attack, paid service, Kaggle/YouTube download,
  remote MLflow/DVC operation, GPU inference, TensorRT build, or full CUDA container build was attempted.
  The Dockerfile itself was checked by BuildKit against the pinned remote base without warnings.
- **Verified fact:** the final code knowledge graph refresh completed with 1,642 nodes and 4,497 edges.

## Coverage ledger

| Tracked area | Files / evidence | Coverage decision | Result |
| --- | --- | --- | --- |
| Root metadata and policy | 18 root files including `README.md`, `CONTRIBUTING.md`, `SECURITY.md`, `LICENSE`, `pyproject.toml`, `requirements.txt`, `Makefile`, Docker/Compose, DVC, params, ignore files | Read and cross-checked against code/CI | Covered |
| Repository instructions | User-supplied `AGENTS.md` instructions plus tracked contributor/copilot/manual docs | Read before implementation; stale claims treated as evidence to verify, not authority on current behavior | Covered |
| GitHub automation | 13 files under `.github/`, including CI, CodeQL, Pages, release, stale, labeler, Dependabot | Reviewed permissions, immutable action pins, Python versions, commands, and release branches | Covered |
| Runtime configuration | 2 supported files under `configs/`, `.env.example`, `params.yaml` | Cross-checked defaults, env precedence, paths, limits, and removed unused Hydra claims | Covered |
| DVC | `.dvc/config`, `.dvc/.gitignore`, `dvc.yaml`, `params.yaml` | Stage/dependencies/outputs inspected and historical local `dvc status` evidence retained; automatic DVC installation removed because its DiskCache dependency has no fixed release | Covered; stage/artifacts and security-reviewed operator CLI remain external |
| API | `src/api/`, `src/main.py`, schemas, API tests | Routes, task lifecycle, auth, CORS, errors, metrics, shutdown, and ASGI tests reviewed | Covered |
| Detection/export/training | `src/detect/` | Inference, input checks, fine-tuning, safe loading, ONNX/TensorRT paths, CLI and tests reviewed | Covered; GPU/TensorRT externally blocked |
| Tracking | `src/track/` | Lifecycle, metric semantics, empty input, graph hand-off, tests reviewed | Covered |
| Graph/GNN | `src/graph/`, `src/models/` | Construction, truncation, memory estimate, deterministic training, trusted dataset loading, checkpoint behavior reviewed | Covered; trained data/checkpoint absent |
| Calibration/analytics | `src/calib/`, `src/analytics/` | Coordinate validation, homography singularities, team unknown-state, tactical failure semantics reviewed | Covered; real calibration absent |
| Batch orchestration | `src/pipeline_full.py`, `src/pipeline_orchestrator.py` | Data flow, partial failure accounting, output integrity, MLflow lifecycle, CLI and smoke reviewed | Covered |
| Live streaming | `src/live/`, `src/detect/video_streaming.py` | URL validation, capture cleanup, error state, detector initialization, routes reviewed | Covered statically/unit; real RTSP/GPU blocked |
| YouTube/audio/media | `src/youtube/`, API tests | URL allowlist, redirect/size/duration controls, temp cleanup, metadata integrity, optional audio behavior reviewed | Covered statically/unit; network run blocked |
| Data/MLOps | `src/data/`, `src/train/`, `src/utils/mlflow_helper.py` | Extraction safety, DVC fail-closed behavior, versioning, locking, atomic publication, MLflow run ownership reviewed | Covered; remote services/data absent |
| Operational utilities | remaining `src/utils/` | Health, monitoring, path containment, network validation, overlays reviewed | Covered; several runtime probes low coverage |
| Scripts | 2 supported worktree files under `scripts/` | Misleading unsafe one-off scripts removed; the experimental generator and data preprocessing CLI retained and reviewed | Covered |
| Tests | 36 worktree files, 251 collected tests after remediation | Assertions, mocks, negative paths, false-positive end-to-end tests, security and performance intent reviewed | Covered |
| Static docs/site | 10 files under `docs/` plus root docs | Claims reconciled with implementation; binary images inventoried only | Covered |

### Preserved untracked/ignored artifacts

| Area | Observed size | Git status | Audit treatment |
| --- | ---: | --- | --- |
| `data/` | 3.7 GB | Ignored after hygiene repair | Preserved; sample frame directory used for one bounded smoke |
| `outputs/` | 16 GB | Ignored | Preserved; not interpreted as validated benchmark evidence |
| `.venv/` | 8.0 GB | Ignored | Preserved; used for early baselines, but historical orphan packages make its final `pip check` fail; not authoritative |
| `mlruns/` | 20 MB | Ignored | Preserved; local smoke wrote a normal local run |
| `yolov8n.pt` | 6.3 MB | Ignored | Used only for the bounded smoke; provenance/license unverified |

## Technology and repository inventory

### Languages, runtimes, and packaging

- **Verified fact:** first-party application code is Python; static documentation includes HTML, YAML,
  Markdown, shell, Dockerfile, and Compose configuration.
- **Verified fact:** project metadata now supports CPython `>=3.12,<3.13`; `.python-version` is `3.12.13`.
  The available environment is CPython 3.12.3.
- **Verified fact:** `pyproject.toml` is the dependency and packaging source of truth. `requirements.txt`
  intentionally contains only `-e ".[dev]"`.
- **Verified fact:** setuptools builds the generic top-level package `src`; four console scripts are
  published: `fifa-detect`, `fifa-live`, `fifa-pipeline`, and `fifa-train-gnn`.

### Major frameworks and services

- CV/ML: Ultralytics YOLO, PyTorch, TorchVision, PyTorch Geometric, OpenCV, NumPy, SciPy,
  scikit-learn, Pillow.
- API/operations: FastAPI, Pydantic, Uvicorn, SlowAPI, Prometheus client, aiohttp, psutil.
- Media: yt-dlp, FFmpeg system CLI; optional PyAV, librosa, Whisper, and ffmpeg-python.
- Export: optional ONNX and ONNX Runtime GPU; TensorRT is imported only when that path is invoked.
- MLOps/data: optional MLflow and Kaggle. DVC configuration is retained, but DVC is intentionally not an
  install extra until its unfixed DiskCache deserialization advisory is resolved or operationally accepted;
  local filesystem is the default artifact/state backend.
- Quality: pytest, pytest-cov/coverage, Ruff, mypy, Bandit, pre-commit, `build`.
- Deployment: a digest-pinned NVIDIA CUDA Ubuntu image, Docker Compose, GitHub Actions, CodeQL,
  Dependabot, Pages, and release-please.

### Storage, queues, caches, and external systems

- **Verified fact:** first-party analysis state is filesystem/process based. Outputs, task artifacts,
  model/data paths, and DVC cache/state are not backed by an application database or object store. The
  optional Compose MLflow profile deliberately uses a SQLite file under its writable state volume.
- **Verified fact:** API background tasks are in-process `asyncio` tasks with a bounded process-local
  result store. There is no external queue and no durable multi-worker coordination.
- **Verified fact:** repository automation schedules only CodeQL (Mondays at 06:30 UTC) and stale-issue
  maintenance (daily at 00:00 UTC). Pages is push/manual, release-please is push-driven, and no model
  training/retraining job is scheduled by GitHub Actions.
- **Verified fact:** apart from that local MLflow SQLite backend, no domain database, Redis, message broker,
  cloud object store, or configured DVC remote is present.
- **Evidence-supported inference:** YouTube, Kaggle, RTSP sources, remote MLflow, and NVIDIA registries are
  deployment integrations rather than repository-owned services.

## Architecture and data flow

```text
frame directory ─┐
local video ─────┼─> input validation ─> Ultralytics detector ─> tracker ─> track history
approved YouTube ┘          │                     │                  │
                            │                     └─> JSON/overlays   ├─> temporal-spatial graph
                            │                                        ├─> optional trusted GNN
                            └─> bounded download/audio metadata       └─> calibrated tactical analytics

batch FastAPI -> bounded background task -> orchestrator -> isolated output root -> status/metrics
live FastAPI  -> validated RTSP capture -> detector/tracker thread -> latest detections/health
training      -> versioned trusted data -> validation-selected checkpoint -> optional MLflow artifacts
```

Major boundaries are meaningful but imperfect: API orchestration delegates to the pipeline, detection
and tracking are separable, graph/model layers are separated, and MLOps helpers are optional. The largest
remaining boundary problem is the roughly 1,400-line `pipeline_full.py`, whose directory orchestration,
artifact writing, graph construction, analytics, CLI, and reporting responsibilities remain coupled.

### Entry points and routes

| Surface | Entry point | Observed interface |
| --- | --- | --- |
| Batch pipeline CLI | `src.pipeline_full:main` / `fifa-pipeline` | frame directory or approved YouTube URL; bounded frames/duration; optional GNN/tactical paths |
| Detection CLI | `src.detect.infer:main` / `fifa-detect` | image/video inference with validated frames and checkpoint path |
| Live CLI | `src.live.run_live:main` / `fifa-live` | camera/file/validated RTSP; bounded queue and cleanup |
| GNN training CLI | `src.models.train_gcn:main` / `fifa-train-gnn` | trusted pickled PyG dataset opt-in, seeded split/training/checkpoints |
| Batch API | `src.api.main:app` | `GET /`, `/health`, `/tasks`, `/tasks/{id}`, `/metrics`, `/pipeline/info`, `/tactical/info`, `/tactical/results/{id}`; `POST /analyze`, `/tactical/compute`; `DELETE /tasks/{id}` |
| Live API | `src.main:app` | `POST /stream/start`, `/stream/stop`; `GET /detections`, `/checkpoint`, `/health` |
| DVC stage | `analyze_frames` | invokes the frame pipeline using `params.yaml`; requires external frames and weights |
| Weekly retraining | `src.train.weekly_retrainer` library flow | explicit loader/trainer/evaluator callbacks; no scheduler or deployment launcher is claimed |

### Configuration precedence

- CLI arguments are authoritative for CLI runs.
- API/runtime security and capacity settings are read from environment variables documented in
  `.env.example`; no `.env` file is loaded or tracked automatically.
- `configs/youtube_pipeline.yaml` supplies health/pipeline defaults; `params.yaml` supplies DVC values.
- Model, media, output, MLflow, API key, CORS, RTSP, retention, and download-limit configuration remains
  deployment-owned. Removed Hydra files were unused and contradicted the real configuration flow.

## Feature reality matrix

| Feature | Where claimed | Entry point/API | Implementation evidence | Tests | Runtime status | Status | Missing work |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Frame detection/tracking/graph | README capability table | `fifa-pipeline` | Real Ultralytics load, tracker update, graph builder, JSON artifacts | pipeline, tracker, graph, smoke tests | One local CPU frame completed; zero detections | Working and verified | Accuracy/throughput evaluation on representative labeled media |
| Image/video detection | README | `fifa-detect` | Input validation, prediction extraction, video cleanup | detection smoke/error tests | Import and synthetic paths pass | Implemented but unverified | Real video benchmark and corrupted-codec matrix |
| YouTube analysis | README/API | `POST /analyze` | Strict host/ID validation, bounded downloader, metadata, orchestration | API, URL, download-security tests | Network run unavailable | Blocked by unavailable service/model | Authorized network/media E2E run |
| Live RTSP inference | README | live API and `fifa-live` | RTSP validation, capture lifecycle, real detector factory, error/health state | live API unit tests | No RTSP source/GPU | Implemented but unverified | Soak test, reconnect test, real source/GPU |
| Tactical pitch control | README | CLI flag and tactical API | Requires calibration and two classified teams; fails instead of inventing shares | calibration/tactical tests | No calibration artifact | Blocked by unavailable data/model | Validated pitch calibration and team-classification accuracy |
| GraphSAGE position model | README | `fifa-train-gnn`, optional inference | Graph model, seeded training, content hash, atomic checkpoints, validation selection | shape, training-integrity, inference tests | No labeled graph/checkpoint | Blocked by unavailable data/model | Versioned dataset, real evaluation, provenance |
| ONNX export | README | `src.detect.export_onnx` | Current Ultralytics export, overwrite refusal, ONNX checker, ORT session validation, atomic cross-filesystem publication | 8 export tests plus real parity run | Real local checkpoint exported; ORT CPU output shape matched and values were close | Working and verified locally | GPU-provider parity, accuracy validation, and checkpoint provenance |
| TensorRT export | README limitation | `src.detect.export_trt` | Real TensorRT path and explicit failure; no fake engine | limited static/unit coverage | NVIDIA tooling absent | Blocked by unavailable hardware/tooling | Build on compatible GPU image and numerical/performance parity |
| MLflow integration | README | helpers in training/pipeline/data | Owns only runs it starts; logs real parameters/metrics/artifacts | MLflow integration tests | Local MLflow run created by smoke | Working and verified locally | Remote auth/outage/retention validation |
| DVC stage | README | `dvc repro analyze_frames` | One real stage with explicit deps/outs and fail-closed missing artifacts | loader/DVC failure tests; historical `dvc status` | Stage reports changed/deleted inputs/outs; CLI excluded from normal extras for security | Blocked by unavailable data/model/tool | Restore/version legitimate artifacts; run DVC only in a security-reviewed isolated operator environment |
| Weekly retraining | README | library orchestrator | Process/thread lock, atomic state, real callbacks, validation gate | race/version/retraining tests | No production loader/evaluator/scheduler | Partially implemented | Deployment scheduler and real data/evaluation integration |
| Event/pass/touch/PPDA statistics | Explicitly disclaimed | none | Fabricated code removed | E2E tests assert absence | Not emitted | Documentation-only claim removed | Implement event model, labels, metrics before re-advertising |
| Player identity | Explicitly disclaimed | none | Tracker IDs remain anonymous identities | tracker/E2E tests | Not emitted as player names | Scaffold or placeholder removed | Roster/re-identification system and consent/provenance |
| WebRTC/HLS restreaming | Explicitly disclaimed | none | No implementation | none | unavailable | Documentation-only claim removed | Separate streaming architecture if required |
| API authentication | README deployment guidance | middleware | Optional constant-time API key check; trusted-host/CORS controls | API security tests | Verified by ASGI tests | Working and verified when enabled | External secret manager and identity-aware authorization for production |

## Baseline and engineering-loop record

### Baseline evidence

The original state was preserved through Git and compared during remediation. Because the repository was
already in an active rebase with user-owned modifications, no destructive checkout was used to rerun the
entire historical tree. The baseline below is therefore deliberately separated by evidence type:

| Baseline observation | Classification | Evidence/result |
| --- | --- | --- |
| Python claim drift | Verified fact | Historical README advertised 3.11 while project metadata required 3.12+ |
| Duplicate manifest drift | Verified fact | Historical `pyproject.toml` and `requirements.txt` declared overlapping but different packages/versions |
| CUDA ecosystem drift | Verified fact | Historical Torch 2.4.1/TorchVision 0.19.1 and separate ORT/CUDA claims were not coordinated across CI/container/docs |
| False-success paths | Verified fact | Historical code emitted fabricated analytical results and placeholder exports/retraining artifacts |
| Unsafe boundaries | Verified fact | Historical output cleanup/path handling, URL acceptance, model loading, and unbounded task/download paths lacked required controls |
| First retained broad suite after early repairs | Verified fact | 226 tests passed with one PyG deprecation warning |
| Final broad suite | Verified fact | 251 tests passed with two emissions of one third-party Torch/PyG deprecation warning; branch-aware coverage 54% |

### Closed-loop ledger

| Iteration | Findings | Hypothesis and coherent change | Principal files | Focused verification | Remaining risk / next action |
| ---: | --- | --- | --- | --- | --- |
| 1 | AUD-001, 023, 026 | Manifest/Python drift caused non-reproducible installs | `pyproject.toml`, `requirements.txt`, `.python-version`, README, Makefile | resolver metadata inspection, imports, `pip check` | Research coordinated ecosystems |
| 2 | AUD-002, 008, 009, 021 | Fabricated outputs came from permissive fallback semantics | pipeline, schemas, evaluation, tactical/team modules | focused pipeline/evaluation/tactical tests | Require real artifacts for real evaluation |
| 3 | AUD-003, 004 | User-controlled paths/URLs crossed trust boundaries | output/network helpers, API, downloader, live RTSP | path/SSRF/download tests | Redirect/network E2E remains blocked |
| 4 | AUD-005, 006, 024 | Pickle use, process races, and assertions could violate fail-closed behavior | GNN/retrainer/training/live/data/health modules | safe-load, race, retraining, Bandit tests | Trusted legacy pickle remains explicit accepted risk |
| 5 | AUD-007, 010, 012 | API/live state had unbounded retention and incomplete lifecycle/error controls | batch/live API and capture | ASGI/lifecycle/security tests | Multi-worker durability requires external queue/store |
| 6 | AUD-011, 013, 014 | Export/MLOps/calibration paths could claim success without valid artifacts | export, calibration, MLflow, DVC/data | export, calibration, MLflow, loader tests | Real GPU/remote runs blocked |
| 7 | AUD-017, 022, 025 | Tests/docs/scripts encoded imagined behavior and masked gaps | tests, README/docs, stale scripts | targeted E2E/import/security suite | Low coverage areas remain |
| 8 | AUD-001, 019 | Infrastructure and ecosystem versions needed coordinated targets | Docker/Compose, CI/actions, Dependabot, manifests | Compose render, workflow inspection, local build | Docker daemon and transitive lock remain blocked/deferred |
| 9 | AUD-015, 018, 020 | Architecture/performance/provenance claims required explicit limits | docs, monitoring, graph memory, reports | graph complexity query, memory tests, artifact inventory | Refactor hotspots and establish provenance |
| 10 | all repaired IDs | Broad regression could expose migration defects | repository-wide | 246-test intermediate suite, lint, format, mypy, Bandit, build/install, smoke | Live dependency/security verification remained |
| 11 | AUD-001, 019, 028 | Reachable vulnerable packages required a coordinated ecosystem change, not isolated patch pins | manifests, Docker, CI, data loader, docs | fresh all-extras resolver, clean install, `pip check`, two `pip-audit` modes, focused/full tests | GPU execution remains external; DVC deferred because no fixed DiskCache exists |
| 12 | AUD-013, 029 | Export success needed structural/runtime proof and atomic cross-filesystem publication | ONNX exporter and tests | 8 unit tests; real YOLO export; ONNX checker; ORT CPU session; raw-output parity | GPU provider and model-provenance validation remain external |
| 13 | AUD-030, 031 | Compose and CI did not realize the documented model/MLflow/GPU/optional-extra contracts | Compose, env example, Docker/CI, README, pre-commit | base/GPU Compose render, BuildKit check, aggregate pre-commit, all-extras clean suite | Full CUDA image build/health remains intentionally unrun |
| 14 | AUD-032 and all IDs | Final packaging/graph/regression checks could reveal stale evidence or metadata | `pyproject.toml`, reports, repository-wide | standard isolated build, installed-wheel API import, 251 tests, graph refresh, diff checks | Genuine hardware/data/provenance blockers documented |

## Findings

No Critical finding was confirmed.

### High findings

| ID | Severity | Confidence | Category | Evidence and classification | Reproduction | Impact | Root cause | Recommended fix | Verification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AUD-001 | High | High | Dependencies/reproducibility | **Verified fact:** duplicate manifests and uncoordinated Torch/CUDA/ORT versions existed | Compare historical `pyproject.toml`, `requirements.txt`, Docker/CI/docs | Resolver drift, binary incompatibility, unreproducible GPU installs | Two sources of truth and independent ecosystem upgrades | Make `pyproject.toml` authoritative; coordinate exact tested targets | Standard build; clean all-extras install; `pip check`; 251 tests; compatibility sources in dependency report |
| AUD-002 | High | High | Functional/ML truthfulness | **Verified fact:** pipeline paths fabricated events, player/team totals, and success metadata | Historical E2E/pipeline code and diff | Consumers could make decisions from invented analytics | Scaffold values were exposed as computed results | Remove claims/results until real computation exists; fail explicitly | Pipeline/E2E tests assert fabricated fields are absent; README disclaims them |
| AUD-003 | High | High | Path security/data integrity | **Verified fact:** task/output cleanup could operate on inadequately constrained paths | Output-path security regression cases | Cross-task overwrite or deletion outside intended root | Caller-controlled names and weak containment | UUID task directories, resolved containment, constrained cleanup | 16 output-path security tests pass |
| AUD-004 | High | High | SSRF/resource exhaustion | **Verified fact:** remote media/RTSP acceptance lacked complete host/IP/size/redirect bounds | URL/private-address/oversize/thumbnail tests | Internal-network access, disk/memory exhaustion, credential leakage | URL syntax validation was treated as network trust validation | Strict YouTube hosts/IDs, public RTSP policy, redirect revalidation, byte/duration/time bounds, redaction | Network/download/API security tests pass; real network E2E blocked |
| AUD-005 | High | High | Unsafe deserialization/supply chain | **Verified fact:** PyTorch/PyG artifact loading could invoke pickle without an explicit trust gate | `load_graph_dataset` negative test and checkpoint loader tests | Arbitrary code execution from malicious model/data artifacts | Legacy serialization APIs were treated as ordinary data reads | Restricted Ultralytics loading; `weights_only` where possible; explicit trusted-PyG opt-in | Safe-model-loading tests pass; Bandit accepted risk is line-scoped |
| AUD-006 | High | High | Reliability/MLOps | **Verified fact:** retraining/version publication had race and placeholder-success behavior | race/version/retraining tests | Concurrent jobs could corrupt state or publish an unevaluated model/version | In-process assumptions and non-atomic state updates | `flock`, private runtime lock path, atomic writes, real evaluator gate, fail closed | 15 race tests plus retraining/version suites pass |
| AUD-007 | High | High | API correctness/reliability | **Verified fact:** analysis tasks could report fake success and grow without bound | API task/security tests | Memory growth, false completion, orphaned work on shutdown | Unbounded global task/result state and scaffold execution | Real orchestrator, capacity/TTL bounds, cancellation, lifespan cleanup, failure propagation | ASGI API/security/YouTube tests pass |
| AUD-008 | High | High | Metric/tracking correctness | **Verified fact:** evaluation aggregation and tracker lifecycle semantics produced incorrect or misleading values on edge cases | evaluation/tracker regression tests | Invalid quality conclusions and identity counts | Denominator/empty-state/lifecycle assumptions | Correct metric aggregation, empty handling, track lifecycle/state validation | Evaluation, tracking, E2E suites pass |
| AUD-009 | High | High | Tactical correctness | **Verified fact:** tactical code could emit 50/50 or 100/0 shares when calibration/team computation failed | tactical/team/calibration regression tests | Fabricated possession/pitch-control claims | “Friendly” fallback substituted invented domain results | Require valid homography and both teams; return explicit unavailable/failure | Tactical/calibration/team tests pass |
| AUD-028 | High | High | Dependency security | **Verified fact:** the previously recommended closure contained known advisories in Torch 2.8, ONNX 1.21, Cryptography 46, and unfixed DiskCache 5.6.3 | Fresh all-extras resolution plus `pip-audit`; OSV records linked in dependency report | DoS/memory-corruption or unsafe-deserialization exposure in reachable supported installs | Compatibility caution had preserved vulnerable versions and DVC pulled an unfixed cache dependency | Upgrade Torch/Vision/PyG, ONNX/ORT, MLflow/Cryptography as tested groups; remove DVC from normal extras until fixed/accepted | Clean all-extras environment and public-version closure both report no known vulnerabilities; 251 tests pass |

### Medium findings

| ID | Severity | Confidence | Category | Evidence and classification | Reproduction | Impact | Root cause | Recommended fix | Verification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AUD-010 | Medium | High | API security | **Verified fact:** auth/CORS/error disclosure defaults and health metadata were inconsistent | API security tests/config review | Accidental public exposure or internal detail leakage | Development defaults leaked into deployable paths | Optional constant-time API key, explicit CORS, sanitized errors, loopback defaults, UTC/version metadata | API security suite passes; operators must enable auth externally |
| AUD-011 | Medium | High | MLflow/DVC lifecycle | **Verified fact:** run ownership and DVC failure reporting were ambiguous | MLflow/loader tests and `dvc status` | Nested/leaked runs or data falsely described as versioned | Helpers did not distinguish caller-owned runs or CLI failure | Own only started runs, log real artifacts, make DVC registration fail closed | MLflow 3.14/loader tests pass; local smoke logged normally; DVC now requires an isolated reviewed CLI |
| AUD-012 | Medium | High | Resource lifecycle | **Verified fact:** media/live/background resources had incomplete cleanup/error handling | lifecycle tests and shutdown inspection | File/thread/capture/task leaks and degraded service state | Success-path-oriented cleanup | Context/finally cleanup, bounded queues/tasks, stop/join/cancel behavior, degraded health | API/live tests and bounded smoke pass |
| AUD-013 | Medium | High | Export correctness | **Verified fact:** export paths could create or accept placeholder artifacts | export tests/diff | Downstream deployment of invalid ONNX/TensorRT files | Scaffold file creation stood in for exporter success | Use current Ultralytics exporter, verify real output, fail without TensorRT/tooling | 8 ONNX tests and a real checker/ORT/parity export pass; TensorRT remains hardware-blocked |
| AUD-014 | Medium | High | Calibration/numerics | **Verified fact:** malformed/singular/non-finite homographies and points were insufficiently rejected | calibration tests | Invalid coordinates and tactical aggregates | Missing finite/shape/invertibility checks | Validate points/matrices/bounds and infinity-safe transforms | Seven calibration tests pass |
| AUD-015 | Medium | High | Maintainability/performance | **Verified fact:** final graph analysis found `process_frames_directory` cyclomatic 74/cognitive 200 (603 lines), fine-tuning 32/81, graph building 18/47, TensorRT export 16/45, and thumbnail download 14/52 | Final 1,642-node graph search/complexity evidence | Change risk, hard-to-test branches, hidden allocation/complexity costs | Multiple orchestration responsibilities in large functions | Incrementally extract artifact writer, frame processor, analytics stage, and download policy objects | Open; tests reduce regression risk but no aesthetic refactor is claimed |
| AUD-016 | Medium | High | Test coverage | **Verified fact:** total branch coverage is 54%; live runner 0%, RTSP 16%, TensorRT 19%, health 17%, YouTube helpers 15–24% | `make test-cov` | Production-only failure paths may escape CI | Heavy external/hardware paths and historically mock-heavy tests | Add hermetic codec/redirect/RTSP fixtures and hardware-profile CI | Open and prioritized |
| AUD-017 | Medium | High | Test validity | **Verified fact:** historical “E2E” assertions accepted imagined artifacts/results | Test/diff review | Green tests could certify nonexistent behavior | Tests mirrored documentation scaffolds instead of contracts | Replace with negative truthfulness and real artifact-integrity assertions | Reworked E2E/pipeline suites pass |
| AUD-018 | Medium | High | GPU/platform verification | **Environment/external blocker:** no NVIDIA GPU, driver, TensorRT, real RTSP, or CUDA execution | `torch.cuda.is_available()`/environment inventory; CPU-only clean environment | GPU compatibility/performance remains unproven locally | Required hardware/runtime unavailable | Run documented GPU matrix in a CUDA 12.6/driver >=560.35.05 environment | Open blocker; BuildKit Dockerfile check and GPU Compose rendering pass, but no GPU runtime is claimed |
| AUD-019 | Medium | High | Build reproducibility | **Verified fact:** direct Python requirements and CUDA image digest are pinned and one fresh Linux closure is tested, but transitive packages and apt packages are not committed as reviewed locks | Manifest/Docker inspection and fresh resolver | Future platform resolutions may change without a first-party diff | No platform constraints/lock; apt repository is time-varying | Generate reviewed per-platform constraints in CI; snapshot/pin OS packages or rebuild/scan regularly | Partially fixed; standard isolated build, wheel install, clean `pip check`, and audit pass |
| AUD-020 | Medium | Medium | Provenance/licensing/privacy | **Unverified hypothesis:** ignored local weights/data/media/outputs may have unknown rights, consent, retention, or model provenance | Git inventory shows none are tracked; local artifacts exist | Legal/privacy/supply-chain risk in deployment | Artifact metadata is outside Git and no model/data cards are supplied | Require checksums, source/license, consent/retention policy, model/data cards | Open; cannot infer provenance from filenames |
| AUD-021 | Medium | High | Partial failure semantics | **Verified fact:** per-frame failures could be hidden behind overall success | pipeline partial-failure tests | Silent data gaps and misleading success rates | Summary tracked totals but not failed-stage accounting | Record failures/warnings/success rate; fail when detection/tracking wholly fails | Pipeline tests pass |
| AUD-022 | Medium | High | Training integrity | **Verified fact:** fine-tuning could proceed without meaningful validation and had accumulation/checkpoint edge defects | `test_train_yolo.py`, GNN training-integrity tests | Overfit/invalid checkpoints could be selected/published | Training loss or placeholder values substituted for validation | Require real validation, finite metrics, corrected accumulation, best-by-validation atomic checkpoint | Training integrity suites pass; real labeled evaluation absent |
| AUD-023 | Medium | High | Python support | **Verified fact:** metadata/docs/local version target disagreed and the local pin was an old security-patch release | `.python-version`, README, metadata comparison and Python upstream status | Onboarding drift and missed interpreter security fixes | Independent version declarations | Align on `>=3.12,<3.13`, pin developer target 3.12.13, document local 3.12.3 mismatch | Metadata/build checks pass; host upgrade remains external |
| AUD-024 | Medium | High | Subprocess/temp/runtime robustness | **Verified fact:** production asserts, partial executable paths, and a shared literal `/tmp` lock path caused Bandit failures and fail-open thread behavior | Initial Bandit: 4 Medium/15 Low; missing-capture and lock tests | Optimized Python removes asserts; path hijack/confusion; silent thread death | Assertions used as runtime validation; executables/lock defaults not resolved securely | Explicit failures, resolved executables/fixed argv, private user-owned 0700 runtime lock directory | Bandit now exits 0; 24 affected tests pass |
| AUD-029 | Medium | High | ONNX integrity/atomicity | **Verified fact:** successful export was not structurally/runtime validated and cross-filesystem publication initially failed with `EXDEV` | Real export to `/tmp` reproduced `EXDEV`; malformed/no-output unit fixtures | Invalid or partially published deployment artifacts | Path existence was treated as validity and atomic rename was assumed to cross filesystems | ONNX checker + ORT session validation, overwrite refusal, fsynced destination-local temp copy, atomic replace | 8 tests pass; real 12.1 MB export validates; PyTorch/ORT raw outputs match shape with max absolute delta 0.000610352 |
| AUD-030 | Medium | High | Deployment integration | **Verified fact:** Compose did not mount the configured model path, app/MLflow tracking URIs conflicted, and GPU intent was implicit | Render historical Compose and inspect container paths | Default deployment could start without usable weights or write MLflow state to the wrong service | Application and service profiles shared undeclared assumptions | Read-only model mount, separate app tracking URI, writable state, selectable build target, explicit GPU override | Base and GPU-overlay Compose render; BuildKit production check reports no warnings |
| AUD-031 | Medium | High | Quality-gate coverage | **Verified fact:** CI omitted supported extras and pre-commit's isolated mypy hook lacked all ML/CV dependencies | Fresh all-extras install and aggregate pre-commit reproduction | Optional code could rot; aggregate gate produced false import failures | CI installed only a subset and hook isolation did not match project semantics | Install all supported extras in CI; run mypy against the installed project environment; run aggregate pre-commit | Clean all-extras tests/pip check pass; aggregate pre-commit passes; CI encodes both gates |

### Low and informational findings

| ID | Severity | Confidence | Category | Evidence and classification | Reproduction | Impact | Root cause | Recommended fix | Verification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AUD-025 | Low | High | Hygiene/docs | **Verified fact:** stale one-off scripts and duplicated security-test prose contradicted supported behavior | Tracked-file/diff review | Maintenance noise and unsafe copy/paste guidance | Experimental scaffolding was promoted into repository surface | Remove misleading scripts/docs; rewrite README/site/contributor guidance | Ruff/tests/docs links pass locally |
| AUD-026 | Low | High | Typing dependencies | **Verified fact:** `types-requests` was redundant with Requests >=2.34, whose official metadata instructs users to uninstall the stub package | Manifest/pre-commit review and official PyPI page | Duplicate/incompatible type information | Historical stub remained after Requests gained inline types | Remove stub from dev and pre-commit environments | mypy passes without it |
| AUD-027 | Informational | High | Packaging architecture | **Verified fact:** the installable top-level package is named `src` | Built wheel content and `pyproject.toml` | Generic namespace can collide and is unconventional | Historical layout packages the source-container name | Consider a future compatibility-planned rename to a project-specific package | Deferred; changing imports is a breaking migration |
| AUD-032 | Low | High | Packaging metadata | **Verified fact:** the standard build warned that the TOML license table becomes unsupported after 2027-02-18 | `python -m build` with setuptools 83 | A future backend upgrade would break otherwise valid builds | Legacy `license = {file = ...}` metadata | Use SPDX `license = "MIT"` and `license-files = ["LICENSE"]` | Repeated standard isolated sdist/wheel build passes without the deprecation; wheel reports `License-Expression: MIT` |
| AUD-033 | Low | High | Local environment hygiene | **Verified fact:** the ignored pre-existing `.venv` retains PyDrive2 1.21.3 and pyOpenSSL 24.2.1, both requiring Cryptography `<44` | `.venv/bin/python -m pip check` | Developers could run an internally inconsistent historical environment instead of the repaired project closure | In-place environments accumulated orphan packages across manifest generations | Recreate `.venv` from `pyproject.toml`; do not downgrade the security-compatible Cryptography target to accommodate orphans | Open user-environment cleanup; fresh all-extras environment passes `pip check`, audit, and 251 tests |

## Security and privacy review

- **Verified fact:** a tracked-file filename-only scan for common AWS, GitHub, private-key, and API-key
  signatures returned no matches. This is not an entropy scanner and is not proof that secrets never
  existed in history.
- **Verified fact:** `.env`, data, model, output, MLflow, build, and cache artifacts are ignored; `.env.example`
  contains placeholders, not working credentials.
- **Verified fact:** API output roots are resolved/contained, task IDs are UUIDs, and cleanup is scoped.
- **Verified fact:** YouTube and RTSP inputs have host/scheme/IP policies; URL credentials are redacted;
  downloads and media metadata are bounded and cleaned.
- **Verified fact:** checkpoint/data loading is restricted by default. The one remaining pickle path is an
  explicit `--trust-pickled-dataset` compatibility path and is documented as code-execution-capable.
- **Verified fact:** API rate limiting, task capacity/TTL, body/schema limits, and bounded output retention
  reduce DoS exposure. They do not replace an upstream reverse proxy or per-tenant quota.
- **Verified fact:** Docker/Compose use a non-root UID, capability drop, read-only runtime settings,
  loopback host publishing, `tini`, and health checks. The app can write only mounted output/state paths.
- **Verified fact:** GitHub Actions use immutable commit SHAs and job-scoped least-privilege permissions.
- **Verified fact:** `pip-audit==2.10.1` found no known vulnerabilities in both (a) the exact clean
  all-extras installed environment and (b) a fully pinned public-version resolver closure. The first scan
  necessarily skipped the editable first-party distribution and the `+cpu` Torch/Vision local labels; the
  second scan explicitly covered upstream `torch==2.13.0` and `torchvision==0.28.0`.
- **Verified fact:** DVC was removed from supported install extras because current DiskCache 5.6.3 has an
  unfixed unsafe-pickle advisory. The DVC stage remains available only to operators who explicitly accept
  and isolate that toolchain.
- **Unverified hypothesis:** retained local media could contain PII/biometrics. Deployment must define lawful
  basis, access, encryption, retention, deletion, and incident handling.

## Performance, scalability, and reliability review

- **Verified fact:** detector/model initialization occurs once per pipeline/live server instance rather than
  once per frame; frame loops release captures and bound queues/tasks/results.
- **Verified fact:** graph construction has explicit memory estimates and truncation preserves references;
  tests cover bounded-memory behavior.
- **Verified fact:** API task state is process-local and bounded. This is suitable for a single-process
  development service, not durable or horizontally scaled job execution.
- **Verified fact:** one local CPU smoke processed one 640-class sample frame through actual model loading,
  detection, tracking, graph writing, and local MLflow logging in a 5.7-second command wall time. It found
  zero detections. This is a smoke result, not an accuracy or throughput benchmark.
- **Verified fact:** a real 12.1 MB YOLO ONNX artifact exported on CPU, passed the ONNX checker and ORT CPU
  session construction, and produced the same `(1, 84, 8400)` raw output shape as PyTorch. Maximum/mean
  absolute deltas were 0.000610352/0.000001334; this is engineering parity, not accuracy certification.
- **Evidence-supported inference:** the oversized pipeline/fine-tuning/export/download functions increase
  the chance of repeated allocations and hard-to-isolate latency. No optimization claim is made without a
  representative benchmark.
- **Environment/external blocker:** no soak, concurrent load, GPU memory, RTSP reconnect, large-video,
  TensorRT, or multi-worker benchmark was possible.

## ML/CV/data/MLOps correctness review

- **Verified fact:** training splits are seeded, dataset identity is content-hashed, checkpoint writes are
  atomic, and best selection requires finite validation metrics.
- **Verified fact:** inference/training input checks reject empty, malformed, or non-image frames; bounding
  boxes are normalized/clamped and oversized boxes filtered before downstream use.
- **Verified fact:** graph node references remain valid after truncation; empty detections/tracks/graphs are
  represented as real empty results, not fabricated players.
- **Verified fact:** calibration validates shapes, finiteness, invertibility, coordinate ranges, and points at
  infinity before tactical computation.
- **Verified fact:** team classification can return unknown; it no longer infers a team from field position.
- **Verified fact:** MLflow 3.14 closes only runs opened by its helper and can be configured fail-closed;
  DVC registration reports missing/failed isolated tooling rather than pretending versioning succeeded.
- **Environment/external blocker:** leakage, class balance/mapping, real precision/recall, ID switches,
  calibration error, tactical validity, cross-camera behavior, and data licenses cannot be assessed without
  the versioned labeled datasets and checkpoints.

## Testing and quality assessment

The 251-test suite contains useful negative/security/unit/API tests and now checks truthfulness of missing artifacts.
It remains uneven: high-coverage graph/tracker/API/retrainer modules coexist with largely unexecuted media,
health, live, TensorRT, and some classifier paths. The 54% total is evidence, not a target certification.
One conditional skip marker exists for environments without PyTorch; it was not activated in the clean
environment, and the final result contains no skipped or xfailed test.

No test or quality gate was disabled or weakened merely to obtain green. Mock-only wall-clock thresholds
were replaced with deterministic call-count/artifact assertions because they did not measure real pipeline
performance; real performance remains explicitly unclaimed. Ten Bandit `nosec` annotations remain
line-scoped: four reviewed subprocess imports, five fixed-argument subprocess calls with resolved/current
executables, and the explicit trusted legacy PyG pickle load. Bandit reports no findings with those reviewed
exceptions. Aggregate pre-commit now uses the installed project environment
for mypy and passes; CI repeats that aggregate gate after installing every supported extra.

## Deployment, CI, and documentation review

- CI runs Python 3.12 with all supported extras, Ruff, format, mypy, aggregate pre-commit, branch coverage,
  Bandit, pip-audit, package metadata checks, base/GPU Compose rendering, and Dockerfile checks. Actions
  are immutable-SHA pinned and current as researched on 2026-07-18.
- Dependabot covers Python, Docker, and GitHub Actions.
- The CUDA image is digest-pinned; Python packaging tools and direct Python packages are exact-pinned.
- Compose validates successfully, mounts weights read-only, separates local/remote MLflow URIs, makes GPU
  allocation explicit, and binds published services to loopback. BuildKit validated the production
  Dockerfile and pinned CUDA base with no warnings. A full image build/health run was not performed because
  it would fetch/build a large CUDA stack; no full-image success is claimed.
- README, contributor guidance, manual setup, `.env.example`, docs site, DVC/MLflow/container sections, and
  capability claims now match implemented behavior and explicitly name artifact/hardware prerequisites.
- Generated build, coverage, cache, model, data, output, and experiment paths are ignored. The build-created
  root `fifa_soccer_ds_analytics.egg-info` directory was removed after validation; ignored `dist/` artifacts
  remain reproducible build evidence and are not part of the intended diff.
- Release automation targets the actual `master` branch. No release was created.

## Final validation record

| Check | Exact command | Result | Classification |
| --- | --- | --- | --- |
| Full tests + branch coverage | `make PYTHON=/tmp/fifa-secure-venv/bin/python test-cov` | PASS — 251 passed, two third-party Torch/PyG warning emissions, 54% total | Verified fact |
| Ruff lint | `/tmp/fifa-secure-venv/bin/ruff check src tests scripts` | PASS | Verified fact |
| Ruff format | `/tmp/fifa-secure-venv/bin/ruff format --check src tests scripts` | PASS — 93 files already formatted | Verified fact |
| Type check | `/tmp/fifa-secure-venv/bin/mypy src` | PASS — 55 source files | Verified fact |
| Security static scan | `/tmp/fifa-secure-venv/bin/bandit -c pyproject.toml -r src` | PASS — no findings, ten reviewed line-scoped exclusions | Verified fact |
| Compilation | `/tmp/fifa-secure-venv/bin/python -m compileall -q src tests scripts` | PASS | Verified fact |
| Aggregate pre-commit | `PRE_COMMIT_HOME=/tmp/fifa-pre-commit-cache pre-commit run --all-files` with the project venv first on `PATH` | PASS — YAML/TOML/EOF/whitespace/Ruff/mypy/Bandit | Verified fact |
| Standard isolated package build | `PIP_CACHE_DIR=/tmp/fifa-pip-install-cache /tmp/fifa-secure-venv/bin/python -m build` | PASS — sdist and wheel using declared setuptools 83/wheel 0.47; no license deprecation | Verified fact |
| Isolated artifact install/import | `pip install --no-deps --target /tmp/fifa-wheel-import.sfsV7N dist/fifa_soccer_ds_analytics-0.1.0-py3-none-any.whl`, then import from `/tmp` | PASS — version 0.1.0, SPDX MIT, installed `src` path, 15 API routes | Verified fact |
| Clean all-extras install | official CPU Torch/Vision wheels, then `pip install -e ".[audio,data,dev,export,mlops]"` in `/tmp/fifa-secure-venv` | PASS — every declared supported direct package installed at its target | Verified fact |
| Dependency consistency | `/tmp/fifa-secure-venv/bin/python -m pip check` | PASS — no broken requirements | Verified fact |
| Pre-existing `.venv` consistency | `.venv/bin/python -m pip check` | FAIL — orphaned PyDrive2 1.21.3 and pyOpenSSL 24.2.1 require Cryptography `<44`, conflicting with secure target 48.0.1 | Verified fact; ignored user environment, not repository closure |
| Installed dependency scan | `/tmp/fifa-secure-venv/bin/pip-audit --local --cache-dir /tmp/fifa-pip-audit-cache` | PASS — no known vulnerabilities; editable project and custom `+cpu` labels skipped as reported by tool | Verified fact with stated limitation |
| Public-version closure scan | `pip-audit -r /tmp/fifa-secure-resolution.txt --no-deps --disable-pip` | PASS — no known vulnerabilities, including Torch 2.13.0 and Vision 0.28.0 | Verified fact |
| Import/startup smoke | installed-wheel import of `src.api.main:app` from `/tmp`, repeated with Compose cache env | PASS — 15 routes; initial host cache warning disappeared with the configured writable `MPLCONFIGDIR`/`YOLO_CONFIG_DIR` paths | Verified fact |
| Representative pipeline smoke | `python -m src.pipeline_full ... --max-frames 1` | PASS — one frame, real local YOLO load, 0 detections, 0 tracks/nodes, valid summary | Verified fact |
| Real ONNX export/parity | `run("yolov8n.pt", "/tmp/fifa-onnx-audit.4mGotf/yolov8n.onnx")`, then PyTorch/ORT raw-output comparison | PASS — 12.1 MB checked artifact; shapes `(1,84,8400)`; `allclose(1e-4,1e-4)`; max/mean absolute delta 0.000610352/0.000001334 | Verified fact; local weight provenance unverified |
| DVC consistency | `DVC_SITE_CACHE_DIR=/tmp/... .venv/bin/dvc status` (before DVC security deferral) | Command PASS; stage reports modified/deleted dependencies and deleted output | Verified fact; artifact/tool blocker |
| Compose validation | `docker compose --profile dev --profile mlops config --quiet` and `docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile mlops config --quiet` | PASS | Verified fact |
| Dockerfile check | `docker build --check --target production .` | PASS — pinned base resolved; no warnings | Verified fact; not a full image build |
| Tracked secret signature scan | filename-only `git grep -Il` common signatures | No matching tracked files | Verified fact with stated limitation |
| Internal documentation links | local relative-link scan of README, both reports, and contributor guide | PASS — no missing local targets | Verified fact |
| GPU/TensorRT/real RTSP/YouTube | documented profile commands | NOT RUN — hardware/services/artifacts unavailable | Environment/external blocker |
| Code-graph refresh | codebase-memory fast index | PASS — 1,642 nodes, 4,497 edges | Verified fact |
| Diff validation/status | `git diff --check`; `git diff --cached --check`; `git status --short --branch` | PASS — no whitespace errors; broad audit changes, five pre-existing staged files, and the detached `rebasing master` state remain uncommitted | Verified fact |

Blocked profiles can be reproduced only after supplying approved artifacts/services. These commands are
**Environment/external blocker** instructions, not passing results:

```bash
# CUDA app and direct device proof (requires NVIDIA Container Toolkit and compatible driver)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build app
docker compose exec app python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'

# TensorRT artifact build/parity prerequisite (requires a reviewed ONNX model and TensorRT in the image)
python -m src.detect.export_trt --onnx /path/to/reviewed.onnx --output /tmp/reviewed.plan --fp16

# Live RTSP soak/reconnect profile (use an approved host and reviewed checkpoint)
RTSP_ALLOWED_HOSTS=camera.example python -m src.live.run_live --rtsp --source rtsp://camera.example/stream --weights /path/to/reviewed.pt --no-preview

# Bounded real YouTube profile (use media you are authorized to process)
python -m src.pipeline_full --youtube-url 'https://www.youtube.com/watch?v=APPROVED_ID' --weights /path/to/reviewed.pt --output-dir outputs/authorized-youtube --max-frames 30

# DVC stage (only inside a security-reviewed isolated operator environment)
dvc repro analyze_frames
```

## Fixed findings, remaining findings, and blockers

All High findings (AUD-001 through AUD-009 and AUD-028) are repaired in the worktree and have focused
regression coverage. Medium AUD-010 through AUD-014, AUD-017, AUD-021 through AUD-024, AUD-029 through
AUD-031, and Low AUD-025/AUD-026/AUD-032 are also repaired. AUD-033 is an ignored user-environment cleanup,
not a repository-code defect. No correctness/security test was weakened to
reach green; replaced mock-timing assertions are documented above.

Remaining work, in priority order:

1. **AUD-033:** recreate the ignored local `.venv` from the authoritative manifest; do not reuse its
   orphaned package set or downgrade Cryptography to make it pass.
2. **AUD-018 / AUD-016:** run GPU, TensorRT, live RTSP, YouTube, codec, load, and soak profiles with
   representative trusted artifacts; add the resulting hermetic regressions.
3. **AUD-020:** create model/data cards with hashes, licenses, provenance, evaluation set version, consent,
   retention, and approved deployment uses.
4. **AUD-019:** generate and continuously test reviewed constraints for CPU and CUDA 12.6 environments;
   address floating OS packages.
5. **AUD-015:** split pipeline, download, export, and training hotspots one behavior-preserving seam at a
   time, using current tests as guardrails.
6. **AUD-027:** consider a future versioned migration from the generic `src` import namespace.
7. Run a full CUDA image build/health check and container vulnerability scan on an adequately provisioned
   builder; local BuildKit definition checks already pass.

## Final health assessment

The repository is materially safer, more truthful, more reproducible, and locally verifiable than its
historical state. Its CPU development path, packaging, API contracts, guarded artifact handling, and
synthetic/integration-style tests are healthy. Production readiness is explicitly **not established**:
model quality, artifact provenance, real-media correctness, GPU compatibility at runtime, capacity,
privacy controls, and external-service behavior remain deployment evidence obligations rather than facts
that can be inferred from a green unit suite.
