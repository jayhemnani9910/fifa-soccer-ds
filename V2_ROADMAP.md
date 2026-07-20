# fifa-soccer-ds: the road from 1.0 to 2.0

Written 2026-07-20, after a full verification pass run with three independent agent
sessions: one explored the codebase from scratch, one ran everything end to end on
this machine, and one adversarially attacked both reports plus the 2026-07-18
AUDIT_REPORT.md. Every claim below is backed by a command that was actually run or
a file:line that was actually read. The three full reports are in
`audits/2026-07-20-verification/`; key evidence is inlined here.

## 1. Where 1.0 actually stands (verified today, not assumed)

What works, reproduced live on this machine:

- Full test suite: 251 passed, 0 failed, in ~15s through `.venv/bin/python -m pytest`.
  Matches the Jul 18 audit's count exactly.
- Real pipeline on real footage: `data/processed/real_sample` (20 El Clasico frames)
  through `python -m src.pipeline_full` produced 82 detections, 34 tracks, a
  76-node/110-edge graph, 100% frame success rate, honest per-frame failure
  accounting in `pipeline_summary.json`.
- Batch API boots clean: `/health`, `/`, `/pipeline/info` all return valid 200 JSON.
- Coverage: 54% branch-aware, reproduced exactly.

Verdict on the ChatGPT audit: mostly trustworthy. 8 of 10 spot-checked claims
reproduced exactly (including the security fixes: UUID task dirs, RTSP SSRF
validation, weights_only loading, flock with 0700 lock dirs, no fabricated
tactical shares). Two claims were inflated, and the bigger one matters: the audit
credits "E2E tests assert absence" of fabricated events, but the branch that
actually assembles the YouTube analysis result (`src/pipeline_full.py:906-1072`)
is executed by zero of the 251 tests. The E2E test injects a fake processor and
asserts on values that `pipeline_orchestrator.py:262-270` hardcodes. Green suite,
untested flagship path.

The repository state itself is the most urgent problem: detached HEAD inside an
interrupted rebase (`.git/rebase-merge` exists), with the entire Jul 18
remediation sitting uncommitted in the worktree. One careless `git checkout .`
destroys a month of audit work. Nothing below matters until this is landed.

## 2. What "impressive to a senior engineer" means here

A concrete bar, not a vibe. A senior engineer who clones the repo should be able to:

1. Run one documented command and get a real analysis of real soccer footage,
   with an annotated clip and metrics they can inspect. Under 10 minutes on CPU.
2. Read the README and find that every claim in it is either demonstrated in the
   repo or explicitly marked as a deployment responsibility.
3. Spend 30 minutes reading code and find no silent-success path, no dead config,
   no documented-but-unrunnable service, no health check that lies.
4. Ask "how good is the model" and get a number: mAP on a named, versioned
   evaluation set, with a model card stating provenance and license.

1.0 fails bars 1, 3 and 4 today. The audit already fixed the worst of bar 3;
what remains is listed in Phase 1 with evidence.

## 3. Phase 0: land the foundation (do first, half a day)

- **0.1 Rescue the git state.** In order: (a) `git bundle create` a full backup of
  the repo including the dirty worktree state copied elsewhere, (b) finish or abort
  the interrupted rebase deliberately (inspect `.git/rebase-merge/done` first to
  see what it was doing), (c) commit the Jul 18 remediation as a small series of
  reviewable commits (deps, security fixes, test rework, docs), (d) push and tag.
  Nothing else in this plan starts before this is done.
- **0.2 Recreate `.venv` from `pyproject.toml`.** The current one fails `pip check`
  (orphaned pydrive2/pyopenssl pinned against cryptography 48). The audit already
  proved a fresh all-extras environment passes cleanly.
- **0.3 Fix the Makefile.** `PYTHON ?= python3` breaks `test`, `test-cov`, `smoke`,
  `run`, `run-pipeline` and `lint-all` out of the box (verified target by target).
  Default `PYTHON` to `.venv/bin/python` when it exists. Separately, `make build`
  fails because the gitignored `build/` output directory shadows PyPA `build` as a
  namespace package under system python; build into `dist-build/` or invoke the
  tool differently.
- **0.4 Decide `.env`.** Nothing loads it (zero dotenv references in `src/`,
  no `env_file:` in compose, verified). Either wire `env_file` into the compose
  dev profile and document it, or delete the file and say env vars only.

## 4. Phase 1: kill the credibility bugs (one to two weeks)

These are the things a reviewer finds fast, each verified with evidence today:

- **1.1 Make `/health` honest.** `src/api/main.py:249-272` reports healthy whenever
  the orchestrator object exists; it never checks that model weights are present or
  loadable. Meanwhile compose bind-mounts `./models` which Docker silently creates
  empty if missing, so a default `docker compose up` passes its healthcheck forever
  while every real request would fail at model load. Split liveness from readiness;
  readiness fails without a loadable checkpoint.
- **1.2 No silent weights default.** `src/pipeline_orchestrator.py:126` falls back
  to the bare relative filename `yolov8n.pt`, which resolves to the untracked,
  provenance-unknown 6.5MB file at the repo root. This contradicts the README's own
  trusted-checkpoint policy. Require an explicit path from config and fail with a
  clear message when unset.
- **1.3 Fix rate limiting behind a proxy.** `Limiter(key_func=get_remote_address)`
  with the app bound to loopback means all real traffic arrives via a reverse proxy
  and shares one rate bucket. Key by API key when auth is enabled, and add explicit
  forwarded-header trust configuration for proxy deployments.
- **1.4 Task store: durable and scoped.** `task_storage` is an in-process dict
  (the code comments admit this at `src/api/main.py:42-44`), and `GET /tasks`
  returns every caller's tasks, URLs and results to any key holder. Move to SQLite
  with per-key scoping, TTL retention as today, and survival across restarts.
- **1.5 Resolve the twin graph builders.** `build_track_graph` and
  `build_track_graph_optimized` disagree badly outside small inputs: 180 vs 150
  nodes and 240 vs 2270 edges on a 60-track dense frame, 2160 vs 7820 edges on a
  30x20 input (measured today with a comparison script). Production uses only the
  former; the "optimized" one exists only in tests, and the one test comparing them
  passes because its data shape happens to land where they roughly agree. Pick one
  implementation, delete the other, and add a regression test in the dense regime.
- **1.6 Test the flagship path.** Cover `pipeline_full.py:906-1072` (YouTube result
  assembly, the `capabilities` dict, the three no-fabrication helpers) with an
  integration test that stubs only the network download and runs the real assembly
  code. This turns the audit's strongest truthfulness claim into something the
  suite actually enforces.
- **1.7 One config surface.** Today there are four that do not reconcile:
  `configs/youtube_pipeline.yaml`, `params.yaml`, ad hoc `os.getenv` in 8+ files,
  and dataclass defaults in `pipeline_full.py`. `configs/tracking_tuned.yaml` is
  referenced by nothing while its exact values (120.0/0.35/20) are hand-copied into
  the Makefile and `params.yaml`. Introduce one pydantic-settings module as the
  single source; everything else reads from it. Delete the dead file.
- **1.8 Fix the confidence-inflating tests.** Three named examples from today's
  review: `test_pipeline_orchestrator_creation` (asserts a constructor result
  `is not None`, which cannot fail), `test_small_graph_memory` (one-sided 20%
  tolerance in a coincidentally-agreeing regime), `test_package_imports` (proves
  imports only in the one environment where they cannot fail). Replace with
  assertions that would actually catch a regression.
- **1.9 Deploy or de-scope the live API.** `src.main:app` (the RTSP/live server,
  76% covered, SSRF-hardened) is documented in the README and the audit's own
  entry-point table, but no Dockerfile CMD, compose service, or Makefile target
  runs it. Recommended: give it a compose profile, a port, a healthcheck and a
  soak test. If not, cut it from the README until it returns.

Each item lands through the same loop used to produce this report: a builder
session implements against a written acceptance command, a separate reviewer
session attacks the diff, findings go back to the builder, three rounds max.

## 5. Phase 2: the flagship result (two to three weeks)

1.0's honest position is "engineering scaffold, no trained soccer model". 2.0's
position should be "here is what it does to a real match, measured."

- **2.1 Train the model and publish the number.** The raw material is already
  local: 3.7GB of Clasico footage, extracted frame sets, a `pseudo_labels/`
  directory, and a working LoRA fine-tune path (`src/detect/yolo_lora_adapter.py`,
  `train_yolo.py` with its validation-gated checkpoint logic). Build a labeled
  eval set of a few hundred frames (player/ball/referee boxes), fine-tune, and
  report mAP against the base checkpoint. Ship a model card: dataset provenance,
  license, hashes, eval set version, known failure modes. This single artifact
  answers the question every senior engineer asks first.
- **2.2 Calibrate one match for real tactical output.** Produce a validated
  homography for the demo clip, run two-team classification, and let the existing
  tactical stack (pitch control, possession by zone) produce real numbers on real
  footage. The code already refuses to fabricate when calibration is missing;
  give it the calibration so it can succeed.
- **2.3 The demo artifact.** A 30 to 60 second annotated clip: boxes and track IDs
  on the broadcast frame, a 2D pitch minimap alongside, metrics JSON next to it.
  GIF in the README, full outputs on the Pages site. This is bar 1 from section 2.
- **2.4 Coverage where it is thinnest.** `src/youtube/` sits at 15-24% across
  1,373 lines; `run_live.py` at 0%. Hermetic fixtures (local media files, fake
  redirect servers) rather than network mocks that mirror implementation. Target:
  no source file below 40%, repo above 70% (from 54%).

## 6. Phase 3: ship like an operator (one week)

- **3.1 Committed lockfiles.** Constraints files for CPU and CUDA 12.6 closures,
  generated and verified in CI, so a resolver drift shows up as a diff instead of
  a surprise. The dependency report already did the research; freeze its result.
- **3.2 GPU evidence.** The CUDA path has never been executed (no NVIDIA runtime
  on this machine). Add a `workflow_dispatch` GPU job or a documented runbook with
  results from a rented GPU box: container build, real inference, TensorRT export
  parity. Publish the measured CPU and GPU fps in the README.
- **3.3 Observability that matches the service.** Add task success/failure/
  cancellation counters and queue-wait histograms to `/metrics` (today it exposes
  only frame-pipeline and system metrics, verified). Ship a Grafana dashboard JSON
  in `docs/`.
- **3.4 Rename the package.** The installable top-level package is literally `src`
  (audit AUD-027). A breaking 2.0 release is the one moment this rename is cheap.
  `fifa_soccer_ds` or `soccervision`, with a deprecation shim for one release.
- **3.5 Release engineering.** release-please is already wired; land 2.0.0 with a
  changelog, a GHCR-published image, and a three-command quickstart that a
  stranger can follow.

## 7. Phase 4: launch (two days)

- README rewritten result-first: demo GIF, the mAP number, the quickstart, then
  architecture. Keep the limitations section; its honesty is a differentiator.
- Pages site refreshed with the demo clip and metrics.
- A short writeup (what it does, what was hard, what it refuses to fake) for
  HN/r/computervision/LinkedIn, plus a 60-second screen recording.
- Optional: a Colab notebook reproducing the demo on CPU.

## 8. Sequencing and effort

| Phase | Contents | Effort | Depends on |
| --- | --- | --- | --- |
| 0 | git rescue, venv, Makefile, .env | 0.5 day | nothing |
| 1 | nine credibility fixes | 1-2 weeks | 0 |
| 2 | model + calibration + demo + coverage | 2-3 weeks | 0 (parallel with 1 after 1.2) |
| 3 | locks, GPU evidence, observability, rename, release | 1 week | 1, 2 |
| 4 | launch | 2 days | 3 |

Roughly five to six calendar weeks part-time. Phases 1 and 2 parallelize well
across builder sessions since they touch mostly disjoint files.

## 9. Definition of done for 2.0

- [ ] Rebase resolved, remediation committed and pushed, tags in place
- [ ] Fresh `.venv` passes `pip check`; every Makefile target works as documented
- [ ] Readiness endpoint fails without a loadable model
- [ ] No silent weights fallback; config is one module; dead config deleted
- [ ] Task store survives restart; tasks scoped per key; rate limits per caller
- [ ] One graph builder, with a dense-regime regression test
- [ ] `pipeline_full.py` YouTube branch covered by a real integration test
- [ ] Live API runnable via compose, or removed from docs
- [ ] Fine-tuned model with mAP on a versioned eval set, model card shipped
- [ ] Demo clip + minimap + metrics in README and on Pages
- [ ] Coverage at 70%+ overall, no file below 40%
- [ ] Lockfiles committed and CI-verified; GPU results documented
- [ ] Package renamed off `src`; 2.0.0 released with changelog and image
