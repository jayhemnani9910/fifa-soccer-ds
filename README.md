[![CI](https://github.com/jayhemnani9910/fifa-soccer-ds/actions/workflows/ci.yml/badge.svg)](https://github.com/jayhemnani9910/fifa-soccer-ds/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# FIFA Soccer DS analytics

An engineering and research codebase for extracting detections, tracker identities,
spatial-temporal graphs, and optional calibrated tactical aggregates from soccer footage.
It supports frame directories, local video/camera/RTSP sources, and approved HTTPS YouTube
video URLs.

This repository is **not a turnkey trained soccer model or a verified production service**.
No model weights, datasets, homography calibration, or trained GNN checkpoint are tracked in
Git. Real-media accuracy, throughput, GPU behavior, and deployment capacity must be validated
with the exact artifacts and hardware used by a deployment. See [AUDIT_REPORT.md](AUDIT_REPORT.md)
and [DEPENDENCY_UPGRADE_REPORT.md](DEPENDENCY_UPGRADE_REPORT.md) for the evidence and remaining
risks from the 2026-07-18 audit.

## Capability status

| Area | Current state | Important limitation |
| --- | --- | --- |
| Frame detection → tracking → graph | Implemented and covered with synthetic/model-double tests | A compatible, trusted YOLO checkpoint and real input frames are required |
| Image/video detection CLI | Implemented | Real inference was not benchmarked by the audit |
| Batch YouTube API | Implemented with bounded tasks, strict URL validation, download limits, and cleanup | Requires network access, yt-dlp/FFmpeg, and model artifacts |
| Live file/camera/RTSP processing | Implemented | No WebRTC or HLS re-streaming; real RTSP/GPU execution was not available during audit |
| Tactical pitch-control aggregates | Implemented behind an explicit flag | Requires a validated homography file and successful two-team classification |
| GraphSAGE/GCN training and inference | Implemented and unit-tested | No trained GNN checkpoint or labeled graph dataset is shipped |
| ONNX export | Implemented with ONNX structural checks and ONNX Runtime load validation | A real export needs a compatible checkpoint; TensorRT needs NVIDIA tooling/hardware |
| MLflow and DVC | MLflow is optional; the DVC stage/configuration is retained | DVC CLI installation is intentionally isolated because its current DiskCache dependency has an unfixed advisory; no remote data is configured here |
| Weekly retraining | Library orchestration exists and fails closed without real loaders/evaluator | No scheduler or deployment-specific data-loader launcher is shipped |
| Event/player identity statistics | Not implemented | Tracker IDs are not player identities; events, passes, touches, PPDA, and possession are not inferred |

## Architecture

```text
frame directory ─┐
local/live video ├─> YOLO/Ultralytics ─> ByteTrack-style tracker ─> track history
YouTube URL ─────┘             │                                  │
                              └─> detection JSON/overlays          ├─> spatial-temporal graph
                                                                 ├─> optional GNN inference
validated homography + classified teams ──────────────────────────└─> tactical aggregates

FastAPI batch service ─> bounded background task ─> pipeline orchestrator ─> isolated output dir
FastAPI live service  ─> RTSP capture + detector/tracker ─> current detections/health endpoints
```

Key entry points:

- `fifa-pipeline` / `python -m src.pipeline_full`: frame-directory or YouTube pipeline.
- `fifa-detect` / `python -m src.detect.infer`: image or local-video detection.
- `fifa-live` / `python -m src.live.run_live`: camera, file, or explicitly allowed RTSP input.
- `fifa-train-gnn` / `python -m src.models.train_gcn`: GraphSAGE training from a trusted PyG dataset.
- `src.api.main:app`: batch YouTube analysis API.
- `src.main:app`: live-stream control API.

## Requirements and installation

- CPython `>=3.12,<3.13` (3.11 is not supported by project metadata).
- Linux is the primary tested platform in CI and containers.
- FFmpeg is needed for video/audio workflows.
- Docker with a compatible NVIDIA runtime/driver is needed for the CUDA container path.

Create an isolated environment and install the core plus development tools:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
make setup
```

`pyproject.toml` is the dependency source of truth. `requirements.txt` is intentionally only
an editable development-install wrapper. Optional integrations can be installed together:

```bash
make setup-all
# Equivalent: python -m pip install -e ".[audio,data,dev,export,mlops]"
```

The PyTorch/CUDA/ONNX Runtime combination is coordinated deliberately; do not upgrade those
packages independently. The tested target and official compatibility sources are recorded in
[DEPENDENCY_UPGRADE_REPORT.md](DEPENDENCY_UPGRADE_REPORT.md).

## Model and data prerequisites

The repository tracks no `.pt`, `.onnx`, TensorRT engine, video, or dataset artifact. Before a
real run:

1. Supply a trusted YOLO checkpoint with known provenance and license.
2. Supply input frames/video for which you have processing rights.
3. Set `--gnn-weights` only when a compatible trained position-classifier checkpoint exists.
4. Enable tactical analytics only with `--calibration-path` pointing to a validated calibration.

Container runs read the ignored `models/` directory at `/app/models`. Place a reviewed
checkpoint at `models/yolov8n.pt`, or set `APP_YOLO_WEIGHTS` to another path mounted into the
container. The host-side `YOLO_WEIGHTS` setting is intentionally separate.

PyTorch/PyG artifacts can contain executable pickle payloads. The GNN training CLI therefore
requires `--trust-pickled-dataset`; use it only for a dataset you created or independently
verified. Ultralytics restricted checkpoint loading is enabled by default. An explicit legacy
opt-out should only be used for a reviewed artifact in a controlled environment.

## Run the frame pipeline

```bash
python -m src.pipeline_full \
  --frames-dir /path/to/frames \
  --output-dir outputs/pipeline_run \
  --weights /path/to/trusted-yolo.pt \
  --max-frames 30
```

The output includes per-frame detection and tracking JSON, overlays, graph metadata at
`graphs/final_graph.json`, and `pipeline_summary.json`. Use a new output directory for each CLI
run: named artifacts inside an explicitly selected existing directory may be replaced.

For calibrated tactical analysis:

```bash
python -m src.pipeline_full \
  --frames-dir /path/to/frames \
  --output-dir outputs/calibrated_run \
  --weights /path/to/trusted-yolo.pt \
  --enable-tactical-analytics \
  --calibration-path /path/to/calibration.json
```

## Run the API

Copy `.env.example` values into your environment, then start the loopback-bound development
server:

```bash
make run
```

OpenAPI documentation is at `http://127.0.0.1:8000/docs`. The batch API exposes health,
analysis task submission/status/cancellation, pipeline information, Prometheus metrics, and
tactical result routes. The live API in `src.main:app` exposes stream start/stop, current
detections, checkpoint metadata, and health.

For any non-loopback deployment, set both:

```bash
export API_REQUIRE_KEY=true
export API_KEY='generate-a-long-random-secret-outside-the-repository'
```

Clients then send `X-API-Key`. Configure `CORS_ALLOWED_ORIGINS` explicitly and keep private RTSP
addresses disabled unless the deployment has a narrowly reviewed allowlist. CORS is not an
authentication mechanism. The built-in task store is process-local, bounded, and not suitable
for durable multi-worker job orchestration.

## YouTube and live-source safety

- Only explicit HTTPS YouTube video hosts/IDs are accepted by the batch workflow.
- Download size and duration are bounded by configuration.
- Public live APIs reject private/link-local RTSP targets unless explicitly allowed.
- Output names are constrained beneath `ANALYSIS_OUTPUT_ROOT`.
- Media and model provenance, retention, privacy, and licensing remain deployment responsibilities.

## DVC, MLflow, and containers

`dvc.yaml` defines one `analyze_frames` stage from `params.yaml`. It intentionally fails when
the configured frames or weights are absent; no sample-data fetch stage or DVC remote is claimed.
The DVC CLI is not installed by `setup-all`: DVC 3.67.1 currently pulls DiskCache 5.6.3, whose
default pickle deserialization has an unfixed advisory. Run DVC only in an isolated, user-owned
environment after that transitive risk is fixed or explicitly accepted.

MLflow is optional and fail-open by default. Set `MLFLOW_REQUIRED=true` when losing experiment
tracking must fail the run. The Compose stack exposes the app and optional MLflow service only on
loopback, uses read-only/non-root runtime settings, and provides `dev` and `mlops` profiles:

```bash
docker compose config
docker compose up --build app
APP_BUILD_TARGET=mlops APP_MLFLOW_TRACKING_URI=http://mlflow:5000 \
  docker compose --profile mlops up --build app mlflow
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build app
```

The last command is the explicit NVIDIA GPU override. It requires the NVIDIA Container Toolkit
and a driver compatible with the pinned CUDA image; the base Compose model does not request a
GPU. The MLOps command builds the app with the MLflow dependency and points it at the Compose
service rather than its private local SQLite store.

## Engineering checks

```bash
make lint          # Ruff lint/format check + mypy
make test          # complete pytest suite
make test-cov      # branch coverage report
make lint-all      # all pre-commit hooks
make build         # wheel and source distribution
make smoke         # marked lightweight smoke tests
```

GPU, network, external-service, real-model, and real-media checks are environment-specific and
must not be inferred from the synthetic unit suite.

## Repository layout

```text
src/api/         batch FastAPI application and task lifecycle
src/analytics/   team classification and calibrated tactical calculations
src/calib/       homography and pixel-to-pitch transformations
src/data/        data loading/versioning helpers
src/detect/      Ultralytics inference, LoRA fine-tuning, ONNX/TensorRT export
src/graph/       spatial-temporal graph construction and position classifier
src/live/        capture, overlay, and live processing
src/models/      GraphSAGE dataset/training pipeline
src/track/       tracker runtime and detection-to-track pipeline
src/youtube/     bounded download, metadata, classification, and media helpers
tests/           unit, API, integration-style, security, and smoke tests
scripts/         supported data/evaluation utilities
```

## Contributing and security

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing the code. Report vulnerabilities through
the private channel in [SECURITY.md](SECURITY.md), not a public issue.

## License

Code is licensed under the [MIT License](LICENSE). This license does not establish rights to any
third-party footage, datasets, model weights, logos, or generated media; verify those separately.
