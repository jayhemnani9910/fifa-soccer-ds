[![CI](https://github.com/jayhemnani9910/fifa-soccer-ds/actions/workflows/ci.yml/badge.svg)](https://github.com/jayhemnani9910/fifa-soccer-ds/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open-2ea44f?style=for-the-badge)](https://jayhemnani9910.github.io/fifa-soccer-ds/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](LICENSE)

# FIFA Soccer DS

**Production-ready computer vision pipeline for analyzing soccer gameplay and YouTube highlights.**

Multi-model tracking with YOLOv8 detection, ByteTrack persistence, and a GraphSAGE scaffold for tactical pattern recognition (inference wired; trained weights pending) — served via FastAPI with full MLOps infrastructure.

> **22 FPS** on 8GB GPU (RTX 3070 class) &bull; **YouTube + FIFA** dual-source support &bull; **Live RTSP** streaming

---

## Pipeline Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
│  Video Source    │     │   Detection  │     │   Tracking   │
│  YouTube / RTSP │────▶│   YOLOv8     │────▶│  ByteTrack   │
│  FIFA Gameplay   │     │  (stock COCO) │     │  + Kalman    │
└─────────────────┘     └──────────────┘     └──────┬───────┘
                                                     │
                     ┌───────────────┐     ┌─────────▼────────┐
                     │  FastAPI      │     │  Graph Builder    │
                     │  REST + Live  │◀────│  Spatial-Temporal │
                     └───────┬───────┘     └─────────┬────────┘
                             │                       │
                     ┌───────▼───────┐     ┌─────────▼────────┐
                     │  MLflow       │     │  GraphSAGE GNN   │
                     │  Tracking     │     │  Tactical Analysis│
                     └───────────────┘     └──────────────────┘
```

Each stage logs metrics to MLflow and can run independently or as a unified pipeline.

---

## Key Features

### Detection & Tracking
- **YOLOv8n (stock COCO)** with soccer-tuned tracking params; repository ships `yolov8n.pt`. A fine-tuned soccer checkpoint (players / ball / referees) is not yet bundled — training pipeline exists in `src/detect/` and `dvc.yaml`.
- **ByteTrack** with Kalman filtering for robust multi-object tracking across occlusions
- Tuned for La Liga footage: `confidence=0.35`, `distance_threshold=120px`, `max_age=20` frames

### Graph Neural Networks
- Constructs **spatial-temporal graphs** from tracked player positions
- **GraphSAGE** architecture for learning player role embeddings and tactical patterns — inference is wired into `pipeline_full.py` via `--gnn-weights`; trained checkpoint not yet shipped (see `src/models/train_gcn.py` for the training skeleton)
- Designed to identify passing networks, defensive formations, pressing behavior
- GCN-based position classification (untrained)

### Live Streaming
- Real-time **RTSP** camera stream processing with frame-by-frame inference
- Low-latency overlay rendering
- WebRTC/HLS re-streaming via FastAPI

### MLOps & Production
- **MLflow**: Experiment tracking, model versioning, metric logging
- **DVC**: Dataset versioning and reproducible preprocessing pipelines
- **FastAPI**: REST endpoints for batch and real-time inference
- **Automated Retraining**: Weekly model updates with performance monitoring

### Dual Video Source
- Automated **YouTube** highlight download and processing
- **FIFA** gameplay footage analysis
- Adaptive preprocessing for different video qualities and formats

---

## Tech Stack

![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=flat&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-3C2179?style=flat&logo=pyg&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-13ADC7?style=flat&logo=dvc&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)

---

## Quickstart

```bash
# 1. Setup environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make setup    # installs git hooks

# 2. Fetch sample data
dvc repro fetch_sample preprocess

# 3. Smoke test
make smoke        # lightweight tests
make run-detect   # outputs to outputs/detect/

# 4. Run the full pipeline (detect → track → graph)
python -m src.pipeline_full \
  --frames-dir data/processed/sample \
  --output-dir outputs/pipeline_run \
  --distance-threshold 120.0 \
  --min-confidence 0.35 \
  --max-age 20

# 5. Live streaming (requires RTSP source)
make run-live URL=rtsp://example/stream
```

---

## Project Structure

```
├── src/
│   ├── detect/       # YOLOv8 inference, ONNX/TRT export, LoRA adapters
│   ├── track/        # ByteTrack + Kalman filter pipeline
│   ├── graph/        # Track-to-graph builders for GNN input
│   ├── models/       # GraphSAGE architecture and training
│   ├── analytics/    # Tactical analysis, team classification, pitch control
│   ├── live/         # RTSP capture + real-time overlay
│   ├── youtube/      # Video download, audio extraction, metadata parsing
│   ├── api/          # FastAPI endpoints (inference + YouTube)
│   ├── calib/        # Homography and pitch transformation
│   └── utils/        # Visualization, MLflow helpers, monitoring
├── scripts/          # Data fetching, preprocessing, performance testing
├── tests/            # Smoke, unit, e2e, and security test suites
├── configs/          # Hydra runtime configurations
├── dvc.yaml          # Declarative DVC pipeline stages
└── Makefile          # Common commands (setup, test, run, deploy)
```

---

## Pipeline Output

Running the full pipeline produces:

| Output | Description |
|--------|-------------|
| `pipeline_summary.json` | Overall statistics (detections, tracks, graph edges) |
| `detections/` | Raw YOLO detection JSON per frame |
| `tracklets.json` | Complete track history with confidence scores |
| `graph/` | Spatial-temporal graph for position classification |

---

## Performance

| Metric | Value |
|--------|-------|
| Inference FPS | **22 FPS** (8GB GPU, RTX 3070 class) |
| Detection model | YOLOv8n (stock COCO; `person` + `sports ball` classes used) |
| Tracking | ByteTrack with Kalman filtering |
| Occlusion recovery | 3-5 frame gaps (`max_age=20`) |

---

## Contributing

- Install git hooks with `make setup`
- Run `make lint-all` and `make test` before submitting PRs
- Follow Conventional Commit messages (`feat:`, `fix:`, `chore:`)
- See `AGENTS.md` for contributor guidelines

---

## License

MIT
