# FIFA Soccer DS Analytics

**Production-ready computer vision pipeline for analyzing FIFA gameplay and YouTube soccer highlights**

This project provides a complete, end-to-end CV pipeline for soccer video analysis with unique capabilities:
- **YouTube Integration**: Analyze both FIFA gameplay footage AND real-world soccer highlights from YouTube
- **Graph Neural Networks**: GraphSAGE-based player interaction modeling for tactical analysis
- **Live Streaming**: Real-time RTSP stream processing and inference
- **Production Ready**: FastAPI deployment, MLflow tracking, DVC data versioning, automated retraining

## What Makes This Different

This project is **NOT**:
- **fifa-ai** - which focuses on playing the FIFA game using AI agents
- **agentic-vscode** - which is a research prototype for AI-assisted development
- **jhsoccer** - which focuses primarily on tactical analytics without the CV pipeline

This project **IS**:
- A complete production pipeline from video ingestion to graph-based analytics
- Optimized for both simulated (FIFA) and real-world (YouTube) soccer footage
- Battle-tested with automated training, monitoring, and deployment workflows

## Technology Stack

- **Detection**: YOLOv8 (fine-tuned for soccer players, ball, referees)
- **Tracking**: ByteTrack with Kalman filtering for robust multi-object tracking
- **Graph ML**: PyTorch Geometric + GraphSAGE for player interaction networks
- **API**: FastAPI for production inference endpoints
- **Monitoring**: MLflow for experiment tracking and model versioning
- **Data**: DVC for dataset versioning and pipeline orchestration
- **Streaming**: RTSP live stream support with real-time processing

## Quickstart

1. Install Python 3.11 and the system libraries needed by PyTorch, OpenCV, and FFmpeg.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   make setup
   ```
3. Optional: fetch and preprocess the toy dataset.
   ```bash
   dvc repro fetch_sample preprocess
   ```
4. Sanity check detection and tracking:
   ```bash
   make smoke      # lightweight tests
   make run-detect # writes outputs/detect/*
   ```
5. Run the full pipeline (detection + tracking + graph):
   ```bash
   python -m src.pipeline_full \
     --frames-dir data/processed/sample \
     --output-dir outputs/pipeline_run \
     --distance-threshold 120.0 \
     --min-confidence 0.35 \
     --max-age 20
   ```
6. Launch the live pipeline once you have a camera/RTSP stream:
   ```bash
   make run-live URL=rtsp://example/stream
   ```

## Project Status

**COMPLETE AND PRODUCTION-READY**

This pipeline is fully implemented with:
- Automated CI/CD testing and deployment
- Production FastAPI endpoints for inference
- Weekly automated model retraining workflows
- Comprehensive monitoring and experiment tracking
- Performance: 22 FPS on 8GB GPU (RTX 3070 class)

## Repository Layout

```
├── src
│   ├── detect      # YOLOv8 inference + export utilities
│   ├── track       # ByteTrack + Kalman filter pipelines
│   ├── graph       # Track-to-graph builders for GNN input
│   ├── models      # GraphSAGE architecture and training
│   ├── live        # RTSP capture + real-time overlay
│   └── utils       # Visualization, MLflow, overlay helpers
├── scripts         # Data fetching (YouTube) and preprocessing
├── tests           # Smoke/unit coverage for all pipeline stages
├── configs         # Hydra runtime configurations
├── docs            # Pipeline guides and experiment results
├── data/           # Raw/processed video assets (DVC tracked)
├── mlruns/         # MLflow experiment tracking artifacts
└── .dvc/           # DVC metadata for dataset versioning
```

## Key Features

### 1. YouTube + FIFA Video Support
- Automated YouTube highlight download and processing
- FIFA gameplay footage analysis
- Unified pipeline handles both simulated and real-world footage
- Adaptive preprocessing for different video qualities and formats

### 2. GraphSAGE Player Interaction Modeling
- Constructs spatial-temporal graphs from tracked player positions
- GraphSAGE GNN learns player role embeddings and tactical patterns
- Identifies passing networks, defensive formations, pressing behavior
- Position classification and tactical analysis

### 3. Live RTSP Streaming
- Real-time processing of RTSP camera streams
- Frame-by-frame detection, tracking, and overlay
- Low-latency inference pipeline optimized for live deployment
- WebRTC/HLS re-streaming capabilities via FastAPI

### 4. Production MLOps
- **MLflow**: Automatic experiment tracking, model versioning, metric logging
- **DVC**: Dataset versioning, reproducible preprocessing pipelines
- **FastAPI**: REST endpoints for batch and real-time inference
- **Automated Retraining**: Weekly model updates with performance monitoring

## Data Management

- DVC orchestrates dataset fetching and preprocessing. Inspect `dvc.yaml` for the declarative pipeline.
- Run `dvc repro fetch_sample preprocess` to download the placeholder clip and generate resized frame stacks under `data/processed/sample/`.
- YouTube integration: The pipeline can automatically fetch and process YouTube soccer highlights
- Replace the URL in `scripts/data/fetch_sample.py` with authorized SoccerNet or Metrica Sports snippets for additional datasets
- Extend `scripts/data/preprocess.py` when adding pitch segmentation or temporal sampling logic; record new steps as additional DVC stages.

## Full Pipeline Usage
The unified pipeline (`src/pipeline_full.py`) integrates detection, tracking, and graph construction:

```bash
python -m src.pipeline_full \
  --frames-dir data/processed/pedri \
  --output-dir outputs/pedri_analysis \
  --confidence 0.35 \
  --distance-threshold 120.0 \
  --max-age 20
```

**Tuned defaults (optimized for La Liga football):**
- `confidence=0.35`: Balances detection recall vs. false positives for distant players
- `distance_threshold=120.0`: Accounts for player movement (~80-120px/frame at 30fps)
- `max_age=20`: Recovers from 3-5 frame occlusions (passes, tackles)

**Output structure:**
- `pipeline_summary.json`: Overall statistics (detections, tracks, graph edges)
- `detections/`: Raw YOLO detection JSON per frame
- `tracklets.json`: Complete track history with confidence scores
- `graph/`: Spatial-temporal graph for position classification

### Detailed Documentation

For comprehensive usage, frame extraction, live API deployment, and troubleshooting:
**See [`docs/PIPELINE_GUIDE.md`](docs/PIPELINE_GUIDE.md)**

Key topics:
- Frame preparation and codec conversion
- Step-by-step pipeline execution
- Output format specifications
- Performance benchmarks (22 FPS on 8GB GPU)
- Live inference via FastAPI
- Weekly retraining automation
- Parameter tuning for different field positions

See also `docs/experiments/` for example runs and parameter tuning results.

## Pipeline Architecture

The complete pipeline consists of these stages:

1. **Video Ingestion** → YouTube downloads or RTSP streams
2. **Detection** → YOLOv8 identifies players, ball, referees
3. **Tracking** → ByteTrack maintains IDs across frames with Kalman filtering
4. **Graph Construction** → Spatial-temporal graphs from tracked positions
5. **GNN Analysis** → GraphSAGE learns player interactions and tactical patterns
6. **Visualization/API** → Annotated overlays or REST endpoints

Each stage logs metrics to MLflow and can be run independently or as a unified pipeline.

## Contributing
- Install Git hooks with `make setup` so formatting and static analysis run automatically.
- Before submitting a PR, execute `make lint-all` and `make test`; include updated smoke/e2e coverage for new behaviours.
- Document notable architectural decisions in `docs/` (create as needed) and surface contributor notes in `AGENTS.md`.
- Follow Conventional Commit messages (`feat:`, `fix:`, `chore:`) and keep each pull request focused with linked issues/screenshots where applicable.

See `AGENTS.md` for contributor guidelines tailored to this project.
