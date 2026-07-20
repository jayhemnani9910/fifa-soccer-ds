"""Full pipeline driver combining detection, tracking, and graph construction.

This script processes a directory of frames through complete analysis pipeline:
1. Detection: YOLO object detection
2. Tracking: ByteTrack multi-object tracking
3. Graph: Spatial/temporal relationship graphs
4. Visualization: Overlays and summaries
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import re
import shutil
import signal
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

try:  # pragma: no cover - optional dependency
    cv2: Any = importlib.import_module("cv2")
except ImportError:
    cv2 = None

from src.detect.infer import InferenceConfig, extract_detections, first_prediction, load_model
from src.graph.build_graph import build_track_graph
from src.schemas import validate_youtube_url
from src.track.bytetrack_runtime import ByteTrackRuntime, Tracklets
from src.track.pipeline import filter_detections
from src.utils.mlflow_helper import start_run
from src.utils.monitoring import (
    DETECTIONS_MADE,
    FRAMES_PROCESSED,
    TRACKS_CREATED,
    timed_processing,
    update_system_metrics,
)
from src.utils.overlay import draw_boxes, draw_track_ids

# Tactical analytics imports
try:
    from src.analytics.tactical import PlayerState, TacticalAnalyzer, TacticalConfig
    from src.analytics.team_classifier import JerseyColorClassifier, TeamAssignment  # noqa: F401
    from src.calib.pitch_transform import PitchCoordinateTransformer

    TACTICAL_AVAILABLE = True
except ImportError:
    TACTICAL_AVAILABLE = False

# YouTube integration imports
try:
    from src.classify.soccer_classifier import SoccerClassifier
    from src.youtube.audio_extractor import AudioExtractor  # noqa: F401
    from src.youtube.metadata_parser import YouTubeMetadataParser  # noqa: F401
    from src.youtube.video_downloader import YouTubeDownloader

    YOUTUBE_AVAILABLE = True
except ImportError:
    YOUTUBE_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    mlflow: Any = importlib.import_module("mlflow")
except ImportError:
    mlflow = None

LOGGER = logging.getLogger(__name__)
FRAME_FILE_PATTERN = re.compile(r"frame_(\d+)\.jpg", re.IGNORECASE)

shutdown_requested = False


def _configure_logging(level: str = "INFO") -> None:
    """Initialise logging handlers once per process."""

    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline.log", encoding="utf-8"),
        ],
    )


def _register_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def signal_handler(signum, frame) -> None:
    """Handle shutdown signals gracefully."""

    del frame
    global shutdown_requested
    LOGGER.info("Received signal %s, initiating graceful shutdown...", signum)
    shutdown_requested = True


@dataclass(slots=True)
class PipelineConfig:
    """Configuration for the full pipeline."""

    # Detection config
    weights: str = "yolov8n.pt"
    device: str = "cuda_if_available"
    confidence: float = 0.25

    # Tracking config
    min_confidence: float = 0.25

    # Detection hygiene
    allowed_classes: tuple[str, ...] = ("person", "sports ball")
    max_bbox_area_ratio: float = 0.35
    nms_iou: float = 0.6
    distance_threshold: float = 80.0
    max_age: int = 15

    # Graph config
    graph_window: int = 30
    graph_distance_threshold: float = 80.0
    include_temporal_edges: bool = True

    # GNN inference config
    gnn_weights: str = ""  # Path to PositionClassifier checkpoint; empty = skip

    # Processing config
    max_frames: int = 30
    max_frame_bytes: int = 25 * 1024 * 1024
    max_frame_pixels: int = 50_000_000

    # Tactical analytics config
    enable_tactical_analytics: bool = False
    tactical_grid_shape: tuple[int, int] = (12, 16)
    enable_team_classification: bool = True
    team_samples_per_track: int = 5
    max_team_sample_frames: int = 100
    calibration_path: str = ""  # Path to homography calibration file

    def __post_init__(self) -> None:
        unit_interval = {
            "confidence": self.confidence,
            "min_confidence": self.min_confidence,
            "nms_iou": self.nms_iou,
        }
        for name, value in unit_interval.items():
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite value between 0 and 1")

        if not math.isfinite(self.max_bbox_area_ratio) or not 0 < self.max_bbox_area_ratio <= 1:
            raise ValueError("max_bbox_area_ratio must be a finite value in (0, 1]")

        positive_floats = {
            "distance_threshold": self.distance_threshold,
            "graph_distance_threshold": self.graph_distance_threshold,
        }
        for name, value in positive_floats.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite value")

        non_negative_integers = {"max_age": self.max_age, "max_frames": self.max_frames}
        for name, value in non_negative_integers.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

        positive_integers = {
            "graph_window": self.graph_window,
            "max_frame_bytes": self.max_frame_bytes,
            "max_frame_pixels": self.max_frame_pixels,
            "team_samples_per_track": self.team_samples_per_track,
            "max_team_sample_frames": self.max_team_sample_frames,
        }
        for name, value in positive_integers.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if len(self.tactical_grid_shape) != 2 or any(
            value <= 0 or value > 256 for value in self.tactical_grid_shape
        ):
            raise ValueError("tactical_grid_shape must contain two values between 1 and 256")
        if self.enable_tactical_analytics:
            if not self.calibration_path:
                raise ValueError("tactical analytics requires a calibrated homography file")
            if not Path(self.calibration_path).is_file():
                raise FileNotFoundError(
                    f"Tactical calibration file not found: {self.calibration_path}"
                )


@dataclass(slots=True)
class PipelineSummary:
    """Summary of pipeline execution."""

    total_frames: int = 0
    attempted_frames: int = 0
    successful_frames: int = 0
    unreadable_frames: int = 0
    detection_failures: int = 0
    tracking_failures: int = 0
    overlay_failures: int = 0
    total_detections: int = 0
    unique_track_ids: list[int] = field(default_factory=list)
    graph_nodes: int = 0
    graph_edges: int = 0
    output_dir: str = ""
    # Tactical analytics summary
    tactical_enabled: bool = False
    avg_home_control_pct: float = 0.0
    avg_away_control_pct: float = 0.0
    team_assignments: dict = field(default_factory=dict)


def _discover_frame_files(frames_dir: Path) -> list[Path]:
    """Return regular, non-symlink JPEG frames in deterministic numeric order."""

    numbered_frames: list[tuple[int, Path]] = []
    for candidate in frames_dir.iterdir():
        match = FRAME_FILE_PATTERN.fullmatch(candidate.name)
        if match and candidate.is_file() and not candidate.is_symlink():
            numbered_frames.append((int(match.group(1)), candidate))
    return [path for _, path in sorted(numbered_frames, key=lambda item: (item[0], item[1].name))]


def _frame_is_safe_to_decode(frame_path: Path, config: PipelineConfig) -> bool:
    """Validate encoded size, format, integrity, and dimensions before OpenCV decode."""

    try:
        encoded_bytes = frame_path.stat(follow_symlinks=False).st_size
        if encoded_bytes <= 0 or encoded_bytes > config.max_frame_bytes:
            LOGGER.warning(
                "Rejected frame %s: encoded size %d is outside 1..%d bytes",
                frame_path.name,
                encoded_bytes,
                config.max_frame_bytes,
            )
            return False

        with Image.open(frame_path) as image:
            width, height = image.size
            if image.format != "JPEG":
                LOGGER.warning("Rejected frame %s: content is not JPEG", frame_path.name)
                return False
            if width <= 0 or height <= 0 or width * height > config.max_frame_pixels:
                LOGGER.warning(
                    "Rejected frame %s: dimensions %dx%d exceed pixel limit %d",
                    frame_path.name,
                    width,
                    height,
                    config.max_frame_pixels,
                )
                return False
            image.verify()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        LOGGER.warning("Rejected unreadable frame %s: %s", frame_path.name, exc)
        return False

    return True


@timed_processing
def process_frames_directory(
    frames_dir: Path,
    output_dir: Path,
    config: PipelineConfig | None = None,
) -> PipelineSummary:
    """Process a directory of frames through the full pipeline.

    Args:
        frames_dir: Directory containing frame images
        output_dir: Directory to save results
        config: Pipeline configuration options

    Returns:
        PipelineSummary: Summary of processing results

    Raises:
        FileNotFoundError: If frames directory doesn't exist
        ValueError: If configuration is invalid
        ImportError: If required dependencies are missing
    """
    cfg = config or PipelineConfig()
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)

    # Validate inputs
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    if not frames_dir.is_dir():
        raise ValueError(f"Frames path is not a directory: {frames_dir}")

    # Create output subdirectories with error handling
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o750)
    for subdir in ["overlays", "detections", "tracks", "graphs"]:
        try:
            (output_dir / subdir).mkdir(exist_ok=True, mode=0o750)
        except OSError as e:
            LOGGER.error("Failed to create directory %s: %s", subdir, e)
            raise

    # Load detector with error handling
    LOGGER.info("Loading YOLO model: %s", cfg.weights)
    try:
        detector_cfg = InferenceConfig(
            weights=cfg.weights,
            device=cfg.device,
            confidence=cfg.confidence,
        )
        detector = load_model(detector_cfg)
    except Exception as e:
        LOGGER.error("Failed to load detector model: %s", e)
        raise RuntimeError("Model loading failed") from e

    # Initialize tracker with error handling
    LOGGER.info(
        "Initializing tracker: min_confidence=%.2f, distance_threshold=%.1f",
        cfg.min_confidence,
        cfg.distance_threshold,
    )
    try:
        tracker = ByteTrackRuntime(
            min_confidence=cfg.min_confidence,
            distance_threshold=cfg.distance_threshold,
            max_age=cfg.max_age,
        )
    except Exception as e:
        LOGGER.error("Failed to initialize tracker: %s", e)
        raise RuntimeError(f"Tracker initialization failed: {e}") from e

    # Collect all frame files with validation
    frame_files = _discover_frame_files(frames_dir)
    if not frame_files:
        raise ValueError(f"No frame files found in {frames_dir}")

    LOGGER.info("Found %d frames to process", len(frame_files))

    # Limit frames if configured
    if cfg.max_frames > 0:
        frame_files = frame_files[: cfg.max_frames]
        LOGGER.info("Processing first %d frames", len(frame_files))

    # Process frames with error handling
    track_history: list[Tracklets] = []
    all_track_ids = set()
    total_detections = 0
    failed_frames = 0
    processed_frames = 0
    detection_failures = 0
    tracking_failures = 0
    overlay_failures = 0
    failed_frame_indices: set[int] = set()

    for frame_idx, frame_path in enumerate(frame_files):
        LOGGER.info("Processing frame %d/%d: %s", frame_idx + 1, len(frame_files), frame_path.name)

        # Read frame with validation
        if cv2 is None:
            raise ImportError("OpenCV required for frame processing")

        if not _frame_is_safe_to_decode(frame_path, cfg):
            failed_frames += 1
            failed_frame_indices.add(frame_idx)
            continue

        try:
            frame = cv2.imread(frame_path.as_posix())
            if frame is None:
                LOGGER.warning("Failed to read frame: %s", frame_path)
                failed_frames += 1
                failed_frame_indices.add(frame_idx)
                continue
        except Exception as e:
            LOGGER.error("Error reading frame %s: %s", frame_path, e)
            failed_frames += 1
            failed_frame_indices.add(frame_idx)
            continue

        processed_frames += 1

        # Detection with error handling
        try:
            results = detector.predict(frame, conf=cfg.confidence, verbose=False)
            prediction = first_prediction(results)
            detections = extract_detections(prediction) if prediction is not None else []
            total_detections += len(detections)

            # Update metrics
            FRAMES_PROCESSED.inc()
            DETECTIONS_MADE.inc(len(detections))
            update_system_metrics()

        except Exception as e:
            LOGGER.error("Detection failed for frame %s: %s", frame_path.name, e)
            detection_failures += 1
            failed_frame_indices.add(frame_idx)
            detections = []

        # Save raw detections with error handling
        try:
            detection_path = output_dir / "detections" / f"{frame_path.stem}_detections.json"
            with detection_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "frame_id": frame_idx,
                        "frame_name": frame_path.name,
                        "num_detections": len(detections),
                        "detections": detections,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            LOGGER.error("Failed to save detections for frame %s: %s", frame_path.name, e)
            raise RuntimeError(f"Failed to persist detections for {frame_path.name}") from e

        # Tracking with error handling
        try:
            filtered = filter_detections(
                detections,
                min_confidence=cfg.min_confidence,
                allowed_classes=set(cfg.allowed_classes) if cfg.allowed_classes else None,
                max_area_ratio=cfg.max_bbox_area_ratio,
                frame_shape=tuple(frame.shape[:2]) if frame is not None else None,
                nms_iou=cfg.nms_iou,
            )
            tracklets = tracker.update(frame_id=frame_idx, detections=filtered)
            track_history.append(tracklets)

            # Collect unique track IDs and update metrics
            for track in tracklets.items:
                all_track_ids.add(track.track_id)

            # Update tracking metrics
            TRACKS_CREATED.inc(len(tracklets.items))

        except Exception as e:
            LOGGER.error("Tracking failed for frame %s: %s", frame_path.name, e)
            tracking_failures += 1
            failed_frame_indices.add(frame_idx)
            tracklets = Tracklets(frame_id=frame_idx, items=[])
            track_history.append(tracklets)

        # Save track info with error handling
        try:
            track_path = output_dir / "tracks" / f"{frame_path.stem}_tracks.json"
            with track_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "frame_id": frame_idx,
                        "frame_name": frame_path.name,
                        "num_tracks": len(tracklets.items),
                        "tracks": [
                            {
                                "track_id": t.track_id,
                                "bbox": t.bbox,
                                "score": t.score,
                                "class_name": t.extras.get("class_name"),
                            }
                            for t in tracklets.items
                        ],
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            LOGGER.error("Failed to save tracks for frame %s: %s", frame_path.name, e)
            raise RuntimeError(f"Failed to persist tracks for {frame_path.name}") from e

        # Visualization with error handling
        if tracklets.items:
            try:
                boxes = [track.bbox for track in tracklets.items]
                labels = [str(track.extras.get("class_name", "")) for track in tracklets.items]
                overlay = draw_boxes(frame.copy(), boxes=boxes, labels=labels)
                ids = [track.track_id for track in tracklets.items]
                overlay = draw_track_ids(overlay, boxes=boxes, track_ids=ids)

                overlay_path = output_dir / "overlays" / f"{frame_path.stem}_overlay.jpg"
                success = cv2.imwrite(overlay_path.as_posix(), overlay)
                if not success:
                    LOGGER.warning("Failed to write overlay image: %s", overlay_path)
                    overlay_failures += 1
            except Exception as e:
                LOGGER.error("Visualization failed for frame %s: %s", frame_path.name, e)
                overlay_failures += 1

    if processed_frames == 0:
        raise RuntimeError("No readable frames were processed")
    if detection_failures == processed_frames:
        raise RuntimeError("Detection failed for every readable frame")
    if tracking_failures == processed_frames:
        raise RuntimeError("Tracking failed for every readable frame")
    if failed_frames or detection_failures or tracking_failures or overlay_failures:
        LOGGER.warning(
            "Partial processing: %d unreadable, %d detection, %d tracking, and %d overlay failures",
            failed_frames,
            detection_failures,
            tracking_failures,
            overlay_failures,
        )

    # ===== TACTICAL ANALYTICS =====
    tactical_results = []
    team_assignments_dict = {}
    avg_home_control = 0.0
    avg_away_control = 0.0

    if cfg.enable_tactical_analytics and TACTICAL_AVAILABLE:
        LOGGER.info("Running tactical analytics...")

        try:
            # Create tactical output directory
            (output_dir / "tactical").mkdir(exist_ok=True)

            transformer = PitchCoordinateTransformer.from_saved_calibration(
                Path(cfg.calibration_path)
            )

            # Initialize tactical analyzer
            tactical_config = TacticalConfig(
                grid_shape=cfg.tactical_grid_shape,
            )
            tactical_analyzer = TacticalAnalyzer(config=tactical_config)

            # Team classification if enabled
            if cfg.enable_team_classification:
                LOGGER.info("Classifying teams based on jersey colors...")
                try:
                    team_classifier = JerseyColorClassifier()

                    # Collect tracklet data for classification
                    all_tracklet_data: list[dict[str, Any]] = []

                    for tracklets in track_history:
                        frame_id = tracklets.frame_id
                        for track in tracklets.items:
                            all_tracklet_data.append(
                                {
                                    "track_id": track.track_id,
                                    "bbox": track.bbox,
                                    "frame_id": frame_id,
                                    "class_name": track.extras.get("class_name", ""),
                                }
                            )

                    samples_by_track: dict[int, int] = {}
                    sampled_frame_ids: set[int] = set()
                    for tracklet in all_tracklet_data:
                        track_id = int(tracklet["track_id"])
                        if samples_by_track.get(track_id, 0) >= cfg.team_samples_per_track:
                            continue
                        sampled_frame_ids.add(int(tracklet["frame_id"]))
                        samples_by_track[track_id] = samples_by_track.get(track_id, 0) + 1

                    frame_images: dict[int, Any] = {}
                    for frame_id in sorted(sampled_frame_ids)[: cfg.max_team_sample_frames]:
                        source_frame = (
                            frame_files[frame_id] if frame_id < len(frame_files) else None
                        )
                        if source_frame is not None and cv2 is not None:
                            image = cv2.imread(source_frame.as_posix())
                            if image is not None:
                                frame_images[frame_id] = image

                    # Run classification
                    team_assignments = team_classifier.classify_tracks(
                        frame_images,
                        all_tracklet_data,
                        sample_frames=cfg.team_samples_per_track,
                    )

                    # Convert to dict for serialization
                    team_assignments_dict = {
                        str(ta.track_id): {
                            "team_id": ta.team_id,
                            "confidence": ta.confidence,
                            "color": ta.dominant_color,
                        }
                        for ta in team_assignments.values()
                    }

                    LOGGER.info("Classified %d players into teams", len(team_assignments))

                except Exception as e:
                    raise RuntimeError("Team classification failed") from e
            else:
                team_assignments = {}

            # Compute tactical analytics for each frame
            for tracklets in track_history:
                frame_id = tracklets.frame_id

                # Convert tracklets to player states
                players = []
                for track in tracklets.items:
                    class_name = track.extras.get("class_name", "")
                    if class_name not in ("person", "player", ""):
                        continue

                    # Get position from bbox
                    position = transformer.bbox_to_pitch_position(track.bbox)

                    # A player's horizontal location is not evidence of team identity.
                    # Omit unknown assignments instead of contaminating tactical metrics.
                    assignment = team_assignments.get(track.track_id)
                    if assignment is None or assignment.team_id not in {0, 1}:
                        continue

                    players.append(
                        PlayerState(
                            player_id=track.track_id,
                            team_id=assignment.team_id,
                            position=position,
                            velocity=None,
                        )
                    )

                if {player.team_id for player in players} != {0, 1}:
                    LOGGER.debug(
                        "Skipping tactical frame %d: both teams were not classified",
                        frame_id,
                    )
                    continue

                # Compute tactical metrics only when both teams are represented.
                result = tactical_analyzer.compute(frame_id, players)
                tactical_results.append(result)

            # Calculate averages
            if not tactical_results:
                raise RuntimeError(
                    "Tactical analytics produced no frames with both teams classified"
                )
            if tactical_results:
                avg_home_control = sum(
                    r.pitch_control.home_control_pct for r in tactical_results
                ) / len(tactical_results)
                avg_away_control = sum(
                    r.pitch_control.away_control_pct for r in tactical_results
                ) / len(tactical_results)

                # Save tactical results
                tactical_output = {
                    "enabled": True,
                    "grid_shape": list(cfg.tactical_grid_shape),
                    "calibration_used": bool(cfg.calibration_path),
                    "team_classification": {
                        "enabled": cfg.enable_team_classification,
                        "assignments": team_assignments_dict,
                    },
                    "aggregates": {
                        "avg_home_control_pct": round(avg_home_control, 2),
                        "avg_away_control_pct": round(avg_away_control, 2),
                        "num_frames_analyzed": len(tactical_results),
                    },
                    "frames": [
                        {
                            "frame_id": r.frame_id,
                            "pitch_control": {
                                "home_pct": round(r.pitch_control.home_control_pct, 2),
                                "away_pct": round(r.pitch_control.away_control_pct, 2),
                            },
                            "obso": {
                                "home_total": round(r.home_obso_total, 4),
                                "away_total": round(r.away_obso_total, 4),
                            },
                        }
                        for r in tactical_results
                    ],
                }

                tactical_path = output_dir / "tactical" / "tactical_analysis.json"
                with tactical_path.open("w", encoding="utf-8") as f:
                    json.dump(tactical_output, f, indent=2)

                LOGGER.info(
                    "Tactical analysis complete: Home %.1f%%, Away %.1f%%",
                    avg_home_control,
                    avg_away_control,
                )

        except Exception as e:
            LOGGER.error("Tactical analytics failed: %s", e)
            raise RuntimeError("Tactical analytics failed") from e

    elif cfg.enable_tactical_analytics and not TACTICAL_AVAILABLE:
        raise RuntimeError("Tactical analytics requested but dependencies are unavailable")

    # Build final graph
    LOGGER.info("Building spatial-temporal graph...")
    try:
        graph_data = build_track_graph(
            track_history,
            window=cfg.graph_window,
            distance_threshold=cfg.graph_distance_threshold,
            include_temporal_edges=cfg.include_temporal_edges,
        )

        if isinstance(graph_data, dict):
            num_nodes = graph_data["x"].shape[0] if "x" in graph_data else 0
            num_edges = graph_data["edge_index"].shape[1] if "edge_index" in graph_data else 0
        else:
            num_nodes = graph_data.x.shape[0]
            num_edges = graph_data.edge_index.shape[1]

        LOGGER.info("Graph: %d nodes, %d edges", num_nodes, num_edges)

        # Save graph metadata
        graph_path = output_dir / "graphs" / "final_graph.json"
        with graph_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_nodes": num_nodes,
                    "num_edges": num_edges,
                    "unique_track_ids": sorted(all_track_ids),
                    "window": cfg.graph_window,
                    "distance_threshold": cfg.graph_distance_threshold,
                },
                f,
                indent=2,
            )

    except Exception as e:
        LOGGER.error("Failed to build graph: %s", e)
        raise RuntimeError("Graph construction failed") from e

    # GNN inference (optional)
    if cfg.gnn_weights and graph_data is not None and num_nodes > 0:
        try:
            from src.graph.gcn_position_classifier import predict_positions

            LOGGER.info("Running GNN position classifier: %s", cfg.gnn_weights)
            predictions = predict_positions(graph_data, cfg.gnn_weights)
            preds_path = output_dir / "graphs" / "predictions.json"
            with preds_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {"num_predictions": len(predictions), "predictions": predictions},
                    f,
                    indent=2,
                )
            LOGGER.info("GNN predictions written: %s (%d nodes)", preds_path, len(predictions))
        except Exception as e:
            LOGGER.error("GNN inference failed: %s", e)
            LOGGER.debug(traceback.format_exc())
            raise RuntimeError("Configured GNN inference failed") from e
    elif cfg.gnn_weights:
        LOGGER.warning("GNN weights provided but graph is empty; skipping inference")

    # Create summary
    summary = PipelineSummary(
        total_frames=processed_frames,
        attempted_frames=len(frame_files),
        successful_frames=len(frame_files) - len(failed_frame_indices),
        unreadable_frames=failed_frames,
        detection_failures=detection_failures,
        tracking_failures=tracking_failures,
        overlay_failures=overlay_failures,
        total_detections=total_detections,
        unique_track_ids=sorted(all_track_ids),
        graph_nodes=num_nodes,
        graph_edges=num_edges,
        output_dir=output_dir.as_posix(),
        tactical_enabled=bool(tactical_results),
        avg_home_control_pct=avg_home_control,
        avg_away_control_pct=avg_away_control,
        team_assignments=team_assignments_dict,
    )

    # Save summary
    summary_path = output_dir / "pipeline_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        summary_data = {
            "total_frames": summary.total_frames,
            "attempted_frames": summary.attempted_frames,
            "successful_frames": summary.successful_frames,
            "processing_success_rate": (
                summary.successful_frames / summary.attempted_frames
                if summary.attempted_frames
                else None
            ),
            "partial_failure": bool(
                summary.successful_frames < summary.attempted_frames or summary.overlay_failures
            ),
            "failures": {
                "unreadable_frames": summary.unreadable_frames,
                "detection_failures": summary.detection_failures,
                "tracking_failures": summary.tracking_failures,
                "overlay_failures": summary.overlay_failures,
            },
            "total_detections": summary.total_detections,
            "unique_track_ids": summary.unique_track_ids,
            "num_unique_tracks": len(summary.unique_track_ids),
            "graph_nodes": summary.graph_nodes,
            "graph_edges": summary.graph_edges,
            "output_dir": summary.output_dir,
            "config": {
                "weights": cfg.weights,
                "confidence": cfg.confidence,
                "min_confidence": cfg.min_confidence,
                "distance_threshold": cfg.distance_threshold,
                "max_age": cfg.max_age,
                "graph_window": cfg.graph_window,
                "graph_distance_threshold": cfg.graph_distance_threshold,
            },
        }

        # Add tactical analytics summary if enabled
        if summary.tactical_enabled:
            summary_data["tactical_analytics"] = {
                "enabled": True,
                "avg_home_control_pct": round(summary.avg_home_control_pct, 2),
                "avg_away_control_pct": round(summary.avg_away_control_pct, 2),
                "team_assignments": summary.team_assignments,
            }

        json.dump(summary_data, f, indent=2)

    LOGGER.info("Pipeline complete! Summary saved to: %s", summary_path)

    # Log to MLflow if available
    if mlflow is not None:
        try:
            with start_run(experiment="pipeline_full", run_name=frames_dir.name) as active_run:
                if active_run is None:
                    LOGGER.warning("MLflow run unavailable; skipping pipeline telemetry")
                else:
                    mlflow.log_params(
                        {
                            "weights": cfg.weights,
                            "confidence": cfg.confidence,
                            "min_confidence": cfg.min_confidence,
                            "distance_threshold": cfg.distance_threshold,
                            "max_age": cfg.max_age,
                            "graph_window": cfg.graph_window,
                            "graph_distance_threshold": cfg.graph_distance_threshold,
                        }
                    )
                    mlflow.log_metrics(
                        {
                            "total_frames": summary.total_frames,
                            "attempted_frames": summary.attempted_frames,
                            "successful_frames": summary.successful_frames,
                            "unreadable_frames": summary.unreadable_frames,
                            "detection_failures": summary.detection_failures,
                            "tracking_failures": summary.tracking_failures,
                            "overlay_failures": summary.overlay_failures,
                            "total_detections": summary.total_detections,
                            "unique_tracks": len(summary.unique_track_ids),
                            "graph_nodes": summary.graph_nodes,
                            "graph_edges": summary.graph_edges,
                            "avg_detections_per_frame": (
                                summary.total_detections / summary.total_frames
                                if summary.total_frames > 0
                                else 0
                            ),
                        }
                    )
                    mlflow.log_artifact(summary_path.as_posix())
                    LOGGER.info("Logged metrics to MLflow")
        except Exception as e:
            LOGGER.warning("Failed to log to MLflow: %s", e)

    return summary


@timed_processing
def process_youtube_video(
    youtube_url: str,
    output_dir: Path,
    config: PipelineConfig | None = None,
    sample_duration: float = 60.0,
    force_full_analysis: bool = False,
    include_audio: bool = False,
    max_duration: int | None = None,
) -> dict:
    """Process YouTube video through Smart YouTube Analyzer.

    Args:
        youtube_url: YouTube video URL
        output_dir: Directory to save results
        config: Pipeline configuration options
        sample_duration: Duration of audio sample for classification (seconds)
        force_full_analysis: Force full soccer analysis even if low confidence
        include_audio: Whether to run audio classification
        max_duration: Reject videos longer than this many seconds when metadata is available

    Returns:
        Dict containing analysis results

    Raises:
        ImportError: If YouTube dependencies are not installed
        ValueError: If YouTube URL is invalid
        RuntimeError: If processing fails
    """
    if not YOUTUBE_AVAILABLE:
        raise ImportError(
            "YouTube dependencies are unavailable. Install the project runtime dependencies."
        )

    cfg = config or PipelineConfig()
    output_dir = Path(output_dir)

    # Validate YouTube URL
    if not _is_valid_youtube_url(youtube_url):
        raise ValueError(f"Invalid YouTube URL: {youtube_url}")
    if max_duration is not None and max_duration <= 0:
        raise ValueError("max_duration must be positive")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        LOGGER.info("Starting YouTube analysis: %s", youtube_url)

        # Step 1: Classify content (soccer vs non-soccer)
        LOGGER.info("Step 1: Classifying content...")
        classifier = SoccerClassifier(confidence_threshold=0.75)
        classification_result = classifier.classify_youtube_content(
            youtube_url,
            sample_duration=sample_duration,
            include_audio=include_audio,
            max_video_duration=max_duration,
        )

        # Step 2: Handle non-soccer content
        if not classification_result["is_soccer"] and not force_full_analysis:
            LOGGER.info("Content classified as non-soccer")

            # Generate minimal summary for non-soccer content
            result: dict[str, Any] = {
                "type": "non-soccer",
                "youtube_url": youtube_url,
                "classification": classification_result,
                "summary": {
                    "confidence": classification_result["confidence"],
                    "reason": "Content does not appear to be soccer-related",
                    "recommendation": "This video does not contain soccer content",
                },
                "processing_info": {
                    "sample_duration": sample_duration,
                    "analysis_type": "classification_only",
                },
            }

            # Save result
            result_path = output_dir / "youtube_analysis.json"
            with result_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            LOGGER.info("Non-soccer analysis complete: %s", result_path)
            return result

        # Step 3: Handle soccer content (full analysis)
        LOGGER.info("Content classified as soccer - proceeding with full analysis")

        # Download video
        LOGGER.info("Step 2: Downloading video...")
        download_dir = Path(tempfile.mkdtemp(prefix="fifa_youtube_download_"))
        try:
            downloader = YouTubeDownloader(cache_dir=download_dir)
            video_info = downloader.download_video(youtube_url, max_duration=max_duration)
            video_path = Path(video_info["video_path"])

            # Extract frames for processing
            LOGGER.info("Step 3: Extracting frames...")
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(exist_ok=True)

            # Use existing frame extraction logic
            _extract_frames_from_video(video_path, frames_dir, max_frames=cfg.max_frames)

            # Run existing pipeline on extracted frames
            LOGGER.info("Step 4: Running soccer analysis pipeline...")
            pipeline_summary = process_frames_directory(
                frames_dir=frames_dir,
                output_dir=output_dir / "soccer_analysis",
                config=config,
            )

            # Generate comprehensive result
            LOGGER.info("Step 5: Generating comprehensive report...")
            report_path = output_dir / "analysis_report.md"
            report_path.write_text(_generate_tactical_report(pipeline_summary), encoding="utf-8")
            result = {
                "type": "soccer",
                "youtube_url": youtube_url,
                "classification": classification_result,
                "match": {
                    "title": classification_result["processing_info"]["video_info"]["title"],
                    "duration": classification_result["processing_info"]["video_info"]["duration"],
                    "uploader": classification_result["processing_info"]["video_info"]["uploader"],
                    "detected_competition": _detect_competition_from_title(
                        classification_result["processing_info"]["video_info"]["title"]
                    ),
                },
                "pipeline_summary": {
                    "total_frames": pipeline_summary.total_frames,
                    "attempted_frames": pipeline_summary.attempted_frames,
                    "successful_frames": pipeline_summary.successful_frames,
                    "processing_success_rate": (
                        pipeline_summary.successful_frames / pipeline_summary.attempted_frames
                        if pipeline_summary.attempted_frames
                        else None
                    ),
                    "failures": {
                        "unreadable_frames": pipeline_summary.unreadable_frames,
                        "detection_failures": pipeline_summary.detection_failures,
                        "tracking_failures": pipeline_summary.tracking_failures,
                        "overlay_failures": pipeline_summary.overlay_failures,
                    },
                    "total_detections": pipeline_summary.total_detections,
                    "unique_track_ids": pipeline_summary.unique_track_ids,
                    "graph_nodes": pipeline_summary.graph_nodes,
                    "graph_edges": pipeline_summary.graph_edges,
                },
                "events": _extract_events_from_analysis(pipeline_summary),
                "players": _analyze_players_from_tracks(pipeline_summary),
                "team_metrics": _calculate_team_metrics(pipeline_summary),
                "tactical_analytics": (
                    {
                        "enabled": True,
                        "avg_home_control_pct": round(pipeline_summary.avg_home_control_pct, 2),
                        "avg_away_control_pct": round(pipeline_summary.avg_away_control_pct, 2),
                        "team_assignments": pipeline_summary.team_assignments,
                    }
                    if pipeline_summary.tactical_enabled
                    else None
                ),
                "capabilities": {
                    "event_detection": "not_implemented",
                    "player_statistics": "not_implemented",
                    "team_possession_ppda_field_tilt": "not_implemented",
                },
                "figures": {
                    "detection_visualizations": str(output_dir / "soccer_analysis" / "overlays"),
                    "graphs": str(output_dir / "soccer_analysis" / "graphs"),
                },
                "written_report_md": str(report_path),
                "processing_info": {
                    "sample_duration": sample_duration,
                    "analysis_type": "full_soccer_analysis",
                    "download_file_size_bytes": video_info["file_size"],
                    "frames_extracted": len(list(frames_dir.glob("*.jpg"))),
                },
            }

            # Save comprehensive result
            result_path = output_dir / "youtube_analysis.json"
            with result_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            LOGGER.info("Full soccer analysis complete: %s", result_path)
            return result

        finally:
            shutil.rmtree(download_dir, ignore_errors=True)
            LOGGER.info("Cleaned up task download directory: %s", download_dir)

    except Exception as e:
        LOGGER.error("YouTube analysis failed: %s", e)
        raise RuntimeError(f"YouTube analysis failed: {e}") from e


def _is_valid_youtube_url(url: str) -> bool:
    """Check if URL is a valid YouTube URL."""
    return validate_youtube_url(url)


def _extract_frames_from_video(video_path: Path, output_dir: Path, max_frames: int = 30) -> None:
    """Extract frames from video file.

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        max_frames: Maximum number of frames to extract
    """
    cap = None
    try:
        import cv2

        output_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame interval for desired number of frames
        if max_frames > 0 and total_frames > max_frames:
            frame_interval = total_frames // max_frames
        else:
            frame_interval = 1

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame at specified intervals
            if frame_count % frame_interval == 0:
                frame_filename = output_dir / f"frame_{saved_count:06d}.jpg"
                if not cv2.imwrite(frame_filename.as_posix(), frame):
                    raise RuntimeError(f"Could not write extracted frame: {frame_filename}")
                saved_count += 1

                if max_frames > 0 and saved_count >= max_frames:
                    break

            frame_count += 1

        if saved_count == 0:
            raise RuntimeError("Video contained no readable frames")
        LOGGER.info("Extracted %d frames from video", saved_count)

    except ImportError:
        raise ImportError(
            "OpenCV required for frame extraction. Install with: pip install opencv-python"
        )
    except Exception as e:
        raise RuntimeError(f"Frame extraction failed: {e}") from e
    finally:
        if cap is not None:
            cap.release()


def _detect_competition_from_title(title: str) -> str | None:
    """Detect soccer competition from video title."""
    title_lower = title.lower()

    competitions = {
        "world cup": ["world cup", "copa mundial", "mundial"],
        "champions league": ["champions league", "liga de campeones", "ucl"],
        "premier league": ["premier league", "epl"],
        "laliga": ["laliga", "la liga", "liga española"],
        "serie a": ["serie a", "italian league"],
        "bundesliga": ["bundesliga", "german league"],
        "copa del rey": ["copa del rey", "spanish cup"],
        "fa cup": ["fa cup", "english cup"],
        "europa league": ["europa league", "uel"],
    }

    for competition, keywords in competitions.items():
        if any(keyword in title_lower for keyword in keywords):
            return competition

    return None


def _extract_events_from_analysis(pipeline_summary: PipelineSummary) -> list:
    """Return no events until a trained event detector is integrated."""
    del pipeline_summary
    return []


def _analyze_players_from_tracks(pipeline_summary: PipelineSummary) -> dict:
    """Expose track identities without inventing player statistics."""
    return {
        "status": "tracking_only",
        "track_ids": pipeline_summary.unique_track_ids,
        "limitations": "Player identity, role, touches, passes, and impact are not inferred.",
    }


def _calculate_team_metrics(pipeline_summary: PipelineSummary) -> dict:
    """Return measured tracking totals without fabricated tactical metrics."""
    return {
        "status": "tracking_summary_only",
        "total_detections": pipeline_summary.total_detections,
        "active_track_ids": len(pipeline_summary.unique_track_ids),
        "limitations": "PPDA, field tilt, and possession are not computed.",
    }


def _generate_tactical_report(pipeline_summary: PipelineSummary) -> str:
    """Generate tactical analysis report."""
    report = f"""# Soccer Video Processing Report

## Match Summary
This run processed {pipeline_summary.total_frames} frames with {pipeline_summary.total_detections} detections and {len(pipeline_summary.unique_track_ids)} tracker identities.

## Key Statistics
- **Total Detections**: {pipeline_summary.total_detections}
- **Unique Tracker IDs**: {len(pipeline_summary.unique_track_ids)}
- **Graph Nodes**: {pipeline_summary.graph_nodes}
- **Graph Edges**: {pipeline_summary.graph_edges}
- **Detection Rate**: {pipeline_summary.total_detections / pipeline_summary.total_frames:.1f} per frame

## Limitations
- Tracker IDs are not verified player identities and can fragment or switch.
- Event statistics, possession, field tilt, PPDA, and player impact are not computed.
- Graph counts describe generated spatial-temporal structure; they are not tactical conclusions.

---
*Generated by FIFA Soccer DS Analytics Pipeline*
"""

    return report


def main() -> None:
    """CLI entrypoint for the full pipeline."""
    try:
        parser = argparse.ArgumentParser(
            description="Run full detection + tracking + graph pipeline or YouTube analysis",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        # Input mode selection
        input_group = parser.add_mutually_exclusive_group(required=True)
        input_group.add_argument(
            "--frames-dir",
            type=str,
            help="Directory containing input frames",
        )
        input_group.add_argument(
            "--youtube-url",
            type=str,
            help="YouTube video URL for analysis",
        )

        parser.add_argument(
            "--output-dir",
            type=str,
            default="outputs/analysis",
            help="Output directory for results",
        )

        # YouTube-specific options
        parser.add_argument(
            "--sample-duration",
            type=float,
            default=60.0,
            help="Duration of audio sample for YouTube classification (seconds)",
        )
        parser.add_argument(
            "--force-full-analysis",
            action="store_true",
            help="Force full soccer analysis even if confidence is low",
        )
        parser.add_argument(
            "--include-audio",
            action="store_true",
            help="Enable optional Whisper/librosa audio classification",
        )
        parser.add_argument(
            "--max-video-duration",
            type=int,
            default=1800,
            help="Reject YouTube videos longer than this many seconds",
        )

        # Model options
        parser.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLO weights file")
        parser.add_argument(
            "--confidence", type=float, default=0.4, help="Detection confidence threshold"
        )
        parser.add_argument(
            "--min-confidence", type=float, default=0.25, help="Minimum confidence for tracking"
        )
        parser.add_argument(
            "--distance-threshold", type=float, default=80.0, help="Distance threshold for tracking"
        )
        parser.add_argument("--max-age", type=int, default=15, help="Max age for lost tracks")
        parser.add_argument(
            "--max-frames", type=int, default=30, help="Maximum frames to process (0=all)"
        )
        parser.add_argument(
            "--graph-window", type=int, default=30, help="Window size for graph construction"
        )
        parser.add_argument(
            "--graph-distance-threshold",
            type=float,
            default=80.0,
            help="Distance threshold for spatial edges",
        )
        parser.add_argument(
            "--max-bbox-area-ratio",
            type=float,
            default=0.35,
            help="Drop bboxes whose area / frame area exceeds this (0.35 = 35%% of frame)",
        )
        parser.add_argument(
            "--nms-iou",
            type=float,
            default=0.6,
            help="Per-class NMS IoU threshold (1.0 disables)",
        )
        parser.add_argument(
            "--gnn-weights",
            type=str,
            default="",
            help="Path to PositionClassifier checkpoint (.pt); empty = skip GNN inference",
        )
        parser.add_argument(
            "--enable-tactical-analytics",
            action="store_true",
            help="Enable calibrated pitch-control analytics",
        )
        parser.add_argument(
            "--calibration-path",
            default="",
            help="Required homography calibration JSON when tactical analytics is enabled",
        )
        parser.add_argument(
            "--log-level",
            type=str,
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Logging level",
        )

        args = parser.parse_args()

        _configure_logging(args.log_level)
        _register_signal_handlers()
        logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

        # Validate arguments
        if not Path(args.weights).exists() and not args.weights.startswith("yolov8"):
            LOGGER.error("Model weights file not found: %s", args.weights)
            sys.exit(1)

        config = PipelineConfig(
            weights=args.weights,
            confidence=args.confidence,
            min_confidence=args.min_confidence,
            distance_threshold=args.distance_threshold,
            max_age=args.max_age,
            max_frames=args.max_frames,
            graph_window=args.graph_window,
            graph_distance_threshold=args.graph_distance_threshold,
            gnn_weights=args.gnn_weights,
            max_bbox_area_ratio=args.max_bbox_area_ratio,
            nms_iou=args.nms_iou,
            enable_tactical_analytics=args.enable_tactical_analytics,
            calibration_path=args.calibration_path,
        )

        # Process based on input mode
        if args.youtube_url:
            # YouTube processing mode
            LOGGER.info("Processing YouTube video: %s", args.youtube_url)
            result = process_youtube_video(
                youtube_url=args.youtube_url,
                output_dir=Path(args.output_dir),
                config=config,
                sample_duration=args.sample_duration,
                force_full_analysis=args.force_full_analysis,
                include_audio=args.include_audio,
                max_duration=args.max_video_duration,
            )

            # Print YouTube analysis summary
            print("\n" + "=" * 60)
            print("YOUTUBE ANALYSIS SUMMARY")
            print("=" * 60)
            print(f"Video URL: {args.youtube_url}")
            print(f"Content Type: {result['type']}")
            print(f"Classification: {result['classification']['classification']}")
            print(f"Confidence: {result['classification']['confidence']:.2f}")
            if result["type"] == "soccer":
                print(f"Total Detections: {result['pipeline_summary']['total_detections']}")
                print(f"Unique Players: {len(result['pipeline_summary']['unique_track_ids'])}")
                print(f"Frames Processed: {result['pipeline_summary']['total_frames']}")
            print(f"Output: {args.output_dir}")
            print("=" * 60)

        else:
            # Frame directory processing mode (existing functionality)
            LOGGER.info("Processing frames directory: %s", args.frames_dir)
            summary = process_frames_directory(
                frames_dir=Path(args.frames_dir),
                output_dir=Path(args.output_dir),
                config=config,
            )

            # Print summary for frame processing mode
            try:
                print("\n" + "=" * 60)
                print("PIPELINE SUMMARY")
                print("=" * 60)
                print(f"Frames processed: {summary.total_frames}")
                print(f"Total detections: {summary.total_detections}")
                print(f"Unique tracks: {len(summary.unique_track_ids)}")
                print(f"Track IDs: {summary.unique_track_ids}")
                print(f"Graph nodes: {summary.graph_nodes}")
                print(f"Graph edges: {summary.graph_edges}")
                print(f"Output: {summary.output_dir}")
                print("=" * 60)
            except Exception as e:
                LOGGER.error("Failed to print summary: %s", e)

    except KeyboardInterrupt:
        LOGGER.info("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        LOGGER.error("Pipeline failed with error: %s", e)
        LOGGER.debug("Full traceback:\n%s", traceback.format_exc())
        sys.exit(1)
    finally:
        # Ensure cleanup on pipeline completion or failure
        LOGGER.info("Pipeline cleanup completed")


if __name__ == "__main__":
    main()
