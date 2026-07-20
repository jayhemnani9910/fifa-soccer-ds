"""Utilities for fetching and preparing La Liga datasets from Kaggle.

This module provides a thin abstraction around the Kaggle API to cache match
metadata locally, extract video frames for downstream pipelines, and generate
pseudo-labels via the existing YOLOv8 detector. It also exposes a helper to
register dataset snapshots with DVC and MLflow so that retraining jobs remain
reproducible.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import logging
import os
import shutil
import stat

# Subprocesses use fixed argument vectors and shutil-resolved executables.
import subprocess  # nosec B404
import tempfile
import threading
import zipfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.detect.infer import InferenceConfig, extract_detections, first_prediction, load_model
from src.utils.mlflow_helper import ensure_local_backend, start_run

try:
    mlflow: Any | None = importlib.import_module("mlflow")
except ImportError:
    mlflow = None

LOGGER = logging.getLogger(__name__)


class ParallelFrameExtractor:
    """Optimized parallel frame extraction for video processing."""

    def __init__(self, max_workers: int = 0, chunk_size: int = 10):
        self.max_workers = max_workers or min(8, os.cpu_count() or 4)
        self.chunk_size = chunk_size
        if self.max_workers < 1 or self.chunk_size < 1:
            raise ValueError("max_workers and chunk_size must be positive")

    def extract_frames_parallel(
        self, video_paths: list[Path], output_root: Path, every_n_frames: int = 1
    ) -> dict[Path, list[Path]]:
        """Extract frames from multiple videos in parallel."""
        if every_n_frames < 1:
            raise ValueError("every_n_frames must be positive")
        output_root.mkdir(parents=True, exist_ok=True)

        video_chunks = [
            video_paths[i : i + self.chunk_size]
            for i in range(0, len(video_paths), self.chunk_size)
        ]

        results: dict[Path, list[Path]] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit chunk jobs
            future_to_chunk = {
                executor.submit(
                    self._process_video_chunk, chunk, output_root, every_n_frames
                ): chunk
                for chunk in video_chunks
            }

            # Collect results
            for future in as_completed(future_to_chunk):
                chunk_results = future.result()
                results.update(chunk_results)

        return results

    def _process_video_chunk(
        self, video_chunk: list[Path], output_root: Path, every_n_frames: int
    ) -> dict[Path, list[Path]]:
        """Process a chunk of videos sequentially."""
        chunk_results: dict[Path, list[Path]] = {}

        for video_path in video_chunk:
            suffix = hashlib.sha256(str(video_path.resolve()).encode()).hexdigest()[:8]
            output_dir = output_root / f"{video_path.stem}-{suffix}"
            output_dir.mkdir(parents=True, exist_ok=True)

            if av is not None:
                frames = _extract_with_pyav(video_path, output_dir, every_n_frames)
            else:
                frames = _extract_with_opencv(video_path, output_dir, every_n_frames)

            chunk_results[video_path] = frames
            LOGGER.debug("Extracted %d frames from %s", len(frames), video_path)

        return chunk_results


_THREAD_LOCAL_MODEL = threading.local()


def _get_thread_local_model(config: InferenceConfig):
    """Return a YOLO model instance scoped to the current worker thread."""

    cache = getattr(_THREAD_LOCAL_MODEL, "cache", None)
    key = (config.weights, config.device, config.confidence)
    if not cache or cache.get("key") != key:
        cache = {"key": key, "model": load_model(config)}
        _THREAD_LOCAL_MODEL.cache = cache
    return cache["model"]


class ParallelPseudoLabeler:
    """Optimized parallel pseudo-labeling for frame processing."""

    def __init__(self, max_workers: int = 0, batch_size: int = 32):
        self.max_workers = max_workers or 1
        self.batch_size = batch_size
        if self.max_workers < 1 or self.batch_size < 1:
            raise ValueError("max_workers and batch_size must be positive")

    def label_frames_parallel(
        self,
        frame_paths: list[Path],
        output_dir: Path,
        confidence_threshold: float = 0.5,
        weights: str = "yolov8n.pt",
        device: str = "cuda_if_available",
    ) -> Path:
        """Label frames in parallel batches."""
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be within [0, 1]")
        if device.startswith("cuda") and self.max_workers > 1:
            raise ValueError(
                "GPU pseudo-labeling uses one worker to avoid duplicate model allocation"
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        model_config = InferenceConfig(
            weights=weights,
            device=device,
            confidence=confidence_threshold,
        )

        # Process frames in batches
        frame_batches = [
            frame_paths[i : i + self.batch_size]
            for i in range(0, len(frame_paths), self.batch_size)
        ]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(
                    self._process_frame_batch,
                    batch,
                    output_dir,
                    confidence_threshold,
                    model_config,
                ): batch
                for batch in frame_batches
            }

            # Wait for all batches to complete
            for future in as_completed(future_to_batch):
                future.result()

        return output_dir

    def _process_frame_batch(
        self,
        frame_batch: list[Path],
        output_dir: Path,
        confidence_threshold: float,
        model_config: InferenceConfig,
    ) -> None:
        """Process a batch of frames."""

        detector = _get_thread_local_model(model_config)
        for frame_path in frame_batch:
            prediction = first_prediction(
                detector.predict(
                    frame_path.as_posix(),
                    conf=confidence_threshold,
                    verbose=False,
                )
            )
            detections = extract_detections(prediction) if prediction is not None else []
            payload = {
                "frame": frame_path.name,
                "path": frame_path.as_posix(),
                "detections": detections,
                "confidence_threshold": confidence_threshold,
            }
            output_file = output_dir / f"{frame_path.stem}.json"
            output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


KaggleApi: Any | None = None


def _resolve_kaggle_api() -> Any | None:
    """Import the Kaggle API class on demand, never at module load time.

    kaggle's own package __init__ calls ``api.authenticate()`` on import,
    which does ``exit(1)`` (raising SystemExit, not ImportError) when no
    credentials are configured. Resolving lazily keeps that from killing the
    process just for importing this module.
    """

    global KaggleApi
    if KaggleApi is not None:
        return KaggleApi
    try:
        KaggleApi = importlib.import_module("kaggle.api.kaggle_api_extended").KaggleApi
    except (ImportError, SystemExit):
        KaggleApi = None
    return KaggleApi


try:  # pragma: no cover - optional dependency
    av: Any | None = importlib.import_module("av")
except ImportError:  # pragma: no cover - fallback to OpenCV/ffmpeg
    av = None

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_RAW_DIR = DEFAULT_DATA_ROOT / "raw"
DEFAULT_PSEUDO_LABEL_DIR = DEFAULT_DATA_ROOT / "pseudo_labels"
DEFAULT_EXPERIMENT_NAME = "la-liga-data"

FRAME_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


class MatchMetadata(BaseModel):
    """Strongly-typed view over match metadata CSV rows."""

    match_id: str = Field(..., description="Unique identifier for the match clip.")
    home_team: str = Field(..., description="Home club name.")
    away_team: str = Field(..., description="Away club name.")
    season: str | None = Field(None, description="Season shortcode (e.g. 2023-24).")
    competition: str | None = Field(None, description="Competition label (La Liga, Copa, etc.).")
    video_path: Path = Field(..., description="Absolute or cache-relative path to the video file.")
    annotations_csv: Path | None = Field(
        None, description="Optional path to supplementary annotations or event logs."
    )

    @field_validator("video_path", mode="before")
    @classmethod
    def _video_as_path(cls, value: str | Path | None) -> Path:
        if value is None or str(value) in {"", "null"}:
            raise ValueError("video_path is required")
        return Path(value)

    @field_validator("annotations_csv", mode="before")
    @classmethod
    def _optional_as_path(cls, value: str | Path | None) -> Path | None:
        if value is None or str(value) in {"", "null"}:
            return None
        return Path(value)

    def resolve_relative_paths(self, root: Path) -> MatchMetadata:
        """Return a copy with any relative asset paths resolved against the cache root."""

        data = self.model_dump()
        root = root.resolve()
        for key in ("video_path", "annotations_csv"):
            path_value = data.get(key)
            if path_value is None:
                continue
            path = Path(path_value)
            resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f"{key} must stay within dataset cache: {path}")
            data[key] = resolved
        return MatchMetadata.model_validate(data)

    model_config = ConfigDict(extra="ignore")


class KaggleAuthenticationError(RuntimeError):
    """Raised when the Kaggle API is unavailable or authentication fails."""


class DVCRegistrationError(RuntimeError):
    """Raised when a requested dataset snapshot cannot be registered with DVC."""


@dataclass(slots=True)
class KaggleDataLoader:
    """Helper to download and cache La Liga datasets sourced from Kaggle."""

    dataset: str
    cache_dir: Path = DEFAULT_RAW_DIR / "la_liga"
    metadata_filename: str | None = None
    max_extract_bytes: int = 20 * 1024**3
    _api: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.cache_dir = self.cache_dir.resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.max_extract_bytes < 1:
            raise ValueError("max_extract_bytes must be positive")

    # Public API -----------------------------------------------------------------
    def download(self, files: Sequence[str] | None = None, force: bool = False) -> list[Path]:
        """Download the configured Kaggle dataset (or a subset of files) to the cache directory."""

        api = self._ensure_api()
        LOGGER.info("Downloading Kaggle dataset %s into %s", self.dataset, self.cache_dir)

        if files:
            downloaded: list[Path] = []
            for file_name in files:
                target = self._cache_path(file_name)
                if target.exists() and not force:
                    LOGGER.debug("Skipping cached file %s", target)
                    downloaded.append(target)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                api.dataset_download_file(
                    self.dataset,
                    file_name=file_name,
                    path=str(target.parent),
                    force=force,
                    quiet=True,
                )
                if target.suffix == ".zip":
                    unpacked = self._maybe_unzip(target)
                    downloaded.extend(unpacked)
                else:
                    downloaded.append(target)
            return downloaded

        api.dataset_download_files(
            self.dataset,
            path=str(self.cache_dir),
            unzip=False,
            force=force,
            quiet=True,
        )
        for archive in self.cache_dir.glob("*.zip"):
            target_dir = archive.with_suffix("")
            if not target_dir.exists():
                self._maybe_unzip(archive)
        return list(self.cache_dir.glob("**/*"))

    def list_cached_files(self) -> list[Path]:
        """Return all cached files beneath the loader cache directory."""

        return [path for path in self.cache_dir.rglob("*") if path.is_file()]

    def load_metadata(self, csv_path: Path | None = None) -> list[MatchMetadata]:
        """Parse match metadata from the provided CSV file (defaults to loader metadata filename)."""

        if csv_path is not None:
            csv_location = csv_path
        elif self.metadata_filename:
            csv_location = self._cache_path(self.metadata_filename)
        else:
            csv_location = None

        if csv_location is None:
            raise FileNotFoundError(
                "No metadata CSV configured; provide csv_path or metadata_filename."
            )

        if not csv_location.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {csv_location}")

        LOGGER.info("Loading match metadata from %s", csv_location)
        with csv_location.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            records = [MatchMetadata.model_validate(row) for row in reader]

        resolved = [record.resolve_relative_paths(self.cache_dir) for record in records]
        return resolved

    # Internal helpers ------------------------------------------------------------
    def _ensure_api(self) -> Any:
        api_cls = _resolve_kaggle_api()
        if api_cls is None:
            raise KaggleAuthenticationError(
                "kaggle package is not installed. Install it or disable Kaggle downloads."
            )
        if self._api is None:
            api = api_cls()
            try:
                api.authenticate()
            except Exception as exc:  # pragma: no cover - depends on Kaggle CLI config
                raise KaggleAuthenticationError("Unable to authenticate with Kaggle API.") from exc
            self._api = api
        return self._api

    def _maybe_unzip(self, archive_path: Path) -> list[Path]:
        target_dir = archive_path.with_suffix("")
        if target_dir.exists():
            raise FileExistsError(f"Refusing to replace existing dataset directory: {target_dir}")
        if not zipfile.is_zipfile(archive_path):
            raise ValueError(f"Unsupported or invalid dataset archive: {archive_path}")

        temporary_dir = Path(tempfile.mkdtemp(prefix="kaggle-extract-", dir=target_dir.parent))
        try:
            with zipfile.ZipFile(archive_path) as archive:
                members = archive.infolist()
                if len(members) > 100_000:
                    raise ValueError("Dataset archive contains too many entries")
                total_size = sum(member.file_size for member in members)
                if total_size > self.max_extract_bytes:
                    raise ValueError(
                        f"Dataset archive expands to {total_size} bytes, exceeding "
                        f"the {self.max_extract_bytes}-byte limit"
                    )
                for member in members:
                    destination = (temporary_dir / member.filename).resolve()
                    if not destination.is_relative_to(temporary_dir.resolve()):
                        raise ValueError(f"Unsafe path in dataset archive: {member.filename}")
                    file_type = (member.external_attr >> 16) & 0o170000
                    if stat.S_ISLNK(file_type):
                        raise ValueError(
                            f"Symlink in dataset archive is not allowed: {member.filename}"
                        )
                archive.extractall(temporary_dir)
            temporary_dir.replace(target_dir)
        except Exception:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise
        return [path for path in target_dir.rglob("*") if path.is_file()]

    def _cache_path(self, relative_name: str) -> Path:
        candidate = (self.cache_dir / relative_name).resolve()
        if Path(relative_name).is_absolute() or not candidate.is_relative_to(self.cache_dir):
            raise ValueError(f"Dataset file must stay within cache_dir: {relative_name}")
        return candidate


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path | None = None,
    every_n_frames: int = 1,
) -> list[Path]:
    """Extract frames from a video onto disk using PyAV, falling back to OpenCV."""

    if every_n_frames <= 0:
        raise ValueError("every_n_frames must be >= 1")

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_root = (output_dir or DEFAULT_RAW_DIR / video_path.stem).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if av is not None:
        return _extract_with_pyav(video_path, output_root, every_n_frames)

    return _extract_with_opencv(video_path, output_root, every_n_frames)


def _extract_with_pyav(video_path: Path, output_dir: Path, every_n_frames: int) -> list[Path]:
    if av is None:  # pragma: no cover - guarded by caller
        raise RuntimeError("PyAV is unavailable")
    container = av.open(video_path.as_posix())
    try:
        if not container.streams.video:
            raise RuntimeError(f"Video contains no video stream: {video_path}")
        stream = container.streams.video[0]
        extracted: list[Path] = []
        for index, frame in enumerate(container.decode(stream)):
            if index % every_n_frames != 0:
                continue
            image = frame.to_image()
            frame_path = output_dir / f"{video_path.stem}_frame_{index:06d}.jpg"
            image.save(frame_path)
            extracted.append(frame_path)
        return extracted
    finally:
        container.close()


def _extract_with_opencv(video_path: Path, output_dir: Path, every_n_frames: int) -> list[Path]:
    capture = cv2.VideoCapture(video_path.as_posix())
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open video with OpenCV: {video_path}")

    extracted: list[Path] = []
    index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if index % every_n_frames == 0:
                frame_path = output_dir / f"{video_path.stem}_frame_{index:06d}.jpg"
                wrote = cv2.imwrite(frame_path.as_posix(), frame)
                if not wrote:
                    raise RuntimeError(f"Failed to write frame {frame_path}")
                extracted.append(frame_path)
            index += 1
        return extracted
    finally:
        capture.release()


def pseudo_label_frames(
    frames_dir: Path,
    output_dir: Path | None = None,
    confidence_threshold: float = 0.5,
    weights: str = "yolov8n.pt",
    device: str = "cuda_if_available",
    model=None,
) -> Path:
    """Auto-label a directory of frames using the pre-existing YOLOv8 detector."""

    if confidence_threshold < 0 or confidence_threshold > 1:
        raise ValueError("confidence_threshold must be within [0, 1].")

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frame directory not found: {frames_dir}")

    frame_paths = sorted(
        [
            path
            for path in frames_dir.iterdir()
            if path.is_file() and path.suffix.lower() in FRAME_EXTENSIONS
        ]
    )
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    detector = model
    if detector is None:
        config = InferenceConfig(weights=weights, device=device, confidence=confidence_threshold)
        detector = load_model(config)

    output_root = (output_dir or DEFAULT_PSEUDO_LABEL_DIR / frames_dir.stem).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    labels: list[dict[str, object]] = []
    for frame_path in frame_paths:
        prediction = first_prediction(
            detector.predict(
                frame_path.as_posix(),
                conf=confidence_threshold,
                verbose=False,
            )
        )
        if prediction is None:
            labels.append({"frame": frame_path.name, "detections": []})
            continue

        detections = [
            detection
            for detection in extract_detections(prediction)
            if detection.get("confidence", 0) and detection["confidence"] >= confidence_threshold
        ]
        labels.append(
            {
                "frame": frame_path.name,
                "path": frame_path.as_posix(),
                "detections": detections,
            }
        )

    output_path = output_root / "pseudo_labels.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"frames": labels, "confidence_threshold": confidence_threshold}, handle, indent=2
        )
    return output_path


def version_data_with_dvc(
    data_dir: Path,
    experiment: str = DEFAULT_EXPERIMENT_NAME,
    run_name: str | None = None,
    param_key: str = "dataset_path",
    hash_key: str = "dataset_hash",
) -> dict[str, str]:
    """Register the dataset snapshot with DVC and log its hash to MLflow."""

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    _run_dvc_add(data_dir)
    dataset_hash = _read_dvc_hash(data_dir) or _hash_directory(data_dir)

    ensure_local_backend()

    if mlflow is None:
        LOGGER.info("MLflow missing; recorded dataset hash %s without logging.", dataset_hash)
        return {"dataset_hash": dataset_hash}

    active_run = mlflow.active_run()
    if active_run is None:
        run_ctx: AbstractContextManager[Any] = start_run(experiment=experiment, run_name=run_name)
    else:
        run_ctx = nullcontext(active_run)

    with run_ctx:
        try:
            mlflow.log_param(param_key, str(data_dir.resolve()))
            mlflow.log_param(hash_key, dataset_hash)
        except Exception as exc:  # pragma: no cover - network/fs issues
            LOGGER.warning("Failed to log dataset metadata to MLflow: %s", exc)

    return {"dataset_hash": dataset_hash}


def _run_dvc_add(data_dir: Path) -> None:
    dvc_executable = shutil.which("dvc")
    if dvc_executable is None:
        raise DVCRegistrationError(
            "DVC CLI not found; install a security-reviewed DVC release in an isolated "
            "operator environment before versioning datasets."
        )
    try:
        # The executable is resolved and every argument is controlled here.
        result = subprocess.run(  # nosec B603
            [dvc_executable, "add", str(data_dir)],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise DVCRegistrationError(
                f"dvc add exited with code {result.returncode}: {result.stderr.strip()}"
            )
    except FileNotFoundError as exc:  # pragma: no cover - executable removed after resolution
        raise DVCRegistrationError("DVC CLI disappeared before it could run.") from exc
    except subprocess.TimeoutExpired as exc:
        raise DVCRegistrationError("dvc add timed out after 300 seconds") from exc


def _hash_directory(directory: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest.update(file_path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(str(file_path.stat().st_size).encode("utf-8"))
        with file_path.open("rb") as handle:
            while chunk := handle.read(8192):
                digest.update(chunk)
    return digest.hexdigest()


def _read_dvc_hash(target_dir: Path) -> str | None:
    """Read the last recorded DVC md5 hash if the .dvc file exists."""

    dvc_file = Path(f"{target_dir}.dvc")
    if not dvc_file.exists():
        return None

    try:
        for line in dvc_file.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("md5:"):
                return stripped.split("md5:", 1)[1].strip()
    except OSError as exc:
        LOGGER.debug("Unable to read DVC metadata from %s: %s", dvc_file, exc)
    return None


__all__ = [
    "DVCRegistrationError",
    "KaggleAuthenticationError",
    "KaggleDataLoader",
    "MatchMetadata",
    "extract_frames_from_video",
    "pseudo_label_frames",
    "version_data_with_dvc",
]
