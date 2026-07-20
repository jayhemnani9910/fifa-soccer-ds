"""Data ingestion utilities for the FIFA Soccer DS analytics pipeline."""

from .la_liga_loader import (
    KaggleDataLoader,
    extract_frames_from_video,
    pseudo_label_frames,
    version_data_with_dvc,
)

__all__ = [
    "KaggleDataLoader",
    "extract_frames_from_video",
    "pseudo_label_frames",
    "version_data_with_dvc",
]
