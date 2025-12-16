from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.data.la_liga_loader import KaggleDataLoader, extract_frames_from_video


def _write_sample_video(video_path: Path, total_frames: int = 5) -> None:
    cv2 = pytest.importorskip("cv2")

    width, height = 64, 64
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path.as_posix(), fourcc, 5, (width, height))
    assert writer.isOpened()

    for idx in range(total_frames):
        frame = np.full((height, width, 3), idx * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_load_metadata_resolves_relative_paths(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    clips_dir = cache_dir / "clips"
    clips_dir.mkdir()
    video_a = clips_dir / "match_a.mp4"
    video_b = clips_dir / "match_b.mp4"
    video_a.touch()
    video_b.touch()

    csv_path = tmp_path / "metadata.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "match_id",
                "home_team",
                "away_team",
                "season",
                "competition",
                "video_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "match_id": "2024-01",
                "home_team": "FC Barcelona",
                "away_team": "Real Madrid",
                "season": "2023-24",
                "competition": "La Liga",
                "video_path": "clips/match_a.mp4",
            }
        )
        writer.writerow(
            {
                "match_id": "2024-02",
                "home_team": "Real Sociedad",
                "away_team": "Sevilla",
                "season": "2023-24",
                "competition": "La Liga",
                "video_path": "clips/match_b.mp4",
            }
        )

    loader = KaggleDataLoader(dataset="user/la-liga", cache_dir=cache_dir)
    metadata = loader.load_metadata(csv_path=csv_path)

    assert len(metadata) == 2
    assert metadata[0].video_path == video_a.resolve()
    assert metadata[1].video_path == video_b.resolve()
    for entry in metadata:
        assert entry.match_id
        assert entry.home_team
        assert entry.away_team


def test_extract_frames_counts_expected_frames(tmp_path: Path) -> None:
    video_path = tmp_path / "fixture.mp4"
    frames_dir = tmp_path / "frames"
    _write_sample_video(video_path, total_frames=6)

    frames = extract_frames_from_video(video_path, output_dir=frames_dir, every_n_frames=1)

    assert len(frames) == 6
    for frame_path in frames:
        assert frame_path.exists()
        assert frame_path.parent == frames_dir.resolve()
