from __future__ import annotations

import csv
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

import src.data.la_liga_loader as loader_module
from src.data.la_liga_loader import (
    DVCRegistrationError,
    KaggleDataLoader,
    extract_frames_from_video,
)


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


def test_dataset_archive_rejects_path_traversal(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    archive = cache_dir / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escaped.txt", "unsafe")
    loader = KaggleDataLoader(dataset="user/example", cache_dir=cache_dir)

    with pytest.raises(ValueError, match="Unsafe path"):
        loader._maybe_unzip(archive)

    assert not (tmp_path / "escaped.txt").exists()


def test_metadata_paths_cannot_escape_dataset_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    csv_path = tmp_path / "metadata.csv"
    csv_path.write_text(
        "match_id,home_team,away_team,video_path\n1,Home,Away,../outside.mp4\n",
        encoding="utf-8",
    )
    loader = KaggleDataLoader(dataset="user/example", cache_dir=cache_dir)

    with pytest.raises(ValueError, match="must stay within dataset cache"):
        loader.load_metadata(csv_path)


def test_importing_module_does_not_authenticate_with_kaggle(tmp_path: Path) -> None:
    """Importing the loader must not eagerly authenticate with Kaggle.

    kaggle's own package __init__ calls exit(1) when no credentials are
    configured, so resolving the Kaggle API at module scope kills the process
    before _ensure_api's KaggleAuthenticationError handling is ever reached.
    """

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"KAGGLE_USERNAME", "KAGGLE_KEY"}
    }
    env["HOME"] = str(fake_home)
    env["KAGGLE_CONFIG_DIR"] = str(fake_home)

    result = subprocess.run(
        [sys.executable, "-c", "import src.data; print('IMPORT_OK')"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "IMPORT_OK" in result.stdout


def test_dvc_registration_failure_is_not_reported_as_versioned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loader_module.shutil, "which", lambda _name: "/usr/bin/dvc")
    monkeypatch.setattr(
        loader_module.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["dvc", "add"], returncode=1, stdout="", stderr="not a DVC repository"
        ),
    )

    with pytest.raises(DVCRegistrationError, match="not a DVC repository"):
        loader_module._run_dvc_add(tmp_path)
