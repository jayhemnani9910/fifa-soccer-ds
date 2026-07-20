from __future__ import annotations

from pathlib import Path

import pytest

from src.youtube.video_downloader import YouTubeDownloader


def test_downloader_extracts_all_approved_url_shapes(tmp_path: Path) -> None:
    downloader = YouTubeDownloader(cache_dir=tmp_path)

    for url in (
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ",
    ):
        assert downloader._extract_video_id(url) == "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "url",
    [
        "http://i.ytimg.com/vi/id/default.jpg",
        "https://127.0.0.1/private.jpg",
        "https://i.ytimg.com.evil.example/image.jpg",
        "https://user:password@i.ytimg.com/image.jpg",
    ],
)
def test_thumbnail_download_rejects_unapproved_targets(url: str) -> None:
    with pytest.raises(ValueError, match="approved HTTPS"):
        YouTubeDownloader._validate_thumbnail_url(url)


def test_cached_video_is_rejected_when_size_limit_is_exceeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("YOUTUBE_MAX_DOWNLOAD_BYTES", "4")
    cached = tmp_path / "dQw4w9WgXcQ.mp4"
    cached.write_bytes(b"12345")
    downloader = YouTubeDownloader(cache_dir=tmp_path)

    with pytest.raises(RuntimeError, match="size limit"):
        downloader.download_video("https://youtu.be/dQw4w9WgXcQ")

    assert cached.exists()
