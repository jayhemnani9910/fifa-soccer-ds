"""Regression tests for API-managed output path boundaries."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas import YouTubeAnalysisRequest, validate_youtube_url
from src.utils.output_paths import create_analysis_output_dir, remove_analysis_output_dir


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ",
    ],
)
def test_youtube_url_validator_accepts_supported_video_urls(url: str) -> None:
    assert validate_youtube_url(url)
    assert str(YouTubeAnalysisRequest(url=url).url).startswith("https://")


@pytest.mark.parametrize(
    "url",
    [
        "http://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com.evil.example/watch?v=dQw4w9WgXcQ",
        "https://127.0.0.1/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/playlist?list=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=too-short",
    ],
)
def test_youtube_url_validator_rejects_non_video_or_untrusted_urls(url: str) -> None:
    assert not validate_youtube_url(url)
    with pytest.raises(ValidationError):
        YouTubeAnalysisRequest(url=url)


@pytest.mark.parametrize("name", ["..", ".", "../data", "/tmp/data", "nested/data"])
def test_analysis_request_rejects_path_like_output_names(name: str) -> None:
    with pytest.raises(ValidationError):
        YouTubeAnalysisRequest(
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            output_dir=name,
        )


def test_managed_output_directory_round_trip(tmp_path) -> None:
    target = create_analysis_output_dir("review", "task_123", root=tmp_path)
    (target / "result.json").write_text("{}", encoding="utf-8")

    assert target.parent == tmp_path.resolve()
    remove_analysis_output_dir(target, root=tmp_path)
    assert not target.exists()


def test_output_removal_refuses_unmanaged_directory(tmp_path) -> None:
    unmanaged = tmp_path / "user-data"
    unmanaged.mkdir()

    with pytest.raises(ValueError, match="unmanaged"):
        remove_analysis_output_dir(unmanaged, root=tmp_path)

    assert unmanaged.is_dir()
