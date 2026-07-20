"""YouTube video downloader using yt-dlp.

This module provides functionality to download YouTube videos
for soccer content analysis.
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import cv2
import requests

from src.schemas import extract_youtube_video_id, validate_youtube_url

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

LOGGER = logging.getLogger(__name__)


class YouTubeDownloader:
    """YouTube video downloader with soccer-optimized settings."""

    def __init__(self, cache_dir: Path | None = None):
        """Initialize the YouTube downloader.

        Args:
            cache_dir: Directory for caching downloaded videos
        """
        if yt_dlp is None:
            raise ImportError("yt-dlp is required. Install with: pip install yt-dlp")

        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "youtube_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # YouTube download options optimized for bounded analysis. TLS certificate
        # verification is intentionally not configurable off.
        self.max_download_bytes = int(os.getenv("YOUTUBE_MAX_DOWNLOAD_BYTES", str(1024**3)))
        if self.max_download_bytes <= 0:
            raise ValueError("YOUTUBE_MAX_DOWNLOAD_BYTES must be a positive integer")

        self.ydl_opts = {
            "format": "best[height<=720]",  # Limit resolution for processing speed
            "outtmpl": str(self.cache_dir / "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "extractaudio": False,  # We'll extract audio separately
            "audioformat": "mp3",
            "audioquality": "192",
            "retries": 3,
            "fragment_retries": 3,
            "socket_timeout": 30,
            "max_filesize": self.max_download_bytes,
            "skip_unavailable_fragments": True,
            "ignoreerrors": False,
            "nocheckcertificate": False,
        }

    def download_video(
        self, youtube_url: str, *, max_duration: int | None = None
    ) -> dict[str, Any]:
        """Download a YouTube video.

        Args:
            youtube_url: YouTube video URL
            max_duration: Reject longer videos when duration metadata is available

        Returns:
            Dict containing download info and file paths

        Raises:
            ValueError: If URL is invalid
            RuntimeError: If download fails
        """
        if not self._is_valid_youtube_url(youtube_url):
            raise ValueError(f"Invalid YouTube URL: {youtube_url}")

        video_id = self._extract_video_id(youtube_url)
        cached_path = next(
            (
                path
                for extension in ("mp4", "mkv", "webm", "avi")
                if (path := self.cache_dir / f"{video_id}.{extension}").exists()
            ),
            None,
        )

        # Check if already cached
        if cached_path is not None:
            if cached_path.is_symlink() or not cached_path.is_file():
                raise RuntimeError(f"Unsafe cached video path: {cached_path}")
            self._validate_downloaded_video(cached_path, max_duration=max_duration)
            LOGGER.info("Using cached video: %s", cached_path)
            return self._get_video_info(cached_path, video_id, youtube_url)

        LOGGER.info("Downloading video: %s", youtube_url)

        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                # Extract info first
                info = ydl.extract_info(youtube_url, download=False)

                if not info:
                    raise RuntimeError("Failed to extract video information")
                duration = info.get("duration")
                if max_duration is not None:
                    if duration is None:
                        raise ValueError(
                            "Video duration is unavailable; refusing an unbounded download"
                        )
                    if duration > max_duration:
                        raise ValueError(
                            f"Video duration {duration}s exceeds configured limit {max_duration}s"
                        )

                # Download the video
                ydl.download([youtube_url])

                # Find the downloaded file
                expected_path = self.cache_dir / f"{video_id}.{info.get('ext', 'mp4')}"

                if not expected_path.exists():
                    # Try alternative extensions
                    for ext in ["mp4", "mkv", "webm", "avi"]:
                        alt_path = self.cache_dir / f"{video_id}.{ext}"
                        if alt_path.exists():
                            expected_path = alt_path
                            break
                    else:
                        raise RuntimeError(f"Downloaded file not found for video {video_id}")

                try:
                    self._validate_downloaded_video(expected_path, max_duration=max_duration)
                except Exception:
                    expected_path.unlink(missing_ok=True)
                    raise
                result = self._get_video_info(expected_path, video_id, youtube_url)
                LOGGER.info("Successfully downloaded video: %s", expected_path)
                return result

        except Exception as e:
            LOGGER.error("Failed to download video %s: %s", youtube_url, e)
            raise RuntimeError(f"Video download failed: {e}") from e

    def get_video_info(self, youtube_url: str) -> dict[str, Any]:
        """Get video information without downloading.

        Args:
            youtube_url: YouTube video URL

        Returns:
            Dict containing video metadata
        """
        if not self._is_valid_youtube_url(youtube_url):
            raise ValueError(f"Invalid YouTube URL: {youtube_url}")

        try:
            with yt_dlp.YoutubeDL(
                {"quiet": True, "socket_timeout": 30, "nocheckcertificate": False}
            ) as ydl:
                info = ydl.extract_info(youtube_url, download=False)

                if not info:
                    raise RuntimeError("Failed to extract video information")

                return {
                    "id": info.get("id"),
                    "title": info.get("title"),
                    "description": info.get("description"),
                    "duration": info.get("duration"),
                    "view_count": info.get("view_count"),
                    "uploader": info.get("uploader"),
                    "upload_date": info.get("upload_date"),
                    "width": info.get("width"),
                    "height": info.get("height"),
                    "fps": info.get("fps"),
                    "format": info.get("format"),
                    "filesize": info.get("filesize"),
                    "url": youtube_url,
                }

        except Exception as e:
            LOGGER.error("Failed to get video info for %s: %s", youtube_url, e)
            raise RuntimeError(f"Failed to get video info: {e}") from e

    def extract_thumbnail(self, youtube_url: str, output_path: Path | None = None) -> Path:
        """Extract video thumbnail.

        Args:
            youtube_url: YouTube video URL
            output_path: Where to save the thumbnail

        Returns:
            Path to the extracted thumbnail
        """
        if not self._is_valid_youtube_url(youtube_url):
            raise ValueError(f"Invalid YouTube URL: {youtube_url}")

        if output_path is None:
            video_id = self._extract_video_id(youtube_url)
            output_path = self.cache_dir / f"{video_id}_thumb.jpg"

        try:
            with yt_dlp.YoutubeDL(
                {
                    "quiet": True,
                    "skip_download": True,
                    "socket_timeout": 30,
                    "nocheckcertificate": False,
                }
            ) as ydl:
                info = ydl.extract_info(youtube_url, download=False)

                if not info or "thumbnail" not in info:
                    raise RuntimeError("No thumbnail available for this video")

                thumbnail_url = info["thumbnail"]
                self._validate_thumbnail_url(thumbnail_url)

                output_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = output_path.with_name(
                    f".{output_path.name}.{uuid.uuid4().hex}.part"
                )
                try:
                    with requests.get(
                        thumbnail_url,
                        timeout=(5, 30),
                        stream=True,
                        allow_redirects=False,
                    ) as response:
                        response.raise_for_status()
                        content_type = response.headers.get("Content-Type", "")
                        if not content_type.lower().startswith("image/"):
                            raise RuntimeError("Thumbnail response is not an image")
                        content_length = response.headers.get("Content-Length")
                        if content_length and int(content_length) > 10 * 1024 * 1024:
                            raise RuntimeError("Thumbnail exceeds the 10 MiB limit")

                        total = 0
                        with temporary_path.open("wb") as output_file:
                            for chunk in response.iter_content(chunk_size=64 * 1024):
                                if not chunk:
                                    continue
                                total += len(chunk)
                                if total > 10 * 1024 * 1024:
                                    raise RuntimeError("Thumbnail exceeds the 10 MiB limit")
                                output_file.write(chunk)
                    temporary_path.replace(output_path)
                finally:
                    temporary_path.unlink(missing_ok=True)

                LOGGER.info("Thumbnail extracted: %s", output_path)
                return output_path

        except Exception as e:
            LOGGER.error("Failed to extract thumbnail for %s: %s", youtube_url, e)
            raise RuntimeError(f"Thumbnail extraction failed: {e}") from e

    def cleanup_cache(self, max_age_days: int = 7) -> None:
        """Clean up old cached videos.

        Args:
            max_age_days: Maximum age of files to keep
        """
        import time

        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        cleaned_files = 0
        cleaned_size = 0

        for file_path in self.cache_dir.glob("*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    cleaned_files += 1
                    cleaned_size += file_size
                    LOGGER.info("Cleaned cache file: %s", file_path.name)

        if cleaned_files > 0:
            LOGGER.info(
                "Cache cleanup complete: %d files, %.1f MB",
                cleaned_files,
                cleaned_size / (1024 * 1024),
            )
        else:
            LOGGER.info("No cache files to clean up")

    def _is_valid_youtube_url(self, url: str) -> bool:
        """Check if URL is a valid YouTube URL."""
        return validate_youtube_url(url)

    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL."""
        return extract_youtube_video_id(url)

    def _get_video_info(self, video_path: Path, video_id: str, url: str) -> dict[str, Any]:
        """Get information about downloaded video."""
        stat = video_path.stat()
        duration = self._probe_video_duration(video_path)

        return {
            "video_id": video_id,
            "video_path": str(video_path),
            "file_size": stat.st_size,
            "duration": duration,
            "url": url,
            "cached": True,
            "download_time": None,  # Could track this
        }

    def _validate_downloaded_video(self, video_path: Path, *, max_duration: int | None) -> None:
        file_size = video_path.stat().st_size
        if file_size <= 0:
            raise RuntimeError("Downloaded video is empty")
        if file_size > self.max_download_bytes:
            raise RuntimeError(
                f"Downloaded video exceeds the {self.max_download_bytes}-byte size limit"
            )
        if max_duration is not None:
            duration = self._probe_video_duration(video_path)
            if duration is None:
                raise RuntimeError("Unable to verify cached/downloaded video duration")
            if duration > max_duration:
                raise ValueError(
                    f"Video duration {duration:.1f}s exceeds configured limit {max_duration}s"
                )

    @staticmethod
    def _probe_video_duration(video_path: Path) -> float | None:
        capture = cv2.VideoCapture(video_path.as_posix())
        try:
            if not capture.isOpened():
                return None
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or frame_count <= 0:
                return None
            return frame_count / fps
        finally:
            capture.release()

    @staticmethod
    def _validate_thumbnail_url(url: str) -> None:
        try:
            parsed = urlsplit(url)
            hostname = (parsed.hostname or "").lower().rstrip(".")
            port = parsed.port
        except ValueError as exc:
            raise ValueError("Thumbnail URL is malformed") from exc
        if (
            parsed.scheme != "https"
            or not hostname.endswith((".ytimg.com", ".ggpht.com"))
            or parsed.username is not None
            or parsed.password is not None
            or port not in {None, 443}
        ):
            raise ValueError("Thumbnail URL must use an approved HTTPS YouTube image host")


# Convenience function
def download_youtube_video(youtube_url: str, cache_dir: Path | None = None) -> dict[str, Any]:
    """Download a YouTube video and return its information.

    Args:
        youtube_url: YouTube video URL
        cache_dir: Optional cache directory

    Returns:
        Dict containing download information
    """
    downloader = YouTubeDownloader(cache_dir)
    return downloader.download_video(youtube_url)


def get_youtube_info(youtube_url: str) -> dict[str, Any]:
    """Get YouTube video information without downloading.

    Args:
        youtube_url: YouTube video URL

    Returns:
        Dict containing video metadata
    """
    downloader = YouTubeDownloader()
    return downloader.get_video_info(youtube_url)


def extract_youtube_thumbnail(youtube_url: str, output_path: Path | None = None) -> Path:
    """Extract YouTube video thumbnail.

    Args:
        youtube_url: YouTube video URL
        output_path: Where to save the thumbnail

    Returns:
        Path to the extracted thumbnail
    """
    downloader = YouTubeDownloader()
    return downloader.extract_thumbnail(youtube_url, output_path)
