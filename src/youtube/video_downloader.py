"""YouTube video downloader using yt-dlp.

This module provides functionality to download YouTube videos
for soccer content analysis.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

LOGGER = logging.getLogger(__name__)


class YouTubeDownloader:
    """YouTube video downloader with soccer-optimized settings."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the YouTube downloader.
        
        Args:
            cache_dir: Directory for caching downloaded videos
        """
        if yt_dlp is None:
            raise ImportError("yt-dlp is required. Install with: pip install yt-dlp")
            
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "youtube_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # YouTube download options optimized for soccer analysis
        # Note: SSL verification is enabled for security (no_check_certificate removed)
        # If you need to disable it for specific environments, set env var YT_DLP_SKIP_SSL=1
        skip_ssl = os.getenv("YT_DLP_SKIP_SSL", "0").lower() in ("1", "true", "yes")

        self.ydl_opts = {
            'format': 'best[height<=720]',  # Limit resolution for processing speed
            'outtmpl': str(self.cache_dir / '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'extractaudio': False,  # We'll extract audio separately
            'audioformat': 'mp3',
            'audioquality': '192',
            'retries': 3,
            'fragment_retries': 3,
            'skip_unavailable_fragments': True,
            'ignoreerrors': True,
            # SSL certificate verification enabled by default for security
            # Only disable if explicitly requested via environment variable
            'nocheckcertificate': skip_ssl,
        }
    
    def download_video(self, youtube_url: str) -> Dict[str, Any]:
        """Download a YouTube video.
        
        Args:
            youtube_url: YouTube video URL
            
        Returns:
            Dict containing download info and file paths
            
        Raises:
            ValueError: If URL is invalid
            RuntimeError: If download fails
        """
        if not self._is_valid_youtube_url(youtube_url):
            raise ValueError(f"Invalid YouTube URL: {youtube_url}")
        
        video_id = self._extract_video_id(youtube_url)
        cached_path = self.cache_dir / f"{video_id}.mp4"
        
        # Check if already cached
        if cached_path.exists():
            LOGGER.info("Using cached video: %s", cached_path)
            return self._get_video_info(cached_path, video_id, youtube_url)
        
        LOGGER.info("Downloading video: %s", youtube_url)
        
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                # Extract info first
                info = ydl.extract_info(youtube_url, download=False)
                
                if not info:
                    raise RuntimeError("Failed to extract video information")
                
                # Download the video
                ydl.download([youtube_url])
                
                # Find the downloaded file
                expected_path = self.cache_dir / f"{video_id}.{info.get('ext', 'mp4')}"
                
                if not expected_path.exists():
                    # Try alternative extensions
                    for ext in ['mp4', 'mkv', 'webm', 'avi']:
                        alt_path = self.cache_dir / f"{video_id}.{ext}"
                        if alt_path.exists():
                            expected_path = alt_path
                            break
                    else:
                        raise RuntimeError(f"Downloaded file not found for video {video_id}")
                
                result = self._get_video_info(expected_path, video_id, youtube_url)
                LOGGER.info("Successfully downloaded video: %s", expected_path)
                return result
                
        except Exception as e:
            LOGGER.error("Failed to download video %s: %s", youtube_url, e)
            raise RuntimeError(f"Video download failed: {e}") from e
    
    def get_video_info(self, youtube_url: str) -> Dict[str, Any]:
        """Get video information without downloading.
        
        Args:
            youtube_url: YouTube video URL
            
        Returns:
            Dict containing video metadata
        """
        if not self._is_valid_youtube_url(youtube_url):
            raise ValueError(f"Invalid YouTube URL: {youtube_url}")
        
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                
                if not info:
                    raise RuntimeError("Failed to extract video information")
                
                return {
                    'id': info.get('id'),
                    'title': info.get('title'),
                    'description': info.get('description'),
                    'duration': info.get('duration'),
                    'view_count': info.get('view_count'),
                    'uploader': info.get('uploader'),
                    'upload_date': info.get('upload_date'),
                    'width': info.get('width'),
                    'height': info.get('height'),
                    'fps': info.get('fps'),
                    'format': info.get('format'),
                    'filesize': info.get('filesize'),
                    'url': youtube_url,
                }
                
        except Exception as e:
            LOGGER.error("Failed to get video info for %s: %s", youtube_url, e)
            raise RuntimeError(f"Failed to get video info: {e}") from e
    
    def extract_thumbnail(self, youtube_url: str, output_path: Optional[Path] = None) -> Path:
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
            with yt_dlp.YoutubeDL({'quiet': True, 'skip_download': True}) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                
                if not info or 'thumbnail' not in info:
                    raise RuntimeError("No thumbnail available for this video")
                
                thumbnail_url = info['thumbnail']
                
                # Download thumbnail
                import requests
                response = requests.get(thumbnail_url, timeout=30)
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
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
            LOGGER.info("Cache cleanup complete: %d files, %.1f MB", 
                       cleaned_files, cleaned_size / (1024 * 1024))
        else:
            LOGGER.info("No cache files to clean up")
    
    def _is_valid_youtube_url(self, url: str) -> bool:
        """Check if URL is a valid YouTube URL."""
        import re
        
        youtube_patterns = [
            r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/',
            r'(https?://)?(www\.)?youtube\.com/watch\?v=',
            r'(https?://)?youtu\.be/',
        ]
        
        return any(re.match(pattern, url) for pattern in youtube_patterns)
    
    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL."""
        import re
        
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
            r'youtube\.com/v/([^&\n?#]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise ValueError(f"Could not extract video ID from URL: {url}")
    
    def _get_video_info(self, video_path: Path, video_id: str, url: str) -> Dict[str, Any]:
        """Get information about downloaded video."""
        stat = video_path.stat()
        
        return {
            'video_id': video_id,
            'video_path': str(video_path),
            'file_size': stat.st_size,
            'duration': None,  # Would need ffprobe to get this
            'url': url,
            'cached': True,
            'download_time': None,  # Could track this
        }


# Convenience function
def download_youtube_video(youtube_url: str, cache_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Download a YouTube video and return its information.
    
    Args:
        youtube_url: YouTube video URL
        cache_dir: Optional cache directory
        
    Returns:
        Dict containing download information
    """
    downloader = YouTubeDownloader(cache_dir)
    return downloader.download_video(youtube_url)


def get_youtube_info(youtube_url: str) -> Dict[str, Any]:
    """Get YouTube video information without downloading.
    
    Args:
        youtube_url: YouTube video URL
        
    Returns:
        Dict containing video metadata
    """
    downloader = YouTubeDownloader()
    return downloader.get_video_info(youtube_url)


def extract_youtube_thumbnail(youtube_url: str, output_path: Optional[Path] = None) -> Path:
    """Extract YouTube video thumbnail.
    
    Args:
        youtube_url: YouTube video URL
        output_path: Where to save the thumbnail
        
    Returns:
        Path to the extracted thumbnail
    """
    downloader = YouTubeDownloader()
    return downloader.extract_thumbnail(youtube_url, output_path)