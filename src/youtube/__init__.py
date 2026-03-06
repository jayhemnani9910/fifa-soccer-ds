"""YouTube integration module for Smart YouTube Analyzer.

This module provides functionality to download, process, and analyze
YouTube videos for soccer content detection and analysis.
"""

from .video_downloader import YouTubeDownloader
from .audio_extractor import AudioExtractor
from .metadata_parser import YouTubeMetadataParser

__all__ = [
    "YouTubeDownloader",
    "AudioExtractor", 
    "YouTubeMetadataParser",
]