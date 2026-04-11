"""YouTube integration module for Smart YouTube Analyzer.

This module provides functionality to download, process, and analyze
YouTube videos for soccer content detection and analysis.
"""

from .audio_extractor import AudioExtractor
from .metadata_parser import YouTubeMetadataParser
from .video_downloader import YouTubeDownloader

__all__ = [
    "YouTubeDownloader",
    "AudioExtractor", 
    "YouTubeMetadataParser",
]