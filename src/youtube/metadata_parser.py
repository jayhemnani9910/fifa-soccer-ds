"""YouTube metadata parser for video information extraction.

This module provides functionality to parse and extract metadata
from YouTube videos for content analysis.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from src.schemas import VideoMetadata, validate_youtube_url, validate_video_metadata

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

LOGGER = logging.getLogger(__name__)


class YouTubeMetadataParser:
    """Parse and analyze YouTube video metadata."""
    
    def __init__(self):
        """Initialize the metadata parser."""
        if yt_dlp is None:
            raise ImportError("yt-dlp is required. Install with: pip install yt-dlp")
        
        # Soccer-related keywords for content analysis
        self.soccer_keywords = [
            # General soccer terms
            'soccer', 'football', 'fútbol', 'futbol',
            # Match events
            'goal', 'gol', 'penalty', 'penalti', 'foul', 'falta',
            'corner', 'esquina', 'free kick', 'tiro libre',
            'offside', 'fuera de juego', 'red card', 'tarjeta roja',
            'yellow card', 'tarjeta amarilla', 'substitution', 'cambio',
            # Positions
            'striker', 'delantero', 'forward', 'atacante',
            'defender', 'defensor', 'midfielder', 'centrocampista',
            'goalkeeper', 'portero', 'goalie', 'arquero',
            # Competitions
            'world cup', 'copa mundial', 'champions league', 'liga de Campeones',
            'premier league', 'laliga', 'serie a', 'bundesliga',
            # Teams
            'barcelona', 'real madrid', 'manchester united', 'manchester city',
            'liverpool', 'arsenal', 'chelsea', 'tottenham',
            'bayern munich', 'borussia dortmund', 'juventus', 'ac milan',
            'inter milan', 'psg', 'marseille', 'lyon',
            # Players
            'messi', 'ronaldo', 'cristiano', 'neymar', 'mbappé',
            'haaland', 'salah', 'mane', 'lewandowski',
            # Commentators/Channels
            'espn', 'fox sports', 'sky sports', 'bt sport',
            'commentary', 'comentarios', 'highlights', 'resumen',
        ]
        
        # Duration patterns (short, medium, long content)
        self.duration_patterns = {
            'short': (0, 300),      # 0-5 minutes
            'medium': (300, 1800),  # 5-30 minutes  
            'long': (1800, 3600),   # 30-60 minutes
            'full_match': (3600, 10800),  # 1-3 hours
        }
    
    def extract_metadata(self, youtube_url: str) -> Dict[str, Any]:
        """Extract comprehensive metadata from YouTube video.
        
        Args:
            youtube_url: YouTube video URL
            
        Returns:
            Dict containing video metadata and analysis
        """
        # Validate input URL
        if not validate_youtube_url(youtube_url):
            raise ValueError(f"Invalid YouTube URL format: {youtube_url}")
        
        try:
            LOGGER.info("Extracting metadata from: %s", youtube_url)
            
            # Extract video info
            info = self._get_video_info(youtube_url)
            
            # Validate and structure metadata using schema
            validated_metadata = self._create_validated_metadata(info)
            
            # Analyze content
            analysis = self._analyze_content(info)
            
            # Validate video metadata comprehensively
            validation_result = validate_video_metadata(info)
            
            if validation_result["status"] == "invalid":
                raise ValueError(f"Video metadata validation failed: {validation_result['issues']}")
            
            # Log warnings if any
            if validation_result["warnings"]:
                LOGGER.warning("Video validation warnings: %s", validation_result["warnings"])
            
            # Combine metadata and analysis
            metadata = {
                **validated_metadata.dict(),
                'analysis': analysis,
                'validation': validation_result,
                'extraction_timestamp': datetime.now().isoformat(),
            }
            
            LOGGER.info("Metadata extraction complete: %s", validated_metadata.title)
            return metadata
            
        except Exception as e:
            LOGGER.error("Failed to extract metadata from %s: %s", youtube_url, e)
            raise RuntimeError(f"Metadata extraction failed: {e}") from e
    
    def _get_video_info(self, youtube_url: str) -> Dict[str, Any]:
        """Get basic video information."""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            
            if not info:
                raise RuntimeError("Failed to extract video information")
            
            # Extract relevant fields
            video_info = {
                'id': info.get('id'),
                'title': info.get('title'),
                'description': info.get('description'),
                'duration': info.get('duration'),  # seconds
                'view_count': info.get('view_count'),
                'like_count': info.get('like_count'),
                'dislike_count': info.get('dislike_count'),
                'comment_count': info.get('comment_count'),
                'uploader': info.get('uploader'),
                'uploader_id': info.get('uploader_id'),
                'channel_id': info.get('channel_id'),
                'channel_url': info.get('channel_url'),
                'upload_date': info.get('upload_date'),
                'upload_date_timestamp': info.get('timestamp'),
                'availability': info.get('availability'),
                'webpage_url': info.get('webpage_url'),
                'thumbnail': info.get('thumbnail'),
                'tags': info.get('tags', []),
                'categories': info.get('categories', []),
                'language': info.get('language'),
                'formats': info.get('formats', []),
                'resolution': info.get('resolution'),
                'fps': info.get('fps'),
                'vcodec': info.get('vcodec'),
                'acodec': info.get('acodec'),
                'filesize': info.get('filesize'),
                'url': youtube_url,
            }
            
            return video_info
    
    def _analyze_content(self, video_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze video content for soccer relevance."""
        title = video_info.get('title', '').lower()
        description = video_info.get('description', '').lower()
        tags = [tag.lower() for tag in video_info.get('tags', [])]
        uploader = video_info.get('uploader', '').lower()
        
        # Combine all text for analysis
        full_text = f"{title} {description} {' '.join(tags)}"
        
        # Keyword matching
        keyword_matches = []
        for keyword in self.soccer_keywords:
            if keyword.lower() in full_text:
                keyword_matches.append(keyword)
        
        # Calculate soccer score
        keyword_score = len(keyword_matches) / len(self.soccer_keywords)
        
        # Duration analysis
        duration = video_info.get('duration', 0)
        duration_category = self._categorize_duration(duration)
        
        # Content type analysis
        content_type = self._analyze_content_type(title, description, tags)
        
        # Channel analysis
        channel_analysis = self._analyze_channel(uploader)
        
        # Engagement analysis
        engagement_analysis = self._analyze_engagement(video_info)
        
        analysis = {
            'soccer_score': keyword_score,
            'keyword_matches': keyword_matches,
            'soccer_relevance': self._calculate_soccer_relevance(keyword_score),
            'duration_category': duration_category,
            'content_type': content_type,
            'channel_analysis': channel_analysis,
            'engagement_analysis': engagement_analysis,
            'text_analysis': {
                'title_length': len(title),
                'description_length': len(description),
                'tags_count': len(tags),
                'language_hint': self._detect_language(title, description),
            }
        }
        
        return analysis
    
    def _categorize_duration(self, duration: Optional[int]) -> str:
        """Categorize video by duration."""
        if duration is None:
            return 'unknown'
        
        for category, (min_duration, max_duration) in self.duration_patterns.items():
            if min_duration <= duration < max_duration:
                return category
        
        return 'very_long'  # 3+ hours
    
    def _analyze_content_type(self, title: str, description: str, tags: List[str]) -> str:
        """Analyze content type based on title, description, and tags."""
        title_lower = title.lower()
        desc_lower = description.lower()
        tags_lower = [tag.lower() for tag in tags]
        
        all_text = f"{title_lower} {desc_lower} {' '.join(tags_lower)}"
        
        # Content type patterns
        if any(word in all_text for word in ['highlight', 'resumen', 'goals', 'goles']):
            return 'highlights'
        elif any(word in all_text for word in ['full match', 'partido completo', 'full game']):
            return 'full_match'
        elif any(word in all_text for word in ['training', 'entrenamiento', 'practice']):
            return 'training'
        elif any(word in all_text for word in ['interview', 'entrevista', 'press conference']):
            return 'interview'
        elif any(word in all_text for word in ['news', 'noticias', 'transfer']):
            return 'news'
        elif any(word in all_text for word in ['tutorial', 'how to', 'tutorial']):
            return 'tutorial'
        else:
            return 'other'
    
    def _analyze_channel(self, uploader: str) -> Dict[str, Any]:
        """Analyze uploader/channel for soccer relevance."""
        soccer_channels = [
            'espn', 'fox sports', 'sky sports', 'bt sport',
            'uefa', 'fifa', 'premier league', 'laliga',
            'barcelona', 'real madrid', 'manchester united',
            'chelsea', 'arsenal', 'liverpool', 'tottenham'
        ]
        
        channel_score = 0
        for channel in soccer_channels:
            if channel in uploader:
                channel_score += 1
        
        return {
            'is_soccer_channel': channel_score > 0,
            'channel_score': channel_score,
            'channel_type': self._classify_channel_type(uploader),
        }
    
    def _classify_channel_type(self, uploader: str) -> str:
        """Classify the type of channel."""
        if any(word in uploader for word in ['official', 'club', 'team']):
            return 'official'
        elif any(word in uploader for word in ['news', 'media', 'sports']):
            return 'media'
        elif any(word in uploader for word in ['fan', 'blog', 'channel']):
            return 'fan'
        else:
            return 'unknown'
    
    def _analyze_engagement(self, video_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze engagement metrics."""
        view_count = video_info.get('view_count', 0)
        like_count = video_info.get('like_count', 0)
        comment_count = video_info.get('comment_count', 0)
        duration = video_info.get('duration', 1)
        
        # Calculate engagement metrics
        like_rate = (like_count / max(view_count, 1)) * 100 if view_count else 0
        comment_rate = (comment_count / max(view_count, 1)) * 100 if view_count else 0
        
        # Views per minute (for duration-adjusted engagement)
        views_per_minute = (view_count / max(duration / 60, 1)) if duration else 0
        
        return {
            'view_count': view_count,
            'like_count': like_count,
            'comment_count': comment_count,
            'like_rate': round(like_rate, 2),
            'comment_rate': round(comment_rate, 2),
            'views_per_minute': round(views_per_minute, 2),
            'engagement_score': self._calculate_engagement_score(like_rate, comment_rate, views_per_minute),
        }
    
    def _calculate_engagement_score(self, like_rate: float, comment_rate: float, views_per_minute: float) -> float:
        """Calculate overall engagement score (0-1)."""
        # Normalize and combine metrics
        like_score = min(like_rate / 5.0, 1.0)  # Assume 5% like rate is excellent
        comment_score = min(comment_rate / 1.0, 1.0)  # Assume 1% comment rate is excellent
        views_score = min(views_per_minute / 100.0, 1.0)  # Assume 100 views/minute is excellent
        
        return round((like_score + comment_score + views_score) / 3.0, 3)
    
    def _calculate_soccer_relevance(self, keyword_score: float) -> str:
        """Calculate soccer relevance level."""
        if keyword_score >= 0.1:
            return 'high'
        elif keyword_score >= 0.05:
            return 'medium'
        elif keyword_score >= 0.01:
            return 'low'
        else:
            return 'none'
    
    def _detect_language(self, title: str, description: str) -> str:
        """Detect the likely language of the content."""
        text = f"{title} {description}".lower()
        
        # Simple language detection based on common words
        if any(word in text for word in ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es']):
            return 'spanish'
        elif any(word in text for word in ['le', 'la', 'de', 'que', 'et', 'en', 'un', 'est']):
            return 'french'
        elif any(word in text for word in ['der', 'die', 'das', 'und', 'in', 'ein', 'ist']):
            return 'german'
        elif any(word in text for word in ['il', 'la', 'di', 'che', 'e', 'in', 'un', 'è']):
            return 'italian'
        else:
            return 'english'  # Default
    
    def predict_soccer_content(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Predict if content is soccer-related based on metadata.
        
        Args:
            metadata: Video metadata from extract_metadata()
            
        Returns:
            Dict containing prediction results
        """
        analysis = metadata.get('analysis', {})
        
        # Combine different signals
        keyword_score = analysis.get('soccer_score', 0)
        relevance = analysis.get('soccer_relevance', 'none')
        content_type = analysis.get('content_type', 'other')
        channel_score = analysis.get('channel_analysis', {}).get('channel_score', 0)
        
        # Weighted scoring
        prediction_score = (
            keyword_score * 0.5 +  # Keyword matching (50%)
            (channel_score * 0.1) * 0.3 +  # Channel relevance (30%)
            (1.0 if content_type in ['highlights', 'full_match'] else 0.0) * 0.2  # Content type (20%)
        )
        
        # Final prediction
        is_soccer = prediction_score >= 0.1  # Threshold for soccer classification
        
        prediction = {
            'is_soccer': is_soccer,
            'confidence': min(prediction_score, 1.0),
            'relevance_level': relevance,
            'content_type': content_type,
            'suitable_for_analysis': is_soccer and content_type in ['highlights', 'full_match'],
            'reasoning': self._generate_prediction_reasoning(keyword_score, channel_score, content_type),
        }
        
        LOGGER.info("Soccer prediction: %.2f confidence (%s)", 
                   prediction['confidence'], "soccer" if is_soccer else "non-soccer")
        
        return prediction
    
    def _generate_prediction_reasoning(self, keyword_score: float, channel_score: float, content_type: str) -> List[str]:
        """Generate reasoning for the prediction."""
        reasoning = []
        
        if keyword_score > 0.05:
            reasoning.append(f"High keyword relevance ({keyword_score:.2%})")
        elif keyword_score > 0.01:
            reasoning.append(f"Moderate keyword relevance ({keyword_score:.2%})")
        
        if channel_score > 0:
            reasoning.append(f"Soccer-related channel detected")
        
        if content_type in ['highlights', 'full_match']:
            reasoning.append(f"Soccer content type: {content_type}")
        elif content_type == 'other':
            reasoning.append("Generic content type")
        
        return reasoning
    
    def _create_validated_metadata(self, video_info: Dict[str, Any]) -> VideoMetadata:
        """Create validated VideoMetadata from raw video info."""
        try:
            # Convert upload_date string to datetime
            upload_date_str = video_info.get('upload_date')
            if upload_date_str:
                upload_date = datetime.strptime(upload_date_str, '%Y%m%d')
            else:
                upload_date = datetime.now()
            
            # Create VideoMetadata with proper field mapping
            metadata = VideoMetadata(
                video_id=video_info.get('id', ''),
                title=video_info.get('title', ''),
                description=video_info.get('description', ''),
                duration_seconds=video_info.get('duration', 0),
                channel_title=video_info.get('uploader', ''),
                channel_id=video_info.get('channel_id', ''),
                view_count=video_info.get('view_count', 0),
                like_count=video_info.get('like_count'),
                comment_count=video_info.get('comment_count'),
                tags=video_info.get('tags', []),
                categories=video_info.get('categories', []),
                publish_date=upload_date,
                resolution=video_info.get('resolution', 'unknown'),
                fps=video_info.get('fps', 30.0),
                audio_codec=video_info.get('acodec', 'unknown'),
                video_codec=video_info.get('vcodec', 'unknown')
            )
            
            # Validate metadata against schema
            validation_errors = validate_video_metadata(metadata)
            if validation_errors:
                raise ValueError(f"Metadata validation failed: {validation_errors}")
            
            return metadata
            
        except Exception as e:
            LOGGER.warning("Failed to create validated metadata: %s", e)
            # Return minimal valid metadata as fallback
            return VideoMetadata(
                video_id=video_info.get('id', 'unknown'),
                title=video_info.get('title', 'Unknown'),
                description='',
                duration_seconds=0,
                channel_title='Unknown',
                channel_id='unknown',
                view_count=0,
                publish_date=datetime.now(),
                resolution='unknown',
                fps=30.0,
                audio_codec='unknown',
                video_codec='unknown'
            )


# Convenience functions
def extract_youtube_metadata(youtube_url: str) -> Dict[str, Any]:
    """Extract metadata from YouTube video.
    
    Args:
        youtube_url: YouTube video URL
        
    Returns:
        Dict containing video metadata and analysis
    """
    parser = YouTubeMetadataParser()
    return parser.extract_metadata(youtube_url)


def predict_soccer_content(youtube_url: str) -> Dict[str, Any]:
    """Predict if YouTube content is soccer-related.
    
    Args:
        youtube_url: YouTube video URL
        
    Returns:
        Dict containing prediction results
    """
    parser = YouTubeMetadataParser()
    metadata = parser.extract_metadata(youtube_url)
    return parser.predict_soccer_content(metadata)