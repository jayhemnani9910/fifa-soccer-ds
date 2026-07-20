"""Soccer content classifier for YouTube videos.

This module combines visual, audio, and metadata analysis to classify
whether YouTube content is soccer-related.
"""

from __future__ import annotations

import logging
import math
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.schemas import validate_youtube_url
from src.youtube.audio_extractor import AudioExtractor
from src.youtube.metadata_parser import YouTubeMetadataParser
from src.youtube.video_downloader import YouTubeDownloader, extract_youtube_thumbnail

LOGGER = logging.getLogger(__name__)


class SoccerClassifier:
    """Multi-modal soccer content classifier."""

    def __init__(self, confidence_threshold: float = 0.75):
        """Initialize the soccer classifier.

        Args:
            confidence_threshold: Threshold for soccer classification (0-1)
        """
        if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be finite and within [0, 1]")
        self.confidence_threshold = confidence_threshold

        # Initialize components
        self.audio_extractor: AudioExtractor | None = None
        self.metadata_parser = YouTubeMetadataParser()

        # Soccer visual indicators (for thumbnail analysis)
        self.soccer_visual_patterns = {
            "field_lines": ["white lines", "rectangle field", "grass pattern"],
            "team_colors": ["jersey colors", "team uniforms"],
            "ball_shape": ["circular object", "soccer ball"],
            "player_formation": ["multiple players", "team formation"],
        }

        LOGGER.info("Soccer classifier initialized with threshold: %.2f", confidence_threshold)

    def classify_youtube_content(
        self,
        youtube_url: str,
        sample_duration: float = 60.0,
        include_audio: bool = False,
        max_video_duration: int | None = None,
    ) -> dict[str, Any]:
        """Classify YouTube content using multi-modal analysis.

        Args:
            youtube_url: YouTube video URL
            sample_duration: Duration of audio/video sample to analyze
            include_audio: Whether to download and analyze an audio sample
            max_video_duration: Reject longer or duration-unknown audio downloads

        Returns:
            Dict containing classification results and scores
        """
        # Validate input using schema
        if not validate_youtube_url(youtube_url):
            raise ValueError(f"Invalid YouTube URL format: {youtube_url}")

        if not math.isfinite(sample_duration) or sample_duration <= 0 or sample_duration > 300:
            raise ValueError("Sample duration must be between 0 and 300 seconds")
        if max_video_duration is not None and max_video_duration <= 0:
            raise ValueError("max_video_duration must be positive")

        LOGGER.info("Classifying YouTube content: %s", youtube_url)

        try:
            # 1. Metadata Analysis
            metadata_result = self.metadata_parser.extract_metadata(youtube_url)
            metadata_prediction = self.metadata_parser.predict_soccer_content(metadata_result)

            # 2. Thumbnail Analysis
            thumbnail_result = self._analyze_thumbnail(youtube_url)

            # 3. Audio Analysis (if video is long enough)
            audio_result = None
            if include_audio:
                audio_result = self._analyze_audio_sample(
                    youtube_url, sample_duration, max_video_duration=max_video_duration
                )

            # 4. Combine Results
            final_result = self._combine_classifications(
                metadata_prediction, thumbnail_result, audio_result
            )

            # 5. Add detailed breakdown
            publish_date = metadata_result.get("publish_date")
            if isinstance(publish_date, datetime):
                publish_date = publish_date.isoformat()

            result = {
                "youtube_url": youtube_url,
                "is_soccer": final_result["is_soccer"],
                "confidence": final_result["confidence"],
                "classification": final_result["classification"],
                "analysis_breakdown": {
                    "metadata_analysis": {
                        "score": metadata_prediction.get("confidence", 0),
                        "relevance": metadata_prediction.get("relevance_level", "none"),
                        "content_type": metadata_prediction.get("content_type", "other"),
                        "reasoning": metadata_prediction.get("reasoning", []),
                    },
                    "thumbnail_analysis": {
                        "score": thumbnail_result.get("visual_score", 0),
                        "soccer_elements": thumbnail_result.get("soccer_elements", []),
                        "confidence": thumbnail_result.get("confidence", 0),
                    },
                    "audio_analysis": audio_result.get("classification", {})
                    if audio_result
                    else None,
                },
                "processing_info": {
                    "sample_duration": sample_duration,
                    "timestamp": metadata_result.get("extraction_timestamp"),
                    "video_info": {
                        "id": metadata_result.get("video_id"),
                        "title": metadata_result.get("title"),
                        "description": metadata_result.get("description", ""),
                        "duration": metadata_result.get("duration_seconds"),
                        "uploader": metadata_result.get("channel_title"),
                        "channel_id": metadata_result.get("channel_id"),
                        "view_count": metadata_result.get("view_count"),
                        "like_count": metadata_result.get("like_count"),
                        "comment_count": metadata_result.get("comment_count"),
                        "tags": metadata_result.get("tags", []),
                        "categories": metadata_result.get("categories", []),
                        "upload_date": publish_date,
                        "resolution": metadata_result.get("resolution"),
                        "fps": metadata_result.get("fps"),
                        "acodec": metadata_result.get("audio_codec"),
                        "vcodec": metadata_result.get("video_codec"),
                    },
                },
            }

            LOGGER.info(
                "Classification complete: %s (confidence: %.2f)",
                "soccer" if result["is_soccer"] else "non-soccer",
                result["confidence"],
            )

            return result

        except Exception as e:
            LOGGER.error("Classification failed for %s: %s", youtube_url, e)
            raise RuntimeError(f"Classification failed: {e}") from e

    def _analyze_thumbnail(self, youtube_url: str) -> dict[str, Any]:
        """Analyze video thumbnail for soccer elements.

        Args:
            youtube_url: YouTube video URL

        Returns:
            Dict containing thumbnail analysis results
        """
        try:
            # Extract thumbnail
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_thumb:
                thumbnail_path = Path(temp_thumb.name)

            try:
                extract_youtube_thumbnail(youtube_url, thumbnail_path)

                # Load and analyze thumbnail
                with Image.open(thumbnail_path) as source_image:
                    image = source_image.convert("RGB")
                    image_size = image.size
                    image_array = np.asarray(image)

                # Basic image analysis
                analysis = self._analyze_image_for_soccer(image_array)

                return {
                    "thumbnail_path": None,
                    "image_size": image_size,
                    "visual_score": analysis["visual_score"],
                    "soccer_elements": analysis["soccer_elements"],
                    "confidence": analysis["confidence"],
                    "analysis": analysis,
                }

            finally:
                # Clean up temporary file
                if thumbnail_path.exists():
                    thumbnail_path.unlink()

        except Exception as e:
            LOGGER.warning("Thumbnail analysis failed: %s", e)
            return {
                "thumbnail_path": None,
                "visual_score": 0.0,
                "soccer_elements": [],
                "confidence": 0.0,
                "error": str(e),
            }

    def _analyze_image_for_soccer(self, image_array) -> dict[str, Any]:
        """Analyze image for soccer-related visual elements.

        Args:
            image_array: numpy array representing the image

        Returns:
            Dict containing visual analysis results
        """
        try:
            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

            # Detect green areas (grass field)
            green_mask = cv2.inRange(
                hsv,
                np.array((40, 40, 40), dtype=np.uint8),
                np.array((80, 255, 255), dtype=np.uint8),
            )
            green_ratio = np.sum(green_mask > 0) / green_mask.size

            # Detect white lines (field markings)
            white_mask = cv2.inRange(
                hsv,
                np.array((0, 0, 200), dtype=np.uint8),
                np.array((180, 30, 255), dtype=np.uint8),
            )
            white_ratio = np.sum(white_mask > 0) / white_mask.size

            # Detect circles (potential soccer balls)
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=20,
                param1=50,
                param2=30,
                minRadius=5,
                maxRadius=50,
            )
            circle_count = circles.shape[1] if circles is not None else 0

            # Detect rectangular shapes (field boundaries)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            rectangular_shapes = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum area threshold
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if len(approx) >= 4:  # Rectangle-like shape
                        rectangular_shapes += 1

            # Calculate visual score
            visual_indicators = {
                "green_field": min(green_ratio * 10, 1.0),  # Grass field
                "white_lines": min(white_ratio * 50, 1.0),  # Field markings
                "circular_objects": min(circle_count / 5.0, 1.0),  # Balls
                "rectangular_shapes": min(rectangular_shapes / 10.0, 1.0),  # Field boundaries
            }

            visual_score = sum(visual_indicators.values()) / len(visual_indicators)

            # Identify detected soccer elements
            soccer_elements = []
            if green_ratio > 0.3:
                soccer_elements.append("grass_field")
            if white_ratio > 0.05:
                soccer_elements.append("field_lines")
            if circle_count > 0:
                soccer_elements.append("circular_objects")
            if rectangular_shapes > 2:
                soccer_elements.append("field_boundaries")

            return {
                "visual_score": visual_score,
                "soccer_elements": soccer_elements,
                "confidence": visual_score,
                "indicators": visual_indicators,
                "image_stats": {
                    "green_ratio": green_ratio,
                    "white_ratio": white_ratio,
                    "circle_count": circle_count,
                    "rectangular_shapes": rectangular_shapes,
                },
            }

        except Exception as e:
            LOGGER.warning("Advanced image analysis failed: %s", e)
            return self._basic_image_analysis(image_array)

    def _basic_image_analysis(self, image_array) -> dict[str, Any]:
        """Basic image analysis without OpenCV dependencies.

        Args:
            image_array: numpy array representing the image

        Returns:
            Dict containing basic visual analysis results
        """
        try:
            # Simple color analysis
            if len(image_array.shape) == 3:
                # Color image
                red_channel = image_array[:, :, 0]
                green_channel = image_array[:, :, 1]
                blue_channel = image_array[:, :, 2]

                # Check for green dominance (grass field)
                green_dominance = np.mean(green_channel > red_channel) * np.mean(
                    green_channel > blue_channel
                )

                # Check for white/light areas (field lines)
                light_areas = np.mean(image_array > 200)

                # Simple score based on color analysis
                visual_score = green_dominance * 0.6 + light_areas * 0.4

                soccer_elements = []
                if green_dominance > 0.4:
                    soccer_elements.append("green_dominant")
                if light_areas > 0.1:
                    soccer_elements.append("light_areas")

            else:
                # Grayscale image
                light_areas = np.mean(image_array > 200)
                visual_score = light_areas
                soccer_elements = ["light_areas"] if light_areas > 0.1 else []

            return {
                "visual_score": float(visual_score),
                "soccer_elements": soccer_elements,
                "confidence": float(visual_score),
                "analysis_type": "basic",
            }

        except Exception as e:
            LOGGER.warning("Basic image analysis failed: %s", e)
            return {
                "visual_score": 0.0,
                "soccer_elements": [],
                "confidence": 0.0,
                "error": str(e),
            }

    def _analyze_audio_sample(
        self,
        youtube_url: str,
        duration: float,
        *,
        max_video_duration: int | None = None,
    ) -> dict[str, Any]:
        """Analyze audio sample for soccer content.

        Args:
            youtube_url: YouTube video URL
            duration: Duration of audio sample to analyze

        Returns:
            Dict containing audio analysis results
        """
        try:
            with tempfile.TemporaryDirectory(prefix="fifa_audio_sample_") as cache_dir:
                downloader = YouTubeDownloader(cache_dir=Path(cache_dir))
                video_info = downloader.download_video(youtube_url, max_duration=max_video_duration)
                video_path = Path(video_info["video_path"])

                # Process audio sample
                if self.audio_extractor is None:
                    self.audio_extractor = AudioExtractor()
                audio_result = self.audio_extractor.process_sample_audio(video_path, duration)

                return {
                    "audio_analysis": audio_result,
                    "classification": audio_result.get("classification", {}),
                }

        except Exception as e:
            LOGGER.warning("Audio analysis failed: %s", e)
            return {
                "audio_analysis": None,
                "classification": {
                    "is_soccer": False,
                    "confidence": 0.0,
                    "error": str(e),
                },
            }

    def _combine_classifications(
        self,
        metadata_result: dict[str, Any],
        thumbnail_result: dict[str, Any],
        audio_result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Combine results from different classification methods.

        Args:
            metadata_result: Metadata analysis results
            thumbnail_result: Thumbnail analysis results
            audio_result: Audio analysis results

        Returns:
            Dict containing combined classification results
        """
        # Extract individual scores
        metadata_score = metadata_result.get("confidence", 0)
        visual_score = thumbnail_result.get("visual_score", 0)
        audio_score = 0.0

        if audio_result and "classification" in audio_result:
            audio_score = audio_result["classification"].get("confidence", 0)

        # Base weights; unavailable modalities are excluded and the remainder
        # is normalized so opting out of audio does not lower confidence.
        base_weights = {
            "metadata": 0.5,  # 50% weight
            "visual": 0.3,  # 30% weight
            "audio": 0.2,  # 20% weight
        }
        available = {"metadata": metadata_score}
        if "error" not in thumbnail_result:
            available["visual"] = visual_score
        audio_classification = audio_result.get("classification", {}) if audio_result else {}
        if audio_result and "error" not in audio_classification:
            available["audio"] = audio_score

        available_weight = sum(base_weights[name] for name in available)
        weights = {name: base_weights[name] / available_weight for name in available}

        # Calculate weighted confidence
        combined_confidence = sum(score * weights[name] for name, score in available.items())

        # Final classification
        is_soccer = combined_confidence >= self.confidence_threshold

        # Determine classification category
        if combined_confidence >= 0.8:
            classification = "highly_soccer"
        elif combined_confidence >= 0.6:
            classification = "probably_soccer"
        elif combined_confidence >= 0.4:
            classification = "possibly_soccer"
        elif combined_confidence >= 0.2:
            classification = "unlikely_soccer"
        else:
            classification = "not_soccer"

        return {
            "is_soccer": is_soccer,
            "confidence": combined_confidence,
            "classification": classification,
            "individual_scores": {
                "metadata_score": metadata_score,
                "visual_score": visual_score,
                "audio_score": audio_score,
            },
            "weights_used": weights,
        }

    def quick_classify(self, youtube_url: str) -> dict[str, Any]:
        """Quick classification using only metadata and thumbnail.

        Args:
            youtube_url: YouTube video URL

        Returns:
            Dict containing quick classification results
        """
        LOGGER.info("Quick classification: %s", youtube_url)

        try:
            # Metadata analysis
            metadata_result = self.metadata_parser.extract_metadata(youtube_url)
            metadata_prediction = self.metadata_parser.predict_soccer_content(metadata_result)

            # Thumbnail analysis
            thumbnail_result = self._analyze_thumbnail(youtube_url)

            # Combine results
            combined = self._combine_classifications(
                metadata_prediction,
                thumbnail_result,
                None,  # No audio analysis
            )

            return {
                "youtube_url": youtube_url,
                "is_soccer": combined["is_soccer"],
                "confidence": combined["confidence"],
                "classification": combined["classification"],
                "analysis_breakdown": {
                    "metadata_analysis": {
                        "score": metadata_prediction.get("confidence", 0),
                        "relevance": metadata_prediction.get("relevance_level", "none"),
                        "content_type": metadata_prediction.get("content_type", "other"),
                    },
                    "thumbnail_analysis": {
                        "score": thumbnail_result.get("visual_score", 0),
                        "soccer_elements": thumbnail_result.get("soccer_elements", []),
                    },
                },
                "processing_time": "fast",
            }

        except Exception as e:
            LOGGER.error("Quick classification failed: %s", e)
            return {
                "youtube_url": youtube_url,
                "is_soccer": False,
                "confidence": 0.0,
                "classification": "error",
                "error": str(e),
            }


# Convenience functions
def classify_youtube_soccer_content(
    youtube_url: str, confidence_threshold: float = 0.75
) -> dict[str, Any]:
    """Classify if YouTube content is soccer-related.

    Args:
        youtube_url: YouTube video URL
        confidence_threshold: Threshold for classification

    Returns:
        Dict containing classification results
    """
    classifier = SoccerClassifier(confidence_threshold)
    return classifier.classify_youtube_content(youtube_url)


def quick_soccer_classify(youtube_url: str) -> dict[str, Any]:
    """Quick soccer classification using metadata and thumbnail only.

    Args:
        youtube_url: YouTube video URL

    Returns:
        Dict containing quick classification results
    """
    classifier = SoccerClassifier()
    return classifier.quick_classify(youtube_url)
