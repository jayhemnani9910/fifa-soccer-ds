"""Audio extraction and processing for YouTube videos.

This module provides functionality to extract audio from YouTube videos
for soccer content classification using speech recognition and audio analysis.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

try:
    import whisper
except ImportError:
    whisper = None

try:
    import librosa
    import numpy as np
except ImportError:
    librosa = None
    np = None

try:
    import ffmpeg
except ImportError:
    ffmpeg = None

LOGGER = logging.getLogger(__name__)


class AudioExtractor:
    """Extract and process audio from video files for soccer classification."""
    
    def __init__(self, model_size: str = "base"):
        """Initialize the audio extractor.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
        """
        if whisper is None:
            raise ImportError("openai-whisper is required. Install with: pip install openai-whisper")
        
        if librosa is None:
            raise ImportError("librosa is required. Install with: pip install librosa")
        
        if ffmpeg is None:
            raise ImportError("ffmpeg is required. Install ffmpeg system package")
        
        self.model_size = model_size
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the Whisper model."""
        try:
            LOGGER.info("Loading Whisper model: %s", self.model_size)
            self.model = whisper.load_model(self.model_size)
            LOGGER.info("Whisper model loaded successfully")
        except Exception as e:
            LOGGER.error("Failed to load Whisper model: %s", e)
            raise RuntimeError(f"Failed to load Whisper model: {e}") from e
    
    def extract_audio(self, video_path: Path, output_path: Optional[Path] = None) -> Path:
        """Extract audio from video file.
        
        Args:
            video_path: Path to video file
            output_path: Where to save extracted audio
            
        Returns:
            Path to extracted audio file
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if output_path is None:
            output_path = video_path.with_suffix('.wav')
        
        try:
            LOGGER.info("Extracting audio from: %s", video_path)
            
            # Extract audio using ffmpeg
            (
                ffmpeg
                .input(str(video_path))
                .output(str(output_path), acodec='pcm_s16le', ac=1, ar=16000)
                .overwrite_output()
                .run(quiet=True)
            )
            
            LOGGER.info("Audio extracted to: %s", output_path)
            return output_path
            
        except Exception as e:
            LOGGER.error("Failed to extract audio from %s: %s", video_path, e)
            raise RuntimeError(f"Audio extraction failed: {e}") from e
    
    def transcribe_audio(self, audio_path: Path, language: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio to text using Whisper.
        
        Args:
            audio_path: Path to audio file
            language: Expected language code (e.g., 'en', 'es')
            
        Returns:
            Dict containing transcription results
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            LOGGER.info("Transcribing audio: %s", audio_path)
            
            # Transcribe with Whisper
            result = self.model.transcribe(
                str(audio_path),
                language=language,
                task="transcribe",
                verbose=False
            )
            
            # Process result
            transcription = {
                'text': result['text'].strip(),
                'language': result.get('language', language),
                'segments': [],
                'duration': 0.0,
            }
            
            # Process segments with timestamps
            for segment in result.get('segments', []):
                transcription['segments'].append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip(),
                    'confidence': segment.get('avg_logprob', 0.0)
                })
            
            # Calculate total duration
            if transcription['segments']:
                transcription['duration'] = max(
                    seg['end'] for seg in transcription['segments']
                )
            else:
                transcription['duration'] = 0.0
            
            LOGGER.info("Transcription complete: %.1f seconds", transcription['duration'])
            return transcription
            
        except Exception as e:
            LOGGER.error("Failed to transcribe audio %s: %s", audio_path, e)
            raise RuntimeError(f"Transcription failed: {e}") from e
    
    def analyze_audio_features(self, audio_path: Path) -> Dict[str, Any]:
        """Analyze audio features for soccer content classification.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict containing audio features
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            LOGGER.info("Analyzing audio features: %s", audio_path)
            
            # Load audio
            y, sr = librosa.load(str(audio_path), sr=16000)
            
            features = {
                'duration': len(y) / sr,
                'sample_rate': sr,
                'energy': float(np.mean(y ** 2)),
                'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y))),
                'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
                'spectral_rolloff': float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
                'mfcc': [],
                'chroma': [],
                'tempo': 0.0,
            }
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc'] = [float(x) for x in np.mean(mfcc, axis=1)]
            
            # Extract chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma'] = [float(x) for x in np.mean(chroma, axis=1)]
            
            # Estimate tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            
            LOGGER.info("Audio analysis complete: %.1f seconds, %.1f BPM", 
                       features['duration'], features['tempo'])
            return features
            
        except Exception as e:
            LOGGER.error("Failed to analyze audio %s: %s", audio_path, e)
            raise RuntimeError(f"Audio analysis failed: {e}") from e
    
    def classify_soccer_audio(self, transcription: Dict[str, Any], 
                            audio_features: Dict[str, Any]) -> Dict[str, Any]:
        """Classify if audio content is soccer-related.
        
        Args:
            transcription: Transcription results
            audio_features: Audio feature analysis
            
        Returns:
            Dict containing classification results
        """
        # Soccer-related keywords and patterns
        soccer_keywords = [
            'goal', 'soccer', 'football', 'player', 'team', 'match', 'game',
            'penalty', 'foul', 'corner', 'free kick', 'offside', 'referee',
            'striker', 'defender', 'midfielder', 'goalkeeper', 'coach',
            'barcelona', 'real madrid', 'messi', 'ronaldo', 'champions league',
            'world cup', 'premier league', 'laliga', 'serie a', 'bundesliga'
        ]
        
        # Audio patterns that suggest soccer
        soccer_audio_patterns = {
            'crowd_noise': audio_features.get('energy', 0) > 0.01,
            'fast_pace': audio_features.get('tempo', 0) > 100,
            'spectral_characteristics': audio_features.get('spectral_centroid', 0) > 1000,
        }
        
        # Analyze transcription
        text = transcription.get('text', '').lower()
        keyword_matches = sum(1 for keyword in soccer_keywords if keyword in text)
        
        # Calculate scores
        keyword_score = min(keyword_matches / 5.0, 1.0)  # Normalize to 0-1
        audio_score = sum(soccer_audio_patterns.values()) / len(soccer_audio_patterns)
        
        # Combined confidence
        confidence = (keyword_score * 0.7 + audio_score * 0.3)
        
        classification = {
            'is_soccer': confidence >= 0.5,
            'confidence': confidence,
            'keyword_score': keyword_score,
            'audio_score': audio_score,
            'keyword_matches': keyword_matches,
            'matched_keywords': [kw for kw in soccer_keywords if kw in text],
            'audio_patterns': soccer_audio_patterns,
            'analysis': {
                'text_length': len(text),
                'audio_duration': audio_features.get('duration', 0),
                'tempo_bpm': audio_features.get('tempo', 0),
                'energy_level': audio_features.get('energy', 0),
            }
        }
        
        LOGGER.info("Audio classification: %.2f confidence (%s)", 
                   confidence, "soccer" if classification['is_soccer'] else "non-soccer")
        
        return classification
    
    def process_sample_audio(self, video_path: Path, duration_seconds: float = 60.0) -> Dict[str, Any]:
        """Process a sample of audio from video for classification.
        
        Args:
            video_path: Path to video file
            duration_seconds: Duration of audio sample to process
            
        Returns:
            Dict containing all analysis results
        """
        try:
            LOGGER.info("Processing audio sample from: %s (%.1f seconds)", 
                       video_path, duration_seconds)
            
            # Create temporary audio file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = Path(temp_audio.name)
            
            try:
                # Extract sample audio
                (
                    ffmpeg
                    .input(str(video_path), ss=0, t=duration_seconds)
                    .output(str(temp_audio_path), acodec='pcm_s16le', ac=1, ar=16000)
                    .overwrite_output()
                    .run(quiet=True)
                )
                
                # Transcribe
                transcription = self.transcribe_audio(temp_audio_path)
                
                # Analyze audio features
                audio_features = self.analyze_audio_features(temp_audio_path)
                
                # Classify
                classification = self.classify_soccer_audio(transcription, audio_features)
                
                result = {
                    'video_path': str(video_path),
                    'sample_duration': duration_seconds,
                    'transcription': transcription,
                    'audio_features': audio_features,
                    'classification': classification,
                }
                
                LOGGER.info("Audio sample processing complete")
                return result
                
            finally:
                # Clean up temporary file
                if temp_audio_path.exists():
                    temp_audio_path.unlink()
                    
        except Exception as e:
            LOGGER.error("Failed to process audio sample from %s: %s", video_path, e)
            raise RuntimeError(f"Audio sample processing failed: {e}") from e


# Convenience functions
def extract_audio_from_video(video_path: Path, output_path: Optional[Path] = None) -> Path:
    """Extract audio from video file.
    
    Args:
        video_path: Path to video file
        output_path: Where to save extracted audio
        
    Returns:
        Path to extracted audio file
    """
    extractor = AudioExtractor()
    return extractor.extract_audio(video_path, output_path)


def transcribe_youtube_audio(audio_path: Path, language: Optional[str] = None) -> Dict[str, Any]:
    """Transcribe audio to text.
    
    Args:
        audio_path: Path to audio file
        language: Expected language code
        
    Returns:
        Dict containing transcription results
    """
    extractor = AudioExtractor()
    return extractor.transcribe_audio(audio_path, language)


def classify_youtube_audio(transcription: Dict[str, Any], 
                         audio_features: Dict[str, Any]) -> Dict[str, Any]:
    """Classify if audio content is soccer-related.
    
    Args:
        transcription: Transcription results
        audio_features: Audio feature analysis
        
    Returns:
        Dict containing classification results
    """
    extractor = AudioExtractor()
    return extractor.classify_soccer_audio(transcription, audio_features)