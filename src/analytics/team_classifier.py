"""Team Classification Module.

Classifies players into teams based on jersey color analysis using K-Means
clustering. Handles automatic team assignment from video frames.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TeamAssignment:
    """Team assignment for a tracked player."""
    track_id: int
    team_id: int  # 0=Team A (home), 1=Team B (away), -1=referee/unknown
    confidence: float
    dominant_color: Tuple[int, int, int]  # BGR color
    cluster_id: int


class JerseyColorClassifier:
    """Classifies players into teams based on jersey colors.

    Uses K-Means clustering on extracted jersey colors to identify
    teams and referees.

    Args:
        n_teams: Number of teams (default 2).
        min_samples: Minimum color samples required for clustering.
        referee_detection: Whether to detect referees as separate cluster.
    """

    def __init__(
        self,
        n_teams: int = 2,
        min_samples: int = 5,
        referee_detection: bool = True
    ) -> None:
        if KMeans is None:
            raise ImportError(
                "scikit-learn is required for team classification. "
                "Install with: pip install scikit-learn"
            )
        if cv2 is None:
            raise ImportError(
                "OpenCV is required for team classification. "
                "Install with: pip install opencv-python"
            )

        self.n_teams = n_teams
        self.min_samples = min_samples
        self.referee_detection = referee_detection

        # Number of clusters: teams + optional referee
        self.n_clusters = n_teams + (1 if referee_detection else 0)

        self.kmeans: Optional[KMeans] = None
        self.cluster_to_team: Dict[int, int] = {}
        self.cluster_colors: Dict[int, Tuple[int, int, int]] = {}

    def extract_jersey_color(
        self,
        frame: np.ndarray,
        bbox: List[float],
        use_hsv: bool = True
    ) -> Optional[np.ndarray]:
        """Extract dominant jersey color from a player bounding box.

        Args:
            frame: BGR image frame.
            bbox: Bounding box [x1, y1, x2, y2].
            use_hsv: Whether to use HSV color space.

        Returns:
            Dominant color as numpy array, or None if extraction fails.
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)

            # Validate bbox
            if x1 >= x2 or y1 >= y2:
                return None
            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                # Clip to frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            # Focus on upper 60% of bbox (jersey region, exclude legs/pitch)
            jersey_y2 = y1 + int(0.6 * (y2 - y1))

            # Also exclude head region (top 15%)
            jersey_y1 = y1 + int(0.15 * (y2 - y1))

            crop = frame[jersey_y1:jersey_y2, x1:x2]

            if crop.size == 0:
                return None

            # Convert to HSV for better color separation
            if use_hsv:
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

                # Mask out green (pitch) - HSV green is roughly H=35-85
                lower_green = np.array([35, 40, 40])
                upper_green = np.array([85, 255, 255])
                green_mask = cv2.inRange(hsv, lower_green, upper_green)

                # Also mask very dark and very bright pixels
                lower_valid = np.array([0, 30, 30])
                upper_valid = np.array([180, 255, 230])
                valid_mask = cv2.inRange(hsv, lower_valid, upper_valid)

                # Combine masks: valid pixels that are not green
                final_mask = cv2.bitwise_and(valid_mask, cv2.bitwise_not(green_mask))

                # Extract valid pixels
                valid_pixels = hsv[final_mask > 0]

                if len(valid_pixels) < 10:
                    # Not enough valid pixels, fall back to full crop
                    valid_pixels = hsv.reshape(-1, 3)
            else:
                valid_pixels = crop.reshape(-1, 3)

            if len(valid_pixels) < 10:
                return None

            # Use median for robustness
            dominant_color = np.median(valid_pixels, axis=0)

            return dominant_color.astype(np.float64)

        except Exception as e:
            LOGGER.warning("Error extracting jersey color: %s", e)
            return None

    def fit(
        self,
        color_samples: List[Tuple[int, np.ndarray]]
    ) -> None:
        """Fit the classifier on collected color samples.

        Args:
            color_samples: List of (track_id, color_array) tuples.
        """
        if len(color_samples) < self.min_samples:
            LOGGER.warning(
                "Insufficient samples for clustering: %d < %d",
                len(color_samples), self.min_samples
            )
            return

        # Extract just the colors for clustering
        colors = np.array([sample[1] for sample in color_samples], dtype=np.float64)
        track_ids = [sample[0] for sample in color_samples]

        LOGGER.info("Fitting K-Means with %d samples, %d clusters", len(colors), self.n_clusters)

        # Fit K-Means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10
        )
        labels = self.kmeans.fit_predict(colors)

        # Analyze clusters to assign team IDs
        self._assign_teams_to_clusters(colors, labels, track_ids)

    def _assign_teams_to_clusters(
        self,
        colors: np.ndarray,
        labels: np.ndarray,
        track_ids: List[int]
    ) -> None:
        """Assign team IDs to clusters based on cluster characteristics."""

        cluster_sizes = {}
        cluster_colors = {}

        for cluster_id in range(self.n_clusters):
            mask = labels == cluster_id
            cluster_sizes[cluster_id] = np.sum(mask)

            if cluster_sizes[cluster_id] > 0:
                # Get centroid color (mean of cluster)
                centroid = self.kmeans.cluster_centers_[cluster_id]
                cluster_colors[cluster_id] = tuple(map(int, centroid))

        self.cluster_colors = cluster_colors

        # Sort clusters by size (largest = team, smallest might be referee)
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: -x[1])

        # Assign teams based on size
        self.cluster_to_team = {}

        for i, (cluster_id, size) in enumerate(sorted_clusters):
            if i < self.n_teams:
                # Assign as team (0 = home, 1 = away)
                self.cluster_to_team[cluster_id] = i
            else:
                # Assign as referee/unknown
                self.cluster_to_team[cluster_id] = -1

        LOGGER.info("Cluster assignments: %s", self.cluster_to_team)
        LOGGER.info("Cluster colors: %s", self.cluster_colors)

    def predict(
        self,
        color: np.ndarray
    ) -> Tuple[int, float]:
        """Predict team ID for a color sample.

        Args:
            color: Color array to classify.

        Returns:
            Tuple of (team_id, confidence).
        """
        if self.kmeans is None:
            LOGGER.warning("Classifier not fitted, returning default team")
            return (0, 0.0)

        # Predict cluster
        cluster_id = self.kmeans.predict([np.asarray(color, dtype=np.float64)])[0]

        # Calculate confidence based on distance to cluster center
        center = self.kmeans.cluster_centers_[cluster_id]
        distance = np.linalg.norm(color - center)

        # Convert distance to confidence (closer = higher confidence)
        # Using exponential decay
        confidence = float(np.exp(-distance / 50.0))

        team_id = self.cluster_to_team.get(cluster_id, -1)

        return (team_id, confidence)

    def classify_tracks(
        self,
        frames: Dict[int, np.ndarray],
        tracklets: List[Dict],
        sample_frames: int = 5
    ) -> Dict[int, TeamAssignment]:
        """Classify all tracks across frames.

        Args:
            frames: Dictionary of frame_id -> frame image.
            tracklets: List of tracklet data with track_id and bbox info.
            sample_frames: Number of frames to sample per track.

        Returns:
            Dictionary of track_id -> TeamAssignment.
        """
        # Step 1: Collect color samples from each track
        color_samples = []
        track_colors: Dict[int, List[np.ndarray]] = {}

        for tracklet in tracklets:
            track_id = tracklet.get("track_id")
            bbox = tracklet.get("bbox")
            frame_id = tracklet.get("frame_id")

            if track_id is None or bbox is None or frame_id is None:
                continue

            if frame_id not in frames:
                continue

            frame = frames[frame_id]
            color = self.extract_jersey_color(frame, bbox)

            if color is not None:
                color_samples.append((track_id, color))

                if track_id not in track_colors:
                    track_colors[track_id] = []
                track_colors[track_id].append(color)

        LOGGER.info("Collected %d color samples from %d tracks", len(color_samples), len(track_colors))

        # Step 2: Fit classifier if enough samples
        if len(color_samples) >= self.min_samples:
            self.fit(color_samples)
        else:
            LOGGER.warning("Insufficient samples, using position-based fallback")
            return self._position_based_fallback(tracklets)

        # Step 3: Classify each track using median color
        assignments: Dict[int, TeamAssignment] = {}

        for track_id, colors in track_colors.items():
            # Use median color for this track
            median_color = np.median(colors, axis=0).astype(np.float64)
            team_id, confidence = self.predict(median_color)

            # Get cluster for this track
            cluster_id = self.kmeans.predict([median_color])[0] if self.kmeans else 0
            dominant_color = self.cluster_colors.get(cluster_id, (128, 128, 128))

            assignments[track_id] = TeamAssignment(
                track_id=track_id,
                team_id=team_id,
                confidence=confidence,
                dominant_color=dominant_color,
                cluster_id=cluster_id
            )

        LOGGER.info("Classified %d tracks into teams", len(assignments))

        return assignments

    def _position_based_fallback(
        self,
        tracklets: List[Dict]
    ) -> Dict[int, TeamAssignment]:
        """Fallback to position-based team assignment.

        Assigns team based on average x-position:
        - Left half of pitch = home team (0)
        - Right half of pitch = away team (1)
        """
        assignments: Dict[int, TeamAssignment] = {}
        track_positions: Dict[int, List[float]] = {}

        for tracklet in tracklets:
            track_id = tracklet.get("track_id")
            bbox = tracklet.get("bbox")

            if track_id is None or bbox is None:
                continue

            # Get center x position (normalized assumption)
            center_x = (bbox[0] + bbox[2]) / 2

            if track_id not in track_positions:
                track_positions[track_id] = []
            track_positions[track_id].append(center_x)

        for track_id, positions in track_positions.items():
            avg_x = np.mean(positions)

            # Assume normalized coordinates or pixel coordinates
            # Use 0.5 as midpoint for normalized, estimate for pixels
            if avg_x < 0.5 or (avg_x > 1 and avg_x < 960):  # Assuming 1920 width
                team_id = 0  # Home team (left side)
            else:
                team_id = 1  # Away team (right side)

            assignments[track_id] = TeamAssignment(
                track_id=track_id,
                team_id=team_id,
                confidence=0.3,  # Low confidence for position-based
                dominant_color=(128, 128, 128),
                cluster_id=-1
            )

        return assignments


__all__ = [
    "JerseyColorClassifier",
    "TeamAssignment",
]
