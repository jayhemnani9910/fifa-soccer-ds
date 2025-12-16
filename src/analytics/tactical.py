"""Tactical Analysis Module.

Provides unified interface for tactical soccer analysis including:
- Pitch control computation
- Expected threat (xT) analysis
- Off-ball scoring opportunities (OBSO)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.analytics._pitch_control import (
    PitchControlModel,
    PitchControlConfig,
    PlayerState,
    PitchControlResult,
)

LOGGER = logging.getLogger(__name__)

# Re-export for convenience
__all__ = [
    "TacticalAnalyzer",
    "TacticalConfig",
    "PlayerState",
    "PitchControlResult",
    "TacticalResult",
]


@dataclass(slots=True)
class TacticalConfig:
    """Configuration for tactical analysis."""
    grid_shape: Tuple[int, int] = (12, 16)  # (rows, cols)
    max_speed: float = 5.0  # m/s
    reaction_time: float = 0.7  # seconds
    pitch_length: float = 105.0  # meters
    pitch_width: float = 68.0  # meters
    time_horizon: float = 5.0  # seconds
    enable_xT: bool = True
    enable_obso: bool = True


@dataclass
class TacticalResult:
    """Complete tactical analysis result for a frame."""
    frame_id: int
    pitch_control: PitchControlResult
    xT_grid: Optional[np.ndarray] = None
    obso_grid: Optional[np.ndarray] = None
    home_obso_total: float = 0.0
    away_obso_total: float = 0.0


class TacticalAnalyzer:
    """Unified tactical analysis combining PitchControl, xT, and OBSO.

    Provides a single interface for computing all tactical metrics
    for a given frame with player positions.

    Args:
        config: Configuration for tactical analysis.
    """

    def __init__(self, config: Optional[TacticalConfig] = None) -> None:
        self.config = config or TacticalConfig()

        # Initialize pitch control model
        pc_config = PitchControlConfig(
            max_speed=self.config.max_speed,
            reaction_time=self.config.reaction_time,
            pitch_length=self.config.pitch_length,
            pitch_width=self.config.pitch_width,
            time_horizon=self.config.time_horizon,
        )
        self.pitch_control = PitchControlModel(
            grid_shape=self.config.grid_shape,
            config=pc_config
        )

        # Initialize xT grid (static expected threat values)
        self.xT_grid = self._create_default_xT_grid()

        LOGGER.info(
            "Initialized TacticalAnalyzer with grid shape %s",
            self.config.grid_shape
        )

    def _create_default_xT_grid(self) -> np.ndarray:
        """Create default Expected Threat (xT) grid.

        Based on Karun Singh's xT model - values represent the probability
        of scoring from each zone of the pitch.

        The grid increases in value towards the opponent's goal.
        """
        rows, cols = self.config.grid_shape

        # Create base xT grid (higher values near goal)
        xT = np.zeros((rows, cols), dtype=np.float32)

        for i in range(rows):
            for j in range(cols):
                # x-position (0 = own goal, 1 = opponent goal)
                x_norm = j / (cols - 1) if cols > 1 else 0.5
                # y-position (0 = left touchline, 1 = right touchline)
                y_norm = i / (rows - 1) if rows > 1 else 0.5

                # Distance from center of pitch (for penalty box effect)
                y_center_dist = abs(y_norm - 0.5) * 2

                # Base xT increases exponentially towards goal
                base_xT = x_norm ** 3 * 0.25

                # Bonus for central positions (near goal center)
                central_bonus = (1 - y_center_dist) * 0.1 * x_norm

                # Extra boost in "penalty box" region
                if x_norm > 0.83 and y_center_dist < 0.3:  # ~17m from goal, central
                    penalty_box_bonus = 0.15
                else:
                    penalty_box_bonus = 0

                xT[i, j] = base_xT + central_bonus + penalty_box_bonus

        # Normalize to [0, 0.5] range (max xT is about 0.5 for direct shots)
        xT = np.clip(xT, 0, 0.5)

        return xT

    def compute(
        self,
        frame_id: int,
        players: Optional[List[PlayerState]] = None
    ) -> TacticalResult:
        """Compute all tactical metrics for a frame.

        Args:
            frame_id: Frame identifier.
            players: List of player states with positions and team IDs.

        Returns:
            TacticalResult with all computed metrics.
        """
        # Compute pitch control
        pc_result = self.pitch_control.compute(frame_id, players)

        # Compute OBSO if enabled
        if self.config.enable_obso:
            obso_home, obso_away, home_total, away_total = self._compute_obso(pc_result.grid)
        else:
            obso_home = obso_away = None
            home_total = away_total = 0.0

        return TacticalResult(
            frame_id=frame_id,
            pitch_control=pc_result,
            xT_grid=self.xT_grid if self.config.enable_xT else None,
            obso_grid=obso_home,  # Return home team's OBSO
            home_obso_total=home_total,
            away_obso_total=away_total,
        )

    def _compute_obso(
        self,
        pitch_control_grid: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Compute Off-Ball Scoring Opportunities.

        OBSO = Pitch Control * xT

        Represents the expected goal value of controlling each zone.
        """
        # Home team OBSO (where they control AND can score)
        obso_home = pitch_control_grid * self.xT_grid

        # Away team OBSO (using flipped xT since they attack opposite direction)
        xT_flipped = np.flip(self.xT_grid, axis=1)
        obso_away = (1 - pitch_control_grid) * xT_flipped

        # Total OBSO for each team
        home_total = float(np.sum(obso_home))
        away_total = float(np.sum(obso_away))

        return obso_home, obso_away, home_total, away_total

    def compute_from_tracklets(
        self,
        frame_id: int,
        tracklets_data: List[Dict[str, Any]],
        team_assignments: Optional[Dict[int, int]] = None,
        image_shape: Optional[Tuple[int, int]] = None
    ) -> TacticalResult:
        """Compute tactical metrics from tracklet data.

        Convenience method that converts tracklet format to PlayerState.

        Args:
            frame_id: Frame identifier.
            tracklets_data: List of tracklet dicts with track_id, bbox, etc.
            team_assignments: Optional mapping of track_id -> team_id.
            image_shape: Image dimensions (height, width) for normalization.

        Returns:
            TacticalResult with all computed metrics.
        """
        players = []

        for track_data in tracklets_data:
            track_id = track_data.get("track_id")
            bbox = track_data.get("bbox")
            class_name = track_data.get("class_name", "")

            # Skip non-players
            if class_name not in ("person", "player", ""):
                continue

            if track_id is None or bbox is None:
                continue

            # Convert bbox to position (use foot point - bottom center)
            x1, y1, x2, y2 = bbox
            pos_x = (x1 + x2) / 2
            pos_y = y2  # Foot position

            # Normalize if image shape provided
            if image_shape is not None:
                height, width = image_shape
                pos_x = pos_x / width
                pos_y = pos_y / height

            # Get team assignment
            if team_assignments and track_id in team_assignments:
                team_id = team_assignments[track_id]
            else:
                # Default: position-based (left = home)
                team_id = 0 if pos_x < 0.5 else 1

            players.append(PlayerState(
                player_id=track_id,
                team_id=team_id,
                position=np.array([pos_x, pos_y]),
                velocity=None
            ))

        return self.compute(frame_id, players)

    def to_dict(self, result: TacticalResult) -> Dict[str, Any]:
        """Convert TacticalResult to JSON-serializable dict."""
        return {
            "frame_id": result.frame_id,
            "pitch_control": {
                "grid": result.pitch_control.grid.tolist(),
                "home_control_pct": round(result.pitch_control.home_control_pct, 2),
                "away_control_pct": round(result.pitch_control.away_control_pct, 2),
            },
            "xT": {
                "grid": result.xT_grid.tolist() if result.xT_grid is not None else None,
            },
            "obso": {
                "grid": result.obso_grid.tolist() if result.obso_grid is not None else None,
                "home_total": round(result.home_obso_total, 4),
                "away_total": round(result.away_obso_total, 4),
            },
        }

    def get_aggregate_stats(
        self,
        results: List[TacticalResult]
    ) -> Dict[str, Any]:
        """Compute aggregate statistics across multiple frames.

        Args:
            results: List of TacticalResult from multiple frames.

        Returns:
            Dict with aggregate statistics.
        """
        if not results:
            return {}

        home_controls = [r.pitch_control.home_control_pct for r in results]
        away_controls = [r.pitch_control.away_control_pct for r in results]
        home_obso = [r.home_obso_total for r in results]
        away_obso = [r.away_obso_total for r in results]

        return {
            "num_frames": len(results),
            "pitch_control": {
                "avg_home_pct": round(float(np.mean(home_controls)), 2),
                "avg_away_pct": round(float(np.mean(away_controls)), 2),
                "std_home_pct": round(float(np.std(home_controls)), 2),
                "std_away_pct": round(float(np.std(away_controls)), 2),
            },
            "obso": {
                "total_home": round(float(np.sum(home_obso)), 4),
                "total_away": round(float(np.sum(away_obso)), 4),
                "avg_home_per_frame": round(float(np.mean(home_obso)), 4),
                "avg_away_per_frame": round(float(np.mean(away_obso)), 4),
            },
        }
