"""Tactical Analysis Module.

Provides unified interface for experimental tactical soccer analysis including:
- Pitch control computation
- A static, heuristic attacking-value surface (xT-like)
- Off-ball opportunity scores derived from those two surfaces
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.analytics._pitch_control import (
    PitchControlConfig,
    PitchControlModel,
    PitchControlResult,
    PlayerState,
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

    grid_shape: tuple[int, int] = (12, 16)  # (rows, cols)
    max_speed: float = 5.0  # m/s
    reaction_time: float = 0.7  # seconds
    pitch_length: float = 105.0  # meters
    pitch_width: float = 68.0  # meters
    time_horizon: float = 5.0  # seconds
    enable_xT: bool = True
    enable_obso: bool = True

    def __post_init__(self) -> None:
        if len(self.grid_shape) != 2 or any(size < 2 or size > 512 for size in self.grid_shape):
            raise ValueError("grid_shape dimensions must each be between 2 and 512")


@dataclass
class TacticalResult:
    """Complete tactical analysis result for a frame."""

    frame_id: int
    pitch_control: PitchControlResult
    xT_grid: np.ndarray | None = None
    obso_grid: np.ndarray | None = None
    home_obso_total: float = 0.0
    away_obso_total: float = 0.0


class TacticalAnalyzer:
    """Unified tactical analysis combining PitchControl, xT, and OBSO.

    Provides a single interface for computing all tactical metrics
    for a given frame with player positions.

    Args:
        config: Configuration for tactical analysis.
    """

    def __init__(self, config: TacticalConfig | None = None) -> None:
        self.config = config or TacticalConfig()

        # Initialize pitch control model
        pc_config = PitchControlConfig(
            max_speed=self.config.max_speed,
            reaction_time=self.config.reaction_time,
            pitch_length=self.config.pitch_length,
            pitch_width=self.config.pitch_width,
            time_horizon=self.config.time_horizon,
        )
        self.pitch_control = PitchControlModel(grid_shape=self.config.grid_shape, config=pc_config)

        # Initialize xT grid (static expected threat values)
        self.xT_grid = self._create_default_xT_grid()

        LOGGER.info("Initialized TacticalAnalyzer with grid shape %s", self.config.grid_shape)

    def _create_default_xT_grid(self) -> np.ndarray:
        """Create a static heuristic attacking-value grid.

        This is not a learned or calibrated expected-threat model. It is a
        deterministic spatial prior that increases toward the opponent's goal.
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
                base_xT = x_norm**3 * 0.25

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

    def compute(self, frame_id: int, players: list[PlayerState] | None = None) -> TacticalResult:
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
            obso_home = None
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
        self, pitch_control_grid: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Compute an experimental off-ball opportunity surface.

        OBSO = Pitch Control * xT

        The result is not calibrated as an expected-goals probability.
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
        tracklets_data: list[dict[str, Any]],
        team_assignments: dict[int, int] | None = None,
        image_shape: tuple[int, int] | None = None,
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
        if image_shape is None or any(dimension <= 0 for dimension in image_shape):
            raise ValueError("image_shape with positive height and width is required")

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

            height, width = image_shape
            pos_x = pos_x / width
            pos_y = pos_y / height

            # Get team assignment
            if not team_assignments or team_assignments.get(track_id) not in {0, 1}:
                continue
            team_id = team_assignments[track_id]

            players.append(
                PlayerState(
                    player_id=track_id,
                    team_id=team_id,
                    position=np.array([pos_x, pos_y]),
                    velocity=None,
                )
            )

        return self.compute(frame_id, players)

    def to_dict(self, result: TacticalResult) -> dict[str, Any]:
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

    def get_aggregate_stats(self, results: list[TacticalResult]) -> dict[str, Any]:
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
