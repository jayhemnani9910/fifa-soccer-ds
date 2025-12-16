"""Pitch Control Analysis Module.

Implements spatial dominance analysis for soccer using a physics-based
time-to-intercept model. The algorithm computes the probability that each
team controls each area of the pitch based on player positions and velocities.

Based on: Spearman (2018) "Beyond Expected Goals" and
Fernandez & Bornn (2018) "Wide Open Spaces: A statistical technique
for measuring space creation in professional soccer"

Ported from jhsoccer project with adaptations for fifa-soccer-ds integration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

# Physical constants for pitch control calculation
DEFAULT_MAX_SPEED = 5.0  # m/s - maximum player running speed
DEFAULT_REACTION_TIME = 0.7  # seconds - time before player starts moving
DEFAULT_SIGMA = 0.45  # uncertainty parameter for arrival time
PITCH_LENGTH = 105.0  # meters (standard pitch)
PITCH_WIDTH = 68.0  # meters (standard pitch)


@dataclass(slots=True)
class PlayerState:
    """State of a player at a given frame."""
    player_id: int
    team_id: int  # 0 = home, 1 = away, -1 = unknown/ball
    position: np.ndarray  # (x, y) in meters or normalized coordinates
    velocity: Optional[np.ndarray] = None  # (vx, vy) in m/s


@dataclass(slots=True)
class PitchControlResult:
    """Result of pitch control computation for a frame."""
    frame_id: int
    grid: np.ndarray  # Shape (grid_y, grid_x) - values in [0, 1] for home team control
    home_control_pct: float = 0.0  # Percentage of pitch controlled by home team
    away_control_pct: float = 0.0  # Percentage controlled by away team


@dataclass(slots=True)
class PitchControlConfig:
    """Configuration for pitch control calculation."""
    max_speed: float = DEFAULT_MAX_SPEED
    reaction_time: float = DEFAULT_REACTION_TIME
    sigma: float = DEFAULT_SIGMA
    pitch_length: float = PITCH_LENGTH
    pitch_width: float = PITCH_WIDTH
    time_horizon: float = 5.0  # seconds - max time to consider for interception


class PitchControlModel:
    """Physics-based pitch control model using time-to-intercept.

    Computes spatial dominance by calculating the probability that each
    team can reach any point on the pitch first, based on player positions
    and velocities.

    Args:
        grid_shape: Tuple of (rows, cols) for the output grid.
        config: Optional configuration for physical parameters.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int] = (12, 16),
        config: Optional[PitchControlConfig] = None
    ) -> None:
        self.grid_shape = grid_shape
        self.config = config or PitchControlConfig()

        # Pre-compute grid coordinates (normalized 0-1)
        self._grid_y, self._grid_x = np.meshgrid(
            np.linspace(0, 1, grid_shape[0]),
            np.linspace(0, 1, grid_shape[1]),
            indexing='ij'
        )
        # Stack for vectorized operations: shape (grid_y, grid_x, 2)
        self._grid_coords = np.stack([self._grid_x, self._grid_y], axis=-1)

    def compute(
        self,
        frame_id: int,
        players: Optional[List[PlayerState]] = None
    ) -> PitchControlResult:
        """Compute pitch control for a frame.

        Args:
            frame_id: Frame identifier.
            players: List of player states. If None, returns uniform control.

        Returns:
            PitchControlResult with control probability grid.
        """
        if players is None or len(players) == 0:
            # No players - return neutral control (0.5 everywhere)
            grid = np.full(self.grid_shape, 0.5, dtype=np.float32)
            LOGGER.debug("No players provided for frame %s, returning neutral", frame_id)
            return PitchControlResult(frame_id, grid, 50.0, 50.0)

        # Separate players by team
        home_players = [p for p in players if p.team_id == 0]
        away_players = [p for p in players if p.team_id == 1]

        if not home_players or not away_players:
            # Only one team - that team controls everything
            if home_players:
                grid = np.ones(self.grid_shape, dtype=np.float32)
                return PitchControlResult(frame_id, grid, 100.0, 0.0)
            elif away_players:
                grid = np.zeros(self.grid_shape, dtype=np.float32)
                return PitchControlResult(frame_id, grid, 0.0, 100.0)
            else:
                grid = np.full(self.grid_shape, 0.5, dtype=np.float32)
                return PitchControlResult(frame_id, grid, 50.0, 50.0)

        # Compute control probability for each grid cell
        grid = self._compute_control_grid(home_players, away_players)

        # Calculate overall control percentages
        home_control_pct = float(np.mean(grid) * 100)
        away_control_pct = 100.0 - home_control_pct

        LOGGER.debug(
            "Computed pitch control for frame %s: home=%.1f%%, away=%.1f%%",
            frame_id, home_control_pct, away_control_pct
        )

        return PitchControlResult(frame_id, grid, home_control_pct, away_control_pct)

    def _compute_control_grid(
        self,
        home_players: List[PlayerState],
        away_players: List[PlayerState]
    ) -> np.ndarray:
        """Compute control probability grid using time-to-intercept model.

        For each grid cell, calculates the probability that the home team
        arrives first based on minimum arrival times.
        """
        # Compute arrival times for all players to all grid points
        home_times = self._compute_team_arrival_times(home_players)
        away_times = self._compute_team_arrival_times(away_players)

        # Get minimum arrival time for each team at each grid point
        home_min_time = np.min(home_times, axis=0)
        away_min_time = np.min(away_times, axis=0)

        # Compute control probability using sigmoid-like function
        # Based on difference in arrival times
        time_diff = away_min_time - home_min_time  # positive = home arrives first

        # Use logistic function for smooth probability transition
        # Scale factor controls sharpness of transition
        scale = self.config.sigma * 2
        control_prob = 1.0 / (1.0 + np.exp(-time_diff / scale))

        return control_prob.astype(np.float32)

    def _compute_team_arrival_times(
        self,
        players: List[PlayerState]
    ) -> np.ndarray:
        """Compute arrival times for all players to all grid points.

        Returns:
            Array of shape (num_players, grid_y, grid_x) with arrival times.
        """
        arrival_times = []

        for player in players:
            time_grid = self._compute_player_arrival_time(player)
            arrival_times.append(time_grid)

        return np.stack(arrival_times, axis=0)

    def _compute_player_arrival_time(self, player: PlayerState) -> np.ndarray:
        """Compute time for a player to reach each grid point.

        Uses physics-based model accounting for:
        - Current position
        - Current velocity (if available)
        - Reaction time
        - Maximum running speed
        """
        # Normalize player position to [0, 1] if needed
        pos = np.asarray(player.position)
        if np.max(pos) > 1.0:
            # Assume position is in meters, normalize
            pos = pos / np.array([self.config.pitch_length, self.config.pitch_width])

        # Compute distance from player to each grid point
        # grid_coords is shape (grid_y, grid_x, 2)
        diff = self._grid_coords - pos  # Broadcasting: (grid_y, grid_x, 2)

        # Scale to meters for physical calculation
        diff_meters = diff * np.array([self.config.pitch_length, self.config.pitch_width])
        distance = np.linalg.norm(diff_meters, axis=-1)  # (grid_y, grid_x)

        # Simple time-to-intercept model
        # Time = reaction_time + distance / max_speed
        # With velocity adjustment if available
        if player.velocity is not None and np.linalg.norm(player.velocity) > 0.01:
            vel = np.asarray(player.velocity)
            # Project velocity onto direction to target
            direction = diff_meters / (distance[..., np.newaxis] + 1e-8)
            vel_component = np.sum(direction * vel, axis=-1)

            # Effective speed is max_speed adjusted by current velocity direction
            effective_speed = np.maximum(
                self.config.max_speed * 0.5,  # minimum speed
                self.config.max_speed + vel_component * 0.3  # velocity boost
            )
            time_to_reach = self.config.reaction_time + distance / effective_speed
        else:
            # No velocity info - use constant speed
            time_to_reach = self.config.reaction_time + distance / self.config.max_speed

        # Clip to time horizon
        time_to_reach = np.clip(time_to_reach, 0, self.config.time_horizon)

        return time_to_reach


__all__ = [
    "PitchControlModel",
    "PitchControlResult",
    "PitchControlConfig",
    "PlayerState"
]
