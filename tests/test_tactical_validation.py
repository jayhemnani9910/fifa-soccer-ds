from __future__ import annotations

import numpy as np
import pytest

from src.analytics._pitch_control import PitchControlConfig, PitchControlModel, PlayerState
from src.analytics.tactical import TacticalAnalyzer


def test_pitch_control_rejects_invalid_configuration_and_positions() -> None:
    with pytest.raises(ValueError, match="positive finite"):
        PitchControlConfig(max_speed=0)

    model = PitchControlModel(grid_shape=(4, 6))
    players = [
        PlayerState(player_id=1, team_id=0, position=np.array([-0.1, 0.5])),
        PlayerState(player_id=2, team_id=1, position=np.array([0.8, 0.5])),
    ]
    with pytest.raises(ValueError, match="outside"):
        model.compute(0, players)


def test_tracklet_conversion_never_invents_team_from_position() -> None:
    analyzer = TacticalAnalyzer()
    with pytest.raises(ValueError, match="both teams"):
        analyzer.compute_from_tracklets(
            frame_id=0,
            tracklets_data=[{"track_id": 1, "bbox": [0, 0, 10, 10], "class_name": "person"}],
            team_assignments={},
            image_shape=(100, 100),
        )


def test_pitch_control_never_infers_complete_control_from_one_team() -> None:
    model = PitchControlModel(grid_shape=(4, 6))
    players = [PlayerState(player_id=1, team_id=0, position=np.array([0.5, 0.5]))]

    with pytest.raises(ValueError, match="both teams"):
        model.compute(0, players)
