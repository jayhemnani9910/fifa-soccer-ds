"""Analytics module for tactical soccer analysis.

This module provides:
- Team classification based on jersey colors
- Pitch control computation
- Expected threat (xT) analysis
- Off-ball scoring opportunities (OBSO)
"""

from src.analytics.tactical import (
    TacticalAnalyzer,
    TacticalConfig,
    PlayerState,
    PitchControlResult,
)
from src.analytics.team_classifier import (
    JerseyColorClassifier,
    TeamAssignment,
)

__all__ = [
    "TacticalAnalyzer",
    "TacticalConfig",
    "PlayerState",
    "PitchControlResult",
    "JerseyColorClassifier",
    "TeamAssignment",
]
