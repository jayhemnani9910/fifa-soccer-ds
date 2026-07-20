from __future__ import annotations

import pytest

from src.analytics.team_classifier import JerseyColorClassifier


def test_insufficient_color_evidence_does_not_invent_team() -> None:
    classifier = JerseyColorClassifier(min_samples=5)

    assignments = classifier.classify_tracks(
        frames={},
        tracklets=[{"track_id": 7, "frame_id": 0, "bbox": [0, 0, 10, 10]}],
    )

    assert assignments[7].team_id == -1
    assert assignments[7].confidence == 0.0


@pytest.mark.parametrize("argument", ["n_teams", "min_samples"])
def test_classifier_rejects_non_positive_configuration(argument: str) -> None:
    kwargs = {argument: 0}
    with pytest.raises(ValueError, match="must be positive"):
        JerseyColorClassifier(**kwargs)
