from __future__ import annotations

import json
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from src.classify.soccer_classifier import SoccerClassifier
from src.youtube.metadata_parser import YouTubeMetadataParser


def _upstream_metadata() -> dict[str, object]:
    return {
        "id": "dQw4w9WgXcQ",
        "title": "Football highlights",
        "description": "Match recap",
        "duration": 120,
        "uploader": "Example Sports",
        "channel_id": "channel-1",
        "view_count": 123,
        "like_count": 4,
        "comment_count": 2,
        "tags": ["football"],
        "categories": ["Sports"],
        "upload_date": "20260717",
        "resolution": "1280x720",
        "fps": 30.0,
        "acodec": "aac",
        "vcodec": "h264",
    }


def test_metadata_parser_preserves_upstream_identity_and_nullable_fields() -> None:
    parser = YouTubeMetadataParser()

    metadata = parser._create_validated_metadata(_upstream_metadata())

    assert metadata.video_id == "dQw4w9WgXcQ"
    assert metadata.channel_title == "Example Sports"
    assert metadata.duration_seconds == 120
    assert metadata.view_count == 123
    assert metadata.publish_date is not None
    assert metadata.publish_date.isoformat() == "2026-07-17T00:00:00+00:00"


def test_metadata_parser_does_not_fabricate_an_identity() -> None:
    parser = YouTubeMetadataParser()
    upstream = _upstream_metadata()
    upstream["id"] = None

    with pytest.raises(ValidationError):
        parser._create_validated_metadata(upstream)


def test_classifier_maps_validated_metadata_to_pipeline_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = YouTubeMetadataParser()
    validated = parser._create_validated_metadata(_upstream_metadata()).model_dump(mode="python")
    validated.update(
        {
            "analysis": {
                "soccer_score": 1.0,
                "soccer_relevance": "high",
                "content_type": "highlights",
                "channel_analysis": {"channel_score": 1},
            },
            "extraction_timestamp": "2026-07-18T00:00:00+00:00",
        }
    )
    metadata_parser = Mock()
    metadata_parser.extract_metadata.return_value = validated
    metadata_parser.predict_soccer_content.return_value = {
        "is_soccer": True,
        "confidence": 0.9,
        "relevance_level": "high",
        "content_type": "highlights",
        "reasoning": [],
    }
    classifier = SoccerClassifier()
    classifier.metadata_parser = metadata_parser
    monkeypatch.setattr(
        classifier,
        "_analyze_thumbnail",
        lambda _url: {"visual_score": 0.9, "soccer_elements": [], "confidence": 0.9},
    )

    result = classifier.classify_youtube_content("https://youtu.be/dQw4w9WgXcQ")

    video_info = result["processing_info"]["video_info"]
    assert video_info["id"] == "dQw4w9WgXcQ"
    assert video_info["duration"] == 120
    assert video_info["uploader"] == "Example Sports"
    assert video_info["upload_date"] == "2026-07-17T00:00:00+00:00"
    assert json.loads(json.dumps(result))["processing_info"]["video_info"] == video_info
