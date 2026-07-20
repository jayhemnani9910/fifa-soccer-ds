import pytest

from src.track.bytetrack_runtime import ByteTrackRuntime


def test_tracker_assigns_incremental_ids():
    tracker = ByteTrackRuntime(min_confidence=0.0)
    detections = [
        {"bbox": [0, 0, 10, 10], "score": 0.9},
        {"bbox": [1, 1, 11, 11], "score": 0.8},
    ]
    tracklets = tracker.update(frame_id=0, detections=detections)

    ids = [track.track_id for track in tracklets.items]
    assert ids == [0, 1]


def test_tracker_does_not_match_different_object_classes():
    tracker = ByteTrackRuntime(min_confidence=0.0, distance_threshold=20)
    first = tracker.update(
        frame_id=0,
        detections=[{"bbox": [0, 0, 10, 10], "score": 0.9, "class_id": 0}],
    )
    second = tracker.update(
        frame_id=1,
        detections=[{"bbox": [0, 0, 10, 10], "score": 0.9, "class_id": 32}],
    )

    assert first.items[0].track_id == 0
    assert second.items[0].track_id == 1


def test_tracker_never_reuses_an_active_id_on_capacity_exhaustion():
    tracker = ByteTrackRuntime(
        min_confidence=0.0,
        distance_threshold=20,
        max_track_id=1,
        id_reuse_delay=30,
    )
    tracker.update(
        frame_id=0,
        detections=[{"bbox": [0, 0, 10, 10], "score": 0.9, "class_id": 0}],
    )

    with pytest.raises(RuntimeError, match="capacity exhausted"):
        tracker.update(
            frame_id=1,
            detections=[
                {"bbox": [1, 1, 11, 11], "score": 0.9, "class_id": 0},
                {"bbox": [100, 100, 110, 110], "score": 0.9, "class_id": 0},
            ],
        )


def test_tracker_rejects_non_monotonic_frames_and_nonfinite_boxes():
    tracker = ByteTrackRuntime(min_confidence=0.0)
    result = tracker.update(
        frame_id=2,
        detections=[{"bbox": [0, 0, float("inf"), 10], "score": 0.9}],
    )
    assert result.items == []

    with pytest.raises(ValueError, match="monotonically"):
        tracker.update(frame_id=1, detections=[])
