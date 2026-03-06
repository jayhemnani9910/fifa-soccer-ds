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
