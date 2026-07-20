import pytest

pytest.importorskip("torch")

from src.graph import build_track_graph, build_track_graph_optimized
from src.track.bytetrack_runtime import Tracklet, Tracklets


def _make_tracklets():
    frame0 = Tracklets(
        frame_id=0,
        items=[
            Tracklet(track_id=0, bbox=[0, 0, 10, 10], score=0.9),
            Tracklet(track_id=1, bbox=[10, 0, 20, 10], score=0.8),
        ],
    )
    frame1 = Tracklets(
        frame_id=1,
        items=[
            Tracklet(track_id=0, bbox=[1, 1, 11, 11], score=0.92),
            Tracklet(track_id=1, bbox=[11, 1, 21, 11], score=0.85),
        ],
    )
    return [frame0, frame1]


def test_build_track_graph_links_consecutive_frames():
    graph_obj = build_track_graph(_make_tracklets(), window=5)

    if isinstance(graph_obj, dict):
        edge_index = graph_obj["edge_index"]
        assert edge_index.shape[1] >= 2
    else:
        assert graph_obj.edge_index.shape[1] >= 2


def test_build_track_graph_truncation_does_not_reference_missing_nodes():
    first_frame = Tracklets(
        frame_id=0,
        items=[Tracklet(track_id=i, bbox=[0, 0, 1, 1], score=0.9) for i in range(1000)],
    )
    truncated_frame = Tracklets(
        frame_id=1,
        items=[Tracklet(track_id=0, bbox=[1, 1, 2, 2], score=0.9)],
    )

    graph_obj = build_track_graph([first_frame, truncated_frame], distance_threshold=0)

    assert graph_obj.x.shape == (1000, 4)


def test_optimized_graph_ignores_truncated_track_history():
    frames = [
        Tracklets(
            frame_id=0,
            items=[
                Tracklet(track_id=0, bbox=[0, 0, 1, 1], score=0.9),
                Tracklet(track_id=1, bbox=[2, 2, 3, 3], score=0.9),
            ],
        ),
        Tracklets(
            frame_id=1,
            items=[
                Tracklet(track_id=0, bbox=[1, 1, 2, 2], score=0.9),
                Tracklet(track_id=1, bbox=[3, 3, 4, 4], score=0.9),
            ],
        ),
        Tracklets(
            frame_id=2,
            items=[
                Tracklet(track_id=1, bbox=[4, 4, 5, 5], score=0.9),
                Tracklet(track_id=0, bbox=[2, 2, 3, 3], score=0.9),
            ],
        ),
    ]

    graph_obj = build_track_graph_optimized(frames, max_nodes_per_frame=1, distance_threshold=0)

    assert graph_obj.x.shape == (3, 4)
    assert graph_obj.edge_index.dtype.is_floating_point is False
