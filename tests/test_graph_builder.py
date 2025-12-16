import pytest

pytest.importorskip("torch")

from src.graph import build_track_graph
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
