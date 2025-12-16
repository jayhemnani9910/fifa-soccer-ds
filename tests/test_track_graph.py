import pytest

from src.detect import InferenceConfig
from src.graph import build_track_graph
from src.track.bytetrack_runtime import ByteTrackRuntime
from src.track.pipeline import TrackingPipelineConfig, run_tracking_pipeline

pytest.importorskip("torch")


class DummyTensor:
    def __init__(self, values):
        self._values = values

    def cpu(self):
        return self

    def detach(self):
        return self

    def numpy(self):
        return self

    def tolist(self):
        return self._values


class DummyBoxes:
    def __init__(self, detections):
        self.xyxy = DummyTensor([det["bbox"] for det in detections])
        self.conf = DummyTensor([det.get("score", 0.0) for det in detections])
        self.cls = DummyTensor([det.get("class_id", 0) for det in detections])


class DummyResult:
    def __init__(self, detections):
        self.boxes = DummyBoxes(detections)
        self.names = {0: "player", 1: "ball"}
        self._detections = detections

    def plot(self, boxes: bool = True):
        return None


class DummyModel:
    def __init__(self, frames):
        self._frames = frames

    def predict(self, source, stream: bool = True, conf: float = 0.25, verbose: bool = False):
        assert stream, "Tracking pipeline expects streaming predictions."
        for frame in self._frames:
            yield DummyResult(frame)


def test_tracker_outputs_edges_for_close_tracks():
    tracker = ByteTrackRuntime(min_confidence=0.0, distance_threshold=30.0, max_age=2)
    frames = [
        [
            {"bbox": [0, 0, 10, 10], "score": 0.9},
            {"bbox": [22, 0, 32, 10], "score": 0.9},
        ],
        [
            {"bbox": [1, 1, 11, 11], "score": 0.92},
            {"bbox": [23, 1, 33, 11], "score": 0.88},
        ],
    ]

    history = []
    for frame_id, detections in enumerate(frames):
        tracklets = tracker.update(frame_id=frame_id, detections=detections)
        history.append(tracklets)

    graph = build_track_graph(history, window=3, distance_threshold=40.0)
    edge_index = graph["edge_index"] if isinstance(graph, dict) else graph.edge_index

    assert edge_index.shape[1] >= 2, "Expected spatial edges between nearby tracklets."


def test_tracking_pipeline_integrates_detection_and_tracking(tmp_path, monkeypatch):
    frames = [
        [
            {"bbox": [0, 0, 8, 8], "score": 0.95, "class_id": 0},
            {"bbox": [40, 40, 52, 52], "score": 0.9, "class_id": 1},
        ],
        [
            {"bbox": [1, 1, 9, 9], "score": 0.96, "class_id": 0},
            {"bbox": [41, 41, 53, 53], "score": 0.91, "class_id": 1},
        ],
    ]

    dummy_model = DummyModel(frames)
    # Patch load_model in the pipeline module where it's imported
    from src.track import pipeline

    monkeypatch.setattr(pipeline, "load_model", lambda cfg: dummy_model)

    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"\x00\x00")

    config = TrackingPipelineConfig(
        detector=InferenceConfig(weights="dummy.pt", device="cpu", confidence=0.0),
        min_confidence=0.0,
        distance_threshold=50.0,
        max_age=2,
        max_frames=5,
    )

    history = run_tracking_pipeline(video_path, config=config)

    assert len(history) == len(frames)
    assert all(len(frame.items) == 2 for frame in history)

    graph = build_track_graph(history, window=5, distance_threshold=60.0)
    edge_index = graph["edge_index"] if isinstance(graph, dict) else graph.edge_index
    assert edge_index.shape[1] > 0
