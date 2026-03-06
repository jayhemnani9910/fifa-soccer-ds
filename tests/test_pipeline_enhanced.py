"""Enhanced integration tests with parameterized configurations."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.pipeline_full import PipelineConfig, process_frames_directory


class DummyTensor:
    """Mock tensor supporting CPU/detach/numpy conversions."""

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
    """Mock YOLO detection boxes."""

    def __init__(self, detections=None):
        if detections is None:
            detections = [[10.0, 10.0, 50.0, 50.0]]
        self.xyxy = DummyTensor(detections)
        self.conf = DummyTensor([0.85] * len(detections))
        self.cls = DummyTensor([0] * len(detections))


class DummyResult:
    """Mock YOLO result object."""

    def __init__(self, detections=None):
        self.boxes = DummyBoxes(detections)
        self.names = {0: "player", 1: "ball"}

    def plot(self, boxes: bool = True):
        return None


class DummyYOLO:
    """Mock YOLO detector."""

    def __init__(self, weights: str):
        self.weights = weights
        self.device = None

    def to(self, device: str):
        self.device = device
        return self

    def predict(self, source, conf: float = 0.25, verbose: bool = False):
        return [DummyResult()]


@pytest.mark.smoke
@pytest.mark.parametrize(
    "num_frames,num_detections_per_frame",
    [
        (1, 1),
        (5, 2),
        (10, 3),
    ],
)
def test_pipeline_with_multiple_frames(
    tmp_path: Path, monkeypatch, num_frames: int, num_detections_per_frame: int
):
    """Test pipeline with varying number of frames and detections."""
    import src.detect.infer as detect_infer

    monkeypatch.setattr(detect_infer, "YOLO", DummyYOLO)

    # Create synthetic frames
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(num_frames):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_path = frames_dir / f"frame_{i:06d}.jpg"
        cv2.imwrite(frame_path.as_posix(), frame)

    output_dir = tmp_path / "output"
    config = PipelineConfig(max_frames=num_frames)
    summary = process_frames_directory(frames_dir, output_dir, config)

    assert summary.total_frames == num_frames
    assert (output_dir / "pipeline_summary.json").exists()

    # Verify summary JSON structure
    with (output_dir / "pipeline_summary.json").open() as f:
        data = json.load(f)
        assert data["total_frames"] == num_frames
        assert "config" in data
        assert "unique_track_ids" in data


@pytest.mark.smoke
@pytest.mark.parametrize(
    "distance_threshold,expected_matches",
    [
        (20.0, 0),  # Very tight threshold
        (100.0, 1),  # Loose threshold
    ],
)
def test_pipeline_with_tracking_parameters(
    tmp_path: Path, monkeypatch, distance_threshold: float, expected_matches: int
):
    """Test pipeline with different tracking thresholds."""
    import src.detect.infer as detect_infer

    monkeypatch.setattr(detect_infer, "YOLO", DummyYOLO)

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(3):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite((frames_dir / f"frame_{i:06d}.jpg").as_posix(), frame)

    output_dir = tmp_path / "output"
    config = PipelineConfig(
        max_frames=3,
        distance_threshold=distance_threshold,
    )
    summary = process_frames_directory(frames_dir, output_dir, config)

    assert summary.total_frames == 3
    assert (output_dir / "pipeline_summary.json").exists()


@pytest.mark.smoke
def test_pipeline_output_directories_created(tmp_path: Path, monkeypatch):
    """Test that all required output directories are created."""
    import src.detect.infer as detect_infer

    monkeypatch.setattr(detect_infer, "YOLO", DummyYOLO)

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite((frames_dir / "frame_000000.jpg").as_posix(), frame)

    output_dir = tmp_path / "output"
    config = PipelineConfig(max_frames=1)
    process_frames_directory(frames_dir, output_dir, config)

    # Verify all output directories exist
    assert (output_dir / "detections").exists()
    assert (output_dir / "tracks").exists()
    assert (output_dir / "overlays").exists()
    assert (output_dir / "graphs").exists()

    # Verify files were created
    assert list((output_dir / "detections").glob("*.json"))
    assert list((output_dir / "tracks").glob("*.json"))


@pytest.mark.smoke
def test_pipeline_graph_construction(tmp_path: Path, monkeypatch):
    """Test that spatial-temporal graph is correctly constructed."""
    import src.detect.infer as detect_infer

    monkeypatch.setattr(detect_infer, "YOLO", DummyYOLO)

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(5):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite((frames_dir / f"frame_{i:06d}.jpg").as_posix(), frame)

    output_dir = tmp_path / "output"
    config = PipelineConfig(max_frames=5, graph_window=5)
    process_frames_directory(frames_dir, output_dir, config)

    # Verify graph metadata
    graph_path = output_dir / "graphs" / "final_graph.json"
    assert graph_path.exists()

    with graph_path.open() as f:
        graph_data = json.load(f)
        assert "num_nodes" in graph_data
        assert "num_edges" in graph_data
        assert "unique_track_ids" in graph_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
