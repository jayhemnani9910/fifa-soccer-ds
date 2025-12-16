"""Smoke tests for the full pipeline driver."""

import json
from pathlib import Path

import pytest

from src.pipeline_full import PipelineConfig, process_frames_directory


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
    def __init__(self):
        self.xyxy = DummyTensor([[10.0, 10.0, 50.0, 50.0]])
        self.conf = DummyTensor([0.85])
        self.cls = DummyTensor([0])


class DummyResult:
    def __init__(self):
        self.boxes = DummyBoxes()
        self.names = {0: "player"}


class DummyYOLO:
    def __init__(self, weights: str):
        self.weights = weights
        self.device = None

    def to(self, device: str):
        self.device = device
        return self

    def predict(self, source, conf: float = 0.25, verbose: bool = False):
        return [DummyResult()]


@pytest.mark.smoke
def test_pipeline_processes_single_frame(tmp_path: Path, monkeypatch):
    """Test pipeline runs end-to-end with minimal input."""
    import cv2
    import numpy as np

    import src.detect.infer as detect_infer

    monkeypatch.setattr(detect_infer, "YOLO", DummyYOLO)

    # Create fake frame
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_path = frames_dir / "frame_000000.jpg"
    cv2.imwrite(frame_path.as_posix(), frame)

    # Run pipeline
    output_dir = tmp_path / "output"
    config = PipelineConfig(max_frames=1)
    summary = process_frames_directory(frames_dir, output_dir, config)

    # Verify outputs
    assert summary.total_frames == 1
    assert summary.total_detections == 1
    assert len(summary.unique_track_ids) > 0

    # Check output structure
    assert (output_dir / "pipeline_summary.json").exists()
    assert (output_dir / "detections").exists()
    assert (output_dir / "tracks").exists()
    assert (output_dir / "overlays").exists()
    assert (output_dir / "graphs").exists()

    # Verify summary JSON
    with (output_dir / "pipeline_summary.json").open() as f:
        data = json.load(f)
        assert data["total_frames"] == 1
        assert data["total_detections"] == 1
        assert "config" in data
