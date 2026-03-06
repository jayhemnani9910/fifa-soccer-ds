import json
from pathlib import Path

import pytest

from src.detect import InferenceConfig
from src.detect import infer as detect_infer
from src.detect.yolo_lora_adapter import YOLOLoRAAdapter


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
        self.xyxy = DummyTensor([[0.0, 0.0, 10.0, 10.0]])
        self.conf = DummyTensor([0.9])
        self.cls = DummyTensor([0])


class DummyResult:
    def __init__(self):
        self.boxes = DummyBoxes()
        self.names = {0: "player"}

    def plot(self, boxes: bool = True):
        return None


class DummyYOLO:
    def __init__(self, weights: str):
        self.weights = weights
        self.device = None

    def to(self, device: str):
        self.device = device
        return self

    def predict(
        self, source, conf: float = 0.25, verbose: bool = False, stream: bool = False, **kwargs
    ):
        if stream:
            return (result for result in [DummyResult()])
        return [DummyResult()]


class DummyYOLOForExport:
    def __init__(self):
        from torch import nn

        self.model = nn.Sequential(nn.Flatten(), nn.Linear(16, 4))
        self._device = "cpu"

    def to(self, device: str):
        self._device = device
        return self

    def export(self, format: str, path: str, **kwargs):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"format": format, "device": self._device}))
        return path

    def loss(self, batch):
        import torch

        return torch.tensor(0.1, requires_grad=True)


@pytest.mark.smoke
def test_run_image_detection_emits_detections(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(detect_infer, "YOLO", DummyYOLO)
    config = InferenceConfig(weights="dummy.pt", device="cpu")
    model = detect_infer.load_model(config)

    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"\x00")

    output_dir = tmp_path / "detect"
    summary = detect_infer.run_image_detection(model, image_path, output_dir, config)

    assert summary["detections"], "Expected at least one detection in the summary."

    json_path: Path = summary["json"]
    assert json_path.exists(), "Detection JSON should be written to disk."
    payload = json.loads(json_path.read_text())
    assert payload["detections"], "Detection JSON should contain detections."


def test_yolo_lora_export_onnx(tmp_path: Path):
    adapter = YOLOLoRAAdapter(model=DummyYOLOForExport(), weights="dummy.pt")
    onnx_path = adapter.export_to_onnx(output_dir=tmp_path)

    assert onnx_path.exists()
    assert onnx_path.suffix == ".onnx"
    contents = onnx_path.read_text().strip()
    assert "onnx" in contents


def test_tensorrt_engine_loads(tmp_path: Path):
    adapter = YOLOLoRAAdapter(model=DummyYOLOForExport(), weights="dummy.pt")
    engine_path = adapter.export_to_tensorrt(output_dir=tmp_path, workspace_size=1 << 20)

    assert engine_path.exists()
    assert engine_path.suffix == ".engine"
    contents = engine_path.read_text().strip()
    assert "tensorrt" in contents
