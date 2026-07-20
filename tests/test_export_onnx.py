from __future__ import annotations

import errno
from pathlib import Path

import pytest

from src.detect import export_onnx


class FakeYOLO:
    exported_path: Path
    export_kwargs: dict[str, object]

    def __init__(self, _weights: str) -> None:
        self.device = None

    def to(self, device: str) -> None:
        self.device = device

    def export(self, **kwargs):
        type(self).export_kwargs = kwargs
        self.exported_path.write_bytes(b"valid onnx artifact")
        return self.exported_path


def test_export_uses_supported_ultralytics_arguments_and_returned_path(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generated = tmp_path / "generated.onnx"
    target = tmp_path / "nested" / "model.onnx"
    FakeYOLO.exported_path = generated
    monkeypatch.setattr(export_onnx, "YOLO", FakeYOLO)
    validated: list[Path] = []
    monkeypatch.setattr(export_onnx, "validate_onnx_artifact", validated.append)

    result = export_onnx.export_to_onnx(
        export_onnx.OnnxExportConfig(weights="model.pt", output=str(target), device="cpu")
    )

    assert result == target
    assert target.read_bytes() == b"valid onnx artifact"
    assert not generated.exists()
    assert validated == [generated]
    assert "outfile" not in FakeYOLO.export_kwargs
    assert "imgsz" not in FakeYOLO.export_kwargs


def test_export_rejects_missing_artifact(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    class MissingArtifactYOLO(FakeYOLO):
        def export(self, **_kwargs):
            return tmp_path / "missing.onnx"

    monkeypatch.setattr(export_onnx, "YOLO", MissingArtifactYOLO)
    monkeypatch.setattr(
        export_onnx,
        "validate_onnx_artifact",
        lambda _artifact: pytest.fail("missing artifacts must not be validated"),
    )

    with pytest.raises(RuntimeError, match="non-empty ONNX"):
        export_onnx.export_to_onnx(
            export_onnx.OnnxExportConfig(output=str(tmp_path / "target.onnx"), device="cpu")
        )


def test_export_refuses_to_overwrite_requested_target(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generated = tmp_path / "generated.onnx"
    target = tmp_path / "target.onnx"
    target.write_bytes(b"user artifact")
    FakeYOLO.exported_path = generated
    monkeypatch.setattr(export_onnx, "YOLO", FakeYOLO)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        export_onnx.export_to_onnx(export_onnx.OnnxExportConfig(output=str(target), device="cpu"))

    assert target.read_bytes() == b"user artifact"


def test_export_refuses_to_overwrite_ultralytics_default_artifact(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = tmp_path / "model.pt"
    existing_export = tmp_path / "model.onnx"
    existing_export.write_bytes(b"user artifact")
    monkeypatch.setattr(export_onnx, "YOLO", FakeYOLO)

    with pytest.raises(FileExistsError, match="Ultralytics overwrite"):
        export_onnx.export_to_onnx(
            export_onnx.OnnxExportConfig(
                weights=str(weights),
                output=str(tmp_path / "elsewhere" / "model.onnx"),
                device="cpu",
            )
        )

    assert existing_export.read_bytes() == b"user artifact"


def test_export_publishes_across_filesystems_atomically(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generated = tmp_path / "generated.onnx"
    target = tmp_path / "destination" / "model.onnx"
    FakeYOLO.exported_path = generated
    monkeypatch.setattr(export_onnx, "YOLO", FakeYOLO)
    monkeypatch.setattr(export_onnx, "validate_onnx_artifact", lambda _artifact: None)
    real_replace = export_onnx.os.replace
    simulated_cross_device_failure = False

    def replace_with_cross_device_fallback(source, destination) -> None:
        nonlocal simulated_cross_device_failure
        if Path(source) == generated and Path(destination) == target:
            simulated_cross_device_failure = True
            raise OSError(errno.EXDEV, "simulated cross-device move")
        real_replace(source, destination)

    monkeypatch.setattr(export_onnx.os, "replace", replace_with_cross_device_fallback)

    result = export_onnx.export_to_onnx(
        export_onnx.OnnxExportConfig(output=str(target), device="cpu")
    )

    assert simulated_cross_device_failure is True
    assert result == target
    assert target.read_bytes() == b"valid onnx artifact"
    assert not generated.exists()
    assert list(target.parent.glob(f".{target.name}.*.tmp")) == []


def test_validate_onnx_checks_structure_and_cpu_runtime(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"generated model")
    checked: list[object] = []
    sessions: list[tuple[str, list[str]]] = []

    class FakeChecker:
        @staticmethod
        def check_model(model: object) -> None:
            checked.append(model)

    class FakeOnnx:
        checker = FakeChecker()

        @staticmethod
        def load(path: str, *, load_external_data: bool) -> object:
            assert path == str(artifact)
            assert load_external_data is False
            return object()

    class FakeSession:
        def __init__(self, path: str, *, providers: list[str]) -> None:
            sessions.append((path, providers))

        @staticmethod
        def get_inputs() -> list[object]:
            return [object()]

        @staticmethod
        def get_outputs() -> list[object]:
            return [object(), object()]

    class FakeOrt:
        InferenceSession = FakeSession

        @staticmethod
        def get_available_providers() -> list[str]:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    modules = {"onnx": FakeOnnx, "onnxruntime": FakeOrt}
    monkeypatch.setattr(
        export_onnx.importlib,
        "import_module",
        lambda name: modules[name],
    )

    export_onnx.validate_onnx_artifact(artifact)

    assert len(checked) == 1
    assert sessions == [(str(artifact), ["CPUExecutionProvider"])]


def test_validate_onnx_requires_export_dependencies(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"generated model")

    def missing_module(_name: str) -> object:
        raise ModuleNotFoundError("not installed")

    monkeypatch.setattr(export_onnx.importlib, "import_module", missing_module)

    with pytest.raises(ImportError, match=r"\[export\]"):
        export_onnx.validate_onnx_artifact(artifact)


def test_validate_onnx_rejects_runtime_graph_without_outputs(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"generated model")

    class FakeChecker:
        @staticmethod
        def check_model(_model: object) -> None:
            return None

    class FakeOnnx:
        checker = FakeChecker()

        @staticmethod
        def load(_path: str, *, load_external_data: bool) -> object:
            assert load_external_data is False
            return object()

    class EmptyOutputSession:
        def __init__(self, _path: str, *, providers: list[str]) -> None:
            assert providers == ["CPUExecutionProvider"]

        @staticmethod
        def get_inputs() -> list[object]:
            return [object()]

        @staticmethod
        def get_outputs() -> list[object]:
            return []

    class FakeOrt:
        InferenceSession = EmptyOutputSession

        @staticmethod
        def get_available_providers() -> list[str]:
            return ["CPUExecutionProvider"]

    modules = {"onnx": FakeOnnx, "onnxruntime": FakeOrt}
    monkeypatch.setattr(
        export_onnx.importlib,
        "import_module",
        lambda name: modules[name],
    )

    with pytest.raises(RuntimeError, match="without graph inputs or outputs"):
        export_onnx.validate_onnx_artifact(artifact)
