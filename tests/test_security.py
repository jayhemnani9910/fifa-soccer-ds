"""Security regression tests for frame ingestion and pipeline error boundaries."""

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from src.pipeline_full import PipelineConfig, process_frames_directory


def _write_jpeg(path: Path, shape: tuple[int, int] = (16, 16)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    path.write_bytes(encoded.tobytes())


@pytest.fixture()
def detector(monkeypatch: pytest.MonkeyPatch) -> Mock:
    fake = Mock(predict=Mock(return_value=[]))
    monkeypatch.setattr("src.pipeline_full.load_model", lambda _config: fake)
    monkeypatch.setattr("src.pipeline_full.mlflow", None)
    return fake


def _config(**overrides) -> PipelineConfig:  # type: ignore[no-untyped-def]
    return PipelineConfig(enable_tactical_analytics=False, **overrides)


class TestInputValidation:
    def test_only_numbered_jpeg_frames_are_processed(self, tmp_path: Path, detector: Mock) -> None:
        frames_dir = tmp_path / "frames"
        valid = frames_dir / "frame_001.jpg"
        _write_jpeg(valid)
        _write_jpeg(frames_dir / "frame_002.png")
        (frames_dir / "malicious.exe").write_bytes(b"MZ")
        (frames_dir / "frame_003.jpg.txt").write_text("not an image", encoding="utf-8")

        with patch("src.pipeline_full.cv2.imread", wraps=cv2.imread) as imread:
            summary = process_frames_directory(frames_dir, tmp_path / "output", _config())

        assert summary.total_frames == 1
        assert detector.predict.call_count == 1
        imread.assert_called_once_with(valid.as_posix())

    @pytest.mark.parametrize(
        "filename",
        ["frame_001;rm.jpg", "frame_secret.jpg", "frame_001.jpg.exe", ".frame_001.jpg"],
    )
    def test_unexpected_frame_names_are_rejected(self, tmp_path: Path, filename: str) -> None:
        _write_jpeg(tmp_path / filename)

        with pytest.raises(ValueError, match="No frame files found"):
            process_frames_directory(tmp_path, tmp_path / "output", _config())

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("confidence", float("inf")),
            ("confidence", float("nan")),
            ("min_confidence", -0.1),
            ("distance_threshold", float("inf")),
            ("graph_distance_threshold", 0.0),
            ("max_bbox_area_ratio", 1.1),
            ("nms_iou", -0.1),
            ("max_frames", -1),
            ("max_frame_bytes", 0),
            ("max_frame_pixels", 0),
        ],
    )
    def test_invalid_numeric_configuration_is_rejected(self, field: str, value: float) -> None:
        with pytest.raises(ValueError, match=field):
            PipelineConfig(**{field: value})

    def test_encoded_size_limit_is_checked_before_decode(
        self, tmp_path: Path, detector: Mock
    ) -> None:
        frame = tmp_path / "frame_001.jpg"
        with frame.open("wb") as handle:
            handle.truncate(1025)

        with (
            patch("src.pipeline_full.cv2.imread") as imread,
            pytest.raises(RuntimeError, match="No readable frames"),
        ):
            process_frames_directory(
                tmp_path,
                tmp_path / "output",
                _config(max_frame_bytes=1024),
            )

        imread.assert_not_called()
        detector.predict.assert_not_called()

    def test_pixel_limit_is_checked_before_decode(self, tmp_path: Path, detector: Mock) -> None:
        _write_jpeg(tmp_path / "frame_001.jpg", shape=(20, 20))

        with (
            patch("src.pipeline_full.cv2.imread") as imread,
            pytest.raises(RuntimeError, match="No readable frames"),
        ):
            process_frames_directory(
                tmp_path,
                tmp_path / "output",
                _config(max_frame_pixels=399),
            )

        imread.assert_not_called()
        detector.predict.assert_not_called()

    def test_symlinked_frames_are_not_followed(self, tmp_path: Path) -> None:
        source = tmp_path / "outside.jpg"
        _write_jpeg(source)
        frames = tmp_path / "frames"
        frames.mkdir()
        (frames / "frame_001.jpg").symlink_to(source)

        with pytest.raises(ValueError, match="No frame files found"):
            process_frames_directory(frames, tmp_path / "output", _config())


class TestSecureDefaults:
    def test_new_output_directory_is_not_world_writable(
        self, tmp_path: Path, detector: Mock
    ) -> None:
        _write_jpeg(tmp_path / "frames" / "frame_001.jpg")
        output = tmp_path / "output"

        process_frames_directory(tmp_path / "frames", output, _config(max_frames=1))

        assert output.stat().st_mode & 0o002 == 0
        assert detector.predict.call_count == 1

    def test_telemetry_excludes_input_and_output_paths(
        self, tmp_path: Path, detector: Mock
    ) -> None:
        frames = tmp_path / "sensitive-input" / "frames"
        output = tmp_path / "sensitive-output"
        _write_jpeg(frames / "frame_001.jpg")
        mlflow_api = Mock()

        with (
            patch("src.pipeline_full.mlflow", mlflow_api),
            patch("src.pipeline_full.start_run", return_value=nullcontext(Mock())),
        ):
            process_frames_directory(frames, output, _config(max_frames=1))

        values = mlflow_api.log_params.call_args.args[0].values()
        assert all(str(tmp_path) not in str(value) for value in values)
        assert detector.predict.call_count == 1


class TestErrorDisclosure:
    def test_model_loader_details_are_logged_but_not_returned(self, tmp_path: Path) -> None:
        _write_jpeg(tmp_path / "frame_001.jpg")

        with (
            patch("src.pipeline_full.load_model", side_effect=RuntimeError("secret /srv/model")),
            pytest.raises(RuntimeError, match="^Model loading failed$") as exc_info,
        ):
            process_frames_directory(tmp_path, tmp_path / "output", _config())

        assert "secret" not in str(exc_info.value)
