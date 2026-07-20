"""Regression tests for pipeline resource bounds and I/O behavior."""

from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from src.pipeline_full import PipelineConfig, process_frames_directory


def _write_frames(frames_dir: Path, count: int) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    for index in range(count):
        assert cv2.imwrite(str(frames_dir / f"frame_{index:03d}.jpg"), image)


@pytest.fixture()
def detector(monkeypatch: pytest.MonkeyPatch) -> Mock:
    fake = Mock(predict=Mock(return_value=[]))
    monkeypatch.setattr("src.pipeline_full.load_model", lambda _config: fake)
    monkeypatch.setattr("src.pipeline_full.mlflow", None)
    return fake


def _config(**overrides) -> PipelineConfig:  # type: ignore[no-untyped-def]
    return PipelineConfig(enable_tactical_analytics=False, **overrides)


class TestPerformanceOptimizations:
    """Test explicit work and output bounds used by the frame pipeline."""

    def test_max_frames_limits_inference_work(self, tmp_path: Path, detector: Mock) -> None:
        frames_dir = tmp_path / "frames"
        output_dir = tmp_path / "output"
        _write_frames(frames_dir, 100)

        summary = process_frames_directory(frames_dir, output_dir, _config(max_frames=5))

        assert summary.total_frames == 5
        assert detector.predict.call_count == 5
        assert len(list((output_dir / "detections").glob("*.json"))) == 5

    def test_each_selected_frame_is_read_once(self, tmp_path: Path, detector: Mock) -> None:
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, 10)

        with patch("src.pipeline_full.cv2.imread", wraps=cv2.imread) as imread:
            summary = process_frames_directory(
                frames_dir, tmp_path / "output", _config(max_frames=10)
            )

        assert summary.total_frames == 10
        assert imread.call_count == 10
        assert detector.predict.call_count == 10

    def test_output_growth_is_bounded_by_max_frames(self, tmp_path: Path, detector: Mock) -> None:
        frames_dir = tmp_path / "frames"
        output_dir = tmp_path / "output"
        _write_frames(frames_dir, 50)

        summary = process_frames_directory(frames_dir, output_dir, _config(max_frames=7))

        assert summary.total_frames == 7
        assert detector.predict.call_count == 7
        assert len(list((output_dir / "detections").glob("*.json"))) == 7
        assert len(list((output_dir / "tracks").glob("*.json"))) == 7

    def test_output_files_written_per_processed_frame(self, tmp_path: Path, detector: Mock) -> None:
        frames_dir = tmp_path / "frames"
        output_dir = tmp_path / "output"
        _write_frames(frames_dir, 20)

        summary = process_frames_directory(frames_dir, output_dir, _config(max_frames=20))

        assert summary.total_frames == 20
        assert detector.predict.call_count == 20
        assert len(list((output_dir / "detections").glob("*_detections.json"))) == 20
        assert len(list((output_dir / "tracks").glob("*_tracks.json"))) == 20


class TestResourceOptimization:
    """Test safe output preservation, error reporting, and configuration forwarding."""

    def test_existing_output_is_preserved(self, tmp_path: Path, detector: Mock) -> None:
        frames_dir = tmp_path / "frames"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        existing = output_dir / "old_file.txt"
        existing.write_text("user data", encoding="utf-8")
        _write_frames(frames_dir, 3)

        process_frames_directory(frames_dir, output_dir, _config(max_frames=3))

        assert existing.read_text(encoding="utf-8") == "user data"
        assert (output_dir / "detections").is_dir()
        assert (output_dir / "tracks").is_dir()
        assert (output_dir / "graphs").is_dir()
        assert detector.predict.call_count == 3

    def test_detection_persistence_failure_is_not_silenced(
        self, tmp_path: Path, detector: Mock
    ) -> None:
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, 1)
        original_open = Path.open

        def fail_detection_write(path: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
            if path.parent.name == "detections":
                raise OSError("disk full")
            return original_open(path, *args, **kwargs)

        with (
            patch.object(Path, "open", fail_detection_write),
            pytest.raises(RuntimeError, match="persist detections"),
        ):
            process_frames_directory(frames_dir, tmp_path / "output", _config(max_frames=1))

        assert detector.predict.call_count == 1

    def test_confidence_is_forwarded_to_inference(self, tmp_path: Path, detector: Mock) -> None:
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, 2)

        for confidence in (0.1, 0.9):
            process_frames_directory(
                frames_dir,
                tmp_path / f"output-{confidence}",
                _config(confidence=confidence, max_frames=1),
            )

        assert [call.kwargs["conf"] for call in detector.predict.call_args_list] == [0.1, 0.9]
