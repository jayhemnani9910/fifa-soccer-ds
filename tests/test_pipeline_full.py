"""Smoke tests for the full pipeline driver."""

import json
from pathlib import Path

import pytest

from src.pipeline_full import PipelineConfig, process_frames_directory, process_youtube_video


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
        self.names = {0: "person"}


class DummyYOLO:
    def __init__(self, weights: str):
        self.weights = weights
        self.device = None

    def to(self, device: str):
        self.device = device
        return self

    def predict(self, source, conf: float = 0.25, verbose: bool = False):
        return [DummyResult()]


class PartialFailureYOLO(DummyYOLO):
    def __init__(self, weights: str):
        super().__init__(weights)
        self.calls = 0

    def predict(self, source, conf: float = 0.25, verbose: bool = False):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("synthetic detector failure")
        return super().predict(source, conf=conf, verbose=verbose)


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
        assert data["attempted_frames"] == 1
        assert data["successful_frames"] == 1
        assert data["processing_success_rate"] == 1.0
        assert data["partial_failure"] is False
        assert data["failures"] == {
            "unreadable_frames": 0,
            "detection_failures": 0,
            "tracking_failures": 0,
            "overlay_failures": 0,
        }
        assert data["total_detections"] == 1
        assert "config" in data


def test_pipeline_reports_partial_frame_failures(tmp_path: Path, monkeypatch) -> None:
    import cv2
    import numpy as np

    import src.detect.infer as detect_infer

    monkeypatch.setattr(detect_infer, "YOLO", PartialFailureYOLO)
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for frame_id in range(2):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite((frames_dir / f"frame_{frame_id:06d}.jpg").as_posix(), frame)

    output_dir = tmp_path / "output"
    summary = process_frames_directory(
        frames_dir,
        output_dir,
        PipelineConfig(max_frames=2),
    )
    payload = json.loads((output_dir / "pipeline_summary.json").read_text(encoding="utf-8"))

    assert summary.attempted_frames == 2
    assert summary.successful_frames == 1
    assert summary.detection_failures == 1
    assert payload["partial_failure"] is True
    assert payload["processing_success_rate"] == 0.5
    assert payload["failures"]["detection_failures"] == 1


# ---------------------------------------------------------------------------
# process_youtube_video: soccer-branch result assembly (anti-fabrication
# contract). These stub only the network/model boundaries -- classification,
# download, and frame extraction from a video file, plus the YOLO detector --
# so process_frames_directory, graph building, and the result assembly in
# process_youtube_video's soccer branch run for real.
# ---------------------------------------------------------------------------


class _FakeYouTubeDownloader:
    """Stands in for YouTubeDownloader so no network call is made."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir

    def download_video(self, youtube_url: str, *, max_duration: int | None = None) -> dict:
        del youtube_url, max_duration
        return {"video_path": "/nonexistent/fake_video.mp4", "file_size": 4096}


def _fake_soccer_classification(
    self, youtube_url, sample_duration=60.0, include_audio=False, max_video_duration=None
) -> dict:
    del self, youtube_url, sample_duration, include_audio, max_video_duration
    return {
        "is_soccer": True,
        "confidence": 0.91,
        "processing_info": {
            "video_info": {
                "title": "Friendly Match Highlights",
                "duration": 120,
                "uploader": "Test Channel",
            }
        },
    }


def _stub_youtube_dependencies(monkeypatch, num_frames: int) -> None:
    import cv2
    import numpy as np

    import src.detect.infer as detect_infer
    import src.pipeline_full as pipeline_full

    monkeypatch.setattr(detect_infer, "YOLO", DummyYOLO)
    monkeypatch.setattr(pipeline_full, "YouTubeDownloader", _FakeYouTubeDownloader)
    monkeypatch.setattr(
        pipeline_full.SoccerClassifier, "classify_youtube_content", _fake_soccer_classification
    )

    def fake_extract_frames_from_video(
        video_path: Path, output_dir: Path, max_frames: int = 30
    ) -> None:
        del video_path, max_frames
        output_dir.mkdir(parents=True, exist_ok=True)
        for index in range(num_frames):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite((output_dir / f"frame_{index:06d}.jpg").as_posix(), frame)

    monkeypatch.setattr(pipeline_full, "_extract_frames_from_video", fake_extract_frames_from_video)


def _run_youtube_soccer_pipeline(tmp_path: Path, monkeypatch, num_frames: int = 2) -> dict:
    _stub_youtube_dependencies(monkeypatch, num_frames)

    output_dir = tmp_path / f"youtube_output_{num_frames}"
    config = PipelineConfig(max_frames=num_frames)

    return process_youtube_video(
        youtube_url="https://www.youtube.com/watch?v=test123abcd",
        output_dir=output_dir,
        config=config,
    )


def test_youtube_soccer_analysis_marks_capabilities_not_implemented_and_no_events(
    tmp_path: Path, monkeypatch
) -> None:
    """The headline result must not claim capabilities the pipeline doesn't have."""
    result = _run_youtube_soccer_pipeline(tmp_path, monkeypatch)

    assert result["type"] == "soccer"
    assert result["capabilities"] == {
        "event_detection": "not_implemented",
        "player_statistics": "not_implemented",
        "team_possession_ppda_field_tilt": "not_implemented",
    }
    assert result["events"] == []


def test_youtube_soccer_analysis_players_and_team_metrics_do_not_fabricate_identities(
    tmp_path: Path, monkeypatch
) -> None:
    """Players/team metrics must expose only tracker output, never invented stats."""
    result = _run_youtube_soccer_pipeline(tmp_path, monkeypatch)

    players = result["players"]
    assert players["status"] == "tracking_only"
    assert players["track_ids"] == result["pipeline_summary"]["unique_track_ids"]
    assert "identity" in players["limitations"].lower()
    forbidden_player_keys = {"name", "player_name", "position", "passes", "touches", "goals"}
    assert forbidden_player_keys.isdisjoint(players.keys())

    team_metrics = result["team_metrics"]
    assert team_metrics["status"] == "tracking_summary_only"
    assert team_metrics["total_detections"] == result["pipeline_summary"]["total_detections"]
    assert team_metrics["active_track_ids"] == len(result["pipeline_summary"]["unique_track_ids"])
    assert "limitations" in team_metrics
    forbidden_team_keys = {"possession", "ppda", "field_tilt"}
    assert forbidden_team_keys.isdisjoint(team_metrics.keys())


def test_youtube_soccer_analysis_summary_counters_reflect_synthetic_frames(
    tmp_path: Path, monkeypatch
) -> None:
    """total_frames/detections must track the actual synthetic input, not a fixed value."""
    result_small = _run_youtube_soccer_pipeline(tmp_path, monkeypatch, num_frames=1)
    result_large = _run_youtube_soccer_pipeline(tmp_path, monkeypatch, num_frames=3)

    assert result_small["pipeline_summary"]["total_frames"] == 1
    assert result_small["pipeline_summary"]["total_detections"] == 1
    assert result_small["processing_info"]["frames_extracted"] == 1

    assert result_large["pipeline_summary"]["total_frames"] == 3
    assert result_large["pipeline_summary"]["total_detections"] == 3
    assert result_large["processing_info"]["frames_extracted"] == 3

    assert (
        result_large["pipeline_summary"]["graph_nodes"]
        > result_small["pipeline_summary"]["graph_nodes"]
    )
