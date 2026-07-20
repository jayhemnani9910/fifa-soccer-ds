from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.evaluate_pipeline import PipelineEvaluator


def test_detection_matching_uses_each_ground_truth_box() -> None:
    evaluator = PipelineEvaluator(iou_threshold=0.5)
    results = [
        {
            "frame_id": 0,
            "detections": [
                {"bbox": [0, 0, 10, 10]},
                {"bbox": [20, 20, 30, 30]},
            ],
        }
    ]
    ground_truth = [
        {"frame_id": 0, "bbox": [0, 0, 10, 10]},
        {"frame_id": 0, "bbox": [20, 20, 30, 30]},
        {"frame_id": 1, "bbox": [40, 40, 50, 50]},
    ]

    metrics = evaluator.evaluate_detections(results, ground_truth)

    assert metrics.true_positives == 2
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 1


def test_tracking_metrics_use_ground_truth_identity_and_iou() -> None:
    evaluator = PipelineEvaluator(iou_threshold=0.5)
    results = [
        {
            "frame_id": 0,
            "detections": [
                {"track_id": 1, "bbox": [0, 0, 10, 10]},
                {"track_id": 10, "bbox": [20, 20, 30, 30]},
            ],
        },
        {
            "frame_id": 1,
            "detections": [{"track_id": 2, "bbox": [1, 0, 11, 10]}],
        },
    ]
    ground_truth = [
        {"frame_id": 0, "track_id": "A", "bbox": [0, 0, 10, 10]},
        {"frame_id": 0, "track_id": "B", "bbox": [20, 20, 30, 30]},
        {"frame_id": 1, "track_id": "A", "bbox": [1, 0, 11, 10]},
        {"frame_id": 1, "track_id": "B", "bbox": [21, 20, 31, 30]},
    ]

    metrics = evaluator.evaluate_tracking(results, ground_truth)

    assert metrics.ground_truth_detections == 4
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 1
    assert metrics.id_switches == 1
    assert metrics.fragmentations == 0
    assert metrics.mostly_tracked == 1
    assert metrics.partially_tracked == 1
    assert metrics.mota == pytest.approx(0.5)
    assert metrics.motp == pytest.approx(1.0)


def test_malformed_boxes_are_counted_without_crashing() -> None:
    evaluator = PipelineEvaluator()
    results = [{"frame_id": 0, "detections": [{"bbox": [1, 2]}]}]
    ground_truth = [{"frame_id": 0, "bbox": [0, 0, 10, 10]}]

    metrics = evaluator.evaluate_detections(results, ground_truth)

    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1


def test_pipeline_performance_is_unknown_without_source_timings(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    ground_truth = tmp_path / "ground_truth.json"
    results.write_text(json.dumps([{"frame_id": 0, "detections": []}]))
    ground_truth.write_text(json.dumps([]))

    metrics = PipelineEvaluator().evaluate_pipeline(results, ground_truth)

    assert metrics.evaluation_time >= 0
    assert metrics.total_time is None
    assert metrics.fps is None
    assert metrics.memory_usage_mb is None
    assert metrics.to_dict()["performance"]["memory_usage_mb"] is None


def test_pipeline_performance_uses_unique_frames_and_supplied_timings(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    ground_truth = tmp_path / "ground_truth.json"
    results.write_text(
        json.dumps(
            [
                {"frame_id": 0, "detections": []},
                {"frame_id": 1, "detections": []},
            ]
        )
    )
    ground_truth.write_text(json.dumps([{"frame_id": 2, "bbox": [0, 0, 1, 1]}]))

    metrics = PipelineEvaluator().evaluate_pipeline(
        results,
        ground_truth,
        timing_data={"total_time": 1.5, "memory_usage_mb": 42.0},
    )

    assert metrics.total_time == 1.5
    assert metrics.fps == pytest.approx(2.0)
    assert metrics.memory_usage_mb == 42.0
