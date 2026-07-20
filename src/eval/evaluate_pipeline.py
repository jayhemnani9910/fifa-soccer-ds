"""Comprehensive pipeline evaluation with metrics and visualization.

This module evaluates the complete FIFA soccer analytics pipeline:
- Object detection accuracy and precision/recall
- Tracking performance (MOTA, MOTP, ID switches)
- Spatial accuracy for homography calibration
- End-to-end pipeline timing analysis
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class DetectionMetrics:
    """Detection performance metrics."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1_score(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)


@dataclass
class TrackingMetrics:
    """Multi-object tracking metrics."""

    total_frames: int = 0
    mostly_tracked: int = 0
    partially_tracked: int = 0
    mostly_lost: int = 0
    id_switches: int = 0
    fragmentations: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    ground_truth_detections: int = 0
    matched_iou_sum: float = 0.0
    matches: int = 0

    @property
    def mota(self) -> float:
        """Multiple Object Tracking Accuracy."""
        if self.ground_truth_detections == 0:
            return 0.0
        errors = self.false_positives + self.false_negatives + self.id_switches
        return 1.0 - errors / self.ground_truth_detections

    @property
    def motp(self) -> float:
        """Multiple Object Tracking Precision."""
        return self.matched_iou_sum / self.matches if self.matches else 0.0


@dataclass
class PipelineMetrics:
    """Complete pipeline performance metrics."""

    detection: DetectionMetrics
    tracking: TrackingMetrics
    evaluation_time: float
    total_time: float | None = None
    fps: float | None = None
    memory_usage_mb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "detection": {
                "precision": self.detection.precision,
                "recall": self.detection.recall,
                "f1_score": self.detection.f1_score,
                "true_positives": self.detection.true_positives,
                "false_positives": self.detection.false_positives,
                "false_negatives": self.detection.false_negatives,
            },
            "tracking": {
                "mota": self.tracking.mota,
                "motp": self.tracking.motp,
                "id_switches": self.tracking.id_switches,
                "fragmentations": self.tracking.fragmentations,
                "false_positives": self.tracking.false_positives,
                "false_negatives": self.tracking.false_negatives,
                "ground_truth_detections": self.tracking.ground_truth_detections,
                "mostly_tracked": self.tracking.mostly_tracked,
                "partially_tracked": self.tracking.partially_tracked,
                "mostly_lost": self.tracking.mostly_lost,
            },
            "performance": {
                "evaluation_time_seconds": self.evaluation_time,
                "total_time_seconds": self.total_time,
                "fps": self.fps,
                "memory_usage_mb": self.memory_usage_mb,
            },
        }


class PipelineEvaluator:
    """Evaluate complete FIFA soccer analytics pipeline."""

    def __init__(self, iou_threshold: float = 0.5):
        if not 0 < iou_threshold <= 1:
            raise ValueError("iou_threshold must be within (0, 1]")
        self.iou_threshold = iou_threshold
        self.detection_metrics = DetectionMetrics()
        self.tracking_metrics = TrackingMetrics()

    def load_results(self, results_path: Path) -> list[dict[str, Any]]:
        """Load tracking results from pipeline."""
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")

        with open(results_path) as f:
            return json.load(f)

    def load_ground_truth(self, gt_path: Path) -> list[dict[str, Any]]:
        """Load ground truth annotations."""
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

        with open(gt_path) as f:
            return json.load(f)

    def calculate_iou(self, box1: list[float], box2: list[float]) -> float:
        """Calculate Intersection over Union for two bounding boxes."""
        if len(box1) != 4 or len(box2) != 4:
            return 0.0
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _bbox(record: dict[str, Any]) -> list[float]:
        value = record.get("bbox", record.get("xyxy", []))
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return []
        try:
            return [float(coordinate) for coordinate in value]
        except (TypeError, ValueError):
            return []

    @staticmethod
    def _group_ground_truth(
        ground_truth: list[dict[str, Any]],
    ) -> dict[int, list[dict[str, Any]]]:
        """Normalize flat or frame-grouped ground-truth records."""
        grouped: dict[int, list[dict[str, Any]]] = {}
        for item in ground_truth:
            try:
                frame_id = int(item.get("frame_id", 0))
            except (TypeError, ValueError):
                continue
            nested = item.get("detections")
            records = nested if isinstance(nested, list) else [item]
            grouped.setdefault(frame_id, []).extend(
                record for record in records if isinstance(record, dict)
            )
        return grouped

    @staticmethod
    def _group_results(
        results: list[dict[str, Any]],
    ) -> dict[int, list[dict[str, Any]]]:
        grouped: dict[int, list[dict[str, Any]]] = {}
        for result in results:
            try:
                frame_id = int(result.get("frame_id", 0))
            except (TypeError, ValueError):
                continue
            detections = result.get("detections", [])
            grouped.setdefault(frame_id, []).extend(
                detection for detection in detections if isinstance(detection, dict)
            )
        return grouped

    def _match_frame(
        self,
        predictions: list[dict[str, Any]],
        annotations: list[dict[str, Any]],
    ) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
        """Greedily match a frame's boxes by descending IoU."""
        candidates: list[tuple[float, int, int]] = []
        for prediction_index, prediction in enumerate(predictions):
            prediction_box = self._bbox(prediction)
            for annotation_index, annotation in enumerate(annotations):
                iou = self.calculate_iou(prediction_box, self._bbox(annotation))
                if iou >= self.iou_threshold:
                    candidates.append((iou, prediction_index, annotation_index))

        matches: list[tuple[int, int, float]] = []
        matched_predictions: set[int] = set()
        matched_annotations: set[int] = set()
        for iou, prediction_index, annotation_index in sorted(candidates, reverse=True):
            if prediction_index in matched_predictions or annotation_index in matched_annotations:
                continue
            matched_predictions.add(prediction_index)
            matched_annotations.add(annotation_index)
            matches.append((prediction_index, annotation_index, iou))
        return matches, matched_predictions, matched_annotations

    def evaluate_detections(
        self, results: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
    ) -> DetectionMetrics:
        """Evaluate detection performance."""
        metrics = DetectionMetrics()
        predictions_by_frame = self._group_results(results)
        annotations_by_frame = self._group_ground_truth(ground_truth)

        for frame_id in predictions_by_frame.keys() | annotations_by_frame.keys():
            predictions = predictions_by_frame.get(frame_id, [])
            annotations = annotations_by_frame.get(frame_id, [])
            matches, matched_predictions, matched_annotations = self._match_frame(
                predictions, annotations
            )
            metrics.true_positives += len(matches)
            metrics.false_positives += len(predictions) - len(matched_predictions)
            metrics.false_negatives += len(annotations) - len(matched_annotations)

        return metrics

    def evaluate_tracking(
        self, results: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
    ) -> TrackingMetrics:
        """Evaluate tracking performance."""
        metrics = TrackingMetrics()
        predictions_by_frame = self._group_results(results)
        annotations_by_frame = self._group_ground_truth(ground_truth)
        metrics.total_frames = len(predictions_by_frame.keys() | annotations_by_frame.keys())
        metrics.ground_truth_detections = sum(map(len, annotations_by_frame.values()))

        # Each observation is (frame, matched predicted track ID). Ground-truth
        # identity is required for cross-frame ID-switch and fragmentation metrics.
        observations: dict[str, list[tuple[int, int | None]]] = {}
        for frame_id in sorted(predictions_by_frame.keys() | annotations_by_frame.keys()):
            predictions = predictions_by_frame.get(frame_id, [])
            annotations = annotations_by_frame.get(frame_id, [])
            matches, matched_predictions, matched_annotations = self._match_frame(
                predictions, annotations
            )
            metrics.false_positives += len(predictions) - len(matched_predictions)
            metrics.false_negatives += len(annotations) - len(matched_annotations)

            match_by_annotation = {
                annotation_index: (prediction_index, iou)
                for prediction_index, annotation_index, iou in matches
            }
            for annotation_index, annotation in enumerate(annotations):
                raw_identity = annotation.get(
                    "track_id", annotation.get("object_id", annotation.get("id"))
                )
                identity = (
                    f"identity:{raw_identity}"
                    if raw_identity is not None
                    else f"anonymous:{frame_id}:{annotation_index}"
                )
                match = match_by_annotation.get(annotation_index)
                predicted_id: int | None = None
                if match is not None:
                    prediction_index, iou = match
                    raw_predicted_id = predictions[prediction_index].get("track_id")
                    if isinstance(raw_predicted_id, int) and not isinstance(raw_predicted_id, bool):
                        predicted_id = raw_predicted_id
                    metrics.matched_iou_sum += iou
                    metrics.matches += 1
                observations.setdefault(identity, []).append((frame_id, predicted_id))

        for trajectory in observations.values():
            matched = [predicted_id is not None for _, predicted_id in trajectory]
            tracked_ratio = sum(matched) / len(matched)
            if tracked_ratio >= 0.8:
                metrics.mostly_tracked += 1
            elif tracked_ratio >= 0.2:
                metrics.partially_tracked += 1
            else:
                metrics.mostly_lost += 1

            matched_segments = 0
            previously_matched = False
            previous_track_id: int | None = None
            for _, predicted_id in trajectory:
                is_matched = predicted_id is not None
                if is_matched and not previously_matched:
                    matched_segments += 1
                if (
                    predicted_id is not None
                    and previous_track_id is not None
                    and predicted_id != previous_track_id
                ):
                    metrics.id_switches += 1
                if predicted_id is not None:
                    previous_track_id = predicted_id
                previously_matched = is_matched
            metrics.fragmentations += max(0, matched_segments - 1)

        return metrics

    def evaluate_pipeline(
        self,
        results_path: Path,
        ground_truth_path: Path,
        timing_data: dict[str, float] | None = None,
    ) -> PipelineMetrics:
        """Evaluate complete pipeline performance."""

        start_time = time.time()

        # Load data
        results = self.load_results(results_path)
        ground_truth = self.load_ground_truth(ground_truth_path)

        # Evaluate components
        detection_metrics = self.evaluate_detections(results, ground_truth)
        tracking_metrics = self.evaluate_tracking(results, ground_truth)

        evaluation_time = time.time() - start_time
        total_time: float | None = None
        fps: float | None = None
        memory_usage: float | None = None
        if timing_data is not None:
            if "total_time" in timing_data:
                total_time = float(timing_data["total_time"])
                if not total_time > 0:
                    raise ValueError("timing_data.total_time must be positive")
                frame_count = len(
                    self._group_results(results).keys()
                    | self._group_ground_truth(ground_truth).keys()
                )
                fps = frame_count / total_time
            if "memory_usage_mb" in timing_data:
                memory_usage = float(timing_data["memory_usage_mb"])
                if memory_usage < 0:
                    raise ValueError("timing_data.memory_usage_mb cannot be negative")

        pipeline_metrics = PipelineMetrics(
            detection=detection_metrics,
            tracking=tracking_metrics,
            evaluation_time=evaluation_time,
            total_time=total_time,
            fps=fps,
            memory_usage_mb=memory_usage,
        )

        log.info("Pipeline evaluation completed:")
        log.info(
            f"  Detection - Precision: {detection_metrics.precision:.3f}, Recall: {detection_metrics.recall:.3f}, F1: {detection_metrics.f1_score:.3f}"
        )
        log.info(
            f"  Tracking - MOTA: {tracking_metrics.mota:.3f}, ID Switches: {tracking_metrics.id_switches}"
        )
        if total_time is not None:
            log.info("  Source performance - FPS: %.1f, Time: %.1fs", fps, total_time)
        else:
            log.info("  Source performance not supplied; evaluation took %.3fs", evaluation_time)

        return pipeline_metrics

    def save_evaluation(
        self, metrics: PipelineMetrics, output_path: Path, save_plots: bool = True
    ) -> None:
        """Save evaluation results and optional plots."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save metrics
        with open(output_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)

        # Generate summary report
        report_path = output_path.parent / "evaluation_report.txt"
        with open(report_path, "w") as f:
            f.write("FIFA Soccer Analytics Pipeline Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write("Detection Performance:\n")
            f.write(f"  Precision: {metrics.detection.precision:.3f}\n")
            f.write(f"  Recall: {metrics.detection.recall:.3f}\n")
            f.write(f"  F1 Score: {metrics.detection.f1_score:.3f}\n")
            f.write(f"  True Positives: {metrics.detection.true_positives}\n")
            f.write(f"  False Positives: {metrics.detection.false_positives}\n")
            f.write(f"  False Negatives: {metrics.detection.false_negatives}\n\n")
            f.write("Tracking Performance:\n")
            f.write(f"  MOTA: {metrics.tracking.mota:.3f}\n")
            f.write(f"  MOTP: {metrics.tracking.motp:.3f}\n")
            f.write(f"  ID Switches: {metrics.tracking.id_switches}\n")
            f.write(f"  Fragmentations: {metrics.tracking.fragmentations}\n\n")
            f.write("Pipeline Performance:\n")
            f.write(f"  Evaluation Time: {metrics.evaluation_time:.3f}s\n")
            f.write(
                f"  Source FPS: {metrics.fps:.1f}\n"
                if metrics.fps is not None
                else "  Source FPS: not supplied\n"
            )
            f.write(
                f"  Source Total Time: {metrics.total_time:.1f}s\n"
                if metrics.total_time is not None
                else "  Source Total Time: not supplied\n"
            )
            f.write(
                f"  Source Memory Usage: {metrics.memory_usage_mb:.1f}MB\n"
                if metrics.memory_usage_mb is not None
                else "  Source Memory Usage: not supplied\n"
            )

        log.info(f"Evaluation results saved to {output_path}")
        log.info(f"Evaluation report saved to {report_path}")


def main() -> None:
    """CLI entrypoint for pipeline evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate FIFA soccer analytics pipeline.")
    parser.add_argument("--results", required=True, help="Path to pipeline results JSON.")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth annotations.")
    parser.add_argument("--output", required=True, help="Output path for evaluation results.")
    parser.add_argument(
        "--iou-threshold", type=float, default=0.5, help="IoU threshold for detection matching."
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    try:
        evaluator = PipelineEvaluator(iou_threshold=args.iou_threshold)
        metrics = evaluator.evaluate_pipeline(
            Path(args.results),
            Path(args.ground_truth),
        )

        evaluator.save_evaluation(metrics, Path(args.output))

    except Exception as e:
        log.error(f"Pipeline evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
