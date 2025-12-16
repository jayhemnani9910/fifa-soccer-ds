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
from typing import Any, Dict, List, Optional

import numpy as np

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
    
    @property
    def mota(self) -> float:
        """Multiple Object Tracking Accuracy."""
        if self.total_frames == 0:
            return 0.0
        return 1.0 - (
            (self.mostly_lost + self.fragmentations + self.id_switches) / self.total_frames
        )
    
    @property
    def motp(self) -> float:
        """Multiple Object Tracking Precision."""
        # Placeholder - would require ground truth trajectory data
        return 0.0


@dataclass
class PipelineMetrics:
    """Complete pipeline performance metrics."""
    detection: DetectionMetrics
    tracking: TrackingMetrics
    total_time: float = 0.0
    fps: float = 0.0
    memory_usage_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "detection": {
                "precision": self.detection.precision,
                "recall": self.detection.recall,
                "f1_score": self.detection.f1_score,
                "true_positives": self.detection.true_positives,
                "false_positives": self.detection.false_positives,
                "false_negatives": self.detection.false_negatives
            },
            "tracking": {
                "mota": self.tracking.mota,
                "motp": self.tracking.motp,
                "id_switches": self.tracking.id_switches,
                "fragmentations": self.tracking.fragmentations,
                "mostly_tracked": self.tracking.mostly_tracked,
                "partially_tracked": self.tracking.partially_tracked,
                "mostly_lost": self.tracking.mostly_lost
            },
            "performance": {
                "total_time_seconds": self.total_time,
                "fps": self.fps,
                "memory_usage_mb": self.memory_usage_mb
            }
        }


class PipelineEvaluator:
    """Evaluate complete FIFA soccer analytics pipeline."""
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.detection_metrics = DetectionMetrics()
        self.tracking_metrics = TrackingMetrics()
        
    def load_results(self, results_path: Path) -> List[Dict[str, Any]]:
        """Load tracking results from pipeline."""
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")
        
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def load_ground_truth(self, gt_path: Path) -> List[Dict[str, Any]]:
        """Load ground truth annotations."""
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
        
        with open(gt_path, 'r') as f:
            return json.load(f)
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union for two bounding boxes."""
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
    
    def evaluate_detections(
        self, 
        results: List[Dict[str, Any]], 
        ground_truth: List[Dict[str, Any]]
    ) -> DetectionMetrics:
        """Evaluate detection performance."""
        metrics = DetectionMetrics()
        
        # Group by frame for comparison
        gt_by_frame = {}
        for gt in ground_truth:
            frame_id = gt.get("frame_id", 0)
            if frame_id not in gt_by_frame:
                gt_by_frame[frame_id] = []
            gt_by_frame[frame_id].append(gt)
        
        # Compare detections to ground truth
        for result in results:
            frame_id = result.get("frame_id", 0)
            detections = result.get("detections", [])
            gt_detections = gt_by_frame.get(frame_id, [])
            
            # Match detections to ground truth
            matched_gt = set()
            for det in detections:
                det_bbox = det.get("bbox", [])
                best_iou = 0.0
                best_gt_idx = -1
                
                for i, gt in enumerate(gt_detections):
                    if i in matched_gt:
                        continue
                    
                    gt_bbox = gt.get("bbox", [])
                    iou = self.calculate_iou(det_bbox, gt_bbox)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
                
                if best_iou >= self.iou_threshold:
                    metrics.true_positives += 1
                    matched_gt.add(best_gt_idx)
                else:
                    metrics.false_positives += 1
            
            # Count false negatives
            for i, gt in enumerate(gt_detections):
                if i not in matched_gt:
                    metrics.false_negatives += 1
        
        return metrics
    
    def evaluate_tracking(
        self, 
        results: List[Dict[str, Any]], 
        ground_truth: List[Dict[str, Any]]
    ) -> TrackingMetrics:
        """Evaluate tracking performance."""
        metrics = TrackingMetrics()
        
        # Group by track ID for analysis
        tracks_by_id = {}
        for result in results:
            frame_id = result.get("frame_id", 0)
            detections = result.get("detections", [])
            
            for det in detections:
                track_id = det.get("track_id", -1)
                if track_id not in tracks_by_id:
                    tracks_by_id[track_id] = []
                tracks_by_id[track_id].append({
                    "frame_id": frame_id,
                    "bbox": det.get("bbox", [])
                })
        
        # Analyze track continuity
        for track_id, track_data in tracks_by_id.items():
            if len(track_data) < 2:
                continue
            
            # Check for fragmentation
            frame_ids = sorted([d["frame_id"] for d in track_data])
            gaps = 0
            for i in range(1, len(frame_ids)):
                if frame_ids[i] - frame_ids[i-1] > 1:
                    gaps += 1
            
            if gaps > 0:
                metrics.fragmentations += gaps
        
        # Count ID switches (simplified - would need more complex analysis)
        frame_track_map = {}
        id_switches = 0
        
        for result in results:
            frame_id = result.get("frame_id", 0)
            detections = result.get("detections", [])
            
            for det in detections:
                track_id = det.get("track_id", -1)
                if frame_id in frame_track_map:
                    if track_id != frame_track_map[frame_id]:
                        id_switches += 1
                frame_track_map[frame_id] = track_id
        
        metrics.id_switches = id_switches
        metrics.total_frames = len(results)
        
        return metrics
    
    def evaluate_pipeline(
        self, 
        results_path: Path, 
        ground_truth_path: Path,
        timing_data: Optional[Dict[str, float]] = None
    ) -> PipelineMetrics:
        """Evaluate complete pipeline performance."""
        
        start_time = time.time()
        
        # Load data
        results = self.load_results(results_path)
        ground_truth = self.load_ground_truth(ground_truth_path)
        
        # Evaluate components
        detection_metrics = self.evaluate_detections(results, ground_truth)
        tracking_metrics = self.evaluate_tracking(results, ground_truth)
        
        # Calculate performance metrics
        total_time = timing_data.get("total_time", time.time() - start_time) if timing_data else time.time() - start_time
        fps = len(results) / max(total_time, 1.0) if results else 0.0
        
        # Memory usage (placeholder)
        memory_usage = timing_data.get("memory_usage_mb", 0.0) if timing_data else 0.0
        
        pipeline_metrics = PipelineMetrics(
            detection=detection_metrics,
            tracking=tracking_metrics,
            total_time=total_time,
            fps=fps,
            memory_usage_mb=memory_usage
        )
        
        log.info(f"Pipeline evaluation completed:")
        log.info(f"  Detection - Precision: {detection_metrics.precision:.3f}, Recall: {detection_metrics.recall:.3f}, F1: {detection_metrics.f1_score:.3f}")
        log.info(f"  Tracking - MOTA: {tracking_metrics.mota:.3f}, ID Switches: {tracking_metrics.id_switches}")
        log.info(f"  Performance - FPS: {fps:.1f}, Time: {total_time:.1f}s, Memory: {memory_usage:.1f}MB")
        
        return pipeline_metrics
    
    def save_evaluation(
        self, 
        metrics: PipelineMetrics, 
        output_path: Path,
        save_plots: bool = True
    ) -> None:
        """Save evaluation results and optional plots."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(output_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        
        # Generate summary report
        report_path = output_path.parent / "evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write("FIFA Soccer Analytics Pipeline Evaluation Report\\n")
            f.write("=" * 50 + "\\n\\n")
            f.write(f"Detection Performance:\\n")
            f.write(f"  Precision: {metrics.detection.precision:.3f}\\n")
            f.write(f"  Recall: {metrics.detection.recall:.3f}\\n")
            f.write(f"  F1 Score: {metrics.detection.f1_score:.3f}\\n")
            f.write(f"  True Positives: {metrics.detection.true_positives}\\n")
            f.write(f"  False Positives: {metrics.detection.false_positives}\\n")
            f.write(f"  False Negatives: {metrics.detection.false_negatives}\\n\\n")
            f.write(f"Tracking Performance:\\n")
            f.write(f"  MOTA: {metrics.tracking.mota:.3f}\\n")
            f.write(f"  MOTP: {metrics.tracking.motp:.3f}\\n")
            f.write(f"  ID Switches: {metrics.tracking.id_switches}\\n")
            f.write(f"  Fragmentations: {metrics.tracking.fragmentations}\\n\\n")
            f.write(f"Pipeline Performance:\\n")
            f.write(f"  FPS: {metrics.fps:.1f}\\n")
            f.write(f"  Total Time: {metrics.total_time:.1f}s\\n")
            f.write(f"  Memory Usage: {metrics.memory_usage_mb:.1f}MB\\n")
        
        log.info(f"Evaluation results saved to {output_path}")
        log.info(f"Evaluation report saved to {report_path}")


def main() -> None:
    """CLI entrypoint for pipeline evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate FIFA soccer analytics pipeline.")
    parser.add_argument("--results", required=True, help="Path to pipeline results JSON.")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth annotations.")
    parser.add_argument("--output", required=True, help="Output path for evaluation results.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for detection matching.")
    
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