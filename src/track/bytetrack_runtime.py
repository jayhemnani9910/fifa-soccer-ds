"""Lightweight ByteTrack-style tracker with a simple Kalman filter.

This module provides multi-object tracking using:
- Constant-velocity Kalman filter for smooth motion prediction
- Greedy distance-based association between predictions and detections
- Track lifecycle management (active, lost, deleted)

Key classes:
    - SimpleKalmanFilter: Constant-velocity Kalman filter for bounding boxes
    - Tracklet: Single track at a specific frame
    - Tracklets: Collection of tracklets for a frame
    - ByteTrackRuntime: Main tracking engine with association logic
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def _bbox_is_valid(bbox: Iterable[float]) -> bool:
    coords = list(bbox)
    return len(coords) == 4 and coords[2] > coords[0] and coords[3] > coords[1]


def _bbox_to_measurement(bbox: Iterable[float]) -> np.ndarray:
    x1, y1, x2, y2 = map(float, bbox)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return np.array([cx, cy, w, h], dtype=np.float32)


def _measurement_to_bbox(measurement: np.ndarray) -> np.ndarray:
    cx, cy, w, h = measurement[:4]
    half_w, half_h = w / 2.0, h / 2.0
    return np.array([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dtype=np.float32)


def _center(bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


class SimpleKalmanFilter:
    """Constant-velocity Kalman filter for axis-aligned bounding boxes."""

    def __init__(self, state: np.ndarray, covariance: np.ndarray | None = None) -> None:
        self.F = np.array(
            [
                [1, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 1],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ],
            dtype=np.float32,
        )
        self.Q = np.eye(6, dtype=np.float32) * 1e-2
        self.R = np.eye(4, dtype=np.float32) * 1e-1
        self.I = np.eye(6, dtype=np.float32)
        self.state = state.reshape(-1, 1).astype(np.float32)
        self.cov = covariance if covariance is not None else np.eye(6, dtype=np.float32)

    @classmethod
    def from_bbox(cls, bbox: Iterable[float]) -> SimpleKalmanFilter:
        measurement = _bbox_to_measurement(bbox)
        state = np.zeros(6, dtype=np.float32)
        state[:4] = measurement
        return cls(state=state)

    def predict(self) -> np.ndarray:
        self.state = self.F @ self.state
        self.cov = self.F @ self.cov @ self.F.T + self.Q
        return _measurement_to_bbox(self.state[:, 0])

    def update_from_bbox(self, bbox: Iterable[float]) -> np.ndarray:
        measurement = _bbox_to_measurement(bbox).reshape(-1, 1)
        z_pred = self.H @ self.state
        innovation = measurement - z_pred
        S = self.H @ self.cov @ self.H.T + self.R
        K = self.cov @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ innovation
        self.cov = (self.I - K @ self.H) @ self.cov
        return _measurement_to_bbox(self.state[:, 0])

    @property
    def bbox(self) -> np.ndarray:
        return _measurement_to_bbox(self.state[:, 0])


@dataclass(slots=True)
class Tracklet:
    """Represents a single track hypothesis."""

    track_id: int
    bbox: list[float]
    score: float
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Tracklets:
    """Collection of tracklets for a frame."""

    frame_id: int
    items: list[Tracklet]


@dataclass(slots=True)
class _TrackState:
    track_id: int
    kalman: SimpleKalmanFilter
    score: float
    label: str | None = None
    class_id: int | None = None
    time_since_update: int = 0
    hits: int = 0
    predicted_bbox: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))

    def predict(self) -> np.ndarray:
        self.predicted_bbox = self.kalman.predict()
        self.time_since_update += 1
        return self.predicted_bbox

    def update(
        self, bbox: Iterable[float], score: float, class_id: int | None, label: str | None
    ) -> None:
        self.kalman.update_from_bbox(bbox)
        self.predicted_bbox = self.kalman.bbox
        self.score = score
        self.class_id = class_id
        self.label = label
        self.time_since_update = 0
        self.hits += 1

    @property
    def bbox(self) -> np.ndarray:
        return self.kalman.bbox


class ByteTrackRuntime:
    """ByteTrack-inspired runtime powered by a lightweight Kalman association."""

    def __init__(
        self,
        min_confidence: float = 0.25,
        distance_threshold: float = 80.0,
        max_age: int = 15,
        max_track_id: int = 10000,  # Prevent ID overflow
        id_reuse_delay: int = 30,  # Frames before ID reuse
    ) -> None:
        self.min_confidence = min_confidence
        self.distance_threshold = distance_threshold
        self.max_age = max_age
        self.max_track_id = max_track_id
        self.id_reuse_delay = id_reuse_delay
        
        # Enhanced ID management with reuse pool
        self._next_track_id = 0
        self._id_pool: deque[int] = deque()  # Reusable IDs
        self._id_release_time: dict[int, int] = {}  # Track when IDs become available
        self._tracks: list[_TrackState] = []
        
        # Statistics for monitoring
        self._total_tracks_created = 0
        self._total_ids_reused = 0

    def update(self, frame_id: int, detections: Iterable[dict[str, Any]]) -> Tracklets:
        """Assign track IDs to detections using greedy matching."""
        
        # Track current frame for ID reuse timing
        self._current_frame = frame_id

        valid_detections: list[dict[str, Any]] = []
        for det in detections:
            score = float(det.get("score", 0.0) or 0.0)
            bbox = det.get("bbox", [])
            if score < self.min_confidence or not _bbox_is_valid(bbox):
                continue
            valid_detections.append(
                {
                    "bbox": np.array(bbox, dtype=np.float32),
                    "score": score,
                    "class_id": det.get("class_id"),
                    "class_name": det.get("class_name"),
                }
            )

        predictions = [track.predict() for track in self._tracks]

        # Build cost matrix: distances between predicted tracks and detections
        cost_matrix = np.zeros((len(self._tracks), len(valid_detections)))
        for track_idx, pred_bbox in enumerate(predictions):
            track_center = _center(pred_bbox)
            for det_idx, det in enumerate(valid_detections):
                det_center = _center(det["bbox"])
                distance = float(np.linalg.norm(track_center - det_center))
                cost_matrix[track_idx, det_idx] = distance

        # Solve optimal assignment using Hungarian algorithm
        if cost_matrix.size > 0:
            track_indices, det_indices = linear_sum_assignment(cost_matrix)
        else:
            track_indices, det_indices = np.array([]), np.array([])

        # Filter matches by distance threshold
        matches: list[tuple[int, int]] = []
        unmatched_tracks = set(range(len(self._tracks)))
        unmatched_detections = set(range(len(valid_detections)))

        for track_idx, det_idx in zip(track_indices, det_indices, strict=False):
            distance = cost_matrix[track_idx, det_idx]
            if distance <= self.distance_threshold:
                matches.append((int(track_idx), int(det_idx)))
                unmatched_tracks.discard(track_idx)
                unmatched_detections.discard(det_idx)

        outputs: list[Tracklet] = []

        for track_idx, det_idx in matches:
            track = self._tracks[track_idx]
            det = valid_detections[det_idx]
            track.update(det["bbox"], det["score"], det.get("class_id"), det.get("class_name"))
            outputs.append(
                Tracklet(
                    track_id=track.track_id,
                    bbox=track.bbox.tolist(),
                    score=track.score,
                    extras={
                        "class_id": track.class_id,
                        "class_name": track.label,
                    },
                )
            )

        for det_idx in unmatched_detections:
            det = valid_detections[det_idx]
            new_track = _TrackState(
                track_id=self._allocate_track_id(),
                kalman=SimpleKalmanFilter.from_bbox(det["bbox"]),
                score=det["score"],
                class_id=det.get("class_id"),
                label=det.get("class_name"),
                time_since_update=0,
                hits=1,
            )
            self._tracks.append(new_track)
            outputs.append(
                Tracklet(
                    track_id=new_track.track_id,
                    bbox=new_track.bbox.tolist(),
                    score=new_track.score,
                    extras={
                        "class_id": new_track.class_id,
                        "class_name": new_track.label,
                    },
                )
            )

        # Clean up old tracks and release their IDs
        active_tracks = []
        for track in self._tracks:
            if track.time_since_update <= self.max_age:
                active_tracks.append(track)
            else:
                # Release ID of expired track for reuse
                self._release_track_id(track.track_id, frame_id)
        
        self._tracks = active_tracks

        return Tracklets(frame_id=frame_id, items=outputs)

    def _allocate_track_id(self) -> int:
        """Allocate track ID with reuse and overflow protection."""
        current_frame = getattr(self, '_current_frame', 0)
        
        # Check for reusable IDs first
        if self._id_pool:
            # Check if enough time has passed for reuse
            reusable_ids = []
            for track_id in self._id_pool:
                release_time = self._id_release_time.get(track_id, 0)
                if current_frame - release_time >= self.id_reuse_delay:
                    reusable_ids.append(track_id)
            
            if reusable_ids:
                # Use the oldest reusable ID
                track_id = reusable_ids[0]
                self._id_pool.remove(track_id)
                del self._id_release_time[track_id]
                self._total_ids_reused += 1
                return track_id
        
        # Allocate new ID if no reusable ones available
        if self._next_track_id < self.max_track_id:
            track_id = self._next_track_id
            self._next_track_id += 1
        else:
            # Handle ID overflow - wrap around or reuse oldest
            if self._id_pool:
                track_id = self._id_pool.popleft()  # Emergency reuse
                if track_id in self._id_release_time:
                    del self._id_release_time[track_id]
                self._total_ids_reused += 1
            else:
                # Last resort: reset to 0 (rare case)
                self._next_track_id = 0
                track_id = self._next_track_id
                self._next_track_id += 1
        
        self._total_tracks_created += 1
        return track_id
    
    def _release_track_id(self, track_id: int, current_frame: int) -> None:
        """Release track ID back to pool after delay."""
        if track_id not in self._id_pool:
            self._id_pool.append(track_id)
            self._id_release_time[track_id] = current_frame
            
            # Limit pool size to prevent memory leaks
            max_pool_size = min(100, self.max_track_id // 10)
            while len(self._id_pool) > max_pool_size:
                old_id = self._id_pool.popleft()
                if old_id in self._id_release_time:
                    del self._id_release_time[old_id]
    
    def get_id_statistics(self) -> dict[str, Any]:
        """Get ID management statistics for monitoring."""
        return {
            "total_tracks_created": self._total_tracks_created,
            "total_ids_reused": self._total_ids_reused,
            "reuse_rate": self._total_ids_reused / max(self._total_tracks_created, 1),
            "current_pool_size": len(self._id_pool),
            "next_track_id": self._next_track_id,
            "max_track_id": self.max_track_id,
        }
