"""Pitch Coordinate Transformation Module.

Provides utilities for converting pixel coordinates to pitch coordinates
(meters or normalized) using homography calibration.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

LOGGER = logging.getLogger(__name__)

# FIFA standard pitch dimensions
PITCH_LENGTH = 105.0  # meters
PITCH_WIDTH = 68.0  # meters
_MAX_CALIBRATION_FILE_BYTES = 1_048_576


def _validated_image_shape(image_shape: tuple[int, int]) -> tuple[int, int]:
    if len(image_shape) != 2 or any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0
        for value in image_shape
    ):
        raise ValueError("image_shape must contain positive integer (height, width) values")
    return int(image_shape[0]), int(image_shape[1])


def _validated_pitch_dimensions(pitch_length: float, pitch_width: float) -> tuple[float, float]:
    length = float(pitch_length)
    width = float(pitch_width)
    if not math.isfinite(length) or not math.isfinite(width) or length <= 0 or width <= 0:
        raise ValueError("Pitch dimensions must be finite and positive")
    return length, width


@dataclass(slots=True)
class PitchCoordinates:
    """Coordinates in pitch space."""

    x: float  # 0 = own goal line, 1 = opponent goal line (normalized)
    y: float  # 0 = left touchline, 1 = right touchline (normalized)
    x_meters: float = 0.0
    y_meters: float = 0.0


class PitchCoordinateTransformer:
    """Unified interface for pixel-to-pitch coordinate transformation.

    Supports multiple modes:
    - manual: Use manually provided homography matrix or keypoints
    - auto: (future) Automatic pitch line detection
    - identity: Simple normalization without proper calibration

    Args:
        mode: Transformation mode.
        image_shape: Image dimensions (height, width) for normalization.
        pitch_length: Pitch length in meters.
        pitch_width: Pitch width in meters.
    """

    def __init__(
        self,
        mode: Literal["manual", "auto", "identity"] = "identity",
        image_shape: tuple[int, int] | None = None,
        pitch_length: float = PITCH_LENGTH,
        pitch_width: float = PITCH_WIDTH,
    ) -> None:
        if mode not in {"manual", "auto", "identity"}:
            raise ValueError(f"Unsupported transformation mode: {mode}")
        if mode == "auto":
            raise NotImplementedError("Automatic pitch calibration is not implemented")

        self.mode = mode
        self.image_shape = _validated_image_shape(image_shape or (1080, 1920))
        self.pitch_length, self.pitch_width = _validated_pitch_dimensions(pitch_length, pitch_width)

        self._homography_matrix: np.ndarray | None = None
        self._inverse_matrix: np.ndarray | None = None

        if mode == "identity":
            LOGGER.warning(
                "Using identity transform - pitch coordinates will be approximate. "
                "Provide calibration for accurate results."
            )

    @classmethod
    def from_homography_matrix(
        cls,
        matrix: np.ndarray,
        image_shape: tuple[int, int],
        pitch_length: float = PITCH_LENGTH,
        pitch_width: float = PITCH_WIDTH,
    ) -> PitchCoordinateTransformer:
        """Create transformer from a 3x3 homography matrix.

        Args:
            matrix: 3x3 homography matrix (pixel -> pitch).
            image_shape: Image dimensions (height, width).
            pitch_length: Pitch length in meters.
            pitch_width: Pitch width in meters.

        Returns:
            Configured transformer instance.
        """
        transformer = cls(
            mode="manual",
            image_shape=image_shape,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
        )
        homography = np.asarray(matrix, dtype=np.float64)
        if homography.shape != (3, 3) or not np.isfinite(homography).all():
            raise ValueError("Homography matrix must be a finite 3x3 array")
        try:
            inverse = np.linalg.inv(homography)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Homography matrix must be invertible") from exc
        if not np.isfinite(inverse).all():
            raise ValueError("Homography inverse contains non-finite values")
        transformer._homography_matrix = homography
        transformer._inverse_matrix = inverse
        return transformer

    @classmethod
    def from_keypoints(
        cls,
        image_points: list[tuple[float, float]],
        world_points: list[tuple[float, float]],
        image_shape: tuple[int, int],
        pitch_length: float = PITCH_LENGTH,
        pitch_width: float = PITCH_WIDTH,
    ) -> PitchCoordinateTransformer:
        """Create transformer from corresponding keypoint pairs.

        Requires at least 4 point correspondences.

        Args:
            image_points: List of (x, y) pixel coordinates.
            world_points: List of (x, y) pitch coordinates in meters.
            image_shape: Image dimensions (height, width).
            pitch_length: Pitch length in meters.
            pitch_width: Pitch width in meters.

        Returns:
            Configured transformer instance.
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV required for keypoint calibration")

        if len(image_points) < 4:
            raise ValueError("At least 4 keypoint pairs required for homography")
        if len(image_points) != len(world_points):
            raise ValueError("Image and world point counts must match")

        # Convert to numpy arrays
        src_pts = np.array(image_points, dtype=np.float32)
        dst_pts = np.array(world_points, dtype=np.float32)
        if src_pts.shape != (len(image_points), 2) or dst_pts.shape != src_pts.shape:
            raise ValueError("Calibration points must be two-dimensional coordinate pairs")
        if not np.isfinite(src_pts).all() or not np.isfinite(dst_pts).all():
            raise ValueError("Calibration points must be finite")

        # Compute homography using RANSAC
        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if matrix is None or mask is None or int(mask.sum()) < 4:
            raise ValueError("Failed to compute homography from keypoints")

        LOGGER.info("Computed homography from %d keypoints", len(image_points))

        return cls.from_homography_matrix(matrix, image_shape, pitch_length, pitch_width)

    @classmethod
    def from_saved_calibration(cls, path: Path) -> PitchCoordinateTransformer:
        """Load transformer from saved calibration file.

        Args:
            path: Path to calibration JSON file.

        Returns:
            Configured transformer instance.
        """
        path = Path(path)

        if path.stat().st_size > _MAX_CALIBRATION_FILE_BYTES:
            raise ValueError("Calibration file exceeds the 1 MiB size limit")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Calibration file must contain a JSON object")

        try:
            matrix = np.asarray(data["homography_matrix"], dtype=np.float64)
            raw_shape = data["image_shape"]
            if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 2:
                raise ValueError("image_shape must contain height and width")
            image_shape = _validated_image_shape((raw_shape[0], raw_shape[1]))
            pitch_length, pitch_width = _validated_pitch_dimensions(
                data.get("pitch_length", PITCH_LENGTH),
                data.get("pitch_width", PITCH_WIDTH),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Calibration file has an invalid schema") from exc

        LOGGER.info("Loaded calibration from %s", path)

        return cls.from_homography_matrix(matrix, image_shape, pitch_length, pitch_width)

    def save_calibration(self, path: Path) -> None:
        """Save calibration to JSON file.

        Args:
            path: Output path for calibration file.
        """
        if self._homography_matrix is None:
            raise ValueError("No calibration to save (identity mode)")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "homography_matrix": self._homography_matrix.tolist(),
            "image_shape": self.image_shape,
            "pitch_length": self.pitch_length,
            "pitch_width": self.pitch_width,
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        LOGGER.info("Saved calibration to %s", path)

    def pixel_to_pitch(self, pixel_point: tuple[float, float]) -> PitchCoordinates:
        """Convert pixel coordinates to pitch coordinates.

        Args:
            pixel_point: (x, y) in pixel coordinates.

        Returns:
            PitchCoordinates with normalized and meter values.
        """
        px, py = pixel_point
        if not math.isfinite(px) or not math.isfinite(py):
            raise ValueError("Pixel coordinates must be finite")

        if self.mode == "manual" and self._homography_matrix is not None:
            # Apply homography transformation
            world_point = self._apply_homography(np.array([px, py]), self._homography_matrix)
            x_meters, y_meters = world_point

            # Normalize to [0, 1]
            x_norm = np.clip(x_meters / self.pitch_length, 0, 1)
            y_norm = np.clip(y_meters / self.pitch_width, 0, 1)

        else:
            # Identity mode - simple normalization
            height, width = self.image_shape
            x_norm = np.clip(px / width, 0, 1)
            y_norm = np.clip(py / height, 0, 1)

            # Approximate meters
            x_meters = x_norm * self.pitch_length
            y_meters = y_norm * self.pitch_width

        return PitchCoordinates(
            x=float(x_norm), y=float(y_norm), x_meters=float(x_meters), y_meters=float(y_meters)
        )

    def pitch_to_pixel(
        self, pitch_point: tuple[float, float], use_meters: bool = False
    ) -> tuple[float, float]:
        """Convert pitch coordinates to pixel coordinates.

        Args:
            pitch_point: (x, y) in pitch coordinates.
            use_meters: If True, input is in meters; else normalized [0,1].

        Returns:
            (x, y) pixel coordinates.
        """
        x, y = pitch_point
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Pitch coordinates must be finite")

        if not use_meters:
            # Convert normalized to meters
            x = x * self.pitch_length
            y = y * self.pitch_width

        if self.mode == "manual" and self._inverse_matrix is not None:
            pixel_point = self._apply_homography(np.array([x, y]), self._inverse_matrix)
            return float(pixel_point[0]), float(pixel_point[1])
        else:
            # Identity mode - simple denormalization
            height, width = self.image_shape
            px = (x / self.pitch_length) * width
            py = (y / self.pitch_width) * height
            return float(px), float(py)

    def bbox_to_pitch_position(self, bbox: list[float], use_foot_point: bool = True) -> np.ndarray:
        """Convert bounding box to pitch position.

        Args:
            bbox: Bounding box [x1, y1, x2, y2] in pixels.
            use_foot_point: If True, use bottom-center (foot position).

        Returns:
            Normalized position as numpy array [x, y].
        """
        if len(bbox) != 4 or not np.isfinite(np.asarray(bbox, dtype=np.float64)).all():
            raise ValueError("bbox must contain four finite coordinates")
        x1, y1, x2, y2 = bbox
        if x2 < x1 or y2 < y1:
            raise ValueError("bbox maximum coordinates must not be below minimum coordinates")

        if use_foot_point:
            # Bottom-center of bbox (foot position)
            pixel_x = (x1 + x2) / 2
            pixel_y = y2
        else:
            # Center of bbox
            pixel_x = (x1 + x2) / 2
            pixel_y = (y1 + y2) / 2

        coords = self.pixel_to_pitch((pixel_x, pixel_y))
        return np.array([coords.x, coords.y], dtype=np.float32)

    def compute_velocity(
        self, positions: list[np.ndarray], frame_ids: list[int], fps: float = 30.0
    ) -> np.ndarray:
        """Compute velocity from position history.

        Args:
            positions: List of normalized [x, y] positions.
            frame_ids: Corresponding frame IDs.
            fps: Video frame rate.

        Returns:
            Velocity in meters per second [vx, vy].
        """
        if len(positions) != len(frame_ids):
            raise ValueError("positions and frame_ids must have matching lengths")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        if len(positions) < 2:
            return np.zeros(2, dtype=np.float32)

        # Use last two positions
        p1 = np.asarray(positions[-2])
        p2 = np.asarray(positions[-1])
        if p1.shape != (2,) or p2.shape != (2,) or not np.isfinite([p1, p2]).all():
            raise ValueError("positions must contain finite two-dimensional coordinates")
        f1, f2 = frame_ids[-2], frame_ids[-1]

        # Time difference
        dt = (f2 - f1) / fps

        if dt <= 0:
            return np.zeros(2, dtype=np.float32)

        # Position difference (in normalized coords)
        dp = p2 - p1

        # Convert to meters
        dp_meters = dp * np.array([self.pitch_length, self.pitch_width])

        # Velocity in m/s
        velocity = dp_meters / dt

        return velocity.astype(np.float32)

    @staticmethod
    def _apply_homography(point: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Apply homography transformation to a point.

        Args:
            point: 2D point [x, y].
            matrix: 3x3 homography matrix.

        Returns:
            Transformed 2D point.
        """
        # Convert to homogeneous coordinates
        vec = np.array([point[0], point[1], 1.0], dtype=np.float64)

        # Apply transformation
        projected = matrix @ vec

        # Convert back from homogeneous
        if not np.isfinite(projected).all() or math.isclose(
            float(projected[2]), 0.0, abs_tol=1e-12
        ):
            raise ValueError("Homography projects the point to infinity")
        projected /= projected[2]

        return projected[:2]


__all__ = [
    "PitchCoordinateTransformer",
    "PitchCoordinates",
    "PITCH_LENGTH",
    "PITCH_WIDTH",
]
