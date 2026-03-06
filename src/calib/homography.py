"""Utilities for estimating a planar homography from correspondences.

This module provides tools for computing planar homography transformations
between two views, useful for pitch-centric sports analytics:
- World-to-image mapping (localize players on pitch coordinates)
- Image-to-world mapping (project detections back to pitch)
- RANSAC-based robust estimation for correspondence noise
- Persistent calibration storage and loading

Key functions:
    - compute_homography: Estimate homography from point correspondences
    - apply_homography: Transform points using computed homography matrix
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore[assignment]
    CV2_AVAILABLE = False

PointList = Iterable[Tuple[float, float]]

def _cv2_circle(img, center, radius, color, thickness=-1):
    """Safely call cv2.circle if available."""
    if CV2_AVAILABLE and cv2 is not None and hasattr(cv2, 'circle'):
        cv2.circle(img, center, radius, color, thickness)

def _cv2_putText(img, text, org, fontFace, fontScale, color, thickness=1, lineType=None):
    """Safely call cv2.putText if available."""
    if CV2_AVAILABLE and cv2 is not None and hasattr(cv2, 'putText'):
        cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType or getattr(cv2, 'LINE_AA', 8))

def _cv2_line(img, pt1, pt2, color, thickness=1):
    """Safely call cv2.line if available."""
    if CV2_AVAILABLE and cv2 is not None and hasattr(cv2, 'line'):
        cv2.line(img, pt1, pt2, color, thickness)

def _cv2_font_hershey_simplex():
    """Get cv2.FONT_HERSHEY_SIMPLEX if available."""
    if CV2_AVAILABLE and cv2 is not None:
        return getattr(cv2, 'FONT_HERSHEY_SIMPLEX', 0)
    return 0

def _cv2_imwrite(filename, img):
    """Safely call cv2.imwrite if available."""
    if CV2_AVAILABLE and cv2 is not None and hasattr(cv2, 'imwrite'):
        cv2.imwrite(filename, img)

def _cv2_find_homography(src_points, dst_points, ransac_threshold):
    """Safely call cv2.findHomography if available."""
    if CV2_AVAILABLE and cv2 is not None and hasattr(cv2, 'findHomography'):
        return cv2.findHomography(
            src_points, dst_points, method=getattr(cv2, 'RANSAC', None), 
            ransacReprojThreshold=ransac_threshold
        )
    return None, None


@dataclass(slots=True)
class HomographyResult:
    """Result of homography estimation."""

    matrix: np.ndarray
    inliers: int
    total: int


def apply_homography(points: PointList, matrix: np.ndarray) -> list[tuple[float, float]]:
    """Apply homography transformation to a set of 2D points.

    Args:
        points: Iterable of (x, y) point coordinates
        matrix: 3x3 homography matrix

    Returns:
        List of transformed (x, y) points

    Raises:
        ValueError: If homography matrix is invalid
    """
    if matrix.shape != (3, 3):
        raise ValueError(f"Homography matrix must be 3x3, got {matrix.shape}")

    points_array = np.asarray(list(points), dtype=np.float32)
    if points_array.shape[0] == 0:
        return []

    # Convert to homogeneous coordinates
    ones = np.ones((points_array.shape[0], 1), dtype=np.float32)
    points_homo = np.hstack([points_array, ones])

    # Apply transformation
    transformed = points_homo @ matrix.T

    # Convert back to Euclidean coordinates
    result = transformed[:, :2] / transformed[:, 2:3]

    return [(float(point[0]), float(point[1])) for point in result]


def pitch_to_image(world_points: PointList, H: np.ndarray) -> list[tuple[float, float]]:
    """Map world (pitch) coordinates to image coordinates using homography.

    Args:
        world_points: Pitch coordinates (width, height in world units)
        H: Homography matrix (world to image)

    Returns:
        Image pixel coordinates
    """
    return apply_homography(world_points, H)


def image_to_pitch(image_points: PointList, H_inv: np.ndarray) -> list[tuple[float, float]]:
    """Map image pixel coordinates back to world (pitch) coordinates.

    Args:
        image_points: Image coordinates (pixel x, pixel y)
        H_inv: Inverse homography matrix (image to world)

    Returns:
        Pitch world coordinates
    """
    return apply_homography(image_points, H_inv)


@dataclass
class CalibrationData:
    """Serializable calibration data structure."""
    homography: List[List[float]]
    inverse_homography: List[List[float]]
    image_points: List[List[float]]
    world_points: List[List[float]]
    reprojection_error: float
    timestamp: str
    image_shape: Tuple[int, int]
    pitch_dimensions: Tuple[float, float]


class HomographyCalibrator:
    """Advanced pitch calibration with persistence and validation.
    
    Provides robust homography estimation with:
    - RANSAC-based outlier rejection
    - Persistent calibration storage
    - Reprojection error calculation
    - Calibration visualization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize calibrator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None
        self.image_points: List[np.ndarray] = []
        self.world_points: List[np.ndarray] = []
        self.reprojection_error: float = float('inf')
        self.image_shape: Optional[Tuple[int, int]] = None
        self.pitch_dimensions: Tuple[float, float] = (105.0, 68.0)  # Standard FIFA pitch
        
    def calibrate_from_points(
        self, 
        image_points: List[Tuple[float, float]], 
        world_points: List[Tuple[float, float]],
        image_shape: Tuple[int, int],
        ransac_threshold: float = 3.0
    ) -> bool:
        """
        Compute homography from point correspondences with validation.
        
        Args:
            image_points: Points in image coordinates (pixels)
            world_points: Corresponding points in world coordinates (meters)
            image_shape: (height, width) of image
            ransac_threshold: RANSAC reprojection threshold in pixels
            
        Returns:
            True if calibration successful and meets quality thresholds
        """
        if len(image_points) < 4:
            raise ValueError("At least 4 point correspondences required for homography")
            
        if len(image_points) != len(world_points):
            raise ValueError("Number of image and world points must match")
            
        self.image_points = [np.array(p, dtype=np.float32) for p in image_points]
        self.world_points = [np.array(p, dtype=np.float32) for p in world_points]
        self.image_shape = image_shape
        
        # Compute homography with RANSAC
        result = compute_homography(self.world_points, self.image_points, ransac_threshold)
        
        if result.matrix is None:
            raise RuntimeError("Homography computation failed - check point correspondences")
            
        self.H = result.matrix
        self.H_inv = np.linalg.inv(self.H)
        
        # Calculate reprojection error
        self.reprojection_error = self._compute_reprojection_error()
        
        # Validate calibration quality
        min_inliers = max(4, int(len(image_points) * 0.7))
        if result.inliers < min_inliers:
            raise ValueError(f"Poor calibration quality: only {result.inliers}/{len(image_points)} inliers")
        
        if self.reprojection_error > ransac_threshold * 2:
            raise ValueError(f"High reprojection error: {self.reprojection_error:.2f}px")
            
        return True
        
    def _compute_reprojection_error(self) -> float:
        """Compute mean reprojection error in pixels."""
        if self.H is None:
            return float('inf')
            
        errors = []
        for img_pt, world_pt in zip(self.image_points, self.world_points):
            projected = self.project_world_to_image(world_pt)
            error = np.linalg.norm(projected - img_pt)
            errors.append(error)
        return float(np.mean(errors))
        
    def project_world_to_image(self, point: np.ndarray) -> np.ndarray:
        """Project world point to image coordinates."""
        if self.H is None:
            raise ValueError("Calibrator not initialized - call calibrate_from_points first")
        point_homog = np.array([point[0], point[1], 1.0], dtype=np.float32)
        image_homog = self.H @ point_homog
        return (image_homog[:2] / image_homog[2]).astype(np.float32)
        
    def project_image_to_world(self, point: np.ndarray) -> np.ndarray:
        """Project image point to world coordinates."""
        if self.H_inv is None:
            raise ValueError("Calibrator not initialized - call calibrate_from_points first")
        point_homog = np.array([point[0], point[1], 1.0], dtype=np.float32)
        world_homog = self.H_inv @ point_homog
        return (world_homog[:2] / world_homog[2]).astype(np.float32)
        
    def save_calibration(self, path: Path) -> None:
        """
        Serialize calibration data to JSON with metadata.
        
        Args:
            path: Output path for calibration file
        """
        if self.H is None or self.H_inv is None:
            raise ValueError("No calibration to save - call calibrate_from_points first")
            
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = CalibrationData(
            homography=self.H.tolist(),
            inverse_homography=self.H_inv.tolist(),
            image_points=[p.tolist() for p in self.image_points],
            world_points=[p.tolist() for p in self.world_points],
            reprojection_error=self.reprojection_error,
            timestamp=datetime.utcnow().isoformat(),
            image_shape=self.image_shape or (0, 0),
            pitch_dimensions=self.pitch_dimensions
        )
        
        with open(path, 'w') as f:
            json.dump(asdict(data), f, indent=2)
            
    @classmethod
    def load_calibration(cls, path: Path, config: Optional[Dict[str, Any]] = None) -> 'HomographyCalibrator':
        """
        Load pre-computed calibration from file with validation.
        
        Args:
            path: Path to calibration JSON
            config: Optional configuration dictionary
            
        Returns:
            Initialized calibrator instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")
            
        with open(path) as f:
            data = json.load(f)
            
        instance = cls(config)
        instance.H = np.array(data["homography"], dtype=np.float32)
        instance.H_inv = np.array(data["inverse_homography"], dtype=np.float32)
        instance.image_points = [np.array(p, dtype=np.float32) for p in data["image_points"]]
        instance.world_points = [np.array(p, dtype=np.float32) for p in data["world_points"]]
        instance.reprojection_error = data["reprojection_error"]
        instance.image_shape = tuple(data["image_shape"])
        instance.pitch_dimensions = tuple(data.get("pitch_dimensions", (105.0, 68.0)))
        
        # Validate loaded calibration
        if instance.reprojection_error > 10.0:  # High error threshold
            raise ValueError(f"Loaded calibration has poor quality: {instance.reprojection_error:.2f}px error")
            
        return instance
        
    def visualize_calibration(self, image: np.ndarray, output_path: Optional[Path] = None) -> np.ndarray:
        """
        Visualize calibration on sample image with point correspondences.
        
        Args:
            image: Input image to draw calibration on
            output_path: Optional path to save visualization
            
        Returns:
            Image with calibration visualization
        """
        if self.H is None:
            raise ValueError("No calibration to visualize")
            
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV (cv2) is required for visualization")
            
        vis = image.copy()
        
        # Draw image points in green
        for i, pt in enumerate(self.image_points):
            _cv2_circle(vis, tuple(map(int, pt)), 5, (0, 255, 0), -1)
            _cv2_putText(vis, f"I{i+1}", tuple(map(int, pt)), 
                        _cv2_font_hershey_simplex(), 0.5, (0, 255, 0), 1, None)        # Draw projected world points in red
        for i, pt in enumerate(self.world_points):
            img_pt = self.project_world_to_image(pt)
            _cv2_circle(vis, tuple(map(int, img_pt)), 3, (0, 0, 255), -1)
            _cv2_putText(vis, f"W{i+1}", tuple(map(int, img_pt)), 
                        _cv2_font_hershey_simplex(), 0.5, (0, 0, 255), 1, None)
            
        # Draw lines connecting correspondences
        for img_pt, world_pt in zip(self.image_points, self.world_points):
            projected = self.project_world_to_image(world_pt)
            _cv2_line(vis, tuple(map(int, img_pt)), tuple(map(int, projected)), (255, 255, 0), 1)
            
        # Add calibration info
        info_text = f"Reprojection Error: {self.reprojection_error:.2f}px"
        _cv2_putText(vis, info_text, (10, 30), _cv2_font_hershey_simplex(), 
                    0.6, (255, 255, 255), 2, None)
        
        if output_path:
            _cv2_imwrite(str(output_path), vis)
            
        return vis
        
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of calibration quality and parameters."""
        if self.H is None:
            return {"status": "not_calibrated"}
            
        return {
            "status": "calibrated",
            "reprojection_error_px": round(self.reprojection_error, 3),
            "num_correspondences": len(self.image_points),
            "image_shape": self.image_shape,
            "pitch_dimensions_m": self.pitch_dimensions,
            "condition_number": np.linalg.cond(self.H).astype(float),
            "determinant": float(np.linalg.det(self.H[:2, :2]))
        }


class PitchCalibration:
    """Manages pitch-centric coordinate systems for sports analytics.

    Maintains homography transformations between image space (pixel coordinates)
    and world space (pitch coordinates in meters).
    """

    def __init__(self, H: np.ndarray | None = None):
        """Initialize calibration.

        Args:
            H: Homography matrix (world to image). If None, identity matrix used.
        """
        self.H = H if H is not None else np.eye(3, dtype=np.float32)
        try:
            self.H_inv = np.linalg.inv(self.H)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Homography matrix is not invertible") from exc

    @classmethod
    def from_correspondences(cls, src: PointList, dst: PointList) -> PitchCalibration:
        """Create calibration from corresponding point pairs.

        Args:
            src: World (pitch) coordinate pairs
            dst: Image coordinate pairs

        Returns:
            PitchCalibration instance
        """
        result = compute_homography(src, dst)
        return cls(H=result.matrix)

    def world_to_image(self, points: PointList) -> list[tuple[float, float]]:
        """Transform world coordinates to image coordinates."""
        return pitch_to_image(points, self.H)

    def image_to_world(self, points: PointList) -> list[tuple[float, float]]:
        """Transform image coordinates to world coordinates."""
        return image_to_pitch(points, self.H_inv)

    def to_dict(self) -> dict:
        """Serialize calibration to dictionary."""
        return {
            "H": self.H.tolist(),
            "H_inv": self.H_inv.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> PitchCalibration:
        """Deserialize calibration from dictionary."""
        H = np.array(data["H"], dtype=np.float32)
        return cls(H=H)

def compute_homography(
    src: PointList, dst: PointList, ransac_threshold: float = 3.0
) -> HomographyResult:
    """Estimate a homography matrix mapping `src` points to `dst` points.

    Raises if OpenCV is unavailable or insufficient point pairs are provided.
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) is required for homography computation.")

    src_points = np.asarray(list(src), dtype=np.float32)
    dst_points = np.asarray(list(dst), dtype=np.float32)
    if src_points.shape != dst_points.shape or src_points.shape[0] < 4:
        raise ValueError("At least four matching point pairs are required.")

    matrix, mask = _cv2_find_homography(
        src_points, dst_points, ransac_threshold
    )
    if matrix is None or mask is None:
        raise RuntimeError("Failed to compute homography.")

    inliers = int(mask.sum())
    return HomographyResult(matrix=matrix, inliers=inliers, total=int(mask.size))
