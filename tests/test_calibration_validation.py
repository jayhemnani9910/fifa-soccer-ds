from __future__ import annotations

import json

import numpy as np
import pytest

from src.calib.homography import HomographyCalibrator, PitchCalibration, apply_homography
from src.calib.pitch_transform import PitchCoordinateTransformer


def test_apply_homography_rejects_invalid_or_infinite_projections() -> None:
    assert apply_homography([(2.0, 3.0)], np.eye(3)) == [(2.0, 3.0)]

    with pytest.raises(ValueError, match="finite 3x3"):
        apply_homography([(1.0, 1.0)], np.full((3, 3), np.nan))
    with pytest.raises(ValueError, match="coordinate pairs"):
        apply_homography([(1.0, 2.0, 3.0)], np.eye(3))  # type: ignore[list-item]
    with pytest.raises(ValueError, match="infinity"):
        apply_homography([(0.0, 0.0)], np.diag([1.0, 1.0, 0.0]))


def test_calibrator_projection_rejects_points_at_infinity() -> None:
    calibrator = HomographyCalibrator()
    calibrator.H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="infinity"):
        calibrator.project_world_to_image(np.array([0.0, 1.0]))


def test_transformer_rejects_unimplemented_and_invalid_calibration() -> None:
    with pytest.raises(NotImplementedError, match="not implemented"):
        PitchCoordinateTransformer(mode="auto")
    with pytest.raises(ValueError, match="positive integer"):
        PitchCoordinateTransformer(image_shape=(0, 1920))
    with pytest.raises(ValueError, match="invertible"):
        PitchCoordinateTransformer.from_homography_matrix(np.zeros((3, 3)), (720, 1280))
    with pytest.raises(ValueError, match="counts must match"):
        PitchCoordinateTransformer.from_keypoints(
            [(0.0, 0.0)] * 4,
            [(0.0, 0.0)] * 5,
            (720, 1280),
        )


def test_transformer_validates_runtime_coordinates_and_timing() -> None:
    transformer = PitchCoordinateTransformer(image_shape=(720, 1280))

    with pytest.raises(ValueError, match="Pixel coordinates"):
        transformer.pixel_to_pitch((float("nan"), 1.0))
    with pytest.raises(ValueError, match="maximum coordinates"):
        transformer.bbox_to_pitch_position([10.0, 10.0, 5.0, 20.0])
    with pytest.raises(ValueError, match="matching lengths"):
        transformer.compute_velocity([np.zeros(2)], [])
    with pytest.raises(ValueError, match="fps"):
        transformer.compute_velocity([], [], fps=0.0)


def test_saved_transformer_calibration_schema_is_validated(tmp_path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "homography_matrix": np.eye(3).tolist(),
                "image_shape": [720, 1280],
                "pitch_length": 105.0,
                "pitch_width": 68.0,
            }
        ),
        encoding="utf-8",
    )

    transformer = PitchCoordinateTransformer.from_saved_calibration(path)
    assert transformer.pixel_to_pitch((52.5, 34.0)).x_meters == pytest.approx(52.5)

    path.write_text('{"homography_matrix": [[1]], "image_shape": [720, 1280]}', encoding="utf-8")
    with pytest.raises(ValueError, match="finite 3x3"):
        PitchCoordinateTransformer.from_saved_calibration(path)


def test_calibrator_recomputes_inverse_and_quality_from_saved_points(tmp_path) -> None:
    homography = np.array([[10.0, 0.0, 5.0], [0.0, 10.0, 7.0], [0.0, 0.0, 1.0]])
    world_points = [[0.0, 0.0], [100.0, 0.0], [100.0, 60.0], [0.0, 60.0]]
    image_points = apply_homography(world_points, homography)
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "homography": homography.tolist(),
                "inverse_homography": np.zeros((3, 3)).tolist(),
                "image_points": image_points,
                "world_points": world_points,
                "reprojection_error": -100.0,
                "image_shape": [720, 1280],
                "pitch_dimensions": [105.0, 68.0],
            }
        ),
        encoding="utf-8",
    )

    calibrator = HomographyCalibrator.load_calibration(path)

    assert calibrator.reprojection_error == pytest.approx(0.0)
    assert calibrator.H_inv is not None
    np.testing.assert_allclose(calibrator.H_inv, np.linalg.inv(homography))


def test_pitch_calibration_rejects_non_finite_matrix() -> None:
    with pytest.raises(ValueError, match="finite 3x3"):
        PitchCalibration(np.full((3, 3), np.inf))
