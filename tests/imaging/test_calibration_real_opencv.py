"""Chessboard calibration against the real OpenCV, with nothing mocked.

The neighbouring mocked test pins the bookkeeping, but it feeds
``findChessboardCorners`` a hand-made ``(N, 1, 2)`` array. OpenCV builds
disagree on that shape -- newer ones return ``(N, 2)`` -- and the mismatch used
to crash the whole calibration script while the mocked test stayed green. These
tests run the real detector so the shape contract is actually exercised.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from aoi_pipeline import (  # noqa: E402
    CalibrationProfileError,
    CameraUndistorter,
    calibrate_from_chessboards,
)

COLUMNS, ROWS = 9, 6          # inner corners
SQUARE_PX = 60
SQUARE_MM = 20.0
WIDTH, HEIGHT = 1280, 960

# Twelve plausible hand-held poses: centred, cornered, and tilted.
POSES = [
    ((0.05, -0.10, 0.02), (-90, -70, 620)),
    ((-0.20, 0.15, -0.05), (-140, -40, 700)),
    ((0.25, 0.05, 0.10), (-60, -110, 660)),
    ((-0.10, -0.25, 0.00), (-120, -90, 580)),
    ((0.30, -0.05, -0.08), (-100, -60, 740)),
    ((-0.05, 0.30, 0.06), (-80, -100, 690)),
    ((0.15, 0.20, -0.12), (-150, -50, 640)),
    ((-0.28, -0.12, 0.04), (-70, -80, 720)),
    ((0.08, 0.02, 0.20), (-110, -75, 600)),
    ((-0.18, 0.22, -0.02), (-95, -95, 760)),
    ((0.22, -0.20, 0.08), (-130, -55, 670)),
    ((-0.02, 0.08, -0.18), (-85, -105, 630)),
]

TRUE_CAMERA_MATRIX = np.array(
    [[1100.0, 0.0, WIDTH / 2 - 12], [0.0, 1095.0, HEIGHT / 2 + 8], [0.0, 0.0, 1.0]]
)
TRUE_DISTORTION = np.array([-0.28, 0.11, 0.001, -0.0008, -0.02])


def _flat_board() -> np.ndarray:
    board = np.zeros(((ROWS + 1) * SQUARE_PX, (COLUMNS + 1) * SQUARE_PX), np.uint8)
    for row in range(ROWS + 1):
        for column in range(COLUMNS + 1):
            if (row + column) % 2 == 0:
                board[
                    row * SQUARE_PX : (row + 1) * SQUARE_PX,
                    column * SQUARE_PX : (column + 1) * SQUARE_PX,
                ] = 255
    return board


@pytest.fixture(scope="module")
def chessboard_views() -> list[tuple[str, np.ndarray]]:
    board = _flat_board()
    source = np.float32(
        [
            [SQUARE_PX, SQUARE_PX],
            [COLUMNS * SQUARE_PX, SQUARE_PX],
            [COLUMNS * SQUARE_PX, ROWS * SQUARE_PX],
            [SQUARE_PX, ROWS * SQUARE_PX],
        ]
    )
    plane = np.float32(
        [
            [0, 0, 0],
            [(COLUMNS - 1) * SQUARE_MM, 0, 0],
            [(COLUMNS - 1) * SQUARE_MM, (ROWS - 1) * SQUARE_MM, 0],
            [0, (ROWS - 1) * SQUARE_MM, 0],
        ]
    )
    views: list[tuple[str, np.ndarray]] = []
    for index, (rotation, translation) in enumerate(POSES):
        projected, _ = cv2.projectPoints(
            plane,
            np.array(rotation, np.float64),
            np.array(translation, np.float64),
            TRUE_CAMERA_MATRIX,
            TRUE_DISTORTION,
        )
        transform = cv2.getPerspectiveTransform(
            source, projected.reshape(-1, 2).astype(np.float32)
        )
        frame = cv2.warpPerspective(
            board,
            transform,
            (WIDTH, HEIGHT),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=110,
        )
        views.append((f"view_{index:02d}.png", cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)))
    return views


def test_real_chessboard_detection_survives_this_opencv_corner_shape(
    chessboard_views: list[tuple[str, np.ndarray]],
) -> None:
    """The regression guard: this used to raise a cv2 type mismatch."""

    run = calibrate_from_chessboards(
        chessboard_views, pattern_size=(COLUMNS, ROWS), square_size=SQUARE_MM
    )
    assert len(run.accepted_images) == len(POSES)
    assert not run.rejected_images
    assert run.profile.images_used == len(POSES)
    assert run.profile.image_size == (WIDTH, HEIGHT)


def test_recovered_focal_lengths_are_close_to_the_truth(
    chessboard_views: list[tuple[str, np.ndarray]],
) -> None:
    run = calibrate_from_chessboards(
        chessboard_views, pattern_size=(COLUMNS, ROWS), square_size=SQUARE_MM
    )
    matrix = run.profile.camera_matrix
    assert matrix[0, 0] == pytest.approx(TRUE_CAMERA_MATRIX[0, 0], rel=0.05)
    assert matrix[1, 1] == pytest.approx(TRUE_CAMERA_MATRIX[1, 1], rel=0.05)


def test_reprojection_error_is_a_pixel_distance_not_a_norm_over_n(
    chessboard_views: list[tuple[str, np.ndarray]],
) -> None:
    """Guards the metric, which used to under-report by sqrt(N).

    With 54 corners that made a 7x worse calibration look acceptable, so the
    per-view number has to stay on the same scale as OpenCV's own RMS.
    """

    run = calibrate_from_chessboards(
        chessboard_views, pattern_size=(COLUMNS, ROWS), square_size=SQUARE_MM
    )
    assert run.per_view_errors
    assert all(error > 0 for error in run.per_view_errors)
    overall = run.profile.rms_reprojection_error
    mean = run.profile.mean_reprojection_error
    assert overall is not None and mean is not None
    # Same units and same order of magnitude as OpenCV's own RMS.
    assert mean == pytest.approx(overall, rel=0.6)
    assert max(run.per_view_errors) < 5.0


def test_the_fitted_profile_can_actually_undistort_a_frame(
    chessboard_views: list[tuple[str, np.ndarray]],
) -> None:
    run = calibrate_from_chessboards(
        chessboard_views, pattern_size=(COLUMNS, ROWS), square_size=SQUARE_MM
    )
    result = CameraUndistorter(run.profile).correct(chessboard_views[0][1])
    assert result.image.shape == chessboard_views[0][1].shape
    assert result.scaled_intrinsics is False
    assert result.calibration_size == (WIDTH, HEIGHT)


def test_too_few_usable_views_is_refused_with_a_clear_message() -> None:
    blank = [(f"blank_{index}.png", np.full((480, 640, 3), 120, np.uint8)) for index in range(12)]
    with pytest.raises(CalibrationProfileError, match="at least 10 are required"):
        calibrate_from_chessboards(blank, pattern_size=(COLUMNS, ROWS), square_size=SQUARE_MM)
