from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from aoi_pipeline import (
    CalibrationProfileError,
    CameraCalibrationProfile,
    ImagePreprocessor,
    PreprocessConfig,
    calibrate_from_chessboards,
)
from app.pipeline_bridge import PipelineBridge


def _profile(width: int = 100, height: int = 80) -> CameraCalibrationProfile:
    return CameraCalibrationProfile(
        camera_matrix=np.array(
            [[120.0, 0.0, width / 2], [0.0, 118.0, height / 2], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        distortion_coefficients=np.array([0.08, -0.03, 0.001, -0.002, 0.0]),
        image_size=(width, height),
        rms_reprojection_error=0.21,
        camera_id="cam-01",
        lens_id="lens-12mm",
    )


def test_calibration_profile_json_round_trip_and_intrinsic_scaling() -> None:
    profile = CameraCalibrationProfile.from_mapping(_profile().to_dict())
    scaled = profile.camera_matrix_for((200, 160), aspect_tolerance=0.01)

    assert profile.image_size == (100, 80)
    assert profile.camera_id == "cam-01"
    assert scaled[0, 0] == pytest.approx(240.0)
    assert scaled[1, 1] == pytest.approx(236.0)
    assert scaled[0, 2] == pytest.approx(100.0)
    assert scaled[1, 2] == pytest.approx(80.0)


def test_calibration_profile_rejects_cropped_or_wrong_aspect_ratio() -> None:
    with pytest.raises(CalibrationProfileError, match="aspect ratio"):
        _profile().camera_matrix_for((100, 100), aspect_tolerance=0.01)


def test_preprocessor_undistorts_before_resize() -> None:
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    cv2.line(image, (0, 40), (99, 40), (255, 255, 255), 2)
    processor = ImagePreprocessor(
        PreprocessConfig(
            undistort=True,
            calibration_profile=_profile().to_dict(),
            max_side=50,
            denoise=False,
            white_balance=False,
            clahe=False,
            normalize=False,
            sharpen=False,
        )
    )

    result = processor.process(image)

    assert result.image.shape == (40, 50, 3)
    assert result.operations[0] == "undistort:pinhole"
    assert result.operations[1] == "resize:50x40"
    assert result.metrics["undistort"] == "applied"
    assert result.metrics["calibration_size"] == "100x80"


def test_preprocessor_requires_a_profile_when_undistort_is_enabled() -> None:
    with pytest.raises(CalibrationProfileError, match="no calibration_profile"):
        ImagePreprocessor(PreprocessConfig(undistort=True))


def test_bridge_never_silently_skips_requested_undistortion() -> None:
    bridge = PipelineBridge.__new__(PipelineBridge)
    bridge.config = {"preprocess": {"undistort": True}}
    bridge.engine = SimpleNamespace(
        preprocess=lambda image: (_ for _ in ()).throw(CalibrationProfileError("bad profile"))
    )

    with pytest.raises(RuntimeError, match="không tự bỏ qua undistort"):
        bridge.preprocess(np.zeros((20, 30, 3), dtype=np.uint8))


def test_bridge_applies_uploaded_profile_through_pipeline_config() -> None:
    bridge = PipelineBridge(
        config={
            "preprocess": {
                "undistort": True,
                "calibration_profile": _profile().to_dict(),
                "resize_enabled": False,
                "denoise": "None",
                "white_balance": False,
                "clahe": False,
                "normalize": False,
                "sharpen": 0.0,
            }
        }
    )

    result = bridge.preprocess(np.zeros((80, 100, 3), dtype=np.uint8))

    assert result.mode == "PIPELINE"
    assert result.metrics["undistort"] == "applied"
    assert result.image.shape == (80, 100, 3)


def test_chessboard_calibration_records_quality_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bookkeeping only, on the older OpenCV ``(N, 1, 2)`` corner shape.

    Newer builds return ``(N, 2)`` instead; that shape is covered for real in
    ``test_calibration_real_opencv.py``. Between the two, both contracts stay
    exercised -- mocking the detector here is what let a crash on the newer
    shape ship unnoticed.
    """

    columns, rows = 3, 2
    corners = np.mgrid[0:columns, 0:rows].T.reshape(-1, 1, 2).astype(np.float32)
    matrix = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    distortion = np.zeros((1, 5), dtype=np.float64)

    monkeypatch.setattr(cv2, "findChessboardCorners", lambda *args, **kwargs: (True, corners.copy()))
    monkeypatch.setattr(cv2, "cornerSubPix", lambda gray, found, *args: found)

    def fake_calibrate(object_points, image_points, image_size, *_):
        count = len(object_points)
        vectors = [np.zeros((3, 1), dtype=np.float64) for _ in range(count)]
        return 0.18, matrix, distortion, vectors, vectors

    monkeypatch.setattr(cv2, "calibrateCamera", fake_calibrate)
    monkeypatch.setattr(
        cv2,
        "projectPoints",
        lambda points, *_: (points[:, :2].reshape(-1, 1, 2).astype(np.float32), None),
    )
    images = [(f"view-{index}.png", np.zeros((80, 100, 3), dtype=np.uint8)) for index in range(10)]

    run = calibrate_from_chessboards(images, pattern_size=(columns, rows), square_size=1.0)

    assert run.profile.images_used == 10
    assert run.profile.rms_reprojection_error == pytest.approx(0.18)
    assert run.profile.mean_reprojection_error == pytest.approx(0.0)
    assert len(run.accepted_images) == 10
    assert not run.rejected_images
