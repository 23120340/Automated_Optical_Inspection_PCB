"""Pinhole-camera calibration profiles and deterministic lens undistortion."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from ..core.exceptions import CalibrationProfileError
from ..core.image_io import ensure_bgr


PROFILE_SCHEMA_VERSION = "1.0"
SUPPORTED_DISTORTION_LENGTHS = {4, 5, 8, 12, 14}


@dataclass(frozen=True, slots=True)
class CameraCalibrationProfile:
    """Validated intrinsic calibration for one camera/lens/resolution recipe."""

    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    image_size: tuple[int, int]
    rms_reprojection_error: float | None = None
    mean_reprojection_error: float | None = None
    camera_id: str | None = None
    lens_id: str | None = None
    pattern_size: tuple[int, int] | None = None
    square_size: float | None = None
    images_used: int | None = None
    camera_model: str = "pinhole"

    def __post_init__(self) -> None:
        matrix = np.asarray(self.camera_matrix, dtype=np.float64)
        coefficients = np.asarray(self.distortion_coefficients, dtype=np.float64).reshape(-1)
        width, height = (int(value) for value in self.image_size)
        if self.camera_model != "pinhole":
            raise CalibrationProfileError(
                f"Unsupported camera_model '{self.camera_model}'; expected 'pinhole'."
            )
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise CalibrationProfileError("camera_matrix must be a finite 3x3 matrix.")
        if coefficients.size not in SUPPORTED_DISTORTION_LENGTHS or not np.all(
            np.isfinite(coefficients)
        ):
            allowed = ", ".join(str(value) for value in sorted(SUPPORTED_DISTORTION_LENGTHS))
            raise CalibrationProfileError(
                f"distortion_coefficients must contain {allowed} finite values."
            )
        if width <= 0 or height <= 0:
            raise CalibrationProfileError("Calibration image_width and image_height must be positive.")
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or abs(matrix[2, 2]) < 1e-12:
            raise CalibrationProfileError("camera_matrix contains invalid focal lengths or scale.")
        for name, value in (
            ("rms_reprojection_error", self.rms_reprojection_error),
            ("mean_reprojection_error", self.mean_reprojection_error),
        ):
            if value is not None and (not np.isfinite(value) or float(value) < 0):
                raise CalibrationProfileError(f"{name} must be a finite non-negative number.")
        if self.pattern_size is not None and any(int(value) < 2 for value in self.pattern_size):
            raise CalibrationProfileError("pattern_size must contain at least 2x2 inner corners.")
        if self.square_size is not None and (
            not np.isfinite(self.square_size) or float(self.square_size) <= 0
        ):
            raise CalibrationProfileError("square_size must be a finite positive number.")
        if self.images_used is not None and int(self.images_used) <= 0:
            raise CalibrationProfileError("images_used must be positive when supplied.")
        object.__setattr__(self, "camera_matrix", matrix.copy())
        object.__setattr__(self, "distortion_coefficients", coefficients.copy())
        object.__setattr__(self, "image_size", (width, height))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> CameraCalibrationProfile:
        """Load the public JSON profile schema and reject ambiguous profiles."""

        if not isinstance(values, Mapping):
            raise CalibrationProfileError("Calibration profile must be a JSON object.")
        schema_version = str(values.get("schema_version", ""))
        if schema_version != PROFILE_SCHEMA_VERSION:
            raise CalibrationProfileError(
                f"Unsupported calibration schema_version '{schema_version}'; "
                f"expected '{PROFILE_SCHEMA_VERSION}'."
            )
        try:
            image_size = (int(values["image_width"]), int(values["image_height"]))
            camera_matrix = np.asarray(values["camera_matrix"], dtype=np.float64)
            distortion = np.asarray(values["distortion_coefficients"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationProfileError(
                "Calibration profile requires image_width, image_height, camera_matrix, "
                "and distortion_coefficients."
            ) from exc
        pattern_size = None
        if values.get("pattern_columns") is not None and values.get("pattern_rows") is not None:
            pattern_size = (int(values["pattern_columns"]), int(values["pattern_rows"]))
        return cls(
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion,
            image_size=image_size,
            rms_reprojection_error=_optional_float(values.get("rms_reprojection_error")),
            mean_reprojection_error=_optional_float(values.get("mean_reprojection_error")),
            camera_id=_optional_text(values.get("camera_id")),
            lens_id=_optional_text(values.get("lens_id")),
            pattern_size=pattern_size,
            square_size=_optional_float(values.get("square_size")),
            images_used=_optional_int(values.get("images_used")),
            camera_model=str(values.get("camera_model", "pinhole")),
        )

    @classmethod
    def from_json(cls, payload: str | bytes) -> CameraCalibrationProfile:
        try:
            values = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CalibrationProfileError(f"Calibration profile is not valid UTF-8 JSON: {exc}") from exc
        return cls.from_mapping(values)

    @classmethod
    def load(cls, path: str | Path) -> CameraCalibrationProfile:
        source = Path(path).expanduser().resolve()
        try:
            return cls.from_json(source.read_bytes())
        except OSError as exc:
            raise CalibrationProfileError(f"Could not read calibration profile '{source}': {exc}") from exc

    def camera_matrix_for(self, image_size: tuple[int, int], aspect_tolerance: float) -> np.ndarray:
        """Scale intrinsics for a full-frame image with the same aspect ratio."""

        width, height = (int(value) for value in image_size)
        calibration_width, calibration_height = self.image_size
        if width <= 0 or height <= 0:
            raise CalibrationProfileError("Input image size must be positive.")
        expected_aspect = calibration_width / calibration_height
        actual_aspect = width / height
        relative_difference = abs(actual_aspect / expected_aspect - 1.0)
        tolerance = float(np.clip(aspect_tolerance, 0.0, 0.25))
        if relative_difference > tolerance:
            raise CalibrationProfileError(
                "Input aspect ratio does not match the calibration profile "
                f"({width}x{height} vs {calibration_width}x{calibration_height}; "
                f"difference {relative_difference:.2%}, tolerance {tolerance:.2%})."
            )
        scale_x = width / calibration_width
        scale_y = height / calibration_height
        scaled = self.camera_matrix.copy()
        scaled[0, 0] *= scale_x
        scaled[0, 1] *= scale_x
        scaled[0, 2] *= scale_x
        scaled[1, 0] *= scale_y
        scaled[1, 1] *= scale_y
        scaled[1, 2] *= scale_y
        return scaled

    def to_dict(self) -> dict[str, Any]:
        values: dict[str, Any] = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "camera_model": self.camera_model,
            "image_width": int(self.image_size[0]),
            "image_height": int(self.image_size[1]),
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.distortion_coefficients.tolist(),
            "rms_reprojection_error": self.rms_reprojection_error,
            "mean_reprojection_error": self.mean_reprojection_error,
            "camera_id": self.camera_id,
            "lens_id": self.lens_id,
            "pattern_columns": None if self.pattern_size is None else int(self.pattern_size[0]),
            "pattern_rows": None if self.pattern_size is None else int(self.pattern_size[1]),
            "square_size": self.square_size,
            "images_used": self.images_used,
        }
        return values


@dataclass(frozen=True, slots=True)
class UndistortionResult:
    image: np.ndarray
    roi: tuple[int, int, int, int]
    input_size: tuple[int, int]
    calibration_size: tuple[int, int]
    scaled_intrinsics: bool
    alpha: float


class CameraUndistorter:
    """Cache OpenCV remap matrices for repeated frames at known resolutions."""

    def __init__(
        self,
        profile: CameraCalibrationProfile,
        *,
        alpha: float = 0.0,
        aspect_tolerance: float = 0.01,
    ) -> None:
        self.profile = profile
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.aspect_tolerance = float(np.clip(aspect_tolerance, 0.0, 0.25))
        self._maps: dict[
            tuple[int, int], tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]
        ] = {}

    def correct(self, image: np.ndarray) -> UndistortionResult:
        source = ensure_bgr(image)
        height, width = source.shape[:2]
        size = (width, height)
        if size not in self._maps:
            matrix = self.profile.camera_matrix_for(size, self.aspect_tolerance)
            new_matrix, roi = cv2.getOptimalNewCameraMatrix(
                matrix,
                self.profile.distortion_coefficients,
                size,
                self.alpha,
                size,
            )
            map_x, map_y = cv2.initUndistortRectifyMap(
                matrix,
                self.profile.distortion_coefficients,
                None,
                new_matrix,
                size,
                cv2.CV_32FC1,
            )
            normalized_roi = tuple(int(value) for value in roi)
            self._maps[size] = map_x, map_y, normalized_roi
        map_x, map_y, roi = self._maps[size]
        corrected = cv2.remap(
            source,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return UndistortionResult(
            image=np.ascontiguousarray(corrected),
            roi=roi,
            input_size=size,
            calibration_size=self.profile.image_size,
            scaled_intrinsics=size != self.profile.image_size,
            alpha=self.alpha,
        )


@dataclass(frozen=True, slots=True)
class CalibrationRun:
    profile: CameraCalibrationProfile
    accepted_images: tuple[str, ...]
    rejected_images: tuple[str, ...]
    per_view_errors: tuple[float, ...]


def calibrate_from_chessboards(
    images: Sequence[tuple[str, np.ndarray]],
    *,
    pattern_size: tuple[int, int] = (9, 6),
    square_size: float = 1.0,
    camera_id: str | None = None,
    lens_id: str | None = None,
) -> CalibrationRun:
    """Estimate pinhole intrinsics from chessboard images.

    ``pattern_size`` is the number of inner corners as ``(columns, rows)``.
    ``square_size`` may use any physical unit; that unit is retained only as
    metadata because intrinsic and distortion coefficients are unitless/pixel based.
    """

    columns, rows = (int(value) for value in pattern_size)
    if columns < 2 or rows < 2:
        raise CalibrationProfileError("Chessboard pattern must have at least 2x2 inner corners.")
    if not np.isfinite(square_size) or float(square_size) <= 0:
        raise CalibrationProfileError("square_size must be a finite positive number.")
    object_template = np.zeros((rows * columns, 3), dtype=np.float32)
    object_template[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    object_template *= float(square_size)

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    accepted: list[str] = []
    rejected: list[str] = []
    image_size: tuple[int, int] | None = None
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        40,
        1e-4,
    )
    for name, image in images:
        bgr = ensure_bgr(image)
        current_size = (int(bgr.shape[1]), int(bgr.shape[0]))
        if image_size is None:
            image_size = current_size
        if current_size != image_size:
            raise CalibrationProfileError(
                f"All calibration images must share one resolution; '{name}' is "
                f"{current_size[0]}x{current_size[1]}, expected {image_size[0]}x{image_size[1]}."
            )
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        found = False
        corners = None
        find_sb = getattr(cv2, "findChessboardCornersSB", None)
        if callable(find_sb):
            sb_flags = (
                cv2.CALIB_CB_NORMALIZE_IMAGE
                | getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0)
                | getattr(cv2, "CALIB_CB_ACCURACY", 0)
            )
            try:
                found, corners = find_sb(gray, (columns, rows), sb_flags)
            except cv2.error:
                found, corners = False, None
        if not found or corners is None:
            found, corners = cv2.findChessboardCorners(
                gray,
                (columns, rows),
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if found and corners is not None:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        if not found or corners is None:
            rejected.append(str(name))
            continue
        object_points.append(object_template.copy())
        image_points.append(np.asarray(corners, dtype=np.float32))
        accepted.append(str(name))

    if image_size is None:
        raise CalibrationProfileError("No calibration images were supplied.")
    if len(accepted) < 10:
        raise CalibrationProfileError(
            f"Only {len(accepted)} usable chessboard images were found; at least 10 are required."
        )
    rms, matrix, distortion, rotation_vectors, translation_vectors = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    if not np.isfinite(rms):
        raise CalibrationProfileError("OpenCV returned a non-finite calibration error.")
    per_view_errors: list[float] = []
    for object_row, image_row, rotation, translation in zip(
        object_points,
        image_points,
        rotation_vectors,
        translation_vectors,
    ):
        projected, _ = cv2.projectPoints(object_row, rotation, translation, matrix, distortion)
        error = cv2.norm(image_row, projected, cv2.NORM_L2) / max(1, len(projected))
        per_view_errors.append(float(error))
    profile = CameraCalibrationProfile(
        camera_matrix=matrix,
        distortion_coefficients=distortion,
        image_size=image_size,
        rms_reprojection_error=float(rms),
        mean_reprojection_error=float(np.mean(per_view_errors)),
        camera_id=_optional_text(camera_id),
        lens_id=_optional_text(lens_id),
        pattern_size=(columns, rows),
        square_size=float(square_size),
        images_used=len(accepted),
    )
    return CalibrationRun(
        profile=profile,
        accepted_images=tuple(accepted),
        rejected_images=tuple(rejected),
        per_view_errors=tuple(per_view_errors),
    )


def save_calibration_profile(profile: CameraCalibrationProfile, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(profile.to_dict(), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return destination


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationProfileError(f"Expected a numeric calibration value, got {value!r}.") from exc


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationProfileError(f"Expected an integer calibration value, got {value!r}.") from exc


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
