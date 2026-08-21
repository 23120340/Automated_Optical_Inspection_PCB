"""Step 1: deterministic OpenCV preprocessing for PCB images."""

from __future__ import annotations

import cv2
import numpy as np

from .calibration import CameraCalibrationProfile, CameraUndistorter
from ..config import PreprocessConfig
from ..exceptions import CalibrationProfileError
from .image_io import ensure_bgr
from ..models import PreprocessResult


class ImagePreprocessor:
    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()
        self.undistorter: CameraUndistorter | None = None
        if self.config.undistort:
            if self.config.calibration_profile is None:
                raise CalibrationProfileError(
                    "Lens undistortion is enabled but no calibration_profile was supplied."
                )
            profile = CameraCalibrationProfile.from_mapping(self.config.calibration_profile)
            self.undistorter = CameraUndistorter(
                profile,
                alpha=self.config.undistort_alpha,
                aspect_tolerance=self.config.calibration_aspect_tolerance,
            )

    def process(self, image: np.ndarray) -> PreprocessResult:
        source = ensure_bgr(image)
        output = source.copy()
        operations: list[str] = []
        warnings: list[str] = []
        metrics: dict[str, object] = {}
        scale = 1.0

        if self.undistorter is not None:
            corrected = self.undistorter.correct(output)
            output = corrected.image
            scaled = ":scaled_intrinsics" if corrected.scaled_intrinsics else ""
            operations.append(f"undistort:pinhole{scaled}")
            metrics.update(
                {
                    "undistort": "applied",
                    "calibration_size": (
                        f"{corrected.calibration_size[0]}x{corrected.calibration_size[1]}"
                    ),
                    "undistort_alpha": corrected.alpha,
                    "valid_pixel_roi": "x".join(str(value) for value in corrected.roi),
                }
            )
            if corrected.roi[2] <= 0 or corrected.roi[3] <= 0:
                warnings.append("Undistortion produced an empty valid-pixel ROI")

        max_side = self.config.max_side
        if max_side is not None:
            if max_side <= 0:
                raise ValueError("max_side must be positive or None")
            current_max = max(output.shape[:2])
            if current_max > max_side:
                scale = max_side / float(current_max)
                new_size = (
                    max(1, int(round(output.shape[1] * scale))),
                    max(1, int(round(output.shape[0] * scale))),
                )
                output = cv2.resize(output, new_size, interpolation=cv2.INTER_AREA)
                operations.append(f"resize:{new_size[0]}x{new_size[1]}")

        if self.config.denoise and self.config.denoise_strength > 0:
            strength = int(np.clip(self.config.denoise_strength, 1, 30))
            method = self.config.denoise_method
            if method == "bilateral":
                output = cv2.bilateralFilter(output, 7, strength * 8, strength * 8)
                operations.append(f"denoise:bilateral:{strength}")
            elif method == "gaussian":
                kernel = max(3, strength if strength % 2 else strength + 1)
                kernel = min(kernel, 31)
                output = cv2.GaussianBlur(output, (kernel, kernel), 0)
                operations.append(f"denoise:gaussian:{strength}")
            else:
                output = cv2.fastNlMeansDenoisingColored(output, None, strength, strength, 7, 21)
                operations.append(f"denoise:nlmeans:{strength}")

        if self.config.white_balance:
            output, applied = _gray_world_white_balance(output)
            if applied:
                operations.append("white_balance:gray_world")
            else:
                warnings.append("White balance skipped because at least one channel is nearly empty")

        if self.config.clahe:
            grid = tuple(max(1, int(value)) for value in self.config.clahe_grid_size)
            lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)
            lightness, channel_a, channel_b = cv2.split(lab)
            clahe = cv2.createCLAHE(
                clipLimit=max(0.1, float(self.config.clahe_clip_limit)),
                tileGridSize=grid,
            )
            lightness = clahe.apply(lightness)
            output = cv2.cvtColor(cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR)
            operations.append("contrast:clahe")

        if self.config.normalize:
            output, applied = _normalize_luminance(
                output,
                self.config.normalize_low_percentile,
                self.config.normalize_high_percentile,
            )
            if applied:
                operations.append("normalize:luminance_percentile")
            else:
                warnings.append("Luminance normalization skipped because the image has no contrast")

        if self.config.sharpen and self.config.sharpen_amount > 0:
            amount = float(np.clip(self.config.sharpen_amount, 0.0, 2.0))
            blurred = cv2.GaussianBlur(output, (0, 0), sigmaX=1.0)
            output = cv2.addWeighted(output, 1.0 + amount, blurred, -amount, 0)
            operations.append(f"sharpen:unsharp:{amount:.2f}")

        return PreprocessResult(
            image=np.ascontiguousarray(output),
            input_shape=tuple(source.shape),
            operations=operations,
            scale=scale,
            warnings=warnings,
            metrics=metrics,
        )


def _gray_world_white_balance(image: np.ndarray) -> tuple[np.ndarray, bool]:
    values = image.astype(np.float32)
    channel_means = values.reshape(-1, 3).mean(axis=0)
    if np.any(channel_means < 1.0):
        return image, False
    target = float(channel_means.mean())
    gains = np.clip(target / channel_means, 0.5, 2.0)
    balanced = values * gains.reshape(1, 1, 3)
    return np.clip(balanced, 0, 255).astype(np.uint8), True


def _normalize_luminance(image: np.ndarray, low: float, high: float) -> tuple[np.ndarray, bool]:
    low = float(np.clip(low, 0.0, 99.0))
    high = float(np.clip(high, low + 0.1, 100.0))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    low_value, high_value = np.percentile(lightness, [low, high])
    if high_value - low_value < 2.0:
        return image, False
    normalized = (lightness.astype(np.float32) - low_value) * (255.0 / (high_value - low_value))
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    merged = cv2.merge((normalized, channel_a, channel_b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR), True
