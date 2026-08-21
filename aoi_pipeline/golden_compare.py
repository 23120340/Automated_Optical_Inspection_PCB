"""Per-slot Golden appearance comparison after local pose compensation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .image_io import ensure_bgr
from .position import PositionResult
from .recipe import SlotRecipe


@dataclass(frozen=True, slots=True)
class GoldenCompareConfig:
    """Algorithmic thresholds; PASS/NG limits remain stored per recipe slot."""

    pixel_diff_threshold: float = 24.0
    edge_low_threshold: int = 50
    edge_high_threshold: int = 150
    edge_tolerance_px: int = 1
    metric_blur_sigma: float = 0.60
    ssim_window_size: int = 11
    ssim_sigma: float = 1.5
    morphology_kernel: int = 3
    min_blob_area_px: int = 3

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.pixel_diff_threshold)) or not 0.0 < float(
            self.pixel_diff_threshold
        ) <= 255.0:
            raise ValueError("pixel_diff_threshold must be in (0, 255]")
        if not 0 <= int(self.edge_low_threshold) < int(self.edge_high_threshold) <= 255:
            raise ValueError("edge thresholds must satisfy 0 <= low < high <= 255")
        if int(self.edge_tolerance_px) < 0:
            raise ValueError("edge_tolerance_px must be non-negative")
        if not math.isfinite(float(self.metric_blur_sigma)) or float(
            self.metric_blur_sigma
        ) < 0.0:
            raise ValueError("metric_blur_sigma must be a non-negative finite value")
        if int(self.ssim_window_size) < 3 or int(self.ssim_window_size) % 2 == 0:
            raise ValueError("ssim_window_size must be an odd integer >= 3")
        if not math.isfinite(float(self.ssim_sigma)) or float(self.ssim_sigma) <= 0.0:
            raise ValueError("ssim_sigma must be a positive finite value")
        if int(self.morphology_kernel) <= 0:
            raise ValueError("morphology_kernel must be positive")
        if int(self.min_blob_area_px) < 0:
            raise ValueError("min_blob_area_px must be non-negative")


@dataclass(slots=True)
class GoldenCompareResult:
    slot_id: str
    status: str
    ssim: float | None
    diff_ratio: float | None
    edge_diff_ratio: float | None
    max_blob_area_px: int | None
    anomaly_blob_count: int | None
    valid_overlap_ratio: float | None
    reason: str
    defect_label: str | None
    anomaly_mask: np.ndarray | None = None
    coordinate_space: str = "golden_board_pixels"

    @property
    def evaluated(self) -> bool:
        return self.status in {"pass", "anomaly"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "status": self.status,
            "ssim": self.ssim,
            "diff_ratio": self.diff_ratio,
            "edge_diff_ratio": self.edge_diff_ratio,
            "max_blob_area_px": self.max_blob_area_px,
            "anomaly_blob_count": self.anomaly_blob_count,
            "valid_overlap_ratio": self.valid_overlap_ratio,
            "reason": self.reason,
            "defect_label": self.defect_label,
            "coordinate_space": self.coordinate_space,
        }


class GoldenComparator:
    """Compare appearance in a fixed Golden ROI after compensating local pose."""

    def __init__(self, config: GoldenCompareConfig | None = None) -> None:
        self.config = config or GoldenCompareConfig()

    def compare(
        self,
        aligned_image: np.ndarray,
        slot: SlotRecipe,
        recipe_root: str | Path,
        position: PositionResult,
        *,
        ignore_mask: np.ndarray | None = None,
        global_valid_mask: np.ndarray | None = None,
    ) -> GoldenCompareResult:
        """Evaluate appearance without recentering from a detector bbox.

        ``position`` describes the Golden-to-observed local Euclidean pose:
        positive X moves right, positive Y moves down, and positive angle uses
        OpenCV's ``getRotationMatrix2D`` convention. The inverse of that pose is
        applied before metrics are measured in the fixed Golden ROI.
        """

        if not position.measurable or position.dx_px is None or position.dy_px is None:
            return _not_evaluated(slot, "position_not_measurable")
        if slot.rotation_period_deg is not None and position.angle_deg is None:
            return _not_evaluated(slot, "rotation_not_measurable")

        image = ensure_bgr(aligned_image)
        if global_valid_mask is not None and (
            not isinstance(global_valid_mask, np.ndarray)
            or global_valid_mask.ndim != 2
            or global_valid_mask.dtype != np.uint8
            or global_valid_mask.shape != image.shape[:2]
        ):
            return _not_evaluated(slot, "global_valid_mask_invalid")
        root = Path(recipe_root).expanduser().resolve()
        template = _read_asset(root, slot.template_path, cv2.IMREAD_COLOR)
        compare_mask = _read_asset(root, slot.compare_mask_path, cv2.IMREAD_GRAYSCALE)
        stored_ignore = (
            None
            if slot.ignore_mask_path is None
            else _read_asset(root, slot.ignore_mask_path, cv2.IMREAD_GRAYSCALE)
        )
        if template is None or compare_mask is None:
            return _not_evaluated(slot, "appearance_assets_unreadable")
        if stored_ignore is None and slot.ignore_mask_path is not None:
            return _not_evaluated(slot, "ignore_mask_unreadable")
        if ignore_mask is not None and (
            not isinstance(ignore_mask, np.ndarray)
            or ignore_mask.ndim != 2
            or ignore_mask.dtype != np.uint8
        ):
            return _not_evaluated(slot, "runtime_ignore_mask_invalid")

        height, width = image.shape[:2]
        x1, y1, x2, y2 = slot.fixed_roi_xyxy.clamp(width, height).to_int()
        expected_shape = (y2 - y1, x2 - x1)
        if expected_shape[0] <= 0 or expected_shape[1] <= 0:
            return _not_evaluated(slot, "fixed_roi_empty")
        masks = [compare_mask]
        if stored_ignore is not None:
            masks.append(stored_ignore)
        if ignore_mask is not None:
            masks.append(ignore_mask)
        if any(mask.shape != expected_shape for mask in masks) or template.shape[:2] != expected_shape:
            return _not_evaluated(slot, "appearance_asset_geometry_mismatch")

        angle_deg = float(position.angle_deg or 0.0)
        golden_to_observed = cv2.getRotationMatrix2D(
            slot.expected_center_px,
            angle_deg,
            1.0,
        )
        golden_to_observed[:, 2] += (float(position.dx_px), float(position.dy_px))
        compensated_full = cv2.warpAffine(
            image,
            golden_to_observed,
            (width, height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
        )
        source_valid = (
            np.full((height, width), 255, dtype=np.uint8)
            if global_valid_mask is None
            else global_valid_mask
        )
        valid_full = cv2.warpAffine(
            source_valid,
            golden_to_observed,
            (width, height),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
        )
        observed = compensated_full[y1:y2, x1:x2]
        valid_warp = valid_full[y1:y2, x1:x2] > 0

        requested = compare_mask > 0
        if stored_ignore is not None:
            requested &= stored_ignore == 0
        if ignore_mask is not None:
            requested &= ignore_mask == 0
        requested_count = int(np.count_nonzero(requested))
        if requested_count == 0:
            return _not_evaluated(slot, "compare_mask_has_no_usable_pixels")
        evaluation = requested & valid_warp
        overlap = float(np.count_nonzero(evaluation) / requested_count)
        if overlap < float(slot.appearance_thresholds.min_valid_overlap_ratio):
            return GoldenCompareResult(
                slot_id=slot.slot_id,
                status="not_evaluated",
                ssim=None,
                diff_ratio=None,
                edge_diff_ratio=None,
                max_blob_area_px=None,
                anomaly_blob_count=None,
                valid_overlap_ratio=overlap,
                reason="valid_overlap_below_threshold",
                defect_label=None,
            )

        metric_golden = _metric_prefilter(template, float(self.config.metric_blur_sigma))
        metric_observed = _metric_prefilter(observed, float(self.config.metric_blur_sigma))
        normalized_observed = _normalize_luminance(
            metric_golden, metric_observed, evaluation
        )
        golden_lab = cv2.cvtColor(metric_golden, cv2.COLOR_BGR2LAB).astype(np.float32)
        observed_lab = cv2.cvtColor(normalized_observed, cv2.COLOR_BGR2LAB).astype(np.float32)
        golden_luma = golden_lab[:, :, 0]
        observed_luma = observed_lab[:, :, 0]
        ssim = _masked_ssim(
            golden_luma,
            observed_luma,
            evaluation,
            window_size=int(self.config.ssim_window_size),
            sigma=float(self.config.ssim_sigma),
        )

        lab_difference = np.max(np.abs(golden_lab - observed_lab), axis=2)
        difference_pixels = (lab_difference > float(self.config.pixel_diff_threshold)) & evaluation
        diff_ratio = float(np.count_nonzero(difference_pixels) / np.count_nonzero(evaluation))

        golden_gray = cv2.cvtColor(metric_golden, cv2.COLOR_BGR2GRAY)
        observed_gray = cv2.cvtColor(normalized_observed, cv2.COLOR_BGR2GRAY)
        golden_edges = cv2.Canny(
            golden_gray,
            int(self.config.edge_low_threshold),
            int(self.config.edge_high_threshold),
        )
        observed_edges = cv2.Canny(
            observed_gray,
            int(self.config.edge_low_threshold),
            int(self.config.edge_high_threshold),
        )
        golden_edge_pixels = golden_edges > 0
        observed_edge_pixels = observed_edges > 0
        tolerance = int(self.config.edge_tolerance_px)
        if tolerance > 0:
            edge_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * tolerance + 1, 2 * tolerance + 1),
            )
            golden_neighborhood = cv2.dilate(
                golden_edge_pixels.astype(np.uint8), edge_kernel
            ) > 0
            observed_neighborhood = cv2.dilate(
                observed_edge_pixels.astype(np.uint8), edge_kernel
            ) > 0
            edge_difference = (
                (golden_edge_pixels & ~observed_neighborhood)
                | (observed_edge_pixels & ~golden_neighborhood)
            )
        else:
            edge_difference = golden_edge_pixels ^ observed_edge_pixels
        edge_difference &= evaluation
        edge_diff_ratio = float(np.count_nonzero(edge_difference) / np.count_nonzero(evaluation))

        anomaly = np.where(difference_pixels | edge_difference, 255, 0).astype(np.uint8)
        anomaly = _clean_anomaly_mask(anomaly, evaluation, self.config)
        blob_count, max_blob_area = _blob_summary(anomaly, int(self.config.min_blob_area_px))
        thresholds = slot.appearance_thresholds
        failures: list[str] = []
        if ssim < float(thresholds.min_ssim):
            failures.append("ssim_below_threshold")
        if diff_ratio > float(thresholds.max_diff_ratio):
            failures.append("diff_ratio_above_threshold")
        if edge_diff_ratio > float(thresholds.max_edge_diff_ratio):
            failures.append("edge_diff_ratio_above_threshold")
        if max_blob_area > int(thresholds.max_blob_area_px):
            failures.append("blob_area_above_threshold")
        anomalous = bool(failures)
        return GoldenCompareResult(
            slot_id=slot.slot_id,
            status="anomaly" if anomalous else "pass",
            ssim=ssim,
            diff_ratio=diff_ratio,
            edge_diff_ratio=edge_diff_ratio,
            max_blob_area_px=max_blob_area,
            anomaly_blob_count=blob_count,
            valid_overlap_ratio=overlap,
            reason=";".join(failures) if failures else "within_thresholds",
            defect_label="appearance_anomaly" if anomalous else None,
            anomaly_mask=anomaly,
        )


def _normalize_luminance(
    golden: np.ndarray, observed: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    golden_lab = cv2.cvtColor(golden, cv2.COLOR_BGR2LAB).astype(np.float32)
    observed_lab = cv2.cvtColor(observed, cv2.COLOR_BGR2LAB).astype(np.float32)
    golden_values = golden_lab[:, :, 0][mask]
    observed_values = observed_lab[:, :, 0][mask]
    golden_mean = float(np.mean(golden_values))
    observed_mean = float(np.mean(observed_values))
    golden_std = float(np.std(golden_values))
    observed_std = float(np.std(observed_values))
    if abs(golden_mean - observed_mean) < 0.05 and abs(golden_std - observed_std) < 0.05:
        return observed.copy()
    if observed_std > 1.0 and golden_std > 1.0:
        observed_lab[:, :, 0] = (
            (observed_lab[:, :, 0] - observed_mean) * (golden_std / observed_std)
            + golden_mean
        )
    else:
        observed_lab[:, :, 0] += golden_mean - observed_mean
    observed_lab[:, :, 0] = np.clip(observed_lab[:, :, 0], 0, 255)
    return cv2.cvtColor(observed_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _metric_prefilter(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return image.copy()
    return cv2.GaussianBlur(image, (3, 3), sigma)


def _masked_ssim(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray,
    *,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Average a Gaussian local-window SSIM map over valid masked pixels."""

    first_float = first.astype(np.float64)
    second_float = second.astype(np.float64)
    weights = mask.astype(np.float64)
    blur_args = ((window_size, window_size), sigma)
    local_weight = cv2.GaussianBlur(weights, *blur_args)
    safe_weight = np.maximum(local_weight, 1e-12)
    mean_first = cv2.GaussianBlur(first_float * weights, *blur_args) / safe_weight
    mean_second = cv2.GaussianBlur(second_float * weights, *blur_args) / safe_weight
    second_moment_first = (
        cv2.GaussianBlur(first_float * first_float * weights, *blur_args) / safe_weight
    )
    second_moment_second = (
        cv2.GaussianBlur(second_float * second_float * weights, *blur_args) / safe_weight
    )
    cross_moment = (
        cv2.GaussianBlur(first_float * second_float * weights, *blur_args) / safe_weight
    )
    variance_first = np.maximum(second_moment_first - mean_first * mean_first, 0.0)
    variance_second = np.maximum(second_moment_second - mean_second * mean_second, 0.0)
    covariance = cross_moment - mean_first * mean_second
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    numerator = (2.0 * mean_first * mean_second + c1) * (
        2.0 * covariance + c2
    )
    denominator = (
        (mean_first * mean_first + mean_second * mean_second + c1)
        * (variance_first + variance_second + c2)
    )
    ssim_map = numerator / np.maximum(denominator, 1e-12)
    evaluation = mask & (local_weight > 1e-6)
    if not np.any(evaluation):
        return -1.0
    return float(np.clip(np.mean(ssim_map[evaluation]), -1.0, 1.0))


def _clean_anomaly_mask(
    anomaly: np.ndarray,
    evaluation: np.ndarray,
    config: GoldenCompareConfig,
) -> np.ndarray:
    kernel_size = int(config.morphology_kernel)
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        anomaly = cv2.morphologyEx(anomaly, cv2.MORPH_OPEN, kernel)
        anomaly = cv2.morphologyEx(anomaly, cv2.MORPH_CLOSE, kernel)
    anomaly[~evaluation] = 0
    return anomaly


def _blob_summary(mask: np.ndarray, min_area: int) -> tuple[int, int]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = [
        int(stats[index, cv2.CC_STAT_AREA])
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= min_area
    ]
    return len(areas), max(areas, default=0)


def _read_asset(root: Path, relative_path: str, mode: int) -> np.ndarray | None:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    return cv2.imread(str(path), mode)


def _not_evaluated(slot: SlotRecipe, reason: str) -> GoldenCompareResult:
    return GoldenCompareResult(
        slot_id=slot.slot_id,
        status="not_evaluated",
        ssim=None,
        diff_ratio=None,
        edge_diff_ratio=None,
        max_blob_area_px=None,
        anomaly_blob_count=None,
        valid_overlap_ratio=None,
        reason=reason,
        defect_label=None,
    )


__all__ = ["GoldenComparator", "GoldenCompareConfig", "GoldenCompareResult"]
