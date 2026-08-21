"""Fixed-ROI component position measurement in Golden board coordinates.

The input image must already be aligned to the recipe's canonical
``golden_board_pixels`` canvas. Runtime detector boxes are presence candidates
only: their coordinates are deliberately ignored for metrology.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..imaging.image_io import ensure_bgr, read_asset_under_root
from ..models import Detection
from .recipe import MetrologyCalibration, SlotRecipe


@dataclass(frozen=True, slots=True)
class PositionQualityGates:
    """Configurable confidence gates for local translation matching."""

    min_score: float = 0.60
    min_peak_margin: float = 0.015
    min_psr: float = 4.0
    min_pose_correlation: float = 0.70
    max_pose_residual: float = 0.35
    min_valid_overlap_ratio: float = 0.80
    max_abs_rotation_deg: float = 15.0
    ecc_iterations: int = 120
    ecc_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.min_score) <= 1.0:
            raise ValueError("min_score must be between 0 and 1")
        if not 0.0 <= float(self.min_peak_margin) <= 1.0:
            raise ValueError("min_peak_margin must be between 0 and 1")
        if not math.isfinite(float(self.min_psr)) or float(self.min_psr) < 0.0:
            raise ValueError("min_psr must be a non-negative finite value")
        if not 0.0 <= float(self.min_pose_correlation) <= 1.0:
            raise ValueError("min_pose_correlation must be between 0 and 1")
        if not math.isfinite(float(self.max_pose_residual)) or not 0.0 <= float(
            self.max_pose_residual
        ) <= 1.0:
            raise ValueError("max_pose_residual must be between 0 and 1")
        if not 0.0 <= float(self.min_valid_overlap_ratio) <= 1.0:
            raise ValueError("min_valid_overlap_ratio must be between 0 and 1")
        if not 0.0 <= float(self.max_abs_rotation_deg) <= 180.0:
            raise ValueError("max_abs_rotation_deg must be between 0 and 180")
        if int(self.ecc_iterations) <= 0:
            raise ValueError("ecc_iterations must be positive")
        if not math.isfinite(float(self.ecc_epsilon)) or float(self.ecc_epsilon) <= 0:
            raise ValueError("ecc_epsilon must be a positive finite value")


@dataclass(frozen=True, slots=True)
class PositionResult:
    """Translation result for one recipe slot.

    ``dx_px`` is positive when the observed component moved right relative to
    Golden; ``dy_px`` is positive when it moved down. Millimetres preserve the
    same signs and are computed independently per axis. Invalid measurements
    expose no numeric displacement, preventing callers from treating a failed
    match as a real zero.
    """

    slot_id: str
    status: str
    dx_px: float | None
    dy_px: float | None
    dx_mm: float | None
    dy_mm: float | None
    angle_deg: float | None
    score: float | None
    peak_margin: float | None
    psr: float | None
    pose_correlation: float | None
    pose_residual: float | None
    valid_overlap_ratio: float | None
    reason: str
    coordinate_space: str = "golden_board_pixels"

    @property
    def measurable(self) -> bool:
        return self.status in {"pass", "ng"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "status": self.status,
            "dx_px": self.dx_px,
            "dy_px": self.dy_px,
            "dx_mm": self.dx_mm,
            "dy_mm": self.dy_mm,
            "angle_deg": self.angle_deg,
            "score": self.score,
            "peak_margin": self.peak_margin,
            "psr": self.psr,
            "pose_correlation": self.pose_correlation,
            "pose_residual": self.pose_residual,
            "valid_overlap_ratio": self.valid_overlap_ratio,
            "reason": self.reason,
            "coordinate_space": self.coordinate_space,
        }


class PositionMeasurer:
    """Measure component translation from a fixed Golden template and mask."""

    def __init__(self, quality_gates: PositionQualityGates | None = None) -> None:
        self.quality_gates = quality_gates or PositionQualityGates()

    def measure(
        self,
        aligned_image: np.ndarray,
        slot: SlotRecipe,
        recipe_root: str | Path,
        metrology: MetrologyCalibration,
        *,
        candidate: Detection | None,
        global_valid_mask: np.ndarray | None = None,
    ) -> PositionResult:
        """Measure X/Y without using ``candidate.bbox`` as the slot centre.

        The candidate is only evidence that the detector associated a component
        with this slot. Coarse search uses the recipe's fixed ROI plus its search
        margin; a quadratic peak fit refines the translation below one pixel.
        ``global_valid_mask`` is the source-validity mask propagated by strict
        alignment. Padded/undistorted-invalid pixels can never yield a numeric
        metrology result.
        """

        if candidate is None:
            return _invalid(slot, "missing_candidate", "No detector candidate was associated")

        image = ensure_bgr(aligned_image)
        if global_valid_mask is None:
            valid_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        elif (
            not isinstance(global_valid_mask, np.ndarray)
            or global_valid_mask.ndim != 2
            or global_valid_mask.shape != image.shape[:2]
        ):
            return _invalid(slot, "unmeasurable", "aligned_valid_mask_geometry_invalid")
        else:
            valid_mask = np.where(global_valid_mask > 0, 255, 0).astype(np.uint8)
        root = Path(recipe_root).expanduser().resolve()
        template = _read_asset(root, slot.template_path, cv2.IMREAD_COLOR)
        mask = _read_asset(root, slot.component_mask_path, cv2.IMREAD_GRAYSCALE)
        if template is None or mask is None:
            return _invalid(slot, "unmeasurable", "Position template or mask is unreadable")
        if template.shape[:2] != mask.shape[:2] or not np.any(mask > 0):
            return _invalid(slot, "unmeasurable", "Position template/mask geometry is invalid")

        height, width = image.shape[:2]
        roi_x1, roi_y1, roi_x2, roi_y2 = slot.fixed_roi_xyxy.clamp(width, height).to_int()
        expected_height = roi_y2 - roi_y1
        expected_width = roi_x2 - roi_x1
        if expected_height <= 0 or expected_width <= 0:
            return _invalid(slot, "unmeasurable", "Fixed ROI is empty on the aligned image")
        if template.shape[:2] != (expected_height, expected_width):
            return _invalid(slot, "unmeasurable", "Template size does not match fixed ROI")

        margin = int(slot.search_margin_px)
        sx1 = max(0, roi_x1 - margin)
        sy1 = max(0, roi_y1 - margin)
        sx2 = min(width, roi_x2 + margin)
        sy2 = min(height, roi_y2 + margin)
        search = image[sy1:sy2, sx1:sx2]
        if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
            return _invalid(slot, "unmeasurable", "Search ROI is smaller than the template")

        template_repr = _gradient_representation(template)
        search_repr = _gradient_representation(search)
        binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        try:
            response = cv2.matchTemplate(
                search_repr,
                template_repr,
                cv2.TM_CCORR_NORMED,
                mask=binary_mask,
            )
        except cv2.error as exc:
            return _invalid(slot, "unmeasurable", f"Template matching failed: {exc}")
        response = np.nan_to_num(response, nan=-1.0, posinf=-1.0, neginf=-1.0)
        if response.size == 0:
            return _invalid(slot, "unmeasurable", "Template matching returned no candidates")

        _, peak, _, peak_location = cv2.minMaxLoc(response)
        peak_x, peak_y = peak_location
        refined_x = float(peak_x) + _quadratic_offset(response[peak_y, :], peak_x)
        refined_y = float(peak_y) + _quadratic_offset(response[:, peak_x], peak_y)
        peak_margin, psr = _peak_quality(response, peak_x, peak_y, float(peak))

        gates = self.quality_gates
        failures: list[str] = []
        if float(peak) < float(gates.min_score):
            failures.append(f"score {peak:.3f} < {gates.min_score:.3f}")
        if peak_margin < float(gates.min_peak_margin):
            failures.append(f"peak_margin {peak_margin:.3f} < {gates.min_peak_margin:.3f}")
        if psr < float(gates.min_psr):
            failures.append(f"psr {psr:.3f} < {gates.min_psr:.3f}")
        if failures:
            return PositionResult(
                slot_id=slot.slot_id,
                status="unmeasurable",
                dx_px=None,
                dy_px=None,
                dx_mm=None,
                dy_mm=None,
                angle_deg=None,
                score=float(peak),
                peak_margin=peak_margin,
                psr=psr,
                pose_correlation=None,
                pose_residual=None,
                valid_overlap_ratio=None,
                reason="; ".join(failures),
            )

        dx_px = float(sx1 + refined_x - roi_x1)
        dy_px = float(sy1 + refined_y - roi_y1)
        tolerance = slot.position_tolerance
        angle_deg: float | None = None
        pose_correlation: float | None = None
        pose_residual: float | None = None
        valid_overlap_ratio = _translation_valid_overlap(
            valid_mask,
            binary_mask,
            (roi_x1, roi_y1, roi_x2, roi_y2),
            dx_px,
            dy_px,
        )
        if valid_overlap_ratio < float(gates.min_valid_overlap_ratio):
            return PositionResult(
                slot_id=slot.slot_id,
                status="unmeasurable",
                dx_px=None,
                dy_px=None,
                dx_mm=None,
                dy_mm=None,
                angle_deg=None,
                score=float(peak),
                peak_margin=peak_margin,
                psr=psr,
                pose_correlation=None,
                pose_residual=None,
                valid_overlap_ratio=valid_overlap_ratio,
                reason="position_overlap_below_gate",
            )
        if slot.rotation_period_deg is not None:
            observed_roi_x1 = max(0, int(round(roi_x1 + dx_px)))
            observed_roi_y1 = max(0, int(round(roi_y1 + dy_px)))
            observed_roi_x2 = min(image.shape[1], observed_roi_x1 + (roi_x2 - roi_x1))
            observed_roi_y2 = min(image.shape[0], observed_roi_y1 + (roi_y2 - roi_y1))
            
            # Pad if the translated ROI hits the image boundary to maintain exact template shape
            observed_roi = np.zeros_like(template)
            observed_valid_mask = np.zeros_like(valid_mask[roi_y1:roi_y2, roi_x1:roi_x2])
            
            src_x1, src_y1 = max(0, observed_roi_x1), max(0, observed_roi_y1)
            src_x2, src_y2 = min(image.shape[1], observed_roi_x1 + (roi_x2 - roi_x1)), min(image.shape[0], observed_roi_y1 + (roi_y2 - roi_y1))
            
            if src_x2 > src_x1 and src_y2 > src_y1:
                dst_x1, dst_y1 = src_x1 - observed_roi_x1, src_y1 - observed_roi_y1
                dst_x2, dst_y2 = dst_x1 + (src_x2 - src_x1), dst_y1 + (src_y2 - src_y1)
                observed_roi[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
                observed_valid_mask[dst_y1:dst_y2, dst_x1:dst_x2] = valid_mask[src_y1:src_y2, src_x1:src_x2]
            
            # The local center MUST be the Golden component's center relative to the template (which is the unshifted ROI)
            local_center = (
                float(slot.expected_center_px[0] - roi_x1),
                float(slot.expected_center_px[1] - roi_y1),
            )
            
            # The remaining initial shift to pass to ECC is just the fractional subpixel part
            residual_dx = dx_px - (observed_roi_x1 - roi_x1)
            residual_dy = dy_px - (observed_roi_y1 - roi_y1)
            refined, failure = _refine_euclidean_pose(
                template,
                observed_roi,
                binary_mask,
                initial_dx=residual_dx,
                initial_dy=residual_dy,
                base_dx=float(observed_roi_x1 - roi_x1),
                base_dy=float(observed_roi_y1 - roi_y1),
                center_px=local_center,
                period_deg=float(slot.rotation_period_deg),
                gates=gates,
                observed_valid_mask=observed_valid_mask,
            )
            if refined is None:
                return PositionResult(
                    slot_id=slot.slot_id,
                    status="unmeasurable",
                    dx_px=None,
                    dy_px=None,
                    dx_mm=None,
                    dy_mm=None,
                    angle_deg=None,
                    score=float(peak),
                    peak_margin=peak_margin,
                    psr=psr,
                    pose_correlation=None,
                    pose_residual=None,
                    valid_overlap_ratio=None,
                    reason=failure,
                )
            (
                dx_px,
                dy_px,
                angle_deg,
                pose_correlation,
                pose_residual,
                valid_overlap_ratio,
            ) = refined
        elif tolerance.max_abs_angle_deg is not None:
            return PositionResult(
                slot_id=slot.slot_id,
                status="unmeasurable",
                dx_px=None,
                dy_px=None,
                dx_mm=None,
                dy_mm=None,
                angle_deg=None,
                score=float(peak),
                peak_margin=peak_margin,
                psr=psr,
                pose_correlation=None,
                pose_residual=None,
                valid_overlap_ratio=None,
                reason="rotation_not_measured",
            )
        dx_mm = dx_px / float(metrology.pixels_per_mm_x)
        dy_mm = dy_px / float(metrology.pixels_per_mm_y)
        passed = (
            abs(dx_mm) <= float(tolerance.max_abs_dx_mm)
            and abs(dy_mm) <= float(tolerance.max_abs_dy_mm)
            and (
                tolerance.max_abs_angle_deg is None
                or (
                    angle_deg is not None
                    and abs(angle_deg) <= float(tolerance.max_abs_angle_deg)
                )
            )
        )
        return PositionResult(
            slot_id=slot.slot_id,
            status="pass" if passed else "ng",
            dx_px=dx_px,
            dy_px=dy_px,
            dx_mm=dx_mm,
            dy_mm=dy_mm,
            angle_deg=angle_deg,
            score=float(peak),
            peak_margin=peak_margin,
            psr=psr,
            pose_correlation=pose_correlation,
            pose_residual=pose_residual,
            valid_overlap_ratio=valid_overlap_ratio,
            reason="within_tolerance" if passed else "position_tolerance_exceeded",
        )


def _refine_euclidean_pose(
    template: np.ndarray,
    observed: np.ndarray,
    component_mask: np.ndarray,
    *,
    initial_dx: float,
    initial_dy: float,
    base_dx: float,
    base_dy: float,
    center_px: tuple[float, float],
    period_deg: float,
    gates: PositionQualityGates,
    observed_valid_mask: np.ndarray,
) -> tuple[tuple[float, float, float, float, float, float] | None, str]:
    """Refine Golden-to-observed translation/rotation without scale or shear."""

    if observed.shape != template.shape or observed_valid_mask.shape != template.shape[:2]:
        return None, "pose_roi_geometry_mismatch"
    template_repr = _normalized_gradient(template)
    observed_repr = _normalized_gradient(observed)
    # Permit the component to move inside its search neighborhood while keeping
    # static PCB background from dominating the Euclidean refinement.
    dilation = max(3, int(round(max(abs(initial_dx), abs(initial_dy)) * 2.0 + 5.0)))
    if dilation % 2 == 0:
        dilation += 1
    ecc_mask = cv2.dilate(
        component_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation)),
    )
    ecc_mask = np.where(observed_valid_mask > 0, ecc_mask, 0).astype(np.uint8)
    if not np.any(ecc_mask > 0):
        return None, "pose_overlap_below_gate"
    warp = np.float32([[1.0, 0.0, initial_dx], [0.0, 1.0, initial_dy]])
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(gates.ecc_iterations),
        float(gates.ecc_epsilon),
    )
    try:
        correlation, warp = cv2.findTransformECC(
            template_repr,
            observed_repr,
            warp,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            ecc_mask,
            3,
        )
    except cv2.error:
        return None, "euclidean_pose_refinement_failed"
    warp = np.asarray(warp, dtype=np.float64)
    if warp.shape != (2, 3) or not np.all(np.isfinite(warp)):
        return None, "euclidean_pose_transform_invalid"

    raw_angle = -math.degrees(math.atan2(float(warp[1, 0]), float(warp[0, 0])))
    if abs(raw_angle) > float(gates.max_abs_rotation_deg):
        return None, "rotation_outside_quality_gate"
    angle_deg = _normalize_periodic_angle(raw_angle, period_deg)
    center = np.float64([center_px[0], center_px[1], 1.0])
    observed_center = warp @ center
    # The output dx/dy must be the total shift: base translation + subpixel ECC refinement
    dx_px = base_dx + float(observed_center[0] - center_px[0])
    dy_px = base_dy + float(observed_center[1] - center_px[1])

    compensated = cv2.warpAffine(
        observed_repr,
        warp,
        (template.shape[1], template.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
    )
    valid = cv2.warpAffine(
        np.where(observed_valid_mask > 0, 255, 0).astype(np.uint8),
        warp,
        (template.shape[1], template.shape[0]),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
    )
    evaluation_mask = (component_mask > 0) & (valid > 0)
    component_count = int(np.count_nonzero(component_mask))
    overlap = float(np.count_nonzero(evaluation_mask) / max(component_count, 1))
    residual = (
        float(np.mean(np.abs(template_repr[evaluation_mask] - compensated[evaluation_mask])))
        if np.any(evaluation_mask)
        else 1.0
    )
    correlation = float(correlation)
    failures: list[str] = []
    if not math.isfinite(correlation) or correlation < float(gates.min_pose_correlation):
        failures.append("pose_correlation_below_gate")
    if not math.isfinite(residual) or residual > float(gates.max_pose_residual):
        failures.append("pose_residual_above_gate")
    if overlap < float(gates.min_valid_overlap_ratio):
        failures.append("pose_overlap_below_gate")
    if failures:
        return None, ";".join(failures)
    return (dx_px, dy_px, angle_deg, correlation, residual, overlap), ""


def _translation_valid_overlap(
    global_valid_mask: np.ndarray,
    component_mask: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    dx_px: float,
    dy_px: float,
) -> float:
    """Return valid observed coverage for a Golden component translated by dx/dy."""

    height, width = global_valid_mask.shape[:2]
    golden_to_observed = np.float32([[1.0, 0.0, dx_px], [0.0, 1.0, dy_px]])
    valid_in_golden = cv2.warpAffine(
        global_valid_mask,
        golden_to_observed,
        (width, height),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
    )
    x1, y1, x2, y2 = roi_xyxy
    component = component_mask > 0
    if component.shape != (y2 - y1, x2 - x1):
        return 0.0
    valid = valid_in_golden[y1:y2, x1:x2] > 0
    return float(np.count_nonzero(component & valid) / max(1, np.count_nonzero(component)))


def _normalized_gradient(image: np.ndarray) -> np.ndarray:
    gradient = _gradient_representation(image)
    maximum = float(np.max(gradient))
    if maximum <= 1e-6:
        return np.zeros_like(gradient, dtype=np.float32)
    return np.asarray(gradient / maximum, dtype=np.float32)


def _normalize_periodic_angle(angle_deg: float, period_deg: float) -> float:
    """Map a relative angle into the nearest equivalent signed period."""

    half = period_deg / 2.0
    normalized = (float(angle_deg) + half) % period_deg - half
    return float(normalized)


def _gradient_representation(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(grad_x, grad_y)


def _quadratic_offset(values: np.ndarray, index: int) -> float:
    if index <= 0 or index >= len(values) - 1:
        return 0.0
    left = float(values[index - 1])
    center = float(values[index])
    right = float(values[index + 1])
    denominator = left - 2.0 * center + right
    if not math.isfinite(denominator) or abs(denominator) < 1e-12:
        return 0.0
    offset = 0.5 * (left - right) / denominator
    return float(np.clip(offset, -1.0, 1.0)) if math.isfinite(offset) else 0.0


def _peak_quality(
    response: np.ndarray, peak_x: int, peak_y: int, peak: float
) -> tuple[float, float]:
    sidelobes = response.copy()
    y1, y2 = max(0, peak_y - 1), min(response.shape[0], peak_y + 2)
    x1, x2 = max(0, peak_x - 1), min(response.shape[1], peak_x + 2)
    sidelobes[y1:y2, x1:x2] = np.nan
    finite = sidelobes[np.isfinite(sidelobes)]
    if finite.size == 0:
        # Keep runtime JSON standards-compliant; this finite sentinel still
        # means the single available peak has no competing sidelobe.
        return 1.0, 1_000_000.0
    second = float(np.max(finite))
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    margin = max(0.0, float(peak) - second)
    psr = (float(peak) - mean) / max(std, 1e-6)
    return margin, psr


_read_asset = read_asset_under_root


def _invalid(slot: SlotRecipe, status: str, reason: str) -> PositionResult:
    return PositionResult(
        slot_id=slot.slot_id,
        status=status,
        dx_px=None,
        dy_px=None,
        dx_mm=None,
        dy_mm=None,
        angle_deg=None,
        score=None,
        peak_margin=None,
        psr=None,
        pose_correlation=None,
        pose_residual=None,
        valid_overlap_ratio=None,
        reason=reason,
    )


__all__ = ["PositionMeasurer", "PositionQualityGates", "PositionResult"]
