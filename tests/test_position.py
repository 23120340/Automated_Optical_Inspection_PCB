from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from aoi_pipeline import BoundingBox, Detection
from aoi_pipeline.golden.position import PositionMeasurer, PositionQualityGates
from aoi_pipeline.golden.recipe import (
    AppearanceThresholds,
    MetrologyCalibration,
    PositionTolerance,
    create_recipe,
)


def _golden() -> np.ndarray:
    image = np.full((140, 180, 3), (35, 85, 42), dtype=np.uint8)
    rng = np.random.default_rng(20260818)
    component = rng.integers(20, 235, size=(32, 44, 3), dtype=np.uint8)
    component = cv2.GaussianBlur(component, (3, 3), 0.45)
    cv2.rectangle(component, (1, 1), (42, 30), (230, 220, 35), 2)
    cv2.circle(component, (13, 16), 7, (25, 25, 230), -1)
    cv2.line(component, (25, 5), (38, 27), (240, 60, 50), 3)
    image[54:86, 68:112] = component
    return image


def _setup(tmp_path: Path, *, bbox: BoundingBox | None = None):
    golden = _golden()
    detection = Detection(
        "ic",
        0.95,
        bbox or BoundingBox(68, 54, 112, 86),
        class_id=10,
        source="ultralytics",
    )
    built = create_recipe(
        golden,
        [detection],
        tmp_path,
        metrology=MetrologyCalibration(20.0, 25.0),
        roi_padding_px=7,
        search_margin_px=6,
        position_tolerance=PositionTolerance(0.04, 0.04, None),
        appearance_thresholds=AppearanceThresholds(0.9, 0.1, 0.1, 20, 0.8),
    )
    measurer = PositionMeasurer(
        PositionQualityGates(min_score=0.50, min_peak_margin=0.005, min_psr=2.5)
    )
    return golden, built.recipe, measurer, detection


def _shift(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return cv2.warpAffine(
        image,
        np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]]),
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _euclidean_shift(
    image: np.ndarray,
    center: tuple[float, float],
    angle_deg: float,
    dx: float,
    dy: float,
) -> np.ndarray:
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    matrix[:, 2] += (dx, dy)
    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


@pytest.mark.parametrize("dx", [-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("dy", [0.0])
def test_position_recovers_fractional_x_shifts_and_sign(
    tmp_path: Path, dx: float, dy: float
) -> None:
    golden, recipe, measurer, candidate = _setup(tmp_path)

    result = measurer.measure(
        _shift(golden, dx, dy),
        recipe.slots[0],
        tmp_path,
        recipe.metrology,
        candidate=candidate,
    )

    assert result.measurable, result.reason
    assert result.dx_px == pytest.approx(dx, abs=0.18)
    assert result.dy_px == pytest.approx(dy, abs=0.10)


@pytest.mark.parametrize("dy", [-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0])
def test_position_recovers_fractional_y_shifts_and_sign(tmp_path: Path, dy: float) -> None:
    golden, recipe, measurer, candidate = _setup(tmp_path)

    result = measurer.measure(
        _shift(golden, 0.0, dy),
        recipe.slots[0],
        tmp_path,
        recipe.metrology,
        candidate=candidate,
    )

    assert result.measurable, result.reason
    assert result.dx_px == pytest.approx(0.0, abs=0.10)
    assert result.dy_px == pytest.approx(dy, abs=0.18)


def test_position_converts_each_axis_to_mm_and_applies_tolerance(tmp_path: Path) -> None:
    golden, recipe, measurer, candidate = _setup(tmp_path)

    result = measurer.measure(
        _shift(golden, 1.0, -1.5),
        recipe.slots[0],
        tmp_path,
        recipe.metrology,
        candidate=candidate,
    )

    assert result.status == "ng"
    assert result.dx_mm == pytest.approx(0.05, abs=0.01)
    assert result.dy_mm == pytest.approx(-0.06, abs=0.01)
    assert result.reason == "position_tolerance_exceeded"


def test_position_same_image_passes(tmp_path: Path) -> None:
    golden, recipe, measurer, candidate = _setup(tmp_path)

    result = measurer.measure(
        golden,
        recipe.slots[0],
        tmp_path,
        recipe.metrology,
        candidate=candidate,
    )

    assert result.status == "pass"
    assert result.dx_px == pytest.approx(0.0, abs=0.05)
    assert result.dy_px == pytest.approx(0.0, abs=0.05)
    assert result.angle_deg is None
    assert result.coordinate_space == "golden_board_pixels"


def test_position_never_exports_measurement_from_invalid_aligned_pixels(
    tmp_path: Path,
) -> None:
    golden, recipe, measurer, candidate = _setup(tmp_path)
    slot = recipe.slots[0]
    valid_mask = np.full(golden.shape[:2], 255, dtype=np.uint8)
    x1, y1, x2, y2 = slot.fixed_roi_xyxy.to_int()
    valid_mask[y1:y2, x1:x2] = 0

    result = measurer.measure(
        golden,
        slot,
        tmp_path,
        recipe.metrology,
        candidate=candidate,
        global_valid_mask=valid_mask,
    )

    assert result.status == "unmeasurable"
    assert result.reason == "position_overlap_below_gate"
    assert result.valid_overlap_ratio == 0.0
    assert result.dx_px is None and result.dy_px is None
    assert result.dx_mm is None and result.dy_mm is None


def test_translation_only_does_not_pass_when_rotation_tolerance_is_required(
    tmp_path: Path,
) -> None:
    golden, recipe, measurer, candidate = _setup(tmp_path)
    slot = replace(
        recipe.slots[0],
        position_tolerance=PositionTolerance(0.04, 0.04, 2.0),
    )

    result = measurer.measure(
        golden,
        slot,
        tmp_path,
        recipe.metrology,
        candidate=candidate,
    )

    assert result.status == "unmeasurable"
    assert result.angle_deg is None
    assert result.dx_px is None
    assert result.dy_px is None
    assert result.reason == "rotation_not_measured"


@pytest.mark.parametrize("angle_deg", [-6.0, -3.0, 2.5, 5.5])
def test_euclidean_pose_recovers_known_rotation_and_translation(
    tmp_path: Path, angle_deg: float
) -> None:
    golden, recipe, measurer, candidate = _setup(tmp_path)
    slot = replace(
        recipe.slots[0],
        rotation_period_deg=360.0,
        position_tolerance=PositionTolerance(0.15, 0.15, 8.0),
    )
    dx, dy = 1.25, -0.75
    observed = _euclidean_shift(
        golden,
        slot.expected_center_px,
        angle_deg,
        dx,
        dy,
    )

    result = measurer.measure(
        observed,
        slot,
        tmp_path,
        recipe.metrology,
        candidate=candidate,
    )

    assert result.measurable, result.reason
    assert result.dx_px == pytest.approx(dx, abs=0.35)
    assert result.dy_px == pytest.approx(dy, abs=0.35)
    assert result.angle_deg == pytest.approx(angle_deg, abs=0.35)
    assert result.pose_correlation is not None and result.pose_correlation > 0.8
    assert result.pose_residual is not None
    assert result.valid_overlap_ratio is not None and result.valid_overlap_ratio > 0.9


def test_rotation_period_180_reports_nearest_equivalent_angle(tmp_path: Path) -> None:
    golden, recipe, measurer, candidate = _setup(tmp_path)
    slot = replace(
        recipe.slots[0],
        rotation_period_deg=180.0,
        position_tolerance=PositionTolerance(0.15, 0.15, 5.0),
    )
    # A nearly 180-degree placement is equivalent to -2 degrees for a slot
    # explicitly declared 180-periodic. Rotate the whole component patch twice
    # symmetrically so the image evidence has the same periodicity.
    x1, y1, x2, y2 = slot.fixed_roi_xyxy.to_int()
    periodic_golden = golden.copy()
    roi = periodic_golden[y1:y2, x1:x2]
    roi_180 = cv2.rotate(roi, cv2.ROTATE_180)
    periodic_golden[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.5, roi_180, 0.5, 0)
    # Refresh the persisted template to match this synthetic periodic Golden.
    assert cv2.imwrite(str(tmp_path / slot.template_path), periodic_golden[y1:y2, x1:x2])
    observed = _euclidean_shift(
        periodic_golden,
        slot.expected_center_px,
        178.0,
        0.0,
        0.0,
    )

    result = measurer.measure(
        observed,
        slot,
        tmp_path,
        recipe.metrology,
        candidate=candidate,
    )

    assert result.measurable, result.reason
    assert result.angle_deg == pytest.approx(-2.0, abs=0.6)


def test_missing_candidate_never_fabricates_numeric_measurement(tmp_path: Path) -> None:
    golden, recipe, measurer, _ = _setup(tmp_path)

    result = measurer.measure(
        golden,
        recipe.slots[0],
        tmp_path,
        recipe.metrology,
        candidate=None,
    )

    assert result.status == "missing_candidate"
    assert result.measurable is False
    assert result.dx_px is None
    assert result.dy_px is None
    assert result.dx_mm is None
    assert result.dy_mm is None


def test_low_quality_match_is_unmeasurable_without_fake_zero(tmp_path: Path) -> None:
    golden, recipe, measurer, candidate = _setup(tmp_path)
    blank = np.full_like(golden, 90)

    result = measurer.measure(
        blank,
        recipe.slots[0],
        tmp_path,
        recipe.metrology,
        candidate=candidate,
    )

    assert result.status == "unmeasurable"
    assert result.dx_px is None
    assert result.dy_px is None


def test_runtime_detector_bbox_jitter_does_not_recenter_position_roi(tmp_path: Path) -> None:
    golden, recipe, measurer, candidate = _setup(tmp_path)
    shifted = _shift(golden, 1.25, -0.75)
    jittered = Detection(
        candidate.label,
        candidate.confidence,
        BoundingBox(74, 49, 119, 83),
        class_id=candidate.class_id,
        source=candidate.source,
    )

    baseline = measurer.measure(
        shifted,
        recipe.slots[0],
        tmp_path,
        recipe.metrology,
        candidate=candidate,
    )
    with_jitter = measurer.measure(
        shifted,
        recipe.slots[0],
        tmp_path,
        recipe.metrology,
        candidate=jittered,
    )

    assert baseline.measurable and with_jitter.measurable
    assert with_jitter.dx_px == pytest.approx(baseline.dx_px, abs=1e-9)
    assert with_jitter.dy_px == pytest.approx(baseline.dy_px, abs=1e-9)


def test_fixed_roi_near_image_boundary_remains_measurable(tmp_path: Path) -> None:
    golden = np.full((80, 100, 3), 40, np.uint8)
    rng = np.random.default_rng(5)
    golden[5:25, 3:26] = rng.integers(0, 255, size=(20, 23, 3), dtype=np.uint8)
    candidate = Detection("connector", 0.9, BoundingBox(3, 5, 26, 25))
    recipe = create_recipe(
        golden,
        [candidate],
        tmp_path,
        metrology=MetrologyCalibration(10, 10),
        roi_padding_px=6,
        search_margin_px=4,
        position_tolerance=PositionTolerance(0.2, 0.2),
        appearance_thresholds=AppearanceThresholds(0.9, 0.1, 0.1, 10, 0.8),
    ).recipe
    result = PositionMeasurer(
        PositionQualityGates(min_score=0.4, min_peak_margin=0.002, min_psr=2.0)
    ).measure(
        _shift(golden, 1.0, 1.0),
        recipe.slots[0],
        tmp_path,
        recipe.metrology,
        candidate=candidate,
    )

    assert result.measurable, result.reason
    assert result.dx_px == pytest.approx(1.0, abs=0.2)
    assert result.dy_px == pytest.approx(1.0, abs=0.2)
