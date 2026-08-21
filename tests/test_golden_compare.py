from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from aoi_pipeline import BoundingBox, Detection
from aoi_pipeline.golden_compare import (
    GoldenComparator,
    GoldenCompareConfig,
    _masked_ssim,
)
from aoi_pipeline.position import PositionMeasurer, PositionQualityGates
from aoi_pipeline.recipe import (
    AppearanceThresholds,
    MetrologyCalibration,
    PositionTolerance,
    create_recipe,
)


def _golden() -> np.ndarray:
    image = np.full((150, 190, 3), (32, 92, 48), dtype=np.uint8)
    rng = np.random.default_rng(42)
    component = rng.integers(25, 220, size=(38, 52, 3), dtype=np.uint8)
    component = cv2.GaussianBlur(component, (3, 3), 0.5)
    cv2.rectangle(component, (1, 1), (50, 36), (220, 210, 40), 2)
    cv2.circle(component, (16, 19), 8, (20, 30, 235), -1)
    cv2.line(component, (29, 6), (44, 31), (240, 55, 45), 4)
    image[56:94, 69:121] = component
    return image


def _setup(tmp_path: Path, *, rotation: bool = False):
    golden = _golden()
    candidate = Detection(
        "ic",
        0.96,
        BoundingBox(69, 56, 121, 94),
        class_id=10,
        source="ultralytics",
    )
    recipe = create_recipe(
        golden,
        [candidate],
        tmp_path,
        metrology=MetrologyCalibration(20.0, 20.0),
        roi_padding_px=9,
        search_margin_px=8,
        position_tolerance=PositionTolerance(0.025, 0.025, 6.0 if rotation else None),
        appearance_thresholds=AppearanceThresholds(
            min_ssim=0.88,
            max_diff_ratio=0.08,
            max_edge_diff_ratio=0.10,
            max_blob_area_px=45,
            min_valid_overlap_ratio=0.88,
        ),
    ).recipe
    if rotation:
        recipe = replace(
            recipe,
            slots=(replace(recipe.slots[0], rotation_period_deg=360.0),),
        )
    position = PositionMeasurer(
        PositionQualityGates(
            min_score=0.50,
            min_peak_margin=0.004,
            min_psr=2.0,
            min_pose_correlation=0.65,
            max_pose_residual=0.40,
        )
    )
    comparator = GoldenComparator(
        GoldenCompareConfig(
            pixel_diff_threshold=24,
            morphology_kernel=3,
            min_blob_area_px=3,
        )
    )
    return golden, recipe, candidate, position, comparator


def _warp(
    image: np.ndarray,
    center: tuple[float, float],
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    angle_deg: float = 0.0,
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


def _measure(position, image, recipe, root, candidate):
    return position.measure(
        image,
        recipe.slots[0],
        root,
        recipe.metrology,
        candidate=candidate,
    )


def test_same_image_appearance_passes(tmp_path: Path) -> None:
    golden, recipe, candidate, position, comparator = _setup(tmp_path)
    pose = _measure(position, golden, recipe, tmp_path, candidate)

    result = comparator.compare(golden, recipe.slots[0], tmp_path, pose)

    assert pose.status == "pass"
    assert result.status == "pass"
    assert result.ssim == pytest.approx(1.0, abs=1e-6)
    assert result.diff_ratio == pytest.approx(0.0)
    assert result.edge_diff_ratio == pytest.approx(0.0)
    assert result.max_blob_area_px == 0


def test_shifted_only_is_position_ng_but_appearance_pass_after_compensation(
    tmp_path: Path,
) -> None:
    golden, recipe, candidate, position, comparator = _setup(tmp_path)
    observed = _warp(golden, recipe.slots[0].expected_center_px, dx=2.0, dy=-1.25)
    pose = _measure(position, observed, recipe, tmp_path, candidate)

    result = comparator.compare(observed, recipe.slots[0], tmp_path, pose)

    assert pose.status == "ng"
    assert pose.dx_px == pytest.approx(2.0, abs=0.25)
    assert pose.dy_px == pytest.approx(-1.25, abs=0.25)
    assert result.status == "pass", result.to_dict()


def test_rotated_and_shifted_only_passes_appearance_after_euclidean_compensation(
    tmp_path: Path,
) -> None:
    golden, recipe, candidate, position, comparator = _setup(tmp_path, rotation=True)
    observed = _warp(
        golden,
        recipe.slots[0].expected_center_px,
        dx=1.25,
        dy=-0.75,
        angle_deg=4.0,
    )
    pose = _measure(position, observed, recipe, tmp_path, candidate)

    result = comparator.compare(observed, recipe.slots[0], tmp_path, pose)

    assert pose.measurable, pose.reason
    assert pose.angle_deg == pytest.approx(4.0, abs=0.4)
    assert result.status == "pass", result.to_dict()


def test_shape_and_color_change_is_appearance_anomaly(tmp_path: Path) -> None:
    golden, recipe, candidate, position, comparator = _setup(tmp_path)
    observed = golden.copy()
    cv2.rectangle(observed, (88, 66), (108, 84), (255, 0, 255), -1)
    cv2.circle(observed, (79, 75), 5, (0, 255, 255), -1)
    pose = _measure(position, observed, recipe, tmp_path, candidate)

    result = comparator.compare(observed, recipe.slots[0], tmp_path, pose)

    assert pose.measurable, pose.reason
    assert result.status == "anomaly"
    assert result.defect_label == "appearance_anomaly"
    assert result.anomaly_blob_count is not None and result.anomaly_blob_count >= 1
    assert result.max_blob_area_px is not None and result.max_blob_area_px > 45


def test_small_global_illumination_change_does_not_trigger_anomaly(tmp_path: Path) -> None:
    golden, recipe, candidate, position, comparator = _setup(tmp_path)
    observed = np.clip(golden.astype(np.float32) * 1.10 + 6.0, 0, 255).astype(np.uint8)
    pose = _measure(position, observed, recipe, tmp_path, candidate)

    result = comparator.compare(observed, recipe.slots[0], tmp_path, pose)

    assert pose.measurable, pose.reason
    assert result.status == "pass", result.to_dict()


def test_missing_or_low_confidence_pose_is_not_evaluated(tmp_path: Path) -> None:
    golden, recipe, _, position, comparator = _setup(tmp_path)
    pose = _measure(position, golden, recipe, tmp_path, None)

    result = comparator.compare(golden, recipe.slots[0], tmp_path, pose)

    assert pose.status == "missing_candidate"
    assert result.status == "not_evaluated"
    assert result.ssim is None
    assert result.anomaly_mask is None
    assert result.reason == "position_not_measurable"


def test_ignore_mask_excludes_known_unstable_region(tmp_path: Path) -> None:
    golden, recipe, candidate, position, comparator = _setup(tmp_path)
    observed = golden.copy()
    cv2.rectangle(observed, (92, 69), (108, 84), (255, 0, 255), -1)
    pose = _measure(position, observed, recipe, tmp_path, candidate)
    slot = recipe.slots[0]
    x1, y1, x2, y2 = slot.fixed_roi_xyxy.to_int()
    ignore = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.rectangle(ignore, (92 - x1, 69 - y1), (108 - x1, 84 - y1), 255, -1)
    ignore_path = tmp_path / "masks" / "slot_0001_ignore.png"
    assert cv2.imwrite(str(ignore_path), ignore)
    slot = replace(slot, ignore_mask_path="masks/slot_0001_ignore.png")

    result = comparator.compare(observed, slot, tmp_path, pose)

    assert result.status == "pass", result.to_dict()
    assert result.valid_overlap_ratio == pytest.approx(1.0)


def test_global_alignment_valid_mask_excludes_padded_roi_pixels(tmp_path: Path) -> None:
    golden, recipe, candidate, position, comparator = _setup(tmp_path)
    pose = _measure(position, golden, recipe, tmp_path, candidate)
    global_valid_mask = np.full(golden.shape[:2], 255, dtype=np.uint8)
    x1, y1, x2, y2 = recipe.slots[0].fixed_roi_xyxy.to_int()
    global_valid_mask[y1:y2, x1:x2] = 0

    result = comparator.compare(
        golden,
        recipe.slots[0],
        tmp_path,
        pose,
        global_valid_mask=global_valid_mask,
    )

    assert result.status == "not_evaluated"
    assert result.reason == "valid_overlap_below_threshold"
    assert result.valid_overlap_ratio == pytest.approx(0.0)
    assert result.ssim is None
    assert result.anomaly_mask is None


def test_ssim_uses_local_windows_instead_of_global_roi_statistics() -> None:
    golden = np.full((96, 96), 128, dtype=np.uint8)
    clustered = golden.copy()
    clustered[44:52, 44:52] = 0
    distributed = golden.copy()
    coordinates = [
        (y, x)
        for y in range(8, 96, 11)
        for x in range(8, 96, 11)
    ]
    for y, x in coordinates[:64]:
        distributed[y, x] = 0
    mask = np.ones(golden.shape, dtype=bool)

    clustered_score = _masked_ssim(golden, clustered, mask)
    distributed_score = _masked_ssim(golden, distributed, mask)

    # Both observations have the same histogram and therefore identical
    # global mean/variance statistics. A local-window map remains sensitive to
    # whether the changed pixels are clustered or spread across the ROI.
    assert np.array_equal(
        np.sort(clustered.reshape(-1)), np.sort(distributed.reshape(-1))
    )
    assert distributed_score < clustered_score - 0.20
