from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

import aoi_pipeline.imaging.alignment as alignment_module
from aoi_pipeline import BoundingBox, Detection, PCBAligner
from aoi_pipeline.imaging.alignment import AnchorMatch
from aoi_pipeline.golden.recipe import (
    AlignmentAnchor,
    AlignmentQualityGates,
    AlignmentRecipe,
    AppearanceThresholds,
    MetrologyCalibration,
    PositionTolerance,
    create_recipe,
    save_recipe,
    validate_recipe_assets,
)


def _textured_golden() -> np.ndarray:
    rng = np.random.default_rng(20260818)
    image = rng.integers(15, 235, size=(180, 240, 3), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    cv2.rectangle(image, (4, 4), (235, 175), (20, 130, 30), 3)
    return image


def _recipe_with_anchors(tmp_path: Path):
    golden = _textured_golden()
    built = create_recipe(
        golden,
        [Detection("ic", 0.95, BoundingBox(92, 72, 142, 112), source="ultralytics")],
        tmp_path,
        metrology=MetrologyCalibration(40.0, 40.0),
        roi_padding_px=4,
        search_margin_px=8,
        position_tolerance=PositionTolerance(0.1, 0.1, 2.0),
        appearance_thresholds=AppearanceThresholds(0.9, 0.1, 0.1, 20, 0.8),
    )
    anchors: list[AlignmentAnchor] = []
    for index, (center_x, center_y) in enumerate(
        ((35, 35), (202, 38), (42, 145), (198, 142)), start=1
    ):
        template_bbox = BoundingBox(center_x - 10, center_y - 10, center_x + 11, center_y + 11)
        search_roi = BoundingBox(center_x - 20, center_y - 20, center_x + 21, center_y + 21)
        x1, y1, x2, y2 = template_bbox.to_int()
        relative_path = f"anchors/anchor_{index:04d}.png"
        output = tmp_path / relative_path
        output.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(output), golden[y1:y2, x1:x2])
        anchors.append(
            AlignmentAnchor(
                anchor_id=f"anchor_{index:04d}",
                reference_point_px=(float(center_x), float(center_y)),
                template_bbox_xyxy=template_bbox,
                search_roi_xyxy=search_roi,
                template_path=relative_path,
            )
        )
    alignment = AlignmentRecipe(
        anchors=tuple(anchors),
        quality_gates=AlignmentQualityGates(
            min_anchors=4,
            min_anchor_score=0.60,
            max_residual_px=0.45,
            ransac_reprojection_threshold_px=0.75,
            min_inlier_ratio=1.0,
            min_scale=0.95,
            max_scale=1.05,
            max_abs_rotation_deg=5.0,
        ),
    )
    recipe = replace(
        built.recipe,
        alignment=alignment,
        asset_sha256={
            **built.recipe.asset_sha256,
            **{
                anchor.template_path: sha256(
                    (tmp_path / anchor.template_path).read_bytes()
                ).hexdigest()
                for anchor in anchors
            },
        },
    )
    save_recipe(recipe, tmp_path / "recipe.json")
    validate_recipe_assets(recipe, tmp_path)
    return golden, recipe


def test_strict_anchor_alignment_recovers_known_translation(tmp_path: Path) -> None:
    golden, recipe = _recipe_with_anchors(tmp_path)
    dx, dy = 4.25, -3.5
    shifted = cv2.warpAffine(
        golden,
        np.float32([[1, 0, dx], [0, 1, dy]]),
        (golden.shape[1], golden.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    result = PCBAligner().align_to_recipe(shifted, recipe, tmp_path)

    assert result.status == "valid"
    assert result.image is not None
    assert result.valid_mask is not None
    assert result.valid_mask.shape == golden.shape[:2]
    assert 0 < np.count_nonzero(result.valid_mask) < result.valid_mask.size
    assert result.transform is not None
    assert result.matched_anchors == 4
    assert result.inliers == 4
    assert result.residual_px is not None and result.residual_px < 0.20
    assert result.transform[0, 2] == pytest.approx(-dx, abs=0.25)
    assert result.transform[1, 2] == pytest.approx(-dy, abs=0.25)
    before = np.mean(np.abs(shifted.astype(np.float32) - golden.astype(np.float32)))
    after = np.mean(np.abs(result.image.astype(np.float32) - golden.astype(np.float32)))
    assert after < before * 0.35


def test_strict_anchor_alignment_recovers_small_rotation_and_translation(
    tmp_path: Path,
) -> None:
    golden, recipe = _recipe_with_anchors(tmp_path)
    center = (golden.shape[1] / 2.0, golden.shape[0] / 2.0)
    golden_to_source = cv2.getRotationMatrix2D(center, 1.25, 1.0)
    golden_to_source[:, 2] += (2.5, -1.75)
    transformed = cv2.warpAffine(
        golden,
        golden_to_source,
        (golden.shape[1], golden.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    expected = cv2.invertAffineTransform(golden_to_source)
    expected_rotation = math.degrees(math.atan2(expected[1, 0], expected[0, 0]))

    result = PCBAligner().align_to_recipe(transformed, recipe, tmp_path)

    assert result.status == "valid"
    assert result.transform is not None
    assert result.rotation_deg == pytest.approx(expected_rotation, abs=0.25)
    assert result.transform[:2] == pytest.approx(expected, abs=0.35)


def test_strict_anchor_alignment_fails_closed_when_anchors_are_missing(tmp_path: Path) -> None:
    golden, recipe = _recipe_with_anchors(tmp_path)
    blank = np.full_like(golden, 127)

    result = PCBAligner().align_to_recipe(blank, recipe, tmp_path)

    assert result.status == "invalid"
    assert result.image is None
    assert result.transform is None
    assert result.matched_anchors < recipe.alignment.quality_gates.min_anchors
    assert "anchors" in result.reason.lower()


def test_strict_anchor_alignment_rejects_implausible_scale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    golden, recipe = _recipe_with_anchors(tmp_path)

    def implausible_affine(*args, **kwargs):
        return np.float64([[2.5, 0.0, 0.0], [0.0, 2.5, 0.0]]), np.ones((4, 1), np.uint8)

    monkeypatch.setattr(cv2, "estimateAffinePartial2D", implausible_affine)
    result = PCBAligner().align_to_recipe(golden, recipe, tmp_path)

    assert result.status == "invalid"
    assert result.image is None
    assert result.transform is None
    assert result.scale == pytest.approx(2.5)
    assert "scale" in result.reason.lower()


def test_strict_alignment_rejects_low_canvas_overlap_even_with_zero_residual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    golden, recipe = _recipe_with_anchors(tmp_path)

    def repeated_wrong_offset(source, anchor, root, minimum_score):
        del source, root, minimum_score
        reference_x, reference_y = anchor.reference_point_px
        return AnchorMatch(
            anchor_id=anchor.anchor_id,
            reference_point_px=anchor.reference_point_px,
            observed_point_px=(reference_x - 1000.0, reference_y),
            score=0.99,
            status="matched",
        )

    monkeypatch.setattr(alignment_module, "_measure_anchor", repeated_wrong_offset)
    result = PCBAligner().align_to_recipe(golden, recipe, tmp_path)

    assert result.status == "invalid"
    assert result.image is None
    assert result.transform is None
    assert result.residual_px == pytest.approx(0.0, abs=1e-4)
    assert result.scale == pytest.approx(1.0, abs=1e-4)
    assert result.canvas_overlap_ratio == pytest.approx(0.0)
    assert "overlap" in result.reason.lower()


def test_existing_demo_alignment_resize_fallback_is_unchanged() -> None:
    source = np.zeros((80, 100, 3), dtype=np.uint8)
    target = np.zeros((60, 90, 3), dtype=np.uint8)

    result = PCBAligner().align(source, target)

    assert result.success is False
    assert result.method == "resize_fallback"
    assert result.image.shape == target.shape
