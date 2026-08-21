from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from aoi_pipeline import BoundingBox, Detection
from aoi_pipeline.imaging.image_io import read_asset_under_root
from aoi_pipeline.golden.recipe import (
    AppearanceThresholds,
    MetrologyCalibration,
    PositionTolerance,
    RecipeValidationError,
    create_grid_alignment_recipe,
    create_recipe,
    load_recipe,
    save_recipe,
    validate_recipe_assets,
)


def _golden() -> np.ndarray:
    y, x = np.indices((80, 120), dtype=np.uint8)
    return np.dstack((x, y, (x.astype(np.uint16) + y) % 256)).astype(np.uint8)


def _position_tolerance() -> PositionTolerance:
    return PositionTolerance(max_abs_dx_mm=0.10, max_abs_dy_mm=0.12, max_abs_angle_deg=2.0)


def _appearance_thresholds() -> AppearanceThresholds:
    return AppearanceThresholds(
        min_ssim=0.90,
        max_diff_ratio=0.08,
        max_edge_diff_ratio=0.10,
        max_blob_area_px=25,
        min_valid_overlap_ratio=0.85,
    )


def _create(tmp_path: Path, detections: list[Detection]):
    return create_recipe(
        _golden(),
        detections,
        tmp_path,
        board_id="BOARD_A",
        side="top",
        metrology=MetrologyCalibration(40.0, 42.0),
        roi_padding_px=3,
        search_margin_px=6,
        position_tolerance=_position_tolerance(),
        appearance_thresholds=_appearance_thresholds(),
        model_identifiers={"component_detector": "detector-best.onnx:a26151e0"},
    )


def test_recipe_slots_are_stable_and_independent_of_detection_order(tmp_path: Path) -> None:
    detections = [
        Detection("ic", 0.88, BoundingBox(70.5, 42.0, 91.0, 60.0), class_id=10),
        Detection("resistor", 0.93, BoundingBox(12.0, 10.0, 30.0, 20.0), class_id=17),
        Detection("capacitor", 0.91, BoundingBox(55.0, 9.0, 65.0, 22.0), class_id=3),
    ]
    first = _create(tmp_path / "first", detections).recipe
    second = _create(tmp_path / "second", list(reversed(detections))).recipe

    assert [slot.slot_id for slot in first.slots] == ["slot_0001", "slot_0002", "slot_0003"]
    assert [slot.label_hint for slot in first.slots] == ["resistor", "capacitor", "ic"]
    assert first.to_dict() == second.to_dict()
    assert all("source_detection_id" not in slot.to_dict() for slot in first.slots)


def test_recipe_clamps_roi_and_persists_pixel_native_assets(tmp_path: Path) -> None:
    result = _create(
        tmp_path,
        [Detection("connector", 0.9, BoundingBox(-2.4, 4.2, 9.6, 15.1), class_id=5)],
    )
    slot = result.recipe.slots[0]
    assert slot.expected_bbox_xyxy.as_xyxy() == [0.0, 4.2, 9.6, 15.1]
    assert slot.fixed_roi_xyxy.as_xyxy() == [0.0, 1.0, 13.0, 19.0]

    # Through the project's reader, not a bare cv2.imread: production never
    # calls imread directly here, and cv2.imread is not necessarily OpenCV's --
    # importing ultralytics replaces it with a version that hands back a
    # trailing channel axis for grayscale. A test that reads the asset a
    # different way from the code under test is testing a different thing.
    template = read_asset_under_root(tmp_path, slot.template_path, cv2.IMREAD_COLOR)
    component_mask = read_asset_under_root(
        tmp_path, slot.component_mask_path, cv2.IMREAD_GRAYSCALE
    )
    compare_mask = read_asset_under_root(
        tmp_path, slot.compare_mask_path, cv2.IMREAD_GRAYSCALE
    )
    assert template.shape == (18, 13, 3)
    assert np.array_equal(template, _golden()[1:19, 0:13])
    assert component_mask.shape == template.shape[:2]
    assert np.array_equal(component_mask, compare_mask)
    assert component_mask[0, 0] == 0
    assert component_mask[4, 1] == 255


def test_grid_alignment_enrollment_is_deterministic_and_lossless(tmp_path: Path) -> None:
    fractions = ((0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85))
    first = create_grid_alignment_recipe(
        _golden(),
        tmp_path / "first",
        template_size_px=15,
        search_margin_px=6,
        grid_fractions=fractions,
    )
    second = create_grid_alignment_recipe(
        _golden(),
        tmp_path / "second",
        template_size_px=15,
        search_margin_px=6,
        grid_fractions=fractions,
    )

    assert first.to_dict() == second.to_dict()
    assert [item.anchor_id for item in first.anchors] == [
        "anchor_0001",
        "anchor_0002",
        "anchor_0003",
        "anchor_0004",
    ]
    for anchor in first.anchors:
        path = tmp_path / "first" / anchor.template_path
        assert path.suffix == ".png"
        assert cv2.imread(str(path), cv2.IMREAD_COLOR).shape == (15, 15, 3)


def test_demo_grid_anchors_never_certify_production_recipe(tmp_path: Path) -> None:
    golden = _golden()
    alignment = create_grid_alignment_recipe(
        golden,
        tmp_path,
        template_size_px=15,
        search_margin_px=6,
        grid_fractions=((0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)),
    )
    result = create_recipe(
        golden,
        [Detection("ic", 0.9, BoundingBox(20, 20, 40, 40), source="ultralytics")],
        tmp_path,
        board_id="BOARD_A",
        side="top",
        metrology=MetrologyCalibration(40.0, 40.0, verified=True),
        roi_padding_px=3,
        search_margin_px=6,
        position_tolerance=_position_tolerance(),
        appearance_thresholds=_appearance_thresholds(),
        alignment=alignment,
        model_identifiers={"component_detector": "best.onnx:test-sha"},
    )

    assert alignment.anchor_provenance == "demo_grid"
    assert result.recipe.production_eligible is False
    payload = result.recipe.to_dict()
    payload["production_eligible"] = True
    path = tmp_path / "forged-production.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RecipeValidationError, match="approved stable alignment anchors"):
        load_recipe(path, validate_assets=False)


def test_recipe_can_enroll_rotation_measurement_contract(tmp_path: Path) -> None:
    result = create_recipe(
        _golden(),
        [Detection("ic", 0.9, BoundingBox(20, 20, 40, 40), class_id=10)],
        tmp_path,
        metrology=MetrologyCalibration(40.0, 40.0, verified=False),
        roi_padding_px=3,
        search_margin_px=6,
        position_tolerance=_position_tolerance(),
        appearance_thresholds=_appearance_thresholds(),
        rotation_period_deg=180.0,
    )

    assert result.recipe.slots[0].expected_angle_deg == 0.0
    assert result.recipe.slots[0].rotation_period_deg == 180.0


def test_recipe_round_trips_and_validates_optional_ignore_mask(tmp_path: Path) -> None:
    built = _create(
        tmp_path,
        [Detection("ic", 0.9, BoundingBox(20, 20, 40, 40), source="ultralytics")],
    )
    slot = built.recipe.slots[0]
    height = int(slot.fixed_roi_xyxy.height)
    width = int(slot.fixed_roi_xyxy.width)
    ignore_path = tmp_path / "masks" / "slot_0001_ignore.png"
    assert cv2.imwrite(str(ignore_path), np.zeros((height, width), dtype=np.uint8))
    recipe = replace(
        built.recipe,
        slots=(replace(slot, ignore_mask_path="masks/slot_0001_ignore.png"),),
        asset_sha256={
            **built.recipe.asset_sha256,
            "masks/slot_0001_ignore.png": sha256(ignore_path.read_bytes()).hexdigest(),
        },
    )
    save_recipe(recipe, tmp_path / "recipe.json")

    loaded = load_recipe(tmp_path / "recipe.json")

    assert loaded.slots[0].ignore_mask_path == "masks/slot_0001_ignore.png"
    assert cv2.imwrite(str(ignore_path), np.zeros((height - 1, width), dtype=np.uint8))
    resized_recipe = replace(
        recipe,
        asset_sha256={
            **recipe.asset_sha256,
            "masks/slot_0001_ignore.png": sha256(ignore_path.read_bytes()).hexdigest(),
        },
    )
    with pytest.raises(RecipeValidationError, match="Ignore mask size mismatch"):
        validate_recipe_assets(resized_recipe, tmp_path)


def test_recipe_rejects_empty_and_fully_outside_detections_with_reasons(
    tmp_path: Path,
) -> None:
    detections = [
        Detection("valid", 0.9, BoundingBox(10, 10, 20, 20)),
        Detection("empty", 0.8, BoundingBox(30, 30, 30, 40)),
        Detection("outside", 0.7, BoundingBox(130, 10, 150, 30)),
    ]
    result = _create(tmp_path, detections)

    assert len(result.recipe.slots) == 1
    assert len(result.recipe.rejected_detections) == 2
    assert {item.reason for item in result.recipe.rejected_detections} == {
        "empty_bbox_after_clamp",
        "bbox_outside_golden_image",
    }


def test_recipe_rejects_opencv_candidates_by_default(tmp_path: Path) -> None:
    detection = Detection(
        "component_candidate",
        0.4,
        BoundingBox(10, 10, 20, 20),
        source="opencv_candidate",
    )
    with pytest.raises(RecipeValidationError, match="No valid production slots"):
        _create(tmp_path, [detection])

    result = create_recipe(
        _golden(),
        [detection],
        tmp_path / "demo",
        board_id="DEMO",
        side="top",
        metrology=MetrologyCalibration(40.0, 40.0),
        roi_padding_px=2,
        search_margin_px=4,
        position_tolerance=_position_tolerance(),
        appearance_thresholds=_appearance_thresholds(),
        allow_demo_sources=True,
    )
    assert result.recipe.slots[0].source == "opencv_candidate"
    assert result.recipe.production_eligible is False


def test_recipe_round_trip_and_hash_validation(tmp_path: Path) -> None:
    result = _create(
        tmp_path,
        [Detection("resistor", 0.9, BoundingBox(20.25, 22.5, 42.75, 35.25), class_id=17)],
    )
    loaded = load_recipe(tmp_path / "recipe.json", validate_assets=True)

    assert loaded.to_dict() == result.recipe.to_dict()
    assert loaded.content_sha256 == result.recipe.content_sha256
    assert loaded.coordinate_space == "golden_board_pixels"
    assert loaded.metrology.pixels_per_mm_y == 42.0

    golden_path = tmp_path / loaded.golden_asset_path
    golden_path.write_bytes(golden_path.read_bytes() + b"tampered")
    with pytest.raises(RecipeValidationError, match="Asset SHA-256 mismatch: golden.png"):
        load_recipe(tmp_path / "recipe.json", validate_assets=True)


@pytest.mark.parametrize("asset_path", ["template_path", "component_mask_path", "compare_mask_path"])
def test_recipe_rejects_same_size_tampering_of_any_slot_asset(
    tmp_path: Path, asset_path: str
) -> None:
    result = _create(
        tmp_path,
        [Detection("resistor", 0.9, BoundingBox(20, 20, 40, 35), class_id=17)],
    )
    relative_path = getattr(result.recipe.slots[0], asset_path)
    path = tmp_path / relative_path
    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR if asset_path == "template_path" else cv2.IMREAD_GRAYSCALE,
    )
    assert image is not None
    image.flat[0] = (int(image.flat[0]) + 17) % 256
    assert cv2.imwrite(str(path), image)

    with pytest.raises(RecipeValidationError, match=f"Asset SHA-256 mismatch: {relative_path}"):
        load_recipe(tmp_path / "recipe.json", validate_assets=True)


def test_recipe_demo_defaults_to_top_side_without_identity_arguments(tmp_path: Path) -> None:
    result = create_recipe(
        _golden(),
        [Detection("resistor", 0.9, BoundingBox(20, 20, 40, 35), class_id=17)],
        tmp_path,
        metrology=MetrologyCalibration(40.0, 40.0),
        roi_padding_px=2,
        search_margin_px=4,
        position_tolerance=_position_tolerance(),
        appearance_thresholds=_appearance_thresholds(),
    )

    assert result.recipe.board_id == "demo_board"
    assert result.recipe.side == "top"
    persisted = json.loads((tmp_path / "recipe.json").read_text(encoding="utf-8"))
    assert persisted["board_id"] == "demo_board"
    assert persisted["side"] == "top"


def test_recipe_without_required_alignment_anchors_is_not_production_eligible(
    tmp_path: Path,
) -> None:
    result = _create(
        tmp_path,
        [Detection("ic", 0.9, BoundingBox(20, 20, 40, 40), source="ultralytics")],
    )

    assert result.recipe.metrology.verified is True
    assert result.recipe.alignment.anchors == ()
    assert result.recipe.production_eligible is False
    assert result.recipe.enrollment["alignment_ready"] is False


def test_recipe_load_rejects_malformed_schema_and_absolute_asset_path(tmp_path: Path) -> None:
    result = _create(
        tmp_path,
        [Detection("ic", 0.9, BoundingBox(20, 20, 40, 40), class_id=10)],
    )
    payload = result.recipe.to_dict()
    payload["schema_version"] = "unknown/9.9"
    (tmp_path / "bad-schema.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RecipeValidationError, match="Unsupported recipe schema"):
        load_recipe(tmp_path / "bad-schema.json", validate_assets=False)

    payload = result.recipe.to_dict()
    payload["golden_asset_path"] = "/tmp/golden.png"
    (tmp_path / "absolute.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RecipeValidationError, match="relative portable path"):
        load_recipe(tmp_path / "absolute.json", validate_assets=False)


def test_recipe_load_rejects_fractional_fixed_roi_geometry(tmp_path: Path) -> None:
    result = _create(
        tmp_path,
        [Detection("ic", 0.9, BoundingBox(20, 20, 40, 40), class_id=10)],
    )
    payload = result.recipe.to_dict()
    payload["slots"][0]["fixed_roi_xyxy"][0] += 0.5
    path = tmp_path / "fractional-fixed-roi.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RecipeValidationError, match="fixed_roi_xyxy must use integer pixels"):
        load_recipe(path, validate_assets=False)


@pytest.mark.parametrize("asset_kind", ["golden", "slot_template", "anchor_template"])
def test_recipe_load_rejects_lossy_image_assets(
    tmp_path: Path, asset_kind: str
) -> None:
    result = _create(
        tmp_path,
        [Detection("ic", 0.9, BoundingBox(20, 20, 40, 40), class_id=10)],
    )
    payload = result.recipe.to_dict()
    if asset_kind == "golden":
        payload["golden_asset_path"] = "golden.jpg"
    elif asset_kind == "slot_template":
        payload["slots"][0]["template_path"] = "templates/slot_0001.jpg"
    else:
        payload["alignment"]["anchors"] = [
            {
                "anchor_id": "anchor_0001",
                "reference_point_px": [10.0, 10.0],
                "template_bbox_xyxy": [5.0, 5.0, 15.0, 15.0],
                "search_roi_xyxy": [2.0, 2.0, 18.0, 18.0],
                "template_path": "anchors/anchor_0001.jpg",
                "mask_path": None,
            }
        ]
    path = tmp_path / f"lossy-{asset_kind}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RecipeValidationError, match="lossless PNG/TIFF"):
        load_recipe(path, validate_assets=False)


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_recipe_load_requires_json_boolean_for_production_eligible(
    tmp_path: Path, value: object
) -> None:
    result = _create(
        tmp_path,
        [Detection("ic", 0.9, BoundingBox(20, 20, 40, 40), source="ultralytics")],
    )
    payload = result.recipe.to_dict()
    if value is None:
        payload.pop("production_eligible")
    else:
        payload["production_eligible"] = value
    path = tmp_path / "bad-production-boolean.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RecipeValidationError, match="production_eligible must be a boolean"):
        load_recipe(path, validate_assets=False)


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_recipe_load_requires_json_boolean_for_calibration_verified(
    tmp_path: Path, value: object
) -> None:
    result = _create(
        tmp_path,
        [Detection("ic", 0.9, BoundingBox(20, 20, 40, 40), source="ultralytics")],
    )
    payload = result.recipe.to_dict()
    if value is None:
        payload["metrology"].pop("verified")
    else:
        payload["metrology"]["verified"] = value
    path = tmp_path / "bad-verified-boolean.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RecipeValidationError, match="verified must be a boolean"):
        load_recipe(path, validate_assets=False)


def test_hand_edited_recipe_cannot_claim_production_without_anchors(
    tmp_path: Path,
) -> None:
    result = _create(
        tmp_path,
        [Detection("ic", 0.9, BoundingBox(20, 20, 40, 40), source="ultralytics")],
    )
    payload = result.recipe.to_dict()
    payload["production_eligible"] = True
    path = tmp_path / "false-production-claim.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RecipeValidationError, match="enough alignment anchors"):
        load_recipe(path, validate_assets=False)


@pytest.mark.parametrize(
    ("x", "y"),
    [(0.0, 40.0), (-1.0, 40.0), (40.0, float("nan"))],
)
def test_metrology_requires_positive_finite_scale(x: float, y: float) -> None:
    with pytest.raises(RecipeValidationError, match="pixels_per_mm"):
        MetrologyCalibration(x, y)


def test_unverified_demo_calibration_cannot_create_production_eligible_recipe(
    tmp_path: Path,
) -> None:
    result = create_recipe(
        _golden(),
        [Detection("ic", 0.9, BoundingBox(20, 20, 40, 40), source="ultralytics")],
        tmp_path,
        metrology=MetrologyCalibration(1.0, 1.0, verified=False),
        roi_padding_px=2,
        search_margin_px=4,
        position_tolerance=_position_tolerance(),
        appearance_thresholds=_appearance_thresholds(),
    )

    assert result.recipe.production_eligible is False
    assert result.recipe.metrology.verified is False
    assert result.recipe.enrollment["calibration_verified"] is False
    assert load_recipe(tmp_path / "recipe.json").metrology.verified is False


def test_recipe_requires_uint8_bgr_measurement_image(tmp_path: Path) -> None:
    with pytest.raises(RecipeValidationError, match="uint8 BGR"):
        create_recipe(
            np.zeros((20, 30), dtype=np.uint8),
            [Detection("ic", 0.9, BoundingBox(2, 2, 10, 10))],
            tmp_path,
            board_id="BOARD_A",
            side="top",
            metrology=MetrologyCalibration(40.0, 40.0),
            roi_padding_px=2,
            search_margin_px=4,
            position_tolerance=_position_tolerance(),
            appearance_thresholds=_appearance_thresholds(),
        )
