from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from aoi_pipeline import (
    AOIInspector,
    BoundingBox,
    Detection,
    MockComponentDetector,
)
from aoi_pipeline.golden_compare import GoldenComparator, GoldenCompareConfig
from aoi_pipeline.inspection import InspectionConfig, _associate_candidates
from aoi_pipeline.position import PositionMeasurer, PositionQualityGates
from aoi_pipeline.recipe import (
    AlignmentAnchor,
    AlignmentQualityGates,
    AlignmentRecipe,
    AppearanceThresholds,
    MetrologyCalibration,
    PositionTolerance,
    create_recipe,
)


def _golden() -> np.ndarray:
    rng = np.random.default_rng(20260819)
    image = rng.integers(20, 210, size=(180, 240, 3), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (5, 5), 0.8)
    component = rng.integers(25, 225, size=(36, 50, 3), dtype=np.uint8)
    component = cv2.GaussianBlur(component, (3, 3), 0.45)
    cv2.rectangle(component, (1, 1), (48, 34), (230, 215, 35), 2)
    cv2.circle(component, (15, 18), 8, (25, 30, 235), -1)
    cv2.line(component, (29, 5), (43, 30), (235, 55, 40), 4)
    image[72:108, 95:145] = component
    return image


def _production_recipe(tmp_path: Path):
    golden = _golden()
    anchors: list[AlignmentAnchor] = []
    for index, (center_x, center_y) in enumerate(
        ((34, 34), (205, 36), (38, 148), (202, 145)), start=1
    ):
        bbox = BoundingBox(center_x - 10, center_y - 10, center_x + 11, center_y + 11)
        search = BoundingBox(center_x - 18, center_y - 18, center_x + 19, center_y + 19)
        x1, y1, x2, y2 = bbox.to_int()
        path = tmp_path / "anchors" / f"anchor_{index:04d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(path), golden[y1:y2, x1:x2])
        anchors.append(
            AlignmentAnchor(
                anchor_id=f"anchor_{index:04d}",
                reference_point_px=(float(center_x), float(center_y)),
                template_bbox_xyxy=bbox,
                search_roi_xyxy=search,
                template_path=f"anchors/anchor_{index:04d}.png",
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
            min_canvas_overlap_ratio=0.90,
        ),
    )
    enrollment_detection = Detection(
        "ic",
        0.96,
        BoundingBox(95, 72, 145, 108),
        class_id=10,
        source="ultralytics",
    )
    recipe = create_recipe(
        golden,
        [enrollment_detection],
        tmp_path,
        board_id="BOARD_TEST",
        side="top",
        metrology=MetrologyCalibration(20.0, 20.0, verified=True),
        roi_padding_px=9,
        search_margin_px=8,
        position_tolerance=PositionTolerance(0.025, 0.025, None),
        appearance_thresholds=AppearanceThresholds(
            min_ssim=0.88,
            max_diff_ratio=0.08,
            max_edge_diff_ratio=0.10,
            max_blob_area_px=45,
            min_valid_overlap_ratio=0.88,
        ),
        alignment=alignment,
        model_identifiers={"component_detector": "best.onnx:test-sha"},
    ).recipe
    assert recipe.production_eligible is True
    runtime_candidate = Detection(
        "ic",
        0.94,
        BoundingBox(95, 72, 145, 108),
        class_id=10,
        source="ultralytics",
        detection_id="runtime_component",
    )
    return golden, recipe, runtime_candidate


def _inspector(detections, *, config: InspectionConfig | None = None):
    return AOIInspector(
        MockComponentDetector(detections),
        position_measurer=PositionMeasurer(
            PositionQualityGates(
                min_score=0.50,
                min_peak_margin=0.004,
                min_psr=2.0,
            )
        ),
        comparator=GoldenComparator(
            GoldenCompareConfig(pixel_diff_threshold=24, min_blob_area_px=3)
        ),
        config=config,
        runtime_detector_identifier="best.onnx:test-sha",
    )


def _shift_slot(image: np.ndarray, bbox: BoundingBox, dx: float, dy: float) -> np.ndarray:
    output = image.copy()
    x1, y1, x2, y2 = bbox.to_int()
    roi = image[y1:y2, x1:x2]
    shifted = cv2.warpAffine(
        roi,
        np.float32([[1, 0, dx], [0, 1, dy]]),
        (roi.shape[1], roi.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    output[y1:y2, x1:x2] = shifted
    return output


def test_inspector_same_board_passes_and_serializes_json(tmp_path: Path) -> None:
    golden, recipe, candidate = _production_recipe(tmp_path)

    run = _inspector([candidate]).inspect(golden, recipe, tmp_path)
    payload = run.to_dict()
    serialized = run.to_json()
    decoded = json.loads(serialized)

    assert run.status == "pass", payload
    assert run.alignment.status == "valid"
    assert len(run.slots) == 1
    assert run.slots[0].status == "pass"
    assert run.slots[0].position.status == "pass"
    assert run.slots[0].appearance.status == "pass"
    assert run.extras == ()
    assert payload["recipe_sha256"] == recipe.content_sha256
    assert decoded["recipe_sha256"] == recipe.content_sha256
    assert payload["model_identifiers"] == {"component_detector": "best.onnx:test-sha"}
    assert str(tmp_path) not in serialized
    assert "anomaly_mask" not in serialized


def test_invalid_measurement_image_returns_invalid_without_detector(tmp_path: Path) -> None:
    _, recipe, candidate = _production_recipe(tmp_path)

    run = _inspector([candidate]).inspect(
        np.zeros((0, 0, 3), dtype=np.uint8), recipe, tmp_path
    )

    assert run.status == "invalid"
    assert run.reason == "invalid_measurement_image"
    assert run.slots == ()


def test_alignment_invalid_stops_before_detector(tmp_path: Path) -> None:
    golden, recipe, _ = _production_recipe(tmp_path)
    calls = 0

    def detector_callback(image):
        nonlocal calls
        calls += 1
        return []

    run = _inspector(detector_callback).inspect(np.full_like(golden, 127), recipe, tmp_path)

    assert run.status == "invalid"
    assert run.alignment.status == "invalid"
    assert run.slots == ()
    assert calls == 0


def test_shifted_component_is_ng_position_but_pass_appearance(tmp_path: Path) -> None:
    golden, recipe, candidate = _production_recipe(tmp_path)
    observed = _shift_slot(golden, recipe.slots[0].fixed_roi_xyxy, 2.0, -1.25)

    run = _inspector([candidate]).inspect(observed, recipe, tmp_path)

    assert run.status == "ng"
    assert run.slots[0].status == "ng_position"
    assert run.slots[0].position.status == "ng"
    assert run.slots[0].appearance.status == "pass"


def test_wrong_appearance_is_ng_appearance(tmp_path: Path) -> None:
    golden, recipe, candidate = _production_recipe(tmp_path)
    observed = golden.copy()
    cv2.rectangle(observed, (112, 82), (135, 101), (255, 0, 255), -1)

    run = _inspector([candidate]).inspect(observed, recipe, tmp_path)

    assert run.status == "ng"
    assert run.slots[0].status in {"ng_appearance", "ng_position_and_appearance"}
    assert run.slots[0].appearance.status == "anomaly"
    assert run.slots[0].appearance.defect_label == "appearance_anomaly"


def test_missing_candidate_is_board_ng_and_appearance_not_evaluated(tmp_path: Path) -> None:
    golden, recipe, _ = _production_recipe(tmp_path)

    run = _inspector([]).inspect(golden, recipe, tmp_path)

    assert run.status == "ng"
    assert run.slots[0].status == "ng_missing"
    assert run.slots[0].position.status == "missing_candidate"
    assert run.slots[0].appearance.status == "not_evaluated"


def test_unmeasurable_candidate_requests_review_instead_of_fake_ng(tmp_path: Path) -> None:
    golden, recipe, candidate = _production_recipe(tmp_path)
    observed = golden.copy()
    x1, y1, x2, y2 = recipe.slots[0].fixed_roi_xyxy.to_int()
    observed[y1:y2, x1:x2] = 100

    run = _inspector([candidate]).inspect(observed, recipe, tmp_path)

    assert run.status == "review"
    assert run.slots[0].status == "review"
    assert run.slots[0].position.status == "unmeasurable"
    assert run.slots[0].appearance.status == "not_evaluated"


def test_unmatched_detection_is_exported_as_extra_and_board_ng(tmp_path: Path) -> None:
    golden, recipe, candidate = _production_recipe(tmp_path)
    extra = Detection(
        "resistor",
        0.88,
        BoundingBox(160, 115, 182, 130),
        class_id=17,
        source="ultralytics",
        detection_id="runtime_extra",
    )

    run = _inspector([candidate, extra]).inspect(golden, recipe, tmp_path)

    assert run.status == "ng"
    assert run.slots[0].status == "pass"
    assert len(run.extras) == 1
    assert run.extras[0].detection_id == "runtime_extra"
    assert run.reason == "1_extra_candidates"


def test_candidate_association_maximizes_cardinality_before_distance(
    tmp_path: Path,
) -> None:
    _, recipe, _ = _production_recipe(tmp_path)
    base = recipe.slots[0]
    first = replace(
        base,
        slot_id="slot_0001",
        expected_bbox_xyxy=BoundingBox(18, 8, 22, 12),
        expected_center_px=(20.0, 10.0),
        fixed_roi_xyxy=BoundingBox(18, 8, 22, 12),
        search_margin_px=6,
    )
    second = replace(
        base,
        slot_id="slot_0002",
        expected_bbox_xyxy=BoundingBox(26, 8, 30, 12),
        expected_center_px=(28.0, 10.0),
        fixed_roi_xyxy=BoundingBox(26, 8, 30, 12),
        search_margin_px=6,
    )
    shared = Detection(
        "ic",
        0.95,
        BoundingBox(20, 8, 24, 12),
        class_id=10,
        source="ultralytics",
        detection_id="shared",
    )
    first_only = Detection(
        "ic",
        0.94,
        BoundingBox(12, 8, 16, 12),
        class_id=10,
        source="ultralytics",
        detection_id="first_only",
    )

    associations, unmatched = _associate_candidates(
        (first, second),
        (shared, first_only),
        allow_class_mismatch=False,
    )

    assert associations["slot_0001"].detection_id == "first_only"
    assert associations["slot_0002"].detection_id == "shared"
    assert unmatched == []


def test_nonproduction_recipe_fails_before_alignment_and_detection(tmp_path: Path) -> None:
    golden, recipe, candidate = _production_recipe(tmp_path)
    demo_recipe = replace(recipe, production_eligible=False)

    run = _inspector([candidate]).inspect(golden, demo_recipe, tmp_path)

    assert run.status == "invalid"
    assert run.reason == "recipe_not_production_eligible"
    assert run.alignment.reason == "recipe_not_production_eligible"


def test_production_inspection_rejects_different_runtime_detector_artifact(
    tmp_path: Path,
) -> None:
    golden, recipe, candidate = _production_recipe(tmp_path)
    calls = 0

    def detector_callback(image):
        nonlocal calls
        calls += 1
        return [candidate]

    inspector = AOIInspector(
        MockComponentDetector(detector_callback),
        config=InspectionConfig(require_production_eligible=True),
        runtime_detector_identifier="other.onnx:other-sha",
    )
    run = inspector.inspect(golden, recipe, tmp_path)

    assert run.status == "invalid"
    assert run.reason == "runtime_detector_mismatch"
    assert run.runtime_detector_identifier == "other.onnx:other-sha"
    assert calls == 0


def test_from_pipeline_reuses_pipeline_detector_and_detection_facade(tmp_path: Path) -> None:
    golden, recipe, candidate = _production_recipe(tmp_path)

    class FakePipeline:
        def __init__(self):
            self.detector = MockComponentDetector([candidate])
            self.calls = 0

        def detect_components(self, image, *, frame_id):
            assert frame_id == "inspection"
            self.calls += 1
            return self.detector.detect(image)

    pipeline = FakePipeline()
    inspector = AOIInspector.from_pipeline(
        pipeline,
        position_measurer=PositionMeasurer(
            PositionQualityGates(min_score=0.5, min_peak_margin=0.004, min_psr=2.0)
        ),
        runtime_detector_identifier="best.onnx:test-sha",
    )

    run = inspector.inspect(golden, recipe, tmp_path)

    assert inspector.detector is pipeline.detector
    assert pipeline.calls == 1
    assert run.status == "pass"
