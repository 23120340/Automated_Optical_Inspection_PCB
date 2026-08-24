from __future__ import annotations

import csv
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import cv2
import numpy as np
import pandas as pd
import pytest

from aoi_pipeline import BoundingBox, Detection, MockComponentDetector
from aoi_pipeline import CameraCalibrationProfile
from app.pipeline_bridge import (
    ClassificationRecord,
    ClassificationResult,
    CropRecord,
    DetectionRecord,
    DetectionResult,
    InspectionRecipeRecord,
    InspectionResult,
    PipelineBridge,
    StageResult,
)


def test_bridge_exposes_failed_alignment_as_fallback() -> None:
    bridge = PipelineBridge()
    source = np.zeros((80, 100, 3), dtype=np.uint8)
    reference = np.zeros((60, 90, 3), dtype=np.uint8)
    result = bridge.align(source, reference)
    assert result.mode == "CV FALLBACK"
    assert result.metrics["success"] is False
    assert result.metrics["method"] == "resize_fallback"
    assert "không đạt gate" in result.message


def test_bridge_exposes_full_image_board_fallback() -> None:
    bridge = PipelineBridge()
    result = bridge.detect_board(np.full((50, 70, 3), 127, dtype=np.uint8))
    assert result.mode == "CV FALLBACK"
    assert result.metrics["method"] == "full_image_fallback"
    assert result.metrics["confidence"] == pytest.approx(0.10)


class _FailingDetector:
    def detect(self, image: np.ndarray):
        raise RuntimeError("broken weights")


def test_selected_model_failure_is_not_downgraded_to_cv_demo() -> None:
    bridge = PipelineBridge.__new__(PipelineBridge)
    bridge.config = {}
    bridge.detector = None
    bridge.model_path = "selected.pt"
    bridge.board_model_path = None
    bridge.extra = {}
    bridge.engine_error = None
    bridge.engine_module = "test"
    bridge.engine = SimpleNamespace(
        detector=_FailingDetector(),
        detect_components=lambda image, board_region=None: (_ for _ in ()).throw(
            RuntimeError("broken weights")
        ),
    )
    with pytest.raises(RuntimeError, match="không tự chuyển sang CV demo"):
        bridge.detect_components(np.zeros((32, 32, 3), dtype=np.uint8))


def test_bridge_returns_core_crop_images_instead_of_recropping() -> None:
    detection = Detection("ic", 0.9, BoundingBox(4, 4, 12, 10))
    core_crop = np.full((17, 23, 3), 77, dtype=np.uint8)
    raw_crop = SimpleNamespace(
        image=core_crop,
        detection_id=detection.detection_id,
        label="ic",
        confidence=0.9,
    )
    bridge = PipelineBridge.__new__(PipelineBridge)
    bridge.config = {"crops": {"padding": 1, "normalize": True, "target_size": 224}}
    bridge.engine = SimpleNamespace(make_crops=lambda image, detections, output_dir=None: [raw_crop])
    record = DetectionRecord(
        detection_id=detection.detection_id,
        label="ic",
        confidence=0.9,
        bbox=(4, 4, 12, 10),
        source="model",
        raw=detection,
    )
    crops = bridge.make_crops(np.zeros((30, 30, 3), dtype=np.uint8), [record])
    assert len(crops) == 1
    assert crops[0].image is core_crop
    assert crops[0].image.shape == (17, 23, 3)


def test_bridge_classification_uses_core_model_and_never_detector_label() -> None:
    raw_crop = SimpleNamespace(detection_id="det_1")
    crop = CropRecord(
        crop_id="crop_0001",
        label="detector_ic",
        image=np.zeros((16, 16, 3), dtype=np.uint8),
        bbox=(0, 0, 16, 16),
        confidence=0.8,
        source="model",
        raw=raw_crop,
    )
    raw_classification = SimpleNamespace(
        crop_id="crop_0001",
        detection_id="det_1",
        family="capacitor",
        probability=0.92,
        unknown_score=0.08,
        decision="accept",
        detector_hint="detector_ic",
        model_version="v1",
        top_k=[SimpleNamespace(label="capacitor", probability=0.92)],
    )
    bridge = PipelineBridge.__new__(PipelineBridge)
    bridge.classifier_model_path = "best.onnx"
    bridge.classifier_manifest_path = "model_manifest.json"
    bridge.engine_error = None
    bridge.engine = SimpleNamespace(
        classify_components=lambda crops: [raw_classification]
    )

    result = bridge.classify_components([crop])

    assert isinstance(result, ClassificationResult)
    assert result.mode == "MODEL"
    assert result.classifications[0].family == "capacitor"
    assert result.classifications[0].family != crop.label
    assert result.classifications[0].decision == "accept"


def _solder_detector_bridge(engine) -> PipelineBridge:
    bridge = PipelineBridge.__new__(PipelineBridge)
    bridge.config = {}
    bridge.engine = engine
    bridge.engine_error = None
    bridge.solder_detector_model_path = "solder-detector.onnx"
    bridge.solder_detector_manifest_path = "model_manifest.json"
    return bridge


def test_solder_segment_findings_are_reported_as_an_independent_layer() -> None:
    finding = SimpleNamespace(
        detection_id="solder_defect_0001",
        label="Short_circuit",
        confidence=0.91,
        bbox=(4, 5, 22, 24),
        source="solder_defect_segment",
        metadata={"mask_polygon": [[4, 5], [22, 5], [22, 24], [4, 24]]},
    )
    engine = SimpleNamespace(
        make_solder_crops=lambda image, detections, output_dir=None: [],
        grade_solder=lambda crops, image: [],
        detect_solder_defects=lambda image: [finding],
        solder_inspector=SimpleNamespace(warnings=[]),
        last_fusion=None,
        cad_warnings=[],
    )
    bridge = _solder_detector_bridge(engine)

    result = bridge.make_solder_crops(np.zeros((40, 50, 3), dtype=np.uint8), [])

    assert result.detector_active is True
    assert result.detector_error is None
    assert len(result.detector_findings) == 1
    assert result.detector_findings[0].label == "Short_circuit"
    assert result.detector_findings[0].metadata["mask_polygon"][0] == [4, 5]
    assert result.metrics["solder_detector_findings"] == 1


def test_solder_segment_failure_does_not_remove_roi_grading() -> None:
    verdict = SimpleNamespace(
        joint_id="joint_0001",
        detection_id="component_0001",
        scope="joint",
        label="good",
        decision="accept",
        source="rules",
        probability=0.8,
        rule_label="good",
        model_label=None,
        model_probability=None,
        designator="R1",
        pin="1",
        component_label="resistor",
        metadata={"bbox": [5, 6, 20, 18]},
        reasons=["rule layer still ran"],
        features=None,
        model_version="rules-only",
    )

    def broken_detector(image):
        raise RuntimeError("segment backend failed")

    engine = SimpleNamespace(
        make_solder_crops=lambda image, detections, output_dir=None: [],
        grade_solder=lambda crops, image: [verdict],
        detect_solder_defects=broken_detector,
        solder_inspector=SimpleNamespace(warnings=[]),
        last_fusion=None,
        cad_warnings=[],
    )
    bridge = _solder_detector_bridge(engine)

    result = bridge.make_solder_crops(np.zeros((40, 50, 3), dtype=np.uint8), [])

    assert result.detector_active is True
    assert "segment backend failed" in (result.detector_error or "")
    assert len(result.verdicts) == 1
    assert result.verdicts[0].decision == "accept"
    assert result.grading_error is None


def test_bridge_builds_and_inspects_recipe_through_core(tmp_path: Path) -> None:
    rng = np.random.default_rng(20260819)
    golden = rng.integers(15, 235, size=(160, 220, 3), dtype=np.uint8)
    golden = cv2.GaussianBlur(golden, (3, 3), 0.4)
    candidate = Detection(
        "ic",
        0.95,
        BoundingBox(85, 62, 135, 100),
        class_id=10,
        source="ultralytics",
    )
    detector = MockComponentDetector([candidate])

    class FakePipeline:
        def __init__(self):
            self.detector = detector

        def detect_components(self, image, board_region=None, *, frame_id):
            del board_region, frame_id
            return self.detector.detect(image)

    bridge = PipelineBridge.__new__(PipelineBridge)
    bridge.config = {}
    bridge.detector = None
    bridge.model_path = None
    bridge.board_model_path = None
    bridge.classifier_model_path = None
    bridge.classifier_manifest_path = None
    bridge.extra = {}
    bridge.engine_error = None
    bridge.engine_module = "test"
    bridge.engine = FakePipeline()

    recipe = bridge.build_inspection_recipe(
        golden,
        tmp_path / "recipe",
        board_id="BOARD_TEST",
        side="top",
        anchor_template_size_px=21,
        anchor_search_margin_px=8,
        roi_padding_px=8,
        search_margin_px=6,
    )
    result = bridge.inspect_board(golden, recipe)
    loaded = bridge.load_inspection_recipe(recipe.recipe_path)

    assert isinstance(recipe, InspectionRecipeRecord)
    assert recipe.slot_count == 1
    assert recipe.anchor_count == 25
    assert recipe.production_eligible is False
    assert loaded.raw.to_dict() == recipe.raw.to_dict()
    assert isinstance(result, InspectionResult)
    assert result.status == "pass"
    assert result.alignment_status == "valid"
    assert result.slots[0]["position"]["status"] == "pass"
    assert result.slots[0]["appearance"]["status"] == "pass"
    assert str(tmp_path) not in result.json_payload
    assert not np.array_equal(result.image, golden)


def test_bridge_blocks_untrusted_pt_before_golden_detector_runs(tmp_path: Path) -> None:
    calls = 0
    candidate = Detection("ic", 0.95, BoundingBox(20, 20, 50, 45), source="ultralytics")
    detector = MockComponentDetector([candidate])

    class FakePipeline:
        def __init__(self):
            self.detector = detector

        def detect_components(self, image, board_region=None, *, frame_id):
            nonlocal calls
            del image, board_region, frame_id
            calls += 1
            return detector.detect(np.zeros((80, 100, 3), dtype=np.uint8))

    bridge = PipelineBridge.__new__(PipelineBridge)
    bridge.config = {}
    bridge.model_path = "untrusted.pt"
    bridge.engine = FakePipeline()
    bridge.engine_error = None

    with pytest.raises(RuntimeError, match="chưa được xác nhận tin cậy"):
        bridge.build_inspection_recipe(
            np.zeros((80, 100, 3), dtype=np.uint8),
            tmp_path / "recipe",
            board_id="BOARD_TEST",
            side="top",
        )

    assert calls == 0


def test_failed_recipe_build_leaves_no_published_or_staging_directory(
    tmp_path: Path,
) -> None:
    detector = MockComponentDetector([])

    class FailingPipeline:
        def __init__(self):
            self.detector = detector

        def detect_components(self, image, board_region=None, *, frame_id):
            del image, board_region, frame_id
            raise RuntimeError("inference failed")

    bridge = PipelineBridge.__new__(PipelineBridge)
    bridge.config = {}
    bridge.model_path = None
    bridge.engine = FailingPipeline()
    bridge.engine_error = None
    destination = tmp_path / "recipe-build"

    with pytest.raises(RuntimeError, match="Detector không tạo được slot Golden"):
        bridge.build_inspection_recipe(
            np.zeros((80, 100, 3), dtype=np.uint8),
            destination,
            board_id="BOARD_TEST",
            side="top",
        )

    assert not destination.exists()
    assert list(tmp_path.glob(".recipe-build.staging-*")) == []


def test_bridge_uses_same_undistortion_domain_for_enrollment_and_inspection(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(20260819)
    golden = cv2.GaussianBlur(
        rng.integers(15, 235, size=(80, 100, 3), dtype=np.uint8), (3, 3), 0.4
    )
    candidate = Detection("ic", 0.95, BoundingBox(34, 25, 64, 52), source="ultralytics")
    detector = MockComponentDetector([candidate])

    class FakePipeline:
        def __init__(self):
            self.detector = detector
            self.images: list[np.ndarray] = []

        def detect_components(self, image, board_region=None, *, frame_id):
            del board_region, frame_id
            self.images.append(image.copy())
            return self.detector.detect(image)

    profile = CameraCalibrationProfile(
        camera_matrix=np.array([[120.0, 0.0, 50.0], [0.0, 118.0, 40.0], [0.0, 0.0, 1.0]]),
        distortion_coefficients=np.array([0.08, -0.03, 0.001, -0.002, 0.0]),
        image_size=(100, 80),
    )
    bridge = PipelineBridge.__new__(PipelineBridge)
    bridge.config = {
        "preprocess": {
            "undistort": True,
            "calibration_profile": profile.to_dict(),
            "undistort_alpha": 0.0,
        }
    }
    bridge.model_path = None
    bridge.engine = FakePipeline()
    bridge.engine_error = None

    record = bridge.build_inspection_recipe(
        golden,
        tmp_path / "recipe",
        board_id="BOARD_TEST",
        side="top",
        anchor_template_size_px=15,
        anchor_search_margin_px=6,
    )
    assert record.raw.enrollment["measurement_domain"]["undistort"] == "pinhole"
    assert len(bridge.engine.images) == 1
    assert bridge.engine.images[0].shape == golden.shape
    assert not np.array_equal(bridge.engine.images[0], golden)

    result = bridge.inspect_board(golden, record)
    assert result.alignment_status == "valid"
    assert len(bridge.engine.images) == 2

    bridge.config = {}
    with pytest.raises(RuntimeError, match="không khớp Golden recipe"):
        bridge.inspect_board(golden, record)


def test_inspection_recipe_output_dirs_are_session_and_build_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import streamlit_app as ui

    monkeypatch.setattr(
        ui.st,
        "session_state",
        type("State", (), {"reference_digest": "a" * 64, "inspection_session_id": "session-a"})(),
    )
    first = ui._inspection_recipe_output_dir(build_id="build-1")
    second = ui._inspection_recipe_output_dir(build_id="build-2")
    assert first != second
    assert "session-a" in first.parts


def test_inspection_ui_helpers_keep_outputs_separate_and_zip_portable(
    tmp_path: Path,
) -> None:
    from app import streamlit_app as ui

    rng = np.random.default_rng(7)
    golden = rng.integers(10, 245, size=(150, 210, 3), dtype=np.uint8)
    candidate = Detection(
        "ic", 0.96, BoundingBox(80, 58, 130, 96), class_id=10
    )
    detector = MockComponentDetector([candidate])

    class FakePipeline:
        def __init__(self):
            self.detector = detector

        def detect_components(self, image, board_region=None, *, frame_id):
            del board_region, frame_id
            return self.detector.detect(image)

    bridge = PipelineBridge.__new__(PipelineBridge)
    bridge.config = {}
    bridge.detector = None
    bridge.model_path = None
    bridge.board_model_path = None
    bridge.classifier_model_path = None
    bridge.classifier_manifest_path = None
    bridge.extra = {}
    bridge.engine_error = None
    bridge.engine_module = "test"
    bridge.engine = FakePipeline()
    recipe = bridge.build_inspection_recipe(
        golden,
        tmp_path / "recipe",
        board_id="BOARD_TEST",
        side="top",
        anchor_template_size_px=21,
        anchor_search_margin_px=8,
        roi_padding_px=8,
        search_margin_px=6,
    )
    result = bridge.inspect_board(golden, recipe)

    position = ui._inspection_position_rows(result)
    appearance = ui._inspection_appearance_rows(result)
    package = ui._inspection_recipe_zip_bytes(recipe)
    package_path = tmp_path / "package.zip"
    package_path.write_bytes(package)

    assert set(position[0]) >= {"position_status", "dx_px", "dy_px", "angle_deg"}
    assert "ssim" not in position[0]
    assert set(appearance[0]) >= {"appearance_status", "ssim", "diff_ratio"}
    assert "dx_px" not in appearance[0]
    with zipfile.ZipFile(package_path) as archive:
        names = archive.namelist()
    assert "recipe.json" in names
    assert "golden.png" in names
    assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in names)


def test_ui_manifest_and_csv_declare_analysis_coordinate_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import streamlit_app as ui

    detection = DetectionRecord(
        detection_id="det_1",
        label="resistor",
        confidence=0.8,
        bbox=(5, 6, 20, 18),
        source="model",
        metadata={
            "frame_id": "import_0000",
            "inference_pass": "tile",
            "tile_id": "tile_r000_c001",
            "touches_tile_border": False,
            "center_in_tile_ownership": True,
        },
    )
    aligned = np.zeros((80, 120, 3), dtype=np.uint8)
    state = SimpleNamespace(
        input_name="large.png",
        input_digest="input-sha",
        input_image=np.zeros((200, 300, 3), dtype=np.uint8),
        reference_name=None,
        reference_digest=None,
        board_model_name=None,
        board_model_digest=None,
        component_model_name="best.onnx",
        component_model_digest="model-sha",
        last_backend_mode="MODEL",
        last_backend_detail="test",
        statuses={0: "done", 1: "done", 2: "done", 3: "done", 4: "done", 5: "pending"},
        latencies={},
        config={},
        crops=[],
        preprocess_result=StageResult(
            image=np.zeros((100, 150, 3), dtype=np.uint8), mode="PIPELINE"
        ),
        alignment_result=StageResult(image=aligned, mode="PIPELINE"),
        board_result=None,
        detection_result=DetectionResult(
            image=aligned.copy(),
            mode="MODEL",
            detections=[detection],
        ),
    )
    monkeypatch.setattr(ui.st, "session_state", state)

    manifest = ui._manifest()
    assert manifest["source"]["width"] == 300
    assert manifest["coordinate_space"] == {
        "id": "analysis_image_pixels",
        "stage": 2,
        "image_role": "aligned",
        "width": 120,
        "height": 80,
        "origin": "top_left",
        "bbox_format": "xyxy",
        "right_bottom": "exclusive",
    }
    assert manifest["detections"][0]["coordinate_space"] == "analysis_image_pixels"
    assert manifest["detections"][0]["metadata"]["tile_id"] == "tile_r000_c001"

    rows = list(
        csv.DictReader(StringIO(ui._detections_csv_bytes().decode("utf-8-sig")))
    )
    assert rows[0]["coordinate_space"] == "analysis_image_pixels"
    assert rows[0]["image_width"] == "120"
    assert rows[0]["image_height"] == "80"
    assert rows[0]["bbox_format"] == "xyxy_right_bottom_exclusive"
    assert rows[0]["frame_id"] == "import_0000"
    assert rows[0]["inference_pass"] == "tile"
    assert rows[0]["tile_id"] == "tile_r000_c001"
    assert rows[0]["center_in_tile_ownership"] == "True"


def test_csv_cells_neutralize_spreadsheet_formulas() -> None:
    from app import streamlit_app as ui

    assert ui._csv_cell("resistor") == "resistor"
    assert ui._csv_cell("=HYPERLINK(\"bad\")") == "'=HYPERLINK(\"bad\")"


def test_detection_table_owner_column_is_arrow_compatible_nullable_boolean() -> None:
    from app import streamlit_app as ui

    detections = [
        DetectionRecord(
            detection_id="det_full",
            label="ic",
            confidence=0.9,
            bbox=(0, 0, 10, 10),
            source="model",
            metadata={"inference_pass": "full_image"},
        ),
        DetectionRecord(
            detection_id="det_tile",
            label="resistor",
            confidence=0.8,
            bbox=(10, 0, 20, 10),
            source="model",
            metadata={
                "inference_pass": "tile",
                "center_in_tile_ownership": True,
            },
        ),
    ]

    frame = ui._detections_frame(detections)

    assert str(frame["owner"].dtype) == "boolean"
    assert pd.isna(frame.loc[0, "owner"])
    assert bool(frame.loc[1, "owner"]) is True


def test_classifier_manifest_warns_for_low_quality_artifact() -> None:
    from app import streamlit_app as ui

    warning = ui._classifier_manifest_quality_warning(
        {"metrics": {"accuracy": 0.1348, "weighted_f1": 0.1633}}
    )

    assert warning is not None
    assert "accuracy=0.135" in warning
    assert "weighted_f1=0.163" in warning
    assert ui._classifier_manifest_quality_warning(
        {"metrics": {"accuracy": 0.8, "weighted_f1": 0.75}}
    ) is None


def test_low_resolution_board_import_is_still_measured_and_described() -> None:
    """The number matters even when it no longer blocks: a fillet needs roughly
    ten pixels across it, so a low-resolution run can look clean while being
    unable to see the defects it was meant to find."""

    from app import streamlit_app as ui

    issue = ui._source_resolution_issue(np.zeros((750, 1000, 3), dtype=np.uint8))

    assert issue is not None
    assert "1000 × 750px" in issue
    assert "1280 × 960px" in issue
    assert ui._source_resolution_issue(np.zeros((960, 1280, 3), dtype=np.uint8)) is None


def test_low_resolution_import_is_not_blocked_by_default() -> None:
    """Development needs to run whatever images exist; the warning still shows."""

    from app import streamlit_app as ui

    assert ui.ENFORCE_SOURCE_RESOLUTION is False
    # No exception: the caller displays the warning instead of refusing.
    ui._require_source_resolution(np.zeros((900, 1280, 3), dtype=np.uint8))


def test_the_gate_can_be_re_armed_for_a_production_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import streamlit_app as ui

    monkeypatch.setattr(ui, "ENFORCE_SOURCE_RESOLUTION", True)
    with pytest.raises(ValueError, match="không đạt"):
        ui._require_source_resolution(np.zeros((900, 1280, 3), dtype=np.uint8))
    assert "đã khóa" in ui._source_resolution_issue(np.zeros((900, 1280, 3), dtype=np.uint8))

def test_classification_csv_exports_decision_and_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import streamlit_app as ui

    item = ClassificationRecord(
        crop_id="crop_0001",
        detection_id="det_1",
        family="resistor",
        probability=0.91,
        unknown_score=0.09,
        decision="accept",
        top_k=[{"label": "resistor", "probability": 0.91}],
        detector_hint="component",
        model_version="v1",
    )
    monkeypatch.setattr(
        ui.st,
        "session_state",
        SimpleNamespace(
            classification_result=ClassificationResult(classifications=[item])
        ),
    )

    rows = list(
        csv.DictReader(StringIO(ui._classifications_csv_bytes().decode("utf-8-sig")))
    )
    assert rows[0]["family"] == "resistor"
    assert rows[0]["decision"] == "accept"
    assert json.loads(rows[0]["top_k"])[0]["label"] == "resistor"


def test_pt_model_requires_explicit_ui_trust(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import streamlit_app as ui

    state = SimpleNamespace(
        component_model_name="best.pt",
        component_model_path="C:/temp/best.pt",
        pt_model_trusted=False,
    )
    monkeypatch.setattr(ui.st, "session_state", state)
    assert ui._pt_model_blocked()
    state.pt_model_trusted = True
    assert not ui._pt_model_blocked()
