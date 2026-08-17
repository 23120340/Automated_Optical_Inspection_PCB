from __future__ import annotations

import csv
from io import StringIO
import json
from types import SimpleNamespace

import numpy as np
import pytest

from aoi_pipeline import BoundingBox, Detection, MockComponentDetector
from app.pipeline_bridge import (
    ClassificationRecord,
    ClassificationResult,
    CropRecord,
    DetectionRecord,
    DetectionResult,
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

    rows = list(
        csv.DictReader(StringIO(ui._detections_csv_bytes().decode("utf-8-sig")))
    )
    assert rows[0]["coordinate_space"] == "analysis_image_pixels"
    assert rows[0]["image_width"] == "120"
    assert rows[0]["image_height"] == "80"
    assert rows[0]["bbox_format"] == "xyxy_right_bottom_exclusive"


def test_csv_cells_neutralize_spreadsheet_formulas() -> None:
    from app import streamlit_app as ui

    assert ui._csv_cell("resistor") == "resistor"
    assert ui._csv_cell("=HYPERLINK(\"bad\")") == "'=HYPERLINK(\"bad\")"


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
