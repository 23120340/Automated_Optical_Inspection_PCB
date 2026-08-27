"""Contract and runtime tests for diagnostic solder instance segmentation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from aoi_pipeline import (
    AOIPipeline,
    BoundingBox,
    Detection,
    DetectorConfigurationError,
    MockComponentDetector,
    PipelineConfig,
    SOLDER_DEFECT_CLASS_NAMES,
    SOLDER_DEFECT_SOURCE,
    SolderDefectDetectionConfig,
    SolderDefectDetector,
    UltralyticsDetector,
    create_solder_defect_detector,
    validate_solder_defect_manifest,
)


class _Boxes:
    xyxy = np.asarray([[5.0, 6.0, 30.0, 28.0]], dtype=np.float32)
    conf = np.asarray([0.91], dtype=np.float32)
    cls = np.asarray([0.0], dtype=np.float32)

    def __len__(self) -> int:
        return 1


class _SegmentModel:
    names = {index: name for index, name in enumerate(SOLDER_DEFECT_CLASS_NAMES)}

    def __init__(self) -> None:
        self.last_predict_kwargs: dict[str, object] = {}

    def predict(self, **kwargs: object) -> list[SimpleNamespace]:
        self.last_predict_kwargs = kwargs
        masks = SimpleNamespace(
            xy=[np.asarray([[-2.0, 4.0], [10.0, 6.0], [105.0, 90.0]])]
        )
        return [
            SimpleNamespace(
                boxes=_Boxes(), masks=masks, names=self.names
            )
        ]


def _manifest(*, sha256: str = "0" * 64, size: int | None = None) -> dict:
    model = {
        "version": "solder-detector-yolov8m-seg-test",
        "architecture": "yolov8m-seg",
        "sha256": sha256,
    }
    if size is not None:
        model["bytes"] = size
    return {
        "schema_version": "aoi-external-yolo-segmentation/1.0",
        "task": "solder_defect_instance_segmentation",
        "pipeline_step": "standalone_solder_defect_localization",
        "model_format": "onnx",
        "class_names": list(SOLDER_DEFECT_CLASS_NAMES),
        "class_map": {
            str(index): name
            for index, name in enumerate(SOLDER_DEFECT_CLASS_NAMES)
        },
        "input": {
            "shape": [1, 3, 640, 640],
            "layout": "NCHW",
            "color_space": "RGB",
            "resize_mode": "letterbox",
        },
        "head": {
            "type": "YOLOv8 Segment",
            "end2end": False,
            "max_det": 300,
        },
        "postprocessing": {
            "recommended_confidence": 0.25,
            "recommended_iou_nms": 0.70,
            "mask_threshold": 0.50,
        },
        "model": model,
    }


def test_generic_adapter_keeps_task_source_and_mask_polygon(tmp_path: Path) -> None:
    model = _SegmentModel()
    detector = UltralyticsDetector(
        tmp_path / "segment.onnx",
        model=model,
        task="segment",
        source="diagnostic_segment",
    )

    detections = detector.detect(np.zeros((80, 100, 3), dtype=np.uint8))

    assert detector.task == "segment"
    assert detections[0].source == "diagnostic_segment"
    assert detections[0].metadata["task"] == "segment"
    assert detections[0].metadata["mask_polygon"] == [
        [0.0, 4.0],
        [10.0, 6.0],
        [100.0, 80.0],
    ]


def test_solder_segmenter_uses_segment_contract_and_manifest_defaults(
    tmp_path: Path,
) -> None:
    model = _SegmentModel()
    detector = create_solder_defect_detector(
        tmp_path / "best.onnx", _manifest(), model=model
    )
    assert isinstance(detector, SolderDefectDetector)

    detections = detector.detect(np.zeros((80, 100, 3), dtype=np.uint8))

    assert detector.task == "segment"
    assert detector.source == SOLDER_DEFECT_SOURCE
    assert detector.config.confidence == pytest.approx(0.25)
    assert detector.config.iou == pytest.approx(0.70)
    assert detector.config.image_size == 640
    assert detector.config.max_detections == 300
    assert model.last_predict_kwargs["end2end"] is False
    assert detections[0].label == "Dry_joint"
    assert detections[0].metadata["diagnostic_only"] is True
    assert detections[0].metadata["model_version"].endswith("-test")


def test_solder_segmenter_requires_a_complete_correct_manifest_pair(
    tmp_path: Path,
) -> None:
    with pytest.raises(DetectorConfigurationError, match="requires both"):
        create_solder_defect_detector(tmp_path / "best.onnx", None)

    classifier_manifest = _manifest()
    classifier_manifest["schema_version"] = "pcb-solder-defect-classifier/1.0"
    classifier_manifest["task"] = "solder_defect_classification"
    with pytest.raises(DetectorConfigurationError, match="schema"):
        create_solder_defect_detector(
            tmp_path / "best.onnx", classifier_manifest, model=_SegmentModel()
        )


def test_solder_segmenter_verifies_model_digest_before_loading(tmp_path: Path) -> None:
    model_path = tmp_path / "best.onnx"
    model_path.write_bytes(b"not the contracted model")
    manifest = _manifest(sha256="f" * 64, size=model_path.stat().st_size)

    with pytest.raises(DetectorConfigurationError, match="SHA-256"):
        create_solder_defect_detector(model_path, manifest)

    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest["model"]["sha256"] = digest
    manifest["model"]["bytes"] += 1
    with pytest.raises(DetectorConfigurationError, match="byte size"):
        create_solder_defect_detector(model_path, manifest)


def test_config_mapping_keeps_segment_detector_separate_from_roi_classifier() -> None:
    config = PipelineConfig.from_mapping(
        {
            "solder_defect_detection": {
                "model": "models/active/solder/segmenter/best.onnx",
                "manifest": "models/active/solder/segmenter/model_manifest.json",
                "conf": 0.31,
                "iou": 0.66,
                "imgsz": 640,
                "max_det": 123,
                "mask_threshold": 0.44,
                "device": "auto",
            }
        }
    )

    segment = config.solder_defect_detection
    assert segment.model_path.endswith("segmenter/best.onnx")
    assert segment.manifest_path.endswith("segmenter/model_manifest.json")
    assert segment.confidence == pytest.approx(0.31)
    assert segment.iou == pytest.approx(0.66)
    assert segment.image_size == 640
    assert segment.max_detections == 123
    assert segment.mask_threshold == pytest.approx(0.44)
    assert segment.device is None
    assert config.solder_grading.model_path is None
    assert config.solder_grading.manifest_path is None


def test_pipeline_exposes_diagnostic_stage_without_changing_component_detector() -> None:
    component = MockComponentDetector(
        [Detection("resistor", 0.9, BoundingBox(2, 3, 12, 14))]
    )
    solder = MockComponentDetector(
        [Detection("Dry_joint", 0.8, BoundingBox(20, 21, 31, 34))]
    )
    pipeline = AOIPipeline(
        detector=component,
        solder_defect_detector=solder,
    )
    image = np.zeros((50, 60, 3), dtype=np.uint8)

    components = pipeline.detect_components(image)
    findings = pipeline.detect_solder_defects(image)

    assert [item.label for item in components] == ["resistor"]
    assert [item.label for item in findings] == ["Dry_joint"]
    assert findings[0].metadata["diagnostic_only"] is True
    assert findings[0].metadata["coordinate_space"] == "analysis_image_pixels"
    assert pipeline.grade_solder([]) == []


def test_shipped_active_segmenter_manifest_matches_runtime_contract() -> None:
    manifest_path = (
        Path(__file__).parents[1]
        / "models"
        / "active"
        / "solder"
        / "segmenter"
        / "model_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    contract = validate_solder_defect_manifest(manifest)

    # Cố ý KHÔNG khoá vào tên lớp hay version cụ thể. Bản đầu assert đúng bốn
    # tên của model yolov8m-seg đầu tiên, nên nó kiểm "có phải model đó không"
    # chứ không kiểm "model đang ship có tự mô tả đúng không" -- và nó chặn luôn
    # mọi model 6.2 khác kể cả model do notebook của chính dự án train ra.
    assert contract.class_names, "manifest phải khai class_names"
    assert contract.ultralytics_task in {"segment", "detect"}
    assert contract.image_size > 0
    assert contract.model_version, "manifest phải khai model.version"

    model_path = manifest_path.with_name("best.onnx")
    assert model_path.stat().st_size == manifest["model"]["bytes"]
    digest = hashlib.sha256()
    with model_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == manifest["model"]["sha256"]
