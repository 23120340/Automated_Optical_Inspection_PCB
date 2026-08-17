from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

from aoi_pipeline import (
    AOIPipeline,
    BoardConfig,
    BoundingBox,
    ClassProbability,
    ComponentClassification,
    Detection,
    MockComponentDetector,
    PipelineConfig,
    PreprocessConfig,
)


class _FakeClassifier:
    def classify(self, crops):
        return [
            ComponentClassification(
                crop_id=f"crop_{index + 1:04d}",
                detection_id=crop.detection_id,
                family="resistor" if index == 0 else "ic",
                probability=0.91,
                top_k=[ClassProbability("resistor" if index == 0 else "ic", 0.91)],
                unknown_score=0.09,
                decision="accept",
                model_version="test-v1",
                detector_hint=crop.label,
            )
            for index, crop in enumerate(crops)
        ]


def test_facade_translates_roi_detections_to_full_image(pcb_image) -> None:
    detector = MockComponentDetector(
        [Detection("capacitor", 0.87, BoundingBox(10, 12, 32, 30))]
    )
    pipeline = AOIPipeline(detector=detector)
    board = pipeline.detect_board(pcb_image)
    detections = pipeline.detect_components(pcb_image, board)
    x1, y1, _, _ = board.bbox.to_int()
    assert detections[0].bbox.x1 == 10 + x1
    assert detections[0].bbox.y1 == 12 + y1
    assert detections[0].metadata["roi_offset"] == [x1, y1]


def test_facade_accepts_ui_mapping_config() -> None:
    pipeline = AOIPipeline(
        config={
            "preprocess": {
                "resize_enabled": False,
                "denoise": "Bilateral",
                "clahe_clip": 3.5,
                "sharpen": 0.2,
            },
            "components": {
                "confidence": 0.42,
                "iou": 0.33,
                "max_candidates": 77,
                "device": "auto",
                "end2end": False,
            },
            "crops": {"padding": 7, "target_size": 128, "normalize": True},
            "classification": {"batch_size": 12, "top_k": 2, "device": "auto"},
        },
        detector=MockComponentDetector([]),
    )
    assert pipeline.config.preprocess.max_side is None
    assert pipeline.config.preprocess.denoise is True
    assert pipeline.config.preprocess.denoise_method == "bilateral"
    assert pipeline.config.preprocess.clahe_clip_limit == 3.5
    assert pipeline.config.cv_detector.nms_iou_threshold == 0.33
    assert pipeline.config.cv_detector.max_detections == 77
    assert pipeline.config.model_detector.confidence == 0.42
    assert pipeline.config.model_detector.iou == 0.33
    assert pipeline.config.model_detector.max_detections == 77
    assert pipeline.config.model_detector.device is None
    assert pipeline.config.model_detector.end2end is False
    assert pipeline.config.crop.padding_pixels == 7
    assert pipeline.config.crop.target_size == (128, 128)
    assert pipeline.config.classification.batch_size == 12
    assert pipeline.config.classification.top_k == 2
    assert pipeline.config.classification.device == "auto"


def test_end_to_end_run_and_json_zip_exports(tmp_path: Path, pcb_image) -> None:
    detector = MockComponentDetector(
        [
            Detection("resistor", 0.93, BoundingBox(15, 18, 42, 35)),
            Detection("ic", 0.89, BoundingBox(80, 55, 125, 100)),
        ]
    )
    config = PipelineConfig(
        preprocess=PreprocessConfig(
            max_side=None,
            denoise=False,
            white_balance=False,
            clahe=False,
            normalize=False,
            sharpen=False,
        ),
        board=BoardConfig(min_area_ratio=0.999, max_area_ratio=0.9999),
    )
    pipeline = AOIPipeline(config=config, detector=detector, classifier=_FakeClassifier())
    run = pipeline.run(pcb_image, source_name="synthetic.png")
    assert run.source_name == "synthetic.png"
    assert len(run.detections) == 2
    assert len(run.crops) == 2
    assert len(run.classifications) == 2
    assert run.alignment_result.method == "not_requested"
    assert run.board_region.method == "full_image_fallback"

    json_path = pipeline.export_json(run, tmp_path / "result.json")
    zip_path = pipeline.export_zip(run, tmp_path / "bundle")
    manifest = json.loads(json_path.read_text(encoding="utf-8"))
    assert manifest["summary"]["component_count"] == 2
    assert manifest["summary"]["labels"] == {"ic": 1, "resistor": 1}
    assert manifest["summary"]["classification_count"] == 2
    assert manifest["summary"]["families"] == {"ic": 1, "resistor": 1}
    assert manifest["summary"]["classification_decisions"] == {"accept": 2}
    assert manifest["classifications"][0]["detection_id"] == run.crops[0].detection_id

    with ZipFile(zip_path) as archive:
        members = set(archive.namelist())
        assert "manifest.json" in members
        assert "images/00_input.png" in members
        assert "images/03_annotated.png" in members
        assert len([member for member in members if member.startswith("crops/")]) == 2
