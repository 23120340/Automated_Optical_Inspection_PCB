from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from aoi_pipeline.classification.package import PackageClassification
from aoi_pipeline.config import (
    PackageClassificationConfig,
    PipelineConfig,
    SolderJointConfig,
)
from aoi_pipeline.models import (
    BoundingBox,
    ClassProbability,
    Detection,
)
from aoi_pipeline.pipeline import AOIPipeline


class _PackageClassifier:
    def __init__(self, package_class: str, decision: str = "accept") -> None:
        self.package_class = package_class
        self.decision = decision

    def classify(self, crops):
        probability = 0.95 if self.decision == "accept" else 0.60
        return [
            PackageClassification(
                crop_id=str(crop.metadata.get("crop_id") or crop.filename),
                detection_id=crop.detection_id,
                package_class=self.package_class,
                probability=probability,
                top_k=[ClassProbability(self.package_class, probability)],
                unknown_score=1.0 - probability,
                decision=self.decision,
                model_version="package-test-v1",
                detector_hint=crop.label,
            )
            for crop in crops
        ]


def _config(**package_overrides: object) -> PipelineConfig:
    return PipelineConfig(
        solder=SolderJointConfig(
            split_pins=False,
            refine_to_metal=False,
            include_body_view=False,
            deconflict_neighbours=False,
        ),
        package_classification=PackageClassificationConfig(**package_overrides),
    )


def _detection(**metadata: object) -> Detection:
    return Detection(
        label="ic",
        confidence=0.92,
        bbox=BoundingBox(40, 45, 140, 95),
        detection_id="det_package",
        metadata=dict(metadata),
    )


def test_no_package_model_is_an_absolute_geometry_noop() -> None:
    image = np.zeros((160, 200, 3), dtype=np.uint8)
    detection = _detection()
    pipeline = AOIPipeline(_config())

    expected = pipeline.solder_cropper.extract(image, [detection])
    actual = pipeline.make_solder_crops(image, [detection])

    assert pipeline.last_package_classifications == []
    assert pipeline.last_package_topology_checks == []
    assert [item.joint.to_dict() for item in actual] == [
        item.joint.to_dict() for item in expected
    ]
    assert pipeline.last_package_detections[0] is detection


def test_only_accepted_package_prediction_changes_solder_geometry() -> None:
    image = np.zeros((160, 200, 3), dtype=np.uint8)
    detection = _detection()
    accepted = AOIPipeline(
        _config(), package_classifier=_PackageClassifier("hai_chan")
    )
    reviewed = AOIPipeline(
        _config(), package_classifier=_PackageClassifier("hai_chan", "review")
    )

    accepted_crops = accepted.make_solder_crops(image, [detection])
    reviewed_crops = reviewed.make_solder_crops(image, [detection])

    assert len(accepted_crops) == 2
    assert {item.joint.terminal_geometry for item in accepted_crops} == {
        "two_terminal"
    }
    assert accepted.last_package_topology_checks[0].status == "pass"
    accepted_metadata = accepted.last_package_detections[0].metadata
    assert accepted_metadata["package_profile"]["package_class"] == "hai_chan"

    assert len(reviewed_crops) == 4
    reviewed_metadata = reviewed.last_package_detections[0].metadata
    assert reviewed_metadata["package_prediction"]["decision"] == "review"
    assert "package_profile" not in reviewed_metadata
    assert reviewed.last_package_topology_checks == []


def test_existing_footprint_profile_outranks_accepted_model() -> None:
    image = np.zeros((160, 200, 3), dtype=np.uint8)
    profile = {
        "package_class": "ic_bon_ben",
        "terminal_geometry": "four_sided",
        "expected_pin_count_range": [8, 128],
        "lead_sides": 4,
        "source": "footprint",
    }
    detection = _detection(
        package_profile=profile,
        terminal_geometry_override="four_sided",
    )
    pipeline = AOIPipeline(
        _config(), package_classifier=_PackageClassifier("hai_chan")
    )

    crops = pipeline.make_solder_crops(image, [detection])

    assert len(crops) == 4
    assert pipeline.last_package_detections[0].metadata["package_profile"] == profile
    assert pipeline.last_package_detections[0].metadata["package_prediction"][
        "package_class"
    ] == "hai_chan"


def test_hidden_terminal_package_is_exported_without_fake_roi(
    tmp_path: Path, pcb_image
) -> None:
    from aoi_pipeline import BoardConfig, MockComponentDetector, PreprocessConfig

    detection = Detection(
        "ic",
        0.94,
        BoundingBox(40, 45, 140, 95),
        detection_id="det_hidden",
    )
    config = _config()
    config.preprocess = PreprocessConfig(
        max_side=None,
        denoise=False,
        white_balance=False,
        clahe=False,
        normalize=False,
        sharpen=False,
    )
    config.board = BoardConfig(min_area_ratio=0.999, max_area_ratio=0.9999)
    pipeline = AOIPipeline(
        config,
        detector=MockComponentDetector([detection]),
        package_classifier=_PackageClassifier("ic_khong_chan"),
    )

    run = pipeline.run(pcb_image, source_name="hidden.png")

    assert run.solder_crops == []
    assert run.package_classifications[0].package_class == "ic_khong_chan"
    assert run.package_topology_checks[0].status == "not_inspectable"
    assert run.to_dict()["summary"]["package_classes"] == {"ic_khong_chan": 1}
    assert run.detections[0].metadata["package_profile"]["source"] == (
        "onnx_package_classifier"
    )

    json_path = pipeline.export_json(run, tmp_path / "package.json")
    zip_path = pipeline.export_zip(run, tmp_path / "package.zip")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["package_topology_checks"][0]["status"] == "not_inspectable"
    with ZipFile(zip_path) as archive:
        members = set(archive.namelist())
        assert "packages/package_classifications.csv" in members
        assert "packages/package_topology_checks.csv" in members
