"""Step 6.2: measurement, rules, and the fusion between rules and a model."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from aoi_pipeline import (
    AOIPipeline,
    BoundingBox,
    ClassProbability,
    Detection,
    PipelineConfig,
    SolderGradingConfig,
    SolderInspector,
    measure_solder,
    segment_solder,
    solder_verdicts_csv,
)
from aoi_pipeline.models import SolderJoint, SolderJointCrop
from aoi_pipeline.detectors import MockComponentDetector
from aoi_pipeline.grading.classifier import ONNXSolderClassifier
from aoi_pipeline.grading.rules import grade_joint_by_rules

BOARD_COLOR = (40, 90, 40)
SOLDER_COLOR = (215, 215, 215)


def _roi(
    solder_boxes: list[tuple[int, int, int, int]],
    size: tuple[int, int] = (40, 80),
    solder_color: tuple[int, int, int] = SOLDER_COLOR,
) -> np.ndarray:
    """A ROI with bright grey 'solder' rectangles on a green mask."""

    height, width = size
    image = np.full((height, width, 3), BOARD_COLOR, np.uint8)
    for x1, y1, x2, y2 in solder_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), solder_color, -1)
    return image


def _joint(
    joint_id: str = "det_1_joint00",
    kind: str = "joint",
    position: str = "terminal_a",
    pin_index: int | None = None,
    detection_id: str = "det_1",
    geometry: str = "two_terminal",
    bbox: BoundingBox | None = None,
    pin: str | None = None,
) -> SolderJoint:
    return SolderJoint(
        detection_id=detection_id,
        joint_id=joint_id,
        label="resistor",
        kind=kind,
        bbox=bbox or BoundingBox(0, 0, 80, 40),
        terminal_geometry=geometry,
        position=position,
        pin_index=pin_index,
        pin=pin,
    )


def _crop(image: np.ndarray, joint: SolderJoint) -> SolderJointCrop:
    return SolderJointCrop(image=image, joint=joint, filename=f"{joint.joint_id}.png")


# --------------------------------------------------------------------------- #
# Segmentation and measurement
# --------------------------------------------------------------------------- #


def test_solder_is_separated_from_the_green_mask() -> None:
    image = _roi([(10, 10, 60, 30)])
    mask = segment_solder(image)
    assert mask.shape == image.shape[:2]
    covered = float(np.count_nonzero(mask)) / mask.size
    assert 0.10 < covered < 0.50
    # The solder rectangle is inside, the board corner is not.
    assert mask[20, 35] > 0
    assert mask[2, 2] == 0


def test_a_few_scattered_bright_pixels_are_not_erased() -> None:
    """Opening is there to drop speckle, not the signal.

    Reporting 0.0 for a ROI that does have metal in it is indistinguishable
    from a bare land, which is the most severe call this stage makes.
    """

    image = np.full((40, 40, 3), (30, 30, 30), np.uint8)
    for x, y in ((10, 10), (25, 20), (7, 31)):
        image[y, x] = (245, 245, 245)
    assert np.count_nonzero(segment_solder(image)) > 0


def test_measurements_scale_with_how_much_solder_is_present() -> None:
    small = measure_solder(_roi([(30, 15, 45, 25)]))
    large = measure_solder(_roi([(5, 5, 75, 35)]))
    assert small.solder_ratio < large.solder_ratio
    assert small.span_ratio < large.span_ratio
    assert large.solder_area_px > small.solder_area_px


def test_an_empty_roi_measures_to_zero_without_raising() -> None:
    features = measure_solder(np.zeros((0, 0, 3), np.uint8))
    assert features.solder_ratio == 0.0
    assert features.solder_area_px == 0


def test_centroid_offset_notices_solder_pushed_to_one_end() -> None:
    centred = measure_solder(_roi([(30, 12, 50, 28)]))
    off_centre = measure_solder(_roi([(2, 12, 22, 28)]))
    assert off_centre.centroid_offset_ratio > centred.centroid_offset_ratio


def test_edge_contact_reports_solder_running_off_both_ends() -> None:
    full = measure_solder(_roi([(0, 10, 79, 30)]))
    assert full.edge_contact_start > 0.5
    assert full.edge_contact_end > 0.5
    contained = measure_solder(_roi([(20, 10, 60, 30)]))
    assert contained.edge_contact_start == 0.0
    assert contained.edge_contact_end == 0.0


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #


def test_bare_land_is_missing_solder() -> None:
    config = SolderGradingConfig()
    features = measure_solder(_roi([]))
    label, reasons = grade_joint_by_rules(features, config)
    assert label == "missing_solder"
    assert reasons and "solder_ratio" in reasons[0]


def test_a_thin_deposit_is_insufficient() -> None:
    config = SolderGradingConfig()
    features = measure_solder(_roi([(35, 18, 45, 23)]))
    label, _ = grade_joint_by_rules(features, config)
    assert label in {"insufficient", "missing_solder"}


def test_a_flooded_roi_is_excess() -> None:
    config = SolderGradingConfig()
    features = measure_solder(_roi([(0, 0, 79, 39)]))
    label, reasons = grade_joint_by_rules(features, config)
    assert label == "excess"
    assert "thừa thiếc" in reasons[0]


def test_a_healthy_joint_passes() -> None:
    config = SolderGradingConfig()
    features = measure_solder(_roi([(12, 8, 68, 32)]))
    label, _ = grade_joint_by_rules(features, config)
    assert label == "good"


def test_every_rule_verdict_carries_a_reason() -> None:
    """A call an operator cannot trace to a number gets overridden on the floor."""

    config = SolderGradingConfig()
    for boxes in ([], [(35, 18, 45, 23)], [(0, 0, 79, 39)], [(12, 8, 68, 32)]):
        _, reasons = grade_joint_by_rules(measure_solder(_roi(boxes)), config)
        assert reasons and all(isinstance(item, str) and item for item in reasons)


def test_adjacent_pins_flooded_across_their_shared_edge_are_a_bridge() -> None:
    inspector = SolderInspector(SolderGradingConfig())
    flooded = _roi([(0, 0, 39, 39)], size=(40, 40))
    crops = [
        _crop(
            flooded,
            _joint(
                joint_id=f"det_2_joint{index:02d}",
                position=f"lead_left_pin{index:02d}",
                pin_index=index,
                detection_id="det_2",
                geometry="multi_pin",
                pin=str(index + 1),
                bbox=BoundingBox(0, index * 40, 40, index * 40 + 40),
            ),
        )
        for index in range(2)
    ]
    verdicts = inspector.inspect(crops)
    assert [verdict.label for verdict in verdicts] == ["bridge", "bridge"]
    assert all("biên chung" in verdict.reasons[0] for verdict in verdicts)


def test_one_lifted_end_of_a_two_terminal_part_is_flagged() -> None:
    """Neither ROI is odd alone; the evidence is only in the comparison."""

    inspector = SolderInspector(SolderGradingConfig())
    crops = [
        _crop(_roi([(12, 8, 68, 32)]), _joint("det_3_a", position="terminal_a", detection_id="det_3")),
        _crop(_roi([]), _joint("det_3_b", position="terminal_b", detection_id="det_3")),
    ]
    labels = [verdict.label for verdict in inspector.inspect(crops)]
    assert labels[0] == "good"
    assert labels[1] == "missing_solder"


# --------------------------------------------------------------------------- #
# Rules-only mode
# --------------------------------------------------------------------------- #


def test_the_stage_runs_with_no_model_at_all() -> None:
    inspector = SolderInspector(SolderGradingConfig())
    assert inspector.has_model is False
    verdicts = inspector.inspect([_crop(_roi([(12, 8, 68, 32)]), _joint())])
    assert len(verdicts) == 1
    assert verdicts[0].source == "rules"
    assert verdicts[0].model_label is None
    assert verdicts[0].decision == "accept"


def test_rules_only_defects_default_to_review_not_reject() -> None:
    inspector = SolderInspector(SolderGradingConfig())
    verdict = inspector.inspect([_crop(_roi([]), _joint())])[0]
    assert verdict.label == "missing_solder"
    assert verdict.decision == "review"

    strict = SolderInspector(SolderGradingConfig(rules_only_defect_decision="reject"))
    assert strict.inspect([_crop(_roi([]), _joint())])[0].decision == "reject"


def test_disabling_the_stage_produces_nothing() -> None:
    inspector = SolderInspector(SolderGradingConfig(enabled=False))
    assert inspector.inspect([_crop(_roi([]), _joint())]) == []


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #


class _StubClassifier:
    """Stands in for ONNXSolderClassifier without needing an ONNX file."""

    scope = "joint"
    model_version = "stub-1"

    def __init__(self, label: str, probability: float) -> None:
        self.label = label
        self.probability = probability

    def predict(self, images):
        other = "insufficient" if self.label == "good" else "good"
        return [
            [
                ClassProbability(self.label, self.probability),
                ClassProbability(other, 1.0 - self.probability),
            ]
            for _ in images
        ]

    def accept_threshold_for(self, label: str) -> float:
        return 0.85


def _inspect_with(model, image, config=None, joint=None):
    inspector = SolderInspector(config or SolderGradingConfig(), classifier=model)
    return inspector.inspect([_crop(image, joint or _joint())])[0]


def test_agreement_between_model_and_rules_is_accepted() -> None:
    verdict = _inspect_with(_StubClassifier("good", 0.97), _roi([(12, 8, 68, 32)]))
    assert verdict.label == "good"
    assert verdict.decision == "accept"
    assert verdict.source == "model+rules"


def test_disagreement_goes_to_review_rather_than_picking_a_winner() -> None:
    verdict = _inspect_with(_StubClassifier("good", 0.99), _roi([]))
    assert verdict.decision == "review"
    assert verdict.source in {"conflict", "escape_guard"}
    assert verdict.rule_label == "missing_solder"
    assert verdict.model_label == "good"


def test_a_low_confidence_agreement_still_goes_to_review() -> None:
    verdict = _inspect_with(_StubClassifier("good", 0.55), _roi([(12, 8, 68, 32)]))
    assert verdict.label == "good"
    assert verdict.decision == "review"


def test_the_escape_guard_overrides_a_confident_but_wrong_pass() -> None:
    """No confidence value may pass a land with almost no solder on it."""

    config = SolderGradingConfig(disagreement_is_review=False)
    verdict = _inspect_with(_StubClassifier("good", 0.999), _roi([]), config)
    assert verdict.label != "good"
    assert verdict.decision == "review"
    assert verdict.source == "escape_guard"
    assert any("chốt chặn" in reason for reason in verdict.reasons)


def test_the_escape_guard_can_be_switched_off_deliberately() -> None:
    config = SolderGradingConfig(
        disagreement_is_review=False, escape_guard_enabled=False
    )
    verdict = _inspect_with(_StubClassifier("good", 0.999), _roi([]), config)
    assert verdict.label == "good"
    assert verdict.source == "model"


def test_a_model_failure_degrades_to_rules_and_reports_it() -> None:
    class _Broken(_StubClassifier):
        def predict(self, images):
            raise RuntimeError("session died")

    inspector = SolderInspector(SolderGradingConfig(), classifier=_Broken("good", 0.9))
    verdicts = inspector.inspect([_crop(_roi([(12, 8, 68, 32)]), _joint())])
    assert verdicts[0].source == "rules"
    assert inspector.warnings and "quay về chấm bằng luật" in inspector.warnings[0]


def test_both_layers_are_kept_side_by_side_for_review() -> None:
    verdict = _inspect_with(_StubClassifier("cold", 0.92), _roi([(12, 8, 68, 32)]))
    assert verdict.rule_label == "good"
    assert verdict.model_label == "cold"
    assert verdict.top_k and verdict.top_k[0].label == "cold"


# --------------------------------------------------------------------------- #
# Manifest contract
# --------------------------------------------------------------------------- #


def _manifest(**overrides) -> dict:
    manifest = {
        "schema_version": "pcb-solder-defect-classifier/1.0",
        "task": "solder_defect_classification",
        "scope": "joint",
        "model_format": "onnx",
        "class_names": ["good", "insufficient", "cold"],
        "good_label": "good",
        "input": {
            "name": "input",
            "size": [128, 128],
            "color_space": "RGB",
            "resize_mode": "letterbox",
            "letterbox_value": 114,
            "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        },
        "output": {"name": "logits"},
        "calibration": {"temperature": 1.0},
        "decision_thresholds": {"accept": 0.85, "review": 0.5},
        "model": {"version": "test-1"},
    }
    manifest.update(overrides)
    return manifest


class _Session:
    def __init__(self, logits: np.ndarray) -> None:
        self.logits = logits

    def get_inputs(self):
        return [type("I", (), {"name": "input"})()]

    def get_outputs(self):
        return [type("O", (), {"name": "logits"})()]

    def run(self, names, feeds):
        batch = next(iter(feeds.values())).shape[0]
        return [np.repeat(self.logits[None, :], batch, axis=0)]


def test_a_valid_manifest_drives_the_runtime() -> None:
    classifier = ONNXSolderClassifier(
        "unused.onnx", _manifest(), session=_Session(np.array([3.0, 0.0, 0.0], np.float32))
    )
    assert classifier.class_names == ["good", "insufficient", "cold"]
    assert classifier.good_label == "good"
    prediction = classifier.predict([np.zeros((20, 20, 3), np.uint8)])[0]
    assert prediction[0].label == "good"
    assert prediction[0].probability > 0.9


class _TTASession:
    """Returns distinct, known logits per one of the 4 flip views so the
    caller's averaging can be checked exactly, not just its shape."""

    PER_VIEW_LOGITS = np.array(
        [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0], [1.0, 1.0, 1.0]],
        dtype=np.float32,
    )

    def __init__(self) -> None:
        self.feed: np.ndarray | None = None

    def get_inputs(self):
        return [type("I", (), {"name": "input"})()]

    def get_outputs(self):
        return [type("O", (), {"name": "logits"})()]

    def run(self, names, feeds):
        self.feed = next(iter(feeds.values()))
        batch = self.feed.shape[0] // 4
        return [np.repeat(self.PER_VIEW_LOGITS, batch, axis=0)]


def _asymmetric_roi() -> np.ndarray:
    image = np.zeros((20, 20, 3), np.uint8)
    image[:, :10, 2] = 255
    image[10:, :, 1] = 200
    return image


def test_solder_tta_off_sends_a_single_view() -> None:
    session = _Session(np.array([3.0, 0.0, 0.0], np.float32))
    classifier = ONNXSolderClassifier(
        "unused.onnx", _manifest(), SolderGradingConfig(tta=False), session=session
    )
    predictions = classifier.predict([_asymmetric_roi(), _asymmetric_roi()])
    # No TTA batching blow-up: one prediction per ROI, not four.
    assert len(predictions) == 2
    assert predictions[0][0].label == "good"


def test_solder_tta_on_stacks_four_flip_views_and_averages() -> None:
    session = _TTASession()
    classifier = ONNXSolderClassifier(
        "unused.onnx", _manifest(), SolderGradingConfig(tta=True), session=session
    )
    roi = _asymmetric_roi()
    base = classifier._preprocess(roi)
    predictions = classifier.predict([roi])[0]

    assert session.feed.shape == (4, 3, 128, 128)
    np.testing.assert_array_equal(session.feed[0], base)
    np.testing.assert_array_equal(session.feed[1], base[:, :, ::-1])
    np.testing.assert_array_equal(session.feed[2], base[:, ::-1, :])
    np.testing.assert_array_equal(session.feed[3], base[:, ::-1, ::-1])

    exponentials = np.exp(
        _TTASession.PER_VIEW_LOGITS - _TTASession.PER_VIEW_LOGITS.max(axis=1, keepdims=True)
    )
    per_view_probs = exponentials / exponentials.sum(axis=1, keepdims=True)
    expected = per_view_probs.mean(axis=0)
    by_label = {item.label: item.probability for item in predictions}
    for label, index in (("good", 0), ("insufficient", 1), ("cold", 2)):
        assert by_label[label] == pytest.approx(float(expected[index]), rel=1e-5)


@pytest.mark.parametrize(
    "override,message",
    [
        ({"schema_version": "other/1.0"}, "schema"),
        ({"task": "something_else"}, "task"),
        ({"model_format": "torchscript"}, "model_format"),
        ({"class_names": ["only_one"]}, "class_names"),
        ({"good_label": "not_a_class"}, "good_label"),
        ({"scope": "wafer"}, "scope"),
    ],
)
def test_a_broken_manifest_is_refused_rather_than_guessed(override, message) -> None:
    """Guessing a class order maps every defect onto a passing label."""

    from aoi_pipeline.exceptions import ClassifierConfigurationError

    with pytest.raises(ClassifierConfigurationError, match=message):
        ONNXSolderClassifier("unused.onnx", _manifest(**override), session=_Session(np.zeros(3, np.float32)))


def test_half_a_contract_is_refused() -> None:
    from aoi_pipeline import create_solder_classifier
    from aoi_pipeline.exceptions import ClassifierConfigurationError

    assert create_solder_classifier(None, None) is None
    with pytest.raises(ClassifierConfigurationError, match="both"):
        create_solder_classifier("best.onnx", None)


# --------------------------------------------------------------------------- #
# Pipeline and export
# --------------------------------------------------------------------------- #


def _board_with_parts() -> tuple[np.ndarray, list[Detection]]:
    image = np.full((300, 400, 3), BOARD_COLOR, np.uint8)
    detections = []
    for index, x in enumerate((60, 180, 300)):
        cv2.rectangle(image, (x - 26, 140), (x - 12, 160), SOLDER_COLOR, -1)
        cv2.rectangle(image, (x + 12, 140), (x + 26, 160), SOLDER_COLOR, -1)
        cv2.rectangle(image, (x - 14, 136), (x + 14, 164), (25, 25, 25), -1)
        detections.append(
            Detection("resistor", 0.9, BoundingBox(x - 14, 136, x + 14, 164))
        )
    return image, detections


def test_the_pipeline_grades_every_roi_without_a_model(tmp_path: Path) -> None:
    image, detections = _board_with_parts()
    pipeline = AOIPipeline(PipelineConfig(), detector=MockComponentDetector(detections))
    run = pipeline.run(image, source_name="board.png")

    assert run.solder_verdicts
    assert len(run.solder_verdicts) == len(run.solder_crops)
    assert all(verdict.source == "rules" for verdict in run.solder_verdicts)
    assert any("Bước 6.2" in warning for warning in run.warnings)

    summary = run.to_dict()["summary"]
    assert summary["solder_verdicts"]
    assert summary["solder_decisions"]


def test_verdicts_export_with_the_measurement_behind_them(tmp_path: Path) -> None:
    image, detections = _board_with_parts()
    pipeline = AOIPipeline(PipelineConfig(), detector=MockComponentDetector(detections))
    run = pipeline.run(image, source_name="board.png")

    rows = list(csv.DictReader(io.StringIO(solder_verdicts_csv(run))))
    assert len(rows) == len(run.solder_verdicts)
    assert rows[0]["solder_ratio"]
    assert rows[0]["reasons"]
    assert rows[0]["decision"] in {"accept", "review", "reject"}

    archive_path = pipeline.export_zip(run, tmp_path / "run.zip")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
    assert "solder_joints/solder_verdicts.csv" in names
    assert "images/06_solder_verdicts.png" in names


def test_measuring_uses_the_raw_roi_not_the_letterboxed_crop() -> None:
    """Letterbox padding would count into every area ratio and halve them."""

    image, detections = _board_with_parts()
    pipeline = AOIPipeline(PipelineConfig(), detector=MockComponentDetector(detections))
    crops = pipeline.make_solder_crops(image, detections)
    joints = [crop for crop in crops if crop.joint.kind == "joint"]

    from_frame = pipeline.solder_inspector.inspect(joints, image)
    from_padded_crop = pipeline.solder_inspector.inspect(joints)
    frame_ratio = np.mean([v.features.solder_ratio for v in from_frame])
    crop_ratio = np.mean([v.features.solder_ratio for v in from_padded_crop])
    assert frame_ratio > crop_ratio
