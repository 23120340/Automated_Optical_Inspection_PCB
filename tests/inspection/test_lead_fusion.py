"""Step 5.5: detected lead/pad boxes take precedence, derived geometry fills the gaps."""

from __future__ import annotations

import numpy as np
import pytest

from aoi_pipeline import (
    AOIPipeline,
    BoundingBox,
    Detection,
    LeadFusionConfig,
    PipelineConfig,
    fuse_detected_leads,
    split_lead_detections,
)
from aoi_pipeline.detectors import MockComponentDetector
from aoi_pipeline.leads import assign_leads_to_components
from aoi_pipeline.solder import SolderJointCropper


def _board() -> np.ndarray:
    return np.full((300, 400, 3), (40, 90, 40), np.uint8)


def _resistor(x: float = 200.0, y: float = 150.0) -> Detection:
    return Detection("resistor", 0.9, BoundingBox(x - 20, y - 10, x + 20, y + 10))


def _lead(x: float, y: float, label: str = "pads", confidence: float = 0.8) -> Detection:
    return Detection(label, confidence, BoundingBox(x - 8, y - 8, x + 8, y + 8))


def _derived(detections):
    return SolderJointCropper().derive(_board(), detections)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #


def test_lead_classes_are_separated_from_component_bodies() -> None:
    detections = [_resistor(), _lead(175, 150), _lead(225, 150), Detection("ic", 0.8, BoundingBox(0, 0, 40, 40))]
    bodies, leads = split_lead_detections(detections)
    assert [d.label for d in bodies] == ["resistor", "ic"]
    assert len(leads) == 2


@pytest.mark.parametrize("label", ["pads", "pad", "pins", "pin", "lead", "leads", "PADS"])
def test_every_lead_alias_is_recognised(label: str) -> None:
    _, leads = split_lead_detections([_lead(10, 10, label=label)])
    assert len(leads) == 1


# --------------------------------------------------------------------------- #
# Assignment
# --------------------------------------------------------------------------- #


def test_leads_attach_to_the_component_they_sit_beside() -> None:
    """Leads sit just outside the body box, so containment alone finds nothing."""

    body = _resistor()
    leads = [_lead(175, 150), _lead(225, 150)]
    assigned = assign_leads_to_components([body], leads, LeadFusionConfig())
    assert assigned[body.detection_id] == leads


def test_a_lead_far_from_every_component_is_left_unassigned() -> None:
    body = _resistor()
    stray = _lead(20, 20)
    assigned = assign_leads_to_components([body], [stray], LeadFusionConfig())
    assert assigned == {}


def test_a_low_confidence_lead_is_ignored() -> None:
    body = _resistor()
    weak = _lead(175, 150, confidence=0.10)
    assigned = assign_leads_to_components([body], [weak], LeadFusionConfig())
    assert assigned == {}


def test_a_lead_goes_to_the_nearer_of_two_components() -> None:
    left = _resistor(100.0, 150.0)
    right = _resistor(300.0, 150.0)
    lead = _lead(325, 150)
    assigned = assign_leads_to_components([left, right], [lead], LeadFusionConfig())
    assert list(assigned) == [right.detection_id]


# --------------------------------------------------------------------------- #
# Fusion precedence
# --------------------------------------------------------------------------- #


def test_with_no_lead_detections_the_derived_rois_pass_through() -> None:
    body = _resistor()
    derived = _derived([body])
    result = fuse_detected_leads([body], [], derived)
    assert result.joints == derived
    assert result.used_detected == 0
    assert result.used_derived == sum(1 for j in derived if j.kind == "joint")


def test_a_detected_lead_replaces_the_derived_roi_it_covers() -> None:
    body = _resistor()
    derived = _derived([body])
    terminals = [j for j in derived if j.kind == "joint"]
    # Put the detection right on top of one derived terminal.
    target = terminals[0]
    lead = Detection("pads", 0.8, target.bbox)

    result = fuse_detected_leads([body], [lead], derived)
    joints = [j for j in result.joints if j.kind == "joint"]
    assert result.used_detected == 1
    # The other terminal is still covered by its derived ROI.
    assert result.used_derived == len(terminals) - 1
    assert len(joints) == len(terminals)
    assert any(j.source == "detected" for j in joints)
    assert any(j.source == "derived" for j in joints)


def test_a_partially_detected_component_keeps_the_terminal_the_model_missed() -> None:
    """The end the detector could not see is the end most likely to be defective."""

    body = _resistor()
    derived = _derived([body])
    terminals = [j for j in derived if j.kind == "joint"]
    assert len(terminals) == 2
    lead = Detection("pads", 0.9, terminals[0].bbox)

    result = fuse_detected_leads([body], [lead], derived)
    joints = [j for j in result.joints if j.kind == "joint"]
    # Both terminals still have a ROI: one detected, one derived.
    assert len(joints) == 2
    assert {j.source for j in joints} == {"detected", "derived"}


def test_a_detection_that_overlaps_nothing_is_added_not_swapped() -> None:
    body = _resistor()
    derived = _derived([body])
    terminals = [j for j in derived if j.kind == "joint"]
    # A lead the derived geometry never predicted, e.g. a thermal pad.
    lead = _lead(200, 168)

    result = fuse_detected_leads([body], [lead], derived)
    joints = [j for j in result.joints if j.kind == "joint"]
    assert len(joints) == len(terminals) + 1
    assert result.used_detected == 1
    assert result.used_derived == len(terminals)


def test_detected_joints_record_where_they_came_from() -> None:
    body = _resistor()
    derived = _derived([body])
    lead = Detection("pins", 0.77, [j for j in derived if j.kind == "joint"][0].bbox)
    result = fuse_detected_leads([body], [lead], derived)
    detected = next(j for j in result.joints if j.source == "detected")
    assert detected.metadata["lead_label"] == "pins"
    assert detected.metadata["lead_confidence"] == pytest.approx(0.77)
    assert detected.metadata["lead_detection_id"] == lead.detection_id


def test_unassigned_leads_are_reported_not_forced_onto_a_component() -> None:
    body = _resistor()
    derived = _derived([body])
    result = fuse_detected_leads([body], [_lead(20, 20)], derived)
    assert result.unassigned_leads == 1
    assert result.warnings and "too far from any component" in result.warnings[0]
    # Not attached to the body it does not belong to: every ROI carrying the
    # body's detection_id must still come from the derived geometry.
    body_rois = [j for j in result.joints if j.detection_id == body.detection_id]
    assert all(j.source != "detected" for j in body_rois)


def test_an_unassigned_lead_is_kept_as_its_own_roi_not_thrown_away() -> None:
    """A pad that belongs to no component is still a real joint.

    Test points, unpopulated footprints, and pads whose component the detector
    missed all land here. Measured on the shipped detector, ``pads`` scores
    0.712 precision against 0.072 recall -- it fires rarely but is usually
    right when it does, so discarding a confident firing throws away the
    scarcest evidence in the pipeline.
    """

    body = _resistor()
    derived = _derived([body])
    orphan = _lead(20, 20, confidence=0.85)
    result = fuse_detected_leads([body], [orphan], derived)

    standalone = [
        j for j in result.joints
        if j.metadata.get("lead_detection_id") == orphan.detection_id
    ]
    assert len(standalone) == 1
    assert standalone[0].source == "detected"
    assert standalone[0].bbox == orphan.bbox
    assert standalone[0].terminal_geometry == "pad_only"
    # It stands on its own rather than being filed under the wrong component.
    assert standalone[0].detection_id == orphan.detection_id


def test_an_unassigned_lead_below_the_confidence_floor_stays_dropped() -> None:
    """Keeping orphans must not resurrect detections the threshold rejected."""

    body = _resistor()
    derived = _derived([body])
    weak = _lead(20, 20, confidence=0.10)
    result = fuse_detected_leads([body], [weak], derived)

    assert result.unassigned_leads == 0
    assert result.used_detected == 0
    assert not any(
        j.metadata.get("lead_detection_id") == weak.detection_id for j in result.joints
    )


def test_keeping_unassigned_leads_can_be_switched_off() -> None:
    body = _resistor()
    derived = _derived([body])
    orphan = _lead(20, 20, confidence=0.85)
    result = fuse_detected_leads(
        [body], [orphan], derived, LeadFusionConfig(keep_unassigned_leads=False)
    )
    assert result.unassigned_leads == 1
    assert result.used_detected == 0
    assert "DROPPED" in result.warnings[0]
    assert not any(
        j.metadata.get("lead_detection_id") == orphan.detection_id for j in result.joints
    )


def test_lead_fusion_can_be_switched_off() -> None:
    body = _resistor()
    derived = _derived([body])
    lead = Detection("pads", 0.9, [j for j in derived if j.kind == "joint"][0].bbox)
    result = fuse_detected_leads([body], [lead], derived, LeadFusionConfig(enabled=False))
    assert result.used_detected == 0
    assert result.joints == derived


# --------------------------------------------------------------------------- #
# Through the pipeline
# --------------------------------------------------------------------------- #


def test_the_pipeline_uses_lead_detections_when_the_detector_reports_them() -> None:
    body = _resistor()
    detections = [body, _lead(175, 150), _lead(225, 150)]
    pipeline = AOIPipeline(PipelineConfig(), detector=MockComponentDetector(detections))
    run = pipeline.run(_board(), source_name="board.png")

    sources = {crop.joint.source for crop in run.solder_crops}
    assert "detected" in sources
    assert pipeline.last_lead_fusion.used_detected == 2
    assert any("chân/pad thật" in warning for warning in run.warnings)


def test_a_pad_detection_is_not_also_treated_as_a_component_to_derive_around() -> None:
    """Deriving terminals around a pad box would inspect the pad's own corners."""

    body = _resistor()
    pad = _lead(175, 150)
    pipeline = AOIPipeline(PipelineConfig(), detector=MockComponentDetector([body, pad]))
    run = pipeline.run(_board(), source_name="board.png")

    detection_ids = {crop.joint.detection_id for crop in run.solder_crops}
    assert pad.detection_id not in detection_ids
    assert body.detection_id in detection_ids


def test_the_pipeline_is_unchanged_when_the_detector_reports_no_leads() -> None:
    body = _resistor()
    pipeline = AOIPipeline(PipelineConfig(), detector=MockComponentDetector([body]))
    run = pipeline.run(_board(), source_name="board.png")
    assert run.solder_crops
    assert all(crop.joint.source in {"derived", "cad", "cad+derived"} for crop in run.solder_crops)
    assert pipeline.last_lead_fusion.used_detected == 0
