"""Step 5.5 tests: derived solder-joint ROIs, crops and their export."""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from aoi_pipeline import (
    BoundingBox,
    CropConfig,
    Detection,
    SolderJointConfig,
    SolderJointCropper,
    derive_solder_joints,
    terminal_geometry,
)
from aoi_pipeline.cropping import ComponentCropper
from aoi_pipeline.exporters import solder_joints_csv
from aoi_pipeline.overlays import render_solder_overlay
from aoi_pipeline.solder import estimate_component_angle

BOARD_SIZE = (400, 700)


def _blank_board() -> np.ndarray:
    return np.full((*BOARD_SIZE, 3), (40, 90, 40), np.uint8)


def _chip_board() -> tuple[np.ndarray, Detection]:
    """Horizontal chip part: dark body at 95..165 with bright lands either side."""

    image = _blank_board()
    cv2.rectangle(image, (60, 96), (100, 124), (200, 200, 200), -1)
    cv2.rectangle(image, (160, 96), (200, 124), (200, 200, 200), -1)
    cv2.rectangle(image, (95, 92), (165, 128), (25, 25, 25), -1)
    return image, Detection("resistor", 0.9, BoundingBox(95, 92, 165, 128))


def _soic_board(pins: int = 6) -> tuple[np.ndarray, Detection]:
    """SOIC with gull-wing leads on the left and right edges only."""

    image = _blank_board()
    cv2.rectangle(image, (440, 120), (560, 190), (20, 20, 20), -1)
    for index in range(pins):
        y = 128 + index * 11
        cv2.rectangle(image, (424, y), (441, y + 6), (215, 215, 215), -1)
        cv2.rectangle(image, (559, y), (576, y + 6), (215, 215, 215), -1)
    return image, Detection("ic", 0.95, BoundingBox(440, 120, 560, 190))


def _joints(detection: Detection, image=None, config=None):
    return derive_solder_joints(
        detection, BOARD_SIZE[1], BOARD_SIZE[0], config or SolderJointConfig(), image
    )


def _by_kind(joints, kind: str):
    return [joint for joint in joints if joint.kind == kind]


# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "label,expected",
    [
        ("resistor", "two_terminal"),
        ("CAPACITOR", "two_terminal"),
        ("led", "two_terminal"),
        ("ic", "multi_pin"),
        ("connector", "multi_pin"),
        ("pads", "pad_only"),
        ("something_new", "multi_pin"),
    ],
)
def test_terminal_geometry_taxonomy(label: str, expected: str) -> None:
    assert terminal_geometry(label) == expected


# --------------------------------------------------------------------------- #
# Two-terminal geometry
# --------------------------------------------------------------------------- #


def test_two_terminal_yields_one_roi_per_end_covering_the_land() -> None:
    image, detection = _chip_board()
    joints = _joints(detection, image)
    terminals = _by_kind(joints, "joint")
    assert len(terminals) == 2
    assert {joint.position for joint in terminals} == {"terminal_a", "terminal_b"}

    left, right = sorted(terminals, key=lambda joint: joint.bbox.x1)
    # The ROIs must reach past the detector box onto the lands at 60..100 and
    # 160..200; this is exactly what the body-tight box cannot show.
    assert left.bbox.x1 < 95 and left.bbox.x2 > 95
    assert right.bbox.x2 > 165 and right.bbox.x1 < 165
    assert left.bbox.x1 <= 75
    assert right.bbox.x2 >= 185
    # They must not swallow each other.
    assert left.bbox.x2 < right.bbox.x1


def test_two_terminal_follows_the_long_axis_when_the_part_is_vertical() -> None:
    detection = Detection("capacitor", 0.8, BoundingBox(292, 235, 328, 305))
    terminals = _by_kind(_joints(detection), "joint")
    assert len(terminals) == 2
    top, bottom = sorted(terminals, key=lambda joint: joint.bbox.y1)
    assert top.bbox.y1 < 235
    assert bottom.bbox.y2 > 305
    # Both ROIs span the same horizontal range, i.e. they are stacked, not
    # placed side by side.
    assert top.bbox.x1 == pytest.approx(bottom.bbox.x1)


def test_body_view_contains_every_joint() -> None:
    image, detection = _chip_board()
    joints = _joints(detection, image)
    bodies = _by_kind(joints, "body")
    assert len(bodies) == 1
    body = bodies[0].bbox
    for joint in _by_kind(joints, "joint"):
        assert body.x1 <= joint.bbox.x1 and body.x2 >= joint.bbox.x2
        assert body.y1 <= joint.bbox.y1 and body.y2 >= joint.bbox.y2


def test_body_view_can_be_disabled() -> None:
    image, detection = _chip_board()
    joints = _joints(detection, image, SolderJointConfig(include_body_view=False))
    assert _by_kind(joints, "body") == []


# --------------------------------------------------------------------------- #
# Multi-pin geometry
# --------------------------------------------------------------------------- #


def test_pin_free_edges_are_dropped_by_the_energy_filter() -> None:
    image, detection = _soic_board()
    positions = {joint.position for joint in _by_kind(_joints(detection, image), "joint")}
    assert positions == {"lead_left", "lead_right"}


def test_all_edges_are_kept_without_an_image() -> None:
    _, detection = _soic_board()
    positions = {joint.position for joint in _by_kind(_joints(detection), "joint")}
    assert positions == {"lead_left", "lead_right", "lead_top", "lead_bottom"}


def test_energy_filter_can_be_disabled() -> None:
    image, detection = _soic_board()
    joints = _joints(detection, image, SolderJointConfig(lead_band_energy_ratio=None))
    assert len(_by_kind(joints, "joint")) == 4


def test_split_pins_finds_one_roi_per_lead() -> None:
    image, detection = _soic_board(pins=6)
    joints = _by_kind(_joints(detection, image, SolderJointConfig(split_pins=True)), "joint")
    assert len(joints) == 12
    assert all(joint.pin_index is not None for joint in joints)

    left = sorted(
        (joint for joint in joints if joint.position.startswith("lead_left")),
        key=lambda joint: joint.bbox.y1,
    )
    assert [joint.pin_index for joint in left] == [0, 1, 2, 3, 4, 5]
    # Each ROI must sit on its own lead, drawn at y = 128 + 11 * index.
    for index, joint in enumerate(left):
        expected_center = 128 + 11 * index + 3
        center = (joint.bbox.y1 + joint.bbox.y2) / 2.0
        assert abs(center - expected_center) <= 3.0


def test_split_pins_falls_back_to_the_band_on_an_unreadable_edge() -> None:
    """A featureless band must stay one ROI rather than invent pins."""

    image = _blank_board()
    cv2.rectangle(image, (440, 120), (560, 190), (20, 20, 20), -1)
    detection = Detection("ic", 0.9, BoundingBox(440, 120, 560, 190))
    joints = _by_kind(
        _joints(
            detection,
            image,
            SolderJointConfig(split_pins=True, lead_band_energy_ratio=None),
        ),
        "joint",
    )
    assert len(joints) == 4
    assert all(joint.pin_index is None for joint in joints)


def test_pad_class_gets_a_single_roi() -> None:
    detection = Detection("pads", 0.6, BoundingBox(100, 100, 140, 130))
    joints = _by_kind(_joints(detection), "joint")
    assert len(joints) == 1
    assert joints[0].position == "pad"
    assert joints[0].bbox.x1 < 100 and joints[0].bbox.x2 > 140


# --------------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------------- #


def test_rois_stay_inside_the_image_for_a_component_on_the_edge() -> None:
    detection = Detection("resistor", 0.7, BoundingBox(0, 0, 40, 18))
    for joint in _joints(detection):
        assert joint.bbox.x1 >= 0 and joint.bbox.y1 >= 0
        assert joint.bbox.x2 <= BOARD_SIZE[1] and joint.bbox.y2 <= BOARD_SIZE[0]


def test_tiny_detection_produces_no_undersized_roi() -> None:
    detection = Detection("resistor", 0.7, BoundingBox(10, 10, 13, 12))
    for joint in _joints(detection, config=SolderJointConfig(min_roi_pixels=6)):
        assert joint.bbox.width >= 6 and joint.bbox.height >= 6


def test_orientation_estimation_is_opt_in_and_abstains_when_axis_aligned() -> None:
    image, detection = _chip_board()
    default = _joints(detection, image)
    assert all(joint.angle == 0.0 for joint in default)

    estimated = _joints(
        detection, image, SolderJointConfig(estimate_orientation=True)
    )
    # The part really is axis aligned, so the estimator must fall back to 0.
    assert all(joint.angle == 0.0 for joint in estimated)


def test_estimate_component_angle_recovers_a_rotated_body() -> None:
    image = _blank_board()
    box_points = cv2.boxPoints(((200.0, 200.0), (90.0, 34.0), 20.0))
    cv2.fillPoly(image, [np.int32(box_points)], (20, 20, 20))
    angle = estimate_component_angle(image, BoundingBox(140, 150, 260, 250))
    assert abs(angle - 20.0) <= 6.0


def test_estimate_component_angle_abstains_on_a_blank_patch() -> None:
    image = _blank_board()
    assert estimate_component_angle(image, BoundingBox(10, 10, 110, 60)) == 0.0


# --------------------------------------------------------------------------- #
# Crops and export
# --------------------------------------------------------------------------- #


def test_cropper_produces_normalized_crops_and_files(tmp_path: Path) -> None:
    image, detection = _chip_board()
    cropper = SolderJointCropper(SolderJointConfig(target_size=(96, 96)))
    crops = cropper.extract(image, [detection], tmp_path)
    assert crops
    for crop in crops:
        assert crop.image.shape == (96, 96, 3)
        assert crop.path is not None and crop.path.exists()
        assert crop.joint.kind in {"joint", "body"}
    assert len({crop.filename for crop in crops}) == len(crops)


def test_cropper_respects_the_enabled_flag() -> None:
    image, detection = _chip_board()
    cropper = SolderJointCropper(SolderJointConfig(enabled=False))
    assert cropper.extract(image, [detection]) == []


def test_solder_crops_do_not_disturb_the_step_6_1_component_crop() -> None:
    """Step 6.1 was trained on body-tight crops; widening them silently is a
    regression, so the default component crop must stay exactly as it was."""

    image, detection = _chip_board()
    default_box = ComponentCropper(CropConfig()).extract(image, [detection])[0].crop_bbox
    solder_box = ComponentCropper(
        CropConfig(solder_aware_padding=True)
    ).extract(image, [detection])[0].crop_bbox
    assert solder_box.width > default_box.width
    # The default must stay the notebook recipe: pad = 0.15 * max(w, h) on
    # every side, no squaring.
    pad = 0.15 * max(detection.bbox.width, detection.bbox.height)
    assert default_box.width == pytest.approx(detection.bbox.width + 2 * pad, abs=1.0)
    assert default_box.height == pytest.approx(detection.bbox.height + 2 * pad, abs=1.0)


def test_solder_aware_padding_uses_a_wider_margin_along_the_long_axis() -> None:
    image, detection = _chip_board()
    config = CropConfig(solder_aware_padding=True, square=False, target_size=None)
    box = ComponentCropper(config).extract(image, [detection])[0].crop_bbox
    grown_x = box.width - detection.bbox.width
    grown_y = box.height - detection.bbox.height
    assert grown_x > grown_y


def test_solder_joints_csv_has_one_row_per_crop_and_a_label_column(tmp_path: Path) -> None:
    from aoi_pipeline import AOIPipeline, PipelineConfig
    from aoi_pipeline.detectors import MockComponentDetector

    image, detection = _chip_board()
    pipeline = AOIPipeline(PipelineConfig(), detector=MockComponentDetector([detection]))
    run = pipeline.run(image, source_name="synthetic.png")
    assert run.solder_crops

    rows = list(csv.DictReader(io.StringIO(solder_joints_csv(run))))
    assert len(rows) == len(run.solder_crops)
    assert rows[0]["defect_class"] == ""
    assert rows[0]["label"] == "resistor"
    assert {row["kind"] for row in rows} <= {"joint", "body"}

    archive_path = pipeline.export_zip(run, tmp_path / "run.zip")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
    assert "solder_joints/solder_joints.csv" in names
    assert any(name.startswith("solder_joints/joints/") for name in names)
    assert "images/05_solder_rois.png" in names


def test_render_solder_overlay_does_not_mutate_the_source() -> None:
    image, detection = _chip_board()
    before = image.copy()
    overlay = render_solder_overlay(image, _joints(detection, image))
    assert np.array_equal(image, before)
    assert not np.array_equal(overlay, before)


# --------------------------------------------------------------------------- #
# Streamlit step-5.5 view helpers
# --------------------------------------------------------------------------- #


def _ui_records():
    from app.pipeline_bridge import SolderCropRecord

    pixels = np.zeros((8, 8, 3), np.uint8)
    return [
        SolderCropRecord(
            "j1", "d1", "resistor", "joint", "terminal_a", "two_terminal",
            pixels, (4, 4, 20, 18), 0.87,
        ),
        SolderCropRecord(
            "j2", "d1", "resistor", "body", "body", "two_terminal",
            pixels, (2, 2, 24, 20), 0.87,
        ),
    ]


def test_ui_label_sheet_has_an_empty_defect_class_per_roi() -> None:
    from app.streamlit_app import _solder_frame

    frame = _solder_frame(_ui_records())
    assert len(frame) == 2
    assert list(frame["defect_class"]) == ["", ""]
    assert list(frame["roi_width_px"]) == [16, 22]


def test_ui_overlay_is_non_destructive_and_honours_the_body_toggle() -> None:
    from app.streamlit_app import _draw_solder_overlay

    records = _ui_records()
    image = np.zeros((40, 40, 3), np.uint8)
    with_body = _draw_solder_overlay(image, records, True)
    without_body = _draw_solder_overlay(image, records, False)
    assert image.sum() == 0
    assert not np.array_equal(with_body, image)
    assert not np.array_equal(with_body, without_body)


# --------------------------------------------------------------------------- #
# Tightening the ROI onto the metal actually inside it
# --------------------------------------------------------------------------- #


def _bare_joint(bbox: BoundingBox, kind: str = "joint", position: str = "terminal_a"):
    from aoi_pipeline.models import SolderJoint

    return SolderJoint(
        detection_id="det_1",
        joint_id="det_1_joint00",
        label="resistor",
        kind=kind,
        bbox=bbox,
        terminal_geometry="two_terminal",
        position=position,
    )


def test_refinement_shrinks_a_roi_onto_its_land() -> None:
    """Ratios place the ROI; only the pixels know how wide the land is."""

    from aoi_pipeline.solder import refine_joint_to_metal

    image = _blank_board()
    cv2.rectangle(image, (40, 40), (60, 60), (215, 215, 215), -1)
    joint = _bare_joint(BoundingBox(20, 20, 80, 80))
    refined = refine_joint_to_metal(joint, image, SolderJointConfig())

    assert refined.metadata.get("refined_to_metal") is True
    assert refined.bbox.area < joint.bbox.area
    # The refined box should sit on the metal, not on the board around it.
    assert refined.bbox.x1 >= 35 and refined.bbox.x2 <= 65


def test_refinement_leaves_an_empty_roi_alone() -> None:
    """A land with no solder must stay a big empty ROI -- that emptiness is the
    evidence step 6.2 needs, and collapsing it would hide the defect."""

    from aoi_pipeline.solder import refine_joint_to_metal

    joint = _bare_joint(BoundingBox(20, 20, 80, 80))
    refined = refine_joint_to_metal(joint, _blank_board(), SolderJointConfig())
    assert refined.bbox.as_xyxy() == joint.bbox.as_xyxy()
    assert "refined_to_metal" not in refined.metadata


def test_refinement_ignores_a_speck() -> None:
    from aoi_pipeline.solder import refine_joint_to_metal

    image = _blank_board()
    image[50, 50] = (240, 240, 240)
    joint = _bare_joint(BoundingBox(20, 20, 80, 80))
    refined = refine_joint_to_metal(joint, image, SolderJointConfig())
    assert refined.bbox.as_xyxy() == joint.bbox.as_xyxy()


def test_refinement_never_touches_the_body_view() -> None:
    from aoi_pipeline.solder import refine_joint_to_metal

    image = _blank_board()
    cv2.rectangle(image, (40, 40), (60, 60), (215, 215, 215), -1)
    body = _bare_joint(BoundingBox(20, 20, 80, 80), kind="body", position="body")
    refined = refine_joint_to_metal(body, image, SolderJointConfig())
    assert refined.bbox.as_xyxy() == body.bbox.as_xyxy()


def test_refinement_can_be_switched_off() -> None:
    image, detection = _chip_board()
    on = SolderJointCropper(SolderJointConfig(refine_to_metal=True)).derive(image, [detection])
    off = SolderJointCropper(SolderJointConfig(refine_to_metal=False)).derive(image, [detection])
    on_area = sum(j.bbox.area for j in on if j.kind == "joint")
    off_area = sum(j.bbox.area for j in off if j.kind == "joint")
    assert on_area < off_area
    assert all("refined_to_metal" not in j.metadata for j in off)
