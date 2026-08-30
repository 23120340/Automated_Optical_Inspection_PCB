"""Step 5.5 tests: derived solder-joint ROIs, crops and their export."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import replace
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
from aoi_pipeline.detection.cropping import ComponentCropper
from aoi_pipeline.reporting.exporters import solder_joints_csv
from aoi_pipeline.reporting.overlays import render_solder_overlay
from aoi_pipeline.solder.geometry import deconflict_joint_rois, estimate_component_angle

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
    # Band mode on purpose: what is under test is which EDGES survive, and
    # per-pin splitting (on by default since it was measured) renames each
    # position to ``lead_left_pinNN``, which would hide that question.
    image, detection = _soic_board()
    joints = _joints(detection, image, SolderJointConfig(split_pins=False))
    positions = {joint.position for joint in _by_kind(joints, "joint")}
    assert positions == {"lead_left", "lead_right"}


def test_all_edges_are_kept_without_an_image() -> None:
    _, detection = _soic_board()
    positions = {joint.position for joint in _by_kind(_joints(detection), "joint")}
    assert positions == {"lead_left", "lead_right", "lead_top", "lead_bottom"}


def test_energy_filter_can_be_disabled() -> None:
    image, detection = _soic_board()
    joints = _joints(
        detection,
        image,
        SolderJointConfig(lead_band_energy_ratio=None, split_pins=False),
    )
    assert len(_by_kind(joints, "joint")) == 4


def _band_split_config(**kwargs) -> SolderJointConfig:
    """Cấu hình tách chân trên **dải đo**, tức chế độ trước khi có dải định vị.

    Bản ship tách riêng một dải nông để định vị (xem
    ``lead_locator_outer_ratio``). Các test dưới đây kiểm chính bộ **tách chân**,
    nên chúng ghim mode cũ để câu hỏi được kiểm không bị đổi.
    """

    return SolderJointConfig(
        split_pins=True,
        lead_locator_inner_ratio=None,
        lead_locator_outer_ratio=None,
        **kwargs,
    )


def test_split_pins_finds_one_roi_per_lead() -> None:
    image, detection = _soic_board(pins=6)
    joints = _by_kind(_joints(detection, image, _band_split_config()), "joint")
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
    from aoi_pipeline.detection.detectors import MockComponentDetector

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


def test_ui_roi_table_shows_what_step_6_2_called_each_roi() -> None:
    """The table hard-coded ``defect_class`` to empty because it doubled as the
    labelling sheet, so the panel looked like step 6.2 had produced nothing
    while every row already had a verdict."""

    from app.pipeline_bridge import SolderVerdictRecord
    from app.streamlit_app import _solder_frame

    records = _ui_records()
    verdicts = [
        SolderVerdictRecord(
            joint_id=records[0].joint_id,
            detection_id=records[0].detection_id,
            scope="joint",
            label="insufficient",
            decision="review",
            source="conflict",
            probability=0.62,
            rule_label="insufficient",
            model_label="good",
            model_probability=0.71,
            designator=None,
            pin=None,
            component_label="resistor",
            bbox=(0, 0, 16, 16),
            reasons=["solder_ratio 0.11 < 0.18"],
        )
    ]

    frame = _solder_frame(records, verdicts)
    assert len(frame) == 2
    assert list(frame["defect_class"]) == ["insufficient", ""]
    assert list(frame["decision"]) == ["review", ""]
    assert list(frame["rule_label"]) == ["insufficient", ""]
    assert list(frame["model_label"]) == ["good", ""]
    assert "solder_ratio" in frame["reasons"][0]
    assert list(frame["roi_width_px"]) == [16, 22]


def test_the_manual_label_column_stays_empty_for_the_person_to_fill() -> None:
    """Filling the machine's call into the human's column would bias whoever
    labels the export, so the two are separate columns."""

    from app.streamlit_app import _solder_frame

    frame = _solder_frame(_ui_records())
    assert list(frame["label_manual"]) == ["", ""]
    assert list(frame["defect_class"]) == ["", ""], "chưa chấm thì không bịa nhãn"


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

    from aoi_pipeline.solder.geometry import refine_joint_to_metal

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

    from aoi_pipeline.solder.geometry import refine_joint_to_metal

    joint = _bare_joint(BoundingBox(20, 20, 80, 80))
    refined = refine_joint_to_metal(joint, _blank_board(), SolderJointConfig())
    assert refined.bbox.as_xyxy() == joint.bbox.as_xyxy()
    assert "refined_to_metal" not in refined.metadata


def test_refinement_ignores_a_speck() -> None:
    from aoi_pipeline.solder.geometry import refine_joint_to_metal

    image = _blank_board()
    image[50, 50] = (240, 240, 240)
    joint = _bare_joint(BoundingBox(20, 20, 80, 80))
    refined = refine_joint_to_metal(joint, image, SolderJointConfig())
    assert refined.bbox.as_xyxy() == joint.bbox.as_xyxy()


def test_refinement_never_touches_the_body_view() -> None:
    from aoi_pipeline.solder.geometry import refine_joint_to_metal

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


# --------------------------------------------------------------------------- #
# Which axis carries the terminals
# --------------------------------------------------------------------------- #


def _square_part_board(pad_axis: str, width: int, height: int):
    """A square-ish two-terminal part whose real lands sit on ONE axis.

    An SMD electrolytic can, a tantalum, a tactile switch: the box is close to
    square, so the box itself cannot say where the leads are.
    """

    image = _blank_board()
    cx, cy = 200, 200
    x1, y1 = cx - width // 2, cy - height // 2
    x2, y2 = cx + width // 2, cy + height // 2
    if pad_axis == "x":
        cv2.rectangle(image, (x1 - 16, y1 + 6), (x1 - 1, y2 - 6), (205, 205, 205), -1)
        cv2.rectangle(image, (x2 + 1, y1 + 6), (x2 + 16, y2 - 6), (205, 205, 205), -1)
    else:
        cv2.rectangle(image, (x1 + 6, y1 - 16), (x2 - 6, y1 - 1), (205, 205, 205), -1)
        cv2.rectangle(image, (x1 + 6, y2 + 1), (x2 - 6, y2 + 16), (205, 205, 205), -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), (28, 28, 32), -1)
    cv2.circle(image, (cx, cy), width // 3, (150, 150, 155), 2)
    return image, Detection("capacitor", 0.9, BoundingBox(x1, y1, x2, y2))


def _terminal_axis(joints) -> str:
    first, second = joints[0].bbox, joints[1].bbox
    horizontal = abs(first.x1 - second.x1) > abs(first.y1 - second.y1)
    return "x" if horizontal else "y"


@pytest.mark.parametrize("pad_axis", ["x", "y"])
@pytest.mark.parametrize("width,height", [(44, 44), (44, 43), (43, 44)])
def test_a_square_part_takes_its_terminal_axis_from_the_metal(
    pad_axis: str, width: int, height: int
) -> None:
    """The box cannot name the axis; the lands can.

    ``max(width, height)`` decided this before, so 44x43 and 43x44 -- one pixel
    apart -- placed the two ROIs 90 degrees from each other. Half the time that
    put both ROIs on bare laminate, which measures as no solder at all.
    """

    image, detection = _square_part_board(pad_axis, width, height)
    joints = _by_kind(_joints(detection, image), "joint")
    assert len(joints) == 2, "trục đã quyết được thì chỉ nên có 2 ROI"
    assert _terminal_axis(joints) == pad_axis


def test_an_undecidable_square_part_keeps_both_axes() -> None:
    """With no image there is no evidence, and guessing is worse than asking.

    Four reviewable ROIs cost an operator seconds; two ROIs on the wrong two
    sides pass every defect on the real joints.
    """

    detection = Detection("capacitor", 0.9, BoundingBox(100, 100, 144, 144))
    joints = _by_kind(_joints(detection), "joint")
    assert len(joints) == 4
    positions = {joint.position for joint in joints}
    assert positions == {
        "terminal_a", "terminal_b", "terminal_a_cross", "terminal_b_cross",
    }


def test_a_clearly_elongated_part_still_uses_its_long_axis() -> None:
    """The measurement only runs where the box is ambiguous; an ordinary chip
    must not pay for it."""

    detection = Detection("resistor", 0.9, BoundingBox(100, 100, 180, 134))
    joints = _by_kind(_joints(detection), "joint")
    assert len(joints) == 2
    assert _terminal_axis(joints) == "x"


# --------------------------------------------------------------------------- #
# Keeping one component's ROIs off its neighbours
# --------------------------------------------------------------------------- #


def _resistor_array(rows: int = 4, columns: int = 2, gap: int = 10):
    """A tightly packed chip array, the layout that breaks isolated derivation."""

    width, height = 78, 34
    detections = []
    for row in range(rows):
        for column in range(columns):
            x = 40 + column * (width + gap)
            y = 40 + row * (height + gap)
            detections.append(
                Detection(
                    "resistor",
                    0.9,
                    BoundingBox(x, y, x + width, y + height),
                    detection_id=f"r{row}{column}",
                )
            )
    return detections


def _cross_component_overlaps(joints):
    """(pairs, worst) over ROIs belonging to *different* components."""

    rois = [joint for joint in joints if joint.kind == "joint"]
    pairs, worst = 0, 0.0
    for index, first in enumerate(rois):
        for second in rois[index + 1:]:
            if first.detection_id == second.detection_id:
                continue
            left = max(first.bbox.x1, second.bbox.x1)
            top = max(first.bbox.y1, second.bbox.y1)
            right = min(first.bbox.x2, second.bbox.x2)
            bottom = min(first.bbox.y2, second.bbox.y2)
            area = max(0.0, right - left) * max(0.0, bottom - top)
            if area <= 0:
                continue
            smaller = min(
                first.bbox.width * first.bbox.height,
                second.bbox.width * second.bbox.height,
            )
            ratio = area / smaller
            if ratio > 0.05:
                pairs += 1
                worst = max(worst, ratio)
    return pairs, worst


def test_neighbouring_components_do_not_share_solder_rois() -> None:
    """Measured before the fix: 22 overlapping pairs, the worst at 97%.

    Two ROIs that are 97% the same box are not two measurements. Step 6.2
    grades the same pixels twice, ``refine_to_metal`` snaps both onto the same
    blob, and a bridge between the two parts is inside both ROIs and therefore
    attributable to neither.
    """

    detections = _resistor_array()
    config = SolderJointConfig(include_body_view=False, refine_to_metal=False)
    derived = []
    for detection in detections:
        derived.extend(derive_solder_joints(detection, 700, 400, config))

    before_pairs, before_worst = _cross_component_overlaps(
        deconflict_joint_rois(
            derived, detections, replace(config, deconflict_neighbours=False)
        )
    )
    assert before_pairs > 0 and before_worst > 0.9, "layout này phải tái hiện được lỗi"

    after_pairs, _ = _cross_component_overlaps(
        deconflict_joint_rois(derived, detections, config)
    )
    assert after_pairs == 0


def test_de_confliction_keeps_the_rois_big_enough_to_inspect() -> None:
    """Cutting must not turn the ROIs into slivers: a joint starved of solder
    has to stay a big empty ROI, because the emptiness is the evidence."""

    detections = _resistor_array()
    config = SolderJointConfig(include_body_view=False, refine_to_metal=False)
    derived = []
    for detection in detections:
        derived.extend(derive_solder_joints(detection, 700, 400, config))
    resolved = deconflict_joint_rois(derived, detections, config)

    for before, after in zip(derived, resolved):
        if before.kind != "joint":
            continue
        kept = (after.bbox.width * after.bbox.height) / (
            before.bbox.width * before.bbox.height
        )
        assert kept >= config.deconflict_min_area_fraction
        assert after.bbox.width >= config.min_roi_pixels
        assert after.bbox.height >= config.min_roi_pixels


def test_a_roi_that_cannot_be_freed_is_marked_not_deleted() -> None:
    """An ROI swallowed whole by a neighbour means the detector merged or
    duplicated a box. Dropping the ROI would hide that; keep it and say so."""

    covered = Detection("resistor", 0.9, BoundingBox(100, 100, 120, 116), detection_id="a")
    swallower = Detection("ic", 0.9, BoundingBox(40, 40, 260, 200), detection_id="b")
    config = SolderJointConfig(include_body_view=False, refine_to_metal=False)
    derived = derive_solder_joints(covered, 400, 400, config)

    resolved = deconflict_joint_rois(derived, [covered, swallower], config)
    assert len(resolved) == len(derived)
    assert all(joint.metadata.get("overlap_unresolved") for joint in resolved)


def test_de_confliction_leaves_a_sparse_board_untouched() -> None:
    """Nothing to fix must mean nothing changed; the fix is not allowed to move
    ROIs on the boards that were already correct."""

    detections = _resistor_array(rows=2, columns=1, gap=300)
    config = SolderJointConfig(include_body_view=False, refine_to_metal=False)
    derived = []
    for detection in detections:
        derived.extend(derive_solder_joints(detection, 900, 900, config))
    resolved = deconflict_joint_rois(derived, detections, config)
    assert [joint.bbox.to_dict() for joint in resolved] == [
        joint.bbox.to_dict() for joint in derived
    ]


def test_pins_of_the_same_component_are_allowed_to_stay_adjacent() -> None:
    """Two pins of one IC are neighbours by design; only ROIs from *different*
    components compete for ground."""

    image, detection = _soic_board(pins=6)
    config = SolderJointConfig(
        include_body_view=False, refine_to_metal=False, split_pins=True
    )
    derived = derive_solder_joints(detection, 700, 400, config, image=image)
    resolved = deconflict_joint_rois(derived, [detection], config)
    assert [joint.bbox.to_dict() for joint in resolved] == [
        joint.bbox.to_dict() for joint in derived
    ]


# --------------------------------------------------------------------------- #
# Which perimeter bands actually hold leads
# --------------------------------------------------------------------------- #


def _speckled_board() -> np.ndarray:
    """Green laminate with the gloss and speckle a real solder mask has.

    Not decoration. ``segment_solder`` falls back to brightness alone when no
    pixel passes its saturation test, and on a perfectly flat synthetic green
    that fallback marks the bare board as metal -- which would make this test
    pass or fail for a reason that has nothing to do with lead detection.
    """

    rng = np.random.default_rng(7)
    image = np.full((*BOARD_SIZE, 3), (40, 90, 40), np.uint8)
    noise = rng.normal(0, 6, image.shape)
    image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
    for _ in range(400):
        x = int(rng.integers(0, BOARD_SIZE[1]))
        y = int(rng.integers(0, BOARD_SIZE[0]))
        cv2.circle(image, (x, y), 1, (150, 155, 150), -1)
    return image


def _sot23_board():
    """Three leads: two on the top edge, one centred on the bottom edge.

    The package that broke this on the real board (D201/D202). The left and
    right edges carry no lead at all, but they sit beside the corner pads, so a
    band that runs past the corners borrows their metal.
    """

    image = _speckled_board()
    body = BoundingBox(150, 150, 214, 197)
    cv2.rectangle(image, (158, 136), (178, 152), (205, 205, 205), -1)
    cv2.rectangle(image, (186, 136), (206, 152), (205, 205, 205), -1)
    cv2.rectangle(image, (172, 195), (192, 211), (205, 205, 205), -1)
    cv2.rectangle(image, (150, 150), (214, 197), (26, 26, 30), -1)
    return image, Detection("ic", 0.9, body)


def test_only_the_edges_that_carry_leads_become_rois() -> None:
    """Measured on the real board before the fix: all four bands survived on
    both SOT-23 parts, so the two lead-free sides were inspected as if they
    held joints -- two wrong ROIs per part, which is exactly what the operator
    reported seeing."""

    image, detection = _sot23_board()
    joints = _by_kind(_joints(detection, image), "joint")
    # Compare by band, not by full position: the shipped config splits each
    # band into per-pin ROIs (``lead_top_pin00``). The claim being made is
    # about which SIDES get inspected, and it has to hold for the config that
    # actually ships, so this test deliberately does not pin the mode.
    bands = {joint.position.split("_pin")[0] for joint in joints}
    assert bands == {"lead_top", "lead_bottom"}


def test_the_lead_free_sides_are_not_saved_by_the_corner_pads() -> None:
    """Bands deliberately run past the corners so a corner pin is not clipped,
    which makes every band overlap its neighbours. Measured whole, a lead-free
    side borrows the corner pads and looks nearly as leaded as the real ones
    (0.47 against 1.00 on the real board). Measured on its corner-free core it
    does not (0.26)."""

    from aoi_pipeline.grading.features import segment_solder
    from aoi_pipeline.solder.geometry import (
        _band_core_rect,
        _component_frame,
        _local_rect_to_bbox,
        _multi_pin_rects,
    )

    image, detection = _sot23_board()
    config = SolderJointConfig()
    frame = _component_frame(detection.bbox, config, image)

    def coverage(rect) -> float:
        bbox = _local_rect_to_bbox(rect, frame, BOARD_SIZE[1], BOARD_SIZE[0])
        x1, y1, x2, y2 = bbox.to_int()
        mask = segment_solder(image[y1:y2, x1:x2], saturation_max=config.saturation_max)
        return float(np.count_nonzero(mask)) / max(1, mask.size)

    rects = _multi_pin_rects(frame, config)
    whole = {rect.position: coverage(rect) for rect in rects}
    core = {rect.position: coverage(_band_core_rect(rect, frame)) for rect in rects}

    def worst_lead_free(values: dict[str, float]) -> float:
        peak = max(values.values()) or 1e-9
        return max(values["lead_left"], values["lead_right"]) / peak

    assert worst_lead_free(whole) > worst_lead_free(core), (
        "cắt góc phải làm cạnh không chân yếu đi tương đối"
    )
    assert worst_lead_free(core) < 0.35


def test_a_band_is_never_filtered_away_to_nothing() -> None:
    """When no band has any metal there is nothing to rank, so all of them are
    kept. Losing every band would drop the part out of inspection without ever
    saying so, which is the one outcome this filter must never produce."""

    image = _blank_board()
    detection = Detection("ic", 0.9, BoundingBox(150, 150, 214, 197))
    assert len(_by_kind(_joints(detection, image), "joint")) == 4


def test_a_lead_free_component_still_keeps_at_least_one_roi() -> None:
    """Even where the ranking does separate bands, the filter has to leave the
    part inspectable."""

    image = _speckled_board()
    cv2.rectangle(image, (150, 150), (214, 197), (26, 26, 30), -1)
    detection = Detection("ic", 0.9, BoundingBox(150, 150, 214, 197))
    assert _by_kind(_joints(detection, image), "joint")


def test_a_two_lead_edge_can_be_split_per_pin() -> None:
    """``min_pins_per_band`` was 3, so the two-lead edge of a SOT-23, SOT-223 or
    DPAK could never be split and its two joints stayed inside one ROI."""

    image, detection = _sot23_board()
    config = SolderJointConfig(include_body_view=False, split_pins=True)
    joints = derive_solder_joints(detection, BOARD_SIZE[1], BOARD_SIZE[0], config, image=image)
    top = [joint for joint in joints if joint.position.startswith("lead_top")]
    assert len(top) >= 2, [joint.position for joint in joints]


# --------------------------------------------------------------------------- #
# Giai đoạn A: ROI mang toạ độ, không mang ảnh
# --------------------------------------------------------------------------- #


def _two_part_board():
    """Two chips far enough apart that nothing here depends on de-confliction."""

    image = np.full((*BOARD_SIZE, 3), (40, 90, 40), np.uint8)
    detections = []
    for index, x in enumerate((60, 260)):
        cv2.rectangle(image, (x - 26, 96), (x - 6, 124), (200, 200, 200), -1)
        cv2.rectangle(image, (x + 66, 96), (x + 86, 124), (200, 200, 200), -1)
        cv2.rectangle(image, (x, 92), (x + 70, 128), (25, 25, 25), -1)
        detections.append(
            Detection("resistor", 0.9, BoundingBox(x, 92, x + 70, 128),
                      detection_id=f"d{index}")
        )
    return image, detections


def _bridge_solder(image, detections, **kwargs):
    from types import SimpleNamespace

    from app.pipeline_bridge import PipelineBridge

    bridge = PipelineBridge(config={"solder": {"enabled": True}})
    wrapped = [SimpleNamespace(raw=item) for item in detections]
    return bridge.make_solder_crops(image, wrapped, **kwargs)


def _held_bytes(result) -> int:
    total = 0
    for crop in result.crops:
        if crop.image is not None:
            total += crop.image.nbytes
        raw_image = getattr(crop.raw, "image", None)
        if raw_image is not None:
            total += raw_image.nbytes
    return total


def test_solder_records_carry_coordinates_not_pixels() -> None:
    """Measured on a real board: 119 ROIs held 11.70 MB of pixels against
    0.028 MB of coordinates -- 414x, and the ROIs cost 3.7x the source image
    they were cut from. The pixels were held twice over, once in ``image`` and
    once again in ``raw``."""

    image, detections = _two_part_board()
    result = _bridge_solder(image, detections)

    assert result.crops, "vẫn phải sinh ra ROI"
    assert _held_bytes(result) == 0
    assert all(crop.image is None for crop in result.crops)
    assert all(crop.bbox is not None for crop in result.crops)


def test_dropping_the_pixels_does_not_change_a_single_verdict() -> None:
    """Grading happens before the records are built, on the core's own crops.
    If holding on to them changed any call, the saving would not be free."""

    image, detections = _two_part_board()
    lean = _bridge_solder(image, detections)
    fat = _bridge_solder(image, detections, keep_images=True)

    assert _held_bytes(fat) > 0, "nhánh giữ ảnh phải thật sự giữ ảnh"
    assert [(v.joint_id, v.label, v.decision) for v in lean.verdicts] == [
        (v.joint_id, v.label, v.decision) for v in fat.verdicts
    ]
    assert [c.bbox for c in lean.crops] == [c.bbox for c in fat.crops]


def test_the_ui_cuts_the_roi_back_out_of_the_analysis_frame() -> None:
    """What the gallery shows must be the same pixels the ROI covers, or the
    saving has quietly turned into a display bug."""

    from app.streamlit_app import _roi_pixels_for_display

    image, detections = _two_part_board()
    result = _bridge_solder(image, detections)
    crop = next(item for item in result.crops if item.kind == "joint")

    shown = _roi_pixels_for_display(image, crop)
    x1, y1, x2, y2 = crop.bbox
    assert shown is not None
    assert np.array_equal(shown, image[y1:y2, x1:x2])


def test_display_falls_back_to_a_carried_image_when_there_is_one() -> None:
    """``keep_images=True`` stays usable, and a record that has its pixels must
    not be re-cut from a frame that may no longer match it."""

    from app.streamlit_app import _roi_pixels_for_display

    image, detections = _two_part_board()
    result = _bridge_solder(image, detections, keep_images=True)
    crop = next(item for item in result.crops if item.kind == "joint")

    assert np.array_equal(_roi_pixels_for_display(image, crop), crop.image)
    assert _roi_pixels_for_display(None, crop) is not None


def test_display_survives_having_no_frame_and_no_pixels() -> None:
    """A stale record after the source image was cleared must not crash a tab."""

    from app.streamlit_app import _roi_pixels_for_display

    image, detections = _two_part_board()
    crop = _bridge_solder(image, detections).crops[0]
    assert _roi_pixels_for_display(None, crop) is None


# --------------------------------------------------------------------------- #
# refine_to_metal phải quan sát được từ app
# --------------------------------------------------------------------------- #


def test_refine_records_both_ends_of_its_own_step() -> None:
    """Measuring refine against the *final* box measures two stages at once --
    de-confliction runs again after fusion. And measuring against the fractional
    pre-refine box measures a rounding artefact: ``to_int`` widens a box outward,
    so a ROI refine kept whole looked like it had grown by a pixel a side, which
    on a 26x29 ROI reads as -11% shrink."""

    image, detection = _chip_board()
    config = SolderJointConfig(include_body_view=False, refine_to_metal=True)
    joints = SolderJointCropper(config).derive(image, [detection])

    refined = [j for j in joints if j.metadata.get("refined_to_metal")]
    assert refined, "board này phải có ROI được siết"
    for joint in refined:
        before = joint.metadata["roi_before_refine"]["xyxy"]
        after = joint.metadata["roi_after_refine"]["xyxy"]
        before_area = (before[2] - before[0]) * (before[3] - before[1])
        after_area = (after[2] - after[0]) * (after[3] - after[1])
        assert after_area <= before_area, (
            f"siết mà lại to ra: {before} -> {after}"
        )
        # Both ends are the integer boxes the step really worked on.
        assert all(float(v).is_integer() for v in (*before, *after))


def test_the_app_can_see_whether_refine_fired_and_by_how_much() -> None:
    """From the app the stage used to be invisible: on by default, no toggle,
    nothing in the table. You could not tell whether it had done anything."""

    from app.streamlit_app import _shrink_percent, _solder_frame

    image, detections = _two_part_board()
    on = _bridge_solder(image, detections)
    frame = _solder_frame(on.crops)

    assert "refined" in frame.columns and "shrink_pct" in frame.columns
    joints = [c for c in on.crops if c.kind == "joint"]
    assert any(c.refined for c in joints), "phải có ROI được siết"

    for crop in joints:
        percent = _shrink_percent(crop)
        if crop.refined and crop.roi_after_refine is not None:
            assert percent is not None and percent >= 0.0, (
                f"{crop.joint_id}: shrink {percent}"
            )
        else:
            assert percent is None, "không siết thì phải là None, không phải 0%"


def test_turning_refine_off_leaves_every_roi_alone() -> None:
    """The toggle has to actually reach the core, or comparing on against off
    in the app compares nothing."""

    from types import SimpleNamespace

    from app.pipeline_bridge import PipelineBridge

    image, detections = _two_part_board()
    wrapped = [SimpleNamespace(raw=item) for item in detections]

    off = PipelineBridge(
        config={"solder": {"enabled": True, "refine_to_metal": False}}
    ).make_solder_crops(image, wrapped)
    on = PipelineBridge(
        config={"solder": {"enabled": True, "refine_to_metal": True}}
    ).make_solder_crops(image, wrapped)

    assert not any(c.refined for c in off.crops)
    assert any(c.refined for c in on.crops)
    assert [c.bbox for c in off.crops] != [c.bbox for c in on.crops]


def test_split_pins_still_covers_the_gap_a_bridge_lives_in() -> None:
    """Vì sao ``split_pins`` được bật mặc định.

    Lý do giữ nó tắt trước đây: cầu chì nối hai chân kề nhau, nên cả một dải
    được cho là đơn vị kiểm tốt hơn một chân lẻ. Đo trên ``pcb03.jpg`` (4 con
    IC, 50 chân) thì lập luận đó không đứng được: ``pin_padding_ratio`` đã nới
    mỗi ROI chân sang tới khe với chân bên cạnh, nên **cả 42 khe vẫn được phủ**
    ở cả hai chế độ — trong khi số ROI chứa đúng một chân đi từ 3 lên 48.

    Test này giữ lại cả hai nửa của phép đo đó, vì bỏ nửa sau đi thì việc bật
    mặc định chỉ còn là một lựa chọn không có bằng chứng.
    """

    image, detection = _soic_board(pins=6)
    joints = _by_kind(_joints(detection, image, _band_split_config()), "joint")

    assert all(joint.pin_index is not None for joint in joints), (
        "mặc định phải tách theo chân"
    )

    left = sorted(
        (joint for joint in joints if joint.position.startswith("lead_left")),
        key=lambda joint: joint.bbox.y1,
    )
    assert len(left) == 6

    # Khe giữa hai chân kề nhau phải nằm trong ít nhất một ROI.
    for first, second in zip(left, left[1:]):
        gap_y = (first.bbox.y2 + second.bbox.y1) / 2.0
        covered = [
            joint for joint in left
            if joint.bbox.y1 <= gap_y <= joint.bbox.y2
        ]
        assert covered, f"khe ở y={gap_y:.1f} không ROI nào phủ — cầu chì sẽ lọt"


# --------------------------------------------------------------------------
# P1 — trần độ sâu ROI theo mm
# --------------------------------------------------------------------------


def _big_can() -> tuple[np.ndarray, Detection]:
    """Tụ hoá can tròn: hộp gần vuông, pad ở trên và dưới.

    Dựng theo đúng con C239 đo được trên board của dự án — hộp 148x136 px ở
    22 px/mm, tức 6,7 x 6,2 mm.
    """

    image = _blank_board()
    cv2.circle(image, (300, 300), 62, (30, 30, 30), -1)
    cv2.rectangle(image, (285, 222), (315, 240), (215, 215, 215), -1)
    cv2.rectangle(image, (285, 360), (315, 378), (215, 215, 215), -1)
    return image, Detection("capacitor", 0.9, BoundingBox(226, 232, 374, 368))


def test_a_big_part_gets_a_roi_sized_like_a_pad_not_like_the_part() -> None:
    """Luật cũ `0,75 x chiều dài` giả định pad to lên theo linh kiện.

    Đo trên board của dự án: tụ hoá C239 dài 6,7 mm cho ra ROI sâu **5,0 mm** —
    to hơn cả con tụ, và không ROI nào rơi trúng pad. Pad là kích thước vật lý
    gần cố định, không co giãn theo thân.
    """

    image, detection = _big_can()
    loose = _joints(detection, image, SolderJointConfig(px_per_mm=None))
    capped = _joints(
        detection, image, SolderJointConfig(px_per_mm=22.0)
    )

    def deepest(joints):
        return max(
            min(joint.bbox.width, joint.bbox.height)
            for joint in _by_kind(joints, "joint")
        )

    # Hai lớp chặn độc lập, và test này đo lớp thứ hai.
    #
    # Lớp thứ nhất là bộ tỉ lệ `compact_*`: hộp gần vuông thì mọi kích thước
    # tính theo cạnh NGẮN, không theo cạnh dài. Riêng nó đã kéo độ sâu ở đây
    # từ 102 px xuống 47,6 px, nên "cái vô lý" mà test này từng đo (>61,6 px)
    # không còn nữa — không phải vì trần mm hết tác dụng, mà vì nó không còn
    # phải một mình gánh.
    #
    # Lớp thứ hai là trần mm, và nó vẫn cắn: 47,6 px vẫn quá 2,0 mm ở 22 px/mm.
    # Đó mới là thứ cần khẳng định — một trần chỉ có ý nghĩa khi còn gì để cắt.
    limit_px = 2.0 * 22.0
    assert deepest(loose) > limit_px, "trần mm phải còn việc để làm thì mới đo được nó"
    assert deepest(capped) <= limit_px + 2


def test_the_cap_leaves_small_chip_parts_untouched() -> None:
    """Tính chất cần có ở bản sửa này: nó chỉ cắn vào chỗ luật cũ vô lý.

    Đo trên ảnh của dự án ở trần 2,0 mm: 6/39 linh kiện bị cắt, 33 con còn lại
    không đổi một pixel. Ở 1,5 mm là 7/39 và trần bắt đầu cắn cả chip 0805, nên
    2,0 mới là chỗ đúng.
    """

    image = _blank_board()
    cv2.rectangle(image, (300, 300), (346, 331), (40, 40, 40), -1)
    chip = Detection("resistor", 0.9, BoundingBox(300, 300, 346, 331))

    loose = _joints(chip, image, SolderJointConfig(px_per_mm=None))
    capped = _joints(
        chip, image, SolderJointConfig(px_per_mm=22.0)
    )
    assert [j.bbox.to_int() for j in loose] == [j.bbox.to_int() for j in capped]


def test_without_a_scale_nothing_is_capped() -> None:
    """Không biết px/mm thì không quy đổi được sang mm. Đoán bừa một thang còn
    tệ hơn giữ nguyên luật tỉ lệ, nên trần chỉ áp khi thang có thật."""

    image, detection = _big_can()
    a = _joints(detection, image, SolderJointConfig(px_per_mm=None))
    b = _joints(detection, image, SolderJointConfig(px_per_mm=0.0))
    assert [j.bbox.to_int() for j in a] == [j.bbox.to_int() for j in b]


def test_the_cap_keeps_the_inner_outer_balance() -> None:
    """Thu cả hai đầu theo tỉ lệ, không cắt cụt một đầu: `inner` với ngược lên
    nắp đầu cực trên thân, `outer` với ra ngoài chỗ land có fillet. Cắt riêng
    một đầu là đổi ý nghĩa của ROI chứ không phải thu nhỏ nó."""

    image, detection = _big_can()
    capped = _by_kind(
        _joints(detection, image, SolderJointConfig(px_per_mm=22.0)), "joint"
    )
    centre_y = (detection.bbox.y1 + detection.bbox.y2) / 2.0
    above = [j for j in capped if (j.bbox.y1 + j.bbox.y2) / 2.0 < centre_y]
    below = [j for j in capped if (j.bbox.y1 + j.bbox.y2) / 2.0 > centre_y]
    if above and below:
        assert abs(above[0].bbox.height - below[0].bbox.height) <= 2


# --------------------------------------------------------------------------
# P2 — dải chân là một cái lược, chữ lụa thì không
# --------------------------------------------------------------------------


def _soic_with_silkscreen() -> tuple[np.ndarray, Detection]:
    """SOIC có chân hai cạnh, cộng một dòng chữ lụa sát cạnh thứ ba.

    Đây là đúng ca đo được trên board của dự án: chữ `HDL01` cạnh con SOIC-8
    ghi **49,5 %** "kim loại" theo phép đo hiện tại, cao hơn cả chân IC thật
    (38,3 % và 29,9 %) — nên bộ lọc năng lượng không thể loại nó.
    """

    image, detection = _soic_board(pins=6)
    cv2.putText(image, "HDL01", (566, 186), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (225, 225, 225), 2, cv2.LINE_AA)
    return image, detection


def test_a_row_of_leads_reads_as_a_comb_and_silkscreen_does_not() -> None:
    """Chỉ số này đo *khoảng cách giữa các đốm có đều không*, không đo sáng tối.

    Trên 9 dải của board dự án nó phát biểu về 5 dải và đúng cả 5: giữ SOIC-8
    trên/dưới (0,99 và 0,98) cùng D201 trên (0,97), bỏ chữ `HDL01` (0,87) và
    viền lụa quanh D201 (0,73).
    """

    from aoi_pipeline.solder.geometry import _band_evenness

    image, _ = _soic_with_silkscreen()
    leads = image[120:196, 424:442]          # cột chân trái: 6 đốm cách đều
    silk = image[170:196, 560:640]           # dòng chữ lụa

    lead_score = _band_evenness(leads, leads.shape[1] >= leads.shape[0], SolderJointConfig())
    silk_score = _band_evenness(silk, silk.shape[1] >= silk.shape[0], SolderJointConfig())

    assert lead_score is not None and lead_score[1] >= 0.95
    if silk_score is not None:
        assert silk_score[1] < lead_score[1]


def test_it_stays_silent_on_an_edge_with_only_two_leads() -> None:
    """Dưới 3 đốm thì "độ đều" luôn bằng 1,0 và vô nghĩa. Cạnh 2 chân là
    chuyện thường (SOT-23, SOT-223), nên im lặng mới đúng — trả về ``None``
    nghĩa là *không biết*, không phải *không có chân*."""

    from aoi_pipeline.solder.geometry import _band_evenness

    image = _blank_board()
    cv2.rectangle(image, (100, 100), (112, 108), (215, 215, 215), -1)
    cv2.rectangle(image, (100, 130), (112, 138), (215, 215, 215), -1)
    band = image[95:145, 98:115]

    assert _band_evenness(band, band.shape[1] >= band.shape[0], SolderJointConfig()) is None


def test_the_profile_follows_the_patch_not_the_local_rect() -> None:
    """Rect nằm trong khung cục bộ của linh kiện, mảng cắt nằm trong khung ảnh,
    và hai khung hoán vị trục khi linh kiện xoay.

    Đo được trước khi sửa: dải `lead_left` của một SOIC-8 cắt ra mảng 112x34
    nằm ngang trong khi rect của nó cao hơn rộng, nên phép chiếu quét ngang qua
    bề dày dải và không đời nào thấy được cái lược. Hệ quả: 7 ROI giả trên chữ
    lụa vẫn sống sót.
    """

    from aoi_pipeline.solder.geometry import _band_evenness

    image = _blank_board()
    for index in range(5):
        x = 200 + index * 20
        cv2.rectangle(image, (x, 300), (x + 10, 316), (215, 215, 215), -1)
    band = image[298:318, 195:310]           # nằm ngang: 115 x 20

    along_patch = _band_evenness(band, band.shape[1] >= band.shape[0], SolderJointConfig())
    across = _band_evenness(band, band.shape[1] < band.shape[0], SolderJointConfig())

    assert along_patch is not None and along_patch[0] == 5
    assert across is None, "chiếu sai trục thì không thấy được cái lược"


# --------------------------------------------------------------------------
# P0 — góc xoay từ file pick-and-place là trục đầu cực ĐÃ BIẾT
# --------------------------------------------------------------------------


def test_a_square_part_emits_both_axes_when_nothing_knows_better() -> None:
    """Hành vi hiện có, giữ lại làm mốc: hộp gần vuông thì không chốt được
    trục, nên phát cả hai cặp và để người xem quyết.

    Trên một ảnh dựng sạch -- chỉ có pad ở trên và dưới, hai cạnh còn lại trống
    -- phép đo kim loại **chốt đúng** và chỉ ra hai ROI. Nó chỉ bó tay khi hai
    cạnh kia cũng có kim loại, và đó chính là board thật: đo trên C232/C231 thì
    hai con này có điện trở chip nằm sát hai bên, nên mối hàn của hàng xóm rơi
    vào đúng vành đang đo và phép chọn trục sai 2 trong 3 lần.
    """

    image, detection = _big_can()
    clean = _by_kind(_joints(detection, image), "joint")
    assert len(clean) == 2, "ảnh sạch thì phải chốt được trục"

    # Thêm mối hàn của hàng xóm ở hai cạnh trái/phải, đúng như board thật.
    crowded = image.copy()
    cv2.rectangle(crowded, (208, 285), (226, 315), (215, 215, 215), -1)
    cv2.rectangle(crowded, (374, 285), (392, 315), (215, 215, 215), -1)
    joints = _by_kind(_joints(detection, crowded), "joint")
    assert any(joint.position.endswith("_cross") for joint in joints), (
        "có kim loại cả bốn phía thì phải phát cả hai cặp chứ không được đoán"
    )


def test_a_known_axis_is_not_second_guessed() -> None:
    """Khi trục đến từ nguồn khác hộp -- góc xoay trong file pick-and-place --
    thì đừng đoán lại.

    Phép đoán dựa vào lượng kim loại ở hai bên, mà trên tụ hoá thì **vỏ can
    chính là kim loại**: đo được thân can 60,3 % "kim loại" so với pad của nó
    64,0 %. Đo thêm cách chỉ nhìn vành ngoài hộp cũng không cứu được: đúng 1
    trong 3 con tụ, vì hai con còn lại có điện trở chip nằm sát hai bên và mối
    hàn của hàng xóm rơi vào đúng vành đang đo.
    """

    from aoi_pipeline.solder.geometry import ComponentFrame, derive_solder_joints

    image, detection = _big_can()
    box = detection.bbox
    frame = ComponentFrame(
        center_x=(box.x1 + box.x2) / 2.0,
        center_y=(box.y1 + box.y2) / 2.0,
        angle=90.0,
        length=box.width,
        span=box.height,
    )
    joints = _by_kind(
        derive_solder_joints(
            detection,
            BOARD_SIZE[1],
            BOARD_SIZE[0],
            SolderJointConfig(),
            image,
            frame=frame,
            geometry="two_terminal",
            axis_known=True,
        ),
        "joint",
    )

    assert len(joints) == 2
    assert not any(joint.position.endswith("_cross") for joint in joints)
    # Góc 90 độ nên hai ROI phải xếp trên–dưới, không phải trái–phải.
    centres = sorted((joint.bbox.y1 + joint.bbox.y2) / 2.0 for joint in joints)
    assert centres[1] - centres[0] > box.height / 2.0


def test_the_pixel_scale_reaches_the_cad_path_too() -> None:
    """``FusionConfig`` mang một ``SolderJointConfig`` riêng.

    ``PipelineConfig.from_mapping`` nối hai cái lại, nhưng ``PipelineConfig()``
    dựng trực tiếp thì không -- và đường CAD đọc đúng cái nằm trong ``fusion``.
    Thiếu chỗ nối này thì trần mm chỉ có tác dụng khi KHÔNG nạp CAD, tức đúng
    lúc cần nó nhất thì lại không có.
    """

    from aoi_pipeline.config import PipelineConfig

    typed = PipelineConfig.from_mapping({"solder": {"max_joint_depth_mm": 2.0}})
    assert typed.fusion.solder is typed.solder

    plain = PipelineConfig()
    plain.solder.px_per_mm = 22.0
    assert plain.fusion.solder.px_per_mm is None, (
        "hai đối tượng vẫn tách rời khi dựng trực tiếp -- đó là lý do "
        "register_cad_to_image phải gán cả hai"
    )


# --------------------------------------------------------------------------
# P3 — dải ĐỊNH VỊ tách khỏi dải ĐO
# --------------------------------------------------------------------------


def _soic_with_standoff(standoff: int, pins: int = 6):
    """SOIC có chân bắt đầu từ mép thân và thò ra ngoài ``standoff`` px."""

    image = np.full((*BOARD_SIZE, 3), (40, 70, 45), np.uint8)
    cv2.rectangle(image, (440, 120), (560, 190), (20, 20, 20), -1)
    for index in range(pins):
        y = 128 + index * 11
        cv2.rectangle(image, (440 - standoff, y), (441, y + 6), (215, 215, 215), -1)
        cv2.rectangle(image, (559, y), (560 + standoff, y + 6), (215, 215, 215), -1)
    return image, Detection("ic", 0.95, BoundingBox(440, 120, 560, 190))


def test_the_locator_strip_is_shallower_than_the_measurement_roi() -> None:
    """Tìm chân và đo chân là hai câu hỏi khác nhau.

    Câu đầu chỉ cần nhìn sát mép thân — với ra xa chỉ rước thêm chữ lụa. Câu
    sau phải với tới pad và fillet, nếu không thì cắt mất đúng thứ cần chấm.
    """

    from aoi_pipeline.solder.geometry import ComponentFrame, _multi_pin_rects

    config = SolderJointConfig()
    frame = ComponentFrame(500.0, 155.0, 0.0, 120.0, 70.0)
    measure = _multi_pin_rects(frame, config)
    locate = _multi_pin_rects(frame, config, locator=True)

    def depth(rect):
        return min(rect.width, rect.height)

    assert depth(locate[0]) < depth(measure[0])


def test_the_locator_survives_leads_that_barely_clear_the_body() -> None:
    """Đây là lý do dải định vị **không** đặt độ sâu ngoài bằng 0.

    Đo trên chân thò ra từ 6 % đến 43 % bề ngang thân: dải định vị hiện tại
    (trong 0,20 / ngoài 0,10) tách được hàng chân ở **mọi** mức, còn chế độ cũ
    — định vị bằng chính dải đo — chỉ ra 2 ROI khi chân thò ít, tức bộ tách
    chân bỏ cuộc và cả một hàng chân bị gộp làm một.

    Trước đây test này đếm 12 ROI. Con số ấy là **5 + 5 chân cộng hai dải
    nguyên nằm trên hai cạnh TRẦN** của SOIC, không phải 12 ROI cho 12 chân:
    fixture có 6 chân mỗi bên và bộ tách chỉ bao giờ tìm được 5 (xem
    ``test_the_split_is_fragile_at_low_contrast_and_that_is_known``). Từ khi
    luật cặp cạnh đối loại hai cạnh trần, phép đếm ấy không còn nói lên điều
    nó định nói, nên ở đây kiểm thẳng thứ cần kiểm: hàng chân vẫn được tách, và
    không ROI nào rơi ra cạnh không có chân.
    """

    for standoff in (4, 10, 22):
        image, detection = _soic_with_standoff(standoff)
        joints = _by_kind(_joints(detection, image, SolderJointConfig()), "joint")
        bands = Counter(joint.position.rsplit("_pin", 1)[0] for joint in joints)
        assert set(bands) == {"lead_left", "lead_right"}, (
            f"thò ra {standoff} px: có ROI trên cạnh không chân — {dict(bands)}"
        )
        assert bands["lead_left"] >= 5 and bands["lead_right"] >= 5, (
            f"thò ra {standoff} px: hàng chân bị gộp — {dict(bands)}"
        )


def test_zeroing_the_locator_reach_loses_leads_on_a_gull_wing_package() -> None:
    """Trực giác "IC không mở box" nếu làm thẳng tay thì hỏng.

    Chân gull-wing của SOIC nằm **ngoài** đường bao thân, nên dải sâu 0 nằm
    trọn trong thân tối và không thấy đủ đốm. Đo trên board của dự án: U201 đi
    từ 8 ROI (đủ 8 chân) xuống còn 4.
    """

    image, detection = _soic_with_standoff(14)
    proper = _by_kind(_joints(detection, image, SolderJointConfig()), "joint")
    flattened = _by_kind(
        _joints(
            detection,
            image,
            SolderJointConfig(lead_locator_inner_ratio=0.14, lead_locator_outer_ratio=0.0),
        ),
        "joint",
    )
    assert len(flattened) < len(proper)


def test_the_split_is_fragile_at_low_contrast_and_that_is_known() -> None:
    """Ghi lại một điểm yếu đã đo, thay vì giấu nó.

    Trên fixture SOIC, kết quả tách chân lật **chỉ vì màu nền**: nền (40,70,45)
    cho 12/12 còn nền (40,90,40) cho 10/12, với cùng độ sáng chân. Dải định vị
    nông có ít tín hiệu hơn nên nhạy hơn với chuyện này.

    Đã thử ba rào chắn để tự phát hiện ca xấu — so số đốm, so độ đều của pitch,
    so tương phản biên dạng — **không cái nào tách được** ca này khỏi ca thật
    trên board (tương phản dải định vị: fixture hỏng 60–67, board thật chạy tốt
    ở 45 và 78). Nên đây là giới hạn đã biết, không phải chuyện đã giải xong.
    """

    def rois(background):
        image = np.full((*BOARD_SIZE, 3), background, np.uint8)
        cv2.rectangle(image, (440, 120), (560, 190), (20, 20, 20), -1)
        for index in range(6):
            y = 128 + index * 11
            cv2.rectangle(image, (424, y), (441, y + 6), (215, 215, 215), -1)
            cv2.rectangle(image, (559, y), (576, y + 6), (215, 215, 215), -1)
        detection = Detection("ic", 0.95, BoundingBox(440, 120, 560, 190))
        return len(_by_kind(_joints(detection, image, SolderJointConfig()), "joint"))

    left, right = rois((40, 70, 45)), rois((40, 90, 40))
    # Cái lật theo màu nền đã đo lại được: nó **không** nằm ở bộ tách chân. Cả
    # hai nền đều luôn cho 5 chân mỗi bên, cả trước lẫn sau. Thứ lật là hai dải
    # nguyên trên hai cạnh TRẦN của SOIC — nền này giữ chúng (5+5+1+1 = 12),
    # nền kia bỏ (5+5 = 10) — nên phép đếm cũ đo độ bám của ROI trên nền trần
    # chứ không đo độ tách chân. Luật cặp cạnh đối bỏ hẳn hai dải đó, nên giờ
    # hai nền cho cùng một kết quả.
    assert left == right == 10
    # Giới hạn thật vẫn còn nguyên và vẫn chưa giải: fixture có **6** chân mỗi
    # bên, bộ tách tìm được 5.
    assert left < 12
