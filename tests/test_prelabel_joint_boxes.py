from __future__ import annotations

from dataclasses import dataclass

from aoi_pipeline.models import BoundingBox, Detection
from scripts.prelabel_joint_boxes import (
    _draft_record,
    crop_order,
    dataset_id,
    geometry_for_row,
    map_detections_to_crops,
)


@dataclass(frozen=True)
class SourceBox:
    cls: str
    cx: float
    cy: float
    w: float
    h: float


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "crop_path": "scene__007__IC.png",
        "component_class": "IC",
        "crop_w": "80",
        "crop_h": "60",
        "body_x": "10",
        "body_y": "10",
    }
    row.update(overrides)
    return row


def test_dataset_id_matches_joint_box_builder_contract() -> None:
    rows = [_row(), _row(crop_path="scene__008__IC.png")]
    assert dataset_id("sample", rows, ["Bad_podu", "Bad_qiaojiao"]) == "9bb591e7f2e081d7"


def test_crop_order_and_geometry_reconstruct_clamped_source_pixels() -> None:
    row = _row()
    box = SourceBox("IC", cx=0.30, cy=0.40, w=0.20, h=0.20)
    geometry = geometry_for_row(row, box, width=200, height=100)

    assert crop_order(row["crop_path"]) == 7
    assert geometry.body.as_xyxy() == [40.0, 30.0, 80.0, 50.0]
    assert geometry.crop.as_xyxy() == [30.0, 20.0, 110.0, 80.0]


def test_scene_detection_is_clipped_and_assigned_to_only_best_crop() -> None:
    first = geometry_for_row(
        _row(crop_path="a__001__IC.png"),
        SourceBox("IC", cx=0.30, cy=0.40, w=0.20, h=0.20),
        200,
        100,
    )
    second = geometry_for_row(
        _row(crop_path="b__002__IC.png", body_x="30"),
        SourceBox("IC", cx=0.55, cy=0.40, w=0.20, h=0.20),
        200,
        100,
    )
    detection = Detection(
        label="Bad_qiaojiao",
        confidence=0.87654321,
        bbox=BoundingBox(72, 28, 88, 39),
        class_id=1,
    )

    mapped, unmapped = map_detections_to_crops(
        [detection], [first, second], ["Bad_podu", "Bad_qiaojiao"]
    )

    assert unmapped == 0
    assert sum(len(boxes) for boxes in mapped.values()) == 1
    chosen = next(boxes[0] for boxes in mapped.values() if boxes)
    assert chosen["cls"] == "Bad_qiaojiao"
    assert chosen["proposal_confidence"] == 0.876543
    assert chosen["w"] >= 2 and chosen["h"] >= 2


def test_detection_in_context_margin_is_not_attached_to_target_component() -> None:
    geometry = geometry_for_row(
        _row(),
        SourceBox("IC", cx=0.30, cy=0.40, w=0.20, h=0.20),
        200,
        100,
    )
    # Visible in the padded crop (x=30..110, y=20..80), but outside the target
    # component body (x=40..80, y=30..50).
    detection = Detection(
        label="Bad_podu",
        confidence=0.8,
        bbox=BoundingBox(91, 55, 101, 65),
        class_id=0,
    )

    mapped, unmapped = map_detections_to_crops(
        [detection], [geometry], ["Bad_podu", "Bad_qiaojiao"]
    )

    assert mapped[geometry.crop_path] == []
    assert unmapped == 1


def test_draft_never_claims_human_verification_or_cleanliness() -> None:
    empty = _draft_record([], confidence=0.25)
    proposed = _draft_record(
        [
            {
                "cls": "Bad_podu",
                "x": 1,
                "y": 2,
                "w": 3,
                "h": 4,
                "proposal_confidence": 0.7,
            }
        ],
        confidence=0.25,
    )

    assert empty["status"] == ""
    assert "CHƯA được coi là sạch" in empty["notes"]
    assert proposed["status"] == ""
    assert proposed["boxes"][0]["cls"] == "Bad_podu"
