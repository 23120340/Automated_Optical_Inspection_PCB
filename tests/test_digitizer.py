from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from aoi_pipeline.digitizer import (
    PIXEL_ARTIFACT_STATUS,
    PIXEL_COORDINATE_SPACE,
    PLACEMENT_DRAFT_STATUS,
    ConsensusConfig,
    build_consensus,
    export_pixel_csv,
    export_pixel_json,
    export_placement_draft,
    placement_draft_rows,
)
from aoi_pipeline.models import BoundingBox, Detection
from aoi_pipeline.solder.cad import load_cad


def _detection(
    label: str,
    center: tuple[float, float],
    *,
    confidence: float = 0.9,
    size: tuple[float, float] = (12.0, 8.0),
) -> Detection:
    x, y = center
    width, height = size
    return Detection(
        label,
        confidence,
        BoundingBox(
            x - width / 2.0,
            y - height / 2.0,
            x + width / 2.0,
            y + height / 2.0,
        ),
        source="ultralytics",
    )


def _ordered_fixture() -> dict[str, list[Detection]]:
    return {
        "frame_c": [
            _detection("capacitor", (60.0, 80.0), confidence=0.82),
            _detection("resistor", (20.0, 20.0), confidence=0.91),
            _detection("pads", (20.0, 30.0)),
        ],
        "frame_a": [
            _detection("resistor", (21.0, 19.0), confidence=0.94),
            _detection("capacitor", (61.0, 79.0), confidence=0.86),
            _detection("pins", (60.0, 70.0)),
        ],
        "frame_b": [
            _detection("capacitor", (59.0, 81.0), confidence=0.84),
            _detection("resistor", (19.0, 21.0), confidence=0.93),
        ],
    }


def test_consensus_is_order_independent_and_excludes_lead_labels() -> None:
    frames = _ordered_fixture()
    first = build_consensus(
        frames,
        config=ConsensusConfig(cluster_radius_px=6.0),
        canvas_size=(100, 120),
    )
    reversed_items = [
        (frame_id, list(reversed(detections)))
        for frame_id, detections in reversed(list(frames.items()))
    ]
    second = build_consensus(
        reversed_items,
        config=ConsensusConfig(cluster_radius_px=6.0),
        canvas_size=(100, 120),
    )

    assert first.to_dict() == second.to_dict()
    assert [item.label for item in first.components] == ["resistor", "capacitor"]
    assert [item.designator for item in first.components] == [
        "R_AUTO_0001",
        "C_AUTO_0001",
    ]
    assert first.excluded_observation_count == 2
    assert all(item.support_ratio == pytest.approx(1.0) for item in first.components)


def test_one_observation_per_frame_per_cluster_and_exact_duplicates_are_audited() -> None:
    duplicate = _detection("resistor", (25.0, 30.0), confidence=0.70)
    stronger = _detection("resistor", (25.0, 30.0), confidence=0.95)
    consensus = build_consensus(
        {
            "a": [duplicate, stronger],
            "b": [_detection("resistor", (26.0, 30.0))],
            "c": [_detection("resistor", (24.0, 30.0))],
        },
        config=ConsensusConfig(cluster_radius_px=5.0),
    )

    assert consensus.duplicate_observation_count == 1
    assert len(consensus.components) == 1
    component = consensus.components[0]
    assert component.observation_count == 3
    assert component.frame_ids == ("a", "b", "c")
    assert len(component.frame_ids) == len(set(component.frame_ids))
    assert component.median_confidence == pytest.approx(0.9)


def test_support_purity_and_center_mad_are_explicit_quality_evidence() -> None:
    consensus = build_consensus(
        {
            "f1": [_detection("resistor", (10.0, 10.0))],
            "f2": [_detection("resistor", (12.0, 10.0))],
            "f3": [_detection("resistor", (14.0, 10.0))],
            "f4": [_detection("capacitor", (12.0, 10.0))],
            "f5": [],
        },
        config=ConsensusConfig(
            cluster_radius_px=6.0,
            min_support_ratio=0.90,
            min_class_purity=0.80,
        ),
    )

    assert len(consensus.components) == 1
    component = consensus.components[0]
    assert component.label == "resistor"
    assert component.observation_count == 4
    assert component.support_ratio == pytest.approx(0.8)
    assert component.class_purity == pytest.approx(0.75)
    assert component.center_px == pytest.approx((12.0, 10.0))
    assert component.center_mad_px == pytest.approx(1.0)
    assert dict(component.class_counts) == {"capacitor": 1, "resistor": 3}
    assert component.consensus_status == PLACEMENT_DRAFT_STATUS
    assert component.review_reasons == ("low_support", "class_ambiguous")


def test_auto_designators_are_spatially_stable_and_never_claim_ocr() -> None:
    consensus = build_consensus(
        {
            "f1": [
                _detection("resistor", (40.0, 10.0)),
                _detection("display", (10.0, 20.0)),
                _detection("resistor", (20.0, 30.0)),
            ],
            "f2": [
                _detection("resistor", (20.0, 30.0)),
                _detection("resistor", (40.0, 10.0)),
                _detection("display", (10.0, 20.0)),
            ],
        },
        config=ConsensusConfig(cluster_radius_px=3.0),
    )

    assert [item.designator for item in consensus.components] == [
        "R_AUTO_0001",
        "AUTO_0001",
        "R_AUTO_0002",
    ]
    assert all(
        item.to_dict()["designator_source"] == "synthetic_auto"
        for item in consensus.components
    )
    assert all(item.to_dict()["rotation_deg"] is None for item in consensus.components)
    assert all(item.to_dict()["footprint"] is None for item in consensus.components)


def test_authoritative_pixel_exports_are_deterministic_and_leave_unknowns_blank(
    tmp_path: Path,
) -> None:
    consensus = build_consensus(
        _ordered_fixture(), config=ConsensusConfig(cluster_radius_px=6.0)
    )
    first_json = export_pixel_json(consensus, tmp_path / "first.json")
    second_json = export_pixel_json(consensus, tmp_path / "second.json")
    first_csv = export_pixel_csv(consensus, tmp_path / "first.csv")
    second_csv = export_pixel_csv(consensus, tmp_path / "second.csv")

    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_csv.read_bytes() == second_csv.read_bytes()
    payload = json.loads(first_json.read_text(encoding="utf-8"))
    assert payload["artifact_status"] == PIXEL_ARTIFACT_STATUS
    assert payload["coordinate_space"] == PIXEL_COORDINATE_SPACE
    assert payload["components"][0]["rotation_deg"] is None
    assert payload["components"][0]["footprint"] is None
    rows = list(csv.DictReader(StringIO(first_csv.read_text(encoding="utf-8"))))
    assert rows[0]["rotation_deg"] == ""
    assert rows[0]["footprint"] == ""
    assert rows[0]["artifact_status"] == PIXEL_ARTIFACT_STATUS


def test_placement_draft_requires_transform_and_never_invents_rotation_or_footprint(
    tmp_path: Path,
) -> None:
    consensus = build_consensus(
        {
            "a": [_detection("resistor", (20.0, 30.0))],
            "b": [_detection("resistor", (20.0, 30.0))],
        },
        config=ConsensusConfig(cluster_radius_px=2.0),
    )
    with pytest.raises(ValueError, match="pixel_to_mm_homography is required"):
        placement_draft_rows(consensus, pixel_to_mm_homography=None)

    # x_mm = 0.5*x_px + 1; y_mm = 0.25*y_px - 2.
    transform = np.array(
        [[0.5, 0.0, 1.0], [0.0, 0.25, -2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    path = export_placement_draft(
        consensus,
        tmp_path / "placement_draft_NEEDS_REVIEW.csv",
        pixel_to_mm_homography=transform,
        side="top",
    )
    rows = list(csv.DictReader(StringIO(path.read_text(encoding="utf-8"))))
    assert len(rows) == 1
    assert float(rows[0]["Mid X"]) == pytest.approx(11.0)
    assert float(rows[0]["Mid Y"]) == pytest.approx(5.5)
    assert rows[0]["Rotation"] == ""
    assert rows[0]["Footprint"] == ""
    assert rows[0]["Status"] == PLACEMENT_DRAFT_STATUS
    assert rows[0]["Designator"] == "R_AUTO_0001"
    assert "verify RefDes, rotation and footprint" in rows[0]["Comment"]
    board = load_cad(path, fmt="placement_csv")
    assert len(board.components) == 1
    assert board.components[0].designator == "R_AUTO_0001"
    assert board.components[0].x == pytest.approx(11.0)
    assert board.components[0].y == pytest.approx(5.5)


@pytest.mark.parametrize(
    "matrix",
    [
        np.eye(2),
        np.zeros((3, 3)),
        np.array([[1.0, 0.0, np.nan], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ],
)
def test_placement_draft_rejects_invalid_homography(matrix: np.ndarray) -> None:
    consensus = build_consensus({"frame": [_detection("ic", (5.0, 5.0))]})
    with pytest.raises(ValueError, match="pixel_to_mm_homography"):
        placement_draft_rows(consensus, pixel_to_mm_homography=matrix)


def test_low_support_sites_stay_in_authoritative_pixels_but_not_default_placement() -> None:
    consensus = build_consensus(
        {
            "a": [
                _detection("resistor", (10.0, 10.0)),
                _detection("capacitor", (80.0, 80.0)),
            ],
            "b": [_detection("resistor", (10.0, 10.0))],
            "c": [_detection("resistor", (10.0, 10.0))],
        },
        config=ConsensusConfig(cluster_radius_px=3.0, min_support_ratio=0.8),
    )
    assert len(consensus.components) == 2
    assert len(consensus.eligible_components) == 1
    default_rows = placement_draft_rows(
        consensus, pixel_to_mm_homography=np.eye(3)
    )
    all_rows = placement_draft_rows(
        consensus,
        pixel_to_mm_homography=np.eye(3),
        include_review_components=True,
    )
    assert len(default_rows) == 1
    assert len(all_rows) == 2
    assert all(row["Status"] == PLACEMENT_DRAFT_STATUS for row in all_rows)


def test_duplicate_frame_names_and_invalid_canvas_fail_closed() -> None:
    with pytest.raises(ValueError, match="duplicate frame ID"):
        build_consensus([("same", []), ("same", [])])
    with pytest.raises(ValueError, match="portable IDs"):
        build_consensus({str(Path.cwd().resolve() / "frame.png"): []})
    with pytest.raises(ValueError, match="canvas_size"):
        build_consensus({"frame": []}, canvas_size=(0, 100))
