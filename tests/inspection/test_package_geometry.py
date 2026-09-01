from __future__ import annotations

import numpy as np
import pytest

from aoi_pipeline.config import SolderJointConfig
from aoi_pipeline.models import BoundingBox, Detection, SolderJoint
from aoi_pipeline.solder.geometry import SolderJointCropper
from aoi_pipeline.solder.package_validation import assess_package_topology


def _detection(package_class: str) -> Detection:
    return Detection(
        label="ic",
        confidence=0.9,
        bbox=BoundingBox(40, 50, 140, 90),
        detection_id=f"det_{package_class}",
        metadata={
            "terminal_geometry_override": package_class,
            "package_profile": {
                "package_class": package_class,
                "source": "test",
            },
        },
    )


def _derive(package_class: str):
    config = SolderJointConfig(
        include_body_view=False,
        split_pins=False,
        refine_to_metal=False,
        deconflict_neighbours=False,
    )
    return SolderJointCropper(config).derive(
        np.zeros((180, 220, 3), dtype=np.uint8),
        [_detection(package_class)],
    )


@pytest.mark.parametrize("package_class", ["hai_chan", "tru_dung"])
def test_two_terminal_package_classes_emit_exactly_one_pair(package_class: str) -> None:
    joints = _derive(package_class)
    assert len(joints) == 2
    assert {item.terminal_geometry for item in joints} == {package_class}


@pytest.mark.parametrize("package_class", ["goi_nho", "ic_hai_ben"])
def test_two_sided_packages_do_not_build_the_other_two_edges(package_class: str) -> None:
    joints = _derive(package_class)
    assert {item.position for item in joints} == {"lead_top", "lead_bottom"}


def test_four_sided_package_keeps_all_perimeter_edges() -> None:
    joints = _derive("ic_bon_ben")
    assert {item.position for item in joints} == {
        "lead_left",
        "lead_right",
        "lead_top",
        "lead_bottom",
    }


def test_hidden_terminal_package_emits_no_fabricated_2d_roi() -> None:
    assert _derive("ic_khong_chan") == []


def test_connector_rows_are_inside_the_component_instead_of_outside_perimeter() -> None:
    detection = _detection("connector")
    joints = _derive("connector")
    assert len(joints) == 2
    assert all(detection.bbox.y1 < item.bbox.center[1] < detection.bbox.y2 for item in joints)


def test_package_topology_passes_an_exact_two_terminal_pair() -> None:
    detection = _detection("hai_chan")
    checks = assess_package_topology([detection], _derive("hai_chan"))
    assert len(checks) == 1
    assert checks[0].status == "pass"
    assert checks[0].actual_pin_count == 2


def test_unsplit_ic_bands_are_review_not_miscounted_as_two_pins() -> None:
    detection = _detection("ic_hai_ben")
    checks = assess_package_topology([detection], _derive("ic_hai_ben"))
    assert checks[0].status == "review"
    assert checks[0].actual_pin_count is None
    assert "could not be split" in checks[0].reason


def test_hidden_terminals_are_explicitly_not_inspectable() -> None:
    detection = _detection("ic_khong_chan")
    checks = assess_package_topology([detection], [])
    assert checks[0].status == "not_inspectable"
    assert not checks[0].review_required


def test_exact_footprint_pin_count_mismatch_requires_review() -> None:
    detection = _detection("hai_chan")
    detection.metadata["package_profile"]["expected_pin_count"] = 4
    joints = _derive("hai_chan")
    for joint in joints:
        joint.metadata["package_profile"]["expected_pin_count"] = 4
    check = assess_package_topology([detection], joints)[0]
    assert check.status == "review"
    assert check.expected_pin_min == check.expected_pin_max == 4

