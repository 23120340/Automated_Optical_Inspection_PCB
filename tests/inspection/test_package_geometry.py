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



def _square_detection(package_class: str) -> Detection:
    """Thân GẦN VUÔNG: 100x94, aspect 1,06 — dưới ``terminal_axis_min_aspect``."""

    return Detection(
        label="capacitor",
        confidence=0.9,
        bbox=BoundingBox(40, 40, 140, 134),
        detection_id=f"sq_{package_class}",
        metadata={"terminal_geometry_override": package_class},
    )


def _derive_square(package_class: str):
    config = SolderJointConfig(
        include_body_view=False, split_pins=False,
        refine_to_metal=False, deconflict_neighbours=False,
    )
    return SolderJointCropper(config).derive(
        np.zeros((200, 220, 3), dtype=np.uint8), [_square_detection(package_class)]
    )


def test_a_round_capacitor_does_not_lose_two_rois_to_a_coin_flip() -> None:
    """``tru_dung`` trên thân gần vuông phải giữ CẢ HAI trục.

    Đo trên 85 tụ trụ gán tay (kế hoạch §6.3c): **55% có hai cạnh lệch dưới
    10%**, nên "cạnh nào dài hơn" do vài pixel của hộp quyết định — nhiễu, không
    phải tín hiệu.

    Bản trước luôn phát đúng một cặp, nên trên chính nhóm đó nó cho 2 ROI trong
    khi ``two_terminal`` cho 4: **gán ĐÚNG ``tru_dung`` lại xoá mất hai ROI
    thật.** Đó là lý do đo được để chưa bật luật tách tụ, chứ không phải vì
    chưa đo được ngưỡng.

    Nguyên tắc §8: luật ảnh được THÊM hoặc GIỮ ROI, chỉ được XOÁ khi có bằng
    chứng dương — mà ở đây bằng chứng là một chênh lệch 6%.
    """

    vertical = _derive_square("tru_dung")
    chip = _derive_square("hai_chan")
    assert len(vertical) == len(chip) == 4, (
        f"gần vuông: tru_dung {len(vertical)} ROI, hai_chan {len(chip)} ROI. "
        "Lệch nhau nghĩa là chọn đúng lớp lại mất ROI"
    )
    assert (sorted(tuple(j.bbox.as_xyxy()) for j in vertical)
            == sorted(tuple(j.bbox.as_xyxy()) for j in chip))
    assert any("_cross" in j.position for j in vertical), (
        "cặp trục thứ hai phải được đặt tên _cross để người duyệt biết đó là "
        "giả thuyết thay thế, không phải bốn chân riêng"
    )


def test_an_elongated_capacitor_still_gets_exactly_one_pair() -> None:
    """Chốt mặt còn lại: thân thuôn dài thì trục KHÔNG mơ hồ, đừng sinh thừa."""

    assert len(_derive("tru_dung")) == 2


def test_measured_lead_edges_beat_the_long_axis_assumption() -> None:
    """§8.3(b): 5.5 đặt ROI theo cạnh ĐO ĐƯỢC, không theo trục dài của thân.

    Thân 100 rộng x 200 cao nên trục dài DỌC, tức hai cạnh dài là trái/phải.
    Nhưng chân đo được nằm ở trên/dưới. Bản trước luôn giữ hai cạnh dài, nên ROI
    rơi trọn vào hai cạnh không có chân — đo được 4/4 pad -> 0/4.
    """

    body = Detection(
        "ic", 0.9, BoundingBox(300, 200, 400, 400), detection_id="B0",
        metadata={
            "terminal_geometry_override": "dual_sided",
            "terminal_lead_edges": ["bottom", "top"],
            "terminal_lead_edges_space": "image",
        },
    )
    config = SolderJointConfig(include_body_view=False, split_pins=False,
                               refine_to_metal=False, deconflict_neighbours=False)
    joints = SolderJointCropper(config).derive(
        np.zeros((600, 800, 3), dtype=np.uint8), [body])

    assert len(joints) == 2
    # Cạnh trên/dưới của ẢNH: hai ROI phải nằm TRÊN và DƯỚI hộp thân.
    ys = sorted(j.bbox.center[1] for j in joints)
    assert ys[0] < 200 and ys[1] > 400, (
        f"ROI ở y={ys}; phải nằm ngoài dải 200-400 theo trục dọc, tức đúng hai "
        "cạnh mà chân được đo thấy"
    )


def test_lead_edges_in_an_undeclared_frame_are_ignored() -> None:
    """Cạnh không khai hệ toạ độ thì bỏ qua, không đoán.

    Đọc nhầm hệ là đặt ROI lệch 90 độ — đúng lỗi mà đường này sinh ra để chặn.
    """

    body = Detection(
        "ic", 0.9, BoundingBox(300, 200, 400, 400), detection_id="B0",
        metadata={
            "terminal_geometry_override": "dual_sided",
            "terminal_lead_edges": ["bottom", "top"],
            "terminal_lead_edges_space": "component",
        },
    )
    config = SolderJointConfig(include_body_view=False, split_pins=False,
                               refine_to_metal=False, deconflict_neighbours=False)
    joints = SolderJointCropper(config).derive(
        np.zeros((600, 800, 3), dtype=np.uint8), [body])
    ys = sorted(j.bbox.center[1] for j in joints)
    assert 200 <= ys[0] <= 400 and 200 <= ys[1] <= 400, (
        "hệ toạ độ lạ thì phải lùi về hai cạnh dài, không dùng cạnh đó"
    )
