"""Bộ luật 5.2 phải quyết đúng, và phải IM LẶNG đúng chỗ.

Nửa sau quan trọng hơn nửa đầu. Luật này thay chỗ một model, mà luật hỏng thì
hỏng có hệ thống chứ không hỏng dần: một lead detector recall kém biến mọi IC
thành "không thấy chân", và gói đó **không sinh ROI** (``PadProfile(0.0, 0.0)``),
tức mất mối hàn mà không ai thấy. Phần lớn test dưới đây canh những ca luật
phải trả về rỗng.
"""

from __future__ import annotations

from aoi_pipeline.classification.package_rules import (
    FAMILY_PACKAGE,
    PackageRuleConfig,
    resolve_packages_by_rule,
)
from aoi_pipeline.config import LeadFusionConfig
from aoi_pipeline.models import BoundingBox, Detection

BODY = BoundingBox(100.0, 100.0, 200.0, 200.0)
LEADS = LeadFusionConfig()


def _body(label: str = "ic", detection_id: str = "body1") -> Detection:
    return Detection(
        label=label, confidence=0.9, bbox=BODY, detection_id=detection_id
    )


def _lead(x1: float, y1: float, x2: float, y2: float, index: int = 0) -> Detection:
    return Detection(
        label="solder_joint",
        confidence=0.9,
        bbox=BoundingBox(x1, y1, x2, y2),
        detection_id=f"lead{index}",
    )


def _edge_leads(edge: str, count: int = 2, start: int = 0) -> list[Detection]:
    """Chân nằm NGOÀI thân, sát một cạnh — đúng như trên board thật."""

    out = []
    for i in range(count):
        offset = 120.0 + i * 25.0
        if edge == "left":
            box = (85.0, offset, 98.0, offset + 12.0)
        elif edge == "right":
            box = (202.0, offset, 215.0, offset + 12.0)
        elif edge == "top":
            box = (offset, 85.0, offset + 12.0, 98.0)
        else:
            box = (offset, 202.0, offset + 12.0, 215.0)
        out.append(_lead(*box, index=start + i))
    return out


def _resolve(bodies, leads, families, *, config=None, decisions=None):
    return resolve_packages_by_rule(
        bodies,
        leads,
        families,
        config=config or PackageRuleConfig(enabled=True),
        lead_fusion_config=LEADS,
        family_decisions=decisions,
    )


# --------------------------------------------------------------- không chạy

def test_the_rule_is_off_until_a_human_turns_it_on() -> None:
    """Có code trên đĩa không được tự đổi hình học ROI — cổng 3 của kế hoạch."""

    assert PackageRuleConfig().enabled is False
    body = _body()
    results = _resolve(
        [body],
        _edge_leads("left") + _edge_leads("right", start=2),
        {"body1": "ic"},
        config=PackageRuleConfig(),
    )
    assert results == []


def test_a_body_with_no_family_is_left_alone() -> None:
    """Luật khoá theo họ 6.1. Không có họ thì không có gì để chia nhỏ."""

    assert _resolve([_body()], _edge_leads("left"), {}) == []


def test_a_family_the_classifier_is_unsure_about_is_skipped() -> None:
    """Họ sai kéo luật sai theo, nên chỉ tin ``accept``."""

    body = _body("resistor", "r1")
    assert _resolve([body], [], {"r1": "resistor"}, decisions={"r1": "review"}) == []
    assert _resolve([body], [], {"r1": "resistor"}, decisions={"r1": "accept"})


# ------------------------------------------------------ họ ứng một kiểu vỏ

def test_families_with_exactly_one_package_need_no_rule_at_all() -> None:
    bodies = [_body(name, f"d{i}") for i, name in enumerate(FAMILY_PACKAGE)]
    families = {f"d{i}": name for i, name in enumerate(FAMILY_PACKAGE)}
    results = _resolve(bodies, [], families)
    got = {item.detection_id: item.package_class for item in results}
    assert got == {f"d{i}": FAMILY_PACKAGE[n] for i, n in enumerate(families.values())}
    assert {item.source for item in results} == {"package_rules"}


def test_a_family_the_plan_never_pinned_down_produces_nothing() -> None:
    """``relay``/``magnetic``/... không lớp nào trong bảy lớp tả đúng.

    Bịa ra một ánh xạ ở đây là đúng loại sai đã tạo ra luật bị gỡ ở 73ce2aa.
    Không kết quả ⇒ ``terminal_geometry`` lùi về đường họ-detector như hôm nay.
    """

    for family in ("relay", "magnetic", "display", "timing", "acoustic"):
        assert _resolve([_body(family, "x")], [], {"x": family}) == [], family


def test_capacitor_is_deliberately_not_split_yet() -> None:
    """Tách tụ trụ đứng khỏi tụ chip cần ngưỡng độ tròn, mà bộ công khai có 0
    ví dụ tụ hoá trụ đứng nên chưa hiệu chuẩn được. Đoán một con số là lặp lại
    lỗi cũ; im lặng giữ nguyên ``two_terminal``, đúng cho tuyệt đại đa số."""

    assert PackageRuleConfig().circularity_threshold is None
    assert _resolve([_body("capacitor", "c1")], [], {"c1": "capacitor"}) == []


# ------------------------------------------------------------ họ ``ic``

def test_leads_on_two_opposite_edges_read_as_a_two_sided_ic() -> None:
    results = _resolve(
        [_body()],
        _edge_leads("left") + _edge_leads("right", start=2),
        {"body1": "ic"},
    )
    assert [item.package_class for item in results] == ["ic_hai_ben"]
    assert results[0].metadata["lead_edges"] == ["left", "right"]


def test_leads_on_all_four_edges_read_as_a_four_sided_ic() -> None:
    leads = (
        _edge_leads("left", start=0)
        + _edge_leads("right", start=2)
        + _edge_leads("top", start=4)
        + _edge_leads("bottom", start=6)
    )
    results = _resolve([_body()], leads, {"body1": "ic"})
    assert [item.package_class for item in results] == ["ic_bon_ben"]


def test_leads_on_two_adjacent_edges_are_not_guessed_at() -> None:
    """Trái+trên không phải gói nào trong ba gói IC. Đoán ở đây làm hỏng ROI."""

    results = _resolve(
        [_body()],
        _edge_leads("left") + _edge_leads("top", start=2),
        {"body1": "ic"},
    )
    assert results == []


def test_one_stray_lead_does_not_decide_an_edge() -> None:
    """SOIC-8 có 4 chân mỗi bên; một hộp lạc không được tạo thành một dải."""

    results = _resolve(
        [_body()],
        _edge_leads("left", count=1) + _edge_leads("right", count=1, start=1),
        {"body1": "ic"},
    )
    assert results == []


# -------------------------------------- ca nguy hiểm: IC không thấy chân

def test_an_ic_without_leads_stays_undecided_when_the_detector_proved_nothing(
) -> None:
    """Đây là ca hỏng-có-hệ-thống của hướng luật.

    Lead detector im lặng trên cả board thì không phân biệt được "IC thật sự
    không có chân" với "detector không chạy". Kết luận ``ic_khong_chan`` ở đây
    làm 5.5 bỏ ROI của một gói CÓ chân — mất mối hàn mà không báo.
    """

    results = _resolve([_body()], [], {"body1": "ic"})
    assert results == []


def test_the_same_ic_is_decided_once_the_detector_works_elsewhere() -> None:
    """Có chân ở linh kiện khác ⇒ detector đã chứng minh nó đang chạy."""

    neighbour = Detection(
        label="ic",
        confidence=0.9,
        bbox=BoundingBox(400.0, 400.0, 500.0, 500.0),
        detection_id="body2",
    )
    neighbour_leads = [
        _lead(385.0, 420.0, 398.0, 432.0, index=10),
        _lead(385.0, 445.0, 398.0, 457.0, index=11),
        _lead(502.0, 420.0, 515.0, 432.0, index=12),
        _lead(502.0, 445.0, 515.0, 457.0, index=13),
    ]
    results = _resolve(
        [_body(), neighbour],
        neighbour_leads,
        {"body1": "ic", "body2": "ic"},
    )
    got = {item.detection_id: item.package_class for item in results}
    assert got == {"body1": "ic_khong_chan", "body2": "ic_hai_ben"}


def test_the_safety_condition_can_be_turned_off_deliberately() -> None:
    results = _resolve(
        [_body()],
        [],
        {"body1": "ic"},
        config=PackageRuleConfig(
            enabled=True, require_lead_evidence_on_board=False
        ),
    )
    assert [item.package_class for item in results] == ["ic_khong_chan"]


# ------------------------------------------------------------- hợp đồng ra

def test_the_rule_reports_itself_as_a_rule_not_as_a_model() -> None:
    """Một con số 1.0 từ luật KHÔNG có nghĩa là tin 100%.

    Nó có nghĩa là luật đã kích hoạt. Độ tin thật đo trên tập kiểm gán tay.
    ``source`` phải nói rõ để không ai đọc nhầm nó là kết quả model.
    """

    item = _resolve([_body("resistor", "r1")], [], {"r1": "resistor"})[0]
    assert item.source == "package_rules"
    assert item.model_version.startswith("package_rules/")
    assert item.probability == 1.0
    assert item.decision == "accept"
    assert item.metadata["reason"]
