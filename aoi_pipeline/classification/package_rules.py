"""Bước 5.2 bằng LUẬT: chia nhỏ gói *bên trong* một họ do 6.1 trả về.

Không thay classifier 6.1 mà nối sau nó. Lý do đo được, ghi ở
``Docs/ke_hoach/ke_hoach_phan_nhom_package.md`` §8.1: trên 16.632 box có nhãn
footprint thật, chip 2 chân chiếm **86,5%**, nên luật "luôn đoán chip" đã đúng
86,5% mà không cần nghĩ. Luật ngưỡng tốt nhất trên **tỉ lệ cạnh** chỉ đạt
**84,5%** -- *tệ hơn* baseline; trên diện tích được 88,7%, tức +2,2 điểm. Luật
hình học toàn cục trên box thân gần như vô dụng. Biết trước họ là thứ xoá đi
mất cân bằng đó.

Đó cũng là lời giải thích bằng số cho luật cũ bị gỡ ở ``73ce2aa``: nó dùng
đúng đặc trưng đo được là *dưới* baseline.

Phạm vi thật vì thế nhỏ: **hai họ** cần chia, không phải bảy lớp tự do.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..models import ClassProbability, Detection
from ..solder.leads import assign_leads_to_components
from .package import PACKAGE_CLASS_NAMES, PackageClassification

__all__ = [
    "FAMILY_PACKAGE",
    "SPLIT_FAMILIES",
    "PackageRuleConfig",
    "resolve_packages_by_rule",
]

#: Họ ứng đúng MỘT kiểu vỏ, không cần luật gì. Chỉ liệt kê những họ kế hoạch
#: §8.2 chốt rõ; ``magnetic``/``protection``/``timing``/``acoustic`` và nhóm
#: hộp lớn nhiều chân (``relay``/``display``/``switch_control``/
#: ``battery_power_input``) CỐ TÌNH để trống -- không lớp nào trong bảy lớp tả
#: đúng chúng, và bịa ra một ánh xạ là đúng kiểu sai đã tạo ra ``73ce2aa``.
#: Không có ánh xạ thì không sinh kết quả, và ``terminal_geometry()`` lùi về
#: đường họ-detector y như hôm nay.
FAMILY_PACKAGE: Mapping[str, str] = {
    "resistor": "hai_chan",
    "led": "hai_chan",
    "diode": "hai_chan",
    "discrete_semiconductor": "goi_nho",
    "connector": "connector",
}

#: Hai họ phải chia nhỏ bằng luật.
SPLIT_FAMILIES: frozenset[str] = frozenset({"capacitor", "ic"})

_EDGES = ("left", "right", "top", "bottom")
_OPPOSITE_PAIRS = (frozenset({"left", "right"}), frozenset({"top", "bottom"}))
_RULE_VERSION = "package_rules/1.0"


@dataclass(slots=True)
class PackageRuleConfig:
    """Chính sách của bộ luật 5.2.

    ``enabled`` mặc định **False**: cổng 3 của kế hoạch (§8.7) đòi bật tay sau
    khi đo lại độ phủ pad trên board thật, đúng như ô ``lead_detector`` đang
    làm. Có file trên đĩa không được tự đổi hình học ROI.
    """

    enabled: bool = False
    #: Một chân lạc không được quyết một cạnh. SOIC-8 có 4 chân mỗi bên nên
    #: ngưỡng 2 vẫn thoáng.
    min_leads_per_edge: int = 2
    #: Chỉ tin họ khi 6.1 tự tin. ``review``/``unknown`` thì không áp luật --
    #: luật khoá theo họ nên họ sai là luật sai theo.
    require_family_accept: bool = True
    #: §8.4: chỉ kết luận "không thấy chân" khi lead detector chứng minh được
    #: nó đang chạy trên chính board này.
    require_lead_evidence_on_board: bool = True
    #: Suy "gói không có chân" từ việc KHÔNG THẤY chân. Mặc định **tắt**.
    #:
    #: Chốt ``require_lead_evidence_on_board`` chỉ chứng minh detector không
    #: chết hẳn; nó không chứng minh detector không bỏ sót đúng con IC này.
    #: Mà ``ic_khong_chan`` có ``PadProfile(0, 0)`` nên kết luận sai làm 5.5 bỏ
    #: SẠCH ROI của linh kiện đó — mất mối hàn, im lặng. Chi phí hai chiều
    #: lệch hẳn: đoán nhầm sang ``multi_pin`` chỉ tốn công xem lại.
    #: Gói ẩn chân nên đến từ footprint/CAD hoặc nhãn tay, tức bằng chứng
    #: DƯƠNG, chứ không từ một phép suy trên sự vắng mặt.
    allow_hidden_from_absence: bool = False
    #: Chân chỉ thấy ở 2 cạnh đối mà thân lại GẦN VUÔNG thì nhiều khả năng là
    #: một QFP mới bị nhìn thấy một nửa, chứ không phải SOIC thật. Đo trên 117
    #: IC gán tay của dự án: ``ic_bon_ben`` có aspect trung vị 1,03 và 64% dưới
    #: 1,3; ``ic_hai_ben`` có trung vị 1,95 và chỉ 13% dưới 1,3.
    #: Dưới ngưỡng này thì trả ``None`` thay vì đoán -- đoán sai ở đây làm 5.5
    #: bỏ hai cạnh chân CÓ THẬT.
    two_sided_min_aspect: float = 1.3
    #: Chưa đo được, nên chưa bật. Bộ dữ liệu công khai có **0** tụ hoá trụ
    #: đứng (§8.8), nên ngưỡng độ tròn ``4piA/P^2`` không có gì để hiệu chuẩn.
    #: Đoán một con số ở đây là lặp lại đúng lỗi ``73ce2aa``. Khi có tập kiểm
    #: gán tay ở §7, đo rồi mới điền.
    circularity_threshold: float | None = None


def _edge_of(lead: Detection, body: Detection) -> str | None:
    """Cạnh mà một chân nằm ngoài. ``None`` nếu tâm chân nằm trong thân."""

    cx, cy = lead.bbox.center
    box = body.bbox
    gaps = {
        "left": box.x1 - cx,
        "right": cx - box.x2,
        "top": box.y1 - cy,
        "bottom": cy - box.y2,
    }
    edge, gap = max(gaps.items(), key=lambda item: item[1])
    return edge if gap > 0 else None


def _ic_package(
    edges: frozenset[str],
    has_leads: bool,
    board_has_leads: bool,
    aspect: float,
    config: PackageRuleConfig,
) -> tuple[str, str] | None:
    """Trả (gói, lý do) cho họ ``ic``, hoặc ``None`` khi không kết luận được."""

    if not has_leads:
        # KHÔNG suy gói ẩn chân từ sự vắng mặt của chân. Xem
        # ``allow_hidden_from_absence``: chốt "board có chân ở chỗ khác" chỉ
        # chứng minh detector không chết hẳn, không chứng minh nó không bỏ sót
        # đúng con này -- mà kết luận sai thì 5.5 bỏ SẠCH ROI của nó.
        if not config.allow_hidden_from_absence:
            return None
        if config.require_lead_evidence_on_board and not board_has_leads:
            return None
        return "ic_khong_chan", "không có chân nào quanh thân"
    if edges in _OPPOSITE_PAIRS:
        if aspect < config.two_sided_min_aspect:
            # Thân gần vuông + chỉ thấy chân hai cạnh = nhiều khả năng QFP mới
            # nhìn thấy một nửa. Nhận ``ic_hai_ben`` ở đây làm 5.5 bỏ hai cạnh
            # chân có thật.
            return None
        return "ic_hai_ben", f"dải chân trên đúng 2 cạnh đối: {sorted(edges)}"
    if len(edges) == 4:
        return "ic_bon_ben", "dải chân trên cả 4 cạnh"
    # 1 cạnh, 3 cạnh, hay 2 cạnh KỀ nhau: không nằm trong ba gói nào. Trả None
    # để lùi về đường cũ thay vì đoán -- đoán ở đây làm hỏng ROI thật.
    return None


def resolve_packages_by_rule(
    bodies: Sequence[Detection],
    leads: Sequence[Detection],
    families: Mapping[str, str],
    *,
    config: PackageRuleConfig,
    lead_fusion_config: Any,
    family_decisions: Mapping[str, str] | None = None,
    crop_ids: Mapping[str, str] | None = None,
) -> list[PackageClassification]:
    """Suy gói cho từng thân linh kiện từ họ 6.1 + vị trí chân.

    Không sinh kết quả cho thân nào không kết luận được. Đó là hành vi đúng:
    thiếu kết quả thì ``terminal_geometry()`` lùi về đường họ-detector như hôm
    nay, còn một kết quả bịa ra thì đổi ROI thật.
    """

    if not config.enabled or not bodies:
        return []

    assigned = assign_leads_to_components(bodies, leads, lead_fusion_config)
    board_has_leads = bool(assigned)
    decisions = family_decisions or {}
    results: list[PackageClassification] = []

    for body in bodies:
        family = str(families.get(body.detection_id, "")).strip().lower()
        if not family:
            continue
        if (
            config.require_family_accept
            and decisions.get(body.detection_id, "accept") != "accept"
        ):
            continue

        own = assigned.get(body.detection_id, [])
        counts = Counter(
            edge for edge in (_edge_of(lead, body) for lead in own) if edge
        )
        edges = frozenset(
            edge for edge in _EDGES if counts[edge] >= config.min_leads_per_edge
        )

        if family == "ic":
            side_long = max(body.bbox.width, body.bbox.height)
            side_short = max(1.0, min(body.bbox.width, body.bbox.height))
            decided = _ic_package(
                edges, bool(own), board_has_leads, side_long / side_short, config
            )
        elif family == "capacitor":
            # Cần độ tròn contour để tách tụ trụ đứng khỏi tụ chip, mà ngưỡng
            # chưa đo được (§8.8: 0 ví dụ công khai). Không sinh kết quả =>
            # ``terminal_geometry("capacitor")`` giữ nguyên ``two_terminal``,
            # đúng cho tuyệt đại đa số tụ chip.
            decided = None
        elif family in FAMILY_PACKAGE:
            decided = (
                FAMILY_PACKAGE[family],
                f"họ '{family}' ứng đúng một kiểu vỏ",
            )
        else:
            decided = None

        if decided is None:
            continue
        package_class, reason = decided
        if package_class not in PACKAGE_CLASS_NAMES:  # pragma: no cover
            raise ValueError(f"luật sinh ra lớp ngoài taxonomy: {package_class}")
        results.append(
            PackageClassification(
                crop_id=(crop_ids or {}).get(body.detection_id, body.detection_id),
                detection_id=body.detection_id,
                package_class=package_class,
                # Luật là tất định. 1.0 nghĩa là "luật đã kích hoạt", KHÔNG
                # phải "tin 100%" -- độ tin thật nằm ở tỉ lệ trúng đo trên tập
                # kiểm gán tay (§7), không nằm ở con số này.
                probability=1.0,
                top_k=[ClassProbability(label=package_class, probability=1.0)],
                unknown_score=0.0,
                decision="accept",
                model_version=_RULE_VERSION,
                source="package_rules",
                detector_hint=body.label,
                metadata={
                    "family": family,
                    "reason": reason,
                    "lead_edges": sorted(edges),
                    "lead_count": len(own),
                    "board_has_leads": board_has_leads,
                },
            )
        )
    return results
