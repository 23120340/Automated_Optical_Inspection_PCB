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
    #: Chỉ nhận ``ic_hai_ben`` khi cặp cạnh có chân TRÙNG cặp cạnh dài của thân.
    #:
    #: Bước 5.5 **không đọc** ``lead_edges``: với ``dual_sided`` nó giữ
    #: ``lead_top``/``lead_bottom`` trong hệ toạ độ *linh kiện*, tức luôn là hai
    #: cạnh **dài** của thân. Nên nếu luật nhận một con IC có chân trên hai cạnh
    #: **ngắn**, ROI rơi trọn vào hai cạnh không có chân gì -- đo trên mẫu tổng
    #: hợp: độ phủ pad 4/4 -> **0/4**.
    #:
    #: Đây là chốt tạm, không phải lời giải. Lời giải là truyền cạnh chân xuống
    #: 5.5 (kế hoạch §10.1); chốt này chỉ biến ROI sai cạnh **im lặng** thành
    #: một lần bỏ qua, tức lùi về đúng hành vi hôm nay. Tắt nó khi và chỉ khi
    #: 5.5 đã biết đọc cạnh thật.
    require_leads_on_long_axis: bool = True
    #: Tách tụ TRỤ ĐỨNG khỏi tụ CHIP: hộp gần vuông thì là trụ đứng nhìn từ
    #: trên. Đo trên 169 tụ gán tay (85 trụ / 84 chip, 32 bo, chia hiệu chỉnh
    #: và nghiệm thu **theo bo**, đóng băng ngưỡng trước khi đo):
    #:
    #:   ============================  =========  ===============
    #:   luật                          ngưỡng     nghiệm thu
    #:   ============================  =========  ===============
    #:   **aspect**                    **1,17**   **90,5%**
    #:   kích thước                    36 px      88,6%
    #:   độ tròn 4piA/P^2              0,88       68,2%
    #:   baseline "luôn đoán chip"      --        68,2%
    #:   ============================  =========  ===============
    #:
    #: Số đo theo **tần suất thật**, cân lại theo tỉ lệ lấy mẫu của từng tầng --
    #: tập 750 lấy phân tầng nên nó over-sample linh kiện to. Wilson 95% trên 42
    #: mẫu nghiệm thu: **75,0%--94,8%**; khoảng còn rộng, cần thêm bo.
    #:
    #: Chọn aspect chứ không chọn kích thước dù hai con số sát nhau: ngưỡng kích
    #: thước không chuyển được giữa các độ phóng đại (§6.3), aspect thì có.
    capacitor_round_max_aspect: float = 1.17
    #: ``4piA/P^2`` -- ý tưởng ở §8.8, **đã đo và loại**. Nó không chỉ chưa hiệu
    #: chuẩn được mà còn **chỉ ngược**: tụ trụ có độ tròn trung vị 0,343 còn tụ
    #: chip 0,635, vì rãnh chữ thập trên nắp nhôm và phản quang làm vỡ contour,
    #: trong khi thân chip cho một hình chữ nhật sạch. Ngưỡng tốt nhất tìm được
    #: đạt 68,2% trên tập nghiệm thu -- đúng bằng baseline, tức không thêm gì.
    circularity_threshold: float | None = None


def _edge_of(lead: Detection, body: Detection) -> str | None:
    """Cạnh mà một chân thò ra. ``None`` khi không quy được về cạnh nào.

    Đo bằng **mép ngoài** của chân, không bằng tâm. Lý do đo được: bước 2 cắt
    một cửa sổ *quanh* linh kiện rồi tìm mối hàn bên trong, nên mối hàn nó trả
    về thường **vắt qua** mép thân — tâm còn nằm trong hộp trong khi mép đã ra
    ngoài. Bản trước đòi *tâm* ra ngoài nên bỏ sót đúng nhóm đó, và nhánh ``ic``
    của luật gần như không bao giờ chạy tới.

    Thử cả năm cách trên 3-4 ảnh thật với lead detector thật (1.972 chân đã gán):

    ==========================================  =========  ==========
    cách                                        chân có     2 cạnh
                                                cạnh        đối
    ==========================================  =========  ==========
    tâm ra ngoài (bản cũ)                          683          16
    **mép ngoài ra ngoài (bản này)**             **1.086**    **22**
    >=25% diện tích ra ngoài                       966          17
    cạnh gần nhất, luôn gán                      1.972          35
    ==========================================  =========  ==========

    "Cạnh gần nhất" phủ cao nhất nhưng **bị loại**: nó gán thêm 886 chân nằm
    **hẳn trong** thân, trong đó **223 nằm sâu hơn 25% nửa bề ngang**. Không chân
    thật nào nằm sâu trong lòng thân, nên đó là nguồn **topology giả** — mà
    topology giả thì 5.5 thu ROI về hai cạnh và mất vùng kiểm thật (§8.3).

    Bản này chỉ nhận thêm nhóm **vắt qua mép** (403 chân): tất cả đều thò ra
    ngoài thân thật, tức đều là ứng viên fillet hợp lý.
    """

    lead_box, box = lead.bbox, body.bbox
    protrusion = {
        "left": box.x1 - lead_box.x1,
        "right": lead_box.x2 - box.x2,
        "top": box.y1 - lead_box.y1,
        "bottom": lead_box.y2 - box.y2,
    }
    edge, gap = max(protrusion.items(), key=lambda item: item[1])
    if gap <= 0:
        return None

    # Hộp chân bao TRỌN thân thì thò ra cả bốn phía và ``max`` chọn bừa một
    # cạnh. Đòi thêm: tâm chân phải lệch về đúng phía cạnh đó so với tâm thân.
    # Chân thật luôn lệch; hộp bao trọn thì tâm gần trùng tâm thân nên bị loại.
    cx, cy = lead_box.center
    bx, by = box.center
    leans = {"left": bx - cx, "right": cx - bx, "top": by - cy, "bottom": cy - by}
    return edge if leans[edge] > 0 else None


def _long_edge_pair(body: Detection) -> frozenset[str]:
    """Cặp cạnh DÀI của thân, trong toạ độ ảnh.

    Hộp rộng hơn cao thì hai cạnh dài là trên/dưới; cao hơn rộng thì là
    trái/phải. Đây chính là cặp mà 5.5 dựng ROI cho ``dual_sided``, vì trục +x
    của hệ toạ độ linh kiện chạy dọc cạnh dài.
    """

    if body.bbox.width >= body.bbox.height:
        return frozenset({"top", "bottom"})
    return frozenset({"left", "right"})


def _ic_package(
    edges: frozenset[str],
    has_leads: bool,
    board_has_leads: bool,
    aspect: float,
    long_edges: frozenset[str],
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
        if config.require_leads_on_long_axis and edges != long_edges:
            # Chân nằm trên hai cạnh NGẮN. 5.5 vẫn sẽ dựng ROI trên hai cạnh
            # DÀI vì nó không đọc ``lead_edges``, nên nhận ở đây là đặt cả hai
            # ROI vào chỗ không có chân. Bỏ qua để lùi về ``multi_pin``, đường
            # đó còn tự dò cặp cạnh bằng pixel (``_dominant_edge_pair``).
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
                edges, bool(own), board_has_leads, side_long / side_short,
                _long_edge_pair(body), config,
            )
        elif family == "capacitor":
            # Tụ trụ đứng nhìn từ trên là một hình TRÒN, nên hộp của nó gần
            # vuông; tụ chip thì thuôn dài. Ngưỡng 1,17 đo trên 169 tụ gán tay,
            # chia hiệu chỉnh/nghiệm thu theo bo -- xem ``capacitor_round_max_aspect``.
            #
            # An toàn được là nhờ một sửa đi kèm ở ``solder/geometry.py``: trước
            # đó ``tru_dung`` luôn phát đúng MỘT cặp ROI, nên trên thân gần
            # vuông nó cho 2 ROI trong khi ``two_terminal`` cho 4 -- gán đúng
            # lớp lại xoá mất hai ROI thật. Giờ hai đường cho ROI y hệt nhau khi
            # trục chưa quyết được, nên bật luật này chỉ có thể THÊM chứ không
            # bớt vùng kiểm.
            side_long = max(body.bbox.width, body.bbox.height)
            side_short = max(1.0, min(body.bbox.width, body.bbox.height))
            aspect = side_long / side_short
            decided = (
                ("tru_dung", f"thân gần vuông (aspect {aspect:.2f} < "
                             f"{config.capacitor_round_max_aspect}) = trụ đứng nhìn từ trên")
                if aspect < config.capacitor_round_max_aspect
                else ("hai_chan", f"thân thuôn dài (aspect {aspect:.2f}) = tụ chip")
            )
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
