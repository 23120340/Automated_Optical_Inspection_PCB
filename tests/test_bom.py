"""BOM: hợp đồng lắp ráp, không phải một file tham khảo.

Khác biệt cốt lõi so với CAD, và là lý do module này tồn tại riêng:

- File **CAD** nói *land nằm ở đâu*, và thường thiếu (thermal pad, shield,
  land cơ khí). Nên `cad_fusion` coi một detection không có trong CAD là
  **ghi nhận**, không phải lỗi.
- File **BOM** nói *board phải có những linh kiện nào*. Nó đầy đủ theo định
  nghĩa. Dưới hợp đồng đó, một linh kiện nằm ở chỗ BOM không liệt kê là
  **lỗi** — linh kiện thừa, đặt nhầm chỗ, hoặc vật lạ.

Phần lớn test ở đây bảo vệ đúng ranh giới đó.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aoi_pipeline.bom import (
    BillOfMaterials,
    BomEntry,
    BomError,
    load_bom,
    reconcile_bom,
)
from aoi_pipeline.models import BoundingBox, Detection


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def _detection(label: str, x: float, y: float, size: float = 20.0,
               detection_id: str = "") -> Detection:
    return Detection(
        label=label, confidence=0.9,
        bbox=BoundingBox(x - size / 2, y - size / 2, x + size / 2, y + size / 2),
        detection_id=detection_id or f"{label}_{int(x)}_{int(y)}",
    )


#: Toạ độ mm -> pixel, một phép chiếu đơn giản đủ để test ghép cặp.
def _identity_projection(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) * 10.0


# --------------------------------------------------------------------------
# Đọc file
# --------------------------------------------------------------------------


def test_a_placement_shaped_bom_gives_one_entry_per_part(tmp_path) -> None:
    path = _write(tmp_path, "bom.csv", """
Designator,Comment,Footprint,Mid X,Mid Y,Rotation,Layer,Width,Height
R1,10k,0402,10.0,20.0,0,Top,1.0,0.5
C1,100nF,0402,15.0,20.0,90,Top,1.0,0.5
U1,ATMEGA,SOIC-8,30.0,25.0,0,Top,5.0,4.0
""")
    bom = load_bom(path)

    assert len(bom) == 3
    assert bom.has_positions
    entry = bom.by_designator()["R1"]
    assert (entry.x, entry.y) == (10.0, 20.0)
    assert (entry.width, entry.height) == (1.0, 0.5)
    assert entry.part_class == "resistor"


def test_a_purchasing_shaped_bom_expands_its_designator_list(tmp_path) -> None:
    """Dạng BOM mua hàng: một dòng cho mỗi LOẠI linh kiện, không có toạ độ."""

    path = _write(tmp_path, "bom.csv", """
Comment,Description,Designator,Footprint,Quantity
100nF,Ceramic capacitor,"C1, C2, C5",0402,3
10k,Resistor,"R1;R2",0402,2
""")
    bom = load_bom(path)

    assert sorted(entry.designator for entry in bom.entries) == [
        "C1", "C2", "C5", "R1", "R2"
    ]
    assert not bom.has_positions
    assert bom.warnings == []


def test_a_quantity_that_disagrees_with_its_own_list_is_reported(tmp_path) -> None:
    """BOM sai trên giấy, trước khi có board nào được kiểm. Tin theo danh sách
    designator vì chúng gọi tên vị trí thật."""

    path = _write(tmp_path, "bom.csv", """
Comment,Designator,Quantity
100nF,"C1, C2, C5",5
""")
    bom = load_bom(path)

    assert len(bom) == 3
    assert any("Quantity ghi 5" in warning for warning in bom.warnings)


def test_a_single_space_is_not_a_separator(tmp_path) -> None:
    """``CONN 1`` là một designator có dấu cách, tách ra sẽ đẻ ra linh kiện ``1``."""

    path = _write(tmp_path, "bom.csv", """
Designator,Comment
CONN 1,Header
""")
    bom = load_bom(path)
    assert [entry.designator for entry in bom.entries] == ["CONN 1"]


def test_a_duplicated_designator_is_kept_once_and_flagged(tmp_path) -> None:
    path = _write(tmp_path, "bom.csv", """
Designator,Comment
R1,10k
R1,22k
""")
    bom = load_bom(path)
    assert len(bom) == 1
    assert any("R1" in warning for warning in bom.warnings)


def test_a_file_without_a_designator_column_is_refused(tmp_path) -> None:
    path = _write(tmp_path, "bom.csv", "Comment,Footprint\n100nF,0402")
    with pytest.raises(BomError, match="designator"):
        load_bom(path)


def test_a_missing_file_says_so_plainly(tmp_path) -> None:
    with pytest.raises(BomError, match="Không tìm thấy"):
        load_bom(tmp_path / "khong-co.csv")


# --------------------------------------------------------------------------
# Đối chiếu — ca quan trọng nhất
# --------------------------------------------------------------------------


def _positioned_bom(**kwargs) -> BillOfMaterials:
    return BillOfMaterials(
        entries=[
            BomEntry(designator="R1", part_class="resistor", x=10.0, y=20.0),
            BomEntry(designator="C1", part_class="capacitor", x=15.0, y=20.0),
        ],
        source="test",
        **kwargs,
    )


def test_a_component_where_the_bom_lists_nothing_is_an_error() -> None:
    """Đây chính là ca người dùng nêu: detect ra linh kiện ở toạ độ không có
    trong BOM thì cũng là một trường hợp sai."""

    bom = _positioned_bom()
    detections = [
        _detection("resistor", 100.0, 200.0),     # khớp R1
        _detection("capacitor", 150.0, 200.0),    # khớp C1
        _detection("resistor", 400.0, 400.0, detection_id="la_hoac_thua"),
    ]

    result = reconcile_bom(bom, detections, _identity_projection)

    unexpected = [item for item in result.findings if item.kind == "unexpected"]
    assert len(unexpected) == 1
    assert unexpected[0].severity == "error"
    assert unexpected[0].detection_id == "la_hoac_thua"
    assert not result.passed


def test_the_same_component_is_only_an_observation_when_the_bom_is_partial() -> None:
    """Ranh giới. Một nguồn chưa đầy đủ thì linh kiện lạ không phải lỗi của
    board — nên `complete` phải tự khai, không được đoán."""

    bom = _positioned_bom(complete=False)
    detections = [
        _detection("resistor", 100.0, 200.0),
        _detection("capacitor", 150.0, 200.0),
        _detection("resistor", 400.0, 400.0),
    ]

    result = reconcile_bom(bom, detections, _identity_projection)

    unexpected = [item for item in result.findings if item.kind == "unexpected"]
    assert len(unexpected) == 1
    assert unexpected[0].severity == "info"
    assert result.passed


def test_a_part_in_the_bom_that_is_not_on_the_board_is_an_error() -> None:
    bom = _positioned_bom()
    result = reconcile_bom(bom, [_detection("resistor", 100.0, 200.0)],
                           _identity_projection)

    missing = [item for item in result.findings if item.kind == "missing"]
    assert [item.designator for item in missing] == ["C1"]
    assert missing[0].severity == "error"


def test_a_board_that_matches_its_bom_passes_clean() -> None:
    bom = _positioned_bom()
    detections = [_detection("resistor", 100.0, 200.0),
                  _detection("capacitor", 150.0, 200.0)]

    result = reconcile_bom(bom, detections, _identity_projection)

    assert result.passed
    assert sorted(d for d, _ in result.matched) == ["C1", "R1"]
    assert result.stats["method"] == "position"


def test_a_wrong_part_at_the_right_place_is_a_warning_not_a_pass() -> None:
    bom = _positioned_bom()
    detections = [_detection("resistor", 100.0, 200.0),
                  _detection("ic", 150.0, 200.0)]    # C1 lẽ ra là tụ

    result = reconcile_bom(bom, detections, _identity_projection)

    mismatch = [item for item in result.findings if item.kind == "class_mismatch"]
    assert len(mismatch) == 1
    assert mismatch[0].designator == "C1"
    assert mismatch[0].expected_class == "capacitor"
    assert mismatch[0].observed_class == "ic"


def test_pairing_takes_the_closest_pairs_first_not_list_order() -> None:
    """Ghép tham lam theo thứ tự danh sách sẽ để một entry sớm chiếm mất
    detection vốn gần một entry sau nó hơn, và mọi cặp phía sau lệch đi một bậc."""

    bom = BillOfMaterials(entries=[
        BomEntry(designator="R1", part_class="resistor", x=10.0, y=10.0),
        BomEntry(designator="R2", part_class="resistor", x=11.0, y=10.0),
    ], source="test")
    # Detection đầu tiên nằm gần R2 hơn R1, nhưng đứng trước trong danh sách.
    detections = [
        _detection("resistor", 110.0, 100.0, detection_id="gan_R2"),
        _detection("resistor", 100.0, 100.0, detection_id="gan_R1"),
    ]

    result = reconcile_bom(bom, detections, _identity_projection)

    assert dict(result.matched) == {"R1": "gan_R1", "R2": "gan_R2"}


def test_a_detection_beyond_the_tolerance_is_not_forced_into_a_match() -> None:
    bom = BillOfMaterials(
        entries=[BomEntry(designator="R1", part_class="resistor", x=10.0, y=10.0)],
        source="test",
    )
    detections = [_detection("resistor", 900.0, 900.0)]

    result = reconcile_bom(bom, detections, _identity_projection,
                           match_tolerance_px=60.0)

    assert result.stats["missing"] == 1
    assert result.stats["unexpected"] == 1
    assert result.matched == []


# --------------------------------------------------------------------------
# Không có toạ độ
# --------------------------------------------------------------------------


def test_without_coordinates_it_still_counts_parts_per_class() -> None:
    """Yếu hơn ghép theo vị trí, và thành thật về điều đó: nói được board
    thiếu một con trở, không nói được thiếu con nào."""

    bom = BillOfMaterials(entries=[
        BomEntry(designator="R1", part_class="resistor"),
        BomEntry(designator="R2", part_class="resistor"),
        BomEntry(designator="C1", part_class="capacitor"),
    ], source="test")
    detections = [_detection("resistor", 10.0, 10.0),
                  _detection("capacitor", 50.0, 10.0)]

    result = reconcile_bom(bom, detections, None)

    assert result.stats["method"] == "count"
    missing = [item for item in result.findings if item.kind == "missing"]
    assert len(missing) == 1
    assert "resistor" in missing[0].message
    assert missing[0].designator is None, "không có toạ độ thì không chỉ được con nào"


def test_without_coordinates_an_extra_part_is_still_an_error() -> None:
    bom = BillOfMaterials(
        entries=[BomEntry(designator="R1", part_class="resistor")], source="test")
    detections = [_detection("resistor", 10.0, 10.0),
                  _detection("resistor", 50.0, 10.0)]

    result = reconcile_bom(bom, detections, None)

    unexpected = [item for item in result.findings if item.kind == "unexpected"]
    assert len(unexpected) == 1
    assert unexpected[0].severity == "error"


# --------------------------------------------------------------------------
# Cầu nối sang bộ máy CAD sẵn có
# --------------------------------------------------------------------------


def test_only_positioned_entries_cross_into_the_cad_model() -> None:
    """Registration giải một phép biến đổi từ các cặp điểm. Một đống linh kiện
    cùng nằm ở (0, 0) sẽ kéo nghiệm đó đi đâu không biết."""

    bom = BillOfMaterials(entries=[
        BomEntry(designator="R1", part_class="resistor", x=10.0, y=20.0),
        BomEntry(designator="R2", part_class="resistor"),          # không toạ độ
    ], source="test")

    board = bom.to_board_cad()

    assert [component.designator for component in board.components] == ["R1"]
    assert board.source_format == "bom"


def test_size_survives_the_trip_into_the_cad_model() -> None:
    bom = BillOfMaterials(entries=[
        BomEntry(designator="U1", part_class="ic", x=1.0, y=2.0,
                 width=5.0, height=4.0),
    ], source="test")

    component = bom.to_board_cad().components[0]
    assert (component.width, component.height) == (5.0, 4.0)
    assert component.has_size


def test_filtering_by_side_keeps_the_completeness_flag() -> None:
    """Mất cờ này là mất luôn ranh giới lỗi/ghi nhận, và mất im lặng."""

    bom = BillOfMaterials(entries=[
        BomEntry(designator="R1", side="top"),
        BomEntry(designator="R2", side="bottom"),
    ], source="test", complete=True)

    top = bom.side("top")
    assert [entry.designator for entry in top.entries] == ["R1"]
    assert top.complete is True
