"""Bản đồ kiểm tra và kế hoạch chụp.

Board thật lớn hơn trường nhìn ở độ phân giải cần cho kiểm tra fillet, nên phải
chụp nhiều khung. Thứ dễ hỏng lặng lẽ nhất ở đây không phải phép chia lưới mà
là **linh kiện lọt qua khe**: một con nằm đúng đường ranh giữa hai khung bị cắt
đôi ở cả hai, và nếu không ai đếm thì nó trôi qua như đã được kiểm.
"""

from __future__ import annotations

import pytest

from aoi_pipeline.inspection_map import (
    CaptureRegion,
    InspectionMap,
    MapComponent,
    MapError,
    build_from_bom,
    build_from_cad,
    components_in_capture,
    crop_boxes_for_capture,
    plan_capture_regions,
    uncovered,
)


def _grid_map(step: float = 20.0, count: int = 5) -> InspectionMap:
    """Lưới linh kiện 5×5, cách nhau `step` mm, mỗi con 4×2 mm."""

    return InspectionMap(
        components=[
            MapComponent(designator=f"R{row}{col}", x=col * step, y=row * step,
                         width=4.0, height=2.0)
            for row in range(count) for col in range(count)
        ],
        source="test",
    )


# --------------------------------------------------------------------------
# Nguồn nào dựng được bản đồ
# --------------------------------------------------------------------------


def test_cad_gives_a_map_with_positions_and_rotation() -> None:
    from aoi_pipeline.solder.cad import BoardCad, CadComponent

    board = BoardCad(components=[
        CadComponent(designator="R1", x=10.0, y=20.0, rotation=90.0,
                     width=3.2, height=1.6, part_class="resistor"),
        CadComponent(designator="C1", x=30.0, y=20.0, part_class="capacitor"),
    ], source="board.csv", source_format="placement_csv", units="mm")

    result = build_from_cad(board)

    assert len(result) == 2
    assert result.components[0].rotation == 90.0
    assert result.components[0].has_size


def test_a_bom_without_coordinates_is_refused_not_guessed() -> None:
    """BOM dạng mua hàng nói board có NHỮNG GÌ, không nói Ở ĐÂU. Trả về một bản
    đồ mà mọi linh kiện chồng lên nhau ở gốc toạ độ thì tệ hơn là từ chối."""

    from aoi_pipeline.bom import BillOfMaterials, BomEntry

    bom = BillOfMaterials(entries=[
        BomEntry(designator="R1"), BomEntry(designator="R2"),
    ], source="mua-hang.csv")

    with pytest.raises(MapError, match="không có toạ độ"):
        build_from_bom(bom)


def test_a_positioned_bom_works_just_like_cad() -> None:
    from aoi_pipeline.bom import BillOfMaterials, BomEntry

    bom = BillOfMaterials(entries=[
        BomEntry(designator="R1", x=5.0, y=6.0, width=2.0, height=1.0),
        BomEntry(designator="R2", x=15.0, y=6.0),
    ], source="centroid.csv")

    result = build_from_bom(bom)
    assert len(result) == 2
    assert result.components[0].has_size


def test_only_the_side_being_inspected_is_mapped() -> None:
    from aoi_pipeline.solder.cad import BoardCad, CadComponent

    board = BoardCad(components=[
        CadComponent(designator="R1", x=1.0, y=1.0, side="top"),
        CadComponent(designator="R2", x=2.0, y=2.0, side="bottom"),
    ], source="b", source_format="placement_csv", units="mm")

    assert [c.designator for c in build_from_cad(board).components] == ["R1"]
    assert len(build_from_cad(board, side=None)) == 2


# --------------------------------------------------------------------------
# Kế hoạch chụp
# --------------------------------------------------------------------------


def test_a_board_that_fits_the_field_of_view_needs_one_shot() -> None:
    regions = plan_capture_regions(_grid_map(), 200.0, 200.0)
    assert len(regions) == 1
    assert len(regions[0].designators) == 25


def test_a_board_bigger_than_the_field_of_view_is_tiled() -> None:
    """Board 80×80 mm, trường nhìn 50×50 → phải hơn một khung."""

    regions = plan_capture_regions(_grid_map(), 50.0, 50.0, overlap=0.15)
    assert len(regions) >= 4
    assert {(r.row, r.column) for r in regions} == {
        (row, col) for row in range(max(r.row for r in regions) + 1)
        for col in range(max(r.column for r in regions) + 1)
    }


def test_every_component_lands_in_at_least_one_capture() -> None:
    """Con số phải nhìn trước khi tin vào một kế hoạch: một linh kiện không
    khung nào phủ TRỌN là một linh kiện sẽ không được kiểm."""

    board = _grid_map()
    regions = plan_capture_regions(board, 50.0, 50.0, overlap=0.2)
    assert uncovered(board, regions) == []


def test_overlap_is_what_stops_a_component_falling_between_two_frames() -> None:
    """Không chồng lấn thì một linh kiện đúng đường ranh bị cắt đôi ở cả hai
    khung, và KHÔNG khung nào kiểm được nó."""

    # Một con nằm đúng chỗ hai khung 40 mm gặp nhau.
    board = InspectionMap(components=[
        MapComponent(designator="R1", x=40.0, y=20.0, width=6.0, height=6.0),
        MapComponent(designator="R2", x=5.0, y=20.0, width=2.0, height=2.0),
        MapComponent(designator="R3", x=75.0, y=20.0, width=2.0, height=2.0),
    ], source="test")

    without = plan_capture_regions(board, 40.0, 40.0, overlap=0.0, margin_mm=0.0)
    with_overlap = plan_capture_regions(board, 40.0, 40.0, overlap=0.3, margin_mm=0.0)

    assert "R1" in uncovered(board, without), "không chồng lấn thì R1 phải lọt"
    assert "R1" not in uncovered(board, with_overlap), "chồng lấn phải cứu được R1"


def test_a_component_is_assigned_only_when_it_fits_whole() -> None:
    """Gán theo TÂM thì linh kiện ở mép bị cắt mà vẫn coi là đã kiểm.

    Con BIG rộng 10 mm, tâm ở x=19 trong một khung 0–20 mm: tâm nằm trong,
    nhưng nửa phải thò ra ngoài. Nó KHÔNG được gán vào khung đó.
    """

    board = InspectionMap(components=[
        MapComponent(designator="BIG", x=19.0, y=10.0, width=10.0, height=2.0),
    ], source="test")
    regions = plan_capture_regions(board, 20.0, 20.0, overlap=0.0, margin_mm=0.0)

    for region in regions:
        left, _, right, _ = region.bounds_mm
        fits = left <= 14.0 and 24.0 <= right
        assert ("BIG" in region.designators) == fits, (
            f"khung {region.index} ({left:.1f}–{right:.1f} mm) gán sai: "
            f"nằm trọn={fits}, đã gán={'BIG' in region.designators}"
        )


def test_a_component_with_no_size_is_assumed_bigger_than_a_point() -> None:
    """Một linh kiện bị cắt đôi ở mép là một linh kiện không kiểm được, nên
    thà giả định to hơn thật còn hơn nhỏ hơn thật."""

    board = InspectionMap(
        components=[MapComponent(designator="R1", x=19.9, y=10.0)], source="test")
    regions = plan_capture_regions(board, 20.0, 20.0, overlap=0.0, margin_mm=0.0,
                                   default_component_mm=6.0)
    for region in regions:
        if "R1" in region.designators:
            left, _, right, _ = region.bounds_mm
            assert left <= 19.9 - 3.0 and 19.9 + 3.0 <= right


def test_rotation_changes_which_way_a_part_is_long() -> None:
    flat = MapComponent(designator="a", x=0.0, y=0.0, width=10.0, height=2.0)
    turned = MapComponent(designator="b", x=0.0, y=0.0, width=10.0, height=2.0,
                          rotation=90.0)

    assert flat.extent_mm(1.0)[0] == pytest.approx(5.0)
    assert turned.extent_mm(1.0)[0] == pytest.approx(1.0, abs=1e-6)
    assert turned.extent_mm(1.0)[1] == pytest.approx(5.0)


def test_an_empty_map_or_a_silly_field_of_view_is_refused() -> None:
    with pytest.raises(MapError, match="rỗng"):
        plan_capture_regions(InspectionMap(), 10.0, 10.0)
    with pytest.raises(MapError, match="lớn hơn 0"):
        plan_capture_regions(_grid_map(), 0.0, 10.0)
    with pytest.raises(MapError, match="Chồng lấn"):
        plan_capture_regions(_grid_map(), 10.0, 10.0, overlap=0.95)


# --------------------------------------------------------------------------
# Từ khung chụp về crop
# --------------------------------------------------------------------------


def test_crop_boxes_map_millimetres_onto_the_captured_pixels() -> None:
    """Không cần dò lại linh kiện trên ảnh vừa chụp — bản đồ đã biết chúng ở
    đâu, và khung chụp cho biết mm nào ứng với pixel nào."""

    board = InspectionMap(components=[
        MapComponent(designator="R1", x=10.0, y=10.0, width=4.0, height=2.0),
    ], source="test")
    region = CaptureRegion(index=0, row=0, column=0, center_x=10.0, center_y=10.0,
                           width=20.0, height=20.0, designators=("R1",))

    boxes = crop_boxes_for_capture(board, region, 1000, 1000, padding=0.0)

    assert "R1" in boxes
    x1, y1, x2, y2 = boxes["R1"].to_int()
    # 20 mm phủ 1000 px -> 50 px/mm. Linh kiện 4×2 mm ở tâm -> 200×100 px giữa ảnh.
    assert (x2 - x1, y2 - y1) == (200, 100)
    assert abs((x1 + x2) / 2 - 500) <= 1 and abs((y1 + y2) / 2 - 500) <= 1


def test_padding_widens_the_crop_so_the_fillet_is_inside_it() -> None:
    board = InspectionMap(components=[
        MapComponent(designator="R1", x=10.0, y=10.0, width=4.0, height=2.0),
    ], source="test")
    region = CaptureRegion(index=0, row=0, column=0, center_x=10.0, center_y=10.0,
                           width=20.0, height=20.0, designators=("R1",))

    tight = crop_boxes_for_capture(board, region, 1000, 1000, padding=0.0)["R1"]
    padded = crop_boxes_for_capture(board, region, 1000, 1000, padding=0.25)["R1"]
    assert padded.width > tight.width and padded.height > tight.height


def test_only_the_components_of_that_capture_are_returned() -> None:
    board = _grid_map()
    regions = plan_capture_regions(board, 50.0, 50.0, overlap=0.15)
    for region in regions:
        items = components_in_capture(board, region)
        assert {item.designator for item in items} == set(region.designators)


# --------------------------------------------------------------------------
# Con số thực tế của dự án
# --------------------------------------------------------------------------


def test_the_camera_plan_matches_the_hardware_document() -> None:
    """`Docs/yeu_cau_phan_cung_camera.md` khuyến nghị cảm biến 20 MP với trường
    nhìn ~137 × 91 mm ở 25 µm/px. Board 200 × 150 mm thì phải chụp nhiều khung
    — đây là lý do module này tồn tại."""

    board = InspectionMap(components=[
        MapComponent(designator=f"R{i}", x=(i % 20) * 10.0, y=(i // 20) * 10.0,
                     width=3.0, height=2.0)
        for i in range(300)
    ], source="test")

    regions = plan_capture_regions(board, 137.0, 91.0, overlap=0.15)
    assert len(regions) > 1, "board 200×150 mm không thể chụp một khung ở 25 µm/px"
    assert uncovered(board, regions) == []
