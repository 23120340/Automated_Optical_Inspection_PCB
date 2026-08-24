from __future__ import annotations

import numpy as np

from aoi_pipeline import (
    BoundingBox,
    Detection,
    TilingConfig,
    detect_with_adaptive_tiling,
    merge_tiled_detections,
    plan_inference_tiles,
)


def test_tile_plan_covers_image_with_overlap_and_exact_outer_edges() -> None:
    width, height = 257, 205
    tiles = plan_inference_tiles(width, height, tile_size=100, overlap_ratio=0.20)

    assert min(tile.x1 for tile in tiles) == 0
    assert min(tile.y1 for tile in tiles) == 0
    assert max(tile.x2 for tile in tiles) == width
    assert max(tile.y2 for tile in tiles) == height

    coverage = np.zeros((height, width), dtype=np.uint8)
    for tile in tiles:
        coverage[tile.y1 : tile.y2, tile.x1 : tile.x2] += 1
        assert tile.width <= 100
        assert tile.height <= 100
    assert np.all(coverage >= 1)
    assert np.any(coverage > 1)
    center = BoundingBox(128, 102, 129, 103)
    assert sum(tile.owns_center(center) for tile in tiles) == 1


def test_adaptive_tiling_maps_seam_boxes_and_removes_duplicate() -> None:
    calls = 0

    def detect(_: np.ndarray) -> list[Detection]:
        nonlocal calls
        calls += 1
        if calls == 1:
            bbox = BoundingBox(80, 30, 100, 55)
        else:
            bbox = BoundingBox(0, 30, 20, 55)
        return [Detection("resistor", 0.9, bbox, class_id=0, detection_id="local")]

    batch = detect_with_adaptive_tiling(
        detect,
        np.zeros((100, 180, 3), dtype=np.uint8),
        TilingConfig(
            mode="on",
            tile_size=100,
            overlap_ratio=0.20,
            include_full_image=False,
            merge_iou_threshold=0.45,
        ),
    )

    assert calls == 2
    assert batch.tiling_applied
    assert batch.raw_detection_count == 2
    assert len(batch.detections) == 1
    assert batch.detections[0].bbox.as_xyxy() == [80.0, 30.0, 100.0, 55.0]
    assert batch.detections[0].metadata["touches_tile_border"] is True
    assert batch.detections[0].metadata["center_in_tile_ownership"] is True
    assert batch.metrics()["duplicates_removed"] == 1


def test_seam_fragments_with_low_iou_are_merged_into_one_complete_box() -> None:
    calls = 0

    def detect(_: np.ndarray) -> list[Detection]:
        nonlocal calls
        calls += 1
        # In frame coordinates these become [65, 100] and [80, 115]. Their
        # IoU is only 0.40, so ordinary global NMS cannot remove the duplicate.
        bbox = (
            BoundingBox(65, 30, 100, 55)
            if calls == 1
            else BoundingBox(0, 30, 35, 55)
        )
        return [Detection("resistor", 0.9, bbox, class_id=0, detection_id="partial")]

    batch = detect_with_adaptive_tiling(
        detect,
        np.zeros((100, 180, 3), dtype=np.uint8),
        TilingConfig(
            mode="on",
            tile_size=100,
            overlap_ratio=0.20,
            include_full_image=False,
            merge_iou_threshold=0.45,
            seam_ios_threshold=0.50,
        ),
    )

    assert len(batch.detections) == 1
    assert batch.detections[0].bbox.as_xyxy() == [65.0, 30.0, 115.0, 55.0]
    assert batch.detections[0].metadata["seam_fragments_merged"] is True
    assert batch.detections[0].metadata["merged_from_tile_ids"] == [
        "tile_r000_c000",
        "tile_r000_c001",
    ]
    assert batch.metrics()["seam_fragments_merged"] == 1
    assert batch.metrics()["duplicates_removed"] == 1


def test_seam_ios_does_not_merge_non_overlapping_neighboring_components() -> None:
    calls = 0

    def detect(_: np.ndarray) -> list[Detection]:
        nonlocal calls
        calls += 1
        bbox = (
            BoundingBox(70, 20, 100, 40)
            if calls == 1
            else BoundingBox(0, 50, 30, 70)
        )
        return [Detection("resistor", 0.9, bbox, class_id=0)]

    batch = detect_with_adaptive_tiling(
        detect,
        np.zeros((100, 180, 3), dtype=np.uint8),
        TilingConfig(mode="on", tile_size=100, include_full_image=False),
    )

    assert len(batch.detections) == 2
    assert batch.metrics()["seam_fragments_merged"] == 0


def test_nested_partial_box_from_another_tile_keeps_complete_detection() -> None:
    complete = Detection(
        "ic",
        0.83,
        BoundingBox(100, 100, 270, 225),
        detection_id="complete",
        metadata={
            "frame_id": "frame",
            "inference_pass": "tile",
            "tile_id": "tile_left",
            "touches_tile_border": False,
            "center_in_tile_ownership": True,
        },
    )
    partial = Detection(
        "ic",
        0.68,
        BoundingBox(237, 110, 272, 220),
        detection_id="partial",
        metadata={
            "frame_id": "frame",
            "inference_pass": "tile",
            "tile_id": "tile_right",
            "touches_tile_border": True,
            "center_in_tile_ownership": False,
        },
    )

    merged = merge_tiled_detections([complete, partial], TilingConfig())

    assert len(merged) == 1
    assert merged[0].detection_id == "complete"
    assert merged[0].bbox == complete.bbox
    assert merged[0].metadata["contained_detection_ids"] == ["partial"]


def test_nested_boxes_from_same_inference_window_are_not_containment_merged() -> None:
    metadata = {
        "frame_id": "frame",
        "inference_pass": "tile",
        "tile_id": "same_tile",
        "touches_tile_border": False,
    }
    detections = [
        Detection("ic", 0.9, BoundingBox(0, 0, 100, 100), metadata=metadata),
        Detection("ic", 0.8, BoundingBox(10, 10, 30, 30), metadata=metadata),
    ]

    merged = merge_tiled_detections(detections, TilingConfig())

    assert len(merged) == 2


def test_global_merge_is_class_aware_for_adjacent_component_types() -> None:
    detections = [
        Detection("resistor", 0.92, BoundingBox(10, 10, 40, 40), class_id=0),
        Detection("capacitor", 0.90, BoundingBox(35, 10, 65, 40), class_id=1),
    ]

    merged = merge_tiled_detections(
        detections,
        TilingConfig(class_aware_merge=True, merge_iou_threshold=0.45),
    )

    assert {item.label for item in merged} == {"resistor", "capacitor"}


def test_global_merge_removes_near_identical_cross_class_hypotheses() -> None:
    detections = [
        Detection("capacitor", 0.62, BoundingBox(10, 10, 40, 40), class_id=0),
        Detection("led", 0.31, BoundingBox(11, 11, 39, 39), class_id=1),
    ]

    merged = merge_tiled_detections(detections, TilingConfig())

    assert len(merged) == 1
    assert merged[0].label == "capacitor"


def test_small_image_uses_one_ordinary_pass_in_auto_mode() -> None:
    calls: list[tuple[int, int]] = []

    def detect(image: np.ndarray) -> list[Detection]:
        calls.append(image.shape[:2])
        return [
            Detection("ic", 0.8, BoundingBox(5, 6, 20, 22), detection_id="single_pass")
        ]

    batch = detect_with_adaptive_tiling(
        detect,
        np.zeros((480, 640, 3), dtype=np.uint8),
        TilingConfig(mode="auto", tile_size=1280),
    )

    assert calls == [(480, 640)]
    assert not batch.tiling_applied
    assert batch.tiles == []
    assert batch.detections[0].detection_id == "single_pass"
    assert "source_detection_id" not in batch.detections[0].metadata


def test_auto_mode_runs_real_detail_pass_for_1000_by_750_board() -> None:
    calls: list[tuple[int, int]] = []

    def detect(image: np.ndarray) -> list[Detection]:
        calls.append(image.shape[:2])
        return []

    batch = detect_with_adaptive_tiling(
        detect,
        np.zeros((750, 1000, 3), dtype=np.uint8),
        TilingConfig(),
    )

    assert batch.tiling_applied
    assert batch.effective_tile_size == 640
    assert len(batch.tiles) == 4
    assert calls[0] == (750, 1000)
    assert calls[1:] == [(640, 640)] * 4


# --------------------------------------------------------------------------
# Manh linh kien bi tile cat doi, roi bi gan nhan khac lop
# --------------------------------------------------------------------------


def _sliver_pair() -> tuple[Detection, Detection]:
    """Đúng hình huống đo được trên ``pcb03.jpg``.

    Đường ranh giữa hai tile rơi vào x=737, cắt đôi SOIC-14 *U1* (x 601–773).
    Tile bên phải chỉ nhìn thấy 36 px cuối của con IC và trả lời ``diode``
    0.32 — cùng lúc trả lời ``ic`` 0.276 cho đúng mảnh đó, và chính cái nhãn
    ĐÚNG mới là cái bị lọc mất vì lọc lồng nhau chỉ so trong cùng một lớp.
    """

    whole = Detection(
        "ic",
        0.62,
        BoundingBox(601, 319, 773, 449),
        detection_id="u1",
        metadata={
            "frame_id": "frame",
            "inference_pass": "tile",
            "tile_id": "tile_r000_c000",
            "touches_tile_border": False,
            "center_in_tile_ownership": True,
        },
    )
    sliver = Detection(
        "diode",
        0.32,
        BoundingBox(737, 334, 777, 444),
        detection_id="sliver",
        metadata={
            "frame_id": "frame",
            "inference_pass": "tile",
            "tile_id": "tile_r000_c001",
            "touches_tile_border": True,
            "center_in_tile_ownership": False,
        },
    )
    return whole, sliver


def test_a_cross_class_sliver_left_by_a_tile_cut_is_dropped() -> None:
    merged = merge_tiled_detections(list(_sliver_pair()), TilingConfig())

    assert [item.label for item in merged] == ["ic"]
    assert merged[0].detection_id == "u1"


def test_nms_alone_can_never_catch_it_which_is_why_the_rule_exists() -> None:
    """IoU của một box nằm LỌT trong box khác bị chặn trên bởi tỉ lệ diện tích.

    Ở đây là 4290/22360 = 0.19, còn ngưỡng khác lớp là 0.70. Không phải chỉnh
    ngưỡng là xong — nới xuống 0.19 thì mọi linh kiện cạnh nhau đều bị gộp.
    """

    whole, sliver = _sliver_pair()
    assert sliver.bbox.area / whole.bbox.area < TilingConfig().cross_class_iou_threshold

    kept = merge_tiled_detections(
        [whole, sliver], TilingConfig(drop_cross_class_edge_fragments=False)
    )
    assert len(kept) == 2, "tắt luật đi thì NMS thường để lọt — đó là điểm mấu chốt"


def test_a_nested_box_the_tile_actually_owns_is_kept() -> None:
    """Bảo vệ đúng thứ không được xoá: hai lớp ``pads`` và ``pins`` của chính
    detector này SINH RA để nằm lồng trong box ``ic``."""

    whole, sliver = _sliver_pair()
    owned = Detection(
        "pins",
        0.40,
        sliver.bbox,
        detection_id="pins",
        metadata={**sliver.metadata, "center_in_tile_ownership": True,
                  "touches_tile_border": False},
    )

    merged = merge_tiled_detections([whole, owned], TilingConfig())
    assert sorted(item.label for item in merged) == ["ic", "pins"]


def test_a_nested_box_from_the_full_image_pass_is_kept() -> None:
    """Lượt chạy toàn ảnh nhìn thấy trọn linh kiện, nên box lồng của nó là một
    quan sát thật chứ không phải mảnh vụn của đường cắt."""

    whole, sliver = _sliver_pair()
    full = Detection(
        "diode",
        0.32,
        sliver.bbox,
        detection_id="full",
        metadata={"frame_id": "frame", "inference_pass": "full_image"},
    )

    merged = merge_tiled_detections([whole, full], TilingConfig())
    assert len(merged) == 2


def test_the_bigger_box_is_never_the_one_dropped() -> None:
    """Mảnh vụn phải là cái NHỎ. Nếu không kiểm, một box lỗi to trùm lên linh
    kiện thật sẽ nuốt mất chính linh kiện đó."""

    whole, sliver = _sliver_pair()
    inflated = Detection(
        "diode",
        0.32,
        BoundingBox(560, 300, 820, 470),
        detection_id="inflated",
        metadata=dict(sliver.metadata),
    )

    merged = merge_tiled_detections([whole, inflated], TilingConfig())
    assert "u1" in {item.detection_id for item in merged}
