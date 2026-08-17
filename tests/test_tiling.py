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
