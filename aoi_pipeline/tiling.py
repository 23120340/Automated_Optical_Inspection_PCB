"""Adaptive tiled inference for high-resolution PCB component detection.

The module operates in a frame-local coordinate space and records ``frame_id``
plus tile geometry on every detection. A future multi-camera adapter can map
those frame-local boxes through its homography before the same global merge
stage, without stitching a panorama before inference.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from .config import TilingConfig
from .image_io import ensure_bgr
from .models import BoundingBox, Detection, intersection_over_smaller, intersection_over_union


@dataclass(frozen=True, slots=True)
class InferenceTile:
    """One right/bottom-exclusive rectangular inference window."""

    tile_id: str
    x1: int
    y1: int
    x2: int
    y2: int
    ownership_x1: float | None = None
    ownership_y1: float | None = None
    ownership_x2: float | None = None
    ownership_y2: float | None = None

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def bbox(self) -> BoundingBox:
        return BoundingBox(float(self.x1), float(self.y1), float(self.x2), float(self.y2))

    @property
    def ownership_bbox(self) -> BoundingBox:
        return BoundingBox(
            float(self.x1 if self.ownership_x1 is None else self.ownership_x1),
            float(self.y1 if self.ownership_y1 is None else self.ownership_y1),
            float(self.x2 if self.ownership_x2 is None else self.ownership_x2),
            float(self.y2 if self.ownership_y2 is None else self.ownership_y2),
        )

    def owns_center(self, bbox: BoundingBox) -> bool:
        ownership = self.ownership_bbox
        center_x = (bbox.x1 + bbox.x2) / 2.0
        center_y = (bbox.y1 + bbox.y2) / 2.0
        return bool(
            ownership.x1 <= center_x < ownership.x2
            and ownership.y1 <= center_y < ownership.y2
        )

    def to_dict(self, offset: tuple[int, int] = (0, 0)) -> dict[str, Any]:
        offset_x, offset_y = offset
        return {
            "tile_id": self.tile_id,
            "xyxy": [
                self.x1 + offset_x,
                self.y1 + offset_y,
                self.x2 + offset_x,
                self.y2 + offset_y,
            ],
            "width": self.width,
            "height": self.height,
            "ownership_xyxy": [
                self.ownership_bbox.x1 + offset_x,
                self.ownership_bbox.y1 + offset_y,
                self.ownership_bbox.x2 + offset_x,
                self.ownership_bbox.y2 + offset_y,
            ],
        }


@dataclass(slots=True)
class TiledDetectionBatch:
    """Detections plus auditable information about the inference passes."""

    detections: list[Detection]
    tiles: list[InferenceTile] = field(default_factory=list)
    tiling_applied: bool = False
    full_image_pass: bool = False
    raw_detection_count: int = 0
    full_image_detection_count: int = 0
    tile_detection_count: int = 0
    duplicates_removed_count: int = 0
    same_class_duplicates_removed_count: int = 0
    cross_class_conflicts_removed_count: int = 0
    seam_fragments_merged_count: int = 0
    containment_duplicates_removed_count: int = 0
    effective_tile_size: int | None = None

    def metrics(self, offset: tuple[int, int] = (0, 0)) -> dict[str, Any]:
        return {
            "tiling_applied": self.tiling_applied,
            "tile_count": len(self.tiles),
            "effective_tile_size": self.effective_tile_size,
            "full_image_pass": self.full_image_pass,
            "raw_detection_count": self.raw_detection_count,
            "full_image_detection_count": self.full_image_detection_count,
            "tile_detection_count": self.tile_detection_count,
            "duplicates_removed": self.duplicates_removed_count,
            "same_class_duplicates_removed": self.same_class_duplicates_removed_count,
            "cross_class_conflicts_removed": self.cross_class_conflicts_removed_count,
            "seam_fragments_merged": self.seam_fragments_merged_count,
            "containment_duplicates_removed": self.containment_duplicates_removed_count,
            "tile_regions": [tile.to_dict(offset) for tile in self.tiles],
        }


def plan_inference_tiles(
    width: int,
    height: int,
    tile_size: int,
    overlap_ratio: float,
) -> list[InferenceTile]:
    """Cover an image completely with evenly distributed overlapping tiles."""

    width, height = int(width), int(height)
    tile_size = int(tile_size)
    overlap_ratio = float(overlap_ratio)
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive")
    if tile_size < 64:
        raise ValueError("tile_size must be at least 64 pixels")
    if not 0.0 <= overlap_ratio < 0.5:
        raise ValueError("overlap_ratio must satisfy 0 <= overlap_ratio < 0.5")

    x_starts = _axis_starts(width, tile_size, overlap_ratio)
    y_starts = _axis_starts(height, tile_size, overlap_ratio)
    x_ownership = _axis_ownership_ranges(x_starts, width, tile_size)
    y_ownership = _axis_ownership_ranges(y_starts, height, tile_size)
    tiles: list[InferenceTile] = []
    for row, y1 in enumerate(y_starts):
        for column, x1 in enumerate(x_starts):
            x2 = min(width, x1 + tile_size)
            y2 = min(height, y1 + tile_size)
            tiles.append(
                InferenceTile(
                    tile_id=f"tile_r{row:03d}_c{column:03d}",
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    ownership_x1=x_ownership[column][0],
                    ownership_y1=y_ownership[row][0],
                    ownership_x2=x_ownership[column][1],
                    ownership_y2=y_ownership[row][1],
                )
            )
    return tiles


def detect_with_adaptive_tiling(
    detect: Callable[[np.ndarray], Sequence[Detection]],
    image: np.ndarray,
    config: TilingConfig | None = None,
    *,
    tile_detect: Callable[[np.ndarray], Sequence[Detection]] | None = None,
    max_detections: int | None = None,
    frame_id: str = "import_0000",
) -> TiledDetectionBatch:
    """Run full-frame/tile inference and merge results in frame coordinates."""

    policy = config or TilingConfig()
    bgr = ensure_bgr(image)
    height, width = bgr.shape[:2]
    _validate_policy(policy)
    effective_tile_size = _effective_tile_size(width, height, policy)
    tiling_applied = _should_tile(width, height, effective_tile_size, policy)
    tiles = (
        plan_inference_tiles(width, height, effective_tile_size, policy.overlap_ratio)
        if tiling_applied
        else []
    )
    # An image no larger than one tile has no seams and should use one ordinary
    # pass even when the user selected mode="on".
    if len(tiles) <= 1:
        tiling_applied = False
        tiles = []

    raw_detections: list[Detection] = []
    full_count = 0
    tile_count = 0
    run_full_image = not tiling_applied or bool(policy.include_full_image)
    if run_full_image:
        full_detections = list(detect(bgr))
        full_count = len(full_detections)
        for detection in full_detections:
            cloned = _map_detection(
                detection,
                width,
                height,
                offset=(0, 0),
                frame_size=(width, height),
                frame_id=frame_id,
                tile=None,
                policy=policy,
                suffix="full" if tiling_applied else None,
            )
            if cloned is not None:
                raw_detections.append(cloned)

    if tiling_applied:
        detail_detector = tile_detect or detect
        for tile in tiles:
            crop = np.ascontiguousarray(bgr[tile.y1 : tile.y2, tile.x1 : tile.x2])
            tile_detections = list(detail_detector(crop))
            tile_count += len(tile_detections)
            for detection in tile_detections:
                cloned = _map_detection(
                    detection,
                    tile.width,
                    tile.height,
                    offset=(tile.x1, tile.y1),
                    frame_size=(width, height),
                    frame_id=frame_id,
                    tile=tile,
                    policy=policy,
                    suffix=tile.tile_id,
                )
                if cloned is not None:
                    raw_detections.append(cloned)

    (
        merged_before_limit,
        same_class_removed,
        cross_class_removed,
        seam_fragments_merged,
        containment_duplicates_removed,
    ) = _merge_with_stats(
        raw_detections,
        policy,
    )
    merged = (
        merged_before_limit[: max(0, int(max_detections))]
        if max_detections is not None
        else merged_before_limit
    )
    return TiledDetectionBatch(
        detections=merged,
        tiles=tiles,
        tiling_applied=tiling_applied,
        full_image_pass=run_full_image,
        raw_detection_count=len(raw_detections),
        full_image_detection_count=full_count,
        tile_detection_count=tile_count,
        duplicates_removed_count=max(0, len(raw_detections) - len(merged_before_limit)),
        same_class_duplicates_removed_count=same_class_removed,
        cross_class_conflicts_removed_count=cross_class_removed,
        seam_fragments_merged_count=seam_fragments_merged,
        containment_duplicates_removed_count=containment_duplicates_removed,
        effective_tile_size=effective_tile_size if tiling_applied else None,
    )


def merge_tiled_detections(
    detections: Sequence[Detection],
    config: TilingConfig | None = None,
    *,
    max_detections: int | None = None,
) -> list[Detection]:
    """Merge tile-seam fragments, then apply class-aware global NMS."""

    policy = config or TilingConfig()
    kept, _, _, _, _ = _merge_with_stats(detections, policy)
    if max_detections is not None:
        return kept[: max(0, int(max_detections))]
    return kept


def _merge_with_stats(
    detections: Sequence[Detection],
    policy: TilingConfig,
) -> tuple[list[Detection], int, int, int, int]:
    same_threshold = float(np.clip(policy.merge_iou_threshold, 0.0, 1.0))
    cross_threshold = float(np.clip(policy.cross_class_iou_threshold, 0.0, 1.0))
    remaining = sorted(detections, key=lambda item: _merge_score(item, policy), reverse=True)
    kept: list[Detection] = []
    same_class_removed = 0
    cross_class_removed = 0
    seam_fragments_merged = 0
    containment_duplicates_removed = 0
    while remaining:
        current = remaining.pop(0)
        survivors: list[Detection] = []
        for candidate in remaining:
            same_class = _same_detection_class(current, candidate)
            threshold = (
                same_threshold
                if same_class or not policy.class_aware_merge
                else cross_threshold
            )
            if _is_seam_fragment_match(current, candidate, policy, same_class):
                current = _merge_seam_fragments(current, candidate)
                seam_fragments_merged += 1
                if same_class:
                    same_class_removed += 1
                else:
                    cross_class_removed += 1
                continue
            if _is_containment_duplicate(current, candidate, policy, same_class):
                current = _prefer_complete_detection(current, candidate)
                containment_duplicates_removed += 1
                same_class_removed += 1
                continue
            if _intersection_over_union(current.bbox, candidate.bbox) > threshold:
                if same_class:
                    same_class_removed += 1
                else:
                    cross_class_removed += 1
                continue
            survivors.append(candidate)
        kept.append(current)
        remaining = survivors
    return (
        kept,
        same_class_removed,
        cross_class_removed,
        seam_fragments_merged,
        containment_duplicates_removed,
    )


def _axis_starts(length: int, tile_size: int, overlap_ratio: float) -> list[int]:
    if length <= tile_size:
        return [0]
    nominal_stride = tile_size * (1.0 - overlap_ratio)
    step_count = max(1, int(math.ceil((length - tile_size) / nominal_stride)))
    span = length - tile_size
    return [int(round(index * span / step_count)) for index in range(step_count + 1)]


def _axis_ownership_ranges(
    starts: Sequence[int],
    length: int,
    tile_size: int,
) -> list[tuple[float, float]]:
    centers = [
        (float(start) + float(min(length, start + tile_size))) / 2.0
        for start in starts
    ]
    boundaries = [0.0]
    boundaries.extend(
        (centers[index] + centers[index + 1]) / 2.0
        for index in range(len(centers) - 1)
    )
    boundaries.append(float(length))
    return [(boundaries[index], boundaries[index + 1]) for index in range(len(starts))]


def _validate_policy(policy: TilingConfig) -> None:
    if policy.mode not in {"auto", "on", "off"}:
        raise ValueError("tiling mode must be auto, on, or off")
    if policy.tile_size < 64:
        raise ValueError("tile_size must be at least 64 pixels")
    if policy.min_tile_size < 64:
        raise ValueError("min_tile_size must be at least 64 pixels")
    if not 0.25 <= float(policy.detail_window_ratio) <= 1.0:
        raise ValueError("detail_window_ratio must satisfy 0.25 <= value <= 1.0")
    if not 0.0 <= float(policy.overlap_ratio) < 0.5:
        raise ValueError("overlap_ratio must satisfy 0 <= overlap_ratio < 0.5")
    if float(policy.auto_trigger_scale) < 1.0:
        raise ValueError("auto_trigger_scale must be at least 1.0")
    if not 0.0 <= float(policy.edge_margin_ratio) < 0.5:
        raise ValueError("edge_margin_ratio must satisfy 0 <= value < 0.5")
    if not 0.0 <= float(policy.cross_class_iou_threshold) <= 1.0:
        raise ValueError("cross_class_iou_threshold must be between 0 and 1")
    if not 0.0 <= float(policy.seam_ios_threshold) <= 1.0:
        raise ValueError("seam_ios_threshold must be between 0 and 1")
    if not 0.0 <= float(policy.containment_ios_threshold) <= 1.0:
        raise ValueError("containment_ios_threshold must be between 0 and 1")
    if not isinstance(policy.detail_class_confidence, dict):
        raise ValueError("detail_class_confidence must be a dictionary")
    for label, threshold in policy.detail_class_confidence.items():
        if not str(label).strip() or not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(
                "detail_class_confidence needs non-empty labels and thresholds in [0, 1]"
            )
    if policy.detail_confidence is not None and not 0.0 <= float(policy.detail_confidence) <= 1.0:
        raise ValueError("detail_confidence must be between 0 and 1 or None")


def _effective_tile_size(width: int, height: int, policy: TilingConfig) -> int:
    upper = max(64, int(policy.tile_size))
    lower = min(upper, max(64, int(policy.min_tile_size)))
    adaptive = int(round(max(width, height) * float(policy.detail_window_ratio)))
    return int(np.clip(adaptive, lower, upper))


def _should_tile(
    width: int,
    height: int,
    effective_tile_size: int,
    policy: TilingConfig,
) -> bool:
    if policy.mode == "off":
        return False
    if policy.mode == "on":
        return width > effective_tile_size or height > effective_tile_size
    threshold = effective_tile_size * float(policy.auto_trigger_scale)
    return max(width, height) > threshold


def _map_detection(
    detection: Detection,
    local_width: int,
    local_height: int,
    *,
    offset: tuple[int, int],
    frame_size: tuple[int, int],
    frame_id: str,
    tile: InferenceTile | None,
    policy: TilingConfig,
    suffix: str | None,
) -> Detection | None:
    local_bbox = detection.bbox.clamp(local_width, local_height)
    if local_bbox.width <= 0 or local_bbox.height <= 0:
        return None
    offset_x, offset_y = offset
    metadata = dict(detection.metadata)
    metadata.update(
        {
            "frame_id": frame_id,
            "coordinate_space": "frame_pixels",
            "inference_pass": "tile" if tile is not None else "full_image",
            "touches_tile_border": False,
        }
    )
    if tile is not None:
        touches = _touches_internal_tile_border(
            local_bbox,
            tile,
            local_width,
            local_height,
            frame_size,
            policy,
        )
        metadata.update(
            {
                "tile_id": tile.tile_id,
                "tile_offset": [tile.x1, tile.y1],
                "tile_bbox": tile.bbox.as_xyxy(),
                "tile_ownership_bbox": tile.ownership_bbox.as_xyxy(),
                "center_in_tile_ownership": tile.owns_center(
                    local_bbox.translated(offset_x, offset_y)
                ),
                "touches_tile_border": touches,
            }
        )
    detection_id = detection.detection_id
    if suffix:
        metadata["source_detection_id"] = detection_id
        detection_id = f"{detection_id}_{suffix}"
    return Detection(
        label=detection.label,
        confidence=detection.confidence,
        bbox=local_bbox.translated(offset_x, offset_y),
        class_id=detection.class_id,
        source=detection.source,
        detection_id=detection_id,
        metadata=metadata,
    )


def _touches_internal_tile_border(
    bbox: BoundingBox,
    tile: InferenceTile,
    local_width: int,
    local_height: int,
    frame_size: tuple[int, int],
    policy: TilingConfig,
) -> bool:
    margin = max(2.0, min(local_width, local_height) * float(policy.edge_margin_ratio))
    frame_width, frame_height = frame_size
    return bool(
        (tile.x1 > 0 and bbox.x1 <= margin)
        or (tile.y1 > 0 and bbox.y1 <= margin)
        or (tile.x2 < frame_width and bbox.x2 >= local_width - margin)
        or (tile.y2 < frame_height and bbox.y2 >= local_height - margin)
    )


def _merge_score(detection: Detection, policy: TilingConfig) -> float:
    penalty = (
        float(policy.edge_confidence_penalty)
        if detection.metadata.get("touches_tile_border")
        else 0.0
    )
    if (
        detection.metadata.get("inference_pass") == "tile"
        and not detection.metadata.get("center_in_tile_ownership", True)
    ):
        penalty += float(policy.non_ownership_confidence_penalty)
    return float(detection.confidence) - penalty


def _same_detection_class(first: Detection, second: Detection) -> bool:
    if first.class_id is not None and second.class_id is not None:
        return first.class_id == second.class_id
    return first.label == second.label


def _is_seam_fragment_match(
    first: Detection,
    second: Detection,
    policy: TilingConfig,
    same_class: bool,
) -> bool:
    """Match partial hypotheses produced by two overlapping tile borders.

    Global NMS uses IoU, which becomes artificially small when each tile only
    sees a different part of one component. Restricting the more permissive
    IoS metric to border detections from distinct tiles avoids changing normal
    full-image NMS behavior.
    """

    first_metadata = first.metadata
    second_metadata = second.metadata
    if (
        first_metadata.get("inference_pass") != "tile"
        or second_metadata.get("inference_pass") != "tile"
        or not first_metadata.get("touches_tile_border")
        or not second_metadata.get("touches_tile_border")
    ):
        return False
    first_tile = first_metadata.get("tile_id")
    second_tile = second_metadata.get("tile_id")
    if not first_tile or not second_tile or first_tile == second_tile:
        return False
    first_frame = first_metadata.get("frame_id")
    second_frame = second_metadata.get("frame_id")
    if first_frame is not None and second_frame is not None and first_frame != second_frame:
        return False

    threshold = float(np.clip(policy.seam_ios_threshold, 0.0, 1.0))
    if policy.class_aware_merge and not same_class:
        threshold = max(
            threshold,
            float(np.clip(policy.cross_class_iou_threshold, 0.0, 1.0)),
        )
    return _intersection_over_smaller(first.bbox, second.bbox) >= threshold


def _merge_seam_fragments(primary: Detection, fragment: Detection) -> Detection:
    """Keep the best hypothesis while expanding its box over both fragments."""

    bbox = BoundingBox(
        min(primary.bbox.x1, fragment.bbox.x1),
        min(primary.bbox.y1, fragment.bbox.y1),
        max(primary.bbox.x2, fragment.bbox.x2),
        max(primary.bbox.y2, fragment.bbox.y2),
    )
    metadata = dict(primary.metadata)
    metadata.update(
        {
            "seam_fragments_merged": True,
            "merged_from_detection_ids": _merged_metadata_values(
                primary,
                fragment,
                key="merged_from_detection_ids",
                fallback=lambda item: item.detection_id,
            ),
            "merged_from_tile_ids": _merged_metadata_values(
                primary,
                fragment,
                key="merged_from_tile_ids",
                fallback=lambda item: item.metadata.get("tile_id"),
            ),
        }
    )
    return Detection(
        label=primary.label,
        confidence=primary.confidence,
        bbox=bbox,
        class_id=primary.class_id,
        source=primary.source,
        detection_id=primary.detection_id,
        metadata=metadata,
    )


def _is_containment_duplicate(
    first: Detection,
    second: Detection,
    policy: TilingConfig,
    same_class: bool,
) -> bool:
    """Identify a complete box plus a nested partial box from another pass/tile."""

    if not same_class:
        return False
    first_metadata = first.metadata
    second_metadata = second.metadata
    first_frame = first_metadata.get("frame_id")
    second_frame = second_metadata.get("frame_id")
    if first_frame is not None and second_frame is not None and first_frame != second_frame:
        return False
    first_origin = (
        first_metadata.get("inference_pass"),
        first_metadata.get("tile_id"),
    )
    second_origin = (
        second_metadata.get("inference_pass"),
        second_metadata.get("tile_id"),
    )
    if first_origin == second_origin:
        return False
    threshold = float(np.clip(policy.containment_ios_threshold, 0.0, 1.0))
    return _intersection_over_smaller(first.bbox, second.bbox) >= threshold


def _prefer_complete_detection(first: Detection, second: Detection) -> Detection:
    """Prefer a non-edge, owner-region, complete hypothesis over a partial box."""

    def priority(item: Detection) -> tuple[int, int, float, float]:
        return (
            int(not bool(item.metadata.get("touches_tile_border"))),
            int(bool(item.metadata.get("center_in_tile_ownership", True))),
            float(item.bbox.area),
            float(item.confidence),
        )

    kept, removed = (first, second) if priority(first) >= priority(second) else (second, first)
    metadata = dict(kept.metadata)
    suppressed = list(metadata.get("contained_detection_ids", []))
    removed_ids = [removed.detection_id]
    stored_removed_ids = removed.metadata.get("contained_detection_ids", [])
    if isinstance(stored_removed_ids, list):
        removed_ids.extend(stored_removed_ids)
    for detection_id in removed_ids:
        if detection_id not in suppressed:
            suppressed.append(detection_id)
    metadata["contained_detection_ids"] = suppressed
    return Detection(
        label=kept.label,
        confidence=kept.confidence,
        bbox=kept.bbox,
        class_id=kept.class_id,
        source=kept.source,
        detection_id=kept.detection_id,
        metadata=metadata,
    )


def _merged_metadata_values(
    first: Detection,
    second: Detection,
    *,
    key: str,
    fallback: Callable[[Detection], Any],
) -> list[Any]:
    values: list[Any] = []
    for detection in (first, second):
        stored = detection.metadata.get(key)
        candidates = stored if isinstance(stored, list) else [fallback(detection)]
        for candidate in candidates:
            if candidate is not None and candidate not in values:
                values.append(candidate)
    return values


_intersection_over_union = intersection_over_union


_intersection_over_smaller = intersection_over_smaller
