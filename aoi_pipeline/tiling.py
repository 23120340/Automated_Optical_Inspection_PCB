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
from .models import BoundingBox, Detection


@dataclass(frozen=True, slots=True)
class InferenceTile:
    """One right/bottom-exclusive rectangular inference window."""

    tile_id: str
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def bbox(self) -> BoundingBox:
        return BoundingBox(float(self.x1), float(self.y1), float(self.x2), float(self.y2))

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

    def metrics(self, offset: tuple[int, int] = (0, 0)) -> dict[str, Any]:
        return {
            "tiling_applied": self.tiling_applied,
            "tile_count": len(self.tiles),
            "full_image_pass": self.full_image_pass,
            "raw_detection_count": self.raw_detection_count,
            "full_image_detection_count": self.full_image_detection_count,
            "tile_detection_count": self.tile_detection_count,
            "duplicates_removed": self.duplicates_removed_count,
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
                )
            )
    return tiles


def detect_with_adaptive_tiling(
    detect: Callable[[np.ndarray], Sequence[Detection]],
    image: np.ndarray,
    config: TilingConfig | None = None,
    *,
    max_detections: int | None = None,
    frame_id: str = "import_0000",
) -> TiledDetectionBatch:
    """Run full-frame/tile inference and merge results in frame coordinates."""

    policy = config or TilingConfig()
    bgr = ensure_bgr(image)
    height, width = bgr.shape[:2]
    _validate_policy(policy)
    tiling_applied = _should_tile(width, height, policy)
    tiles = (
        plan_inference_tiles(width, height, policy.tile_size, policy.overlap_ratio)
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
        for tile in tiles:
            crop = np.ascontiguousarray(bgr[tile.y1 : tile.y2, tile.x1 : tile.x2])
            tile_detections = list(detect(crop))
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

    merged_before_limit = merge_tiled_detections(raw_detections, policy)
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
    )


def merge_tiled_detections(
    detections: Sequence[Detection],
    config: TilingConfig | None = None,
    *,
    max_detections: int | None = None,
) -> list[Detection]:
    """Class-aware global NMS with a small penalty for internal tile seams."""

    policy = config or TilingConfig()
    threshold = float(np.clip(policy.merge_iou_threshold, 0.0, 1.0))
    remaining = sorted(detections, key=lambda item: _merge_score(item, policy), reverse=True)
    kept: list[Detection] = []
    while remaining:
        current = remaining.pop(0)
        kept.append(current)
        remaining = [
            candidate
            for candidate in remaining
            if not (
                _same_merge_class(current, candidate, policy.class_aware_merge)
                and _intersection_over_union(current.bbox, candidate.bbox) > threshold
            )
        ]
    if max_detections is not None:
        return kept[: max(0, int(max_detections))]
    return kept


def _axis_starts(length: int, tile_size: int, overlap_ratio: float) -> list[int]:
    if length <= tile_size:
        return [0]
    nominal_stride = tile_size * (1.0 - overlap_ratio)
    step_count = max(1, int(math.ceil((length - tile_size) / nominal_stride)))
    span = length - tile_size
    return [int(round(index * span / step_count)) for index in range(step_count + 1)]


def _validate_policy(policy: TilingConfig) -> None:
    if policy.mode not in {"auto", "on", "off"}:
        raise ValueError("tiling mode must be auto, on, or off")
    if policy.tile_size < 64:
        raise ValueError("tile_size must be at least 64 pixels")
    if not 0.0 <= float(policy.overlap_ratio) < 0.5:
        raise ValueError("overlap_ratio must satisfy 0 <= overlap_ratio < 0.5")
    if float(policy.auto_trigger_scale) < 1.0:
        raise ValueError("auto_trigger_scale must be at least 1.0")
    if not 0.0 <= float(policy.edge_margin_ratio) < 0.5:
        raise ValueError("edge_margin_ratio must satisfy 0 <= value < 0.5")


def _should_tile(width: int, height: int, policy: TilingConfig) -> bool:
    if policy.mode == "off":
        return False
    if policy.mode == "on":
        return width > policy.tile_size or height > policy.tile_size
    threshold = policy.tile_size * float(policy.auto_trigger_scale)
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
    return float(detection.confidence) - penalty


def _same_merge_class(first: Detection, second: Detection, class_aware: bool) -> bool:
    if not class_aware:
        return True
    if first.class_id is not None and second.class_id is not None:
        return first.class_id == second.class_id
    return first.label == second.label


def _intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    x1, y1 = max(first.x1, second.x1), max(first.y1, second.y1)
    x2, y2 = min(first.x2, second.x2), min(first.y2, second.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = first.area + second.area - intersection
    return intersection / union if union > 0 else 0.0
