"""Runtime models shared by the local app and Kaggle-facing code."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import numpy as np


def utc_now_iso() -> str:
    """Return a timezone-aware timestamp that is stable in JSON exports."""

    return datetime.now(timezone.utc).isoformat()


def shape_dict(image: np.ndarray | None) -> dict[str, int] | None:
    if image is None:
        return None
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else int(image.shape[2])
    return {"height": int(height), "width": int(width), "channels": channels}


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned box in ``xyxy`` pixel coordinates (right/bottom exclusive)."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Bounding-box coordinates must be finite")
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("Bounding box must satisfy x2 >= x1 and y2 >= y1")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_xyxy(self) -> list[float]:
        return [float(self.x1), float(self.y1), float(self.x2), float(self.y2)]

    def tolist(self) -> list[float]:
        """NumPy-like alias used by lightweight UI adapters."""

        return self.as_xyxy()

    def as_xywh(self) -> list[float]:
        return [float(self.x1), float(self.y1), float(self.width), float(self.height)]

    def to_int(self) -> tuple[int, int, int, int]:
        return (
            int(np.floor(self.x1)),
            int(np.floor(self.y1)),
            int(np.ceil(self.x2)),
            int(np.ceil(self.y2)),
        )

    def clamp(self, width: int, height: int) -> BoundingBox:
        return BoundingBox(
            min(max(self.x1, 0.0), float(width)),
            min(max(self.y1, 0.0), float(height)),
            min(max(self.x2, 0.0), float(width)),
            min(max(self.y2, 0.0), float(height)),
        )

    def translated(self, dx: float, dy: float) -> BoundingBox:
        return BoundingBox(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)

    def to_dict(self) -> dict[str, Any]:
        return {"xyxy": self.as_xyxy(), "xywh": self.as_xywh(), "area": float(self.area)}


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float
    bbox: BoundingBox
    class_id: int | None = None
    source: str = "unknown"
    detection_id: str = field(default_factory=lambda: f"det_{uuid4().hex[:12]}")
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("Detection confidence must be between 0 and 1")
        if not self.label:
            raise ValueError("Detection label cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.detection_id,
            "label": self.label,
            "class_id": self.class_id,
            "confidence": float(self.confidence),
            "bbox": self.bbox.to_dict(),
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class PreprocessResult:
    image: np.ndarray
    input_shape: tuple[int, ...]
    operations: list[str] = field(default_factory=list)
    scale: float = 1.0
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_shape": list(self.input_shape),
            "output_shape": shape_dict(self.image),
            "operations": self.operations,
            "scale": float(self.scale),
            "warnings": self.warnings,
            "metrics": self.metrics,
        }


@dataclass(slots=True)
class AlignmentResult:
    image: np.ndarray
    method: str
    success: bool
    homography: np.ndarray | None = None
    source_keypoints: int = 0
    reference_keypoints: int = 0
    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    correlation: float | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "success": bool(self.success),
            "output_shape": shape_dict(self.image),
            "homography": self.homography.tolist() if self.homography is not None else None,
            "source_keypoints": int(self.source_keypoints),
            "reference_keypoints": int(self.reference_keypoints),
            "good_matches": int(self.good_matches),
            "inliers": int(self.inliers),
            "inlier_ratio": float(self.inlier_ratio),
            "correlation": None if self.correlation is None else float(self.correlation),
            "message": self.message,
        }


@dataclass(slots=True)
class BoardRegion:
    bbox: BoundingBox
    polygon: list[tuple[float, float]]
    confidence: float
    method: str
    mask: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox.to_dict(),
            "polygon": [[float(x), float(y)] for x, y in self.polygon],
            "confidence": float(self.confidence),
            "method": self.method,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ComponentCrop:
    image: np.ndarray
    detection_id: str
    label: str
    confidence: float
    source_bbox: BoundingBox
    crop_bbox: BoundingBox
    filename: str
    path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "label": self.label,
            "confidence": float(self.confidence),
            "source_bbox": self.source_bbox.to_dict(),
            "crop_bbox": self.crop_bbox.to_dict(),
            "filename": self.filename,
            # Export only the portable filename, not an absolute workstation path.
            "path": self.path.name if self.path is not None else None,
            "image_shape": shape_dict(self.image),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class PipelineRun:
    source_name: str
    input_image: np.ndarray
    preprocess_result: PreprocessResult
    alignment_result: AlignmentResult
    board_region: BoardRegion
    detections: list[Detection]
    crops: list[ComponentCrop]
    run_id: str = field(default_factory=lambda: f"run_{uuid4().hex[:12]}")
    started_at: str = field(default_factory=utc_now_iso)
    finished_at: str = field(default_factory=utc_now_iso)
    warnings: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def final_image(self) -> np.ndarray:
        return self.alignment_result.image

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "source_name": self.source_name,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "input_shape": shape_dict(self.input_image),
            "preprocess": self.preprocess_result.to_dict(),
            "alignment": self.alignment_result.to_dict(),
            "board": self.board_region.to_dict(),
            "detections": [detection.to_dict() for detection in self.detections],
            "crops": [crop.to_dict() for crop in self.crops],
            "summary": {
                "component_count": len(self.detections),
                "crop_count": len(self.crops),
                "labels": _count_labels(self.detections),
            },
            "warnings": self.warnings,
            "config": self.config,
        }


def _count_labels(detections: Sequence[Detection]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for detection in detections:
        counts[detection.label] = counts.get(detection.label, 0) + 1
    return dict(sorted(counts.items()))
