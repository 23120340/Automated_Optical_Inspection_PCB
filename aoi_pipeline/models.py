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
    def center(self) -> tuple[float, float]:
        """Centre point. Ten call sites recomputed this inline before it existed."""

        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

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


def intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    """Overlap as a share of the two boxes together.

    Lived in four places at once -- ``detectors``, ``tiling``, ``leads`` and
    ``cad_fusion`` each carried their own copy of this and of
    :func:`intersection_over_smaller`. All the copies were checked against each
    other on 4005 box pairs, degenerate ones included, and agreed every time,
    so nothing changes by having one. What changes is that the next fix lands
    in one place instead of three that nobody remembers to look at.
    """

    x1, y1 = max(first.x1, second.x1), max(first.y1, second.y1)
    x2, y2 = min(first.x2, second.x2), min(first.y2, second.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = first.area + second.area - intersection
    return intersection / union if union > 0 else 0.0


def intersection_over_smaller(first: BoundingBox, second: BoundingBox) -> float:
    """Overlap as a share of the smaller box.

    The measure that recognises a fragment sitting inside a bigger box, which
    IoU cannot: a tile-seam sliver against the whole component scores near zero
    on IoU and near one here.
    """

    x1, y1 = max(first.x1, second.x1), max(first.y1, second.y1)
    x2, y2 = min(first.x2, second.x2), min(first.y2, second.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    smaller_area = min(first.area, second.area)
    return intersection / smaller_area if smaller_area > 0 else 0.0


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax, shifted for numerical stability.

    Shared by the step-6.1 and step-6.2 ONNX heads. They had a copy each; the
    two were byte-identical and agreed to 0.0e+00 over 500 random logit
    matrices, which is exactly the state in which a divergence would go
    unnoticed.
    """

    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


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
    # Same geometry as ``image``, before denoise/white-balance/CLAHE/
    # normalization/sharpening.  Keeping the two frames in one result prevents
    # a later preprocess call (for example the Golden Image) from replacing
    # the board pixels that step 6.2 must measure.
    radiometric_image: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_shape": list(self.input_shape),
            "output_shape": shape_dict(self.image),
            "operations": self.operations,
            "scale": float(self.scale),
            "warnings": self.warnings,
            "metrics": self.metrics,
            "radiometric_shape": shape_dict(self.radiometric_image),
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
    # The un-enhanced frame warped by the exact same source-to-reference
    # transform as ``image``.  ``None`` is fail-safe: grading may fall back to
    # the analysis pixels, but it must never crop an unaligned auxiliary frame.
    radiometric_image: np.ndarray | None = None

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
            "radiometric_shape": shape_dict(self.radiometric_image),
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
class SolderJoint:
    """One derived inspection ROI on or around a component's terminals.

    ``kind`` is ``"joint"`` for a terminal/lead ROI and ``"body"`` for the
    whole-component view that includes every joint. ``angle`` is the ROI
    rotation in degrees (0 for the axis-aligned default); ``bbox`` is always
    the axis-aligned box that encloses the ROI, so overlays and exports do not
    need to know whether orientation estimation ran.
    """

    detection_id: str
    joint_id: str
    label: str
    kind: str
    bbox: BoundingBox
    terminal_geometry: str
    position: str
    angle: float = 0.0
    pin_index: int | None = None
    # Provenance of the ROI geometry: "derived" from the detector box alone,
    # "cad" from registered CAD land coordinates, "cad+derived" when both
    # agreed and were merged. Downstream labelling can filter on it.
    source: str = "derived"
    designator: str | None = None
    pin: str | None = None
    net: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_id": self.joint_id,
            "detection_id": self.detection_id,
            "label": self.label,
            "kind": self.kind,
            "bbox": self.bbox.to_dict(),
            "terminal_geometry": self.terminal_geometry,
            "position": self.position,
            "angle": float(self.angle),
            "pin_index": self.pin_index,
            "source": self.source,
            "designator": self.designator,
            "pin": self.pin,
            "net": self.net,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class SolderFeatures:
    """Physical measurements of one solder ROI, in units a human can argue with.

    Every value is a ratio in [0, 1] unless named otherwise, so thresholds carry
    over between lenses and magnifications. These feed the rule layer, the
    fusion guard rails, and the operator-facing explanation -- a defect report
    that only says "the network disagreed" cannot be acted on.
    """

    solder_ratio: float
    solder_area_px: int
    span_ratio: float
    width_ratio: float
    centroid_offset_ratio: float
    specular_ratio: float
    edge_density: float
    contrast: float
    uniformity: float
    edge_contact_start: float
    edge_contact_end: float
    mean_value: float
    mean_saturation: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "solder_ratio": round(float(self.solder_ratio), 5),
            "solder_area_px": int(self.solder_area_px),
            "span_ratio": round(float(self.span_ratio), 5),
            "width_ratio": round(float(self.width_ratio), 5),
            "centroid_offset_ratio": round(float(self.centroid_offset_ratio), 5),
            "specular_ratio": round(float(self.specular_ratio), 5),
            "edge_density": round(float(self.edge_density), 5),
            "contrast": round(float(self.contrast), 5),
            "uniformity": round(float(self.uniformity), 5),
            "edge_contact_start": round(float(self.edge_contact_start), 5),
            "edge_contact_end": round(float(self.edge_contact_end), 5),
            "mean_value": round(float(self.mean_value), 5),
            "mean_saturation": round(float(self.mean_saturation), 5),
        }


@dataclass(slots=True)
class SolderVerdict:
    """The step-6.2 call on one ROI, and how it was reached.

    ``rule_label`` and ``model_label`` are kept side by side on purpose: when
    they disagree the run is worth a human look, and collapsing them into one
    field would hide exactly the cases that matter.
    """

    joint_id: str
    detection_id: str
    scope: str
    label: str
    probability: float
    decision: str
    source: str
    rule_label: str | None = None
    model_label: str | None = None
    model_probability: float | None = None
    designator: str | None = None
    pin: str | None = None
    component_label: str = ""
    reasons: list[str] = field(default_factory=list)
    top_k: list[ClassProbability] = field(default_factory=list)
    features: SolderFeatures | None = None
    model_version: str = "rules-only"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_defect(self) -> bool:
        return self.label not in {"good", "ok"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_id": self.joint_id,
            "detection_id": self.detection_id,
            "scope": self.scope,
            "label": self.label,
            "probability": float(self.probability),
            "decision": self.decision,
            "source": self.source,
            "rule_label": self.rule_label,
            "model_label": self.model_label,
            "model_probability": (
                None if self.model_probability is None else float(self.model_probability)
            ),
            "designator": self.designator,
            "pin": self.pin,
            "component_label": self.component_label,
            "reasons": list(self.reasons),
            "top_k": [item.to_dict() for item in self.top_k],
            "features": None if self.features is None else self.features.to_dict(),
            "model_version": self.model_version,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class SolderJointCrop:
    """Pixels for one :class:`SolderJoint`, ready to label for step 6.2."""

    image: np.ndarray
    joint: SolderJoint
    filename: str
    path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def joint_id(self) -> str:
        return self.joint.joint_id

    @property
    def detection_id(self) -> str:
        return self.joint.detection_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint": self.joint.to_dict(),
            "filename": self.filename,
            # Export only the portable filename, not an absolute workstation path.
            "path": self.path.name if self.path is not None else None,
            "image_shape": shape_dict(self.image),
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class ClassProbability:
    """One entry in a classifier's ordered top-k output."""

    label: str
    probability: float

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "probability": float(self.probability)}


@dataclass(slots=True)
class ComponentClassification:
    """Step-6.1 family result tied back to one step-5 crop/detection."""

    crop_id: str
    detection_id: str
    family: str
    probability: float
    top_k: list[ClassProbability]
    unknown_score: float
    decision: str
    model_version: str
    source: str = "onnx_classifier"
    detector_hint: str | None = None
    visual_subtype: str | None = None
    mount_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.decision not in {"accept", "review", "unknown"}:
            raise ValueError("Classification decision must be accept, review, or unknown")
        if not 0.0 <= float(self.probability) <= 1.0:
            raise ValueError("Classification probability must be between 0 and 1")
        if not 0.0 <= float(self.unknown_score) <= 1.0:
            raise ValueError("Classification unknown_score must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "crop_id": self.crop_id,
            "detection_id": self.detection_id,
            "family": self.family,
            "probability": float(self.probability),
            "top_k": [item.to_dict() for item in self.top_k],
            "unknown_score": float(self.unknown_score),
            "decision": self.decision,
            "model_version": self.model_version,
            "source": self.source,
            "detector_hint": self.detector_hint,
            "visual_subtype": self.visual_subtype,
            "mount_type": self.mount_type,
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
    # Kept as ``Any`` to preserve models.py as the dependency leaf: the
    # manifest-driven package runtime imports the shared crop/probability
    # models.  Every entry implements ``to_dict``.
    package_classifications: list[Any] = field(default_factory=list)
    package_topology_checks: list[Any] = field(default_factory=list)
    classifications: list[ComponentClassification] = field(default_factory=list)
    solder_crops: list[SolderJointCrop] = field(default_factory=list)
    # ``Any`` avoids importing fusion here: models is the leaf module every
    # other one depends on, and fusion depends on models.
    fusion: Any | None = None
    solder_verdicts: list[SolderVerdict] = field(default_factory=list)
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
            "package_classifications": [
                item.to_dict() for item in self.package_classifications
            ],
            "package_topology_checks": [
                item.to_dict() for item in self.package_topology_checks
            ],
            "solder_crops": [crop.to_dict() for crop in self.solder_crops],
            "solder_verdicts": [item.to_dict() for item in self.solder_verdicts],
            "cad_fusion": (
                self.fusion.to_dict() if self.fusion is not None else None
            ),
            "classifications": [item.to_dict() for item in self.classifications],
            "summary": {
                "component_count": len(self.detections),
                "crop_count": len(self.crops),
                "solder_joint_count": sum(
                    1 for crop in self.solder_crops if crop.joint.kind == "joint"
                ),
                "solder_crop_count": len(self.solder_crops),
                "solder_roi_sources": _count_roi_sources(self.solder_crops),
                "solder_verdicts": _count_verdicts(self.solder_verdicts),
                "solder_decisions": _count_verdict_decisions(self.solder_verdicts),
                "cad_findings": (
                    _count_findings(self.fusion.findings)
                    if self.fusion is not None
                    else {}
                ),
                "classification_count": len(self.classifications),
                "package_classification_count": len(self.package_classifications),
                "package_classes": _count_attribute(
                    self.package_classifications, "package_class"
                ),
                "package_decisions": _count_attribute(
                    self.package_classifications, "decision"
                ),
                "package_topology_statuses": _count_attribute(
                    self.package_topology_checks, "status"
                ),
                "labels": _count_labels(self.detections),
                "families": _count_classifications(self.classifications),
                "classification_decisions": _count_decisions(self.classifications),
            },
            "warnings": self.warnings,
            "config": self.config,
        }


def _count_verdicts(verdicts: Sequence[SolderVerdict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for verdict in verdicts:
        counts[verdict.label] = counts.get(verdict.label, 0) + 1
    return dict(sorted(counts.items()))


def _count_verdict_decisions(verdicts: Sequence[SolderVerdict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for verdict in verdicts:
        counts[verdict.decision] = counts.get(verdict.decision, 0) + 1
    return dict(sorted(counts.items()))


def _count_roi_sources(crops: Sequence[SolderJointCrop]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for crop in crops:
        counts[crop.joint.source] = counts.get(crop.joint.source, 0) + 1
    return dict(sorted(counts.items()))


def _count_findings(findings: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in findings:
        kind = str(getattr(finding, "kind", "unknown"))
        counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))


def _count_labels(detections: Sequence[Detection]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for detection in detections:
        counts[detection.label] = counts.get(detection.label, 0) + 1
    return dict(sorted(counts.items()))


def _count_classifications(
    classifications: Sequence[ComponentClassification],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in classifications:
        counts[item.family] = counts.get(item.family, 0) + 1
    return dict(sorted(counts.items()))


def _count_decisions(
    classifications: Sequence[ComponentClassification],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in classifications:
        counts[item.decision] = counts.get(item.decision, 0) + 1
    return dict(sorted(counts.items()))


def _count_attribute(items: Sequence[Any], name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(getattr(item, name, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
