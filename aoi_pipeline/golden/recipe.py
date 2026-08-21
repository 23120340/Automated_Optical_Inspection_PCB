"""Versioned Golden inspection recipe contracts and enrollment helpers.

All boxes use ``xyxy`` coordinates with exclusive right/bottom edges. Recipe
geometry exists only in ``golden_board_pixels``. Golden and slot assets retain
the native uint8 BGR measurement-image resolution; classifier crops are never
used here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from ..exceptions import RecipeValidationError
from ..models import BoundingBox, Detection


RECIPE_SCHEMA_VERSION = "aoi-inspection-recipe/1.1"
GOLDEN_COORDINATE_SPACE = "golden_board_pixels"
DEMO_BOARD_ID = "demo_board"
DEFAULT_BOARD_SIDE = "top"
APPROVED_ANCHOR_PROVENANCE = "approved_stable_features"
DEMO_GRID_ANCHOR_PROVENANCE = "demo_grid"


@dataclass(frozen=True, slots=True)
class ImageSize:
    width: int
    height: int

    def __post_init__(self) -> None:
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise RecipeValidationError("Recipe image width and height must be positive")

    def to_dict(self) -> dict[str, int]:
        return {"width": int(self.width), "height": int(self.height)}

    @classmethod
    def from_mapping(cls, value: object) -> ImageSize:
        mapping = _mapping(value, "image_size")
        return cls(_integer(mapping, "width"), _integer(mapping, "height"))


@dataclass(frozen=True, slots=True)
class MetrologyCalibration:
    """Axis-aligned calibration for canonical Golden pixels to millimetres."""

    pixels_per_mm_x: float
    pixels_per_mm_y: float
    verified: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.verified, bool):
            raise RecipeValidationError("verified must be a boolean")
        for name, value in (
            ("pixels_per_mm_x", self.pixels_per_mm_x),
            ("pixels_per_mm_y", self.pixels_per_mm_y),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise RecipeValidationError(f"{name} must be a positive finite value")

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "pixels_per_mm_x": float(self.pixels_per_mm_x),
            "pixels_per_mm_y": float(self.pixels_per_mm_y),
            "verified": bool(self.verified),
        }

    @classmethod
    def from_mapping(cls, value: object) -> MetrologyCalibration:
        mapping = _mapping(value, "metrology")
        return cls(
            _number(mapping, "pixels_per_mm_x"),
            _number(mapping, "pixels_per_mm_y"),
            verified=_boolean(mapping, "verified"),
        )


@dataclass(frozen=True, slots=True)
class PositionTolerance:
    max_abs_dx_mm: float
    max_abs_dy_mm: float
    max_abs_angle_deg: float | None = None

    def __post_init__(self) -> None:
        _nonnegative_finite(self.max_abs_dx_mm, "max_abs_dx_mm")
        _nonnegative_finite(self.max_abs_dy_mm, "max_abs_dy_mm")
        if self.max_abs_angle_deg is not None:
            _nonnegative_finite(self.max_abs_angle_deg, "max_abs_angle_deg")

    def to_dict(self) -> dict[str, float | None]:
        return {
            "max_abs_dx_mm": float(self.max_abs_dx_mm),
            "max_abs_dy_mm": float(self.max_abs_dy_mm),
            "max_abs_angle_deg": None if self.max_abs_angle_deg is None else float(self.max_abs_angle_deg),
        }

    @classmethod
    def from_mapping(cls, value: object) -> PositionTolerance:
        mapping = _mapping(value, "position_tolerance")
        angle = mapping.get("max_abs_angle_deg")
        return cls(
            _number(mapping, "max_abs_dx_mm"),
            _number(mapping, "max_abs_dy_mm"),
            None if angle is None else _finite_number(angle, "max_abs_angle_deg"),
        )


@dataclass(frozen=True, slots=True)
class AppearanceThresholds:
    min_ssim: float
    max_diff_ratio: float
    max_edge_diff_ratio: float
    max_blob_area_px: int
    min_valid_overlap_ratio: float

    def __post_init__(self) -> None:
        for name, value in (
            ("min_ssim", self.min_ssim),
            ("max_diff_ratio", self.max_diff_ratio),
            ("max_edge_diff_ratio", self.max_edge_diff_ratio),
            ("min_valid_overlap_ratio", self.min_valid_overlap_ratio),
        ):
            numeric = float(value)
            if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                raise RecipeValidationError(f"{name} must be between 0 and 1")
        if int(self.max_blob_area_px) < 0:
            raise RecipeValidationError("max_blob_area_px must be non-negative")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "min_ssim": float(self.min_ssim),
            "max_diff_ratio": float(self.max_diff_ratio),
            "max_edge_diff_ratio": float(self.max_edge_diff_ratio),
            "max_blob_area_px": int(self.max_blob_area_px),
            "min_valid_overlap_ratio": float(self.min_valid_overlap_ratio),
        }

    @classmethod
    def from_mapping(cls, value: object) -> AppearanceThresholds:
        mapping = _mapping(value, "appearance_thresholds")
        return cls(
            min_ssim=_number(mapping, "min_ssim"),
            max_diff_ratio=_number(mapping, "max_diff_ratio"),
            max_edge_diff_ratio=_number(mapping, "max_edge_diff_ratio"),
            max_blob_area_px=_integer(mapping, "max_blob_area_px"),
            min_valid_overlap_ratio=_number(mapping, "min_valid_overlap_ratio"),
        )


@dataclass(frozen=True, slots=True)
class AlignmentQualityGates:
    min_anchors: int = 3
    min_anchor_score: float = 0.70
    max_residual_px: float = 0.5
    ransac_reprojection_threshold_px: float = 1.0
    min_inlier_ratio: float = 0.75
    min_scale: float = 0.5
    max_scale: float = 2.0
    max_abs_rotation_deg: float = 30.0
    min_canvas_overlap_ratio: float = 0.80

    def __post_init__(self) -> None:
        if int(self.min_anchors) < 2:
            raise RecipeValidationError("Alignment min_anchors must be at least 2")
        if not 0.0 <= float(self.min_anchor_score) <= 1.0:
            raise RecipeValidationError("Alignment min_anchor_score must be between 0 and 1")
        if not math.isfinite(float(self.max_residual_px)) or self.max_residual_px <= 0:
            raise RecipeValidationError("Alignment max_residual_px must be positive")
        if (
            not math.isfinite(float(self.ransac_reprojection_threshold_px))
            or self.ransac_reprojection_threshold_px <= 0
        ):
            raise RecipeValidationError(
                "Alignment ransac_reprojection_threshold_px must be positive"
            )
        if not 0.0 <= float(self.min_inlier_ratio) <= 1.0:
            raise RecipeValidationError("Alignment min_inlier_ratio must be between 0 and 1")
        if self.min_scale <= 0 or self.max_scale < self.min_scale:
            raise RecipeValidationError("Alignment scale gates are invalid")
        if not 0.0 <= float(self.max_abs_rotation_deg) <= 180.0:
            raise RecipeValidationError("Alignment max_abs_rotation_deg must be in [0, 180]")
        if not 0.0 <= float(self.min_canvas_overlap_ratio) <= 1.0:
            raise RecipeValidationError(
                "Alignment min_canvas_overlap_ratio must be between 0 and 1"
            )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "min_anchors": int(self.min_anchors),
            "min_anchor_score": float(self.min_anchor_score),
            "max_residual_px": float(self.max_residual_px),
            "ransac_reprojection_threshold_px": float(
                self.ransac_reprojection_threshold_px
            ),
            "min_inlier_ratio": float(self.min_inlier_ratio),
            "min_scale": float(self.min_scale),
            "max_scale": float(self.max_scale),
            "max_abs_rotation_deg": float(self.max_abs_rotation_deg),
            "min_canvas_overlap_ratio": float(self.min_canvas_overlap_ratio),
        }

    @classmethod
    def from_mapping(cls, value: object) -> AlignmentQualityGates:
        mapping = _mapping(value, "alignment.quality_gates")
        return cls(
            min_anchors=_integer(mapping, "min_anchors"),
            min_anchor_score=_number(mapping, "min_anchor_score"),
            max_residual_px=_number(mapping, "max_residual_px"),
            ransac_reprojection_threshold_px=_number(
                mapping, "ransac_reprojection_threshold_px"
            ),
            min_inlier_ratio=_number(mapping, "min_inlier_ratio"),
            min_scale=_number(mapping, "min_scale"),
            max_scale=_number(mapping, "max_scale"),
            max_abs_rotation_deg=_number(mapping, "max_abs_rotation_deg"),
            # A hand-edited quality-gate object missing this field remains
            # fail-closed through the conservative runtime default.
            min_canvas_overlap_ratio=_finite_number(
                mapping.get("min_canvas_overlap_ratio", 0.80),
                "min_canvas_overlap_ratio",
            ),
        )


@dataclass(frozen=True, slots=True)
class AlignmentAnchor:
    anchor_id: str
    reference_point_px: tuple[float, float]
    template_bbox_xyxy: BoundingBox
    search_roi_xyxy: BoundingBox
    template_path: str
    mask_path: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.anchor_id, "anchor_id")
        _point_from(self.reference_point_px, "reference_point_px")
        _lossless_asset_path(self.template_path, "anchor template_path")
        if self.mask_path is not None:
            _lossless_asset_path(self.mask_path, "anchor mask_path")

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "reference_point_px": [float(value) for value in self.reference_point_px],
            "template_bbox_xyxy": self.template_bbox_xyxy.as_xyxy(),
            "search_roi_xyxy": self.search_roi_xyxy.as_xyxy(),
            "template_path": self.template_path,
            "mask_path": self.mask_path,
        }

    @classmethod
    def from_mapping(cls, value: object) -> AlignmentAnchor:
        mapping = _mapping(value, "alignment anchor")
        return cls(
            anchor_id=_text(mapping, "anchor_id"),
            reference_point_px=_point_from(mapping.get("reference_point_px"), "reference_point_px"),
            template_bbox_xyxy=_bbox_from(
                mapping.get("template_bbox_xyxy"), "template_bbox_xyxy"
            ),
            search_roi_xyxy=_bbox_from(mapping.get("search_roi_xyxy"), "search_roi_xyxy"),
            template_path=_text(mapping, "template_path"),
            mask_path=_optional_text(mapping.get("mask_path"), "mask_path"),
        )


@dataclass(frozen=True, slots=True)
class AlignmentRecipe:
    anchors: tuple[AlignmentAnchor, ...] = ()
    quality_gates: AlignmentQualityGates = field(default_factory=AlignmentQualityGates)
    transform_model: str = "partial_affine"
    anchor_provenance: str = APPROVED_ANCHOR_PROVENANCE

    def __post_init__(self) -> None:
        if self.transform_model != "partial_affine":
            raise RecipeValidationError("Only partial_affine alignment is supported for inspection")
        if self.anchor_provenance not in {
            APPROVED_ANCHOR_PROVENANCE,
            DEMO_GRID_ANCHOR_PROVENANCE,
        }:
            raise RecipeValidationError("Unsupported alignment anchor provenance")
        identifiers = [anchor.anchor_id for anchor in self.anchors]
        if len(identifiers) != len(set(identifiers)):
            raise RecipeValidationError("Alignment anchor IDs must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform_model": self.transform_model,
            "anchor_provenance": self.anchor_provenance,
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "quality_gates": self.quality_gates.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: object) -> AlignmentRecipe:
        mapping = _mapping(value, "alignment")
        anchors = mapping.get("anchors")
        if not isinstance(anchors, list):
            raise RecipeValidationError("alignment.anchors must be a list")
        return cls(
            anchors=tuple(AlignmentAnchor.from_mapping(item) for item in anchors),
            quality_gates=AlignmentQualityGates.from_mapping(mapping.get("quality_gates")),
            transform_model=_text(mapping, "transform_model"),
            anchor_provenance=_text(mapping, "anchor_provenance"),
        )


@dataclass(frozen=True, slots=True)
class SlotRecipe:
    slot_id: str
    label_hint: str
    class_id: int | None
    expected_bbox_xyxy: BoundingBox
    expected_center_px: tuple[float, float]
    expected_angle_deg: float | None
    rotation_period_deg: float | None
    fixed_roi_xyxy: BoundingBox
    template_path: str
    component_mask_path: str
    compare_mask_path: str
    ignore_mask_path: str | None
    search_margin_px: int
    position_tolerance: PositionTolerance
    appearance_thresholds: AppearanceThresholds
    source: str
    source_confidence: float

    def __post_init__(self) -> None:
        if not self.slot_id.startswith("slot_") or not self.slot_id[5:].isdigit():
            raise RecipeValidationError(f"Invalid stable slot ID: {self.slot_id}")
        _nonempty(self.label_hint, "label_hint")
        _point_from(self.expected_center_px, "expected_center_px")
        if self.expected_angle_deg is not None:
            _finite_number(self.expected_angle_deg, "expected_angle_deg")
        if self.rotation_period_deg not in {None, 180.0, 360.0}:
            raise RecipeValidationError("rotation_period_deg must be null, 180, or 360")
        if int(self.search_margin_px) < 0:
            raise RecipeValidationError("search_margin_px must be non-negative")
        if not 0.0 <= float(self.source_confidence) <= 1.0:
            raise RecipeValidationError("source_confidence must be between 0 and 1")
        for name, path in (
            ("template_path", self.template_path),
            ("component_mask_path", self.component_mask_path),
            ("compare_mask_path", self.compare_mask_path),
        ):
            _lossless_asset_path(path, name)
        if self.ignore_mask_path is not None:
            _lossless_asset_path(self.ignore_mask_path, "ignore_mask_path")

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "label_hint": self.label_hint,
            "class_id": self.class_id,
            "expected_bbox_xyxy": self.expected_bbox_xyxy.as_xyxy(),
            "expected_center_px": [float(value) for value in self.expected_center_px],
            "expected_angle_deg": self.expected_angle_deg,
            "rotation_period_deg": self.rotation_period_deg,
            "fixed_roi_xyxy": self.fixed_roi_xyxy.as_xyxy(),
            "template_path": self.template_path,
            "component_mask_path": self.component_mask_path,
            "compare_mask_path": self.compare_mask_path,
            "ignore_mask_path": self.ignore_mask_path,
            "search_margin_px": int(self.search_margin_px),
            "position_tolerance": self.position_tolerance.to_dict(),
            "appearance_thresholds": self.appearance_thresholds.to_dict(),
            "source": self.source,
            "source_confidence": float(self.source_confidence),
        }

    @classmethod
    def from_mapping(cls, value: object) -> SlotRecipe:
        mapping = _mapping(value, "slot")
        class_id = mapping.get("class_id")
        angle = mapping.get("expected_angle_deg")
        period = mapping.get("rotation_period_deg")
        return cls(
            slot_id=_text(mapping, "slot_id"),
            label_hint=_text(mapping, "label_hint"),
            class_id=None if class_id is None else _integer_value(class_id, "class_id"),
            expected_bbox_xyxy=_bbox_from(mapping.get("expected_bbox_xyxy"), "expected_bbox_xyxy"),
            expected_center_px=_point_from(mapping.get("expected_center_px"), "expected_center_px"),
            expected_angle_deg=None if angle is None else _finite_number(angle, "expected_angle_deg"),
            rotation_period_deg=None if period is None else _finite_number(period, "rotation_period_deg"),
            fixed_roi_xyxy=_bbox_from(mapping.get("fixed_roi_xyxy"), "fixed_roi_xyxy"),
            template_path=_text(mapping, "template_path"),
            component_mask_path=_text(mapping, "component_mask_path"),
            compare_mask_path=_text(mapping, "compare_mask_path"),
            ignore_mask_path=_optional_text(
                mapping.get("ignore_mask_path"), "ignore_mask_path"
            ),
            search_margin_px=_integer(mapping, "search_margin_px"),
            position_tolerance=PositionTolerance.from_mapping(mapping.get("position_tolerance")),
            appearance_thresholds=AppearanceThresholds.from_mapping(mapping.get("appearance_thresholds")),
            source=_text(mapping, "source"),
            source_confidence=_number(mapping, "source_confidence"),
        )


@dataclass(frozen=True, slots=True)
class RejectedDetection:
    label: str
    source: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "source": self.source,
            "confidence": float(self.confidence),
            "bbox_xyxy": [float(value) for value in self.bbox_xyxy],
            "reason": self.reason,
        }

    @classmethod
    def from_mapping(cls, value: object) -> RejectedDetection:
        mapping = _mapping(value, "rejected detection")
        return cls(
            label=_text(mapping, "label"),
            source=_text(mapping, "source"),
            confidence=_number(mapping, "confidence"),
            bbox_xyxy=_bbox_values(mapping.get("bbox_xyxy"), "bbox_xyxy"),
            reason=_text(mapping, "reason"),
        )


@dataclass(frozen=True, slots=True)
class InspectionRecipe:
    board_id: str
    side: str
    golden_sha256: str
    golden_asset_path: str
    image_size: ImageSize
    metrology: MetrologyCalibration
    alignment: AlignmentRecipe
    slots: tuple[SlotRecipe, ...]
    asset_sha256: Mapping[str, str]
    model_identifiers: Mapping[str, str] = field(default_factory=dict)
    rejected_detections: tuple[RejectedDetection, ...] = ()
    production_eligible: bool = True
    enrollment: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = RECIPE_SCHEMA_VERSION
    coordinate_space: str = GOLDEN_COORDINATE_SPACE

    def __post_init__(self) -> None:
        if self.schema_version != RECIPE_SCHEMA_VERSION:
            raise RecipeValidationError(f"Unsupported recipe schema: {self.schema_version}")
        if self.coordinate_space != GOLDEN_COORDINATE_SPACE:
            raise RecipeValidationError(f"Recipe coordinate_space must be {GOLDEN_COORDINATE_SPACE}")
        _nonempty(self.board_id, "board_id")
        _nonempty(self.side, "side")
        if self.side not in {"top", "bottom"}:
            raise RecipeValidationError("side must be 'top' or 'bottom'")
        if len(self.golden_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.golden_sha256.lower()
        ):
            raise RecipeValidationError("golden_sha256 must be a 64-character hexadecimal digest")
        _lossless_asset_path(self.golden_asset_path, "golden_asset_path")
        referenced_assets = _referenced_asset_paths(
            self.golden_asset_path, self.alignment, self.slots
        )
        if set(self.asset_sha256) != set(referenced_assets):
            missing = sorted(set(referenced_assets) - set(self.asset_sha256))
            unexpected = sorted(set(self.asset_sha256) - set(referenced_assets))
            raise RecipeValidationError(
                "asset_sha256 must contain exactly every recipe asset "
                f"(missing={missing}, unexpected={unexpected})"
            )
        for path, digest in self.asset_sha256.items():
            _lossless_asset_path(str(path), "asset_sha256 path")
            if not isinstance(digest, str) or len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest.lower()
            ):
                raise RecipeValidationError(
                    f"asset_sha256[{path!r}] must be a 64-character hexadecimal digest"
                )
        if self.asset_sha256[self.golden_asset_path].lower() != self.golden_sha256:
            raise RecipeValidationError("golden_sha256 must match the Golden asset digest")
        if not isinstance(self.production_eligible, bool):
            raise RecipeValidationError("production_eligible must be a boolean")
        if not self.slots:
            raise RecipeValidationError("Inspection recipe must contain at least one slot")
        expected_ids = [f"slot_{index:04d}" for index in range(1, len(self.slots) + 1)]
        if [slot.slot_id for slot in self.slots] != expected_ids:
            raise RecipeValidationError("Slot IDs must be sequential and deterministically ordered")
        for key, value in self.model_identifiers.items():
            _nonempty(str(key), "model identifier key")
            _nonempty(str(value), "model identifier value")
        if self.production_eligible:
            if not self.metrology.verified:
                raise RecipeValidationError(
                    "Production-eligible recipe requires verified metrology"
                )
            if len(self.alignment.anchors) < self.alignment.quality_gates.min_anchors:
                raise RecipeValidationError(
                    "Production-eligible recipe requires enough alignment anchors"
                )
            if self.alignment.anchor_provenance != APPROVED_ANCHOR_PROVENANCE:
                raise RecipeValidationError(
                    "Production-eligible recipe requires approved stable alignment anchors"
                )
            if not self.model_identifiers.get("component_detector"):
                raise RecipeValidationError(
                    "Production-eligible recipe requires a component detector identifier"
                )
            if any(slot.source == "opencv_candidate" for slot in self.slots):
                raise RecipeValidationError(
                    "Production-eligible recipe cannot contain opencv_candidate slots"
                )
        _validate_geometry(self)

    @property
    def content_sha256(self) -> str:
        return sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "board_id": self.board_id,
            "side": self.side,
            "golden_sha256": self.golden_sha256,
            "golden_asset_path": self.golden_asset_path,
            "coordinate_space": self.coordinate_space,
            "image_size": self.image_size.to_dict(),
            "metrology": self.metrology.to_dict(),
            "alignment": self.alignment.to_dict(),
            "asset_sha256": dict(sorted(self.asset_sha256.items())),
            "model_identifiers": dict(sorted(self.model_identifiers.items())),
            "production_eligible": bool(self.production_eligible),
            "enrollment": dict(self.enrollment),
            "slots": [slot.to_dict() for slot in self.slots],
            "rejected_detections": [item.to_dict() for item in self.rejected_detections],
        }

    @classmethod
    def from_mapping(cls, value: object) -> InspectionRecipe:
        mapping = _mapping(value, "recipe")
        schema = _text(mapping, "schema_version")
        if schema != RECIPE_SCHEMA_VERSION:
            raise RecipeValidationError(f"Unsupported recipe schema: {schema}")
        slots = mapping.get("slots")
        rejected = mapping.get("rejected_detections", [])
        identifiers = mapping.get("model_identifiers", {})
        enrollment = mapping.get("enrollment", {})
        asset_sha256 = mapping.get("asset_sha256")
        if not isinstance(slots, list):
            raise RecipeValidationError("recipe.slots must be a list")
        if not isinstance(rejected, list):
            raise RecipeValidationError("recipe.rejected_detections must be a list")
        if not isinstance(identifiers, Mapping) or not all(
            isinstance(key, str) and isinstance(item, str) for key, item in identifiers.items()
        ):
            raise RecipeValidationError("model_identifiers must map strings to strings")
        if not isinstance(enrollment, Mapping):
            raise RecipeValidationError("enrollment must be an object")
        if not isinstance(asset_sha256, Mapping) or not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in asset_sha256.items()
        ):
            raise RecipeValidationError("asset_sha256 must map asset paths to SHA-256 strings")
        return cls(
            schema_version=schema,
            board_id=_text(mapping, "board_id"),
            side=_text(mapping, "side"),
            golden_sha256=_text(mapping, "golden_sha256").lower(),
            golden_asset_path=_text(mapping, "golden_asset_path"),
            coordinate_space=_text(mapping, "coordinate_space"),
            image_size=ImageSize.from_mapping(mapping.get("image_size")),
            metrology=MetrologyCalibration.from_mapping(mapping.get("metrology")),
            alignment=AlignmentRecipe.from_mapping(mapping.get("alignment")),
            slots=tuple(SlotRecipe.from_mapping(item) for item in slots),
            asset_sha256={key: item.lower() for key, item in asset_sha256.items()},
            model_identifiers=dict(identifiers),
            rejected_detections=tuple(RejectedDetection.from_mapping(item) for item in rejected),
            production_eligible=_boolean(mapping, "production_eligible"),
            enrollment=dict(enrollment),
        )


@dataclass(frozen=True, slots=True)
class RecipeBuildResult:
    recipe: InspectionRecipe
    output_dir: Path

    @property
    def rejected_detections(self) -> tuple[RejectedDetection, ...]:
        return self.recipe.rejected_detections


@dataclass(frozen=True, slots=True)
class _AcceptedDetection:
    detection: Detection
    bbox: BoundingBox
    roi: BoundingBox


def create_recipe(
    golden_image: np.ndarray,
    detections: Sequence[Detection],
    output_dir: str | Path,
    *,
    board_id: str = DEMO_BOARD_ID,
    side: str = DEFAULT_BOARD_SIDE,
    metrology: MetrologyCalibration,
    roi_padding_px: int,
    search_margin_px: int,
    position_tolerance: PositionTolerance,
    appearance_thresholds: AppearanceThresholds,
    alignment: AlignmentRecipe | None = None,
    model_identifiers: Mapping[str, str] | None = None,
    allow_demo_sources: bool = False,
    rotation_period_deg: float | None = None,
    measurement_metadata: Mapping[str, Any] | None = None,
) -> RecipeBuildResult:
    """Enroll deterministic fixed slots from Golden detector proposals.

    ``board_id`` and ``side`` default to ``demo_board`` and ``top`` for the
    current single-board demo. Random ``Detection.detection_id`` values are
    intentionally not persisted.
    Runtime position measurement must use the fixed native template/ROI assets,
    never a recentered test detection box.
    """

    golden = _measurement_image(golden_image)
    if int(roi_padding_px) < 0:
        raise RecipeValidationError("roi_padding_px must be non-negative")
    if int(search_margin_px) < 0:
        raise RecipeValidationError("search_margin_px must be non-negative")
    if rotation_period_deg not in {None, 180.0, 360.0}:
        raise RecipeValidationError("rotation_period_deg must be null, 180, or 360")
    _nonempty(board_id, "board_id")
    _nonempty(side, "side")
    height, width = golden.shape[:2]
    accepted: list[_AcceptedDetection] = []
    rejected: list[RejectedDetection] = []

    for detection in detections:
        rejection = _rejection_reason(detection, width, height, allow_demo_sources)
        if rejection is not None:
            rejected.append(_rejected(detection, rejection))
            continue
        bbox = detection.bbox.clamp(width, height)
        if bbox.width <= 0 or bbox.height <= 0:
            rejected.append(_rejected(detection, "empty_bbox_after_clamp"))
            continue
        roi = _fixed_roi(bbox, width, height, int(roi_padding_px))
        if roi.width <= 0 or roi.height <= 0:
            rejected.append(_rejected(detection, "empty_roi_after_clamp"))
            continue
        accepted.append(_AcceptedDetection(detection, bbox, roi))

    accepted.sort(key=_accepted_sort_key)
    rejected.sort(key=_rejected_sort_key)
    if not accepted:
        reasons = ", ".join(sorted({item.reason for item in rejected})) or "no detections"
        raise RecipeValidationError(f"No valid production slots were created ({reasons})")

    destination = Path(output_dir).expanduser().resolve()
    (destination / "templates").mkdir(parents=True, exist_ok=True)
    (destination / "masks").mkdir(parents=True, exist_ok=True)
    golden_bytes = _png_bytes(golden)
    (destination / "golden.png").write_bytes(golden_bytes)

    slots: list[SlotRecipe] = []
    for index, item in enumerate(accepted, start=1):
        slot_id = f"slot_{index:04d}"
        x1, y1, x2, y2 = item.roi.to_int()
        template = np.ascontiguousarray(golden[y1:y2, x1:x2].copy())
        component_mask = _bbox_mask(item.bbox, item.roi)
        template_path = f"templates/{slot_id}.png"
        component_mask_path = f"masks/{slot_id}_component.png"
        compare_mask_path = f"masks/{slot_id}_compare.png"
        (destination / template_path).write_bytes(_png_bytes(template))
        mask_bytes = _png_bytes(component_mask)
        (destination / component_mask_path).write_bytes(mask_bytes)
        (destination / compare_mask_path).write_bytes(mask_bytes)
        slots.append(
            SlotRecipe(
                slot_id=slot_id,
                label_hint=item.detection.label,
                class_id=item.detection.class_id,
                expected_bbox_xyxy=item.bbox,
                expected_center_px=((item.bbox.x1 + item.bbox.x2) / 2.0, (item.bbox.y1 + item.bbox.y2) / 2.0),
                expected_angle_deg=(None if rotation_period_deg is None else 0.0),
                rotation_period_deg=rotation_period_deg,
                fixed_roi_xyxy=item.roi,
                template_path=template_path,
                component_mask_path=component_mask_path,
                compare_mask_path=compare_mask_path,
                ignore_mask_path=None,
                search_margin_px=int(search_margin_px),
                position_tolerance=position_tolerance,
                appearance_thresholds=appearance_thresholds,
                source=item.detection.source,
                source_confidence=float(item.detection.confidence),
            )
        )

    alignment_recipe = alignment or AlignmentRecipe()
    alignment_ready = (
        len(alignment_recipe.anchors) >= alignment_recipe.quality_gates.min_anchors
    )
    production_eligible = (
        bool(metrology.verified)
        and alignment_ready
        and alignment_recipe.anchor_provenance == APPROVED_ANCHOR_PROVENANCE
        and all(item.detection.source != "opencv_candidate" for item in accepted)
        and bool((model_identifiers or {}).get("component_detector"))
    )
    referenced_assets = _referenced_asset_paths("golden.png", alignment_recipe, slots)
    asset_sha256 = _asset_digests(destination, referenced_assets)
    recipe = InspectionRecipe(
        board_id=board_id,
        side=side,
        golden_sha256=sha256(golden_bytes).hexdigest(),
        golden_asset_path="golden.png",
        image_size=ImageSize(width=width, height=height),
        metrology=metrology,
        alignment=alignment_recipe,
        slots=tuple(slots),
        asset_sha256=asset_sha256,
        model_identifiers=dict(model_identifiers or {}),
        rejected_detections=tuple(rejected),
        production_eligible=production_eligible,
        enrollment={
            "roi_padding_px": int(roi_padding_px),
            "mask_strategy": "expected_bbox_rectangle",
            "allow_demo_sources": bool(allow_demo_sources),
            "calibration_verified": bool(metrology.verified),
            "alignment_ready": alignment_ready,
            "alignment_anchor_provenance": alignment_recipe.anchor_provenance,
            "measurement_domain": dict(measurement_metadata or {}),
        },
    )
    save_recipe(recipe, destination / "recipe.json")
    validate_recipe_assets(recipe, destination)
    return RecipeBuildResult(recipe=recipe, output_dir=destination)


def create_grid_alignment_recipe(
    golden_image: np.ndarray,
    output_dir: str | Path,
    *,
    template_size_px: int = 65,
    search_margin_px: int = 48,
    grid_fractions: Sequence[tuple[float, float]] | None = None,
    quality_gates: AlignmentQualityGates | None = None,
) -> AlignmentRecipe:
    """Persist deterministic native-resolution patches for demo alignment.

    Grid patches are convenient enrollment candidates for the local demo, but
    they are not automatically certified fiducials. A production recipe still
    requires the selected patches, camera setup, and calibration to be reviewed
    on repeated real captures.
    """

    golden = _measurement_image(golden_image)
    size = int(template_size_px)
    margin = int(search_margin_px)
    if size < 9 or size % 2 == 0:
        raise RecipeValidationError("template_size_px must be an odd integer >= 9")
    if margin < 0:
        raise RecipeValidationError("alignment search_margin_px must be non-negative")
    points = (
        tuple(
            (fraction_x, fraction_y)
            for fraction_y in (0.15, 0.325, 0.50, 0.675, 0.85)
            for fraction_x in (0.15, 0.325, 0.50, 0.675, 0.85)
        )
        if grid_fractions is None
        else tuple(grid_fractions)
    )
    if not points:
        raise RecipeValidationError("At least one alignment grid point is required")
    height, width = golden.shape[:2]
    if size > width or size > height:
        raise RecipeValidationError("Alignment template is larger than the Golden image")

    gates = quality_gates or AlignmentQualityGates(
        min_anchors=4,
        min_anchor_score=0.70,
        max_residual_px=0.75,
        ransac_reprojection_threshold_px=1.0,
        min_inlier_ratio=0.50,
        min_scale=0.95,
        max_scale=1.05,
        max_abs_rotation_deg=5.0,
        min_canvas_overlap_ratio=0.90,
    )
    if len(points) < int(gates.min_anchors):
        raise RecipeValidationError(
            "Alignment grid does not contain enough points for its quality gate"
        )

    destination = Path(output_dir).expanduser().resolve()
    anchor_dir = destination / "anchors"
    anchor_dir.mkdir(parents=True, exist_ok=True)
    half = size // 2
    anchors: list[AlignmentAnchor] = []
    seen_centers: set[tuple[int, int]] = set()
    for fraction_x, fraction_y in points:
        fx = float(fraction_x)
        fy = float(fraction_y)
        if not math.isfinite(fx) or not math.isfinite(fy) or not (
            0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0
        ):
            raise RecipeValidationError("Alignment grid fractions must be in [0, 1]")
        center_x = int(round(fx * (width - 1)))
        center_y = int(round(fy * (height - 1)))
        x1 = min(max(0, center_x - half), width - size)
        y1 = min(max(0, center_y - half), height - size)
        x2 = x1 + size
        y2 = y1 + size
        center_x = x1 + half
        center_y = y1 + half
        if (center_x, center_y) in seen_centers:
            raise RecipeValidationError(
                "Alignment grid points collapse to duplicate pixel centers"
            )
        seen_centers.add((center_x, center_y))
        anchor_id = f"anchor_{len(anchors) + 1:04d}"
        relative_path = f"anchors/{anchor_id}.png"
        patch = np.ascontiguousarray(golden[y1:y2, x1:x2].copy())
        (destination / relative_path).write_bytes(_png_bytes(patch))
        anchors.append(
            AlignmentAnchor(
                anchor_id=anchor_id,
                reference_point_px=(float(center_x), float(center_y)),
                template_bbox_xyxy=BoundingBox(x1, y1, x2, y2),
                search_roi_xyxy=BoundingBox(
                    max(0, x1 - margin),
                    max(0, y1 - margin),
                    min(width, x2 + margin),
                    min(height, y2 + margin),
                ),
                template_path=relative_path,
            )
        )
    return AlignmentRecipe(
        anchors=tuple(anchors),
        quality_gates=gates,
        anchor_provenance=DEMO_GRID_ANCHOR_PROVENANCE,
    )


def save_recipe(recipe: InspectionRecipe, path: str | Path) -> Path:
    """Serialize a validated recipe without embedding workstation paths."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(recipe.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def load_recipe(path: str | Path, *, validate_assets: bool = True) -> InspectionRecipe:
    """Load a recipe and optionally verify all referenced lossless assets."""

    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecipeValidationError(f"Could not read recipe JSON: {exc}") from exc
    recipe = InspectionRecipe.from_mapping(payload)
    if validate_assets:
        validate_recipe_assets(recipe, source.parent)
    return recipe


def validate_recipe_assets(recipe: InspectionRecipe, root: str | Path) -> None:
    """Validate hashes, dimensions, and native slot asset geometry."""

    base = Path(root).expanduser().resolve()
    for asset_path, expected_digest in recipe.asset_sha256.items():
        resolved = _resolve_asset(base, asset_path)
        try:
            encoded = resolved.read_bytes()
        except OSError as exc:
            raise RecipeValidationError(
                f"Could not read recipe asset {asset_path}: {exc}"
            ) from exc
        if sha256(encoded).hexdigest() != expected_digest:
            raise RecipeValidationError(f"Asset SHA-256 mismatch: {asset_path}")

    golden_path = _resolve_asset(base, recipe.golden_asset_path)
    try:
        golden_bytes = golden_path.read_bytes()
    except OSError as exc:
        raise RecipeValidationError(f"Could not read Golden asset: {exc}") from exc
    if sha256(golden_bytes).hexdigest() != recipe.golden_sha256:
        raise RecipeValidationError("Golden SHA-256 mismatch")
    golden = cv2.imdecode(np.frombuffer(golden_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if golden is None:
        raise RecipeValidationError("Golden asset is not a readable lossless image")
    if golden.shape[:2] != (recipe.image_size.height, recipe.image_size.width):
        raise RecipeValidationError("Golden asset dimensions do not match recipe image_size")

    for anchor in recipe.alignment.anchors:
        expected_shape = (
            int(anchor.template_bbox_xyxy.height),
            int(anchor.template_bbox_xyxy.width),
        )
        template = _read_asset(base, anchor.template_path, cv2.IMREAD_COLOR)
        if template.shape[:2] != expected_shape:
            raise RecipeValidationError(f"Anchor template size mismatch for {anchor.anchor_id}")
        if anchor.mask_path is not None:
            mask = _read_asset(base, anchor.mask_path, cv2.IMREAD_GRAYSCALE)
            if mask.shape != expected_shape:
                raise RecipeValidationError(f"Anchor mask size mismatch for {anchor.anchor_id}")

    for slot in recipe.slots:
        expected_shape = (int(slot.fixed_roi_xyxy.height), int(slot.fixed_roi_xyxy.width))
        template = _read_asset(base, slot.template_path, cv2.IMREAD_COLOR)
        component_mask = _read_asset(base, slot.component_mask_path, cv2.IMREAD_GRAYSCALE)
        compare_mask = _read_asset(base, slot.compare_mask_path, cv2.IMREAD_GRAYSCALE)
        ignore_mask = (
            None
            if slot.ignore_mask_path is None
            else _read_asset(base, slot.ignore_mask_path, cv2.IMREAD_GRAYSCALE)
        )
        if template.shape[:2] != expected_shape:
            raise RecipeValidationError(f"Template size mismatch for {slot.slot_id}")
        if component_mask.shape != expected_shape or compare_mask.shape != expected_shape:
            raise RecipeValidationError(f"Mask size mismatch for {slot.slot_id}")
        if ignore_mask is not None and ignore_mask.shape != expected_shape:
            raise RecipeValidationError(f"Ignore mask size mismatch for {slot.slot_id}")


def _measurement_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise RecipeValidationError("Golden measurement image must be a uint8 BGR array")
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise RecipeValidationError("Golden measurement image must be a uint8 BGR array")
    return np.ascontiguousarray(image)


def _rejection_reason(detection: Detection, width: int, height: int, allow_demo_sources: bool) -> str | None:
    bbox = detection.bbox
    if detection.source == "opencv_candidate" and not allow_demo_sources:
        return "opencv_candidate_not_allowed_for_production"
    if bbox.width <= 0 or bbox.height <= 0:
        return "empty_bbox_after_clamp"
    if bbox.x2 <= 0 or bbox.y2 <= 0 or bbox.x1 >= width or bbox.y1 >= height:
        return "bbox_outside_golden_image"
    return None


def _fixed_roi(bbox: BoundingBox, width: int, height: int, padding: int) -> BoundingBox:
    return BoundingBox(
        float(max(0, math.floor(bbox.x1) - padding)),
        float(max(0, math.floor(bbox.y1) - padding)),
        float(min(width, math.ceil(bbox.x2) + padding)),
        float(min(height, math.ceil(bbox.y2) + padding)),
    )


def _bbox_mask(bbox: BoundingBox, roi: BoundingBox) -> np.ndarray:
    roi_x1, roi_y1, roi_x2, roi_y2 = roi.to_int()
    mask = np.zeros((roi_y2 - roi_y1, roi_x2 - roi_x1), dtype=np.uint8)
    x1 = max(0, math.floor(bbox.x1) - roi_x1)
    y1 = max(0, math.floor(bbox.y1) - roi_y1)
    x2 = min(mask.shape[1], math.ceil(bbox.x2) - roi_x1)
    y2 = min(mask.shape[0], math.ceil(bbox.y2) - roi_y1)
    mask[y1:y2, x1:x2] = 255
    return mask


def _accepted_sort_key(item: _AcceptedDetection) -> tuple[Any, ...]:
    bbox = item.bbox
    return (
        (bbox.y1 + bbox.y2) / 2.0,
        (bbox.x1 + bbox.x2) / 2.0,
        bbox.y1,
        bbox.x1,
        bbox.y2,
        bbox.x2,
        item.detection.label,
        -1 if item.detection.class_id is None else item.detection.class_id,
        -float(item.detection.confidence),
        item.detection.source,
    )


def _rejected(detection: Detection, reason: str) -> RejectedDetection:
    return RejectedDetection(
        label=detection.label,
        source=detection.source,
        confidence=float(detection.confidence),
        bbox_xyxy=tuple(detection.bbox.as_xyxy()),
        reason=reason,
    )


def _rejected_sort_key(item: RejectedDetection) -> tuple[Any, ...]:
    return (*item.bbox_xyxy, item.label, item.source, -item.confidence, item.reason)


def _validate_geometry(recipe: InspectionRecipe) -> None:
    width, height = recipe.image_size.width, recipe.image_size.height
    for anchor in recipe.alignment.anchors:
        for name, bbox in (
            ("template_bbox_xyxy", anchor.template_bbox_xyxy),
            ("search_roi_xyxy", anchor.search_roi_xyxy),
        ):
            if bbox.width <= 0 or bbox.height <= 0:
                raise RecipeValidationError(f"{anchor.anchor_id} has an empty {name}")
            if bbox.x1 < 0 or bbox.y1 < 0 or bbox.x2 > width or bbox.y2 > height:
                raise RecipeValidationError(f"{anchor.anchor_id} {name} is outside Golden image")
            if not all(float(value).is_integer() for value in bbox.as_xyxy()):
                raise RecipeValidationError(f"{anchor.anchor_id} {name} must use integer pixels")
        point_x, point_y = anchor.reference_point_px
        if not (
            anchor.template_bbox_xyxy.x1 <= point_x < anchor.template_bbox_xyxy.x2
            and anchor.template_bbox_xyxy.y1 <= point_y < anchor.template_bbox_xyxy.y2
        ):
            raise RecipeValidationError(
                f"{anchor.anchor_id} reference point must lie inside its template bbox"
            )
    for slot in recipe.slots:
        for name, bbox in (("expected_bbox_xyxy", slot.expected_bbox_xyxy), ("fixed_roi_xyxy", slot.fixed_roi_xyxy)):
            if bbox.width <= 0 or bbox.height <= 0:
                raise RecipeValidationError(f"{slot.slot_id} has an empty {name}")
            if bbox.x1 < 0 or bbox.y1 < 0 or bbox.x2 > width or bbox.y2 > height:
                raise RecipeValidationError(f"{slot.slot_id} {name} is outside Golden image")
            if name == "fixed_roi_xyxy" and not all(
                float(value).is_integer() for value in bbox.as_xyxy()
            ):
                raise RecipeValidationError(
                    f"{slot.slot_id} fixed_roi_xyxy must use integer pixels"
                )
        if (
            slot.fixed_roi_xyxy.x1 > slot.expected_bbox_xyxy.x1
            or slot.fixed_roi_xyxy.y1 > slot.expected_bbox_xyxy.y1
            or slot.fixed_roi_xyxy.x2 < slot.expected_bbox_xyxy.x2
            or slot.fixed_roi_xyxy.y2 < slot.expected_bbox_xyxy.y2
        ):
            raise RecipeValidationError(f"{slot.slot_id} fixed ROI does not contain expected bbox")
        expected_center = (
            (slot.expected_bbox_xyxy.x1 + slot.expected_bbox_xyxy.x2) / 2.0,
            (slot.expected_bbox_xyxy.y1 + slot.expected_bbox_xyxy.y2) / 2.0,
        )
        if not np.allclose(slot.expected_center_px, expected_center, atol=1e-9):
            raise RecipeValidationError(f"{slot.slot_id} expected center does not match bbox")


def _read_asset(root: Path, path: str, mode: int) -> np.ndarray:
    resolved = _resolve_asset(root, path)
    try:
        encoded = resolved.read_bytes()
    except OSError as exc:
        raise RecipeValidationError(f"Could not read recipe asset {path}: {exc}") from exc
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), mode)
    if image is None:
        raise RecipeValidationError(f"Recipe asset is not a readable image: {path}")
    return image


def _referenced_asset_paths(
    golden_asset_path: str,
    alignment: AlignmentRecipe,
    slots: Sequence[SlotRecipe],
) -> tuple[str, ...]:
    paths: set[str] = {golden_asset_path}
    for anchor in alignment.anchors:
        paths.add(anchor.template_path)
        if anchor.mask_path is not None:
            paths.add(anchor.mask_path)
    for slot in slots:
        paths.update(
            {
                slot.template_path,
                slot.component_mask_path,
                slot.compare_mask_path,
            }
        )
        if slot.ignore_mask_path is not None:
            paths.add(slot.ignore_mask_path)
    return tuple(sorted(paths))


def _asset_digests(root: Path, paths: Sequence[str]) -> dict[str, str]:
    digests: dict[str, str] = {}
    for asset_path in paths:
        resolved = _resolve_asset(root, asset_path)
        try:
            digests[asset_path] = sha256(resolved.read_bytes()).hexdigest()
        except OSError as exc:
            raise RecipeValidationError(
                f"Could not hash recipe asset {asset_path}: {exc}"
            ) from exc
    return digests


def _resolve_asset(root: Path, path: str) -> Path:
    _portable_path(path)
    resolved = (root / Path(PurePosixPath(path))).resolve()
    if resolved != root and root not in resolved.parents:
        raise RecipeValidationError(f"Recipe asset escapes its root: {path}")
    return resolved


def _portable_path(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise RecipeValidationError("Recipe asset path cannot be empty")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise RecipeValidationError(f"Recipe assets must use a relative portable path: {value}")


def _lossless_asset_path(value: str, name: str) -> None:
    _portable_path(value)
    if PurePosixPath(value).suffix.lower() not in {".png", ".tif", ".tiff"}:
        raise RecipeValidationError(
            f"{name} must reference a lossless PNG/TIFF asset"
        )


def _png_bytes(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RecipeValidationError("Could not encode a lossless PNG recipe asset")
    return encoded.tobytes()


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecipeValidationError(f"{name} must be an object")
    return value


def _text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RecipeValidationError(f"{key} must be a non-empty string")
    return value


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RecipeValidationError(f"{name} must be null or a non-empty string")
    return value


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise RecipeValidationError(f"{name} must be a non-empty string")


def _number(mapping: Mapping[str, Any], key: str) -> float:
    return _finite_number(mapping.get(key), key)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise RecipeValidationError(f"{name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RecipeValidationError(f"{name} must be a finite number")
    return numeric


def _nonnegative_finite(value: object, name: str) -> None:
    if _finite_number(value, name) < 0:
        raise RecipeValidationError(f"{name} must be non-negative")


def _integer(mapping: Mapping[str, Any], key: str) -> int:
    return _integer_value(mapping.get(key), key)


def _boolean(mapping: Mapping[str, Any], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise RecipeValidationError(f"{key} must be a boolean")
    return value


def _integer_value(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RecipeValidationError(f"{name} must be an integer")
    return int(value)


def _bbox_values(value: object, name: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise RecipeValidationError(f"{name} must contain four xyxy values")
    values = tuple(_finite_number(item, name) for item in value)
    return values  # type: ignore[return-value]


def _bbox_from(value: object, name: str) -> BoundingBox:
    try:
        return BoundingBox(*_bbox_values(value, name))
    except ValueError as exc:
        raise RecipeValidationError(f"Invalid {name}: {exc}") from exc


def _point_from(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise RecipeValidationError(f"{name} must contain two values")
    return (_finite_number(value[0], name), _finite_number(value[1], name))


__all__ = [
    "AppearanceThresholds",
    "AlignmentAnchor",
    "AlignmentQualityGates",
    "AlignmentRecipe",
    "DEFAULT_BOARD_SIDE",
    "DEMO_BOARD_ID",
    "GOLDEN_COORDINATE_SPACE",
    "ImageSize",
    "InspectionRecipe",
    "MetrologyCalibration",
    "PositionTolerance",
    "RECIPE_SCHEMA_VERSION",
    "RecipeBuildResult",
    "RecipeValidationError",
    "RejectedDetection",
    "SlotRecipe",
    "create_recipe",
    "load_recipe",
    "save_recipe",
    "validate_recipe_assets",
]
