"""Configuration dataclasses for steps 1 through 6.1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from collections.abc import Mapping
from typing import Any, Literal


@dataclass(slots=True)
class PreprocessConfig:
    """Image enhancement options used by step 1.

    The defaults are intentionally conservative. They improve uneven illumination
    without aggressively destroying tiny silkscreen or component edges.
    """

    undistort: bool = False
    calibration_profile: dict[str, Any] | None = None
    undistort_alpha: float = 0.0
    calibration_aspect_tolerance: float = 0.01
    # Preserve enough detail for tiled component detection. The detector still
    # receives bounded tiles, so this does not enlarge its model input tensor.
    max_side: int | None = 4096
    denoise: bool = True
    denoise_method: Literal["nlmeans", "bilateral", "gaussian"] = "nlmeans"
    denoise_strength: int = 5
    white_balance: bool = True
    clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: tuple[int, int] = (8, 8)
    normalize: bool = True
    normalize_low_percentile: float = 1.0
    normalize_high_percentile: float = 99.0
    sharpen: bool = True
    sharpen_amount: float = 0.35


@dataclass(slots=True)
class AlignmentConfig:
    """ORB/homography settings and the ECC fallback policy for step 2."""

    enabled: bool = True
    orb_features: int = 4000
    ratio_test: float = 0.75
    min_good_matches: int = 12
    ransac_reprojection_threshold: float = 4.0
    min_inliers: int = 8
    min_inlier_ratio: float = 0.25
    use_ecc_fallback: bool = True
    ecc_iterations: int = 120
    ecc_epsilon: float = 1e-5
    min_ecc_correlation: float = 0.65
    strict: bool = False


@dataclass(slots=True)
class BoardConfig:
    """Contour-based PCB localization settings for step 3."""

    min_area_ratio: float = 0.08
    max_area_ratio: float = 0.995
    min_rectangularity: float = 0.45
    morphology_kernel_ratio: float = 0.015
    fallback_to_full_image: bool = True
    padding_ratio: float = 0.005


@dataclass(slots=True)
class CVDetectorConfig:
    """Heuristic proposal detector used before a trained model is available."""

    min_area_ratio: float = 0.00004
    max_area_ratio: float = 0.035
    min_width: int = 4
    min_height: int = 4
    max_aspect_ratio: float = 12.0
    morphology_kernel: int = 3
    nms_iou_threshold: float = 0.35
    max_detections: int = 500


@dataclass(slots=True)
class ModelDetectorConfig:
    """Ultralytics inference options for a ``.pt`` or ``.onnx`` model."""

    confidence: float = 0.25
    iou: float = 0.45
    image_size: int = 1280
    device: str | None = None
    max_detections: int = 2000
    # ``None`` preserves the head declared by arbitrary Ultralytics artifacts.
    # The local UI explicitly sets False for its Kaggle one-to-many/NMS recipe.
    end2end: bool | None = None


@dataclass(slots=True)
class TilingConfig:
    """Adaptive high-resolution inference policy for step 4."""

    mode: Literal["auto", "on", "off"] = "auto"
    # ``tile_size`` is the upper bound. Auto mode may choose a smaller detail
    # window so a 1000px PCB is not incorrectly treated as a single 1280px tile.
    tile_size: int = 1280
    min_tile_size: int = 640
    detail_window_ratio: float = 0.64
    overlap_ratio: float = 0.20
    auto_trigger_scale: float = 1.25
    include_full_image: bool = True
    detail_confidence: float | None = 0.20
    # Detail tiles improve recall, but some visually ambiguous classes need a
    # stricter floor. In the current detector, bright solder joints are the
    # main low-confidence false positive for ``led``.
    detail_class_confidence: dict[str, float] = field(
        default_factory=lambda: {"led": 0.35}
    )
    merge_iou_threshold: float = 0.45
    # IoU alone misses duplicate partial boxes on opposite sides of a tile seam.
    # IoS (intersection / smaller box) recognizes those fragments without
    # merging ordinary neighboring components that do not overlap.
    seam_ios_threshold: float = 0.50
    # Remove a partial/tight box almost completely enclosed by another box for
    # the same class when they originate from different inference windows.
    containment_ios_threshold: float = 0.80
    class_aware_merge: bool = True
    # Different class labels may still be duplicate hypotheses for one object.
    # Use a stricter threshold than same-class NMS to preserve adjacent parts.
    cross_class_iou_threshold: float = 0.70
    edge_margin_ratio: float = 0.03
    edge_confidence_penalty: float = 0.10
    non_ownership_confidence_penalty: float = 0.10


@dataclass(slots=True)
class CropConfig:
    """Crop and normalization options used by step 5."""

    padding_ratio: float = 0.12
    padding_pixels: int = 2
    square: bool = True
    target_size: tuple[int, int] | None = (224, 224)
    letterbox_color: tuple[int, int, int] = (114, 114, 114)
    image_extension: Literal[".png", ".jpg"] = ".png"
    jpeg_quality: int = 95


@dataclass(slots=True)
class ClassificationConfig:
    """Runtime policy for the step-6.1 component-family classifier.

    Preprocessing details and the default confidence policy live in the model
    manifest. Non-``None`` values below are explicit deployment overrides.
    """

    batch_size: int = 32
    top_k: int = 3
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    accept_threshold: float | None = None
    review_threshold: float | None = None
    temperature: float | None = None


@dataclass(slots=True)
class PipelineConfig:
    """Top-level configuration consumed by :class:`AOIPipeline`."""

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    board: BoardConfig = field(default_factory=BoardConfig)
    cv_detector: CVDetectorConfig = field(default_factory=CVDetectorConfig)
    model_detector: ModelDetectorConfig = field(default_factory=ModelDetectorConfig)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    detector_mode: Literal["auto", "cv"] = "auto"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> PipelineConfig:
        """Translate permissive UI dictionaries into the typed core config.

        Unknown UI-only keys are ignored so the facade can evolve independently
        from Streamlit controls. Direct ``PipelineConfig`` use remains preferred
        for scripts and tests.
        """

        if not values:
            return cls()
        config = cls()
        preprocess_values = _section(values, "preprocess")
        alignment_values = _section(values, "alignment")
        board_values = _section(values, "board")
        component_values = _section(values, "components", "cv_detector")
        model_values = _section(values, "model_detector", "model")
        tiling_values = _section(values, "tiling")
        crop_values = _section(values, "crops", "crop")
        classification_values = _section(values, "classification", "classifier")

        _assign_known(
            config.preprocess,
            preprocess_values,
            aliases={"clahe_clip": "clahe_clip_limit"},
        )
        if preprocess_values.get("resize_enabled") is False:
            config.preprocess.max_side = None
        denoise = preprocess_values.get("denoise")
        if isinstance(denoise, str):
            normalized_denoise = denoise.strip().lower().replace(" ", "")
            config.preprocess.denoise = normalized_denoise not in {"none", "off", "false"}
            method_aliases = {
                "nlmeans": "nlmeans",
                "fastnlmeans": "nlmeans",
                "bilateral": "bilateral",
                "gaussian": "gaussian",
            }
            if normalized_denoise in method_aliases:
                config.preprocess.denoise_method = method_aliases[normalized_denoise]
        sharpen = preprocess_values.get("sharpen")
        if isinstance(sharpen, (int, float)) and not isinstance(sharpen, bool):
            config.preprocess.sharpen = sharpen > 0
            config.preprocess.sharpen_amount = float(sharpen)

        _assign_known(
            config.alignment,
            alignment_values,
            aliases={
                "features": "orb_features",
                "match_ratio": "ratio_test",
                "ransac_threshold": "ransac_reprojection_threshold",
            },
        )
        _assign_known(config.board, board_values)
        _assign_known(
            config.cv_detector,
            component_values,
            aliases={
                "max_candidates": "max_detections",
                "iou": "nms_iou_threshold",
            },
        )
        _assign_known(
            config.model_detector,
            model_values,
            aliases={"conf": "confidence", "imgsz": "image_size", "max_det": "max_detections"},
        )
        # The local UI deliberately exposes shared detector controls in its
        # ``components`` section. Apply those values to learned-model inference
        # as well as the OpenCV proposal detector.
        _assign_known(
            config.model_detector,
            component_values,
            aliases={
                "conf": "confidence",
                "imgsz": "image_size",
                "max_det": "max_detections",
                "max_candidates": "max_detections",
            },
        )
        if config.model_detector.device == "auto":
            config.model_detector.device = None
        _assign_known(config.tiling, tiling_values)
        _assign_known(
            config.tiling,
            component_values,
            aliases={
                "tiling_mode": "mode",
                "tile_overlap": "overlap_ratio",
                "tile_trigger_scale": "auto_trigger_scale",
                "full_image_pass": "include_full_image",
                "tile_confidence": "detail_confidence",
                "merge_iou": "merge_iou_threshold",
                "seam_ios": "seam_ios_threshold",
                "containment_ios": "containment_ios_threshold",
                "cross_class_iou": "cross_class_iou_threshold",
            },
        )
        if "tile_led_confidence" in component_values:
            config.tiling.detail_class_confidence["led"] = float(
                component_values["tile_led_confidence"]
            )
        _assign_known(
            config.crop,
            crop_values,
            aliases={"padding": "padding_pixels"},
        )
        target_size = crop_values.get("target_size")
        if isinstance(target_size, (int, float)):
            side = int(target_size)
            config.crop.target_size = (side, side) if side > 0 else None
        if crop_values.get("normalize") is False:
            config.crop.target_size = None
        _assign_known(config.classification, classification_values)
        detector_mode = values.get("detector_mode")
        if detector_mode in {"auto", "cv"}:
            config.detector_mode = detector_mode
        return config


def _section(values: Mapping[str, Any], *names: str) -> dict[str, Any]:
    for name in names:
        section = values.get(name)
        if isinstance(section, Mapping):
            return dict(section)
    return dict(values)


def _assign_known(
    target: Any,
    values: Mapping[str, Any],
    aliases: Mapping[str, str] | None = None,
) -> None:
    aliases = aliases or {}
    fields = target.__dataclass_fields__
    for source_name, value in values.items():
        target_name = aliases.get(source_name, source_name)
        if target_name in fields and value is not None:
            setattr(target, target_name, value)
