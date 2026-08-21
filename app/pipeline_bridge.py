"""UI-facing adapter for the AOI pipeline.

The Streamlit app keeps images in OpenCV BGR format.  This bridge prefers the
project's ``AOIPipeline`` implementation when it is importable, while retaining
an explicitly-labelled OpenCV demo mode so that the UI can be exercised before
trained weights are available.

The fallback component boxes are *candidate regions*, not component classes or
confidence scores.  The distinction is deliberately carried in every result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import importlib
import inspect
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


@dataclass
class StageResult:
    image: np.ndarray
    mode: str
    message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    raw: Any = None


@dataclass
class BoardResult(StageResult):
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class DetectionRecord:
    detection_id: str
    label: str
    confidence: float | None
    bbox: tuple[int, int, int, int]
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: Any = None

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class DetectionResult(StageResult):
    detections: list[DetectionRecord] = field(default_factory=list)


@dataclass
class CropRecord:
    crop_id: str
    label: str
    image: np.ndarray
    bbox: tuple[int, int, int, int]
    confidence: float | None
    source: str
    raw: Any = None


@dataclass
class ClassificationRecord:
    crop_id: str
    detection_id: str
    family: str
    probability: float
    unknown_score: float
    decision: str
    top_k: list[dict[str, Any]]
    detector_hint: str | None
    model_version: str
    raw: Any = None


@dataclass
class ClassificationResult:
    classifications: list[ClassificationRecord] = field(default_factory=list)
    mode: str = "MODEL"
    message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    raw: Any = None


@dataclass
class InspectionRecipeRecord:
    recipe_root: Path
    recipe_path: Path
    board_id: str
    side: str
    schema_version: str
    coordinate_space: str
    slot_count: int
    anchor_count: int
    rejected_count: int
    production_eligible: bool
    raw: Any
    elapsed_ms: float = 0.0


@dataclass
class InspectionResult(StageResult):
    status: str = "invalid"
    alignment_status: str = "invalid"
    alignment: dict[str, Any] = field(default_factory=dict)
    slots: list[dict[str, Any]] = field(default_factory=list)
    extras: list[dict[str, Any]] = field(default_factory=list)
    json_payload: str = ""


class PipelineBridge:
    """Small compatibility layer between Streamlit and ``AOIPipeline``.

    Expected project facade::

        AOIPipeline(config=None, detector=None, model_path=None)
        preprocess(image) -> PreprocessResult
        align(image, reference=None) -> AlignmentResult
        detect_board(image) -> BoardRegion
        detect_components(image, board_region=None) -> list[Detection]
        make_crops(image, detections, output_dir=None) -> list[ComponentCrop]

    Optional keyword arguments/config are tolerated so the UI remains usable
    while the project implementation evolves.
    """

    ENGINE_CANDIDATES = (
        ("src.pipeline", "AOIPipeline"),
        ("src.aoi_pipeline", "AOIPipeline"),
        ("aoi_pipeline", "AOIPipeline"),
    )

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        detector: Any = None,
        model_path: str | None = None,
        board_model_path: str | None = None,
        classifier_model_path: str | None = None,
        classifier_manifest_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = dict(config or {})
        self.detector = detector
        self.model_path = model_path
        self.board_model_path = board_model_path
        self.classifier_model_path = classifier_model_path
        self.classifier_manifest_path = classifier_manifest_path
        self.extra = kwargs
        self.engine: Any = None
        self.engine_error: str | None = None
        self.engine_module: str | None = None
        self._load_engine()

    def _inspection_measurement_image(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Apply only the canonical lens-undistortion measurement stage.

        Inspection must not reuse presentation preprocessing: no resize,
        letterbox, denoise, or contrast operation is permitted here. The same
        calibration identity is persisted at enrollment and required at run
        time, while the returned mask tracks remap padding.
        """

        from aoi_pipeline.calibration import CameraCalibrationProfile, CameraUndistorter
        from aoi_pipeline.image_io import ensure_bgr

        source = ensure_bgr(image)
        preprocess_config = self.config.get("preprocess", self.config)
        if not bool(preprocess_config.get("undistort", False)):
            return (
                np.ascontiguousarray(source),
                np.full(source.shape[:2], 255, dtype=np.uint8),
                {"undistort": "not_requested", "calibration_sha256": None, "alpha": None},
            )
        profile_mapping = preprocess_config.get("calibration_profile")
        if not isinstance(profile_mapping, Mapping):
            raise RuntimeError(
                "Đã bật undistort cho Golden Inspection nhưng thiếu calibration_profile."
            )
        profile = CameraCalibrationProfile.from_mapping(profile_mapping)
        alpha = float(preprocess_config.get("undistort_alpha", 0.0))
        corrected = CameraUndistorter(
            profile,
            alpha=alpha,
            aspect_tolerance=float(
                preprocess_config.get("calibration_aspect_tolerance", 0.01)
            ),
        ).correct(source)
        profile_json = json.dumps(
            profile.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return (
            corrected.image,
            corrected.valid_mask,
            {
                "undistort": "pinhole",
                "calibration_sha256": sha256(profile_json.encode("utf-8")).hexdigest(),
                "alpha": float(corrected.alpha),
            },
        )

    def _assert_inspection_model_is_trusted(self, allow_trusted_pt: bool) -> None:
        """Block pickle-backed .pt loading before any Golden inference occurs."""

        detector = getattr(self.engine, "detector", None) if self.engine is not None else None
        candidates = (self.model_path, getattr(detector, "model_path", None))
        has_pt = any(
            candidate is not None and Path(str(candidate)).suffix.lower() == ".pt"
            for candidate in candidates
        )
        if has_pt and not allow_trusted_pt:
            raise RuntimeError(
                "Detector .pt chưa được xác nhận tin cậy; Golden Inspection sẽ không nạp hoặc chạy nó."
            )

    @property
    def backend_mode(self) -> str:
        if self.engine is not None:
            if self._engine_uses_demo_detector():
                return "CV DEMO"
            return "MODEL" if self.model_path else "PIPELINE"
        return "CV DEMO"

    @property
    def backend_detail(self) -> str:
        if self.engine is not None:
            return f"AOIPipeline từ {self.engine_module}"
        if self.engine_error:
            return f"Chưa nạp được AOIPipeline: {self.engine_error}"
        return "Chưa tìm thấy AOIPipeline; đang dùng fallback OpenCV."

    def _load_engine(self) -> None:
        errors: list[str] = []
        engine_cls: Any = None
        for module_name, class_name in self.ENGINE_CANDIDATES:
            try:
                module = importlib.import_module(module_name)
                engine_cls = getattr(module, class_name)
                self.engine_module = module_name
                break
            except (ImportError, AttributeError) as exc:
                errors.append(f"{module_name}: {exc}")

        if engine_cls is None:
            self.engine_error = " | ".join(errors[-2:]) if errors else "not found"
            return

        init_kwargs = {
            "config": self.config or None,
            "detector": self.detector,
            "model_path": self.model_path,
            "classifier_model_path": self.classifier_model_path,
            "classifier_manifest_path": self.classifier_manifest_path,
        }
        init_kwargs.update(self.extra)
        try:
            self.engine = _call_supported(engine_cls, **init_kwargs)
        except Exception as exc:  # External pipeline failures must not kill UI.
            self.engine = None
            self.engine_error = f"{type(exc).__name__}: {exc}"

    def _engine_uses_demo_detector(self) -> bool:
        """Identify the facade's heuristic detector without importing its type."""

        detector = getattr(self.engine, "detector", None) if self.engine is not None else None
        detector_name = type(detector).__name__.lower()
        detector_module = type(detector).__module__.lower()
        return any(
            token in f"{detector_module}.{detector_name}"
            for token in ("cvcomponent", "opencv", "candidate", "mock")
        )

    def preprocess(self, image: np.ndarray, **kwargs: Any) -> StageResult:
        started = time.perf_counter()
        preprocess_config = self.config.get("preprocess", self.config)
        undistort_required = bool(preprocess_config.get("undistort", False))
        if self.engine is not None and hasattr(self.engine, "preprocess"):
            try:
                raw = _call_supported(self.engine.preprocess, image, **kwargs)
                output = _extract_image(raw, ("image", "processed_image", "output"))
                if output is None:
                    raise ValueError("PreprocessResult không chứa ảnh đầu ra")
                return StageResult(
                    image=output,
                    mode="PIPELINE",
                    message="Tiền xử lý bởi AOIPipeline.",
                    metrics={
                        "elapsed_ms": _elapsed_ms(started),
                        **_extract_metrics(raw),
                    },
                    raw=raw,
                )
            except Exception as exc:
                if undistort_required:
                    raise RuntimeError(
                        "Hiệu chỉnh méo ống kính thất bại; pipeline không tự bỏ qua undistort: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                fallback_note = f"AOIPipeline lỗi ({type(exc).__name__}: {exc}); dùng CV demo."
            else:  # pragma: no cover - explicit for readability
                fallback_note = ""
        else:
            if undistort_required:
                raise RuntimeError(
                    "Đã bật hiệu chỉnh méo ống kính nhưng AOIPipeline không sẵn sàng; "
                    "pipeline không tự chạy ảnh chưa undistort."
                )
            fallback_note = "AOIPipeline chưa sẵn sàng; dùng CV demo."

        output = _fallback_preprocess(image, preprocess_config)
        return StageResult(
            image=output,
            mode="CV DEMO",
            message=fallback_note,
            metrics={
                "elapsed_ms": _elapsed_ms(started),
                "input_size": f"{image.shape[1]}×{image.shape[0]}",
                "output_size": f"{output.shape[1]}×{output.shape[0]}",
            },
        )

    def align(
        self,
        image: np.ndarray,
        reference: np.ndarray | None = None,
        **kwargs: Any,
    ) -> StageResult:
        started = time.perf_counter()
        if reference is None:
            return StageResult(
                image=image.copy(),
                mode="SKIPPED",
                message="Chưa có Golden Image/reference; giữ nguyên ảnh đầu vào.",
                metrics={"elapsed_ms": _elapsed_ms(started)},
            )

        if self.engine is not None and hasattr(self.engine, "align"):
            try:
                raw = _call_supported(self.engine.align, image, reference=reference, **kwargs)
                output = _extract_image(raw, ("image", "aligned_image", "warped_image", "output"))
                if output is None:
                    raise ValueError("AlignmentResult không chứa ảnh đầu ra")
                success = _attr(raw, ("success",), None)
                method = str(_attr(raw, ("method",), "unknown"))
                core_message = str(_attr(raw, ("message",), "")).strip()
                metrics = {
                    "elapsed_ms": _elapsed_ms(started),
                    **_extract_metrics(raw),
                    "method": method,
                }
                for metric_name in (
                    "source_keypoints",
                    "reference_keypoints",
                    "good_matches",
                    "inliers",
                    "inlier_ratio",
                    "correlation",
                ):
                    metric_value = _attr(raw, (metric_name,), None)
                    if metric_value is not None:
                        metrics[metric_name] = metric_value
                if success is not None:
                    metrics["success"] = bool(success)
                if success is False:
                    return StageResult(
                        image=output,
                        mode="CV FALLBACK",
                        message=(
                            f"Alignment không đạt gate ({method}). "
                            f"{core_message or 'Chỉ dùng ảnh fallback; cần kiểm tra trước bước tiếp theo.'}"
                        ),
                        metrics=metrics,
                        raw=raw,
                    )
                return StageResult(
                    image=output,
                    mode="PIPELINE",
                    message=core_message or f"Căn chỉnh bởi AOIPipeline ({method}).",
                    metrics=metrics,
                    raw=raw,
                )
            except Exception as exc:
                fallback_note = f"AOIPipeline lỗi ({type(exc).__name__}: {exc}); dùng ORB demo."
        else:
            fallback_note = "AOIPipeline chưa sẵn sàng; dùng ORB demo."

        output, metrics, warning = _fallback_align(
            image,
            reference,
            self.config.get("alignment", self.config),
        )
        return StageResult(
            image=output,
            mode="CV DEMO",
            message=" ".join(part for part in (fallback_note, warning) if part),
            metrics={"elapsed_ms": _elapsed_ms(started), **metrics},
        )

    def detect_board(self, image: np.ndarray, **kwargs: Any) -> BoardResult:
        started = time.perf_counter()
        if self.engine is not None and hasattr(self.engine, "detect_board"):
            try:
                raw = _call_supported(self.engine.detect_board, image, **kwargs)
                bbox = _extract_bbox(raw, image.shape)
                annotated = _draw_board(image, bbox, "PCB")
                method = str(_attr(raw, ("method",), "unknown"))
                confidence = _attr(raw, ("confidence",), None)
                fallback = method == "full_image_fallback"
                metrics = {
                    "elapsed_ms": _elapsed_ms(started),
                    "board_coverage": _bbox_coverage(bbox, image.shape),
                    **_extract_metrics(raw),
                    "method": method,
                }
                if confidence is not None:
                    metrics["confidence"] = confidence
                return BoardResult(
                    image=annotated,
                    bbox=bbox,
                    mode="CV FALLBACK" if fallback else ("MODEL" if self.board_model_path else "PIPELINE"),
                    message=(
                        "Không tìm thấy contour PCB; đang dùng toàn ảnh làm ROI."
                        if fallback
                        else f"Vùng PCB từ AOIPipeline ({method})."
                    ),
                    metrics=metrics,
                    raw=raw,
                )
            except Exception as exc:
                fallback_note = f"AOIPipeline lỗi ({type(exc).__name__}: {exc}); dùng CV demo."
        else:
            fallback_note = "AOIPipeline chưa sẵn sàng; dùng CV demo."

        bbox = _fallback_board_bbox(image)
        annotated = _draw_board(image, bbox, "PCB candidate · CV demo")
        return BoardResult(
            image=annotated,
            bbox=bbox,
            mode="CV DEMO",
            message=f"{fallback_note} Bounding box chỉ là ước lượng hình học.",
            metrics={
                "elapsed_ms": _elapsed_ms(started),
                "board_coverage": _bbox_coverage(bbox, image.shape),
            },
        )

    def detect_components(
        self,
        image: np.ndarray,
        board_region: BoardResult | Any | None = None,
        **kwargs: Any,
    ) -> DetectionResult:
        started = time.perf_counter()
        external_board = board_region.raw if isinstance(board_region, BoardResult) and board_region.raw is not None else board_region
        if self.engine is not None and hasattr(self.engine, "detect_components"):
            try:
                raw = _call_supported(
                    self.engine.detect_components,
                    image,
                    board_region=external_board,
                    **kwargs,
                )
                # Preserve the pipeline-provided source.  In particular,
                # ``opencv_candidate`` must never be presented as a model hit.
                engine_demo = self._engine_uses_demo_detector()
                detections = _normalize_detections(
                    raw,
                    image.shape,
                    source=(
                        "opencv_candidate"
                        if engine_demo
                        else ("model" if self.model_path else "pipeline")
                    ),
                )
                core_metrics = getattr(self.engine, "last_detection_metrics", {})
                if not isinstance(core_metrics, Mapping):
                    core_metrics = {}
                annotated = _draw_detections(image, detections)
                component_config = self.config.get("components", self.config)
                if bool(component_config.get("show_tile_grid", False)):
                    annotated = _draw_tile_grid(annotated, core_metrics.get("tile_regions", []))
                demo_count = sum(_is_demo_source(item.source) for item in detections)
                inferred_count = len(detections) - demo_count
                if demo_count and inferred_count:
                    result_mode = "MIXED DEMO"
                    result_message = (
                        "AOIPipeline trả về cả model detections và OpenCV candidates. "
                        "Các candidate không phải nhận dạng đáng tin cậy."
                    )
                elif demo_count or engine_demo:
                    result_mode = "CV DEMO"
                    result_message = (
                        "CV candidate demo – không phải nhận dạng đáng tin cậy; "
                        "không coi candidate là model detection."
                    )
                else:
                    result_mode = "MODEL" if self.model_path else "PIPELINE"
                    result_message = "Kết quả nhận dạng từ AOIPipeline."
                if core_metrics.get("tiling_applied"):
                    effective_size = core_metrics.get("effective_tile_size", "?")
                    result_message += (
                        f" Adaptive tiling đã chạy {int(core_metrics.get('tile_count', 0))} "
                        f"khung {effective_size}px và gộp detection trong tọa độ ảnh phân tích."
                    )
                return DetectionResult(
                    image=annotated,
                    detections=detections,
                    mode=result_mode,
                    message=result_message,
                    metrics={
                        "elapsed_ms": _elapsed_ms(started),
                        "count": len(detections),
                        **dict(core_metrics),
                    },
                    raw=raw,
                )
            except Exception as exc:
                if self.model_path:
                    raise RuntimeError(
                        f"Model đã được chọn nhưng inference thất bại; không tự chuyển sang CV demo: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                fallback_note = f"AOIPipeline/model lỗi ({type(exc).__name__}: {exc})."
        else:
            fallback_note = "AOIPipeline/model chưa sẵn sàng."

        if self.model_path:
            detail = self.engine_error or fallback_note
            raise RuntimeError(
                f"Model đã được chọn nhưng backend không khởi tạo được; "
                f"không tự chuyển sang CV demo: {detail}"
            )

        bbox = (
            board_region.bbox
            if isinstance(board_region, BoardResult)
            else (0, 0, image.shape[1], image.shape[0])
        )
        config = self.config.get("components", self.config)
        detections = _fallback_component_candidates(image, bbox, config)
        annotated = _draw_detections(image, detections)
        model_note = " Có file model nhưng backend không nạp được model." if self.model_path else ""
        return DetectionResult(
            image=annotated,
            detections=detections,
            mode="CV DEMO",
            message=(
                f"{fallback_note}{model_note} CV candidate demo – không phải nhận dạng "
                "đáng tin cậy; nhãn/confidence không có ý nghĩa model."
            ),
            metrics={"elapsed_ms": _elapsed_ms(started), "count": len(detections)},
        )

    def make_crops(
        self,
        image: np.ndarray,
        detections: Sequence[DetectionRecord],
        output_dir: str | None = None,
        **kwargs: Any,
    ) -> list[CropRecord]:
        """Make crops with the core recipe, or the UI fallback if no core exists.

        A core failure is surfaced instead of silently substituting crops with
        different geometry or normalization semantics.
        """

        config = dict(self.config.get("crops", {}))
        config.update(kwargs)
        padding = max(0, int(config.get("padding", 4)))
        target_size = int(config.get("target_size", 224))
        normalize = bool(config.get("normalize", True))

        if self.engine is not None and hasattr(self.engine, "make_crops"):
            raw_detections = [item.raw if item.raw is not None else item for item in detections]
            try:
                raw_crops = _call_supported(
                    self.engine.make_crops,
                    image,
                    raw_detections,
                    output_dir=output_dir,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"AOIPipeline crop thất bại; không thay bằng recipe khác: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(raw_crops, Sequence):
                raise RuntimeError("AOIPipeline.make_crops phải trả về một sequence")

            detections_by_id = {item.detection_id: item for item in detections}
            core_records: list[CropRecord] = []
            for index, raw_crop in enumerate(raw_crops):
                crop_image = _extract_image(raw_crop, ("image", "crop", "output"))
                if crop_image is None:
                    raise RuntimeError(f"ComponentCrop thứ {index} không chứa ảnh")
                detection_id = str(
                    _attr(raw_crop, ("detection_id", "id"), f"det_{index + 1:04d}")
                )
                detection = detections_by_id.get(detection_id)
                if detection is None and index < len(detections):
                    detection = detections[index]
                label = str(
                    _attr(raw_crop, ("label", "class_name"), detection.label if detection else "component")
                )
                confidence_value = _attr(
                    raw_crop,
                    ("confidence", "score"),
                    detection.confidence if detection else None,
                )
                confidence = (
                    float(confidence_value) if confidence_value is not None else None
                )
                bbox = detection.bbox if detection is not None else _extract_bbox(raw_crop, image.shape)
                source = detection.source if detection is not None else "pipeline"
                core_records.append(
                    CropRecord(
                        crop_id=f"crop_{index + 1:04d}",
                        label=label,
                        image=crop_image,
                        bbox=bbox,
                        confidence=confidence,
                        source=source,
                        raw=raw_crop,
                    )
                )
            return core_records

        crops: list[CropRecord] = []
        for index, detection in enumerate(detections):
            x1, y1, x2, y2 = _pad_bbox(detection.bbox, padding, image.shape)
            crop = image[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue
            if normalize and target_size > 0:
                crop = _letterbox(crop, target_size)
            crops.append(
                CropRecord(
                    crop_id=f"crop_{index + 1:04d}",
                    label=detection.label,
                    image=crop,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    source=detection.source,
                    raw=None,
                )
            )
        return crops

    def classify_components(
        self,
        crops: Sequence[CropRecord],
        **kwargs: Any,
    ) -> ClassificationResult:
        """Run the real step-6.1 classifier; there is deliberately no CV fallback."""

        started = time.perf_counter()
        if not self.classifier_model_path or not self.classifier_manifest_path:
            raise RuntimeError(
                "Bước 6.1 cần đủ best.onnx và model_manifest.json từ notebook train."
            )
        if self.engine is None or not hasattr(self.engine, "classify_components"):
            detail = self.engine_error or "AOIPipeline không hỗ trợ bước 6.1"
            raise RuntimeError(
                "Classifier đã được chọn nhưng backend không khởi tạo được; "
                f"không tạo nhãn giả: {detail}"
            )
        raw_crops = [item.raw for item in crops]
        if any(item is None for item in raw_crops):
            raise RuntimeError(
                "Crop không mang metadata lõi của bước 5; hãy chạy lại bước 5 trước bước 6.1."
            )
        try:
            raw = _call_supported(self.engine.classify_components, raw_crops, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                "Classifier inference thất bại; không lấy nhãn detector thay thế: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(raw, Sequence):
            raise RuntimeError("AOIPipeline.classify_components phải trả về một sequence")
        records: list[ClassificationRecord] = []
        for index, item in enumerate(raw):
            top_k_raw = _attr(item, ("top_k",), [])
            top_k: list[dict[str, Any]] = []
            for score in top_k_raw if isinstance(top_k_raw, Sequence) else []:
                top_k.append(
                    {
                        "label": str(_attr(score, ("label", "family"), "")),
                        "probability": float(_attr(score, ("probability", "score"), 0.0)),
                    }
                )
            records.append(
                ClassificationRecord(
                    crop_id=str(_attr(item, ("crop_id",), f"crop_{index + 1:04d}")),
                    detection_id=str(_attr(item, ("detection_id",), "")),
                    family=str(_attr(item, ("family", "label"), "unknown")),
                    probability=float(_attr(item, ("probability", "confidence"), 0.0)),
                    unknown_score=float(_attr(item, ("unknown_score",), 1.0)),
                    decision=str(_attr(item, ("decision",), "unknown")),
                    top_k=top_k,
                    detector_hint=_attr(item, ("detector_hint",), None),
                    model_version=str(_attr(item, ("model_version",), "unknown")),
                    raw=item,
                )
            )
        counts: dict[str, int] = {}
        for item in records:
            counts[item.decision] = counts.get(item.decision, 0) + 1
        return ClassificationResult(
            classifications=records,
            mode="MODEL",
            message=(
                f"Đã phân loại {len(records)} crop bằng ONNX; kết quả review/unknown "
                "cần được kiểm tra, không tự động coi là nhãn đúng."
            ),
            metrics={
                "elapsed_ms": _elapsed_ms(started),
                "count": len(records),
                "decisions": counts,
            },
            raw=raw,
        )

    def build_inspection_recipe(
        self,
        golden_image: np.ndarray,
        output_dir: str | Path,
        *,
        board_id: str,
        side: str,
        pixels_per_mm_x: float = 1.0,
        pixels_per_mm_y: float = 1.0,
        calibration_verified: bool = False,
        roi_padding_px: int = 12,
        search_margin_px: int = 16,
        max_abs_dx_mm: float = 0.20,
        max_abs_dy_mm: float = 0.20,
        max_abs_angle_deg: float | None = None,
        rotation_period_deg: float | None = None,
        min_ssim: float = 0.88,
        max_diff_ratio: float = 0.08,
        max_edge_diff_ratio: float = 0.10,
        max_blob_area_px: int = 45,
        min_valid_overlap_ratio: float = 0.88,
        anchor_template_size_px: int = 65,
        anchor_search_margin_px: int = 48,
        allow_demo_sources: bool = True,
        allow_trusted_pt: bool = False,
    ) -> InspectionRecipeRecord:
        """Forward Golden enrollment to the versioned core recipe service."""

        if self.engine is None or not hasattr(self.engine, "detect_components"):
            detail = self.engine_error or "AOIPipeline không hỗ trợ detection"
            raise RuntimeError(f"Không thể build recipe nếu core không sẵn sàng: {detail}")
        if not isinstance(board_id, str) or not board_id.strip():
            raise ValueError("board_id là bắt buộc khi build Golden recipe")
        if side not in {"top", "bottom"}:
            raise ValueError("side phải là 'top' hoặc 'bottom'")
        self._assert_inspection_model_is_trusted(allow_trusted_pt)
        from aoi_pipeline import (
            AppearanceThresholds,
            Detection,
            MetrologyCalibration,
            PositionTolerance,
            create_grid_alignment_recipe,
            create_recipe,
        )

        started = time.perf_counter()
        measurement_golden, _golden_valid_mask, measurement_domain = (
            self._inspection_measurement_image(golden_image)
        )
        destination = Path(output_dir).expanduser().resolve()
        if destination.exists():
            raise RuntimeError(
                f"Recipe output đã tồn tại và không được ghi đè: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
        )
        try:
            detections = list(
                _call_supported(
                    self.engine.detect_components,
                    measurement_golden,
                    frame_id="golden_enrollment",
                )
            )
        except Exception as exc:
            shutil.rmtree(staging, ignore_errors=True)
            raise RuntimeError(
                "Detector không tạo được slot Golden; không dùng bbox giả thay thế: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not detections or not all(isinstance(item, Detection) for item in detections):
            shutil.rmtree(staging, ignore_errors=True)
            raise RuntimeError("Detector không trả về Detection hợp lệ để tạo Golden recipe")

        try:
            alignment = create_grid_alignment_recipe(
                measurement_golden,
                staging,
                template_size_px=int(anchor_template_size_px),
                search_margin_px=int(anchor_search_margin_px),
            )
            from aoi_pipeline.detectors import detector_identifier

            runtime_detector = getattr(self.engine, "detector", None)
            if runtime_detector is None:
                raise RuntimeError("AOIPipeline không cung cấp detector runtime")
            built = create_recipe(
                measurement_golden,
                detections,
                staging,
                board_id=board_id.strip(),
                side=side,
                metrology=MetrologyCalibration(
                    float(pixels_per_mm_x),
                    float(pixels_per_mm_y),
                    verified=bool(calibration_verified),
                ),
                roi_padding_px=int(roi_padding_px),
                search_margin_px=int(search_margin_px),
                position_tolerance=PositionTolerance(
                    float(max_abs_dx_mm),
                    float(max_abs_dy_mm),
                    None if max_abs_angle_deg is None else float(max_abs_angle_deg),
                ),
                appearance_thresholds=AppearanceThresholds(
                    float(min_ssim),
                    float(max_diff_ratio),
                    float(max_edge_diff_ratio),
                    int(max_blob_area_px),
                    float(min_valid_overlap_ratio),
                ),
                alignment=alignment,
                model_identifiers={
                    "component_detector": detector_identifier(runtime_detector)
                },
                allow_demo_sources=bool(allow_demo_sources),
                rotation_period_deg=rotation_period_deg,
                measurement_metadata=measurement_domain,
            )
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        try:
            os.replace(staging, destination)
        except OSError as exc:
            shutil.rmtree(staging, ignore_errors=True)
            raise RuntimeError(f"Không thể publish recipe atomically: {exc}") from exc
        return _inspection_recipe_record(
            built.recipe,
            destination,
            elapsed_ms=_elapsed_ms(started),
        )

    def load_inspection_recipe(
        self,
        recipe_path: str | Path,
    ) -> InspectionRecipeRecord:
        """Load and validate an existing local recipe with all of its assets."""

        from aoi_pipeline import load_recipe

        path = Path(recipe_path).expanduser().resolve()
        recipe = load_recipe(path, validate_assets=True)
        return _inspection_recipe_record(recipe, path.parent)

    def inspect_board(
        self,
        test_image: np.ndarray,
        recipe_record: InspectionRecipeRecord,
        *,
        require_production_eligible: bool = False,
        allow_trusted_pt: bool = False,
    ) -> InspectionResult:
        """Run the core inspector and normalize its contract for Streamlit."""

        if self.engine is None:
            detail = self.engine_error or "AOIPipeline không sẵn sàng"
            raise RuntimeError(f"Không thể inspect board: {detail}")
        self._assert_inspection_model_is_trusted(allow_trusted_pt)
        from aoi_pipeline import (
            AOIInspector,
            InspectionConfig,
            render_inspection_overlay,
        )

        started = time.perf_counter()
        measurement_test, source_valid_mask, measurement_domain = (
            self._inspection_measurement_image(test_image)
        )
        enrolled_domain = recipe_record.raw.enrollment.get("measurement_domain", {})
        if enrolled_domain != measurement_domain:
            raise RuntimeError(
                "Calibration/undistortion đang dùng không khớp Golden recipe; inspection dừng fail-closed."
            )
        inspector = AOIInspector.from_pipeline(
            self.engine,
            config=InspectionConfig(
                require_production_eligible=bool(require_production_eligible)
            ),
        )
        run = inspector.inspect(
            measurement_test,
            recipe_record.raw,
            recipe_record.recipe_root,
            source_valid_mask=source_valid_mask,
        )
        payload = run.to_dict()
        overlay = render_inspection_overlay(run, recipe_record.raw)
        if overlay is None:
            overlay = np.ascontiguousarray(measurement_test.copy())
        status_messages = {
            "pass": "Board PASS theo toàn bộ status do core trả về.",
            "ng": "Board NG; xem riêng Position và Appearance theo từng slot.",
            "review": "Board cần review vì có phép đo không đủ tin cậy.",
            "invalid": "Inspection INVALID; pipeline đã fail closed.",
        }
        return InspectionResult(
            image=overlay,
            mode="INSPECTION",
            message=status_messages.get(run.status, run.reason),
            metrics={
                "elapsed_ms": _elapsed_ms(started),
                "slot_count": len(run.slots),
                "extra_count": len(run.extras),
            },
            raw=run,
            status=run.status,
            alignment_status=run.alignment.status,
            alignment=payload["alignment"],
            slots=list(payload["slots"]),
            extras=list(payload["extras"]),
            json_payload=run.to_json(),
        )


def _inspection_recipe_record(
    recipe: Any,
    root: Path,
    *,
    elapsed_ms: float = 0.0,
) -> InspectionRecipeRecord:
    return InspectionRecipeRecord(
        recipe_root=root,
        recipe_path=root / "recipe.json",
        board_id=str(recipe.board_id),
        side=str(recipe.side),
        schema_version=str(recipe.schema_version),
        coordinate_space=str(recipe.coordinate_space),
        slot_count=len(recipe.slots),
        anchor_count=len(recipe.alignment.anchors),
        rejected_count=len(recipe.rejected_detections),
        production_eligible=bool(recipe.production_eligible),
        raw=recipe,
        elapsed_ms=float(elapsed_ms),
    )


def _call_supported(callable_obj: Any, *args: Any, **kwargs: Any) -> Any:
    """Call a changing project facade without forwarding unsupported kwargs."""

    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return callable_obj(*args, **kwargs)

    accepts_var_kwargs = any(
        item.kind is inspect.Parameter.VAR_KEYWORD
        for item in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return callable_obj(*args, **kwargs)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return callable_obj(*args, **supported)


def _attr(value: Any, names: Iterable[str], default: Any = None) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _extract_image(raw: Any, names: Sequence[str]) -> np.ndarray | None:
    if isinstance(raw, np.ndarray):
        return raw
    candidate = _attr(raw, names)
    return candidate if isinstance(candidate, np.ndarray) else None


def _extract_metrics(raw: Any) -> dict[str, Any]:
    metrics = _attr(raw, ("metrics", "metadata", "stats"), {})
    return dict(metrics) if isinstance(metrics, Mapping) else {}


def _extract_bbox(raw: Any, shape: Sequence[int]) -> tuple[int, int, int, int]:
    candidate = _attr(raw, ("bbox", "xyxy", "box", "bounds"))
    if candidate is None:
        values = [_attr(raw, (name,)) for name in ("x1", "y1", "x2", "y2")]
        candidate = values if all(value is not None for value in values) else None
    if hasattr(candidate, "tolist"):
        candidate = candidate.tolist()
    if isinstance(candidate, Sequence) and len(candidate) >= 4:
        return _clip_bbox(tuple(int(round(float(value))) for value in candidate[:4]), shape)
    return (0, 0, int(shape[1]), int(shape[0]))


def _normalize_detections(
    raw: Any,
    shape: Sequence[int],
    source: str,
) -> list[DetectionRecord]:
    items = _attr(raw, ("detections", "items", "results"), raw)
    if items is None:
        return []
    if isinstance(items, Mapping) or not isinstance(items, Iterable):
        items = [items]
    records: list[DetectionRecord] = []
    for index, item in enumerate(items):
        bbox = _extract_bbox(item, shape)
        label = str(_attr(item, ("label", "class_name", "name", "class_id"), "component"))
        confidence_value = _attr(item, ("confidence", "score", "conf", "probability"))
        try:
            confidence = float(confidence_value) if confidence_value is not None else None
        except (TypeError, ValueError):
            confidence = None
        detection_id = str(_attr(item, ("detection_id", "id"), f"det_{index + 1:04d}"))
        item_source = str(
            _attr(
                item,
                ("source", "backend", "origin", "detector_source"),
                source,
            )
        )
        item_metadata = _attr(item, ("metadata",), {})
        if not isinstance(item_metadata, Mapping):
            item_metadata = {}
        records.append(
            DetectionRecord(
                detection_id=detection_id,
                label=label,
                confidence=confidence,
                bbox=bbox,
                source=item_source,
                metadata=dict(item_metadata),
                raw=item,
            )
        )
    return records


def _fallback_preprocess(image: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    output = image.copy()
    if bool(config.get("resize_enabled", True)):
        max_side = max(320, int(config.get("max_side", 1600)))
        height, width = output.shape[:2]
        scale = min(1.0, max_side / max(height, width))
        if scale < 1.0:
            output = cv2.resize(
                output,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )

    denoise = str(config.get("denoise", "Bilateral"))
    strength = max(1, int(config.get("denoise_strength", 5)))
    if denoise == "Gaussian":
        kernel = strength if strength % 2 else strength + 1
        output = cv2.GaussianBlur(output, (kernel, kernel), 0)
    elif denoise == "Bilateral":
        output = cv2.bilateralFilter(output, 7, strength * 8, strength * 8)
    elif denoise == "NLMeans":
        output = cv2.fastNlMeansDenoisingColored(output, None, strength, strength, 7, 21)

    if bool(config.get("white_balance", True)):
        means = output.reshape(-1, 3).mean(axis=0)
        target = float(means.mean())
        gains = target / np.maximum(means, 1.0)
        output = np.clip(output.astype(np.float32) * gains, 0, 255).astype(np.uint8)

    if bool(config.get("clahe", True)):
        lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=float(config.get("clahe_clip", 2.0)),
            tileGridSize=(8, 8),
        )
        lightness = clahe.apply(lightness)
        output = cv2.cvtColor(cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    if bool(config.get("normalize", False)):
        output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)

    sharpen = float(config.get("sharpen", 0.35))
    if sharpen > 0:
        blurred = cv2.GaussianBlur(output, (0, 0), 1.2)
        output = cv2.addWeighted(output, 1.0 + sharpen, blurred, -sharpen, 0)
    return output


def _fallback_align(
    image: np.ndarray,
    reference: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any], str]:
    target_height, target_width = reference.shape[:2]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=max(500, int(config.get("features", 3000))))
    keypoints_a, descriptors_a = orb.detectAndCompute(gray_image, None)
    keypoints_b, descriptors_b = orb.detectAndCompute(gray_reference, None)
    metrics: dict[str, Any] = {
        "keypoints_input": len(keypoints_a),
        "keypoints_reference": len(keypoints_b),
        "good_matches": 0,
        "inliers": 0,
    }
    if descriptors_a is None or descriptors_b is None:
        resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        return resized, metrics, "Không đủ descriptor; chỉ resize theo reference."

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    ratio = float(config.get("match_ratio", 0.75))
    good = [first for first, second in pairs if first.distance < ratio * second.distance]
    metrics["good_matches"] = len(good)
    if len(good) < 8:
        resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        return resized, metrics, "Không đủ match cho homography; chỉ resize theo reference."

    points_a = np.float32([keypoints_a[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
    points_b = np.float32([keypoints_b[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(
        points_a,
        points_b,
        cv2.RANSAC,
        float(config.get("ransac_threshold", 4.0)),
    )
    if matrix is None:
        resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        return resized, metrics, "Không tìm được homography; chỉ resize theo reference."
    metrics["inliers"] = int(mask.sum()) if mask is not None else 0
    warped = cv2.warpPerspective(image, matrix, (target_width, target_height))
    return warped, metrics, ""


def _fallback_board_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(gray, 40, 120)
    kernel_size = max(5, int(round(min(height, width) * 0.02)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, width, height)
    candidates = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in candidates:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width * box_height >= 0.12 * width * height:
            return _clip_bbox((x, y, x + box_width, y + box_height), image.shape)
    return (0, 0, width, height)


def _fallback_component_candidates(
    image: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    config: Mapping[str, Any],
) -> list[DetectionRecord]:
    x1, y1, x2, y2 = _clip_bbox(board_bbox, image.shape)
    board = image[y1:y2, x1:x2]
    if board.size == 0:
        return []
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    equalized = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    edges = cv2.Canny(equalized, 45, 130)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    board_area = max(1, board.shape[0] * board.shape[1])
    min_area = max(12, int(float(config.get("min_area_ratio", 0.00004)) * board_area))
    max_area = int(float(config.get("max_area_ratio", 0.035)) * board_area)
    max_candidates = max(1, int(config.get("max_candidates", 160)))
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        bx, by, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        aspect = max(bw / max(1, bh), bh / max(1, bw))
        if min_area <= area <= max_area and aspect <= 12.0 and bw >= 4 and bh >= 4:
            boxes.append((bx + x1, by + y1, bx + bw + x1, by + bh + y1))
    boxes.sort(key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
    boxes = _nms_boxes(boxes, iou_threshold=0.35)[:max_candidates]
    boxes.sort(key=lambda box: (box[1], box[0]))
    return [
        DetectionRecord(
            detection_id=f"candidate_{index + 1:04d}",
            label="candidate",
            confidence=None,
            bbox=box,
            source="cv-demo",
        )
        for index, box in enumerate(boxes)
    ]


def _draw_board(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    label: str,
) -> np.ndarray:
    output = image.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(output, (x1, y1), (x2, y2), (245, 182, 48), 3)
    _draw_label(output, label, x1, y1, (245, 182, 48))
    return output


def _draw_detections(image: np.ndarray, detections: Sequence[DetectionRecord]) -> np.ndarray:
    output = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        color = (43, 166, 255) if _is_demo_source(detection.source) else (45, 196, 128)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        confidence = "" if detection.confidence is None else f" {detection.confidence:.2f}"
        _draw_label(output, f"{detection.label}{confidence}", x1, y1, color)
    return output


def _draw_tile_grid(image: np.ndarray, tile_regions: Any) -> np.ndarray:
    """Draw optional debug geometry without changing detection coordinates."""

    output = image.copy()
    if not isinstance(tile_regions, Sequence):
        return output
    for index, region in enumerate(tile_regions):
        if not isinstance(region, Mapping):
            continue
        xyxy = region.get("xyxy")
        if not isinstance(xyxy, Sequence) or len(xyxy) < 4:
            continue
        x1, y1, x2, y2 = (int(round(float(value))) for value in xyxy[:4])
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 120, 40), 1)
        ownership_xyxy = region.get("ownership_xyxy")
        if isinstance(ownership_xyxy, Sequence) and len(ownership_xyxy) >= 4:
            ox1, oy1, ox2, oy2 = (
                int(round(float(value))) for value in ownership_xyxy[:4]
            )
            cv2.rectangle(output, (ox1, oy1), (ox2, oy2), (190, 70, 255), 1)
        _draw_label(
            output,
            str(region.get("tile_id", f"tile_{index:03d}")),
            x1,
            y1,
            (255, 120, 40),
        )
    return output


def _is_demo_source(source: str | None) -> bool:
    normalized = str(source or "").strip().lower().replace("-", "_")
    return any(
        token in normalized
        for token in ("opencv", "cv_demo", "candidate", "fallback", "heuristic")
    )


def _draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    top = max(0, y - height - baseline - 6)
    cv2.rectangle(image, (x, top), (x + width + 8, top + height + baseline + 6), color, -1)
    cv2.putText(
        image,
        text,
        (x + 4, top + height + 1),
        font,
        font_scale,
        (12, 20, 31),
        thickness,
        cv2.LINE_AA,
    )


def _nms_boxes(
    boxes: Sequence[tuple[int, int, int, int]],
    iou_threshold: float,
) -> list[tuple[int, int, int, int]]:
    kept: list[tuple[int, int, int, int]] = []
    for candidate in boxes:
        if all(_iou(candidate, existing) < iou_threshold for existing in kept):
            kept.append(candidate)
    return kept


def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _pad_bbox(
    bbox: tuple[int, int, int, int],
    padding: int,
    shape: Sequence[int],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return _clip_bbox((x1 - padding, y1 - padding, x2 + padding, y2 + padding), shape)


def _clip_bbox(
    bbox: tuple[int, int, int, int],
    shape: Sequence[int],
) -> tuple[int, int, int, int]:
    height, width = int(shape[0]), int(shape[1])
    x1, y1, x2, y2 = bbox
    x1 = min(max(0, x1), max(0, width - 1))
    y1 = min(max(0, y1), max(0, height - 1))
    x2 = min(max(x1 + 1, x2), width)
    y2 = min(max(y1 + 1, y2), height)
    return (x1, y1, x2, y2)


def _bbox_coverage(bbox: tuple[int, int, int, int], shape: Sequence[int]) -> float:
    area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
    return round(area / max(1, int(shape[0]) * int(shape[1])), 4)


def _letterbox(image: np.ndarray, target_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(target_size / max(1, width), target_size / max(1, height))
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.full((target_size, target_size, 3), 24, dtype=np.uint8)
    offset_x = (target_size - resized_width) // 2
    offset_y = (target_size - resized_height) // 2
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return canvas


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 2)
