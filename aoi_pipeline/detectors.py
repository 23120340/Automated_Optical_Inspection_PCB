"""Step 4: interchangeable component detector adapters.

The CV detector emits proposals labelled ``component_candidate``. It is useful
for testing steps 0-5 before training, but it is not a replacement for the
Kaggle-trained model. ``UltralyticsDetector`` supports both ``.pt`` and
``.onnx`` artifacts and never silently falls back when a configured model fails.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import CVDetectorConfig, ModelDetectorConfig
from .exceptions import DetectorConfigurationError, ModelDependencyError
from .image_io import ensure_bgr
from .models import BoundingBox, Detection


class ComponentDetector(ABC):
    """Minimal adapter interface shared by local and learned detectors."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        raise NotImplementedError


class MockComponentDetector(ComponentDetector):
    """Deterministic detector for UI demos and unit tests."""

    def __init__(
        self,
        detections: Sequence[Detection] | Callable[[np.ndarray], Sequence[Detection]],
    ) -> None:
        self._detections = detections

    def detect(self, image: np.ndarray) -> list[Detection]:
        bgr = ensure_bgr(image)
        detections = self._detections(bgr) if callable(self._detections) else self._detections
        width, height = bgr.shape[1], bgr.shape[0]
        output: list[Detection] = []
        for detection in detections:
            bbox = detection.bbox.clamp(width, height)
            if bbox.width <= 0 or bbox.height <= 0:
                continue
            output.append(
                Detection(
                    label=detection.label,
                    confidence=detection.confidence,
                    bbox=bbox,
                    class_id=detection.class_id,
                    source="mock",
                    detection_id=detection.detection_id,
                    metadata=dict(detection.metadata),
                )
            )
        return output


class CVComponentDetector(ComponentDetector):
    """Find component-like contour proposals without claiming a component class."""

    def __init__(self, config: CVDetectorConfig | None = None) -> None:
        self.config = config or CVDetectorConfig()

    def detect(self, image: np.ndarray) -> list[Detection]:
        bgr = ensure_bgr(image)
        height, width = bgr.shape[:2]
        image_area = float(height * width)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        masks = _component_masks(gray, self.config.morphology_kernel)
        proposals: list[Detection] = []

        for mask_name, mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, box_width, box_height = cv2.boundingRect(contour)
                if box_width < self.config.min_width or box_height < self.config.min_height:
                    continue
                box_area = float(box_width * box_height)
                area_ratio = box_area / image_area
                if not self.config.min_area_ratio <= area_ratio <= self.config.max_area_ratio:
                    continue
                aspect = max(box_width / box_height, box_height / box_width)
                if aspect > self.config.max_aspect_ratio:
                    continue
                contour_area = abs(float(cv2.contourArea(contour)))
                fill = float(np.clip(contour_area / max(box_area, 1.0), 0.0, 1.0))
                contrast = _local_contrast(gray, x, y, box_width, box_height)
                confidence = float(np.clip(0.16 + 0.38 * fill + 0.34 * contrast, 0.05, 0.88))
                proposals.append(
                    Detection(
                        label="component_candidate",
                        confidence=confidence,
                        bbox=BoundingBox(float(x), float(y), float(x + box_width), float(y + box_height)),
                        source="opencv_candidate",
                        metadata={
                            "proposal_mask": mask_name,
                            "area_ratio": area_ratio,
                            "fill_ratio": fill,
                            "classification_ready": False,
                        },
                    )
                )

        kept = non_maximum_suppression(proposals, self.config.nms_iou_threshold)
        return kept[: max(0, int(self.config.max_detections))]


class UltralyticsDetector(ComponentDetector):
    """Adapter around an Ultralytics detection model (PyTorch or ONNX)."""

    SUPPORTED_SUFFIXES = {".pt", ".onnx"}

    def __init__(
        self,
        model_path: str | Path,
        config: ModelDetectorConfig | None = None,
        *,
        model: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.config = config or ModelDetectorConfig()
        if self.model_path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
            raise DetectorConfigurationError(
                "Component model must be an Ultralytics-compatible .pt or .onnx file"
            )
        if model is None and not self.model_path.is_file():
            raise DetectorConfigurationError(f"Component model does not exist: {self.model_path}")
        self._model = model

    def detect(
        self, image: np.ndarray, *, confidence: float | None = None
    ) -> list[Detection]:
        bgr = ensure_bgr(image)
        model = self._get_model()
        kwargs: dict[str, Any] = {
            "source": bgr,
            "conf": float(self.config.confidence if confidence is None else confidence),
            "iou": float(self.config.iou),
            "imgsz": int(self.config.image_size),
            "max_det": int(self.config.max_detections),
            "verbose": False,
        }
        if self.config.device:
            kwargs["device"] = self.config.device
        if self.config.end2end is not None:
            kwargs["end2end"] = bool(self.config.end2end)
        try:
            results = model.predict(**kwargs)
        except Exception as exc:
            raise DetectorConfigurationError(
                f"Ultralytics inference failed for '{self.model_path.name}': {exc}"
            ) from exc

        detections: list[Detection] = []
        height, width = bgr.shape[:2]
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            coordinates = _to_numpy(boxes.xyxy)
            confidences = _to_numpy(boxes.conf).reshape(-1)
            classes = _to_numpy(boxes.cls).astype(int).reshape(-1)
            names = getattr(result, "names", getattr(model, "names", {}))
            for coordinates_row, confidence, class_id in zip(coordinates, confidences, classes):
                bbox = BoundingBox(*(float(value) for value in coordinates_row[:4])).clamp(width, height)
                if bbox.width <= 0 or bbox.height <= 0:
                    continue
                detections.append(
                    Detection(
                        label=_class_name(names, int(class_id)),
                        confidence=float(np.clip(confidence, 0.0, 1.0)),
                        bbox=bbox,
                        class_id=int(class_id),
                        source="ultralytics",
                        metadata={"model": self.model_path.name},
                    )
                )
        return detections

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelDependencyError(
                "A .pt/.onnx component model was configured, but 'ultralytics' is not installed. "
                "Install the optional model dependencies; the pipeline will not silently use CV proposals."
            ) from exc
        try:
            self._model = YOLO(str(self.model_path), task="detect")
        except Exception as exc:
            raise DetectorConfigurationError(
                f"Could not load component model '{self.model_path}': {exc}"
            ) from exc
        return self._model


def create_detector(
    model_path: str | Path | None = None,
    *,
    mode: str = "auto",
    cv_config: CVDetectorConfig | None = None,
    model_config: ModelDetectorConfig | None = None,
) -> ComponentDetector:
    """Build a detector without concealing model configuration errors."""

    if mode not in {"auto", "cv"}:
        raise DetectorConfigurationError(f"Unsupported detector mode: {mode}")
    if mode == "cv":
        if model_path is not None:
            raise DetectorConfigurationError(
                "detector_mode='cv' cannot be combined with model_path; remove model_path to choose CV explicitly"
            )
        return CVComponentDetector(cv_config)
    if model_path is not None:
        return UltralyticsDetector(model_path, model_config)
    return CVComponentDetector(cv_config)


def non_maximum_suppression(
    detections: Sequence[Detection], iou_threshold: float = 0.35
) -> list[Detection]:
    threshold = float(np.clip(iou_threshold, 0.0, 1.0))
    remaining = sorted(detections, key=lambda detection: detection.confidence, reverse=True)
    kept: list[Detection] = []
    while remaining:
        current = remaining.pop(0)
        kept.append(current)
        remaining = [
            candidate
            for candidate in remaining
            if _intersection_over_union(current.bbox, candidate.bbox) <= threshold
        ]
    return kept


def _component_masks(gray: np.ndarray, configured_kernel: int) -> list[tuple[str, np.ndarray]]:
    kernel_size = max(3, int(configured_kernel))
    if kernel_size % 2 == 0:
        kernel_size += 1
    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    feature_size = max(7, int(round(min(gray.shape[:2]) * 0.035)))
    if feature_size % 2 == 0:
        feature_size += 1
    feature_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (feature_size, feature_size))

    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, feature_kernel)
    _, dark = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, feature_kernel)
    _, bright = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(gray, 45, 135)

    masks: list[tuple[str, np.ndarray]] = []
    for name, mask in (("dark", dark), ("bright", bright), ("edges", edges)):
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, small_kernel, iterations=2)
        masks.append((name, closed))
    return masks


def _local_contrast(gray: np.ndarray, x: int, y: int, width: int, height: int) -> float:
    padding = max(2, int(round(max(width, height) * 0.4)))
    outer_x1, outer_y1 = max(0, x - padding), max(0, y - padding)
    outer_x2 = min(gray.shape[1], x + width + padding)
    outer_y2 = min(gray.shape[0], y + height + padding)
    patch = gray[y : y + height, x : x + width]
    outer = gray[outer_y1:outer_y2, outer_x1:outer_x2]
    if patch.size == 0 or outer.size == 0:
        return 0.0
    mean_difference = abs(float(patch.mean()) - float(outer.mean())) / 96.0
    texture = float(patch.std()) / 96.0
    return float(np.clip(0.65 * mean_difference + 0.35 * texture, 0.0, 1.0))


def _intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    x1, y1 = max(first.x1, second.x1), max(first.y1, second.y1)
    x2, y2 = min(first.x2, second.x2), min(first.y2, second.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = first.area + second.area - intersection
    return intersection / union if union > 0 else 0.0


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, f"class_{class_id}"))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return f"class_{class_id}"
