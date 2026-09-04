"""Step 4: interchangeable component detector adapters.

The CV detector emits proposals labelled ``component_candidate``. It is useful
for testing steps 0-5 before training, but it is not a replacement for the
Kaggle-trained model. ``UltralyticsDetector`` supports both ``.pt`` and
``.onnx`` artifacts and never silently falls back when a configured model fails.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import CVDetectorConfig, ModelDetectorConfig
from ..exceptions import DetectorConfigurationError, ModelDependencyError
from ..imaging.image_io import ensure_bgr
from ..models import BoundingBox, Detection, intersection_over_union


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
    """Adapter around an Ultralytics detection/segmentation model.

    ``task`` and ``source`` default to their historical component-detection
    values.  Supplying them explicitly lets a separately contracted model use
    the same battle-tested Ultralytics/ONNX adapter without pretending that an
    instance-segmentation head is the step-4 component detector.
    """

    SUPPORTED_SUFFIXES = {".pt", ".onnx"}

    def __init__(
        self,
        model_path: str | Path,
        config: ModelDetectorConfig | None = None,
        *,
        model: Any | None = None,
        task: str = "detect",
        source: str = "ultralytics",
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.config = config or ModelDetectorConfig()
        self.task = str(task).strip().lower()
        self.source = str(source).strip()
        if not self.task:
            raise DetectorConfigurationError("Ultralytics task cannot be empty")
        if not self.source:
            raise DetectorConfigurationError("Detection source cannot be empty")
        if self.model_path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
            raise DetectorConfigurationError(
                "Component model must be an Ultralytics-compatible .pt or .onnx file"
            )
        if model is None and not self.model_path.is_file():
            raise DetectorConfigurationError(f"Component model does not exist: {self.model_path}")
        self._model = model
        # Filled lazily on first inference. ``None`` means "not looked yet";
        # ``0`` means "looked, this artifact has no fixed size". A bool sentinel
        # would be wrong here -- ``isinstance(False, int)`` is True in Python,
        # so False would be returned as the image size and reach cv2.resize.
        self._fixed_image_size: int | None = None

    def detect(
        self, image: np.ndarray, *, confidence: float | None = None
    ) -> list[Detection]:
        bgr = ensure_bgr(image)
        model = self._get_model()
        # An ONNX exported with dynamic=False accepts exactly one input size.
        # The configured imgsz is what the *pipeline* wants; the graph decides
        # what it will actually take, and disagreeing is a hard ONNX Runtime
        # error rather than a resize. Shipped exports differ (640, 1280, 1536),
        # so the size has to come from the artifact, not from a default.
        image_size = self._resolve_image_size()
        kwargs: dict[str, Any] = {
            "source": bgr,
            "conf": float(self.config.confidence if confidence is None else confidence),
            "iou": float(self.config.iou),
            "imgsz": image_size,
            "max_det": int(self.config.max_detections),
            "verbose": False,
        }
        if self.config.device:
            kwargs["device"] = self.config.device
        if self.config.end2end is not None:
            kwargs["end2end"] = bool(self.config.end2end)
        if self.config.tta:
            kwargs["augment"] = True
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
            mask_polygons = _result_mask_polygons(result, width, height)
            for index, (coordinates_row, confidence, class_id) in enumerate(
                zip(coordinates, confidences, classes)
            ):
                bbox = BoundingBox(*(float(value) for value in coordinates_row[:4])).clamp(width, height)
                if bbox.width <= 0 or bbox.height <= 0:
                    continue
                metadata: dict[str, Any] = {
                    "model": self.model_path.name,
                    "task": self.task,
                }
                if index < len(mask_polygons) and mask_polygons[index]:
                    metadata["mask_polygon"] = mask_polygons[index]
                detections.append(
                    Detection(
                        label=_class_name(names, int(class_id)),
                        confidence=float(np.clip(confidence, 0.0, 1.0)),
                        bbox=bbox,
                        class_id=int(class_id),
                        source=self.source,
                        metadata=metadata,
                    )
                )
        return detections

    def _resolve_image_size(self) -> int:
        """The size this artifact will actually accept.

        Returns the configured size for ``.pt`` weights and for ONNX graphs with
        a dynamic spatial axis, since those genuinely resize. A fixed-shape ONNX
        returns its own size instead: feeding it anything else raises
        ``INVALID_ARGUMENT ... Got: 1280 Expected: 1536`` deep inside ONNX
        Runtime, which surfaces as an opaque inference failure.
        """

        if self._fixed_image_size is None:
            self._fixed_image_size = _onnx_fixed_image_size(self.model_path) or 0
        if self._fixed_image_size > 0:
            return self._fixed_image_size
        return int(self.config.image_size)

    @property
    def image_size(self) -> int:
        """The size inference will really run at, after the artifact is consulted."""

        return self._resolve_image_size()

    @property
    def native_image_size(self) -> int | None:
        """Cỡ mà đồ thị BẮT BUỘC nhận, hoặc ``None`` khi nó nhận cỡ nào cũng được.

        Khác ``image_size`` ở đúng một chỗ, và chỗ đó quan trọng: khi artifact
        không khoá shape, ``image_size`` trả về giá trị *cấu hình* — một mặc
        định chung, không phải một ràng buộc. Ai lấy nó để ghi đè một tham số
        người dùng đặt tường minh sẽ ghi đè bằng mặc định, và đó là lỗi.
        """

        if self._fixed_image_size is None:
            self._fixed_image_size = _onnx_fixed_image_size(self.model_path) or 0
        return self._fixed_image_size or None

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
            self._model = YOLO(str(self.model_path), task=self.task)
        except Exception as exc:
            raise DetectorConfigurationError(
                f"Could not load component model '{self.model_path}': {exc}"
            ) from exc
        return self._model


def detector_identifier(detector: ComponentDetector) -> str:
    """Return a portable identifier for the exact runtime detector artifact.

    Production Ultralytics identifiers bind the artifact filename and SHA-256;
    absolute workstation paths are never exported. Non-artifact detectors are
    explicitly marked as demo/test implementations.
    """

    if isinstance(detector, UltralyticsDetector):
        try:
            digest = _file_sha256(detector.model_path)
        except OSError:
            digest = "unavailable"
        return f"{detector.model_path.name}:{digest}"
    if isinstance(detector, CVComponentDetector):
        return "CVComponentDetector:demo"
    if isinstance(detector, MockComponentDetector):
        return "MockComponentDetector:test"
    return f"{type(detector).__name__}:custom"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


_intersection_over_union = intersection_over_union


def _onnx_fixed_image_size(path: Path) -> int | None:
    """The square input size an ONNX graph is locked to, if it is locked at all.

    ``None`` for non-ONNX files, for graphs whose spatial axes are dynamic
    (exported with ``dynamic=True``, where the caller's size is honoured), for
    non-square inputs, and whenever the file cannot be read -- in every one of
    those cases the configured size remains the right answer, so a failure here
    must not become a failure to detect.
    """

    if path.suffix.lower() != ".onnx" or not path.is_file():
        return None
    try:
        import onnx
    except ImportError:
        return None
    try:
        graph = onnx.load(str(path)).graph
        dims = graph.input[0].type.tensor_type.shape.dim
    except Exception:
        return None
    if len(dims) != 4:
        return None
    # dim_value is 0 when the axis is dynamic (dim_param carries its name).
    height, width = dims[2].dim_value, dims[3].dim_value
    if height <= 0 or width <= 0 or height != width:
        return None
    return int(height)


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


def _result_mask_polygons(
    result: Any, width: int, height: int
) -> list[list[list[float]]]:
    """Return Ultralytics mask contours in original-image pixel coordinates.

    ``Masks.xy`` is already scaled back from the model's letterboxed input.
    Clipping here gives downstream overlays the same image-bound guarantee as
    :class:`BoundingBox`; an absent mask head simply produces an empty list, so
    all existing detection models keep exactly their previous behaviour.
    """

    masks = getattr(result, "masks", None)
    raw_polygons = getattr(masks, "xy", None)
    if raw_polygons is None:
        return []
    polygons: list[list[list[float]]] = []
    try:
        values = list(raw_polygons)
    except TypeError:
        return []
    for value in values:
        polygon = _to_numpy(value)
        if polygon.size == 0:
            polygons.append([])
            continue
        try:
            points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        except (TypeError, ValueError):
            polygons.append([])
            continue
        points[:, 0] = np.clip(points[:, 0], 0.0, float(width))
        points[:, 1] = np.clip(points[:, 1], 0.0, float(height))
        polygons.append([[float(x), float(y)] for x, y in points])
    return polygons
