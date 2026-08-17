"""Step 6.1: manifest-driven ONNX component-family classification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .config import ClassificationConfig
from .exceptions import ClassifierConfigurationError, ModelDependencyError
from .image_io import ensure_bgr
from .models import ClassProbability, ComponentClassification, ComponentCrop


MANIFEST_SCHEMA = "pcb-component-classifier/1.0"
MAX_MANIFEST_BYTES = 1_048_576


class ComponentClassifier(Protocol):
    """Interface accepted by :class:`AOIPipeline` for testable inference."""

    def classify(
        self, crops: Sequence[ComponentCrop]
    ) -> list[ComponentClassification]: ...


class ONNXComponentClassifier:
    """Run a trusted raw-logit ONNX classifier using its model manifest."""

    def __init__(
        self,
        model_path: str | Path,
        manifest_path: str | Path | Mapping[str, Any],
        config: ClassificationConfig | None = None,
        *,
        session: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.config = config or ClassificationConfig()
        self.manifest = _load_manifest(manifest_path)
        self.contract = _validate_manifest(self.manifest)
        self.class_names: list[str] = self.contract["class_names"]
        self.input_size: tuple[int, int] = self.contract["input_size"]
        self.mean: np.ndarray = np.asarray(self.contract["mean"], dtype=np.float32)
        self.std: np.ndarray = np.asarray(self.contract["std"], dtype=np.float32)
        self.letterbox_value: int = self.contract["letterbox_value"]
        self.model_version: str = self.contract["model_version"]
        self.temperature = _deployment_value(
            self.config.temperature,
            self.contract["temperature"],
            "temperature",
            minimum=0.0,
        )
        self.accept_threshold = _deployment_value(
            self.config.accept_threshold,
            self.contract["accept_threshold"],
            "accept_threshold",
        )
        self.review_threshold = _deployment_value(
            self.config.review_threshold,
            self.contract["review_threshold"],
            "review_threshold",
        )
        if self.review_threshold > self.accept_threshold:
            raise ClassifierConfigurationError(
                "review_threshold must be less than or equal to accept_threshold"
            )
        self.accept_by_class: dict[str, float] = self.contract["accept_by_class"]
        self._session = session
        self._input_name: str | None = None
        self._output_name: str | None = None

        if session is None:
            if self.model_path.suffix.lower() != ".onnx":
                raise ClassifierConfigurationError(
                    "Step 6.1 only accepts .onnx artifacts; .pt/pickle is not loaded."
                )
            if not self.model_path.is_file():
                raise ClassifierConfigurationError(
                    f"Classifier model does not exist: {self.model_path}"
                )
            expected_sha = self.contract["model_sha256"]
            if expected_sha:
                actual_sha = _sha256_file(self.model_path)
                if actual_sha.lower() != expected_sha.lower():
                    raise ClassifierConfigurationError(
                        "Classifier SHA-256 does not match model_manifest.json"
                    )

    def classify(
        self, crops: Sequence[ComponentCrop]
    ) -> list[ComponentClassification]:
        if not crops:
            return []
        session = self._get_session()
        batch_size = max(1, int(self.config.batch_size))
        top_k_count = min(max(1, int(self.config.top_k)), len(self.class_names))
        results: list[ComponentClassification] = []
        for start in range(0, len(crops), batch_size):
            batch_crops = crops[start : start + batch_size]
            tensor = np.stack([self._preprocess(item.image) for item in batch_crops])
            output = session.run([self._output_name], {self._input_name: tensor})[0]
            logits = np.asarray(output, dtype=np.float32)
            expected_shape = (len(batch_crops), len(self.class_names))
            if logits.ndim != 2 or logits.shape != expected_shape:
                raise ClassifierConfigurationError(
                    "Classifier output must be [batch, classes]; "
                    f"received {tuple(logits.shape)}, expected {expected_shape}"
                )
            probabilities = _softmax(logits / self.temperature)
            for offset, (crop, row) in enumerate(zip(batch_crops, probabilities)):
                order = np.argsort(-row, kind="stable")[:top_k_count]
                top_k = [
                    ClassProbability(self.class_names[int(index)], float(row[int(index)]))
                    for index in order
                ]
                prediction = top_k[0]
                accept_threshold = self.accept_by_class.get(
                    prediction.label, self.accept_threshold
                )
                if prediction.probability >= accept_threshold:
                    decision = "accept"
                elif prediction.probability >= self.review_threshold:
                    decision = "review"
                else:
                    decision = "unknown"
                crop_id = str(
                    crop.metadata.get("crop_id") or f"crop_{start + offset + 1:04d}"
                )
                results.append(
                    ComponentClassification(
                        crop_id=crop_id,
                        detection_id=crop.detection_id,
                        family=prediction.label,
                        probability=prediction.probability,
                        top_k=top_k,
                        unknown_score=float(1.0 - prediction.probability),
                        decision=decision,
                        model_version=self.model_version,
                        detector_hint=crop.label,
                        metadata={
                            "accept_threshold": float(accept_threshold),
                            "review_threshold": float(self.review_threshold),
                            "temperature": float(self.temperature),
                        },
                    )
                )
        return results

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        bgr = ensure_bgr(image)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = _letterbox(rgb, self.input_size, self.letterbox_value)
        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - self.mean) / self.std
        return np.ascontiguousarray(tensor.transpose(2, 0, 1), dtype=np.float32)

    def _get_session(self) -> Any:
        if self._session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ModelDependencyError(
                    "A step-6.1 ONNX classifier was configured, but 'onnxruntime' "
                    "is not installed. Install requirements-model.txt."
                ) from exc
            available = set(ort.get_available_providers())
            if self.config.device == "cuda":
                if "CUDAExecutionProvider" not in available:
                    raise ModelDependencyError(
                        "CUDA was requested for step 6.1, but ONNX Runtime CUDA is unavailable."
                    )
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif self.config.device == "auto" and "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(str(self.model_path), providers=providers)
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or not outputs:
            raise ClassifierConfigurationError(
                "Classifier ONNX must expose exactly one input and at least one output"
            )
        self._input_name = self.contract["input_name"] or inputs[0].name
        self._output_name = self.contract["output_name"] or outputs[0].name
        return self._session


def create_classifier(
    model_path: str | Path | None,
    manifest_path: str | Path | Mapping[str, Any] | None,
    config: ClassificationConfig | None = None,
) -> ONNXComponentClassifier | None:
    """Create the classifier only when both model and contract are supplied."""

    if model_path is None and manifest_path is None:
        return None
    if model_path is None or manifest_path is None:
        raise ClassifierConfigurationError(
            "Step 6.1 requires both best.onnx and model_manifest.json"
        )
    return ONNXComponentClassifier(model_path, manifest_path, config)


def _load_manifest(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise ClassifierConfigurationError(f"Classifier manifest does not exist: {path}")
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ClassifierConfigurationError("Classifier manifest exceeds the 1 MB limit")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ClassifierConfigurationError(f"Invalid classifier manifest JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ClassifierConfigurationError("Classifier manifest root must be an object")
    return value


def _validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ClassifierConfigurationError(
            f"Unsupported classifier manifest schema; expected {MANIFEST_SCHEMA}"
        )
    if manifest.get("task") != "component_family_classification":
        raise ClassifierConfigurationError(
            "Classifier manifest task must be component_family_classification"
        )
    if str(manifest.get("model_format", "")).lower() != "onnx":
        raise ClassifierConfigurationError("Classifier manifest model_format must be onnx")
    raw_classes = manifest.get("class_names")
    if not isinstance(raw_classes, list) or len(raw_classes) < 2:
        raise ClassifierConfigurationError("Classifier manifest needs at least two class_names")
    class_names = [str(value).strip() for value in raw_classes]
    if any(not value for value in class_names) or len(set(class_names)) != len(class_names):
        raise ClassifierConfigurationError("Classifier class_names must be unique and non-empty")

    input_spec = _mapping(manifest.get("input"), "input")
    if str(input_spec.get("color_space", "")).upper() != "RGB":
        raise ClassifierConfigurationError("Classifier input color_space must be RGB")
    if str(input_spec.get("resize_mode", "")).lower() != "letterbox":
        raise ClassifierConfigurationError("Classifier input resize_mode must be letterbox")
    input_size = _input_size(input_spec.get("size"))
    normalization = _mapping(input_spec.get("normalization"), "input.normalization")
    mean = _three_floats(normalization.get("mean"), "normalization.mean")
    std = _three_floats(normalization.get("std"), "normalization.std")
    if any(value <= 0 for value in std):
        raise ClassifierConfigurationError("normalization.std values must be positive")

    calibration = _mapping(manifest.get("calibration", {}), "calibration")
    thresholds = _mapping(manifest.get("decision_thresholds"), "decision_thresholds")
    model = _mapping(manifest.get("model"), "model")
    accept = _probability(thresholds.get("accept"), "decision_thresholds.accept")
    review = _probability(thresholds.get("review"), "decision_thresholds.review")
    if review > accept:
        raise ClassifierConfigurationError(
            "decision_thresholds.review must be <= decision_thresholds.accept"
        )
    raw_per_class = thresholds.get("accept_by_class", {})
    if not isinstance(raw_per_class, Mapping):
        raise ClassifierConfigurationError("accept_by_class must be an object")
    accept_by_class: dict[str, float] = {}
    for label, value in raw_per_class.items():
        if label not in class_names:
            raise ClassifierConfigurationError(f"Unknown accept_by_class label: {label}")
        accept_by_class[str(label)] = _probability(value, f"accept_by_class.{label}")

    temperature = float(calibration.get("temperature", 1.0))
    if not np.isfinite(temperature) or temperature <= 0:
        raise ClassifierConfigurationError("calibration.temperature must be positive")
    letterbox_value = int(input_spec.get("letterbox_value", 114))
    if not 0 <= letterbox_value <= 255:
        raise ClassifierConfigurationError("input.letterbox_value must be in [0, 255]")
    output_spec = _mapping(manifest.get("output", {}), "output")
    return {
        "class_names": class_names,
        "input_size": input_size,
        "mean": mean,
        "std": std,
        "letterbox_value": letterbox_value,
        "input_name": str(input_spec.get("name") or ""),
        "output_name": str(output_spec.get("name") or ""),
        "temperature": temperature,
        "accept_threshold": accept,
        "review_threshold": review,
        "accept_by_class": accept_by_class,
        "model_version": str(model.get("version") or manifest.get("created_at") or "unknown"),
        "model_sha256": str(model.get("sha256") or ""),
    }


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ClassifierConfigurationError(f"Classifier manifest {name} must be an object")
    return value


def _input_size(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        result = (value, value)
    elif (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
    ):
        result = (int(value[0]), int(value[1]))
    else:
        raise ClassifierConfigurationError("input.size must be an integer or [height, width]")
    if min(result) <= 0:
        raise ClassifierConfigurationError("input.size values must be positive")
    return result


def _three_floats(value: Any, name: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        raise ClassifierConfigurationError(f"{name} must contain three values")
    result = [float(item) for item in value]
    if not all(np.isfinite(item) for item in result):
        raise ClassifierConfigurationError(f"{name} values must be finite")
    return result


def _probability(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ClassifierConfigurationError(f"{name} must be in [0, 1]")
    return number


def _deployment_value(
    override: float | None,
    fallback: float,
    name: str,
    *,
    minimum: float | None = None,
) -> float:
    value = fallback if override is None else float(override)
    if not np.isfinite(value):
        raise ClassifierConfigurationError(f"{name} must be finite")
    if minimum is not None:
        if value <= minimum:
            raise ClassifierConfigurationError(f"{name} must be greater than {minimum}")
    elif not 0.0 <= value <= 1.0:
        raise ClassifierConfigurationError(f"{name} must be in [0, 1]")
    return value


def _letterbox(image: np.ndarray, size: tuple[int, int], value: int) -> np.ndarray:
    target_height, target_width = size
    height, width = image.shape[:2]
    scale = min(target_width / max(1, width), target_height / max(1, height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.full((target_height, target_width, 3), value, dtype=np.uint8)
    x = (target_width - resized_width) // 2
    y = (target_height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
