"""Layer B of step 6.2: the trained solder-defect classifier.

Deliberately the same shape of contract as step 6.1: a raw-logit ONNX file plus
a ``model_manifest.json`` that pins class order, preprocessing, calibration and
the decision policy. Anyone who has shipped a 6.1 model already knows how to
ship this one, and the runtime refuses anything whose contract it cannot read
rather than guessing at the class order -- a silently transposed taxonomy turns
every defect into a pass.

Nothing here is required to run step 6.2. With no artifacts configured the
inspector runs on rules alone and says so.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from ..config import SolderGradingConfig
from ..exceptions import ClassifierConfigurationError, ModelDependencyError
from ..imaging.image_io import ensure_bgr, letterbox_normalize
from ..models import ClassProbability, softmax

__all__ = [
    "MANIFEST_SCHEMA",
    "ONNXSolderClassifier",
    "create_solder_classifier",
]

MANIFEST_SCHEMA = "pcb-solder-defect-classifier/1.0"
MAX_MANIFEST_BYTES = 1024 * 1024


class ONNXSolderClassifier:
    """Run a trusted raw-logit ONNX solder-defect model from its manifest."""

    def __init__(
        self,
        model_path: str | Path,
        manifest_path: str | Path | Mapping[str, Any],
        config: SolderGradingConfig | None = None,
        *,
        session: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.config = config or SolderGradingConfig()
        self.manifest = _load_manifest(manifest_path)
        contract = _validate_manifest(self.manifest)

        self.class_names: list[str] = contract["class_names"]
        self.scope: str = contract["scope"]
        self.input_size: tuple[int, int] = contract["input_size"]
        self.mean = np.asarray(contract["mean"], dtype=np.float32)
        self.std = np.asarray(contract["std"], dtype=np.float32)
        self.letterbox_value: int = contract["letterbox_value"]
        self.model_version: str = contract["model_version"]
        self.good_label: str = contract["good_label"]

        self.temperature = _override(
            self.config.temperature, contract["temperature"], "temperature", minimum=0.0
        )
        self.accept_threshold = _override(
            self.config.accept_threshold, contract["accept_threshold"], "accept_threshold"
        )
        self.review_threshold = _override(
            self.config.review_threshold, contract["review_threshold"], "review_threshold"
        )
        if self.review_threshold > self.accept_threshold:
            raise ClassifierConfigurationError(
                "review_threshold must be less than or equal to accept_threshold"
            )
        self.accept_by_class: dict[str, float] = contract["accept_by_class"]

        self._session = session
        self._input_name: str | None = None
        self._output_name: str | None = None
        self._contract = contract

        if session is None:
            if self.model_path.suffix.lower() != ".onnx":
                raise ClassifierConfigurationError(
                    "Step 6.2 only accepts .onnx artifacts; .pt/pickle is not loaded."
                )
            if not self.model_path.is_file():
                raise ClassifierConfigurationError(
                    f"Solder model does not exist: {self.model_path}"
                )
            expected = contract["model_sha256"]
            if expected and _sha256_file(self.model_path).lower() != expected.lower():
                raise ClassifierConfigurationError(
                    "Solder model SHA-256 does not match model_manifest.json"
                )

    # ------------------------------------------------------------------ #

    def predict(
        self, images: Sequence[np.ndarray]
    ) -> list[list[ClassProbability]]:
        """Return the full ordered distribution for each ROI image.

        The whole distribution rather than a top-1 label: fusion needs the
        probability of the good class specifically, not only whichever class
        happened to win.
        """

        if not images:
            return []
        session = self._get_session()
        batch_size = max(1, int(self.config.batch_size))
        results: list[list[ClassProbability]] = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            base = np.stack([self._preprocess(image) for image in batch])
            probabilities = self._infer_probabilities(session, base)
            for row in probabilities:
                order = np.argsort(-row, kind="stable")
                results.append(
                    [
                        ClassProbability(self.class_names[int(index)], float(row[int(index)]))
                        for index in order
                    ]
                )
        return results

    def accept_threshold_for(self, label: str) -> float:
        return self.accept_by_class.get(label, self.accept_threshold)

    def _infer_probabilities(self, session: Any, base: np.ndarray) -> np.ndarray:
        """Run the model, averaging over 4 flip views when ``config.tta`` is set.

        Same identity/horizontal/vertical/both averaging as the step-6.1
        classifier. Solder joints have no canonical orientation, so this is
        expected to help by the same reasoning, but -- unlike step 6.1 -- it
        has not been measured against this specific trained model.
        """

        batch_count = base.shape[0]
        if self.config.tta:
            views = np.concatenate(
                [base, base[:, :, :, ::-1], base[:, :, ::-1, :], base[:, :, ::-1, ::-1]],
                axis=0,
            )
            tensor = np.ascontiguousarray(views)
        else:
            tensor = base
        output = session.run([self._output_name], {self._input_name: tensor})[0]
        logits = np.asarray(output, dtype=np.float32)
        view_count = 4 if self.config.tta else 1
        expected = (batch_count * view_count, len(self.class_names))
        if logits.ndim != 2 or logits.shape != expected:
            raise ClassifierConfigurationError(
                "Solder model output must be [batch, classes]; "
                f"received {tuple(logits.shape)}, expected {expected}"
            )
        probabilities = _softmax(logits / self.temperature)
        if self.config.tta:
            probabilities = probabilities.reshape(view_count, batch_count, -1).mean(axis=0)
        return probabilities

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(ensure_bgr(image), cv2.COLOR_BGR2RGB)
        resized = letterbox_normalize(
            rgb, self.input_size, (self.letterbox_value,) * 3
        )
        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - self.mean) / self.std
        return np.ascontiguousarray(tensor.transpose(2, 0, 1), dtype=np.float32)

    def _get_session(self) -> Any:
        if self._session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ModelDependencyError(
                    "A step-6.2 ONNX model was configured, but 'onnxruntime' is not "
                    "installed. Install requirements-model.txt."
                ) from exc
            available = set(ort.get_available_providers())
            if self.config.device == "cuda":
                if "CUDAExecutionProvider" not in available:
                    raise ModelDependencyError(
                        "CUDA was requested for step 6.2, but ONNX Runtime CUDA is unavailable."
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
                "Solder ONNX must expose exactly one input and at least one output"
            )
        self._input_name = self._contract["input_name"] or inputs[0].name
        self._output_name = self._contract["output_name"] or outputs[0].name
        return self._session


def create_solder_classifier(
    model_path: str | Path | None,
    manifest_path: str | Path | Mapping[str, Any] | None,
    config: SolderGradingConfig | None = None,
) -> ONNXSolderClassifier | None:
    """Build the classifier only when both artifacts are supplied.

    Half a contract is worse than none: a model without its manifest would have
    to have its class order guessed, and guessing wrong maps every defect onto
    a passing label.
    """

    if model_path is None and manifest_path is None:
        return None
    if model_path is None or manifest_path is None:
        raise ClassifierConfigurationError(
            "Step 6.2 requires both the .onnx model and its model_manifest.json"
        )
    return ONNXSolderClassifier(model_path, manifest_path, config)


# ---------------------------------------------------------------------- #
# Manifest
# ---------------------------------------------------------------------- #


def _load_manifest(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise ClassifierConfigurationError(f"Solder manifest does not exist: {path}")
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ClassifierConfigurationError("Solder manifest exceeds the 1 MB limit")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ClassifierConfigurationError(f"Invalid solder manifest JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ClassifierConfigurationError("Solder manifest root must be an object")
    return value


def _validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ClassifierConfigurationError(
            f"Unsupported solder manifest schema; expected {MANIFEST_SCHEMA}"
        )
    if manifest.get("task") != "solder_defect_classification":
        raise ClassifierConfigurationError(
            "Solder manifest task must be solder_defect_classification"
        )
    if str(manifest.get("model_format", "")).lower() != "onnx":
        raise ClassifierConfigurationError("Solder manifest model_format must be onnx")

    scope = str(manifest.get("scope", "joint")).lower()
    if scope not in {"joint", "component"}:
        raise ClassifierConfigurationError("Solder manifest scope must be joint or component")

    raw_classes = manifest.get("class_names")
    if not isinstance(raw_classes, list) or len(raw_classes) < 2:
        raise ClassifierConfigurationError("Solder manifest needs at least two class_names")
    class_names = [str(value).strip() for value in raw_classes]
    if any(not value for value in class_names) or len(set(class_names)) != len(class_names):
        raise ClassifierConfigurationError("Solder class_names must be unique and non-empty")

    # Fusion has to know which label means "no defect"; without it the escape
    # guard cannot tell a pass from a fail.
    good_label = str(manifest.get("good_label", "good")).strip()
    if good_label not in class_names:
        raise ClassifierConfigurationError(
            f"good_label '{good_label}' is not in class_names"
        )

    input_spec = _mapping(manifest.get("input"), "input")
    if str(input_spec.get("color_space", "")).upper() != "RGB":
        raise ClassifierConfigurationError("Solder input color_space must be RGB")
    if str(input_spec.get("resize_mode", "")).lower() != "letterbox":
        raise ClassifierConfigurationError("Solder input resize_mode must be letterbox")
    input_size = _input_size(input_spec.get("size"))
    normalization = _mapping(input_spec.get("normalization"), "input.normalization")
    mean = _three_floats(normalization.get("mean"), "normalization.mean")
    std = _three_floats(normalization.get("std"), "normalization.std")
    if any(value <= 0 for value in std):
        raise ClassifierConfigurationError("normalization.std values must be positive")
    letterbox_value = int(input_spec.get("letterbox_value", 114))
    if not 0 <= letterbox_value <= 255:
        raise ClassifierConfigurationError("input.letterbox_value must be in [0, 255]")

    thresholds = _mapping(manifest.get("decision_thresholds"), "decision_thresholds")
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

    calibration = _mapping(manifest.get("calibration", {}), "calibration")
    temperature = float(calibration.get("temperature", 1.0))
    if not np.isfinite(temperature) or temperature <= 0:
        raise ClassifierConfigurationError("calibration.temperature must be positive")

    model = _mapping(manifest.get("model", {}), "model")
    output_spec = _mapping(manifest.get("output", {}), "output")
    return {
        "class_names": class_names,
        "good_label": good_label,
        "scope": scope,
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
        raise ClassifierConfigurationError(f"Solder manifest '{name}' must be an object")
    return value


def _input_size(value: Any) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        width, height = (int(item) for item in value)
    elif isinstance(value, Mapping):
        width, height = int(value.get("width", 0)), int(value.get("height", 0))
    elif isinstance(value, (int, float)):
        width = height = int(value)
    else:
        raise ClassifierConfigurationError("input.size must be [width, height] or a number")
    if width <= 0 or height <= 0:
        raise ClassifierConfigurationError("input.size values must be positive")
    return (width, height)


def _three_floats(value: Any, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ClassifierConfigurationError(f"{name} must hold three numbers")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ClassifierConfigurationError(f"{name} must hold three numbers") from exc


def _probability(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ClassifierConfigurationError(f"{name} must be a number in [0, 1]") from exc
    if not 0.0 <= number <= 1.0:
        raise ClassifierConfigurationError(f"{name} must be in [0, 1]")
    return number


def _override(
    deployment: float | None, contract: float, name: str, minimum: float = 0.0
) -> float:
    if deployment is None:
        return contract
    value = float(deployment)
    if not np.isfinite(value) or value < minimum:
        raise ClassifierConfigurationError(f"{name} override must be >= {minimum}")
    return value


_softmax = softmax


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
