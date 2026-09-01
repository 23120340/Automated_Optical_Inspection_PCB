"""Manifest-driven ONNX classification for step-5.2 package geometry.

This head classifies the visible package topology of an already cropped
component.  It deliberately has a separate manifest identity from the
step-6.1 component-family head: accepting one model in the other slot could
silently remove or invent solder ROIs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from ..config import ClassificationConfig
from ..exceptions import ClassifierConfigurationError
from ..models import ClassProbability, ComponentCrop
from .family import (
    MANIFEST_SCHEMA as _FAMILY_MANIFEST_SCHEMA,
    ONNXComponentClassifier,
    _load_manifest,
)


MANIFEST_SCHEMA = "pcb-package-classifier/1.0"
MANIFEST_TASK = "component_package_classification"
PACKAGE_INPUT_SIZE = (128, 128)
PACKAGE_CLASS_NAMES = (
    "hai_chan",
    "tru_dung",
    "goi_nho",
    "ic_hai_ben",
    "ic_bon_ben",
    "ic_khong_chan",
    "connector",
)


class PackageClassifier(Protocol):
    """Interface consumed by the step-5.2 package-classification stage."""

    def classify(
        self, crops: Sequence[ComponentCrop]
    ) -> list[PackageClassification]: ...


@dataclass(slots=True)
class PackageClassification:
    """One package-topology result tied to its crop and component detection."""

    crop_id: str
    detection_id: str
    package_class: str
    probability: float
    top_k: list[ClassProbability]
    unknown_score: float
    decision: Literal["accept", "review", "unknown"]
    model_version: str
    source: str = "onnx_package_classifier"
    detector_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.package_class not in PACKAGE_CLASS_NAMES:
            raise ValueError(f"Unknown package class: {self.package_class}")
        if not 0.0 <= float(self.probability) <= 1.0:
            raise ValueError("Package probability must be between 0 and 1")
        if not 0.0 <= float(self.unknown_score) <= 1.0:
            raise ValueError("Package unknown_score must be between 0 and 1")
        if self.decision not in {"accept", "review", "unknown"}:
            raise ValueError("Package decision must be accept, review, or unknown")

    def to_dict(self) -> dict[str, Any]:
        return {
            "crop_id": self.crop_id,
            "detection_id": self.detection_id,
            "package_class": self.package_class,
            "probability": float(self.probability),
            "top_k": [item.to_dict() for item in self.top_k],
            "unknown_score": float(self.unknown_score),
            "decision": self.decision,
            "model_version": self.model_version,
            "source": self.source,
            "detector_hint": self.detector_hint,
            "metadata": self.metadata,
        }


class ONNXPackageClassifier:
    """Run the seven-class package head under its own strict manifest contract.

    The ONNX execution, RGB letterboxing, normalization, batching, calibration
    and confidence policy are shared with :class:`ONNXComponentClassifier`.
    Only a copied manifest is adapted for that internal runner; the caller's
    package manifest and its task identity remain unchanged.
    """

    def __init__(
        self,
        model_path: str | Path,
        manifest_path: str | Path | Mapping[str, Any],
        config: ClassificationConfig | None = None,
        *,
        session: Any | None = None,
    ) -> None:
        manifest = _load_manifest(manifest_path)
        _validate_package_identity(manifest)

        adapted_manifest = dict(manifest)
        adapted_manifest["schema_version"] = _FAMILY_MANIFEST_SCHEMA
        adapted_manifest["task"] = "component_family_classification"
        delegate = ONNXComponentClassifier(
            model_path,
            adapted_manifest,
            config,
            session=session,
        )
        if delegate.input_size != PACKAGE_INPUT_SIZE:
            raise ClassifierConfigurationError(
                "Package classifier input.size must be 128 or [128, 128]"
            )

        self.manifest = manifest
        self.model_path = delegate.model_path
        self.config = delegate.config
        self.class_names = list(delegate.class_names)
        self.input_size = delegate.input_size
        self.model_version = delegate.model_version
        self._delegate = delegate

    def classify(
        self, crops: Sequence[ComponentCrop]
    ) -> list[PackageClassification]:
        """Classify crops without inventing output for an empty input batch."""

        family_results = self._delegate.classify(crops)
        return [
            PackageClassification(
                crop_id=item.crop_id,
                detection_id=item.detection_id,
                package_class=item.family,
                probability=item.probability,
                top_k=list(item.top_k),
                unknown_score=item.unknown_score,
                decision=item.decision,
                model_version=item.model_version,
                detector_hint=item.detector_hint,
                metadata=dict(item.metadata),
            )
            for item in family_results
        ]


def create_package_classifier(
    model_path: str | Path | None,
    manifest_path: str | Path | Mapping[str, Any] | None,
    config: ClassificationConfig | None = None,
) -> ONNXPackageClassifier | None:
    """Create step 5.2 only from a complete ONNX/manifest artifact pair."""

    if model_path is None and manifest_path is None:
        return None
    if model_path is None or manifest_path is None:
        raise ClassifierConfigurationError(
            "Step 5.2 package classification requires both best.onnx "
            "and model_manifest.json"
        )
    return ONNXPackageClassifier(model_path, manifest_path, config)


def _validate_package_identity(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ClassifierConfigurationError(
            "Unsupported package classifier manifest schema; "
            f"expected {MANIFEST_SCHEMA}"
        )
    if manifest.get("task") != MANIFEST_TASK:
        raise ClassifierConfigurationError(
            f"Package classifier manifest task must be {MANIFEST_TASK}"
        )
    if manifest.get("class_names") != list(PACKAGE_CLASS_NAMES):
        raise ClassifierConfigurationError(
            "Package classifier class_names must exactly match the ordered "
            f"seven-class contract: {list(PACKAGE_CLASS_NAMES)}"
        )
    input_spec = manifest.get("input")
    if isinstance(input_spec, Mapping):
        input_size = input_spec.get("size")
        is_128_square = input_size == 128 or input_size in (
            [128, 128],
            (128, 128),
        )
        if not is_128_square:
            raise ClassifierConfigurationError(
                "Package classifier input.size must be 128 or [128, 128]"
            )
