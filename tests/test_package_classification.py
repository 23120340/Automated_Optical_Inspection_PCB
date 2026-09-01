from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from aoi_pipeline.classification.package import (
    MANIFEST_SCHEMA,
    MANIFEST_TASK,
    PACKAGE_CLASS_NAMES,
    ONNXPackageClassifier,
    PackageClassification,
    create_package_classifier,
)
from aoi_pipeline.config import ClassificationConfig
from aoi_pipeline.exceptions import ClassifierConfigurationError
from aoi_pipeline.models import BoundingBox, ComponentCrop


def _manifest() -> dict[str, object]:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "task": MANIFEST_TASK,
        "model_format": "onnx",
        "class_names": list(PACKAGE_CLASS_NAMES),
        "input": {
            "name": "images",
            "size": [128, 128],
            "color_space": "RGB",
            "resize_mode": "letterbox",
            "letterbox_value": 114,
            "normalization": {
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
            },
        },
        "output": {"name": "logits"},
        "calibration": {"temperature": 1.0},
        "decision_thresholds": {"accept": 0.70, "review": 0.45},
        "model": {"version": "package-test-v1", "sha256": ""},
    }


def _crop(index: int) -> ComponentCrop:
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    image[:, :, 2] = 255  # BGR red; the manifest requires RGB input.
    bbox = BoundingBox(0, 0, 40, 20)
    return ComponentCrop(
        image=image,
        detection_id=f"det_{index}",
        label="ic",
        confidence=0.9,
        source_bbox=bbox,
        crop_bbox=bbox,
        filename=f"crop_{index}.png",
        metadata={"crop_id": f"crop_{index}"},
    )


class _FakeSession:
    def __init__(self) -> None:
        self.feed: np.ndarray | None = None

    def get_inputs(self):
        return [SimpleNamespace(name="images")]

    def get_outputs(self):
        return [SimpleNamespace(name="logits")]

    def run(self, output_names, feed):
        assert output_names == ["logits"]
        self.feed = feed["images"]
        return [
            np.asarray(
                [
                    [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
        ]


def test_package_classifier_applies_rgb_letterbox_top_k_and_decisions() -> None:
    session = _FakeSession()
    classifier = ONNXPackageClassifier(
        "unused.onnx",
        _manifest(),
        ClassificationConfig(batch_size=8, top_k=2),
        session=session,
    )

    results = classifier.classify([_crop(1), _crop(2), _crop(3)])

    assert all(isinstance(item, PackageClassification) for item in results)
    assert [item.decision for item in results] == ["accept", "review", "unknown"]
    assert results[0].package_class == "ic_hai_ben"
    assert [item.label for item in results[0].top_k] == ["ic_hai_ben", "hai_chan"]
    assert results[0].crop_id == "crop_1"
    assert results[0].detection_id == "det_1"
    assert results[0].detector_hint == "ic"
    assert results[0].source == "onnx_package_classifier"
    assert classifier.manifest["schema_version"] == MANIFEST_SCHEMA

    assert session.feed is not None
    assert session.feed.shape == (3, 3, 128, 128)
    assert session.feed[0, 0, 64, 64] == pytest.approx(1.0)  # BGR -> RGB
    assert session.feed[0, 2, 64, 64] == pytest.approx(0.0)
    assert session.feed[0, :, 0, 0] == pytest.approx(np.full(3, 114 / 255.0))

    exported = results[0].to_dict()
    assert exported["package_class"] == "ic_hai_ben"
    assert exported["decision"] == "accept"
    assert exported["model_version"] == "package-test-v1"
    assert len(exported["top_k"]) == 2


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("schema_version", "pcb-package-classifier/2.0", "schema"),
        ("task", "component_family_classification", "task"),
        ("class_names", list(reversed(PACKAGE_CLASS_NAMES)), "class_names"),
    ],
)
def test_package_classifier_rejects_wrong_manifest_identity(
    key: str, value: object, message: str
) -> None:
    manifest = _manifest()
    manifest[key] = value

    with pytest.raises(ClassifierConfigurationError, match=message):
        ONNXPackageClassifier("unused.onnx", manifest, session=_FakeSession())


def test_package_classifier_requires_manifest_driven_128_input() -> None:
    manifest = _manifest()
    manifest["input"] = {  # type: ignore[dict-item]
        **manifest["input"],  # type: ignore[arg-type]
        "size": [224, 224],
    }

    with pytest.raises(ClassifierConfigurationError, match="128"):
        ONNXPackageClassifier("unused.onnx", manifest, session=_FakeSession())


def test_package_classifier_reuses_rgb_letterbox_manifest_validation() -> None:
    manifest = _manifest()
    manifest["input"] = {  # type: ignore[dict-item]
        **manifest["input"],  # type: ignore[arg-type]
        "color_space": "BGR",
    }

    with pytest.raises(ClassifierConfigurationError, match="RGB"):
        ONNXPackageClassifier("unused.onnx", manifest, session=_FakeSession())


def test_package_classifier_is_noop_without_an_artifact_pair() -> None:
    assert create_package_classifier(None, None) is None

    with pytest.raises(ClassifierConfigurationError, match="both best.onnx"):
        create_package_classifier("best.onnx", None)
    with pytest.raises(ClassifierConfigurationError, match="both best.onnx"):
        create_package_classifier(None, _manifest())


def test_package_classifier_verifies_onnx_sha256_before_loading(tmp_path) -> None:
    model_path = tmp_path / "best.onnx"
    model_path.write_bytes(b"not-a-real-model")
    manifest = _manifest()
    manifest["model"] = {"version": "package-test-v1", "sha256": "0" * 64}

    with pytest.raises(ClassifierConfigurationError, match="SHA-256"):
        ONNXPackageClassifier(model_path, manifest)

    manifest["model"] = {
        "version": "package-test-v1",
        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    classifier = ONNXPackageClassifier(model_path, manifest)
    assert classifier.model_path == model_path.resolve()


def test_package_classifier_does_not_open_a_session_for_empty_crops() -> None:
    classifier = ONNXPackageClassifier(
        "unused.onnx", _manifest(), session=_FakeSession()
    )
    assert classifier.classify([]) == []
