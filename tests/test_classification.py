from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from aoi_pipeline import (
    BoundingBox,
    ClassificationConfig,
    ClassifierConfigurationError,
    ComponentCrop,
    ONNXComponentClassifier,
    create_classifier,
)


def _manifest() -> dict[str, object]:
    return {
        "schema_version": "pcb-component-classifier/1.0",
        "task": "component_family_classification",
        "model_format": "onnx",
        "class_names": ["capacitor", "resistor", "ic"],
        "input": {
            "name": "images",
            "size": [8, 8],
            "color_space": "RGB",
            "resize_mode": "letterbox",
            "letterbox_value": 114,
            "normalization": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        },
        "output": {"name": "logits"},
        "calibration": {"temperature": 1.0},
        "decision_thresholds": {"accept": 0.7, "review": 0.45},
        "model": {"version": "test-v1", "sha256": ""},
    }


def _crop(index: int) -> ComponentCrop:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :, 2] = 255  # BGR red; the classifier contract requires RGB.
    bbox = BoundingBox(0, 0, 8, 8)
    return ComponentCrop(
        image=image,
        detection_id=f"det_{index}",
        label="detector_hint",
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
                    [0.0, 3.0, 0.0],  # accept resistor
                    [0.7, 0.0, 0.0],  # review capacitor
                    [0.2, 0.0, 0.0],  # unknown / low confidence
                ],
                dtype=np.float32,
            )
        ]


def test_onnx_classifier_applies_manifest_preprocessing_and_decisions() -> None:
    session = _FakeSession()
    classifier = ONNXComponentClassifier(
        "unused.onnx",
        _manifest(),
        ClassificationConfig(batch_size=8, top_k=2),
        session=session,
    )
    results = classifier.classify([_crop(1), _crop(2), _crop(3)])

    assert [item.decision for item in results] == ["accept", "review", "unknown"]
    assert results[0].family == "resistor"
    assert results[0].top_k[0].label == "resistor"
    assert results[0].crop_id == "crop_1"
    assert results[0].detection_id == "det_1"
    assert results[0].detector_hint == "detector_hint"
    assert session.feed is not None
    assert session.feed.shape == (3, 3, 8, 8)
    assert session.feed[0, 0].mean() == pytest.approx(1.0)  # BGR -> RGB
    assert session.feed[0, 2].mean() == pytest.approx(0.0)


def test_classifier_rejects_ambiguous_artifact_configuration() -> None:
    with pytest.raises(ClassifierConfigurationError, match="both best.onnx"):
        create_classifier("best.onnx", None)


def test_classifier_rejects_incompatible_manifest() -> None:
    manifest = _manifest()
    manifest["schema_version"] = "some-other-schema"
    with pytest.raises(ClassifierConfigurationError, match="Unsupported"):
        ONNXComponentClassifier("unused.onnx", manifest, session=_FakeSession())


def test_classifier_never_invents_results_for_empty_crops() -> None:
    classifier = ONNXComponentClassifier(
        "unused.onnx", _manifest(), session=_FakeSession()
    )
    assert classifier.classify([]) == []


def test_classifier_verifies_onnx_sha256_before_loading(tmp_path) -> None:
    model_path = tmp_path / "best.onnx"
    model_path.write_bytes(b"not-a-real-model")
    manifest = _manifest()
    manifest["model"] = {"version": "test-v1", "sha256": "0" * 64}

    with pytest.raises(ClassifierConfigurationError, match="SHA-256"):
        ONNXComponentClassifier(model_path, manifest)
