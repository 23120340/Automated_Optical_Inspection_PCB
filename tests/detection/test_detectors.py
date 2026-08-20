"""Step 4: the CV proposal detector and the Ultralytics adapter."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from aoi_pipeline import (
    BoundingBox,
    CVComponentDetector,
    Detection,
    DetectorConfigurationError,
    MockComponentDetector,
    ModelDetectorConfig,
    UltralyticsDetector,
    create_detector,
)


def test_cv_detector_returns_only_valid_candidate_boxes(pcb_image: np.ndarray) -> None:
    detections = CVComponentDetector().detect(pcb_image[30:211, 35:306])
    assert detections
    assert all(detection.label == "component_candidate" for detection in detections)
    assert all(detection.source == "opencv_candidate" for detection in detections)
    assert all(detection.bbox.area > 0 for detection in detections)
    assert all(0 <= detection.confidence <= 1 for detection in detections)


def test_mock_detector_clamps_boxes(pcb_image: np.ndarray) -> None:
    detector = MockComponentDetector(
        [Detection("resistor", 0.9, BoundingBox(-5, -4, 20, 22))]
    )
    result = detector.detect(pcb_image)
    assert result[0].bbox.as_xyxy() == [0.0, 0.0, 20.0, 22.0]
    assert result[0].source == "mock"


class _FakeBoxes:
    xyxy = np.asarray([[5.0, 6.0, 22.0, 30.0], [30.0, 10.0, 55.0, 26.0]])
    conf = np.asarray([0.91, 0.73])
    cls = np.asarray([0, 2])

    def __len__(self) -> int:
        return 2


class _FakeModel:
    names = {0: "resistor", 2: "ic"}

    def __init__(self) -> None:
        self.last_predict_kwargs = None

    def predict(self, **kwargs: object) -> list[SimpleNamespace]:
        self.last_predict_kwargs = kwargs
        return [SimpleNamespace(boxes=_FakeBoxes(), names=self.names)]


@pytest.mark.parametrize("suffix", [".pt", ".onnx"])
def test_ultralytics_adapter_parses_pt_and_onnx_results(tmp_path: Path, suffix: str) -> None:
    model = _FakeModel()
    detector = UltralyticsDetector(tmp_path / f"model{suffix}", model=model)
    detections = detector.detect(np.zeros((80, 100, 3), dtype=np.uint8))
    assert [detection.label for detection in detections] == ["resistor", "ic"]
    assert detections[0].confidence == pytest.approx(0.91)
    assert "end2end" not in model.last_predict_kwargs


def test_ultralytics_adapter_can_pin_kaggle_one_to_many_head(tmp_path: Path) -> None:
    model = _FakeModel()
    detector = UltralyticsDetector(
        tmp_path / "model.pt",
        ModelDetectorConfig(end2end=False),
        model=model,
    )
    detector.detect(np.zeros((80, 100, 3), dtype=np.uint8))
    assert model.last_predict_kwargs["end2end"] is False


def test_ultralytics_adapter_leaves_augment_unset_by_default(tmp_path: Path) -> None:
    model = _FakeModel()
    detector = UltralyticsDetector(tmp_path / "model.pt", model=model)
    detector.detect(np.zeros((80, 100, 3), dtype=np.uint8))
    assert "augment" not in model.last_predict_kwargs


def test_ultralytics_adapter_can_enable_inference_time_tta(tmp_path: Path) -> None:
    model = _FakeModel()
    detector = UltralyticsDetector(
        tmp_path / "model.pt",
        ModelDetectorConfig(tta=True),
        model=model,
    )
    detector.detect(np.zeros((80, 100, 3), dtype=np.uint8))
    assert model.last_predict_kwargs["augment"] is True


def test_ultralytics_adapter_accepts_detail_confidence_override(tmp_path: Path) -> None:
    model = _FakeModel()
    detector = UltralyticsDetector(
        tmp_path / "model.onnx",
        ModelDetectorConfig(confidence=0.35),
        model=model,
    )

    detector.detect(np.zeros((80, 100, 3), dtype=np.uint8), confidence=0.20)

    assert model.last_predict_kwargs["conf"] == pytest.approx(0.20)


def test_bad_model_path_never_silently_falls_back(tmp_path: Path) -> None:
    with pytest.raises(DetectorConfigurationError, match="does not exist"):
        create_detector(tmp_path / "missing.pt")
    with pytest.raises(DetectorConfigurationError, match=".pt or .onnx"):
        UltralyticsDetector(tmp_path / "weights.bin", model=_FakeModel())
