from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from aoi_pipeline import (
    BoardConfig,
    BoundingBox,
    ComponentCropper,
    CropConfig,
    CVComponentDetector,
    Detection,
    DetectorConfigurationError,
    MockComponentDetector,
    ModelDetectorConfig,
    PCBLocalizer,
    UltralyticsDetector,
    create_detector,
)


def test_board_localizer_finds_synthetic_pcb(pcb_image: np.ndarray) -> None:
    region = PCBLocalizer().locate(pcb_image)
    assert region.method.startswith("contour:")
    assert region.confidence > 0.45
    assert region.bbox.x1 <= 45
    assert region.bbox.y1 <= 40
    assert region.bbox.x2 >= 295
    assert region.bbox.y2 >= 200
    assert region.mask is not None
    assert region.mask[120, 170] == 255
    assert region.mask[0, 0] == 0


def test_board_localizer_has_explicit_full_image_fallback() -> None:
    blank = np.full((50, 70, 3), 127, dtype=np.uint8)
    region = PCBLocalizer(BoardConfig()).locate(blank)
    assert region.method == "full_image_fallback"
    assert region.bbox.as_xyxy() == [0.0, 0.0, 70.0, 50.0]


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


def _onnx_with_input_shape(path: Path, height: object, width: object) -> Path:
    """A minimal ONNX file whose only purpose is to declare an input shape."""

    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    shape = [1, 3, height, width]
    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["out"])],
        "g",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, shape)],
    )
    onnx.save(helper.make_model(graph), str(path))
    return path


def test_a_fixed_shape_onnx_overrides_the_configured_image_size(tmp_path: Path) -> None:
    """An ONNX exported with ``dynamic=False`` accepts exactly one size.

    Feeding it anything else fails inside ONNX Runtime with
    ``INVALID_ARGUMENT ... Got: 1280 Expected: 1536``, which reaches the user as
    an opaque "inference failed". The shipped detectors are exported at 640,
    1280 and 1536, so no single configured default can be right for all of them
    -- the artifact has to be asked.
    """

    path = _onnx_with_input_shape(tmp_path / "fixed.onnx", 1536, 1536)
    model = _FakeModel()
    detector = UltralyticsDetector(path, ModelDetectorConfig(image_size=1280), model=model)

    assert detector.image_size == 1536
    detector.detect(np.zeros((80, 100, 3), dtype=np.uint8))
    assert model.last_predict_kwargs["imgsz"] == 1536


def test_a_dynamic_onnx_keeps_the_configured_image_size(tmp_path: Path) -> None:
    """Exported with ``dynamic=True``, the graph really does resize, so the
    pipeline's choice is the one that should win."""

    path = _onnx_with_input_shape(tmp_path / "dynamic.onnx", "h", "w")
    model = _FakeModel()
    detector = UltralyticsDetector(path, ModelDetectorConfig(image_size=1280), model=model)

    assert detector.image_size == 1280
    detector.detect(np.zeros((80, 100, 3), dtype=np.uint8))
    assert model.last_predict_kwargs["imgsz"] == 1280


def test_pt_weights_keep_the_configured_image_size(tmp_path: Path) -> None:
    """Regression: the "no fixed size" sentinel must not be a bool.

    ``isinstance(False, int)`` is True in Python, so a False sentinel was
    returned as the image size itself and reached ``cv2.resize`` as
    ``imgsz=False`` -- which fails with an assertion about ``inv_scale_x``,
    nowhere near the actual mistake.
    """

    model = _FakeModel()
    detector = UltralyticsDetector(
        tmp_path / "model.pt", ModelDetectorConfig(image_size=1280), model=model
    )

    assert detector.image_size == 1280
    assert detector.image_size is not False
    detector.detect(np.zeros((80, 100, 3), dtype=np.uint8))
    assert model.last_predict_kwargs["imgsz"] == 1280


def test_a_non_square_onnx_input_falls_back_to_the_configured_size(tmp_path: Path) -> None:
    """Only a square lock can be expressed as one ``imgsz`` number."""

    path = _onnx_with_input_shape(tmp_path / "oblong.onnx", 640, 1024)
    detector = UltralyticsDetector(
        path, ModelDetectorConfig(image_size=1280), model=_FakeModel()
    )
    assert detector.image_size == 1280


def test_ultralytics_adapter_can_pin_kaggle_one_to_many_head(tmp_path: Path) -> None:
    model = _FakeModel()
    detector = UltralyticsDetector(
        tmp_path / "model.pt",
        ModelDetectorConfig(end2end=False),
        model=model,
    )
    detector.detect(np.zeros((80, 100, 3), dtype=np.uint8))
    assert model.last_predict_kwargs["end2end"] is False


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


def test_cropper_pads_normalizes_and_optionally_writes(tmp_path: Path, pcb_image: np.ndarray) -> None:
    detection = Detection("IC / QFN", 0.95, BoundingBox(90, 80, 130, 105))
    cropper = ComponentCropper(CropConfig(target_size=(96, 64), square=True))
    crops = cropper.extract(pcb_image, [detection], tmp_path)
    assert len(crops) == 1
    assert crops[0].image.shape == (64, 96, 3)
    assert crops[0].crop_bbox.width >= detection.bbox.width
    assert crops[0].path is not None and crops[0].path.is_file()
    assert "IC_QFN" in crops[0].filename
    assert crops[0].to_dict()["path"] == crops[0].path.name


def test_cropper_sanitizes_detection_id_in_output_filename(
    tmp_path: Path, pcb_image: np.ndarray
) -> None:
    detection = Detection(
        "resistor",
        0.8,
        BoundingBox(20, 20, 40, 35),
        detection_id="../../outside",
    )
    crop = ComponentCropper(CropConfig(target_size=None)).extract(
        pcb_image, [detection], tmp_path
    )[0]
    assert "/" not in crop.filename and "\\" not in crop.filename
    assert crop.path is not None and crop.path.parent == tmp_path.resolve()
