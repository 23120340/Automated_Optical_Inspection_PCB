"""Manifest-bound full-board solder-defect instance segmentation.

This stage is diagnostic and independent from both solder-joint ROI grading
and the pass-2 lead detector.  Keeping the contract here prevents a classifier
manifest from being paired with a YOLO segmentation graph (or vice versa).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..evidence import file_digest as _sha256_file
from ..config import ModelDetectorConfig, SolderDefectDetectionConfig
from ..detection.detectors import UltralyticsDetector
from ..exceptions import DetectorConfigurationError
from ..models import Detection


SOLDER_DEFECT_MANIFEST_SCHEMA = "aoi-external-yolo-segmentation/1.0"
SOLDER_DEFECT_TASK = "solder_defect_instance_segmentation"
SOLDER_DEFECT_PIPELINE_STEP = "standalone_solder_defect_localization"
SOLDER_DEFECT_SOURCE = "solder_defect_segment"
#: Tên lớp của model segmentation đầu tiên được nạp vào vai trò này. Giữ lại để
#: các test và script cũ còn tham chiếu được, **không** còn là ràng buộc.
SOLDER_DEFECT_CLASS_NAMES = (
    "Dry_joint",
    "Incorrect_installation",
    "PCB_damage",
    "Short_circuit",
)

# Vai trò này từng bị hàn cứng vào đúng một artifact: schema phải là
# ``aoi-external-yolo-segmentation/1.0``, task phải là instance segmentation, và
# ``class_names`` phải trùng từng chữ với bốn tên trên. Hệ quả là một detector
# train từ chính notebook của dự án -- đúng bài toán, đúng bước, hai lớp khác --
# bị từ chối, và cách duy nhất để nạp được là sửa manifest cho khớp, tức là bắt
# hồ sơ nói dối về model.
#
# Giờ vai trò nhận cả hai hình thái. Tên lớp **đọc từ manifest**, vì chỉ manifest
# mới biết model đã học gì; danh sách cứng ở trên không biết.
SOLDER_DEFECT_SCHEMAS = {
    "aoi-external-yolo-segmentation/1.0": "segment",
    "aoi-solder-defect-detection/1.0": "detect",
}
SOLDER_DEFECT_TASKS = {
    "solder_defect_instance_segmentation": "segment",
    "solder_defect_detection": "detect",
}
MAX_MANIFEST_BYTES = 1_048_576


@dataclass(frozen=True, slots=True)
class SolderDefectContract:
    """Validated subset of the external YOLO segmentation manifest."""

    class_names: tuple[str, ...]
    image_size: int
    confidence: float
    iou: float
    mask_threshold: float
    max_detections: int
    end2end: bool
    model_version: str
    model_sha256: str
    model_bytes: int | None
    #: ``segment`` hoặc ``detect``. Ultralytics cần biết đúng hình thái để giải
    #: mã output; đoán sai thì nó đi tìm hệ số mask trong một tensor không có.
    #: Mặc định giữ ``segment`` cho các contract dựng bằng tay trong test cũ.
    ultralytics_task: str = "segment"


class SolderDefectDetector(UltralyticsDetector):
    """YOLOv8-seg adapter whose class order and artifact hash are pinned."""

    def __init__(
        self,
        model_path: str | Path,
        manifest_path: str | Path | Mapping[str, Any],
        config: SolderDefectDetectionConfig | None = None,
        *,
        model: Any | None = None,
    ) -> None:
        settings = config or SolderDefectDetectionConfig()
        resolved_path = Path(model_path).expanduser().resolve()
        if resolved_path.suffix.lower() != ".onnx":
            raise DetectorConfigurationError(
                "Solder defect segmentation only accepts an .onnx artifact"
            )

        self.manifest = _load_manifest(manifest_path)
        self.contract = validate_solder_defect_manifest(self.manifest)
        self.class_names = self.contract.class_names
        self.model_version = self.contract.model_version
        self.mask_threshold = _probability(
            settings.mask_threshold
            if settings.mask_threshold is not None
            else self.contract.mask_threshold,
            "mask_threshold",
        )

        # A model injection is exclusively a test seam, matching the base
        # adapter. Production paths always verify both size and digest before
        # Ultralytics gets an opportunity to load the graph.
        if model is None:
            if not resolved_path.is_file():
                raise DetectorConfigurationError(
                    f"Solder defect model does not exist: {resolved_path}"
                )
            if (
                self.contract.model_bytes is not None
                and resolved_path.stat().st_size != self.contract.model_bytes
            ):
                raise DetectorConfigurationError(
                    "Solder defect model byte size does not match model_manifest.json"
                )
            actual_sha256 = _sha256_file(resolved_path)
            if actual_sha256.lower() != self.contract.model_sha256.lower():
                raise DetectorConfigurationError(
                    "Solder defect model SHA-256 does not match model_manifest.json"
                )

        runtime_config = ModelDetectorConfig(
            confidence=_probability(
                settings.confidence
                if settings.confidence is not None
                else self.contract.confidence,
                "confidence",
            ),
            iou=_probability(
                settings.iou if settings.iou is not None else self.contract.iou,
                "iou",
            ),
            image_size=_positive_int(
                settings.image_size
                if settings.image_size is not None
                else self.contract.image_size,
                "image_size",
            ),
            device=settings.device,
            max_detections=_positive_int(
                settings.max_detections
                if settings.max_detections is not None
                else self.contract.max_detections,
                "max_detections",
            ),
            end2end=self.contract.end2end,
            tta=bool(settings.tta),
        )
        super().__init__(
            resolved_path,
            runtime_config,
            model=model,
            # Lấy từ contract, không ghi cứng: model detect-only chỉ có một
            # output [1, 4+nc, N], không có prototype mask để giải.
            task=self.contract.ultralytics_task,
            source=SOLDER_DEFECT_SOURCE,
        )

    def detect(
        self, image: np.ndarray, *, confidence: float | None = None
    ) -> list[Detection]:
        """Localize solder defects without contributing a 6.2 verdict."""

        detections = super().detect(image, confidence=confidence)
        for detection in detections:
            class_id = detection.class_id
            if class_id is None or not 0 <= class_id < len(self.class_names):
                raise DetectorConfigurationError(
                    "Solder defect model returned a class outside its manifest"
                )
            expected_label = self.class_names[class_id]
            if detection.label != expected_label:
                raise DetectorConfigurationError(
                    "Solder defect model class names do not match model_manifest.json: "
                    f"class {class_id} is '{detection.label}', expected '{expected_label}'"
                )
            detection.metadata.update(
                {
                    "manifest_schema": SOLDER_DEFECT_MANIFEST_SCHEMA,
                    "model_version": self.model_version,
                    "mask_threshold": float(self.mask_threshold),
                    "diagnostic_only": True,
                }
            )
        return detections


def create_solder_defect_detector(
    model_path: str | Path | None,
    manifest_path: str | Path | Mapping[str, Any] | None,
    config: SolderDefectDetectionConfig | None = None,
    *,
    model: Any | None = None,
) -> SolderDefectDetector | None:
    """Create the diagnostic detector only from a complete artifact pair."""

    if model_path is None and manifest_path is None:
        return None
    if model_path is None or manifest_path is None:
        raise DetectorConfigurationError(
            "Solder defect detection requires both best.onnx and its "
            "segmentation model_manifest.json"
        )
    return SolderDefectDetector(model_path, manifest_path, config, model=model)


def validate_solder_defect_manifest(
    manifest: Mapping[str, Any],
) -> SolderDefectContract:
    """Validate that a manifest describes the supported YOLOv8-seg stage."""

    schema = str(manifest.get("schema_version", ""))
    # Bản detector do notebook của dự án xuất ghi ``schema_version: 1``; chấp nhận
    # nó khi ``task`` đã tự nói rõ hình thái, thay vì bắt sửa file.
    if schema in SOLDER_DEFECT_SCHEMAS:
        model_shape = SOLDER_DEFECT_SCHEMAS[schema]
    elif str(manifest.get("task", "")) in SOLDER_DEFECT_TASKS:
        model_shape = SOLDER_DEFECT_TASKS[str(manifest["task"])]
    else:
        raise DetectorConfigurationError(
            "Unsupported solder defect manifest; schema_version must be one of "
            + ", ".join(sorted(SOLDER_DEFECT_SCHEMAS))
            + " or task one of "
            + ", ".join(sorted(SOLDER_DEFECT_TASKS))
        )
    task = str(manifest.get("task", ""))
    if task not in SOLDER_DEFECT_TASKS:
        raise DetectorConfigurationError(
            "Solder defect manifest task must be one of "
            + ", ".join(sorted(SOLDER_DEFECT_TASKS))
        )
    if SOLDER_DEFECT_TASKS[task] != model_shape:
        raise DetectorConfigurationError(
            f"Solder defect manifest task {task!r} does not match schema {schema!r}"
        )
    # Hai tên cùng chỉ một bước; bản đầu nhận đúng một chuỗi nên notebook của
    # chính dự án cũng bị từ chối.
    if manifest.get("pipeline_step") not in {
        SOLDER_DEFECT_PIPELINE_STEP,
        "6_2_solder_defect_localization",
    }:
        raise DetectorConfigurationError(
            "Solder defect manifest pipeline_step must be "
            f"{SOLDER_DEFECT_PIPELINE_STEP}"
        )
    if str(manifest.get("model_format", "")).lower() != "onnx":
        raise DetectorConfigurationError(
            "Solder defect manifest model_format must be onnx"
        )

    raw_classes = manifest.get("class_names")
    if not isinstance(raw_classes, list):
        raise DetectorConfigurationError(
            "Solder defect manifest class_names must be a list"
        )
    class_names = tuple(str(value).strip() for value in raw_classes)
    if not class_names or any(not name for name in class_names):
        raise DetectorConfigurationError(
            "Solder defect manifest class_names must be a non-empty list of names"
        )
    raw_class_map = manifest.get("class_map")
    if raw_class_map is not None:
        class_map = _mapping(raw_class_map, "class_map")
        expected_map = {str(index): name for index, name in enumerate(class_names)}
        normalized_map = {str(key): str(value) for key, value in class_map.items()}
        if normalized_map != expected_map:
            raise DetectorConfigurationError(
                "Solder defect manifest class_map disagrees with class_names"
            )

    input_spec = _mapping(manifest.get("input"), "input")
    if str(input_spec.get("layout", "")).upper() != "NCHW":
        raise DetectorConfigurationError("Solder defect input layout must be NCHW")
    if str(input_spec.get("color_space", "")).upper() != "RGB":
        raise DetectorConfigurationError("Solder defect input color_space must be RGB")
    if str(input_spec.get("resize_mode", "")).lower() != "letterbox":
        raise DetectorConfigurationError(
            "Solder defect input resize_mode must be letterbox"
        )
    shape = input_spec.get("shape")
    if (
        not isinstance(shape, Sequence)
        or isinstance(shape, (str, bytes))
        or len(shape) != 4
    ):
        raise DetectorConfigurationError(
            "Solder defect input shape must be [batch, channels, height, width]"
        )
    try:
        batch, channels, height, width = (int(value) for value in shape)
    except (TypeError, ValueError) as exc:
        raise DetectorConfigurationError(
            "Solder defect input shape must contain integers"
        ) from exc
    if batch != 1 or channels != 3 or height <= 0 or height != width:
        raise DetectorConfigurationError(
            "Solder defect input must be a fixed [1, 3, size, size] tensor"
        )

    head = _mapping(manifest.get("head"), "head")
    head_type = str(head.get("type", "")).lower()
    wanted = "segment" if model_shape == "segment" else "detect"
    if wanted not in head_type:
        raise DetectorConfigurationError(
            f"Solder defect manifest head.type must describe {wanted} "
            f"to match task {task!r}, got {head_type!r}"
        )
    if bool(head.get("end2end", False)):
        raise DetectorConfigurationError(
            "Solder defect model must expose the external-NMS segmentation head"
        )
    max_detections = _positive_int(head.get("max_det", 300), "head.max_det")

    postprocessing = _mapping(manifest.get("postprocessing"), "postprocessing")
    confidence = _probability(
        postprocessing.get("recommended_confidence"),
        "postprocessing.recommended_confidence",
    )
    iou = _probability(
        postprocessing.get("recommended_iou_nms"),
        "postprocessing.recommended_iou_nms",
    )
    mask_threshold = _probability(
        postprocessing.get("mask_threshold"),
        "postprocessing.mask_threshold",
    )

    model_spec = _mapping(manifest.get("model"), "model")
    model_version = str(model_spec.get("version", "")).strip()
    if not model_version:
        raise DetectorConfigurationError(
            "Solder defect manifest model.version cannot be empty"
        )
    architecture = str(model_spec.get("architecture", "")).strip().lower()
    # ``-seg`` chỉ bắt buộc khi manifest tự khai là segmentation. Một detector
    # thì kiến trúc của nó *không* được kết thúc bằng ``-seg``, và ép nó khai như
    # vậy chính là bắt hồ sơ nói dối -- thứ vừa gây ra một model sai nguồn.
    if model_shape == "segment" and not architecture.endswith("-seg"):
        raise DetectorConfigurationError(
            "Solder defect manifest model.architecture must be a segmentation model"
        )
    if model_shape == "detect" and architecture.endswith("-seg"):
        raise DetectorConfigurationError(
            "Solder defect manifest declares task=detect but a -seg architecture"
        )
    model_sha256 = str(model_spec.get("sha256", "")).strip().lower()
    if len(model_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in model_sha256
    ):
        raise DetectorConfigurationError(
            "Solder defect manifest model.sha256 must be a 64-character hex digest"
        )
    raw_bytes = model_spec.get("bytes")
    model_bytes = (
        None
        if raw_bytes is None
        else _positive_int(raw_bytes, "model.bytes")
    )

    return SolderDefectContract(
        class_names=class_names,
        image_size=height,
        confidence=confidence,
        iou=iou,
        mask_threshold=mask_threshold,
        max_detections=max_detections,
        end2end=False,
        model_version=model_version,
        model_sha256=model_sha256,
        model_bytes=model_bytes,
        ultralytics_task=model_shape,
    )


def _load_manifest(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise DetectorConfigurationError(
            f"Solder defect manifest does not exist: {path}"
        )
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise DetectorConfigurationError(
            "Solder defect manifest exceeds the 1 MB limit"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DetectorConfigurationError(
            f"Invalid solder defect manifest JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise DetectorConfigurationError(
            "Solder defect manifest root must be an object"
        )
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DetectorConfigurationError(
            f"Solder defect manifest {name} must be an object"
        )
    return value


def _probability(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DetectorConfigurationError(f"{name} must be a number in [0, 1]") from exc
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        raise DetectorConfigurationError(f"{name} must be in [0, 1]")
    return number


def _positive_int(value: Any, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise DetectorConfigurationError(f"{name} must be a positive integer") from exc
    if isinstance(value, (float, np.floating)) and not float(value).is_integer():
        raise DetectorConfigurationError(f"{name} must be a positive integer")
    if number <= 0:
        raise DetectorConfigurationError(f"{name} must be a positive integer")
    return number


