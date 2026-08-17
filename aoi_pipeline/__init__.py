"""OpenCV/Ultralytics/ONNX-ready AOI PCB pipeline for steps 0 through 6.1."""

from .alignment import PCBAligner
from .board import PCBLocalizer
from .classification import (
    ComponentClassifier,
    MANIFEST_SCHEMA,
    ONNXComponentClassifier,
    create_classifier,
)
from .calibration import (
    CalibrationRun,
    CameraCalibrationProfile,
    CameraUndistorter,
    UndistortionResult,
    calibrate_from_chessboards,
    save_calibration_profile,
)
from .config import (
    AlignmentConfig,
    BoardConfig,
    ClassificationConfig,
    CropConfig,
    CVDetectorConfig,
    ModelDetectorConfig,
    PipelineConfig,
    PreprocessConfig,
    TilingConfig,
)
from .cropping import ComponentCropper
from .detectors import (
    CVComponentDetector,
    ComponentDetector,
    MockComponentDetector,
    UltralyticsDetector,
    create_detector,
    non_maximum_suppression,
)
from .exceptions import (
    AlignmentError,
    AOIPipelineError,
    CalibrationProfileError,
    ClassifierConfigurationError,
    DetectorConfigurationError,
    ExportError,
    InvalidImageError,
    ModelDependencyError,
)
from .exporters import export_json, export_zip, render_annotations
from .image_io import encode_image, ensure_bgr, load_image
from .models import (
    AlignmentResult,
    BoardRegion,
    BoundingBox,
    ClassProbability,
    ComponentClassification,
    ComponentCrop,
    Detection,
    PipelineRun,
    PreprocessResult,
)
from .pipeline import AOIPipeline
from .preprocessing import ImagePreprocessor
from .tiling import (
    InferenceTile,
    TiledDetectionBatch,
    detect_with_adaptive_tiling,
    merge_tiled_detections,
    plan_inference_tiles,
)

__all__ = [
    "AOIPipeline",
    "AOIPipelineError",
    "AlignmentConfig",
    "AlignmentError",
    "AlignmentResult",
    "BoardConfig",
    "BoardRegion",
    "BoundingBox",
    "CalibrationProfileError",
    "CalibrationRun",
    "ClassProbability",
    "ClassificationConfig",
    "ClassifierConfigurationError",
    "CVComponentDetector",
    "CVDetectorConfig",
    "CameraCalibrationProfile",
    "CameraUndistorter",
    "ComponentCrop",
    "ComponentClassification",
    "ComponentClassifier",
    "ComponentCropper",
    "ComponentDetector",
    "CropConfig",
    "Detection",
    "DetectorConfigurationError",
    "ExportError",
    "ImagePreprocessor",
    "InferenceTile",
    "InvalidImageError",
    "MockComponentDetector",
    "ModelDependencyError",
    "ModelDetectorConfig",
    "MANIFEST_SCHEMA",
    "ONNXComponentClassifier",
    "PCBAligner",
    "PCBLocalizer",
    "PipelineConfig",
    "PipelineRun",
    "PreprocessConfig",
    "PreprocessResult",
    "TiledDetectionBatch",
    "TilingConfig",
    "UltralyticsDetector",
    "UndistortionResult",
    "calibrate_from_chessboards",
    "create_detector",
    "create_classifier",
    "detect_with_adaptive_tiling",
    "encode_image",
    "ensure_bgr",
    "export_json",
    "export_zip",
    "load_image",
    "merge_tiled_detections",
    "non_maximum_suppression",
    "plan_inference_tiles",
    "render_annotations",
    "save_calibration_profile",
]

__version__ = "0.2.0"
