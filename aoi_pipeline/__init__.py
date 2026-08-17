"""OpenCV/Ultralytics-ready AOI PCB pipeline for steps 0 through 5."""

from .alignment import PCBAligner
from .board import PCBLocalizer
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
    CropConfig,
    CVDetectorConfig,
    ModelDetectorConfig,
    PipelineConfig,
    PreprocessConfig,
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
    ComponentCrop,
    Detection,
    PipelineRun,
    PreprocessResult,
)
from .pipeline import AOIPipeline
from .preprocessing import ImagePreprocessor

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
    "CVComponentDetector",
    "CVDetectorConfig",
    "CameraCalibrationProfile",
    "CameraUndistorter",
    "ComponentCrop",
    "ComponentCropper",
    "ComponentDetector",
    "CropConfig",
    "Detection",
    "DetectorConfigurationError",
    "ExportError",
    "ImagePreprocessor",
    "InvalidImageError",
    "MockComponentDetector",
    "ModelDependencyError",
    "ModelDetectorConfig",
    "PCBAligner",
    "PCBLocalizer",
    "PipelineConfig",
    "PipelineRun",
    "PreprocessConfig",
    "PreprocessResult",
    "UltralyticsDetector",
    "UndistortionResult",
    "calibrate_from_chessboards",
    "create_detector",
    "encode_image",
    "ensure_bgr",
    "export_json",
    "export_zip",
    "load_image",
    "non_maximum_suppression",
    "render_annotations",
    "save_calibration_profile",
]

__version__ = "0.2.0"
