"""OpenCV/Ultralytics-ready AOI PCB pipeline for steps 0 through 5."""

from .alignment import PCBAligner
from .board import PCBLocalizer
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
    "CVComponentDetector",
    "CVDetectorConfig",
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
    "create_detector",
    "encode_image",
    "ensure_bgr",
    "export_json",
    "export_zip",
    "load_image",
    "non_maximum_suppression",
    "render_annotations",
]

__version__ = "0.1.0"
