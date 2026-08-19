"""OpenCV/Ultralytics/ONNX-ready AOI PCB pipeline for steps 0 through 6.1.

The package is laid out by pipeline stage. Imports below follow the same order
as the data does, from the foundation up to the facade:

    core/ -> imaging/ -> board/ -> detection/ -> inspection/ -> classification
                                                            -> export/

Everything a caller needs is re-exported here, so ``from aoi_pipeline import X``
stays stable no matter which module X actually lives in.
"""

# --- foundation ------------------------------------------------------------
from .core.exceptions import (
    AlignmentError,
    AOIPipelineError,
    CalibrationProfileError,
    ClassifierConfigurationError,
    DetectorConfigurationError,
    ExportError,
    InvalidImageError,
    ModelDependencyError,
)
from .core.image_io import encode_image, ensure_bgr, letterbox_normalize, load_image
from .core.models import (
    AlignmentResult,
    BoardRegion,
    BoundingBox,
    ClassProbability,
    ComponentClassification,
    ComponentCrop,
    Detection,
    PipelineRun,
    PreprocessResult,
    SolderFeatures,
    SolderJoint,
    SolderJointCrop,
    SolderVerdict,
)

# --- every stage's knobs, in one place -------------------------------------
from .config import (
    AlignmentConfig,
    BoardConfig,
    CadConfig,
    ClassificationConfig,
    CropConfig,
    CVDetectorConfig,
    FusionConfig,
    ModelDetectorConfig,
    PadProfile,
    PipelineConfig,
    PreprocessConfig,
    LeadFusionConfig,
    SolderGradingConfig,
    SolderJointConfig,
    TilingConfig,
    terminal_geometry,
)

# --- step 0-1: make the image measurable -----------------------------------
from .imaging.calibration import (
    CalibrationRun,
    CameraCalibrationProfile,
    CameraUndistorter,
    UndistortionResult,
    calibrate_from_chessboards,
    save_calibration_profile,
)
from .imaging.preprocessing import ImagePreprocessor

# --- step 2-3: find the board in the frame ---------------------------------
from .board.alignment import PCBAligner
from .board.localization import PCBLocalizer

# --- step 4: detect components ---------------------------------------------
from .detection.detectors import (
    ComponentDetector,
    CVComponentDetector,
    MockComponentDetector,
    UltralyticsDetector,
    create_detector,
    non_maximum_suppression,
)
from .detection.tiling import (
    InferenceTile,
    TiledDetectionBatch,
    detect_with_adaptive_tiling,
    merge_tiled_detections,
    plan_inference_tiles,
)

# --- step 5-5.5: cut the regions to inspect --------------------------------
from .inspection.cropping import ComponentCropper
from .inspection.solder import (
    ComponentFrame,
    SolderJointCropper,
    derive_solder_joints,
    estimate_component_angle,
)
from .inspection.cad import (
    BoardCad,
    CAD_LOADERS,
    CadComponent,
    CadError,
    CadPad,
    CadRegistration,
    classes_agree,
    designator_to_class,
    is_informative_label,
    load_cad,
    register_cad,
    register_from_fiducials,
    save_cad_json,
)
from .inspection.fusion import CadFinding, FusionResult, fuse_solder_joints
from .inspection.leads import (
    LEAD_CLASSES,
    LeadFusionResult,
    fuse_detected_leads,
    split_lead_detections,
)

# --- step 6.2: grade the solder ---------------------------------------------
from .grading.classifier import (
    MANIFEST_SCHEMA as SOLDER_MANIFEST_SCHEMA,
    ONNXSolderClassifier,
    create_solder_classifier,
)
from .grading.features import measure_solder, segment_solder
from .grading.inspector import SolderInspector
from .grading.rules import (
    COMPONENT_CLASSES,
    JOINT_CLASSES,
    grade_component_by_rules,
    grade_joint_by_rules,
)

# --- step 6.1: classify component families ---------------------------------
from .classification import (
    ComponentClassifier,
    MANIFEST_SCHEMA,
    ONNXComponentClassifier,
    create_classifier,
)

# --- packaging the result --------------------------------------------------
from .export.exporters import (
    cad_findings_csv,
    export_json,
    export_zip,
    solder_joints_csv,
    solder_verdicts_csv,
)
from .export.overlays import (
    render_annotations,
    render_solder_overlay,
    render_verdict_overlay,
)

# --- the facade ------------------------------------------------------------
from .pipeline import AOIPipeline


__all__ = [
    "AOIPipeline",
    "AOIPipelineError",
    "AlignmentConfig",
    "AlignmentError",
    "AlignmentResult",
    "BoardCad",
    "BoardConfig",
    "BoardRegion",
    "BoundingBox",
    "CadComponent",
    "CadConfig",
    "CadError",
    "CadFinding",
    "CadPad",
    "CadRegistration",
    "CalibrationProfileError",
    "CalibrationRun",
    "ClassProbability",
    "ClassificationConfig",
    "ClassifierConfigurationError",
    "CAD_LOADERS",
    "COMPONENT_CLASSES",
    "CVComponentDetector",
    "CVDetectorConfig",
    "CameraCalibrationProfile",
    "CameraUndistorter",
    "ComponentCrop",
    "ComponentClassification",
    "ComponentClassifier",
    "ComponentCropper",
    "ComponentDetector",
    "ComponentFrame",
    "CropConfig",
    "Detection",
    "DetectorConfigurationError",
    "ExportError",
    "FusionConfig",
    "FusionResult",
    "ImagePreprocessor",
    "InferenceTile",
    "JOINT_CLASSES",
    "LEAD_CLASSES",
    "LeadFusionConfig",
    "LeadFusionResult",
    "InvalidImageError",
    "MockComponentDetector",
    "ModelDependencyError",
    "ModelDetectorConfig",
    "MANIFEST_SCHEMA",
    "ONNXComponentClassifier",
    "PCBAligner",
    "PCBLocalizer",
    "PadProfile",
    "PipelineConfig",
    "PipelineRun",
    "PreprocessConfig",
    "PreprocessResult",
    "SOLDER_MANIFEST_SCHEMA",
    "SolderFeatures",
    "SolderGradingConfig",
    "SolderInspector",
    "SolderJoint",
    "SolderJointConfig",
    "SolderJointCrop",
    "SolderJointCropper",
    "SolderVerdict",
    "ONNXSolderClassifier",
    "TiledDetectionBatch",
    "TilingConfig",
    "UltralyticsDetector",
    "UndistortionResult",
    "cad_findings_csv",
    "classes_agree",
    "calibrate_from_chessboards",
    "create_detector",
    "create_classifier",
    "create_solder_classifier",
    "derive_solder_joints",
    "designator_to_class",
    "detect_with_adaptive_tiling",
    "encode_image",
    "ensure_bgr",
    "estimate_component_angle",
    "export_json",
    "fuse_detected_leads",
    "fuse_solder_joints",
    "is_informative_label",
    "export_zip",
    "grade_component_by_rules",
    "grade_joint_by_rules",
    "letterbox_normalize",
    "load_cad",
    "load_image",
    "measure_solder",
    "merge_tiled_detections",
    "non_maximum_suppression",
    "plan_inference_tiles",
    "register_cad",
    "register_from_fiducials",
    "render_annotations",
    "render_solder_overlay",
    "render_verdict_overlay",
    "save_cad_json",
    "save_calibration_profile",
    "segment_solder",
    "split_lead_detections",
    "solder_joints_csv",
    "solder_verdicts_csv",
    "terminal_geometry",
]

__version__ = "0.4.0"
