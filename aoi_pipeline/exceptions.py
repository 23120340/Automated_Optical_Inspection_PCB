"""Domain-specific exceptions for the AOI pipeline."""


class AOIPipelineError(RuntimeError):
    """Base error raised by this package."""


class InvalidImageError(AOIPipelineError, ValueError):
    """Raised when an input cannot be interpreted as a BGR image."""


class CalibrationProfileError(AOIPipelineError, ValueError):
    """Raised when camera calibration data is missing or incompatible."""


class AlignmentError(AOIPipelineError):
    """Raised when strict image alignment is requested and cannot be completed."""


class DetectorConfigurationError(AOIPipelineError, ValueError):
    """Raised when a component detector is configured incorrectly."""


class ModelDependencyError(AOIPipelineError, ImportError):
    """Raised when an optional model runtime is not installed."""


class ExportError(AOIPipelineError):
    """Raised when a run cannot be exported."""
