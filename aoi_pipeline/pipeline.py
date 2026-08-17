"""Facade that composes the AOI PCB steps 0 through 5."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from collections.abc import Mapping
from typing import Sequence

import numpy as np

from .alignment import PCBAligner
from .board import PCBLocalizer
from .config import PipelineConfig
from .cropping import ComponentCropper
from .detectors import ComponentDetector, CVComponentDetector, create_detector
from .exceptions import DetectorConfigurationError
from .exporters import export_json as write_json
from .exporters import export_zip as write_zip
from .image_io import ImageSource, load_image
from .models import (
    AlignmentResult,
    BoardRegion,
    ComponentCrop,
    Detection,
    PipelineRun,
    PreprocessResult,
    utc_now_iso,
)
from .preprocessing import ImagePreprocessor


class AOIPipeline:
    """Reusable local pipeline for the diagram's steps 0-5.

    Images passed to individual stage methods are BGR NumPy arrays. ``run`` also
    accepts uploaded bytes or a local path through the step-0 loader.
    """

    def __init__(
        self,
        config: PipelineConfig | Mapping[str, object] | None = None,
        detector: ComponentDetector | None = None,
        model_path: str | Path | None = None,
    ) -> None:
        self.config = (
            config
            if isinstance(config, PipelineConfig)
            else PipelineConfig.from_mapping(config)
        )
        if detector is not None and model_path is not None:
            raise DetectorConfigurationError("Pass either detector or model_path, not both")
        self.preprocessor = ImagePreprocessor(self.config.preprocess)
        self.aligner = PCBAligner(self.config.alignment)
        self.localizer = PCBLocalizer(self.config.board)
        self.detector = detector or create_detector(
            model_path,
            mode=self.config.detector_mode,
            cv_config=self.config.cv_detector,
            model_config=self.config.model_detector,
        )
        self.cropper = ComponentCropper(self.config.crop)

    @staticmethod
    def load_image(source: ImageSource) -> np.ndarray:
        """Step 0: decode an imported image without accessing a camera."""

        return load_image(source)

    def preprocess(self, image: np.ndarray) -> PreprocessResult:
        """Step 1: optionally undistort, then resize and enhance an image."""

        return self.preprocessor.process(image)

    def align(
        self, image: np.ndarray, reference: np.ndarray | None = None
    ) -> AlignmentResult:
        """Step 2: align an image to a golden/reference image."""

        return self.aligner.align(image, reference)

    def detect_board(self, image: np.ndarray) -> BoardRegion:
        """Step 3: locate the PCB region."""

        return self.localizer.locate(image)

    def detect_components(
        self, image: np.ndarray, board_region: BoardRegion | None = None
    ) -> list[Detection]:
        """Step 4: detect components in a board ROI and return full-image boxes."""

        bgr = load_image(image)
        if board_region is None:
            return self.detector.detect(bgr)

        height, width = bgr.shape[:2]
        x1, y1, x2, y2 = board_region.bbox.clamp(width, height).to_int()
        roi = bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return []
        local_detections = self.detector.detect(roi)
        translated: list[Detection] = []
        for detection in local_detections:
            global_bbox = detection.bbox.translated(x1, y1).clamp(width, height)
            if global_bbox.width <= 0 or global_bbox.height <= 0:
                continue
            metadata = dict(detection.metadata)
            metadata["roi_offset"] = [x1, y1]
            translated.append(
                Detection(
                    label=detection.label,
                    confidence=detection.confidence,
                    bbox=global_bbox,
                    class_id=detection.class_id,
                    source=detection.source,
                    detection_id=detection.detection_id,
                    metadata=metadata,
                )
            )
        return translated

    def make_crops(
        self,
        image: np.ndarray,
        detections: Sequence[Detection],
        output_dir: str | Path | None = None,
    ) -> list[ComponentCrop]:
        """Step 5: extract padded and normalized component crops."""

        return self.cropper.extract(image, detections, output_dir)

    def run(
        self,
        image: ImageSource,
        reference: ImageSource | None = None,
        source_name: str | None = None,
        crop_dir: str | Path | None = None,
    ) -> PipelineRun:
        """Execute steps 0-5 and retain all artifacts for the local UI/export."""

        started_at = utc_now_iso()
        input_image = load_image(image)
        inferred_name = source_name or _source_name(image)
        preprocessed = self.preprocess(input_image)

        reference_image: np.ndarray | None = None
        if reference is not None:
            decoded_reference = load_image(reference)
            reference_image = self.preprocess(decoded_reference).image
        aligned = self.align(preprocessed.image, reference_image)
        board = self.detect_board(aligned.image)
        detections = self.detect_components(aligned.image, board)
        crops = self.make_crops(aligned.image, detections, crop_dir)

        warnings = list(preprocessed.warnings)
        if not aligned.success:
            warnings.append(aligned.message)
        if board.method == "full_image_fallback":
            warnings.append("PCB contour was not found; the complete image was used as the board ROI.")
        if isinstance(self.detector, CVComponentDetector):
            warnings.append(
                "Step 4 is using OpenCV candidate proposals; labels are not component classifications."
            )

        return PipelineRun(
            source_name=inferred_name,
            input_image=input_image,
            preprocess_result=preprocessed,
            alignment_result=aligned,
            board_region=board,
            detections=detections,
            crops=crops,
            started_at=started_at,
            finished_at=utc_now_iso(),
            warnings=warnings,
            config=asdict(self.config),
        )

    @staticmethod
    def export_json(run: PipelineRun, path: str | Path) -> Path:
        return write_json(run, path)

    @staticmethod
    def export_zip(
        run: PipelineRun,
        path: str | Path,
        *,
        include_input: bool = True,
        include_intermediate: bool = True,
        include_crops: bool = True,
    ) -> Path:
        return write_zip(
            run,
            path,
            include_input=include_input,
            include_intermediate=include_intermediate,
            include_crops=include_crops,
        )


def _source_name(source: ImageSource) -> str:
    if isinstance(source, (str, Path)):
        return Path(source).name
    if hasattr(source, "name") and isinstance(source.name, str):
        return Path(source.name).name
    return "uploaded_image"
