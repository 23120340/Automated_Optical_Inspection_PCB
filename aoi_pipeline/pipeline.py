"""Facade that composes the AOI PCB steps 0 through 6.1."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from collections.abc import Mapping
from typing import Sequence

import numpy as np

from .alignment import PCBAligner
from .board import PCBLocalizer
from .classification import ComponentClassifier, create_classifier
from .config import PipelineConfig
from .cropping import ComponentCropper
from .detectors import (
    ComponentDetector,
    CVComponentDetector,
    UltralyticsDetector,
    create_detector,
)
from .exceptions import DetectorConfigurationError
from .exporters import export_json as write_json
from .exporters import export_zip as write_zip
from .image_io import ImageSource, load_image
from .models import (
    AlignmentResult,
    BoardRegion,
    ComponentClassification,
    ComponentCrop,
    Detection,
    PipelineRun,
    PreprocessResult,
    SolderJointCrop,
    utc_now_iso,
)
from .preprocessing import ImagePreprocessor
from .solder import SolderJointCropper
from .tiling import detect_with_adaptive_tiling


class AOIPipeline:
    """Reusable local pipeline for the diagram's steps 0-6.1.

    Images passed to individual stage methods are BGR NumPy arrays. ``run`` also
    accepts uploaded bytes or a local path through the step-0 loader.
    """

    def __init__(
        self,
        config: PipelineConfig | Mapping[str, object] | None = None,
        detector: ComponentDetector | None = None,
        model_path: str | Path | None = None,
        classifier: ComponentClassifier | None = None,
        classifier_model_path: str | Path | None = None,
        classifier_manifest_path: str | Path | Mapping[str, object] | None = None,
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
        self.last_detection_metrics: dict[str, object] = {}
        self.cropper = ComponentCropper(self.config.crop)
        self.solder_cropper = SolderJointCropper(self.config.solder)
        if classifier is not None and (
            classifier_model_path is not None or classifier_manifest_path is not None
        ):
            raise DetectorConfigurationError(
                "Pass either classifier or classifier artifact paths, not both"
            )
        self.classifier = classifier or create_classifier(
            classifier_model_path,
            classifier_manifest_path,
            self.config.classification,
        )

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
        self,
        image: np.ndarray,
        board_region: BoardRegion | None = None,
        *,
        frame_id: str = "import_0000",
    ) -> list[Detection]:
        """Step 4: detect a board ROI with adaptive tiled inference."""

        bgr = load_image(image)
        height, width = bgr.shape[:2]
        if board_region is None:
            x1, y1, x2, y2 = 0, 0, width, height
        else:
            x1, y1, x2, y2 = board_region.bbox.clamp(width, height).to_int()
        roi = bgr[y1:y2, x1:x2]
        if roi.size == 0:
            self.last_detection_metrics = {
                "tiling_applied": False,
                "tile_count": 0,
                "raw_detection_count": 0,
                "duplicates_removed": 0,
            }
            return []

        tiling_policy = self.config.tiling
        tiling_reason: str | None = None
        if isinstance(self.detector, CVComponentDetector):
            # CV proposals use area ratios tied to the complete ROI and are an
            # explicit demo, so tiling them would change their semantics.
            tiling_policy = replace(tiling_policy, mode="off")
            tiling_reason = "disabled_for_cv_demo"
        max_detections = (
            self.config.cv_detector.max_detections
            if isinstance(self.detector, CVComponentDetector)
            else self.config.model_detector.max_detections
        )
        tile_detect = None
        applied_detail_confidence = tiling_policy.detail_confidence
        detail_class_confidence = dict(tiling_policy.detail_class_confidence)
        if (
            isinstance(self.detector, UltralyticsDetector)
            and tiling_policy.detail_confidence is not None
        ):
            applied_detail_confidence = min(
                float(self.config.model_detector.confidence),
                float(tiling_policy.detail_confidence),
            )

            def detect_detail_tile(tile: np.ndarray) -> list[Detection]:
                detections = self.detector.detect(
                    tile,
                    confidence=applied_detail_confidence,
                )
                return [
                    detection
                    for detection in detections
                    if detection.confidence
                    >= float(
                        detail_class_confidence.get(
                            detection.label,
                            applied_detail_confidence,
                        )
                    )
                ]

            tile_detect = detect_detail_tile
        batch = detect_with_adaptive_tiling(
            self.detector.detect,
            roi,
            tiling_policy,
            tile_detect=tile_detect,
            max_detections=max_detections,
            frame_id=frame_id,
        )
        self.last_detection_metrics = {
            **batch.metrics(offset=(x1, y1)),
            "coordinate_space": "analysis_image_pixels",
            "roi_bbox": [x1, y1, x2, y2],
            "detail_confidence": applied_detail_confidence,
            "detail_class_confidence": detail_class_confidence,
        }
        if tiling_reason:
            self.last_detection_metrics["tiling_reason"] = tiling_reason

        translated: list[Detection] = []
        for detection in batch.detections:
            global_bbox = detection.bbox.translated(x1, y1).clamp(width, height)
            if global_bbox.width <= 0 or global_bbox.height <= 0:
                continue
            metadata = dict(detection.metadata)
            metadata["roi_offset"] = [x1, y1]
            metadata["coordinate_space"] = "analysis_image_pixels"
            for bbox_key in ("tile_bbox", "tile_ownership_bbox"):
                tile_bbox = metadata.get(bbox_key)
                if isinstance(tile_bbox, list) and len(tile_bbox) == 4:
                    metadata[f"{bbox_key}_roi"] = list(tile_bbox)
                    metadata[bbox_key] = [
                        float(tile_bbox[0]) + x1,
                        float(tile_bbox[1]) + y1,
                        float(tile_bbox[2]) + x1,
                        float(tile_bbox[3]) + y1,
                    ]
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

    def make_solder_crops(
        self,
        image: np.ndarray,
        detections: Sequence[Detection],
        output_dir: str | Path | None = None,
    ) -> list[SolderJointCrop]:
        """Step 5.5: derive solder-joint ROIs and cut them out.

        Kept separate from :meth:`make_crops` on purpose. Step 6.1 was trained
        on body-tight crops, so widening those to reveal the fillet would shift
        the classifier's input distribution; joint inspection gets its own ROIs
        instead.
        """

        return self.solder_cropper.extract(image, detections, output_dir)

    def classify_components(
        self, crops: Sequence[ComponentCrop]
    ) -> list[ComponentClassification]:
        """Step 6.1: classify component families or return no fabricated result."""

        if self.classifier is None:
            return []
        return self.classifier.classify(crops)

    def run(
        self,
        image: ImageSource,
        reference: ImageSource | None = None,
        source_name: str | None = None,
        crop_dir: str | Path | None = None,
        solder_crop_dir: str | Path | None = None,
    ) -> PipelineRun:
        """Execute steps 0-6.1 and retain all artifacts for the local UI/export."""

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
        solder_crops = self.make_solder_crops(aligned.image, detections, solder_crop_dir)
        classifications = self.classify_components(crops)

        warnings = list(preprocessed.warnings)
        if not aligned.success:
            warnings.append(aligned.message)
        if board.method == "full_image_fallback":
            warnings.append("PCB contour was not found; the complete image was used as the board ROI.")
        if isinstance(self.detector, CVComponentDetector):
            warnings.append(
                "Step 4 is using OpenCV candidate proposals; labels are not component classifications."
            )
        if self.classifier is None:
            warnings.append(
                "Step 6.1 was not run because best.onnx and model_manifest.json were not configured."
            )
        if self.config.solder.enabled and detections and not solder_crops:
            warnings.append(
                "Step 5.5 derived no solder ROI; check that detections are larger "
                "than the minimum ROI size."
            )

        return PipelineRun(
            source_name=inferred_name,
            input_image=input_image,
            preprocess_result=preprocessed,
            alignment_result=aligned,
            board_region=board,
            detections=detections,
            crops=crops,
            solder_crops=solder_crops,
            classifications=classifications,
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
