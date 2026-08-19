"""Facade that composes the AOI PCB steps 0 through 6.1."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from collections.abc import Mapping
from typing import Sequence

import numpy as np

from .board.alignment import PCBAligner
from .board.localization import PCBLocalizer
from .inspection.cad import BoardCad, CadError, CadRegistration, load_cad, register_cad, register_from_fiducials
from .classification import ComponentClassifier, create_classifier
from .config import PipelineConfig
from .inspection.cropping import ComponentCropper
from .detection.detectors import (
    ComponentDetector,
    CVComponentDetector,
    UltralyticsDetector,
    create_detector,
)
from .core.exceptions import DetectorConfigurationError
from .inspection.fusion import FusionResult, fuse_solder_joints
from .export.exporters import export_json as write_json
from .export.exporters import export_zip as write_zip
from .core.image_io import ImageSource, load_image
from .core.models import (
    AlignmentResult,
    BoardRegion,
    ComponentClassification,
    ComponentCrop,
    Detection,
    PipelineRun,
    PreprocessResult,
    SolderJoint,
    SolderJointCrop,
    utc_now_iso,
)
from .imaging.preprocessing import ImagePreprocessor
from .inspection.solder import SolderJointCropper
from .detection.tiling import detect_with_adaptive_tiling


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
        cad: BoardCad | str | Path | None = None,
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
        self.cad: BoardCad | None = None
        self.cad_registration: CadRegistration | None = None
        self.cad_warnings: list[str] = []
        self.last_fusion: FusionResult = FusionResult()
        self._load_cad(cad)
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

    def _load_cad(self, cad: BoardCad | str | Path | None) -> None:
        """Bring in CAD data if any was given, and never fail the run over it.

        A missing or unreadable board file downgrades the pipeline to the
        detector-only path with a warning. Inspection without CAD is the
        supported baseline, so it must not become an error.
        """

        settings = self.config.cad
        if isinstance(cad, BoardCad):
            self.cad = cad
        else:
            source = cad if cad is not None else settings.path
            if source:
                try:
                    self.cad = load_cad(
                        source, fmt=settings.fmt, units=settings.units, side=settings.side
                    )
                except CadError as exc:
                    self.cad_warnings.append(f"CAD not loaded: {exc}")
                    self.cad = None
        if self.cad is None:
            return

        if settings.registration:
            try:
                self.cad_registration = CadRegistration.from_dict(settings.registration)
            except (CadError, KeyError, TypeError, ValueError) as exc:
                self.cad_warnings.append(f"CAD registration not applied: {exc}")
        elif settings.registration_path:
            try:
                payload = json.loads(
                    Path(settings.registration_path).expanduser().read_text(encoding="utf-8")
                )
                self.cad_registration = CadRegistration.from_dict(payload)
            except (OSError, CadError, KeyError, TypeError, ValueError) as exc:
                self.cad_warnings.append(f"CAD registration file not read: {exc}")
        elif settings.fiducials_mm and settings.fiducials_px:
            try:
                self.cad_registration = register_from_fiducials(
                    settings.fiducials_mm,
                    settings.fiducials_px,
                    perspective=settings.fiducial_perspective,
                )
            except CadError as exc:
                self.cad_warnings.append(f"CAD fiducials rejected: {exc}")

    def register_cad_to_image(
        self,
        image: np.ndarray,
        detections: Sequence[Detection],
        board_region: BoardRegion | None = None,
    ) -> CadRegistration | None:
        """Auto-align the loaded CAD board to this frame's detections.

        Only used when no registration was supplied. The result is cached on the
        instance so a fixed camera and fixture pay for it once.
        """

        if self.cad is None or not self.config.cad.auto_register:
            return self.cad_registration
        settings = self.config.cad
        height, width = image.shape[:2]
        registration = register_cad(
            self.cad,
            detections,
            (width, height),
            board_polygon=board_region.polygon if board_region is not None else None,
            min_matches=settings.auto_min_matches,
            match_tolerance_ratio=settings.auto_match_tolerance_ratio,
            refine_rounds=settings.auto_refine_rounds,
        )
        if registration is None:
            self.cad_warnings.append(
                "CAD auto-registration did not converge; run with fiducials or a "
                "saved registration matrix."
            )
            return None
        self.cad_registration = registration
        return registration

    def fuse_solder_rois(
        self,
        image: np.ndarray,
        detections: Sequence[Detection],
        derived_joints: Sequence[SolderJoint],
        board_region: BoardRegion | None = None,
    ) -> FusionResult:
        """Step 5.5b: combine CAD lands with the derived ROIs."""

        if self.cad is None:
            return FusionResult(joints=list(derived_joints), used_cad=False)
        height, width = image.shape[:2]
        registration = self.cad_registration or self.register_cad_to_image(
            image, detections, board_region
        )
        return fuse_solder_joints(
            detections,
            derived_joints,
            width,
            height,
            board=self.cad,
            registration=registration,
            config=self.config.fusion,
            image=image,
        )

    def make_solder_crops(
        self,
        image: np.ndarray,
        detections: Sequence[Detection],
        output_dir: str | Path | None = None,
        board_region: BoardRegion | None = None,
    ) -> list[SolderJointCrop]:
        """Step 5.5: derive solder-joint ROIs and cut them out.

        Kept separate from :meth:`make_crops` on purpose. Step 6.1 was trained
        on body-tight crops, so widening those to reveal the fillet would shift
        the classifier's input distribution; joint inspection gets its own ROIs
        instead.
        """

        derived = self.solder_cropper.derive(image, detections)
        fusion = self.fuse_solder_rois(image, detections, derived, board_region)
        self.last_fusion = fusion
        return self.solder_cropper.extract_joints(image, fusion.joints, output_dir)

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
        solder_crops = self.make_solder_crops(
            aligned.image, detections, solder_crop_dir, board_region=board
        )
        fusion = self.last_fusion
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
        warnings.extend(self.cad_warnings)
        warnings.extend(fusion.warnings)
        if self.cad is not None and not fusion.used_cad:
            warnings.append(
                "CAD was loaded but not applied; step 5.5 used detector-derived "
                "ROIs only."
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
            fusion=fusion,
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
