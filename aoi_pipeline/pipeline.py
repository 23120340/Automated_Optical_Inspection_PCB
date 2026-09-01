"""Facade that composes the AOI PCB steps 0 through 6.1."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from collections.abc import Mapping
from typing import Sequence

import cv2
import numpy as np

from .imaging.alignment import PCBAligner
from .imaging.board import PCBLocalizer
from .solder.cad import BoardCad, CadError, CadRegistration, load_cad, register_cad, register_from_fiducials
from .classification.family import ComponentClassifier, create_classifier
from .classification.package import (
    PackageClassification,
    PackageClassifier,
    create_package_classifier,
)
from .config import ModelDetectorConfig, PipelineConfig
from .detection.cropping import ComponentCropper
from .detection.detectors import (
    ComponentDetector,
    CVComponentDetector,
    UltralyticsDetector,
    create_detector,
)
from .exceptions import DetectorConfigurationError
from .grading.inspector import SolderInspector
from .solder.cad_fusion import FusionResult, fuse_solder_joints
from .solder.lead_detection import detect_leads_in_components
from .solder.leads import fuse_detected_leads, split_lead_detections
from .reporting.exporters import export_json as write_json
from .reporting.exporters import export_zip as write_zip
from .imaging.image_io import ImageSource, load_image
from .models import (
    AlignmentResult,
    BoardRegion,
    ComponentClassification,
    ComponentCrop,
    Detection,
    PipelineRun,
    PreprocessResult,
    SolderJoint,
    SolderJointCrop,
    SolderVerdict,
    utc_now_iso,
)
from .imaging.preprocessing import ImagePreprocessor
from .solder.geometry import SolderJointCropper, deconflict_joint_rois
from .solder.package_validation import (
    PackageTopologyCheck,
    assess_package_topology,
)
from .placement.footprints import profile_for_package_class
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
        lead_detector: object | None = None,
        package_classifier: PackageClassifier | None = None,
        package_model_path: str | Path | None = None,
        package_manifest_path: str | Path | Mapping[str, object] | None = None,
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
        self.localizer = PCBLocalizer(self.config.board, self.config.fiducials)
        self.detector = detector or create_detector(
            model_path,
            mode=self.config.detector_mode,
            cv_config=self.config.cv_detector,
            model_config=self.config.model_detector,
        )
        self.last_detection_metrics: dict[str, object] = {}
        self.cropper = ComponentCropper(self.config.crop)
        self.solder_cropper = SolderJointCropper(self.config.solder)
        self.solder_inspector = SolderInspector(self.config.solder_grading)
        self.cad: BoardCad | None = None
        self.cad_registration: CadRegistration | None = None
        self.cad_warnings: list[str] = []
        self.last_fusion: FusionResult = FusionResult()
        self.last_lead_fusion = None
        self.last_package_classifications: list[PackageClassification] = []
        self.last_package_topology_checks: list[PackageTopologyCheck] = []
        self.last_package_detections: list[Detection] = []
        # Pass 2 stays absent until someone names a model for it -- injected
        # here, or configured under ``lead_detection.model_path`` so the UI and
        # the CLI can reach it too. An injected detector wins; the two are never
        # combined, for the same reason as every other role above.
        lead_config = self.config.lead_detection
        if lead_detector is not None and lead_config.model_path:
            raise DetectorConfigurationError(
                "Pass either lead_detector or lead_detection.model_path, not both"
            )
        if lead_detector is not None:
            self.lead_detector = lead_detector
        elif lead_config.enabled and lead_config.model_path:
            self.lead_detector = create_detector(
                lead_config.model_path,
                model_config=ModelDetectorConfig(confidence=lead_config.confidence),
            )
        else:
            self.lead_detector = None
        self.last_pass2_leads: list[Detection] = []
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
        if package_classifier is not None and (
            package_model_path is not None or package_manifest_path is not None
        ):
            raise DetectorConfigurationError(
                "Pass either package_classifier or package artifact paths, not both"
            )
        self.package_classifier = package_classifier or create_package_classifier(
            package_model_path,
            package_manifest_path,
            self.config.package_classification,
        )

    @staticmethod
    def load_image(source: ImageSource) -> np.ndarray:
        """Step 0: decode an imported image without accessing a camera."""

        return load_image(source)

    def preprocess(self, image: np.ndarray) -> PreprocessResult:
        """Step 1: optionally undistort, then resize and enhance an image.

        The returned result carries both the enhanced analysis frame and its
        un-enhanced, geometrically identical radiometric frame.  No frame is
        stored on the pipeline instance, so preprocessing a Golden Image cannot
        replace the board pixels that a later stage will grade.
        """

        return self.preprocessor.process(image)

    def align(
        self,
        image: np.ndarray,
        reference: np.ndarray | None = None,
        *,
        radiometric_image: np.ndarray | None = None,
    ) -> AlignmentResult:
        """Step 2: align analysis and radiometric frames as one bundle.

        ``radiometric_image`` must already share the analysis coordinate space.
        It is warped with the exact source-to-reference homography returned by
        the aligner.  When that transform cannot be applied safely, the
        auxiliary frame is set to ``None`` instead of leaving stale pixels.
        """

        result = self.aligner.align(image, reference)
        result.radiometric_image = _warp_auxiliary_frame(
            radiometric_image,
            source_shape=image.shape,
            alignment=result,
        )
        return result

    def detect_board(
        self,
        image: np.ndarray,
        fiducials: Sequence[tuple[float, float]] | None = None,
    ) -> BoardRegion:
        """Step 3: locate the PCB region.

        With three or more ``fiducials`` the region comes from them instead of
        from contours. Contour finding looks for the largest rectangle-ish
        blob, which fails on the cases that matter: a background the same
        colour as the board, a partly covered board, or two boards in frame.
        """

        return self.localizer.locate(image, fiducials)

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
        # Thang px/mm chỉ có ở đây. Bước 5.5 cần nó để áp trần độ sâu ROI theo
        # mm; không có CAD thì trần không áp được, và đó là hành vi đúng --
        # đoán một thang còn tệ hơn giữ nguyên luật tỉ lệ.
        scale = float(getattr(registration, "scale_px_per_mm", 0.0) or 0.0)
        if scale > 0.0:
            self.config.solder.px_per_mm = scale
            # ``FusionConfig`` mang một ``SolderJointConfig`` RIÊNG.
            # ``from_mapping`` nối hai cái lại, nhưng ``PipelineConfig()`` dựng
            # trực tiếp thì không -- và đường CAD đọc đúng cái bên trong
            # ``fusion``. Bỏ dòng này thì trần mm chỉ có tác dụng khi không nạp
            # CAD, tức đúng lúc cần nó nhất thì lại không có.
            self.config.fusion.solder.px_per_mm = scale
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
        *,
        component_crops: Sequence[ComponentCrop] | None = None,
        package_classifications: Sequence[PackageClassification] | None = None,
    ) -> list[SolderJointCrop]:
        """Step 5.5: derive solder-joint ROIs and cut them out.

        Kept separate from :meth:`make_crops` on purpose. Step 6.1 was trained
        on body-tight crops, so widening those to reveal the fillet would shift
        the classifier's input distribution; joint inspection gets its own ROIs
        instead.
        """

        # Lead/pad detections describe the ROI directly, so they win where they
        # exist; everything else keeps the derived geometry. Split them out
        # first so a pad box is never also treated as a component to derive
        # terminals around.
        bodies, leads = split_lead_detections(detections)
        if package_classifications is None:
            if (
                self.package_classifier is None
                or not self.config.package_classification.enabled
            ):
                # Absolute no-op: with no explicitly configured model, do not
                # even create an otherwise-unused second set of body crops.
                package_classifications = []
            else:
                package_crops = (
                    list(component_crops)
                    if component_crops is not None
                    else self.make_crops(image, bodies)
                )
                body_ids = {item.detection_id for item in bodies}
                package_classifications = self.classify_packages(
                    [crop for crop in package_crops if crop.detection_id in body_ids]
                )
        self.last_package_classifications = list(package_classifications)
        bodies = self.apply_package_classifications(bodies, package_classifications)
        self.last_package_detections = [*bodies, *leads]
        # Đăng ký CAD trước khi suy hình học, không phải sau: phép đăng ký là
        # nơi duy nhất biết px/mm, mà trần độ sâu ROI cần con số đó ngay ở bước
        # suy. Gọi lại là rẻ -- kết quả được nhớ trong ``cad_registration``.
        if self.cad is not None and self.cad_registration is None:
            self.register_cad_to_image(image, bodies or detections, board_region)
        # Pass 2: look for the leads inside each component box. Returns nothing
        # when no detector is configured, which leaves the derived geometry
        # below exactly as it was.
        pass2 = detect_leads_in_components(
            image, bodies or detections, self.lead_detector, self.config.lead_detection
        )
        self.last_pass2_leads = pass2
        leads = [*leads, *pass2]
        derived = self.solder_cropper.derive(image, bodies or detections)
        lead_result = fuse_detected_leads(
            bodies or detections, leads, derived, self.config.lead_fusion
        )
        self.last_lead_fusion = lead_result
        fusion = self.fuse_solder_rois(
            image, bodies or detections, lead_result.joints, board_region
        )
        self.last_fusion = fusion
        # Again after fusion, not only inside ``derive``: lead detection and CAD
        # re-anchoring both rewrite ROIs, and either can push one component's
        # ROI back onto its neighbour. This is the last point where every joint
        # exists together, so it is the only place the check is complete.
        joints = deconflict_joint_rois(
            fusion.joints, bodies or detections, self.config.solder
        )
        self.last_package_topology_checks = assess_package_topology(
            bodies or detections,
            joints,
        )
        return self.solder_cropper.extract_joints(image, joints, output_dir)

    def grade_solder(
        self,
        crops: Sequence[SolderJointCrop],
        image: np.ndarray | None = None,
        *,
        radiometric_image: np.ndarray | None = None,
    ) -> list[SolderVerdict]:
        """Step 6.2: measure every solder ROI and call it.

        Runs on rules alone until a trained model is configured, so the stage
        produces verdicts from the first board rather than waiting for a
        training cycle to finish.
        """

        crops, image = self._radiometric_crops(
            crops,
            image,
            radiometric_image=radiometric_image,
        )
        return self.solder_inspector.inspect(crops, image)

    def _radiometric_crops(
        self,
        crops: Sequence[SolderJointCrop],
        image: np.ndarray | None,
        *,
        radiometric_image: np.ndarray | None = None,
    ) -> tuple[Sequence[SolderJointCrop], np.ndarray | None]:
        """Re-cut the ROIs from the un-enhanced frame, when there is one.

        Geometry stays where it was: the ROI boxes were derived on the enhanced
        image *on purpose* -- measured on the project board, that is where the
        band filters keep 0 silkscreen ROIs instead of 3. Only the pixels handed
        to the rule layer change, because that is the stage the clipping hurts.
        """

        source = radiometric_image
        if not self.config.solder_grading.prefer_radiometric_image or source is None:
            return crops, image
        if image is not None and image.shape[:2] != source.shape[:2]:
            return crops, image
        height, width = source.shape[:2]
        recut: list[SolderJointCrop] = []
        for crop in crops:
            box = getattr(crop, "joint", None)
            bbox = getattr(box, "bbox", None)
            if bbox is None:
                return crops, image
            x1, y1, x2, y2 = bbox.clamp(width, height).to_int()
            patch = source[y1:y2, x1:x2]
            if patch.size == 0:
                recut.append(crop)
                continue
            recut.append(replace(crop, image=patch))
        return recut, source

    def classify_components(
        self, crops: Sequence[ComponentCrop]
    ) -> list[ComponentClassification]:
        """Step 6.1: classify component families or return no fabricated result."""

        if self.classifier is None:
            return []
        return self.classifier.classify(crops)

    def classify_packages(
        self, crops: Sequence[ComponentCrop]
    ) -> list[PackageClassification]:
        """Step 5.2: classify visible package topology when explicitly loaded."""

        if (
            self.package_classifier is None
            or not self.config.package_classification.enabled
        ):
            return []
        return self.package_classifier.classify(crops)

    def apply_package_classifications(
        self,
        detections: Sequence[Detection],
        classifications: Sequence[PackageClassification],
    ) -> list[Detection]:
        """Attach accepted step-5.2 evidence without overriding stronger data.

        Every prediction remains visible in ``package_prediction``.  Only an
        ``accept`` decision may alter solder geometry, and even then an
        existing package profile (for example BOM/PnP/CAD footprint evidence)
        wins.  Review/unknown predictions therefore cannot manufacture ROIs.
        """

        by_detection = {item.detection_id: item for item in classifications}
        annotated: list[Detection] = []
        for detection in detections:
            result = by_detection.get(detection.detection_id)
            if result is None:
                annotated.append(detection)
                continue
            metadata = dict(detection.metadata)
            metadata["package_prediction"] = result.to_dict()
            may_apply = (
                result.decision == "accept"
                and self.config.package_classification.apply_to_solder_geometry
                and not isinstance(metadata.get("package_profile"), Mapping)
            )
            if may_apply:
                profile = profile_for_package_class(
                    result.package_class,
                    source=result.source,
                )
                payload = profile.to_dict()
                payload.update(
                    {
                        "source": result.source,
                        "probability": float(result.probability),
                        "decision": result.decision,
                        "model_version": result.model_version,
                    }
                )
                metadata["package_profile"] = payload
                metadata["terminal_geometry_override"] = profile.terminal_geometry
            annotated.append(replace(detection, metadata=metadata))
        return annotated

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
        aligned = self.align(
            preprocessed.image,
            reference_image,
            radiometric_image=preprocessed.radiometric_image,
        )
        board = self.detect_board(aligned.image)
        detections = self.detect_components(aligned.image, board)
        crops = self.make_crops(aligned.image, detections, crop_dir)
        body_ids = {
            item.detection_id for item in split_lead_detections(detections)[0]
        }
        package_classifications = self.classify_packages(
            [crop for crop in crops if crop.detection_id in body_ids]
        )
        detections = self.apply_package_classifications(
            detections,
            package_classifications,
        )
        solder_crops = self.make_solder_crops(
            aligned.image,
            detections,
            solder_crop_dir,
            board_region=board,
            component_crops=crops,
            package_classifications=package_classifications,
        )
        package_detection_by_id = {
            item.detection_id: item for item in self.last_package_detections
        }
        detections = [
            package_detection_by_id.get(item.detection_id, item) for item in detections
        ]
        fusion = self.last_fusion
        package_topology_checks = self.last_package_topology_checks
        classifications = self.classify_components(crops)
        solder_verdicts = self.grade_solder(
            solder_crops,
            aligned.image,
            radiometric_image=aligned.radiometric_image,
        )

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
        warnings.extend(self.solder_inspector.warnings)
        package_review_count = sum(
            item.decision != "accept" for item in package_classifications
        )
        if package_review_count:
            warnings.append(
                f"Step 5.2 sent {package_review_count} package prediction(s) "
                "to review; their predictions did not alter solder geometry."
            )
        topology_review_count = sum(
            item.status == "review" for item in package_topology_checks
        )
        if topology_review_count:
            warnings.append(
                f"Step 5.5 found {topology_review_count} package/ROI topology "
                "mismatch(es) requiring review."
            )
        hidden_count = sum(
            item.status == "not_inspectable" for item in package_topology_checks
        )
        if hidden_count:
            warnings.append(
                f"Step 5.5 marked {hidden_count} hidden-terminal package(s) as "
                "not inspectable by top-down 2D solder ROIs."
            )
        lead_result = self.last_lead_fusion
        if lead_result is not None:
            warnings.extend(lead_result.warnings)
            if lead_result.used_detected:
                warnings.append(
                    f"Bước 5.5: dùng {lead_result.used_detected} ROI từ detection "
                    f"chân/pad thật và {lead_result.used_derived} ROI suy ra."
                )
        if self.config.solder_grading.enabled and solder_crops and not self.solder_inspector.has_model:
            warnings.append(
                "Bước 6.2 đang chấm bằng luật đo hình học; nạp model ONNX + "
                "model_manifest.json để bật thêm tầng phân loại."
            )
        if (
            self.config.solder.enabled
            and detections
            and not solder_crops
            and not hidden_count
        ):
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
            package_classifications=package_classifications,
            package_topology_checks=package_topology_checks,
            solder_crops=solder_crops,
            fusion=fusion,
            solder_verdicts=solder_verdicts,
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


def _warp_auxiliary_frame(
    frame: np.ndarray | None,
    *,
    source_shape: Sequence[int],
    alignment: AlignmentResult,
) -> np.ndarray | None:
    """Apply an alignment result to an auxiliary frame, or fail closed.

    ``AlignmentResult.homography`` is source-to-reference for every legacy
    method, including identity, disabled-resize and resize fallback.  Reusing
    it is the only way to keep the radiometric pixels on the same canvas as the
    analysis image without estimating a second, potentially different warp.
    """

    if frame is None or frame.ndim not in (2, 3):
        return None
    if tuple(frame.shape[:2]) != tuple(source_shape[:2]):
        return None
    matrix = alignment.homography
    if matrix is None or alignment.image is None:
        return None
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return None
    try:
        condition = float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        return None
    # OpenCV accepts singular homographies and quietly returns a constant
    # border image.  Treat singular or numerically unusable transforms as a
    # failed auxiliary warp; grading the fabricated pixels would be worse than
    # falling back to the aligned analysis frame.
    if not np.isfinite(condition) or condition > 1.0e12:
        return None
    target_height, target_width = alignment.image.shape[:2]
    if target_height <= 0 or target_width <= 0:
        return None
    try:
        warped = cv2.warpPerspective(
            frame,
            matrix,
            (int(target_width), int(target_height)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
    except cv2.error:
        return None
    if warped.shape[:2] != (target_height, target_width):
        return None
    return np.ascontiguousarray(warped)
