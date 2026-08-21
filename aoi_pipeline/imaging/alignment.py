"""Step 2: reference alignment using ORB/homography with an ECC fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import AlignmentConfig
from ..exceptions import AlignmentError, RecipeValidationError
from .image_io import drop_singleton_channel, ensure_bgr
from ..models import AlignmentResult, shape_dict
from ..golden.recipe import AlignmentAnchor, InspectionRecipe, validate_recipe_assets


@dataclass(frozen=True, slots=True)
class AnchorMatch:
    """One fixed-anchor measurement in canonical Golden coordinates."""

    anchor_id: str
    reference_point_px: tuple[float, float]
    observed_point_px: tuple[float, float] | None
    score: float | None
    status: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "reference_point_px": [float(value) for value in self.reference_point_px],
            "observed_point_px": (
                None
                if self.observed_point_px is None
                else [float(value) for value in self.observed_point_px]
            ),
            "score": None if self.score is None else float(self.score),
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(slots=True)
class StrictAlignmentResult:
    """Fail-closed source-to-Golden partial-affine alignment result.

    ``transform`` maps undistorted measurement-image pixels into
    ``golden_board_pixels``. Invalid results deliberately expose neither an
    aligned image nor a transform, so inspection cannot continue by accident.
    ``rotation_deg`` is ``atan2(m10, m00)`` of that source-to-Golden matrix in
    image coordinates (X right, Y down); positive therefore denotes the
    OpenCV/image-coordinate correction direction, not a Y-up Cartesian angle.
    """

    status: str
    image: np.ndarray | None
    transform: np.ndarray | None
    residual_px: float | None
    matched_anchors: int
    inliers: int
    inlier_ratio: float
    scale: float | None
    rotation_deg: float | None
    canvas_overlap_ratio: float | None
    valid_mask: np.ndarray | None
    reason: str
    anchor_matches: list[AnchorMatch] = field(default_factory=list)
    coordinate_space: str = "golden_board_pixels"

    @property
    def success(self) -> bool:
        return self.status == "valid"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "output_shape": shape_dict(self.image),
            "transform": None if self.transform is None else self.transform.tolist(),
            "residual_px": None if self.residual_px is None else float(self.residual_px),
            "matched_anchors": int(self.matched_anchors),
            "inliers": int(self.inliers),
            "inlier_ratio": float(self.inlier_ratio),
            "scale": None if self.scale is None else float(self.scale),
            "rotation_deg": None if self.rotation_deg is None else float(self.rotation_deg),
            "canvas_overlap_ratio": (
                None
                if self.canvas_overlap_ratio is None
                else float(self.canvas_overlap_ratio)
            ),
            "valid_pixel_ratio": (
                None
                if self.valid_mask is None
                else float(np.count_nonzero(self.valid_mask) / self.valid_mask.size)
            ),
            "reason": self.reason,
            "anchor_matches": [item.to_dict() for item in self.anchor_matches],
            "coordinate_space": self.coordinate_space,
        }


class PCBAligner:
    def __init__(self, config: AlignmentConfig | None = None) -> None:
        self.config = config or AlignmentConfig()

    def align(self, image: np.ndarray, reference: np.ndarray | None = None) -> AlignmentResult:
        source = ensure_bgr(image)
        if reference is None:
            return AlignmentResult(
                image=source.copy(),
                method="not_requested",
                success=True,
                homography=np.eye(3, dtype=np.float64),
                message="No reference image was supplied; alignment was skipped.",
            )

        target = ensure_bgr(reference)
        if not self.config.enabled:
            resized = _resize_to_reference(source, target)
            return AlignmentResult(
                image=resized,
                method="disabled",
                success=True,
                homography=_resize_homography(source, target),
                message="Alignment is disabled; source was resized to the reference canvas.",
            )

        orb_result, reason = self._align_orb(source, target)
        if orb_result is not None:
            return orb_result

        if self.config.use_ecc_fallback:
            ecc_result, ecc_reason = self._align_ecc(source, target)
            if ecc_result is not None:
                ecc_result.message = f"ORB unavailable ({reason}); ECC fallback succeeded."
                return ecc_result
            reason = f"{reason}; ECC failed ({ecc_reason})"

        if self.config.strict:
            raise AlignmentError(f"Could not align image to reference: {reason}")

        return AlignmentResult(
            image=_resize_to_reference(source, target),
            method="resize_fallback",
            success=False,
            homography=_resize_homography(source, target),
            message=f"Feature alignment failed; image was only resized. Reason: {reason}",
        )

    def align_to_recipe(
        self,
        image: np.ndarray,
        recipe: InspectionRecipe,
        recipe_root: str | Path,
        *,
        source_valid_mask: np.ndarray | None = None,
    ) -> StrictAlignmentResult:
        """Align a measurement image using only recipe-defined stable anchors.

        This production API never delegates to ORB/ECC or ``resize_fallback``.
        Expected alignment failures return ``status='invalid'`` and stop before
        warp. The returned transform, when valid, maps source pixels to the
        recipe's canonical Golden canvas.
        """

        source = ensure_bgr(image)
        if source_valid_mask is None:
            source_valid = np.full(source.shape[:2], 255, dtype=np.uint8)
        elif (
            not isinstance(source_valid_mask, np.ndarray)
            or source_valid_mask.ndim != 2
            or source_valid_mask.shape != source.shape[:2]
        ):
            return _invalid_strict(
                "Source valid mask must be a 2D array matching the measurement image"
            )
        else:
            source_valid = ((source_valid_mask > 0).view(np.uint8) * 255)
        gates = recipe.alignment.quality_gates
        anchors = recipe.alignment.anchors
        if len(anchors) < int(gates.min_anchors):
            return _invalid_strict(
                f"Recipe provides only {len(anchors)} anchors; at least {gates.min_anchors} are required"
            )
        try:
            validate_recipe_assets(recipe, recipe_root)
        except RecipeValidationError as exc:
            return _invalid_strict(f"Recipe asset validation failed: {exc}")

        root = Path(recipe_root).expanduser().resolve()
        matches = [
            _measure_anchor(source, anchor, root, float(gates.min_anchor_score))
            for anchor in anchors
        ]
        accepted = [item for item in matches if item.status == "matched"]
        if len(accepted) < int(gates.min_anchors):
            return _invalid_strict(
                f"Only {len(accepted)} anchors passed the score gate; {gates.min_anchors} are required",
                matches=matches,
                matched=len(accepted),
            )

        source_points = np.float32(
            [item.observed_point_px for item in accepted]
        ).reshape(-1, 1, 2)
        reference_points = np.float32(
            [item.reference_point_px for item in accepted]
        ).reshape(-1, 1, 2)
        matrix, inlier_mask = cv2.estimateAffinePartial2D(
            source_points,
            reference_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(gates.ransac_reprojection_threshold_px),
            maxIters=2000,
            confidence=0.99,
            refineIters=10,
        )
        if matrix is None or inlier_mask is None or matrix.shape != (2, 3):
            return _invalid_strict(
                "OpenCV could not estimate a partial-affine anchor transform",
                matches=matches,
                matched=len(accepted),
            )
        matrix = np.asarray(matrix, dtype=np.float64)
        if not np.all(np.isfinite(matrix)):
            return _invalid_strict(
                "Anchor transform contains non-finite values",
                matches=matches,
                matched=len(accepted),
            )

        inlier_flags = np.asarray(inlier_mask).reshape(-1).astype(bool)
        inliers = int(inlier_flags.sum())
        inlier_ratio = inliers / max(1, len(accepted))
        scale = float(math.hypot(matrix[0, 0], matrix[1, 0]))
        rotation_deg = float(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])))
        if not gates.min_scale <= scale <= gates.max_scale:
            return _invalid_strict(
                f"Estimated scale {scale:.6f} is outside [{gates.min_scale}, {gates.max_scale}]",
                matches=matches,
                matched=len(accepted),
                inliers=inliers,
                inlier_ratio=inlier_ratio,
                scale=scale,
                rotation_deg=rotation_deg,
            )
        if abs(rotation_deg) > float(gates.max_abs_rotation_deg):
            return _invalid_strict(
                f"Estimated rotation {rotation_deg:.3f} deg exceeds gate {gates.max_abs_rotation_deg:.3f} deg",
                matches=matches,
                matched=len(accepted),
                inliers=inliers,
                inlier_ratio=inlier_ratio,
                scale=scale,
                rotation_deg=rotation_deg,
            )
        if inliers < int(gates.min_anchors):
            return _invalid_strict(
                f"Only {inliers} anchor inliers; {gates.min_anchors} are required",
                matches=matches,
                matched=len(accepted),
                inliers=inliers,
                inlier_ratio=inlier_ratio,
                scale=scale,
                rotation_deg=rotation_deg,
            )
        if inlier_ratio < float(gates.min_inlier_ratio):
            return _invalid_strict(
                f"Anchor inlier ratio {inlier_ratio:.3f} is below {gates.min_inlier_ratio:.3f}",
                matches=matches,
                matched=len(accepted),
                inliers=inliers,
                inlier_ratio=inlier_ratio,
                scale=scale,
                rotation_deg=rotation_deg,
            )

        predicted = cv2.transform(source_points, matrix).reshape(-1, 2)
        expected = reference_points.reshape(-1, 2)
        errors = np.linalg.norm(predicted - expected, axis=1)
        inlier_errors = errors[inlier_flags]
        residual = float(np.sqrt(np.mean(np.square(inlier_errors))))
        if not math.isfinite(residual) or residual > float(gates.max_residual_px):
            return _invalid_strict(
                f"Anchor RMS residual {residual:.4f}px exceeds {gates.max_residual_px:.4f}px",
                matches=matches,
                matched=len(accepted),
                inliers=inliers,
                inlier_ratio=inlier_ratio,
                residual=residual,
                scale=scale,
                rotation_deg=rotation_deg,
            )

        canvas_overlap_ratio = _affine_canvas_overlap_ratio(
            source.shape[:2],
            (recipe.image_size.height, recipe.image_size.width),
            matrix,
        )
        if (
            not math.isfinite(canvas_overlap_ratio)
            or canvas_overlap_ratio < float(gates.min_canvas_overlap_ratio)
        ):
            return _invalid_strict(
                "Transformed source canvas overlap "
                f"{canvas_overlap_ratio:.3f} is below "
                f"{gates.min_canvas_overlap_ratio:.3f}",
                matches=matches,
                matched=len(accepted),
                inliers=inliers,
                inlier_ratio=inlier_ratio,
                residual=residual,
                scale=scale,
                rotation_deg=rotation_deg,
                canvas_overlap_ratio=canvas_overlap_ratio,
            )

        transform = np.eye(3, dtype=np.float64)
        transform[:2, :] = matrix
        aligned = cv2.warpAffine(
            source,
            matrix,
            (recipe.image_size.width, recipe.image_size.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        valid_mask = cv2.warpAffine(
            source_valid,
            matrix,
            (recipe.image_size.width, recipe.image_size.height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return StrictAlignmentResult(
            status="valid",
            image=aligned,
            transform=transform,
            residual_px=residual,
            matched_anchors=len(accepted),
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            scale=scale,
            rotation_deg=rotation_deg,
            canvas_overlap_ratio=canvas_overlap_ratio,
            valid_mask=valid_mask,
            reason="",
            anchor_matches=matches,
        )

    def _align_orb(
        self, source: np.ndarray, target: np.ndarray
    ) -> tuple[AlignmentResult | None, str]:
        gray_source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        gray_target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=max(100, int(self.config.orb_features)))
        keypoints_source, descriptors_source = orb.detectAndCompute(gray_source, None)
        keypoints_target, descriptors_target = orb.detectAndCompute(gray_target, None)
        source_count = len(keypoints_source)
        target_count = len(keypoints_target)

        if descriptors_source is None or descriptors_target is None:
            return None, f"insufficient descriptors ({source_count}/{target_count} keypoints)"

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        pairs = matcher.knnMatch(descriptors_source, descriptors_target, k=2)
        ratio = float(np.clip(self.config.ratio_test, 0.1, 0.99))
        good_matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
        if len(good_matches) < max(4, int(self.config.min_good_matches)):
            return None, f"only {len(good_matches)} good ORB matches"

        source_points = np.float32(
            [keypoints_source[match.queryIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)
        target_points = np.float32(
            [keypoints_target[match.trainIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(
            source_points,
            target_points,
            cv2.RANSAC,
            float(self.config.ransac_reprojection_threshold),
        )
        if homography is None or mask is None:
            return None, "OpenCV could not estimate a homography"

        inliers = int(mask.ravel().sum())
        inlier_ratio = inliers / max(1, len(good_matches))
        if inliers < int(self.config.min_inliers):
            return None, f"only {inliers} homography inliers"
        if inlier_ratio < float(self.config.min_inlier_ratio):
            return None, f"low homography inlier ratio ({inlier_ratio:.3f})"
        if not _reasonable_homography(homography, source.shape[:2], target.shape[:2]):
            return None, "estimated homography is geometrically implausible"

        height, width = target.shape[:2]
        aligned = cv2.warpPerspective(
            source,
            homography,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return (
            AlignmentResult(
                image=aligned,
                method="orb_homography",
                success=True,
                homography=homography,
                source_keypoints=source_count,
                reference_keypoints=target_count,
                good_matches=len(good_matches),
                inliers=inliers,
                inlier_ratio=inlier_ratio,
                message="ORB feature matching and RANSAC homography succeeded.",
            ),
            "",
        )

    def _align_ecc(
        self, source: np.ndarray, target: np.ndarray
    ) -> tuple[AlignmentResult | None, str]:
        resized = _resize_to_reference(source, target)
        source_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        source_gray = cv2.GaussianBlur(source_gray, (5, 5), 0)
        target_gray = cv2.GaussianBlur(target_gray, (5, 5), 0)
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            max(1, int(self.config.ecc_iterations)),
            max(float(self.config.ecc_epsilon), 1e-9),
        )
        try:
            correlation, warp = cv2.findTransformECC(
                target_gray,
                source_gray,
                warp,
                cv2.MOTION_AFFINE,
                criteria,
                None,
                5,
            )
        except cv2.error as exc:
            return None, _short_cv_error(exc)

        minimum_correlation = float(np.clip(self.config.min_ecc_correlation, -1.0, 1.0))
        if not np.isfinite(correlation) or float(correlation) < minimum_correlation:
            return (
                None,
                f"low ECC correlation ({float(correlation):.3f} < {minimum_correlation:.3f})",
            )

        height, width = target.shape[:2]
        aligned = cv2.warpAffine(
            resized,
            warp,
            (width, height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
        )
        # findTransformECC returns the destination-to-source transform used with
        # WARP_INVERSE_MAP. Export the more conventional original-source to
        # reference transform, including the initial canvas resize.
        inverse_warp = np.eye(3, dtype=np.float64)
        inverse_warp[:2, :] = warp
        try:
            inverse_warp = np.linalg.inv(inverse_warp)
        except np.linalg.LinAlgError:
            return None, "ECC returned a singular affine transform"
        resize_transform = _resize_homography(source, target)
        source_to_reference = inverse_warp @ resize_transform
        return (
            AlignmentResult(
                image=aligned,
                method="ecc_affine",
                success=True,
                homography=source_to_reference,
                correlation=float(correlation),
                message="ECC affine fallback succeeded.",
            ),
            "",
        )


def _measure_anchor(
    source: np.ndarray,
    anchor: AlignmentAnchor,
    recipe_root: Path,
    minimum_score: float,
) -> AnchorMatch:
    # Not a bare cv2.imread: see drop_singleton_channel. The anchor mask is
    # compared against the template shape three lines down, and that comparison
    # is exactly what a trailing channel axis breaks.
    template = drop_singleton_channel(
        cv2.imread(str(recipe_root / anchor.template_path), cv2.IMREAD_GRAYSCALE)
    )
    if template is None or template.size == 0:
        return AnchorMatch(
            anchor.anchor_id,
            anchor.reference_point_px,
            None,
            None,
            "missing",
            "anchor template is unreadable",
        )
    mask: np.ndarray | None = None
    if anchor.mask_path is not None:
        mask = drop_singleton_channel(
            cv2.imread(str(recipe_root / anchor.mask_path), cv2.IMREAD_GRAYSCALE)
        )
        if mask is None or mask.shape != template.shape:
            return AnchorMatch(
                anchor.anchor_id,
                anchor.reference_point_px,
                None,
                None,
                "missing",
                "anchor mask is unreadable or has the wrong size",
            )

    height, width = source.shape[:2]
    search_bbox = anchor.search_roi_xyxy.clamp(width, height)
    search_x1, search_y1, search_x2, search_y2 = search_bbox.to_int()
    search = cv2.cvtColor(
        source[search_y1:search_y2, search_x1:search_x2], cv2.COLOR_BGR2GRAY
    )
    if (
        search.size == 0
        or search.shape[0] < template.shape[0]
        or search.shape[1] < template.shape[1]
    ):
        return AnchorMatch(
            anchor.anchor_id,
            anchor.reference_point_px,
            None,
            None,
            "missing",
            "anchor search ROI is empty or smaller than its template",
        )

    search_repr = _anchor_representation(search)
    template_repr = _anchor_representation(template)
    method = cv2.TM_CCOEFF_NORMED
    kwargs: dict[str, np.ndarray] = {}
    if mask is not None:
        method = cv2.TM_CCORR_NORMED
        kwargs["mask"] = mask
    try:
        response = cv2.matchTemplate(search_repr, template_repr, method, **kwargs)
    except cv2.error as exc:
        return AnchorMatch(
            anchor.anchor_id,
            anchor.reference_point_px,
            None,
            None,
            "missing",
            f"template matching failed: {_short_cv_error(exc)}",
        )
    response = np.nan_to_num(response, nan=-1.0, posinf=-1.0, neginf=-1.0)
    _, maximum, _, peak = cv2.minMaxLoc(response)
    score = float(maximum)
    if not math.isfinite(score) or score < minimum_score:
        return AnchorMatch(
            anchor.anchor_id,
            anchor.reference_point_px,
            None,
            score,
            "low_score",
            f"anchor score {score:.4f} is below {minimum_score:.4f}",
        )

    subpixel_x, subpixel_y = _subpixel_peak(response, peak)
    offset_x = anchor.reference_point_px[0] - anchor.template_bbox_xyxy.x1
    offset_y = anchor.reference_point_px[1] - anchor.template_bbox_xyxy.y1
    observed = (
        float(search_x1 + subpixel_x + offset_x),
        float(search_y1 + subpixel_y + offset_y),
    )
    return AnchorMatch(
        anchor.anchor_id,
        anchor.reference_point_px,
        observed,
        score,
        "matched",
    )


def _anchor_representation(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    gradient_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gradient_x, gradient_y)


def _subpixel_peak(response: np.ndarray, peak: tuple[int, int]) -> tuple[float, float]:
    x, y = peak
    offset_x = _quadratic_peak_offset(response[y, x - 1 : x + 2]) if 0 < x < response.shape[1] - 1 else 0.0
    offset_y = _quadratic_peak_offset(response[y - 1 : y + 2, x]) if 0 < y < response.shape[0] - 1 else 0.0
    return float(x + offset_x), float(y + offset_y)


def _quadratic_peak_offset(values: np.ndarray) -> float:
    left, center, right = (float(value) for value in np.asarray(values).reshape(3))
    denominator = left - 2.0 * center + right
    if not math.isfinite(denominator) or abs(denominator) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))


def _invalid_strict(
    reason: str,
    *,
    matches: list[AnchorMatch] | None = None,
    matched: int = 0,
    inliers: int = 0,
    inlier_ratio: float = 0.0,
    residual: float | None = None,
    scale: float | None = None,
    rotation_deg: float | None = None,
    canvas_overlap_ratio: float | None = None,
) -> StrictAlignmentResult:
    return StrictAlignmentResult(
        status="invalid",
        image=None,
        transform=None,
        residual_px=residual,
        matched_anchors=matched,
        inliers=inliers,
        inlier_ratio=inlier_ratio,
        scale=scale,
        rotation_deg=rotation_deg,
        canvas_overlap_ratio=canvas_overlap_ratio,
        valid_mask=None,
        reason=reason,
        anchor_matches=list(matches or []),
    )


def _affine_canvas_overlap_ratio(
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    matrix: np.ndarray,
) -> float:
    """Return coverage of the Golden canvas by transformed source corners."""

    source_height, source_width = source_shape
    target_height, target_width = target_shape
    source_corners = np.float32(
        [[0, 0], [source_width, 0], [source_width, source_height], [0, source_height]]
    ).reshape(-1, 1, 2)
    transformed = cv2.transform(source_corners, matrix).reshape(-1, 2)
    if not np.all(np.isfinite(transformed)):
        return 0.0
    target_corners = np.float32(
        [[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]]
    )
    try:
        intersection_area, _ = cv2.intersectConvexConvex(
            transformed.astype(np.float32), target_corners
        )
    except cv2.error:
        return 0.0
    target_area = float(target_width * target_height)
    return float(np.clip(float(intersection_area) / max(target_area, 1.0), 0.0, 1.0))


def _resize_to_reference(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    height, width = target.shape[:2]
    if source.shape[:2] == target.shape[:2]:
        return source.copy()
    return cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)


def _resize_homography(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_height, source_width = source.shape[:2]
    target_height, target_width = target.shape[:2]
    return np.array(
        [
            [target_width / source_width, 0.0, 0.0],
            [0.0, target_height / source_height, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _reasonable_homography(
    homography: np.ndarray,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> bool:
    if homography.shape != (3, 3) or not np.all(np.isfinite(homography)):
        return False
    source_height, source_width = source_shape
    target_height, target_width = target_shape
    corners = np.float32(
        [[0, 0], [source_width, 0], [source_width, source_height], [0, source_height]]
    ).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    if not np.all(np.isfinite(transformed)):
        return False
    area = abs(float(cv2.contourArea(transformed.astype(np.float32))))
    target_area = float(target_width * target_height)
    if not 0.1 * target_area <= area <= 4.0 * target_area:
        return False
    margin_x, margin_y = 2.0 * target_width, 2.0 * target_height
    return bool(
        np.all(transformed[:, 0] >= -margin_x)
        and np.all(transformed[:, 0] <= target_width + margin_x)
        and np.all(transformed[:, 1] >= -margin_y)
        and np.all(transformed[:, 1] <= target_height + margin_y)
    )


def _short_cv_error(exc: cv2.error) -> str:
    message = str(exc).replace("\r", " ").replace("\n", " ")
    return " ".join(message.split())[:240]
