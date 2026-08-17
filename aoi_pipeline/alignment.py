"""Step 2: reference alignment using ORB/homography with an ECC fallback."""

from __future__ import annotations

import cv2
import numpy as np

from .config import AlignmentConfig
from .exceptions import AlignmentError
from .image_io import ensure_bgr
from .models import AlignmentResult


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
