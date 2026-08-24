from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from aoi_pipeline.golden.enrollment import (
    ReferenceSelectionConfig,
    ReferenceSelectionError,
    select_reference,
)
from aoi_pipeline.models import AlignmentResult


def _write_png(path: Path, image: np.ndarray) -> Path:
    assert cv2.imwrite(str(path), image)
    return path


def _textured_image(height: int = 180, width: int = 240) -> np.ndarray:
    rng = np.random.default_rng(20260824)
    image = rng.integers(35, 220, size=(height, width, 3), dtype=np.uint8)
    cv2.rectangle(image, (18, 20), (width - 18, height - 20), (25, 135, 55), 4)
    cv2.circle(image, (width // 3, height // 2), 24, (230, 225, 35), 3)
    cv2.putText(
        image,
        "PCB-A17",
        (35, height - 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return image


def _identity_alignment(source: np.ndarray, reference: np.ndarray) -> AlignmentResult:
    assert source.shape == reference.shape
    return AlignmentResult(
        image=source.copy(),
        method="identity_test",
        success=True,
        homography=np.eye(3, dtype=np.float64),
        inlier_ratio=1.0,
        correlation=1.0,
    )


def test_selects_real_source_medoid_deterministically(tmp_path: Path) -> None:
    base = _textured_image()
    darker = cv2.subtract(base, np.full_like(base, 18))
    brighter = cv2.add(base, np.full_like(base, 18))
    paths = [
        _write_png(tmp_path / "capture_dark.png", darker),
        _write_png(tmp_path / "capture_mid.png", base),
        _write_png(tmp_path / "capture_bright.png", brighter),
    ]

    first = select_reference(paths, align=_identity_alignment)
    reversed_order = select_reference(list(reversed(paths)), align=_identity_alignment)

    assert first.reference_path == paths[1]
    assert reversed_order.reference_path == paths[1]
    assert first.report.selected_basename == "capture_mid.png"
    assert first.report.selected_sha256 == hashlib.sha256(paths[1].read_bytes()).hexdigest()
    assert first.report.to_json() == reversed_order.report.to_json()

    # Enrollment must select an acquisition verbatim, never a mean/stacked image.
    input_payloads = {path.read_bytes() for path in paths}
    input_hashes = {hashlib.sha256(payload).hexdigest() for payload in input_payloads}
    assert first.reference_path.read_bytes() in input_payloads
    assert first.report.selected_sha256 in input_hashes

    serialized = first.report.to_json()
    assert str(tmp_path) not in serialized
    assert "reference_path" not in serialized
    assert {item.basename for item in first.report.candidates} == {
        "capture_dark.png",
        "capture_mid.png",
        "capture_bright.png",
    }
    assert all(len(item.sha256) == 64 for item in first.report.candidates)
    selected_report = next(item for item in first.report.candidates if item.selected)
    assert selected_report.medoid_score == pytest.approx(
        min(
            item.medoid_score
            for item in first.report.candidates
            if item.medoid_score is not None
        )
    )


def test_quality_metrics_reject_blur_clipping_and_exposure_outliers(
    tmp_path: Path,
) -> None:
    base = _textured_image()
    good_images = [
        cv2.subtract(base, np.full_like(base, 3)),
        base,
        cv2.add(base, np.full_like(base, 3)),
    ]
    paths = [
        _write_png(tmp_path / f"good_{index}.png", image)
        for index, image in enumerate(good_images)
    ]
    paths.extend(
        [
            _write_png(
                tmp_path / "blurred.png",
                cv2.GaussianBlur(base, (31, 31), sigmaX=9.0),
            ),
            _write_png(tmp_path / "underexposed.png", np.zeros_like(base)),
            _write_png(tmp_path / "overexposed.png", np.full_like(base, 255)),
        ]
    )
    config = ReferenceSelectionConfig(
        min_blur_variance=10.0,
        min_relative_blur=0.35,
        max_clipped_ratio=0.90,
        min_mean_luma=0.05,
        max_mean_luma=0.95,
    )

    result = select_reference(paths, config=config, align=_identity_alignment)
    by_name = {candidate.basename: candidate for candidate in result.report.candidates}

    assert result.report.quality_candidate_count == 3
    assert "blur_below_batch_gate" in by_name["blurred.png"].rejection_reasons
    assert "clipping_ratio_above_gate" in by_name["underexposed.png"].rejection_reasons
    assert "exposure_below_gate" in by_name["underexposed.png"].rejection_reasons
    assert "clipping_ratio_above_gate" in by_name["overexposed.png"].rejection_reasons
    assert "exposure_above_gate" in by_name["overexposed.png"].rejection_reasons
    assert by_name["underexposed.png"].quality.dark_clipped_ratio == pytest.approx(1.0)
    assert by_name["overexposed.png"].quality.bright_clipped_ratio == pytest.approx(1.0)
    assert by_name["blurred.png"].quality.laplacian_variance < (
        by_name["good_1.png"].quality.laplacian_variance
    )
    assert not by_name["blurred.png"].eligible


def test_fails_closed_when_no_candidate_has_enough_aligned_peers(
    tmp_path: Path,
) -> None:
    base = _textured_image()
    paths = [
        _write_png(tmp_path / f"capture_{index}.png", np.roll(base, index, axis=1))
        for index in range(3)
    ]

    def failed_alignment(source: np.ndarray, reference: np.ndarray) -> AlignmentResult:
        return AlignmentResult(
            image=source.copy(),
            method="resize_fallback",
            success=False,
            homography=np.eye(3, dtype=np.float64),
        )

    with pytest.raises(ReferenceSelectionError) as caught:
        select_reference(paths, align=failed_alignment)

    report = caught.value.report
    assert report is not None
    assert report.selected_basename is None
    assert report.selected_sha256 is None
    assert report.failure_reason == "insufficient_alignment_consensus"
    assert all(
        any(
            reason.startswith("insufficient_aligned_peers:0/")
            for reason in item.rejection_reasons
        )
        for item in report.candidates
    )
    assert str(tmp_path) not in str(caught.value)
    assert str(tmp_path) not in report.to_json()


def test_mixed_image_sizes_fail_closed_with_path_free_report(tmp_path: Path) -> None:
    base = _textured_image()
    paths = [
        _write_png(tmp_path / "same_a.png", base),
        _write_png(tmp_path / "same_b.png", cv2.add(base, np.full_like(base, 2))),
        _write_png(tmp_path / "wrong_size.png", base[:-12, :]),
    ]

    with pytest.raises(ReferenceSelectionError) as caught:
        select_reference(paths, align=_identity_alignment)

    report = caught.value.report
    assert report is not None
    assert report.failure_reason == "image_size_mismatch"
    assert report.selected_basename is None
    by_name = {candidate.basename: candidate for candidate in report.candidates}
    assert any(
        reason.startswith("image_size_mismatch:")
        for reason in by_name["wrong_size.png"].rejection_reasons
    )
    assert str(tmp_path) not in report.to_json()
    assert str(tmp_path) not in str(caught.value)


def test_default_pcb_aligner_provides_pairwise_diagnostics(tmp_path: Path) -> None:
    base = _textured_image(height=240, width=320)
    transforms = [
        np.float32([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.float32([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0]]),
        np.float32([[1.0, 0.0, -2.0], [0.0, 1.0, 3.0]]),
    ]
    paths = [
        _write_png(
            tmp_path / f"aligned_capture_{index}.png",
            cv2.warpAffine(
                base,
                matrix,
                (base.shape[1], base.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            ),
        )
        for index, matrix in enumerate(transforms)
    ]

    result = select_reference(paths)

    assert result.reference_path in paths
    assert result.report.required_peers == 2
    assert all(candidate.aligned_peer_count == 2 for candidate in result.report.candidates)
    assert all(candidate.alignment_method_counts for candidate in result.report.candidates)
    assert all(
        (candidate.median_overlap_ratio or 0.0) >= 0.80
        for candidate in result.report.candidates
    )
