from __future__ import annotations

import cv2
import numpy as np
import pytest

from aoi_pipeline import AlignmentConfig, ImagePreprocessor, PCBAligner, PreprocessConfig


def test_preprocessor_preserves_uint8_bgr_and_limits_size(pcb_image: np.ndarray) -> None:
    processor = ImagePreprocessor(PreprocessConfig(max_side=170, denoise=False))
    result = processor.process(pcb_image)
    assert result.image.dtype == np.uint8
    assert result.image.ndim == 3 and result.image.shape[2] == 3
    assert max(result.image.shape[:2]) == 170
    assert result.scale == 0.5
    assert any(operation.startswith("resize:") for operation in result.operations)
    assert "contrast:clahe" in result.operations


def test_orb_alignment_recovers_translated_textured_board(pcb_image: np.ndarray) -> None:
    translation = np.float32([[1, 0, 13], [0, 1, 9]])
    shifted = cv2.warpAffine(pcb_image, translation, (pcb_image.shape[1], pcb_image.shape[0]))
    aligner = PCBAligner(
        AlignmentConfig(min_good_matches=6, min_inliers=5, min_inlier_ratio=0.2)
    )
    result = aligner.align(shifted, pcb_image)
    assert result.success
    assert result.method in {"orb_homography", "ecc_affine"}
    assert result.image.shape == pcb_image.shape
    before = np.mean(np.abs(shifted.astype(np.float32) - pcb_image.astype(np.float32)))
    after = np.mean(np.abs(result.image.astype(np.float32) - pcb_image.astype(np.float32)))
    assert after < before


def test_alignment_uses_resize_fallback_for_featureless_images() -> None:
    source = np.zeros((80, 100, 3), dtype=np.uint8)
    target = np.zeros((60, 90, 3), dtype=np.uint8)
    result = PCBAligner(AlignmentConfig(use_ecc_fallback=True)).align(source, target)
    assert not result.success
    assert result.method == "resize_fallback"
    assert result.image.shape == target.shape


def test_alignment_without_reference_is_explicitly_skipped(pcb_image: np.ndarray) -> None:
    result = PCBAligner().align(pcb_image)
    assert result.success
    assert result.method == "not_requested"
    assert np.array_equal(result.image, pcb_image)


def test_ecc_rejects_a_converged_but_low_correlation_transform(
    monkeypatch: pytest.MonkeyPatch,
    pcb_image: np.ndarray,
) -> None:
    def low_quality_ecc(*args, **kwargs):
        return 0.31, np.eye(2, 3, dtype=np.float32)

    monkeypatch.setattr(cv2, "findTransformECC", low_quality_ecc)
    result = PCBAligner(
        AlignmentConfig(
            min_good_matches=99999,
            use_ecc_fallback=True,
            min_ecc_correlation=0.65,
        )
    ).align(pcb_image, pcb_image)
    assert not result.success
    assert result.method == "resize_fallback"
    assert "low ECC correlation (0.310 < 0.650)" in result.message
