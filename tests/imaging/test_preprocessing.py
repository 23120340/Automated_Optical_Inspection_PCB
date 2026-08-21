"""Step 1: image enhancement that must not change dtype or colour order."""

from __future__ import annotations

import numpy as np

from aoi_pipeline import ImagePreprocessor, PreprocessConfig


def test_preprocessor_preserves_uint8_bgr_and_limits_size(pcb_image: np.ndarray) -> None:
    processor = ImagePreprocessor(PreprocessConfig(max_side=170, denoise=False))
    result = processor.process(pcb_image)
    assert result.image.dtype == np.uint8
    assert result.image.ndim == 3 and result.image.shape[2] == 3
    assert max(result.image.shape[:2]) == 170
    assert result.scale == 0.5
    assert any(operation.startswith("resize:") for operation in result.operations)
    assert "contrast:clahe" in result.operations
