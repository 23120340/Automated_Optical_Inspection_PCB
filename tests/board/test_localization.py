"""Step 3: locating the PCB inside the frame."""

from __future__ import annotations

import numpy as np

from aoi_pipeline import BoardConfig, PCBLocalizer


def test_board_localizer_finds_synthetic_pcb(pcb_image: np.ndarray) -> None:
    region = PCBLocalizer().locate(pcb_image)
    assert region.method.startswith("contour:")
    assert region.confidence > 0.45
    assert region.bbox.x1 <= 45
    assert region.bbox.y1 <= 40
    assert region.bbox.x2 >= 295
    assert region.bbox.y2 >= 200
    assert region.mask is not None
    assert region.mask[120, 170] == 255
    assert region.mask[0, 0] == 0


def test_board_localizer_has_explicit_full_image_fallback() -> None:
    blank = np.full((50, 70, 3), 127, dtype=np.uint8)
    region = PCBLocalizer(BoardConfig()).locate(blank)
    assert region.method == "full_image_fallback"
    assert region.bbox.as_xyxy() == [0.0, 0.0, 70.0, 50.0]
