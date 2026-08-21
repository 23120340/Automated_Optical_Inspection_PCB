"""Step 5: component crops for the 6.1 classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aoi_pipeline import BoundingBox, ComponentCropper, CropConfig, Detection


def test_cropper_pads_normalizes_and_optionally_writes(tmp_path: Path, pcb_image: np.ndarray) -> None:
    detection = Detection("IC / QFN", 0.95, BoundingBox(90, 80, 130, 105))
    cropper = ComponentCropper(CropConfig(target_size=(96, 64), square=True))
    crops = cropper.extract(pcb_image, [detection], tmp_path)
    assert len(crops) == 1
    assert crops[0].image.shape == (64, 96, 3)
    assert crops[0].crop_bbox.width >= detection.bbox.width
    assert crops[0].path is not None and crops[0].path.is_file()
    assert "IC_QFN" in crops[0].filename
    assert crops[0].to_dict()["path"] == crops[0].path.name


def test_cropper_sanitizes_detection_id_in_output_filename(
    tmp_path: Path, pcb_image: np.ndarray
) -> None:
    detection = Detection(
        "resistor",
        0.8,
        BoundingBox(20, 20, 40, 35),
        detection_id="../../outside",
    )
    crop = ComponentCropper(CropConfig(target_size=None)).extract(
        pcb_image, [detection], tmp_path
    )[0]
    assert "/" not in crop.filename and "\\" not in crop.filename
    assert crop.path is not None and crop.path.parent == tmp_path.resolve()
