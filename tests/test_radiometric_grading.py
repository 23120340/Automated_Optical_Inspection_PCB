"""Bước 6.2 chấm điểm trên ảnh chưa tăng cường quang học.

Chuỗi CLAHE + normalize + sharpen giúp khâu *khoanh* ROI nhưng phá khâu *chấm*.
Đo trên board của dự án (`00001__1024__1648___4120.png`), 131 ROI:

    ảnh dùng để chấm     bridge   insufficient   phải xem tay
    đã tăng cường            12              1             16
    nguồn                     2              6             10

10 trong 12 cái ``bridge`` là gọi oan. Cơ chế đo được: trong đúng 10 ROI đó, tỉ
lệ pixel cháy sáng đi từ 35 % (nguồn) lên **71 %** (tăng cường) và độ phủ kim
loại từ 53 % lên 73 % — hai pad cạnh nhau nhoè vào nhau thành một khối, và luật
đọc ra là cầu chì.

Hướng ngược lại cũng đã đo và cũng thật: **khoanh ROI thì ảnh tăng cường tốt
hơn** — 0 so với 3 ROI rơi trên chữ lụa. Nên chỉ khâu chấm đổi ảnh.
"""

from __future__ import annotations

import cv2
import numpy as np

from aoi_pipeline.config import PipelineConfig
from aoi_pipeline.models import BoundingBox, Detection
from aoi_pipeline.pipeline import AOIPipeline


def _board(width: int = 400, height: int = 300) -> np.ndarray:
    rng = np.random.default_rng(7)
    image = np.zeros((height, width, 3), np.uint8)
    image[:, :] = (40, 80, 45)
    cv2.rectangle(image, (150, 120), (250, 180), (25, 25, 25), -1)
    for x in (140, 255):
        cv2.rectangle(image, (x, 130), (x + 12, 170), (205, 205, 205), -1)
    noise = rng.integers(-5, 5, image.shape, dtype=np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def test_preprocess_keeps_the_unenhanced_frame() -> None:
    pipeline = AOIPipeline(PipelineConfig.from_mapping({"preprocess": {"clahe": True}}))
    board = _board()
    result = pipeline.preprocess(board)

    assert pipeline.radiometric_image is not None
    assert pipeline.radiometric_image.shape == result.image.shape
    # Đúng là ảnh CHƯA tăng cường, không phải bản sao của ảnh đã xử lý.
    assert not np.array_equal(pipeline.radiometric_image, result.image)


def test_a_geometric_step_disables_it_rather_than_guessing() -> None:
    """Undistort hoặc resize làm hai ảnh lệch hệ toạ độ. Cắt lại sai chỗ còn tệ
    hơn cháy sáng, nên thà không dùng."""

    pipeline = AOIPipeline(
        PipelineConfig.from_mapping({"preprocess": {"max_side": 200}})
    )
    result = pipeline.preprocess(_board(width=400, height=300))

    assert result.image.shape[1] <= 200
    assert pipeline.radiometric_image is None


def test_grading_reads_the_unenhanced_pixels() -> None:
    pipeline = AOIPipeline(PipelineConfig())
    board = _board()
    analysis = pipeline.preprocess(board).image
    detection = Detection("resistor", 0.9, BoundingBox(150, 120, 250, 180))
    crops = pipeline.make_solder_crops(analysis, [detection])
    assert crops

    seen: list[np.ndarray] = []
    original = pipeline.solder_inspector.inspect

    def spy(items, image=None):
        seen.extend(item.image for item in items if item.image is not None)
        return original(items, image)

    pipeline.solder_inspector.inspect = spy  # type: ignore[method-assign]
    pipeline.grade_solder(crops, analysis)

    assert seen, "phải có ROI được chấm"
    box = crops[0].joint.bbox.to_int()
    from_source = pipeline.radiometric_image[box[1]:box[3], box[0]:box[2]]
    assert np.array_equal(seen[0], from_source)


def test_turning_it_off_restores_the_old_pixels() -> None:
    config = PipelineConfig()
    config.solder_grading.prefer_radiometric_image = False
    pipeline = AOIPipeline(config)
    board = _board()
    analysis = pipeline.preprocess(board).image
    detection = Detection("resistor", 0.9, BoundingBox(150, 120, 250, 180))
    crops = pipeline.make_solder_crops(analysis, [detection])

    before = [crop.image for crop in crops]
    kept, image = pipeline._radiometric_crops(crops, analysis)
    assert [crop.image for crop in kept] == before
    assert image is analysis


def test_the_roi_boxes_do_not_move() -> None:
    """Chỉ pixel đưa vào luật đổi. Hình học giữ nguyên, vì đo được là ảnh tăng
    cường mới cho khoanh ROI đúng hơn."""

    pipeline = AOIPipeline(PipelineConfig())
    board = _board()
    analysis = pipeline.preprocess(board).image
    detection = Detection("resistor", 0.9, BoundingBox(150, 120, 250, 180))
    crops = pipeline.make_solder_crops(analysis, [detection])

    boxes = [crop.joint.bbox.to_int() for crop in crops]
    recut, _ = pipeline._radiometric_crops(crops, analysis)
    assert [crop.joint.bbox.to_int() for crop in recut] == boxes
