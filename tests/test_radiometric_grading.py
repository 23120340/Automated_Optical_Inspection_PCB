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
from aoi_pipeline.detection.detectors import MockComponentDetector
from aoi_pipeline.models import AlignmentResult, BoundingBox, Detection
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

    assert result.radiometric_image is not None
    assert result.radiometric_image.shape == result.image.shape
    # Đúng là ảnh CHƯA tăng cường, không phải bản sao của ảnh đã xử lý.
    assert not np.array_equal(result.radiometric_image, result.image)
    payload = result.to_dict()
    assert payload["radiometric_shape"] == {
        "height": result.image.shape[0],
        "width": result.image.shape[1],
        "channels": result.image.shape[2],
    }
    assert "radiometric_image" not in payload


def test_a_geometric_preprocess_step_is_applied_to_both_frames() -> None:
    """Radiometric is captured *after* resize, so its coordinates stay exact."""

    pipeline = AOIPipeline(
        PipelineConfig.from_mapping({"preprocess": {"max_side": 200}})
    )
    result = pipeline.preprocess(_board(width=400, height=300))

    assert result.image.shape[1] <= 200
    assert result.radiometric_image is not None
    assert result.radiometric_image.shape == result.image.shape


def test_grading_reads_the_unenhanced_pixels() -> None:
    pipeline = AOIPipeline(PipelineConfig())
    board = _board()
    preprocessed = pipeline.preprocess(board)
    analysis = preprocessed.image
    detection = Detection("resistor", 0.9, BoundingBox(150, 120, 250, 180))
    crops = pipeline.make_solder_crops(analysis, [detection])
    assert crops

    seen: list[np.ndarray] = []
    original = pipeline.solder_inspector.inspect

    def spy(items, image=None):
        seen.extend(item.image for item in items if item.image is not None)
        return original(items, image)

    pipeline.solder_inspector.inspect = spy  # type: ignore[method-assign]
    pipeline.grade_solder(
        crops,
        analysis,
        radiometric_image=preprocessed.radiometric_image,
    )

    assert seen, "phải có ROI được chấm"
    box = crops[0].joint.bbox.to_int()
    from_source = preprocessed.radiometric_image[box[1]:box[3], box[0]:box[2]]
    assert np.array_equal(seen[0], from_source)


def test_turning_it_off_restores_the_old_pixels() -> None:
    config = PipelineConfig()
    config.solder_grading.prefer_radiometric_image = False
    pipeline = AOIPipeline(config)
    board = _board()
    preprocessed = pipeline.preprocess(board)
    analysis = preprocessed.image
    detection = Detection("resistor", 0.9, BoundingBox(150, 120, 250, 180))
    crops = pipeline.make_solder_crops(analysis, [detection])

    before = [crop.image for crop in crops]
    kept, image = pipeline._radiometric_crops(
        crops,
        analysis,
        radiometric_image=preprocessed.radiometric_image,
    )
    assert [crop.image for crop in kept] == before
    assert image is analysis


def test_the_roi_boxes_do_not_move() -> None:
    """Chỉ pixel đưa vào luật đổi. Hình học giữ nguyên, vì đo được là ảnh tăng
    cường mới cho khoanh ROI đúng hơn."""

    pipeline = AOIPipeline(PipelineConfig())
    board = _board()
    preprocessed = pipeline.preprocess(board)
    analysis = preprocessed.image
    detection = Detection("resistor", 0.9, BoundingBox(150, 120, 250, 180))
    crops = pipeline.make_solder_crops(analysis, [detection])

    boxes = [crop.joint.bbox.to_int() for crop in crops]
    recut, _ = pipeline._radiometric_crops(
        crops,
        analysis,
        radiometric_image=preprocessed.radiometric_image,
    )
    assert [crop.joint.bbox.to_int() for crop in recut] == boxes


def test_preprocessing_a_golden_image_cannot_replace_the_board_pixels() -> None:
    """Regression for the staged UI and ``run`` using one pipeline instance."""

    board = _board()
    golden = np.full_like(board, (7, 31, 91))
    detection = Detection("resistor", 0.9, BoundingBox(150, 120, 250, 180))
    pipeline = AOIPipeline(
        PipelineConfig(),
        detector=MockComponentDetector([detection]),
    )
    seen: dict[str, np.ndarray | None] = {}

    def spy(items, image=None):
        seen["image"] = image
        seen["crop"] = items[0].image if items else None
        return []

    pipeline.solder_inspector.inspect = spy  # type: ignore[method-assign]
    pipeline.run(board, reference=golden, source_name="board.png")

    assert np.array_equal(seen["image"], board)
    assert not np.array_equal(seen["image"], golden)


def test_run_warps_the_radiometric_frame_with_the_golden_alignment() -> None:
    """A shifted Golden must move the measurement pixels with the ROI canvas."""

    board = _board()
    golden = np.roll(board, shift=(11, 23), axis=(0, 1))
    transform = np.array(
        [[1.0, 0.0, 23.0], [0.0, 1.0, 11.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    expected = cv2.warpPerspective(
        board,
        transform,
        (board.shape[1], board.shape[0]),
    )
    detection = Detection("resistor", 0.9, BoundingBox(173, 131, 273, 191))
    pipeline = AOIPipeline(
        PipelineConfig(),
        detector=MockComponentDetector([detection]),
    )

    def aligned(image, reference):
        assert reference is not None
        return AlignmentResult(
            image=cv2.warpPerspective(
                image,
                transform,
                (reference.shape[1], reference.shape[0]),
            ),
            method="known_translation",
            success=True,
            homography=transform,
        )

    pipeline.aligner.align = aligned  # type: ignore[method-assign]
    seen: dict[str, np.ndarray | None] = {}

    def spy(items, image=None):
        seen["image"] = image
        return []

    pipeline.solder_inspector.inspect = spy  # type: ignore[method-assign]
    run = pipeline.run(board, reference=golden, source_name="shifted.png")

    assert run.alignment_result.radiometric_image is not None
    assert np.array_equal(run.alignment_result.radiometric_image, expected)
    assert np.array_equal(seen["image"], expected)


def test_missing_alignment_transform_drops_radiometric_pixels() -> None:
    pipeline = AOIPipeline(PipelineConfig())
    preprocessed = pipeline.preprocess(_board())
    pipeline.aligner.align = lambda image, reference: AlignmentResult(  # type: ignore[method-assign]
        image=image.copy(),
        method="broken_backend",
        success=False,
        homography=None,
    )

    result = pipeline.align(
        preprocessed.image,
        np.zeros_like(preprocessed.image),
        radiometric_image=preprocessed.radiometric_image,
    )

    assert result.radiometric_image is None


def test_singular_alignment_transform_drops_radiometric_pixels() -> None:
    """OpenCV otherwise accepts H=zeros and silently returns a black frame."""

    pipeline = AOIPipeline(PipelineConfig())
    preprocessed = pipeline.preprocess(_board())
    pipeline.aligner.align = lambda image, reference: AlignmentResult(  # type: ignore[method-assign]
        image=image.copy(),
        method="singular_backend",
        success=False,
        homography=np.zeros((3, 3), dtype=np.float64),
    )

    result = pipeline.align(
        preprocessed.image,
        np.zeros_like(preprocessed.image),
        radiometric_image=preprocessed.radiometric_image,
    )

    assert result.radiometric_image is None


def test_a_wrong_radiometric_frame_cannot_be_caught_by_looking_at_it() -> None:
    """Vì sao không có phép kiểm nội dung nào canh chỗ này.

    Sau khi hai lỗi im lặng đầu tiên (nạp golden ghi đè, và ``align`` không warp
    theo) đều là "cùng kích thước, khác nội dung", phản xạ tự nhiên là thêm một
    vân tay nội dung để chặn cặp ảnh không khớp. **Đã thử và đo: không được.**

    Tương quan chuẩn hoá giữa khung phân tích và khung radiometric, lấy ở bốn
    độ phân giải:

        cặp                            16×16   64×64  128×128  256×256
        đúng cặp, ảnh thật             0,985   0,986    0,985    0,984
        NHẦM: radiometric của golden   0,985   0,986    0,985    0,984
        đúng cặp, ảnh dựng trong test  0,683   0,588    0,581    0,573

    Cặp nhầm ghi điểm **bằng đúng** cặp đúng, còn fixture đúng lại ghi *thấp
    hơn* cặp nhầm — không ngưỡng nào tách được.

    Lý do không phải là chọn sai thước đo, mà là bản chất bài toán: ``align``
    warp ảnh board **về hệ toạ độ của golden**, nên radiometric của golden thật
    sự cùng hệ toạ độ. Sai ở đây là *nhầm board*, không phải *lệch toạ độ*. Mà
    golden thì trông giống hệt board đang kiểm — đó chính là công dụng của nó.

    Nên chỗ này được canh bằng **cấu trúc**, không phải bằng phép kiểm: hai
    khung ra khỏi ``preprocess``/``align`` trong cùng một kết quả, và người gọi
    lấy cả hai từ cùng một chỗ. Kiểm ngược lại chính điều đó.
    """

    pipeline = AOIPipeline(PipelineConfig())
    board = _board()
    result = pipeline.preprocess(board)

    # Hợp đồng thật: hai khung đi cùng nhau, không ai phải tự ghép.
    assert result.radiometric_image is not None
    assert result.radiometric_image.shape == result.image.shape

    # Và không còn khung nào nằm trên pipeline để một lần preprocess khác ghi đè.
    assert not hasattr(pipeline, "radiometric_image") or (
        getattr(pipeline, "radiometric_image", None) is None
    )
