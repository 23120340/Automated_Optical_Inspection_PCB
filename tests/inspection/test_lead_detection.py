"""Pass 2: detecting leads inside a component crop, and getting them back onto
the board.

The whole idea rests on one thing being exactly right: a box found in crop
coordinates has to land on the same pixels of the board. A translation that is
off by the margin, or by a clamp, would put every ROI slightly wrong in a way
no aggregate metric would show.
"""

from __future__ import annotations

import numpy as np
import pytest

from aoi_pipeline import BoundingBox, Detection, LeadDetectionConfig, PipelineConfig
from aoi_pipeline.solder.lead_detection import (
    component_crop_window,
    detect_leads_in_components,
    to_board_coordinates,
)

BOARD = (400, 700)  # height, width


def _board() -> np.ndarray:
    return np.full((*BOARD, 3), (40, 90, 40), np.uint8)


def _component(x1=200, y1=150, x2=270, y2=190, label="resistor", ident="d0"):
    return Detection(label, 0.9, BoundingBox(x1, y1, x2, y2), detection_id=ident)


class _MarkerDetector:
    """Reports one box at a position it is told, in crop coordinates.

    Deliberately not a real model. The question under test is the coordinate
    algebra, and a stub is the only way to know the expected answer exactly.
    """

    def __init__(self, boxes, label="pads", confidence=0.9):
        self.boxes = boxes
        self.label = label
        self.confidence = confidence
        self.seen: list[tuple[int, int]] = []

    def detect(self, image):
        self.seen.append((image.shape[1], image.shape[0]))
        return [
            Detection(self.label, self.confidence, BoundingBox(*box))
            for box in self.boxes
        ]


# --------------------------------------------------------------------------- #
# The crop window
# --------------------------------------------------------------------------- #


def test_the_crop_reaches_past_the_body_onto_the_land() -> None:
    """A crop that stops at the component silhouette hides the fillet, which is
    the only thing pass 2 exists to find."""

    config = LeadDetectionConfig(crop_margin_ratio=0.35, crop_margin_min_px=0)
    window = component_crop_window(BoundingBox(200, 150, 270, 190), 700, 400, config)
    margin = round(0.35 * 70)
    assert window == (200 - margin, 150 - margin, 270 + margin, 190 + margin)


def test_a_tiny_part_still_gets_a_usable_margin() -> None:
    """35% of a 10 px box is 4 px. The floor keeps the crop from being all body."""

    config = LeadDetectionConfig(crop_margin_ratio=0.35, crop_margin_min_px=6)
    window = component_crop_window(BoundingBox(100, 100, 110, 108), 700, 400, config)
    assert window == (94, 94, 116, 114)


def test_the_crop_stops_at_the_frame_edge() -> None:
    config = LeadDetectionConfig()
    window = component_crop_window(BoundingBox(2, 2, 40, 30), 700, 400, config)
    assert window[0] == 0 and window[1] == 0
    assert window[2] <= 700 and window[3] <= 400


# --------------------------------------------------------------------------- #
# Crop coordinates back onto the board
# --------------------------------------------------------------------------- #


def test_a_box_in_the_crop_lands_on_the_same_pixels_of_the_board() -> None:
    """The identity the whole two-pass idea depends on."""

    window = (180, 130, 290, 210)
    local = Detection("pads", 0.9, BoundingBox(4, 6, 24, 26))
    board = to_board_coordinates(local, window, 700, 400)
    assert (board.x1, board.y1, board.x2, board.y2) == (184, 136, 204, 156)


def test_the_translation_is_exact_for_every_corner_of_the_crop() -> None:
    """Off-by-one here would shift every ROI on the board by a pixel."""

    window = (180, 130, 290, 210)
    width, height = window[2] - window[0], window[3] - window[1]
    for local_box in (
        (0, 0, 5, 5),
        (width - 5, 0, width, 5),
        (0, height - 5, 5, height),
        (width - 5, height - 5, width, height),
    ):
        board = to_board_coordinates(
            Detection("pads", 0.9, BoundingBox(*local_box)), window, 700, 400
        )
        assert board.x1 == local_box[0] + window[0]
        assert board.y1 == local_box[1] + window[1]
        assert board.x2 == local_box[2] + window[0]
        assert board.y2 == local_box[3] + window[1]


def test_the_pixels_under_the_translated_box_are_the_pixels_the_detector_saw() -> None:
    """Not algebra this time: cut the crop, cut the board at the translated
    coordinates, and require the two patches to be identical."""

    image = _board()
    rng = np.random.default_rng(3)
    image[:] = rng.integers(0, 255, image.shape, dtype=np.uint8)
    component = _component()
    config = LeadDetectionConfig()
    window = component_crop_window(component.bbox, 700, 400, config)

    crop = image[window[1]:window[3], window[0]:window[2]]
    local = Detection("pads", 0.9, BoundingBox(5, 7, 25, 22))
    board = to_board_coordinates(local, window, 700, 400)

    from_crop = crop[7:22, 5:25]
    from_board = image[int(board.y1):int(board.y2), int(board.x1):int(board.x2)]
    assert np.array_equal(from_crop, from_board)


# --------------------------------------------------------------------------- #
# Running the stage
# --------------------------------------------------------------------------- #


def test_no_detector_means_no_leads_and_no_work() -> None:
    """The stage has to be inert until a model exists, or it changes today's
    behaviour for no benefit."""

    assert detect_leads_in_components(_board(), [_component()], None) == []


def test_a_disabled_stage_does_nothing_even_with_a_detector() -> None:
    detector = _MarkerDetector([(4, 4, 20, 20)])
    found = detect_leads_in_components(
        _board(), [_component()], detector, LeadDetectionConfig(enabled=False)
    )
    assert found == []
    assert detector.seen == [], "tắt rồi thì không được gọi detector"


def test_every_lead_comes_back_in_board_coordinates() -> None:
    image = _board()
    component = _component(200, 150, 270, 190)
    config = LeadDetectionConfig(crop_margin_ratio=0.35, crop_margin_min_px=0)
    window = component_crop_window(component.bbox, 700, 400, config)

    detector = _MarkerDetector([(2, 3, 18, 19), (30, 3, 46, 19)])
    leads = detect_leads_in_components(image, [component], detector, config)

    assert len(leads) == 2
    assert detector.seen == [(window[2] - window[0], window[3] - window[1])]
    for lead, local in zip(leads, detector.boxes):
        assert lead.bbox.x1 == local[0] + window[0]
        assert lead.bbox.y1 == local[1] + window[1]
        assert lead.source == "lead_pass2"
        assert lead.metadata["parent_detection_id"] == component.detection_id


def test_a_low_confidence_lead_is_dropped() -> None:
    detector = _MarkerDetector([(4, 4, 20, 20)], confidence=0.10)
    config = LeadDetectionConfig(confidence=0.25)
    assert detect_leads_in_components(_board(), [_component()], detector, config) == []


def test_a_speck_is_not_a_lead() -> None:
    detector = _MarkerDetector([(4, 4, 6, 6)])
    config = LeadDetectionConfig(min_lead_px=5)
    assert detect_leads_in_components(_board(), [_component()], detector, config) == []


def test_a_component_too_small_to_read_is_skipped_not_guessed() -> None:
    detector = _MarkerDetector([(1, 1, 4, 4)])
    tiny = _component(10, 10, 16, 14)
    config = LeadDetectionConfig(min_crop_px=40, crop_margin_ratio=0.0,
                                 crop_margin_min_px=0)
    assert detect_leads_in_components(_board(), [tiny], detector, config) == []
    assert detector.seen == []


def test_one_unreadable_component_does_not_lose_the_others() -> None:
    """Pass 1 already produced usable boxes for the rest of the board; one
    failed forward pass must not throw them away."""

    class _FlakyDetector:
        def __init__(self):
            self.calls = 0

        def detect(self, image):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("mô phỏng inference hỏng")
            return [Detection("pads", 0.9, BoundingBox(4, 4, 20, 20))]

    detector = _FlakyDetector()
    components = [_component(ident="a"), _component(300, 150, 370, 190, ident="b")]
    leads = detect_leads_in_components(_board(), components, detector)

    assert detector.calls == 2
    assert len(leads) == 1
    assert leads[0].metadata["parent_detection_id"] == "b"


# --------------------------------------------------------------------------- #
# Wiring into the pipeline
# --------------------------------------------------------------------------- #


def _chip_board():
    import cv2

    image = _board()
    cv2.rectangle(image, (174, 156), (194, 184), (200, 200, 200), -1)
    cv2.rectangle(image, (276, 156), (296, 184), (200, 200, 200), -1)
    cv2.rectangle(image, (200, 150), (270, 190), (25, 25, 25), -1)
    return image, [_component()]


def test_the_pipeline_is_unchanged_when_pass_2_has_no_model() -> None:
    """The stage ships switched off. If its mere presence moved a single ROI,
    that would be a regression dressed up as a feature."""

    from aoi_pipeline import AOIPipeline

    image, detections = _chip_board()
    without = AOIPipeline(config=PipelineConfig()).make_solder_crops(image, detections)
    with_stage = AOIPipeline(
        config=PipelineConfig(), lead_detector=None
    ).make_solder_crops(image, detections)

    assert [c.joint.bbox.to_dict() for c in without] == [
        c.joint.bbox.to_dict() for c in with_stage
    ]


def test_detected_leads_reach_the_fusion_stage() -> None:
    """Pass 2 is only useful if what it finds actually replaces the derived
    geometry, per terminal, through the fusion that already exists."""

    from aoi_pipeline import AOIPipeline

    image, detections = _chip_board()
    config = PipelineConfig()
    window = component_crop_window(
        detections[0].bbox, BOARD[1], BOARD[0], config.lead_detection
    )
    # A lead exactly over the left land at 174..194 x 156..184 on the board.
    local = (174 - window[0], 156 - window[1], 194 - window[0], 184 - window[1])
    detector = _MarkerDetector([local])

    pipeline = AOIPipeline(config=config, lead_detector=detector)
    crops = pipeline.make_solder_crops(image, detections)

    assert pipeline.last_pass2_leads, "pass 2 phải chạy"
    sources = {crop.joint.source for crop in crops}
    assert sources != {"derived"}, f"chân đo được phải thắng hình học: {sources}"

    matched = [
        crop for crop in crops
        if crop.joint.kind == "joint" and crop.joint.source != "derived"
    ]
    assert matched, "ít nhất một ROI phải đến từ chân đo được"
    box = matched[0].joint.bbox
    assert abs(box.x1 - 174) <= 2 and abs(box.y1 - 156) <= 2


# --------------------------------------------------------------------------
# Gom lô. Đo được trên máy này, yolo11n @ imgsz 256, 64 crop:
#   từng cái 28.9 ms/crop · lô 4 19.6 ms · lô 16 28.6 ms · lô 64 33.2 ms
# Nên lô nhỏ có lợi thật, nhưng lô lớn thì hại. Cái đáng test không phải tốc
# độ mà là HỢP ĐỒNG VỊ TRÍ: kết quả thứ i phải thuộc về crop thứ i, vì lệch
# một bậc là gán chân của linh kiện này sang linh kiện khác mà không báo lỗi.
# --------------------------------------------------------------------------


class _BatchDetector:
    """Ghi lại kích thước từng lô, trả về một box mang dấu vết crop."""

    def __init__(self, *, batch_returns=None):
        self.batch_sizes: list[int] = []
        self.single_calls = 0
        self._batch_returns = batch_returns

    def _one(self, image):
        # Nhãn mang chiều rộng crop, để test truy được kết quả về đúng crop.
        return [Detection(label=f"w{image.shape[1]}", confidence=0.9,
                          bbox=BoundingBox(1.0, 1.0, 9.0, 9.0))]

    def detect(self, image):
        self.single_calls += 1
        return self._one(image)

    def detect_batch(self, images):
        self.batch_sizes.append(len(images))
        if self._batch_returns is not None:
            return self._batch_returns(images)
        return [self._one(image) for image in images]


def _board_with_components(count: int):
    """Một khung ảnh và ``count`` linh kiện, mỗi cái rộng khác nhau."""

    frame = np.zeros((200, 60 * count + 80, 3), dtype=np.uint8)
    detections = [
        Detection(
            label="resistor", confidence=0.9,
            bbox=BoundingBox(20.0 + 60 * index, 40.0,
                             20.0 + 60 * index + 24.0 + index, 80.0),
            detection_id=f"d{index}",
        )
        for index in range(count)
    ]
    return frame, detections


def test_crops_go_through_detect_batch_in_chunks_of_the_configured_size() -> None:
    frame, detections = _board_with_components(10)
    detector = _BatchDetector()

    detect_leads_in_components(frame, detections, detector,
                               LeadDetectionConfig(batch_size=4))

    assert detector.batch_sizes == [4, 4, 2]
    assert detector.single_calls == 0


def test_each_result_lands_on_the_component_its_crop_came_from() -> None:
    """Lệch một bậc ở đây là gán chân sai linh kiện, và không có gì báo lỗi."""

    frame, detections = _board_with_components(7)
    detector = _BatchDetector()

    leads = detect_leads_in_components(frame, detections, detector,
                                       LeadDetectionConfig(batch_size=3))

    assert len(leads) == 7
    for lead in leads:
        parent = lead.metadata["parent_detection_id"]
        window = lead.metadata["crop_window"]
        crop_width = window[2] - window[0]
        # Nhãn do detector đặt theo chiều rộng crop nó THỰC SỰ nhận được.
        assert lead.label == f"w{crop_width}", (
            f"{parent} nhận kết quả của crop rộng {lead.label[1:]}, "
            f"nhưng crop của nó rộng {crop_width}"
        )


def test_a_detector_without_detect_batch_still_works_one_crop_at_a_time() -> None:
    frame, detections = _board_with_components(5)

    class _SingleOnly:
        def __init__(self):
            self.calls = 0

        def detect(self, image):
            self.calls += 1
            return [Detection(label="pads", confidence=0.9,
                              bbox=BoundingBox(1.0, 1.0, 9.0, 9.0))]

    detector = _SingleOnly()
    leads = detect_leads_in_components(frame, detections, detector,
                                       LeadDetectionConfig(batch_size=4))

    assert detector.calls == 5
    assert len(leads) == 5


def test_a_batch_that_returns_the_wrong_count_falls_back_instead_of_guessing() -> None:
    """Trả về ít kết quả hơn số crop là phá hợp đồng vị trí. Ghép theo may
    rủi sẽ đặt chân lên nhầm linh kiện, nên phải quay về gọi từng cái."""

    frame, detections = _board_with_components(6)
    detector = _BatchDetector(batch_returns=lambda images: [[]])   # 1 thay vì 4

    leads = detect_leads_in_components(frame, detections, detector,
                                       LeadDetectionConfig(batch_size=4))

    assert detector.single_calls == 6, "phải quay về gọi từng crop"
    assert len(leads) == 6
    for lead in leads:
        window = lead.metadata["crop_window"]
        assert lead.label == f"w{window[2] - window[0]}"


def test_a_batch_that_raises_skips_only_that_batch() -> None:
    """Một lô hỏng không được làm hỏng cả board: lượt 1 đã cho box dùng được
    và bước 5.5 vẫn suy ra được hình học cho phần còn lại."""

    frame, detections = _board_with_components(6)

    class _RaisesAlways:
        def detect(self, image):
            raise RuntimeError("model chết")

        def detect_batch(self, images):
            raise RuntimeError("model chết")

    leads = detect_leads_in_components(frame, detections, _RaisesAlways(),
                                       LeadDetectionConfig(batch_size=4))
    assert leads == []
