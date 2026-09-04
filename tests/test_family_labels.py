"""Nhãn họ 6.1 thay nhãn detector khi detector chỉ khoanh thân.

Detector thân linh kiện mới có MỘT lớp ``component``, mà bước 5.5 đọc hình học
chân từ ``terminal_geometry(detection.label)`` — và ``component`` rơi vào nhánh
mặc định ``multi_pin``, tức dựng dải quanh cả 4 cạnh của cả linh kiện 2 chân.

Test cuối file là cổng thật: nó đo trên 28 pad đếm tay của một bo thật, giữ
nguyên box và chỉ đổi nhãn, nên nó tách riêng đúng tác động của hợp đồng nhãn.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest

from aoi_pipeline import BoundingBox, Detection, SolderJointConfig, derive_solder_joints
from aoi_pipeline.config import PipelineConfig, PreprocessConfig, terminal_geometry
from aoi_pipeline.imaging.preprocessing import ImagePreprocessor
from aoi_pipeline.models import ClassProbability, ComponentClassification
from aoi_pipeline.pipeline import AOIPipeline
from aoi_pipeline.solder.geometry import deconflict_joint_rois

DATA_DIR = Path(__file__).resolve().parent / "data" / "solder_geometry"
MIN_PAD_COVERAGE = 0.50


def _pipeline() -> AOIPipeline:
    return AOIPipeline(PipelineConfig())


def _detection(label: str, detection_id: str = "det_1") -> Detection:
    return Detection(
        label=label,
        confidence=0.9,
        bbox=BoundingBox(10.0, 10.0, 60.0, 30.0),
        detection_id=detection_id,
    )


def _family(
    family: str, decision: str = "accept", detection_id: str = "det_1"
) -> ComponentClassification:
    return ComponentClassification(
        crop_id="crop_0001",
        detection_id=detection_id,
        family=family,
        probability=0.9,
        top_k=[ClassProbability(family, 0.9)],
        unknown_score=0.1,
        decision=decision,
        model_version="test",
    )


def test_a_one_class_detector_gets_its_label_from_the_classifier() -> None:
    out = _pipeline().apply_family_labels(
        [_detection("component")], [_family("capacitor")]
    )
    assert out[0].label == "capacitor"
    assert terminal_geometry(out[0].label) == "two_terminal"
    # Nguồn nhãn phải truy được: báo cáo hiển thị nhãn này, và người đọc cần
    # biết nó đến từ 6.1 chứ không từ detector.
    assert out[0].metadata["detector_label"] == "component"
    assert out[0].metadata["label_source"] == "family_classifier"


def test_a_detector_that_does_know_its_classes_is_left_alone() -> None:
    """Detector 22 lớp đang chạy được train riêng cho việc phân lớp. Đảo ưu
    tiên sang 6.1 là đổi hành vi của đường đang chạy mà không ai yêu cầu."""

    out = _pipeline().apply_family_labels(
        [_detection("resistor")], [_family("ic")]
    )
    assert out[0].label == "resistor"
    assert "label_source" not in out[0].metadata


@pytest.mark.parametrize("decision", ["review", "unknown"])
def test_an_unsure_family_does_not_replace_anything(decision: str) -> None:
    """``multi_pin`` là mặc định AN TOÀN: dựng thừa ROI thì xem lại được, thiếu
    thì không. Một họ đoán sai có thể dựng 2 ROI cho một con IC 16 chân."""

    out = _pipeline().apply_family_labels(
        [_detection("component")], [_family("capacitor", decision=decision)]
    )
    assert out[0].label == "component"


def test_background_is_never_promoted_to_a_label() -> None:
    out = _pipeline().apply_family_labels(
        [_detection("component")], [_family("false_crop_background")]
    )
    assert out[0].label == "component"


def test_no_classifier_means_no_change() -> None:
    detections = [_detection("component")]
    assert _pipeline().apply_family_labels(detections, None) == detections
    assert _pipeline().apply_family_labels(detections, []) == detections


def test_only_the_matching_detection_is_relabelled() -> None:
    out = _pipeline().apply_family_labels(
        [_detection("component", "a"), _detection("component", "b")],
        [_family("resistor", detection_id="a")],
    )
    assert [item.label for item in out] == ["resistor", "component"]


# --------------------------------------------------------------- cổng thật

def _coverage(roi: BoundingBox, pad: list[int]) -> float:
    px1, py1, px2, py2 = pad
    ix = max(0.0, min(roi.x2, px2) - max(roi.x1, px1))
    iy = max(0.0, min(roi.y2, py2) - max(roi.y1, py1))
    area = float((px2 - px1) * (py2 - py1))
    return (ix * iy) / area if area > 0 else 0.0


def _covered_pads(relabel: bool) -> tuple[int, int, int]:
    truth = json.loads(
        (DATA_DIR / "board_smd_00001.json").read_text(encoding="utf-8")
    )
    raw = cv2.imread(str(DATA_DIR / truth["image"]))
    assert raw is not None, f"thiếu asset {truth['image']}"
    image = ImagePreprocessor(PreprocessConfig()).process(raw).image
    height, width = image.shape[:2]

    # Mọi nhãn thành ``component``: đúng thứ detector thân linh kiện trả về.
    detections = [
        Detection("component", d["confidence"], BoundingBox(*d["box"]))
        for d in truth["detections"]
    ]
    if relabel:
        families = [
            _family(raw_row["label"], detection_id=det.detection_id)
            for det, raw_row in zip(detections, truth["detections"])
        ]
        detections = _pipeline().apply_family_labels(detections, families)

    config = SolderJointConfig()
    index_of = {d.detection_id: i for i, d in enumerate(detections)}
    joints: list = []
    for detection in detections:
        joints.extend(
            derive_solder_joints(detection, width, height, config=config, image=image)
        )
    per: dict[int, list] = {i: [] for i in range(len(detections))}
    for joint in deconflict_joint_rois(joints, detections, config):
        if joint.kind == "joint":
            per[index_of[joint.detection_id]].append(joint)

    total = covered = 0
    for entry in truth["components"].values():
        rois = [j.bbox for j in per[entry["detection_index"]]]
        for pad in entry["pads"]:
            total += 1
            if max((_coverage(r, pad) for r in rois), default=0.0) >= MIN_PAD_COVERAGE:
                covered += 1
    return covered, total, sum(len(v) for v in per.values())


def test_the_classifier_label_restores_every_hand_measured_pad() -> None:
    """Cổng promote của detector một lớp.

    Không vá thì 21/28 pad, và số ROI *tăng* 47% — nó dựng dải quanh cả 4 cạnh
    của linh kiện 2 chân rồi trượt khỏi land thật. Vá xong phải về đúng con số
    của đường nhãn thật: 28/28 pad, 90 ROI.
    """

    without, total, roi_without = _covered_pads(relabel=False)
    with_fix, _, roi_with = _covered_pads(relabel=True)

    assert without < total, (
        "nhãn một lớp lẽ ra phải làm mất pad; nếu test này đỏ thì hoặc 5.5 đã "
        "thôi đọc detection.label, hoặc fixture đã đổi"
    )
    assert with_fix == total, (
        f"vá xong vẫn chỉ phủ {with_fix}/{total} pad (trước khi vá {without})"
    )
    assert roi_with < roi_without, (
        f"vá xong vẫn dựng {roi_with} ROI, không ít hơn {roi_without} — dấu hiệu "
        "vẫn đang đi nhánh multi_pin cho linh kiện 2 chân"
    )


def test_unresolved_generic_labels_are_counted_for_the_report() -> None:
    """Nạp detector một lớp mà quên nạp 6.1 là hỏng im lặng: ROI vẫn dựng ra,
    chỉ là sai nhánh. Đếm để ``run()`` nói thẳng con số."""

    pipeline = _pipeline()
    pipeline.apply_family_labels([_detection("component")], None)
    assert pipeline.last_unresolved_generic_labels == 1

    pipeline.apply_family_labels([_detection("component")], [_family("capacitor")])
    assert pipeline.last_unresolved_generic_labels == 0

    pipeline.apply_family_labels(
        [_detection("component")], [_family("capacitor", decision="review")]
    )
    assert pipeline.last_unresolved_generic_labels == 1

    pipeline.apply_family_labels([_detection("resistor")], None)
    assert pipeline.last_unresolved_generic_labels == 0, (
        "detector 22 lớp không mang nhãn chung chung nên không có gì chưa giải"
    )


def test_tiles_are_cut_at_the_size_the_artifact_really_accepts() -> None:
    """Một ONNX khoá cứng shape chỉ nhận đúng một cỡ, nên tile lớn hơn bị
    letterbox thu nhỏ và linh kiện xuất hiện nhỏ hơn lúc train.

    Đo trên 640 box tay: tile 1280 với artifact native 1024 làm recall tổng tụt
    89,7% -> 83,3%, và dải >=250px tụt 92,7% -> 78,0%, trong khi precision không
    khá hơn để bù. Cỡ phải lấy từ artifact, không gán cứng — model 22 lớp cũ
    native 1280 nên nó phải giữ nguyên hành vi.

    Chạy qua ĐÚNG ``detect_components`` chứ không chép lại phép tính: một test
    chép logic sẽ xanh cả khi bản vá bị gỡ.
    """

    import numpy as np

    from aoi_pipeline.config import TilingConfig

    class _FixedSizeDetector:
        """Ghi lại cạnh của từng tile mà pipeline đưa xuống."""

        def __init__(self, size: int) -> None:
            self.native_image_size = size
            self.image_size = size
            self.seen: list[int] = []

        def detect(self, image, *, confidence=None):
            self.seen.append(max(image.shape[:2]))
            return []

    board = np.zeros((3072, 3072, 3), dtype=np.uint8)
    for native in (1024, 1280):
        pipeline = AOIPipeline(PipelineConfig(tiling=TilingConfig(tile_size=1280)))
        pipeline.detector = _FixedSizeDetector(native)
        pipeline.detect_components(board)
        assert pipeline.last_detection_metrics["effective_tile_size"] == native, (
            f"native {native} nhưng cắt tile "
            f"{pipeline.last_detection_metrics['effective_tile_size']}"
        )
        # include_full_image cho chạy thêm một lượt trên NGUYÊN ảnh; đó là
        # hành vi cố ý, không phải tile. Mọi lượt còn lại phải vừa artifact.
        tiles = [s for s in pipeline.detector.seen if s != max(board.shape[:2])]
        oversized = [s for s in tiles if s > native]
        assert not oversized, (
            f"native {native} mà vẫn có tile cạnh {sorted(set(oversized))} — "
            "chúng sẽ bị letterbox thu nhỏ"
        )
