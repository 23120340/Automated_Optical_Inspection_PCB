"""Chốt thứ tự các bước, vì ở đây thứ tự là hợp đồng chứ không phải tuỳ ý.

Hai ràng buộc, cả hai đều phục vụ kế hoạch phân package bằng LUẬT
(`docs/plans/ke_hoach_phan_nhom_package.md` §4, §5):

1. **6.1 chạy trước 5.2/5.5.** Luật phân package chia nhỏ gói *bên trong một
   họ* do 6.1 trả về (``capacitor`` -> tròn/vuông, ``ic`` -> chân 2 bên/4 bên/
   không chân), nên họ phải có trước.
2. **Pass 2 (lead detector) chạy trước bước phân package.** Luật đọc vị trí
   chân. Detector thành phần mới là một lớp ``component``, không sinh
   ``pads``/``pins``, nên pass 2 là nguồn chân duy nhất — chạy sau thì luật
   nhận đầu vào rỗng và im lặng trả ``unknown`` cho mọi IC.

Vì sao cần file này: trước khi có nó, **cả 1087 test đều xanh với thứ tự sai**.
Không có gì canh, nên một lần refactor đảo lại là không ai biết.
"""

from __future__ import annotations

import numpy as np

from aoi_pipeline.classification.package import PackageClassification
from aoi_pipeline.config import (
    BoardConfig,
    PackageClassificationConfig,
    PipelineConfig,
    PreprocessConfig,
    SolderJointConfig,
)
from aoi_pipeline.detection.detectors import MockComponentDetector
from aoi_pipeline.models import (
    BoundingBox,
    ClassProbability,
    ComponentClassification,
    Detection,
)
from aoi_pipeline.pipeline import AOIPipeline


class _SpyLeadDetector:
    """Pass 2: ghi lại thời điểm được gọi, trả một chân ở giữa crop."""

    def __init__(self, log: list[str]) -> None:
        self.log = log

    def detect(self, image):
        self.log.append("pass2")
        height, width = image.shape[:2]
        return [
            Detection(
                "pads",
                0.9,
                BoundingBox(width // 4, height // 4, width // 2, height // 2),
            )
        ]


class _SpyPackageClassifier:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def classify(self, crops):
        self.log.append("package")
        return [
            PackageClassification(
                crop_id=str(crop.metadata.get("crop_id") or crop.filename),
                detection_id=crop.detection_id,
                package_class="hai_chan",
                probability=0.95,
                top_k=[ClassProbability("hai_chan", 0.95)],
                unknown_score=0.05,
                decision="accept",
                model_version="order-test-v1",
                detector_hint=crop.label,
            )
            for crop in crops
        ]


class _SpyFamilyClassifier:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def classify(self, crops):
        self.log.append("family")
        return [
            ComponentClassification(
                crop_id=f"crop_{index + 1:04d}",
                detection_id=crop.detection_id,
                family="capacitor",
                probability=0.91,
                top_k=[ClassProbability("capacitor", 0.91)],
                unknown_score=0.09,
                decision="accept",
                model_version="order-test-v1",
                detector_hint=crop.label,
            )
            for index, crop in enumerate(crops)
        ]


def _solder_config(**package_overrides: object) -> PipelineConfig:
    return PipelineConfig(
        solder=SolderJointConfig(
            split_pins=False,
            refine_to_metal=False,
            include_body_view=False,
            deconflict_neighbours=False,
        ),
        package_classification=PackageClassificationConfig(**package_overrides),
    )


def _body() -> Detection:
    return Detection(
        label="capacitor",
        confidence=0.92,
        bbox=BoundingBox(40, 45, 140, 95),
        detection_id="det_body",
    )


def test_pass_two_runs_before_package_classification() -> None:
    log: list[str] = []
    pipeline = AOIPipeline(
        _solder_config(),
        lead_detector=_SpyLeadDetector(log),
        package_classifier=_SpyPackageClassifier(log),
    )

    pipeline.make_solder_crops(np.zeros((160, 200, 3), dtype=np.uint8), [_body()])

    assert "pass2" in log and "package" in log, log
    assert log.index("pass2") < log.index("package"), (
        "Luật phân package đọc vị trí chân, nên pass 2 phải chạy trước. "
        f"Thứ tự thực tế: {log}"
    )


def test_package_snapshot_excludes_pass_two_leads() -> None:
    """``last_package_detections`` là ảnh chụp của tập ĐẦU VÀO sau khi gắn nhãn
    package — ``run()`` dùng nó để ghi đè lại ``detections``. Chân do pass 2
    sinh ra là hộp mới, không nằm trong tập đầu vào, nên không được lọt vào."""

    pipeline = AOIPipeline(
        _solder_config(),
        lead_detector=_SpyLeadDetector([]),
        package_classifier=_SpyPackageClassifier([]),
    )

    pipeline.make_solder_crops(np.zeros((160, 200, 3), dtype=np.uint8), [_body()])

    assert pipeline.last_pass2_leads, "pass 2 phải thật sự tìm được chân"
    snapshot_ids = {item.detection_id for item in pipeline.last_package_detections}
    pass2_ids = {item.detection_id for item in pipeline.last_pass2_leads}
    assert snapshot_ids == {"det_body"}, snapshot_ids
    assert not (snapshot_ids & pass2_ids), "chân pass 2 lọt vào ảnh chụp package"


def test_family_classifier_runs_before_package_and_solder(pcb_image) -> None:
    log: list[str] = []
    pipeline = AOIPipeline(
        config=PipelineConfig(
            preprocess=PreprocessConfig(
                max_side=None,
                denoise=False,
                white_balance=False,
                clahe=False,
                normalize=False,
                sharpen=False,
            ),
            board=BoardConfig(min_area_ratio=0.999, max_area_ratio=0.9999),
            solder=SolderJointConfig(
                split_pins=False,
                refine_to_metal=False,
                include_body_view=False,
                deconflict_neighbours=False,
            ),
        ),
        detector=MockComponentDetector(
            [Detection("capacitor", 0.93, BoundingBox(15, 18, 42, 35))]
        ),
        classifier=_SpyFamilyClassifier(log),
        lead_detector=_SpyLeadDetector(log),
        package_classifier=_SpyPackageClassifier(log),
    )

    run = pipeline.run(pcb_image, source_name="order.png")

    assert len(run.classifications) == 1, "6.1 vẫn phải cho kết quả sau khi đổi chỗ"
    assert log.index("family") < log.index("package"), (
        "Luật phân package chia nhỏ gói bên trong một HỌ, nên 6.1 phải chạy "
        f"trước 5.2. Thứ tự thực tế: {log}"
    )
    assert log.index("family") < log.index("pass2"), (
        f"6.1 phải chạy trước cả bước 5.5. Thứ tự thực tế: {log}"
    )
