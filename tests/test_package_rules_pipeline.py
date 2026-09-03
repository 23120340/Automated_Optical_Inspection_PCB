"""Bộ luật 5.2 phải thật sự đến được ROI, và phải kêu khi nó không chạy.

`test_package_rules.py` kiểm chính cái luật. File này kiểm phần NỐI: luật có
nhận được họ 6.1 không, có bị model ghi đè không, và khi bật mà thiếu đầu vào
thì có im lặng không. Ca cuối là ca đáng sợ nhất — bật một tính năng rồi không
thấy gì xảy ra là loại lỗi khó truy nhất.
"""

from __future__ import annotations

import numpy as np

from aoi_pipeline.classification.package import PackageClassification
from aoi_pipeline.config import (
    PackageClassificationConfig,
    PackageRulesConfig,
    PipelineConfig,
    SolderJointConfig,
)
from aoi_pipeline.models import (
    BoundingBox,
    ClassProbability,
    ComponentClassification,
    Detection,
)
from aoi_pipeline.pipeline import AOIPipeline

IMAGE = np.zeros((160, 260, 3), dtype=np.uint8)
BODY_BOX = BoundingBox(60.0, 45.0, 160.0, 95.0)


def _config(**rule_overrides: object) -> PipelineConfig:
    return PipelineConfig(
        solder=SolderJointConfig(
            split_pins=False,
            refine_to_metal=False,
            include_body_view=False,
            deconflict_neighbours=False,
        ),
        package_classification=PackageClassificationConfig(enabled=False),
        package_rules=PackageRulesConfig(**rule_overrides),
    )


def _body() -> Detection:
    return Detection(
        label="ic", confidence=0.92, bbox=BODY_BOX, detection_id="det_body"
    )


def _pads() -> list[Detection]:
    """Chân sát hai cạnh NGẮN của thân, nằm ngoài hộp."""

    out = []
    for index, (x1, x2) in enumerate(((44.0, 57.0), (163.0, 176.0))):
        for row, y in enumerate((55.0, 75.0)):
            out.append(
                Detection(
                    label="pads",
                    confidence=0.9,
                    bbox=BoundingBox(x1, y, x2, y + 12.0),
                    detection_id=f"pad_{index}_{row}",
                )
            )
    return out


def _family(family: str = "ic") -> list[ComponentClassification]:
    return [
        ComponentClassification(
            crop_id="crop_0001",
            detection_id="det_body",
            family=family,
            probability=0.93,
            top_k=[ClassProbability(family, 0.93)],
            unknown_score=0.07,
            decision="accept",
            model_version="rule-wiring-test",
        )
    ]


def test_the_rule_reaches_the_detection_when_it_is_turned_on() -> None:
    pipeline = AOIPipeline(_config(enabled=True))
    pipeline.make_solder_crops(
        IMAGE,
        [_body(), *_pads()],
        family_classifications=_family(),
    )

    results = pipeline.last_package_classifications
    assert [item.package_class for item in results] == ["ic_hai_ben"]
    assert results[0].source == "package_rules"

    body = next(
        item
        for item in pipeline.last_package_detections
        if item.detection_id == "det_body"
    )
    assert body.metadata["package_prediction"]["package_class"] == "ic_hai_ben"


def test_nothing_happens_while_the_rule_is_off() -> None:
    """Mặc định của repo. Có code trên đĩa không được tự đổi hình học ROI."""

    pipeline = AOIPipeline(_config())
    pipeline.make_solder_crops(
        IMAGE, [_body(), *_pads()], family_classifications=_family()
    )
    assert pipeline.last_package_classifications == []
    assert pipeline.last_package_rule_skip is None


def test_turning_the_rule_on_without_families_says_so_out_loud() -> None:
    """Luật khoá theo họ 6.1. Gọi ``make_solder_crops`` mà quên truyền họ —
    đúng như Streamlit đang gọi — thì luật không chạy được. Im lặng trả rỗng ở
    đây là bẫy: người dùng bật tính năng, không thấy gì, và không có manh mối
    nào."""

    pipeline = AOIPipeline(_config(enabled=True))
    pipeline.make_solder_crops(IMAGE, [_body(), *_pads()])

    assert pipeline.last_package_classifications == []
    assert pipeline.last_package_rule_skip is not None
    assert "6.1" in pipeline.last_package_rule_skip


def test_a_running_model_is_never_overwritten_by_the_rule() -> None:
    """Luật NỐI SAU classifier, không thay nó. Thân nào đã có kết quả thì luật
    không đụng vào."""

    from_model = PackageClassification(
        crop_id="crop_0001",
        detection_id="det_body",
        package_class="ic_bon_ben",
        probability=0.88,
        top_k=[ClassProbability("ic_bon_ben", 0.88)],
        unknown_score=0.12,
        decision="accept",
        model_version="model-v1",
    )
    pipeline = AOIPipeline(_config(enabled=True))
    pipeline.make_solder_crops(
        IMAGE,
        [_body(), *_pads()],
        package_classifications=[from_model],
        family_classifications=_family(),
    )

    results = pipeline.last_package_classifications
    assert [item.package_class for item in results] == ["ic_bon_ben"]
    assert results[0].model_version == "model-v1"


def test_the_rule_changes_the_terminal_topology_it_is_there_to_change() -> None:
    """Giá trị của cả bước này nằm ở đây: ``ic`` mặc định là ``multi_pin``, còn
    luật cho biết chân nằm ở hai cạnh nào."""

    from aoi_pipeline.config import terminal_geometry

    assert terminal_geometry("ic") == "multi_pin"
    assert terminal_geometry("ic", package="ic_hai_ben") == "ic_hai_ben"
