from __future__ import annotations

import numpy as np

from aoi_pipeline.config import FusionConfig, SolderJointConfig
from aoi_pipeline.models import BoundingBox, Detection
from aoi_pipeline.solder.cad import BoardCad, CadComponent, CadRegistration
from aoi_pipeline.solder.cad_fusion import fuse_solder_joints
from aoi_pipeline.solder.geometry import SolderJointCropper


def _registration() -> CadRegistration:
    return CadRegistration(
        matrix=np.asarray([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]])
    )


def _config() -> FusionConfig:
    return FusionConfig(
        solder=SolderJointConfig(
            split_pins=False,
            refine_to_metal=False,
            include_body_view=False,
            deconflict_neighbours=False,
        )
    )


def _fuse(footprint: str):
    image = np.zeros((240, 240, 3), dtype=np.uint8)
    detection = Detection(
        "resistor",
        0.9,
        BoundingBox(60, 80, 140, 120),
        detection_id="det_footprint",
    )
    board = BoardCad(
        components=[
            CadComponent(
                designator="U1",
                x=10.0,
                y=10.0,
                rotation=0.0,
                footprint=footprint,
                part_class="resistor",
            )
        ]
    )
    derived = SolderJointCropper(_config().solder).derive(image, [detection])
    result = fuse_solder_joints(
        [detection],
        derived,
        240,
        240,
        board=board,
        registration=_registration(),
        config=_config(),
        image=image,
    )
    return detection, result


def test_placement_footprint_overrides_wrong_detector_family_topology() -> None:
    detection, result = _fuse("SOIC-16")
    joints = [item for item in result.joints if item.kind == "joint"]

    assert result.used_cad is True
    assert {item.terminal_geometry for item in joints} == {"dual_sided"}
    assert {item.position for item in joints} == {"lead_top", "lead_bottom"}
    assert detection.metadata["package_profile"]["source"] == "footprint"
    assert detection.metadata["package_profile"]["expected_pin_count"] == 16
    assert all(item.metadata["package_profile"]["package_class"] == "ic_hai_ben" for item in joints)


def test_hidden_terminal_footprint_suppresses_cad_reanchored_fake_rois() -> None:
    detection, result = _fuse("QFN-32")

    assert result.used_cad is True
    assert result.joints == []
    assert detection.metadata["package_profile"]["package_class"] == "ic_khong_chan"
    assert detection.metadata["terminal_geometry_override"] == "hidden_terminals"


def test_cad_evidence_clears_the_image_rules_measured_edges() -> None:
    """CAD tiếp quản thì cạnh chân do LUẬT ẢNH đo được phải bị xoá.

    CAD mang cả hình học lẫn hướng, nên giữ lại cạnh cũ là **trộn hai nguồn**:
    hình học của CAD ghép với cạnh của luật ảnh. Trong một lượt chạy thì
    ``derive`` đã xong trước khi CAD ghi, nên chưa lộ ra — nhưng detection bị sửa
    TẠI CHỖ và ``AOIPipeline.last_package_detections`` giữ chính object đó, nên
    gọi ``make_solder_crops`` lần nữa trên chúng là dính.
    """

    import numpy as np

    image = np.zeros((240, 240, 3), dtype=np.uint8)
    detection = Detection(
        "resistor", 0.9, BoundingBox(60, 80, 140, 120),
        detection_id="det_footprint",
        metadata={
            "terminal_lead_edges": ["left", "right"],
            "terminal_lead_edges_space": "image",
        },
    )
    board = BoardCad(components=[CadComponent(
        designator="U1", x=10.0, y=10.0, rotation=0.0,
        footprint="SOIC-16", part_class="resistor")])
    derived = SolderJointCropper(_config().solder).derive(image, [detection])
    fuse_solder_joints([detection], derived, 240, 240, board=board,
                       registration=_registration(), config=_config(), image=image)

    assert "terminal_lead_edges" not in detection.metadata, (
        "cạnh của luật ảnh còn lại sau khi CAD tiếp quản; lần dựng ROI sau sẽ "
        "ghép hình học CAD với cạnh của luật"
    )
    assert "terminal_lead_edges_space" not in detection.metadata
