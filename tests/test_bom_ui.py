"""Luồng BOM chạy thật qua giao diện, không chỉ qua hàm.

`tests/test_bom.py` kiểm phần logic. Ở đây kiểm phần dễ hỏng lặng lẽ hơn: nút
bấm có nối đúng hàm không, state có tên đúng không, và bảng đối chiếu có thật
sự hiện ra khi board lệch BOM không. Một lỗi chính tả trong key session state
không làm test logic đỏ, nhưng làm tính năng biến mất khỏi giao diện.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from aoi_pipeline.bom import BillOfMaterials, BomEntry
from aoi_pipeline.models import BoundingBox, Detection

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture
def app() -> AppTest:
    instance = AppTest.from_file(APP, default_timeout=180)
    instance.session_state["workspace_mode"] = "pipeline_lab"
    instance.run()
    return instance


def _detection(label: str, x: float, y: float, detection_id: str) -> Detection:
    return Detection(
        label=label, confidence=0.9,
        bbox=BoundingBox(x - 10, y - 10, x + 10, y + 10),
        detection_id=detection_id,
    )


def test_the_bom_uploader_is_on_the_sidebar(app: AppTest) -> None:
    assert any(item.key == "bom_uploader" for item in app.sidebar.get("file_uploader"))


def test_completeness_defaults_to_complete(app: AppTest) -> None:
    """Mặc định phải là "đủ board". Đây là cái quyết định linh kiện lạ bị coi
    là LỖI hay chỉ là ghi nhận, và mặc định lỏng sẽ để lỗi trôi qua im lặng."""

    checkbox = next(item for item in app.sidebar.checkbox if item.key == "bom_complete")
    assert checkbox.value is True


def test_state_starts_with_no_bom(app: AppTest) -> None:
    assert app.session_state["bom"] is None
    assert app.session_state["bom_name"] is None


def test_the_ui_reads_the_same_state_keys_the_loader_writes() -> None:
    """Lỗi chính tả ở đây làm tính năng biến mất khỏi giao diện mà không test
    logic nào đỏ."""

    source = Path(APP).read_text(encoding="utf-8")
    for key in ("bom", "bom_name", "bom_complete"):
        assert f'"{key}"' in source or f"st.session_state.{key}" in source


def test_reconciliation_renders_when_a_board_matches_its_bom(app: AppTest) -> None:
    app.session_state["bom"] = BillOfMaterials(
        entries=[BomEntry(designator="R1", part_class="resistor")],
        source="test",
    )
    app.session_state["active_step"] = 4
    app.run()
    assert not app.exception


def test_an_unlisted_component_reaches_the_operator_as_an_error(app: AppTest) -> None:
    """Ca người dùng nêu, đi hết đường từ state tới màn hình."""

    from aoi_pipeline.bom import reconcile_bom

    bom = BillOfMaterials(
        entries=[BomEntry(designator="R1", part_class="resistor")],
        source="test", complete=True,
    )
    detections = [
        _detection("resistor", 100.0, 100.0, "d0"),
        _detection("capacitor", 400.0, 400.0, "d1"),   # BOM không có
    ]
    result = reconcile_bom(bom, detections, None)

    assert not result.passed
    unexpected = [item for item in result.findings if item.kind == "unexpected"]
    assert unexpected and unexpected[0].severity == "error"

    # Bảng hiển thị phải nói bằng tiếng Việt, không phải kind thô.
    import app.streamlit_app as ui

    frame = ui._bom_findings_frame(result.findings)
    assert "LỖI" in frame["mức"].tolist()
    assert "không có trong BOM" in frame["loại"].tolist()


def test_findings_are_sorted_with_errors_first() -> None:
    """Người vận hành đọc từ trên xuống. Lỗi nằm dưới ba dòng ghi nhận là lỗi
    bị bỏ qua."""

    import app.streamlit_app as ui
    from aoi_pipeline.bom import BomFinding

    findings = [
        BomFinding(kind="class_mismatch", severity="warning", message="w"),
        BomFinding(kind="bom_inconsistent", severity="info", message="i"),
        BomFinding(kind="missing", severity="error", message="e"),
    ]
    frame = ui._bom_findings_frame(findings)
    assert frame["mức"].tolist() == ["LỖI", "cảnh báo", "ghi nhận"]


def test_a_positioned_bom_without_an_image_renders_instead_of_crashing(
    app: AppTest,
) -> None:
    """BOM có toạ độ nhưng chưa có ảnh phân tích thì không căn được. Phải rơi
    về đếm theo lớp và vẫn vẽ ra màn hình — ghép theo toạ độ mà chưa biết board
    nằm đâu sẽ cho một bảng trông rất thuyết phục và sai toàn bộ."""

    app.session_state["bom"] = BillOfMaterials(
        entries=[BomEntry(designator="R1", part_class="resistor", x=1.0, y=2.0)],
        source="test",
    )
    app.session_state["active_step"] = 4
    app.run()

    assert not app.exception


def test_the_projection_the_ui_hands_over_is_actually_callable() -> None:
    """Test này tồn tại vì một bug thật: `_bom_projection` từng đọc sai key
    session state VÀ gọi sai tên method (`project` thay vì `to_image`). Cả hai
    lỗi đều trả về None một cách im lặng, nên đường ghép theo toạ độ chết hẳn
    mà không test nào đỏ và giao diện vẫn hiện một bảng trông hợp lý.

    Nên ở đây kiểm chính thứ đã hỏng: registration thật, gọi thật, ra pixel.
    """

    import numpy as np
    from aoi_pipeline.solder.cad import register_cad

    # Board mm và detection pixel của cùng một bố cục, tỉ lệ 10 px/mm.
    entries = [
        BomEntry(designator=f"R{index}", part_class="resistor",
                 x=float(x), y=float(y))
        for index, (x, y) in enumerate(
            [(5, 5), (25, 5), (45, 5), (5, 25), (25, 25), (45, 25)], start=1)
    ]
    bom = BillOfMaterials(entries=entries, source="test")
    detections = [
        _detection("resistor", entry.x * 10.0, entry.y * 10.0, f"d{index}")
        for index, entry in enumerate(entries)
    ]

    registration = register_cad(bom.to_board_cad(), detections, (600, 400))
    assert registration is not None, "RANSAC phải căn được bố cục khớp hoàn toàn"

    project = registration.to_image
    projected = project(np.array([[5.0, 5.0]], dtype=np.float64))
    assert projected.shape == (1, 2)
    assert np.allclose(projected[0], [50.0, 50.0], atol=5.0), (
        f"chiếu (5, 5) mm ra {projected[0]}, phải gần (50, 50) px"
    )

    from aoi_pipeline.bom import reconcile_bom

    result = reconcile_bom(bom, detections, project)
    assert result.stats["method"] == "position"
    assert result.passed, [item.message for item in result.errors]
