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


def test_position_matching_is_skipped_without_a_registration(app: AppTest) -> None:
    """Ghép theo toạ độ mà chưa biết board nằm đâu trong ảnh sẽ cho ra một bảng
    trông rất thuyết phục và sai toàn bộ. Phải rơi về đếm theo lớp."""

    import app.streamlit_app as ui
    from aoi_pipeline.bom import reconcile_bom

    assert ui._bom_projection() is None

    bom = BillOfMaterials(
        entries=[BomEntry(designator="R1", part_class="resistor", x=1.0, y=2.0)],
        source="test",
    )
    result = reconcile_bom(bom, [_detection("resistor", 5.0, 5.0, "d0")],
                           ui._bom_projection())
    assert result.stats["method"] == "count"
