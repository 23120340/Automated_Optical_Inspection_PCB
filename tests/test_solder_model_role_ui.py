"""UI contract for the step-6.2 ROI classifier slot.

Trước đây file này giữ hợp đồng "tách hai vai trò solder": ô classifier ROI và ô
detect lỗi toàn board phải độc lập, không nhận manifest của nhau. Tầng detect
toàn board **đã được gỡ** khỏi app và pipeline — dự án đi theo hướng lượt 2
(định vị mối hàn) + 6.2 (chấm ROI) — nên vế thứ hai của hợp đồng đó không còn
chủ thể.

Cái còn lại và vẫn phải giữ: ô classifier từ chối một manifest không phải của
nó. Manifest của model detect lỗi vẫn nằm trên đĩa nên vẫn dùng làm mẫu thử
được.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import app.streamlit_app as ui


#: Manifest của model detect lỗi toàn board. Tầng đó đã gỡ khỏi app, nhưng file
#: vẫn trên đĩa và vẫn là mẫu "manifest không phải của ô classifier" tốt nhất.
FOREIGN_MANIFEST = (
    ui.PROJECT_ROOT / "models" / "active" / "solder" / "defect" / "model_manifest.json"
)
CLASSIFIER_MODEL = (
    ui.PROJECT_ROOT / "models" / "active" / "solder" / "classifier" / "best.onnx"
)
CLASSIFIER_MANIFEST = CLASSIFIER_MODEL.with_name("model_manifest.json")


def test_the_sidebar_offers_the_roi_classifier_and_pass_two_only() -> None:
    """Tầng detect toàn board đã gỡ. Nếu nhãn của nó còn sót lại trong sidebar
    thì người dùng vẫn thấy một tính năng không còn chạy được nữa."""

    source = inspect.getsource(ui._render_sidebar)

    assert "Classifier ROI mối hàn · raw logits" in source
    assert 'Manifest classifier ROI (model_manifest.json)' in source
    assert '_render_model_picker("solder")' in source
    assert "_render_pass2_lead_controls()" in source

    for gone in ("solder_segmenter", "Model khoanh lỗi", "toàn board"):
        assert gone not in source, f"nhãn của tầng đã gỡ còn sót: {gone!r}"


def test_the_classifier_pair_is_held_back_until_it_is_complete() -> None:
    """Nửa cặp thì không được đưa vào runtime: một ONNX không có manifest thì
    thứ tự lớp phải đoán, mà đoán sai là ánh xạ mọi lỗi thành đạt."""

    config = ui._default_config()
    config["solder_grading"]["model_path"] = str(CLASSIFIER_MODEL)

    guarded = ui._engine_config(config)

    assert guarded["solder_grading"]["model_path"] is None
    assert guarded["solder_grading"]["manifest_path"] is None


def test_the_whole_board_defect_stage_is_gone_from_the_config() -> None:
    """Gỡ tính năng nghĩa là app không còn ghi khoá đó nữa. Để sót lại một
    section chết thì lần sau ai đó lại nối dây vào nó."""

    assert "solder_defect_detection" not in ui._default_config()
    assert "solder_defect_detection" not in ui._engine_config(ui._default_config())


ROLE_HARNESS = '''
import sys
from pathlib import Path

sys.path.insert(0, {root!r})

import streamlit as st
import app.streamlit_app as ui


class _Upload:
    def __init__(self, path: str) -> None:
        source = Path(path)
        self.name = source.name
        self._data = source.read_bytes()

    def getvalue(self) -> bytes:
        return self._data


ui._init_state()
ui._adopt_active_models()

# Ô classifier tự nạp bản trong models/active.
assert Path(st.session_state.solder_model_path) == Path({classifier_model!r})
assert Path(st.session_state.solder_manifest_path) == Path({classifier_manifest!r})

# Và nó phải TỪ CHỐI một manifest không phải của nó, giữ nguyên cặp đang có.
before = st.session_state.solder_manifest_path
try:
    ui._set_solder_manifest(_Upload({foreign_manifest!r}))
except ValueError as exc:
    assert "classifier ROI" in str(exc), str(exc)
else:
    raise AssertionError("một manifest lạ đã được ô classifier nhận")
assert st.session_state.solder_manifest_path == before

st.markdown("ROLE_SPLIT_OK")
'''


def test_the_classifier_slot_refuses_a_manifest_that_is_not_its_own(
    tmp_path: Path,
) -> None:
    from streamlit.testing.v1 import AppTest

    assert CLASSIFIER_MODEL.is_file() and CLASSIFIER_MANIFEST.is_file()
    assert FOREIGN_MANIFEST.is_file(), "cần một manifest lạ để thử"
    script = tmp_path / "solder_role_harness.py"
    script.write_text(
        ROLE_HARNESS.format(
            root=str(ui.PROJECT_ROOT),
            classifier_model=str(CLASSIFIER_MODEL),
            classifier_manifest=str(CLASSIFIER_MANIFEST),
            foreign_manifest=str(FOREIGN_MANIFEST),
        ),
        encoding="utf-8",
    )
    app = AppTest.from_file(str(script), default_timeout=120).run()

    assert not app.exception, [str(item.value) for item in app.exception]
    assert any("ROLE_SPLIT_OK" in item.value for item in app.markdown)



def test_pass_two_appears_in_the_picker_so_a_model_can_be_found() -> None:
    """Trước đây mục lượt 2 chỉ có ô nhập đường dẫn bằng tay, nên một model đã
    cài vào ``models/active/lead_detector`` là vô hình trong app."""

    source = inspect.getsource(ui._render_pass2_lead_controls)
    assert '_render_model_picker("lead_detector")' in source
    assert "lead_detector" in ui._MODEL_SLOTS


def test_pass_two_is_the_one_slot_that_never_loads_itself() -> None:
    """Mọi ô model khác tự nạp bản trong ``models/active`` khi mở app. Lượt 2
    thì không, và đó là chủ ý: nó THAY ROI hình học, mà model hiện có làm độ phủ
    pad tụt từ 28/28 xuống 26/28 trên board thật. Một thay đổi như thế phải do
    người bật sau khi đọc số đo, không được xảy ra chỉ vì file có mặt."""

    assert "lead_detector" in ui._NO_AUTO_ADOPT
    source = inspect.getsource(ui._adopt_active_models)
    assert "_NO_AUTO_ADOPT" in source, "vòng tự nạp không đọc danh sách loại trừ"


def test_choosing_a_pass_two_model_is_what_switches_it_on() -> None:
    """Vì ô này không tự nạp, hành động chọn phải tự ghi vào config -- nếu không
    người dùng chọn xong mà lượt 2 vẫn nằm im, không có gì báo tại sao."""

    source = inspect.getsource(ui._use_model_entry)
    assert 'slot == "lead_detector"' in source
    assert '"lead_detection"' in source


def test_the_pass_two_section_warns_before_anyone_turns_it_on() -> None:
    """Bảng chọn chỉ hiện mAP50 0.871. Con số đó một mình mời người ta bật lên,
    trong khi phép đo trên board thật nói ngược lại."""

    source = inspect.getsource(ui._render_pass2_lead_controls)
    assert "0/28" in source and "2/28" in source
    assert "on_board_validation" in source


def test_choosing_a_pass_two_model_works_on_a_fresh_config() -> None:
    """``_default_config`` không tạo sẵn khoá ``lead_detection`` -- lượt 2 chỉ
    ghi khoá đó khi có ai chỉnh nó lần đầu. Bản đầu của hàm này truy cập thẳng
    ``config["lead_detection"]`` nên app chết với KeyError ngay tại thao tác
    chọn model, tức đúng lúc người dùng làm đúng thứ được hướng dẫn."""

    config = ui._default_config()
    assert "lead_detection" not in config, (
        "nếu khoá này đã có sẵn thì test mất ý nghĩa; sửa lại cho khớp"
    )

    class _Session(dict):
        __getattr__ = dict.get
        def __setattr__(self, key, value): self[key] = value

    session = _Session(config=config, messages=[])
    entry = ui.ModelEntry(
        name="lead_detector/best.onnx",
        kind="lead_detector",
        model_path=Path("models/active/lead_detector/best.onnx"),
        manifest_path=Path("models/active/lead_detector/model_manifest.json"),
        origin="active",
    )
    original = ui.st.session_state
    ui.st.session_state = session
    try:
        ui._use_model_entry("lead_detector", entry)
    finally:
        ui.st.session_state = original

    assert session["config"]["lead_detection"]["model_path"].endswith("best.onnx")
    assert session["config"]["lead_detection"]["enabled"] is True
