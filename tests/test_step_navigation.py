"""Every navigable step must actually be renderable.

The step list, the per-step status map, the reset map and the renderer table are
four separate structures that all have to agree. They drifted the moment step
6.2 was added to the navigation only: ``statuses`` was built from a literal
``range(7)``, so the sidebar raised ``KeyError: 7`` on the very first render and
the whole app died before showing anything.

These tests compare the four against each other rather than against a hard-coded
count, so adding a step 8 later cannot reintroduce the same class of bug.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

import app.streamlit_app as ui


def _step_indices() -> list[int]:
    return [step for step, *_ in ui.STEP_DEFINITIONS]


def test_step_indices_are_unique_and_contiguous_from_zero() -> None:
    """Chỉ số nội bộ phải phủ kín 0..N-1, nhưng KHÔNG cần tăng dần theo thứ tự
    hiển thị: Golden Inspection mang chỉ số 8 mà đứng ở vị trí 3.5 trong đường
    ống. Chỉ số là khoá của `renderers`/`statuses`; thứ tự công việc do
    `STEP_ORDER` giữ."""

    indices = _step_indices()
    assert len(set(indices)) == len(indices), "chỉ số trùng nhau"
    assert sorted(indices) == list(range(len(indices))), "chỉ số phải phủ kín 0..N-1"


def test_the_displayed_numbers_are_unique_and_follow_the_work_order() -> None:
    """Số hiển thị mới là thứ người dùng đọc. Hai bước cùng số là mơ hồ."""

    shown = [row[4] for row in ui.STEP_DEFINITIONS]
    assert len(set(shown)) == len(shown), f"số hiển thị trùng: {shown}"
    assert [float(value) for value in shown] == sorted(float(v) for v in shown), (
        "số hiển thị phải tăng dần theo thứ tự trong bảng"
    )


def test_golden_inspection_sits_after_the_board_roi_step() -> None:
    """Nó chỉ cần ảnh đã căn và vùng board — không cần detect hay phân loại.
    Đặt nó cuối danh sách chỉ vì được thêm sau cùng là sai thứ tự công việc."""

    order = [row[1] for row in ui.STEP_DEFINITIONS]
    assert order.index("Golden Inspection") == order.index("Khoanh vùng PCB") + 1


@pytest.fixture(scope="module")
def rendered_app():
    """The app after one real render, so these assertions test the running
    thing rather than a re-implementation of its initialisation."""

    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(ui.__file__)), default_timeout=120)
    app.run()
    assert not app.exception, [str(e.value) for e in app.exception]
    return app


def test_the_default_status_map_covers_every_navigable_step(rendered_app) -> None:
    """The exact failure seen in the app: KeyError: 7 raised by the sidebar
    before a single widget was drawn."""

    statuses = rendered_app.session_state["statuses"]
    missing = [step for step in _step_indices() if step not in statuses]
    assert missing == [], f"bước không có status mặc định: {missing}"


def test_init_state_declares_every_session_key_the_ui_reads() -> None:
    """Streamlit does not invent attributes; an undeclared key is an
    AttributeError the first time a tab touches it. Step 6.2 needed eight of
    them (``solder_result``, the model/manifest triples, ``cad_summary``) and
    none were declared when the section was first wired up."""

    source = Path(ui.__file__).read_text(encoding="utf-8")
    # ``st.session_state.get(...)`` is a Mapping method, not a key. Reading it
    # as one made this test fail the moment the UI stopped hard-coding a key.
    MAPPING_METHODS = {
        "get", "keys", "items", "values", "pop", "setdefault", "update", "clear",
        "to_dict", "copy",
    }
    used = set(re.findall(r"st\.session_state\.([a-z][a-z0-9_]*)", source)) - MAPPING_METHODS
    declared = set(re.findall(r'^\s*"([a-z][a-z0-9_]*)":', inspect.getsource(ui._init_state), re.M))
    # Widget-bound keys are created by Streamlit on render, not by _init_state.
    widget_keys = set(re.findall(r'key="([a-z][a-z0-9_]*)"', source))
    missing = sorted(used - declared - widget_keys)
    assert missing == [], f"key chưa khai báo trong _init_state: {missing}"


def test_every_step_has_a_renderer() -> None:
    source = inspect.getsource(ui.main)
    for step in _step_indices():
        assert f"{step}: _render_step_" in source, f"bước {step} không có renderer"


def test_resetting_a_step_invalidates_every_later_step() -> None:
    """``_invalidate_after`` walks the step list; a literal bound would leave
    the last step holding stale results from a previous board."""

    source = inspect.getsource(ui._invalidate_after)
    assert "STEP_ORDER" in source, (
        "_invalidate_after phải đi theo STEP_ORDER, không dùng số cứng và không "
        "dùng range() — từ khi Golden thành bước 3.5 thì 'các bước sau' không "
        "còn trùng với 'chỉ số lớn hơn'"
    )
    covered = {int(index) for index in re.findall(r"^\s*(\d+): \"\w+\",", source, re.M)}
    missing = [step for step in _step_indices() if step not in covered]
    assert missing == [], f"bước không có result_key khi reset: {missing}"


def test_no_renderer_calls_a_name_that_does_not_exist() -> None:
    """A NameError inside a Streamlit renderer is invisible until that panel is
    drawn, so importing the module proves nothing. A merge dropped
    ``_draw_verdict_overlay``, ``_verdict_frame`` and ``import collections``
    from step 6.2 and every test still passed.

    ``symtable`` gives the compiler's own view of which names each function
    resolves globally; anything not in the module or in builtins is a
    NameError waiting for a user to open that tab.
    """

    import builtins
    import symtable

    import app.pipeline_bridge as bridge

    for module in (ui, bridge):
        path = Path(module.__file__)
        table = symtable.symtable(path.read_text(encoding="utf-8"), str(path), "exec")
        missing: set[tuple[str, str]] = set()

        def walk(scope) -> None:
            for symbol in scope.get_symbols():
                if symbol.is_global() and not symbol.is_assigned():
                    name = symbol.get_name()
                    if not hasattr(module, name) and not hasattr(builtins, name):
                        missing.add((scope.get_name(), name))
            for child in scope.get_children():
                walk(child)

        walk(table)
        assert missing == set(), f"{path.name}: tên chưa định nghĩa {sorted(missing)}"


# --------------------------------------------------------------------------
# Golden Inspection là một BƯỚC, không phải một workspace riêng
# --------------------------------------------------------------------------


def test_golden_inspection_is_a_step_in_the_pipeline() -> None:
    """Nó vốn đã kiểm chính tấm ảnh của bước 0, nên tách ra một chế độ riêng
    chỉ bắt người dùng nạp lại đúng những thứ đã nạp."""

    names = {index: name for index, name, *_ in ui.STEP_DEFINITIONS}
    assert names[8] == "Golden Inspection"
    assert "8: _render_step_eight" in inspect.getsource(ui.main)


def test_the_workspace_switch_is_gone_from_the_sidebar() -> None:
    source = Path(ui.__file__).read_text(encoding="utf-8")
    assert "workspace_mode" not in source, (
        "còn sót chế độ workspace; Golden nay là bước 8 của cùng một đường ống"
    )


def test_resetting_the_source_image_clears_the_golden_run_too() -> None:
    """Bước 8 kiểm chính ảnh của bước 0, nên ảnh mới thì kết quả cũ là kết quả
    của một board khác. Recipe thì KHÔNG bị xoá — nó dựng từ ảnh Golden riêng
    và sống lâu hơn từng board."""

    source = inspect.getsource(ui._invalidate_after)
    # Bỏ dòng chú thích ra: chúng nhắc tới `inspection_recipe` để giải thích vì
    # sao nó KHÔNG có mặt, và đó không phải là code.
    code = chr(10).join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    assert '8: "inspection_run"' in code
    assert "inspection_recipe" not in code


def test_rerunning_detection_does_not_wipe_the_golden_result() -> None:
    """Hệ quả thật của việc Golden thành bước 3.5: nó phụ thuộc ảnh đã căn và
    vùng board, KHÔNG phụ thuộc detect. Chạy lại detect mà xoá kết quả Golden
    là bắt người dùng làm lại một việc chẳng liên quan.

    Với `range(step+1, ...)` cũ thì điều này sai: Golden mang chỉ số 8 nên mọi
    bước đều xoá nó.
    """

    order = list(ui.STEP_ORDER)
    golden = 8
    detect = 4
    assert order.index(golden) < order.index(detect), "Golden phải đứng trước detect"

    after_detect = order[order.index(detect) + 1:]
    assert golden not in after_detect, "chạy lại detect không được xoá kết quả Golden"

    after_board = order[order.index(3) + 1:]
    assert golden in after_board, "đổi vùng board thì kết quả Golden phải bị xoá"
