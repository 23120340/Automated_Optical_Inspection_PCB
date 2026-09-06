"""Kho lịch sử của app phải ở chỗ sống được lâu — và không lên GitHub.

Hai điều kiện này không phải chi tiết trang trí:

* Câu hỏi mà kho sinh ra để trả lời — *bo này đã kiểm mặt nào rồi* — chỉ có
  nghĩa nếu dữ liệu **sống qua lần khởi động lại**. Recipe workspace của app đặt
  trong thư mục tạm và như thế là đúng với nó; kho lịch sử mà đi theo thì mất
  sạch sau một lần dọn máy.
* Kho chứa **ảnh cắt từ bo thật**. Đưa nhầm nó vào git là đẩy ảnh sản xuất của
  khách lên một repo công khai — hỏng một lần là không rút lại được.
"""

from __future__ import annotations

import subprocess

import app.streamlit_app as ui


def test_the_history_store_outlives_a_restart() -> None:
    """Không nằm trong thư mục tạm, và nằm trong chính dự án."""

    import tempfile
    from pathlib import Path

    root = ui._history_store_root()
    assert ui.PROJECT_ROOT in root.parents
    assert Path(tempfile.gettempdir()).resolve() not in root.resolve().parents


def test_the_history_store_is_never_committed() -> None:
    """Ảnh bo thật không được lên repo. Hỏi thẳng git, không đoán."""

    root = ui._history_store_root()
    probe = root / "images" / "aa" / "probe.png"
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(probe)],
        cwd=ui.PROJECT_ROOT,
        capture_output=True,
    )
    assert result.returncode == 0, f"{probe} KHÔNG bị .gitignore chặn"


def test_the_result_page_offers_to_save_the_run() -> None:
    """Chỗ nối thật sự tồn tại: trang kết quả có gọi khối lưu lịch sử.

    Lớp lưu trữ trước đây đủ test nhưng **không nơi nào gọi** — thư viện không ai
    gọi thì chưa biết nó có vừa dữ liệu thật không.
    """

    import inspect

    source = inspect.getsource(ui._render_inspect_board_mode)
    assert "_render_inspection_history(result)" in source
