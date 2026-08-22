"""Không notebook nào được chết ở cell xuất ONNX.

Chuyện đã xảy ra thật, 2026-08-22, trên `pcb_classifier_v2_kaggle`:

    ModuleNotFoundError: No module named 'onnxscript'

Từ torch 2.9, `torch.onnx.export` mặc định `dynamo=True`, và đường đó uỷ quyền
cho `onnxscript`. Image Kaggle không có gói ấy. Cell xuất ONNX là cell **cuối
cùng**, nên khi nó hỏng thì cả run không để lại artifact nào — dù trọng số đã
được lưu an toàn từ trước đó.

Điều làm lỗi này đắt không phải độ khó sửa (một tham số) mà là **thời điểm**:
nó chỉ nổ sau khi toàn bộ giờ GPU đã tiêu xong.

Mẫu đúng đã có sẵn trong `pcb_component_classification_kaggle` từ trước;
notebook v2 đơn giản là không kế thừa. Test này giữ mẫu đó cho mọi file.

Ghi chú: các notebook YOLO không nằm trong phạm vi test — chúng xuất qua
`model.export(format="onnx")` của ultralytics, mà ultralytics đã tự truyền
`dynamo=False` cho torch >= 2.4.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: Mọi file gọi thẳng `torch.onnx.export`. `.ipynb` được sinh từ `.py` nên chỉ
#: cần kiểm bản `.py`.
SOURCES = sorted(
    path
    for path in list((ROOT / "training").rglob("*.py"))
    + list((ROOT / "scripts").rglob("*.py"))
    if "torch.onnx.export" in path.read_text(encoding="utf-8", errors="replace")
)


def _export_calls(tree: ast.AST) -> list[ast.Call]:
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if (
            isinstance(target, ast.Attribute)
            and target.attr == "export"
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "onnx"
        ):
            calls.append(node)
    return calls


def test_there_is_something_to_check() -> None:
    """Nếu không file nào gọi torch.onnx.export nữa thì test dưới sẽ xanh một
    cách vô nghĩa. Bắt điều đó ngay ở đây."""

    assert SOURCES, "không tìm thấy file nào gọi torch.onnx.export"


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_every_export_states_which_exporter_it_wants(path: Path) -> None:
    """`dynamo` phải được nêu rõ, không để mặc định.

    Mặc định đổi theo phiên bản torch: cùng một notebook chạy được hôm nay và
    hỏng sau một lần Kaggle nâng image, mà không có dòng code nào thay đổi.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = _export_calls(tree)
    assert calls, f"{path.name}: chuỗi có nhưng AST không thấy lời gọi nào"

    for call in calls:
        names = {keyword.arg for keyword in call.keywords if keyword.arg}
        assert "dynamo" in names or any(keyword.arg is None for keyword in call.keywords), (
            f"{path.name}:{call.lineno} gọi torch.onnx.export mà không nêu "
            "`dynamo`. Mặc định của torch đổi theo phiên bản, và từ 2.9 nó cần "
            "`onnxscript` — gói mà Kaggle không có."
        )


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_every_export_has_a_second_path_to_fall_back_to(path: Path) -> None:
    """Một lời gọi duy nhất, không dự phòng, là một điểm hỏng đơn ở cuối run.

    Bộ xuất TorchScript đã deprecated và sẽ bị bỏ; đường dynamo cần thêm gói.
    Cả hai đều có thể vắng mặt, nên phải có đường thứ hai.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = _export_calls(tree)

    guarded = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        in_try = {id(call) for call in _export_calls(ast.Module(body=node.body, type_ignores=[]))}
        in_handlers = set()
        for handler in node.handlers:
            in_handlers |= {
                id(call)
                for call in _export_calls(ast.Module(body=handler.body, type_ignores=[]))
            }
        if in_try and in_handlers:
            guarded |= in_try | in_handlers

    unguarded = [call for call in calls if id(call) not in guarded]
    assert not unguarded, (
        f"{path.name}: lời gọi ở dòng "
        f"{[call.lineno for call in unguarded]} không có đường dự phòng. "
        "Đặt trong try/except với lời gọi thứ hai dùng bộ xuất còn lại — "
        "cell này chạy sau khi mọi giờ GPU đã tiêu xong."
    )


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_the_no_dependency_path_is_tried_first(path: Path) -> None:
    """`dynamo=False` không cần gói nào và giữ đúng opset được yêu cầu — đường
    dynamo âm thầm nâng opset lên bản nó hỗ trợ (đo được: xin 12, nhận 18).

    Nên nó phải là lựa chọn đầu, không phải phương án chữa cháy.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        first = _export_calls(ast.Module(body=node.body, type_ignores=[]))
        if not first:
            continue
        for call in first:
            for keyword in call.keywords:
                if keyword.arg == "dynamo":
                    assert keyword.value.value is False, (
                        f"{path.name}:{call.lineno} thử đường dynamo trước. "
                        "Trên Kaggle nó luôn hỏng rồi mới rơi về dự phòng — "
                        "in ra một traceback đáng sợ mà không cần thiết."
                    )
