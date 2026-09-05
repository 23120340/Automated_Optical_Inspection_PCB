"""Cổng đường luật 5.2 phải FAIL khi mất pad, và phải nói được vì sao bỏ qua.

Một cổng luôn PASS thì vô dụng, nên phần lớn test dưới đây dựng ra tình huống
mất pad rồi kiểm cổng có bắt được không.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_package_rule_gate import BoardResult, evaluate_board, main

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "data" / "solder_geometry"


def _board() -> BoardResult:
    return evaluate_board(FIXTURES / "board_smd_00001.json", "truth", None)


def test_the_gate_runs_and_counts_every_hand_measured_pad() -> None:
    result = _board()
    assert result.pads_total == 28
    assert result.covered_before == 28, (
        "đường baseline lẽ ra phủ đủ 28 pad; lệch nghĩa là fixture hoặc 5.5 đã đổi"
    )


def test_losing_a_baseline_pad_fails_the_gate(monkeypatch) -> None:
    """Hợp đồng quan trọng nhất của cổng: mất pad ⇒ exit code khác 0.

    Test cũ của tôi ở đây là GIẢ — nó tự thêm vào ``result.lost`` rồi assert
    chính cái list đó, nên đột biến "``main`` luôn trả 0" không bị bắt. Bản này
    đi qua đúng ``main()``.
    """

    from scripts import evaluate_package_rule_gate as gate

    def _fake(path, families_mode, lead_model):
        result = BoardResult(board=path.stem, pads_total=2)
        result.covered_before = 2
        result.covered_after = 1
        result.lost.append("U1 pad0")
        return result

    monkeypatch.setattr(gate, "evaluate_board", _fake)
    assert gate.main([str(FIXTURES), "--no-leads"]) == 1, (
        "mất pad baseline mà cổng vẫn trả 0 thì nó không phải cổng"
    )


def test_a_clean_board_passes(monkeypatch) -> None:
    from scripts import evaluate_package_rule_gate as gate

    def _fake(path, families_mode, lead_model):
        result = BoardResult(board=path.stem, pads_total=2)
        result.covered_before = result.covered_after = 2
        return result

    monkeypatch.setattr(gate, "evaluate_board", _fake)
    assert gate.main([str(FIXTURES), "--no-leads"]) == 0


def test_the_gate_exits_zero_on_the_shipped_fixture() -> None:
    assert main([str(FIXTURES), "--no-leads"]) == 0


def test_the_gate_says_WHY_it_abstained_not_just_how_many() -> None:
    """"Bỏ qua 15" không dùng được. "3 con ic vì 0 cạnh có dải chân" thì dùng."""

    result = _board()
    assert result.abstain_reasons, "phải ghi lý do bỏ qua"
    assert all("-" in key for key in result.abstain_reasons), (
        "mỗi lý do phải có dạng '<họ> - <n> canh co dai chan'"
    )
    assert sum(result.abstain_reasons.values()) == result.abstained


def test_the_gate_flags_leads_that_land_inside_the_body() -> None:
    """Chân có tâm trong box thân thì không đóng góp cạnh nào.

    Cổng phải nói ra chứ không im lặng báo PASS — PASS khi nhánh ``ic`` chưa
    chạy lần nào thì không có nghĩa là luật đã được kiểm. Cổng KHÔNG kết luận
    nguyên nhân: có thể do quy ước box cũ, cũng có thể do lượt 2 vốn tìm mối
    hàn trong một cửa sổ quanh linh kiện.
    """

    result = evaluate_board(
        FIXTURES / "board_smd_00001.json", "truth",
        Path("models/active/lead_detector/best.onnx"),
    )
    if result.leads_found == 0:
        pytest.skip("không có lead detector trong môi trường này")
    assert result.leads_inside_body > 0, (
        "fixture này dùng quy ước box cũ nên phải có chân nằm trong thân"
    )
