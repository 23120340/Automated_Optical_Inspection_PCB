"""Cổng đường luật 5.2 phải FAIL khi mất pad, và phải nói được vì sao bỏ qua.

Một cổng luôn PASS thì vô dụng, nên phần lớn test dưới đây dựng ra tình huống
mất pad rồi kiểm cổng có bắt được không.

Nhóm cuối file canh một hợp đồng nữa, thêm 2026-09-05 sau khi nó đã hỏng thật:
**cổng phải dựng ROI bằng đúng hàm runtime của 5.5, không được dựng lại.** Bản
đầu tự lặp lại ``derive_solder_joints`` và quên ``geometry=``, nên nó không
nhìn thấy gói mà luật vừa gán — ``before`` và ``after`` bằng nhau theo cấu
trúc, và cổng báo PASS suốt. Mọi bản sao của đường runtime đều sẽ trôi, và
trôi về phía im lặng báo PASS.
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
    """Baseline là **19/28**, không phải 28/28 — và con số đó là phát hiện thật.

    Trước 2026-09-05 cổng tự dựng lại ROI bằng ``derive_solder_joints`` trần,
    nên nó bỏ luôn ``refine_to_metal`` (mặc định **True** trong runtime). Bật
    lại cho đúng đường runtime thì độ phủ tụt 28 -> 19: cùng **90 ROI**, nhưng
    refine co chúng về mảng kim loại và 9 pad rơi xuống dưới ngưỡng phủ 50%.

    Đo tách bạch trên chính fixture này::

        runtime refine=True  :  90 ROI, phủ 19/28 pad
        runtime refine=False :  90 ROI, phủ 28/28 pad

    **Đây là số liệu về bước 5.5, không phải về bộ luật**, nên nó không chặn
    cổng — cổng so trước/sau, và refine tác động như nhau lên cả hai vế. Nhưng
    nó đáng để riêng một dòng: 9/28 pad đếm tay tuột khỏi ROI sau khi refine là
    việc phải đi đo, không phải hằng số để sửa cho test xanh.
    """

    result = _board()
    assert result.pads_total == 28
    assert result.covered_before == 19, (
        "baseline trên đường runtime (có refine_to_metal) là 19/28; lệch nghĩa "
        "là fixture, 5.5, hoặc đường đo của cổng đã đổi"
    )


def test_losing_a_baseline_pad_fails_the_gate(monkeypatch) -> None:
    """Hợp đồng quan trọng nhất của cổng: mất pad ⇒ exit code khác 0.

    Test cũ của tôi ở đây là GIẢ — nó tự thêm vào ``result.lost`` rồi assert
    chính cái list đó, nên đột biến "``main`` luôn trả 0" không bị bắt. Bản này
    đi qua đúng ``main()``.
    """

    from scripts import evaluate_package_rule_gate as gate

    def _fake(path, families_mode, lead_model, leads_mode="model"):
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

    def _fake(path, families_mode, lead_model, leads_mode="model"):
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


# ------------------------------------------------- cổng phải NHÌN THẤY thay đổi


def test_the_gate_measures_on_the_same_path_as_the_runtime() -> None:
    """Hợp đồng đã hỏng một lần: cổng phải dựng ROI bằng ĐÚNG hàm của 5.5.

    Bản đầu tự lặp lại ``derive_solder_joints`` và quên truyền ``geometry=``,
    nên nó lấy ``terminal_geometry(detection.label)`` và bỏ qua sạch
    ``terminal_geometry_override`` mà luật vừa ghi. Hậu quả: ``before`` và
    ``after`` bằng nhau **theo cấu trúc**, cổng luôn báo PASS, và bản báo cáo
    "0 mất pad" ngày 2026-09-05 hoá ra không đo gì cả.

    Ở đây dùng ``ic_khong_chan`` vì nó là ca cực đoan nhất — ``PadProfile(0, 0)``
    nghĩa là **không ROI nào**. Cổng nào không thấy nổi thay đổi này thì không
    thấy nổi thay đổi nào.
    """

    import numpy as np
    from dataclasses import replace as _replace

    from aoi_pipeline import BoundingBox, Detection, SolderJointConfig
    from scripts.evaluate_package_rule_gate import _rois

    rng = np.random.default_rng(0)
    image = (rng.random((300, 300, 3)) * 60 + 40).astype("uint8")
    body = Detection("ic", 0.9, BoundingBox(100, 100, 200, 200), detection_id="B0")
    config = SolderJointConfig()

    before = _rois(image, [body], config)
    assert before, "thân này phải sinh ra ROI khi chưa có gói nào"

    hidden = _replace(
        body, metadata={"terminal_geometry_override": "hidden_terminals"}
    )
    assert _rois(image, [hidden], config) == [], (
        "gói ẩn chân KHÔNG sinh ROI. Cổng vẫn thấy ROI ở đây nghĩa là nó đang "
        "đọc nhãn detector chứ không đọc gói — nó mù với chính thứ nó phải đo"
    )


def test_the_gate_reports_a_geometry_change_end_to_end(tmp_path) -> None:
    """Cùng hợp đồng, nhưng đi qua trọn ``evaluate_board()``.

    Dùng họ ``connector``: luật ánh xạ thẳng ``connector -> connector`` mà
    **không cần chân nào**, và ``connector_rows`` khác hẳn ``multi_pin`` mà
    nhãn detector sẽ trả. Nên đây là đường ngắn nhất để bắt cổng phải thấy một
    thay đổi hình học thật, không cần lead detector.
    """

    import cv2
    import numpy as np

    from aoi_pipeline import BoundingBox, Detection, SolderJointConfig
    from aoi_pipeline.imaging.preprocessing import ImagePreprocessor
    from aoi_pipeline.config import PreprocessConfig
    from aoi_pipeline.placement.footprints import profile_for_package_class
    from aoi_pipeline.solder.geometry import SolderJointCropper
    from dataclasses import replace as _replace

    rng = np.random.default_rng(1)
    raw = (rng.random((300, 400, 3)) * 60 + 40).astype("uint8")
    cv2.imwrite(str(tmp_path / "board.png"), raw)
    box = [120.0, 100.0, 280.0, 190.0]
    (tmp_path / "board.json").write_text(json.dumps({
        "image": "board.png",
        "detections": [{"label": "connector", "confidence": 0.9, "box": box}],
        "components": {"J1": {"pads": [[124, 104, 140, 120]]}},
    }), encoding="utf-8")

    result = evaluate_board(tmp_path / "board.json", "truth", None)
    assert result.by_package == {"connector": 1}, (
        "luật phải quyết được thân này; không thì test đo nhầm thứ khác"
    )

    # Con số đối chiếu lấy từ chính API runtime, không lặp lại phép tính của cổng.
    image = ImagePreprocessor(PreprocessConfig()).process(raw).image
    detection = Detection("connector", 0.9, BoundingBox(*box), detection_id="d0")
    annotated = _replace(detection, metadata={
        "terminal_geometry_override":
            profile_for_package_class("connector", source="package_rules")
            .terminal_geometry,
    })

    def _runtime(dets):
        return [
            j.bbox.as_xyxy()
            for j in SolderJointCropper(SolderJointConfig()).derive(image, dets)
            if j.kind == "joint"
        ]

    plain, overridden = _runtime([detection]), _runtime([annotated])
    assert plain != overridden, (
        "mẫu này không tạo ra thay đổi hình học nào, nên nó không kiểm được gì "
        "— sửa mẫu, đừng nới khẳng định"
    )

    # Đối chiếu bằng DIỆN TÍCH chứ không bằng SỐ ROI: ``connector_rows`` và
    # ``multi_pin`` ở đây cùng cho 2 ROI, nên đếm số thì hai đường sai khác nhau
    # vẫn trông giống hệt. Đúng loại tín hiệu quá yếu đã để lọt lỗi lần đầu.
    def _area(boxes):
        return sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes)

    assert result.roi_area_after == pytest.approx(_area(overridden)), (
        f"cổng báo diện tích {result.roi_area_after:.1f}, runtime dựng "
        f"{_area(overridden):.1f} — cổng đang chạy một đường khác 5.5"
    )
    assert result.roi_area_before == pytest.approx(_area(plain))
    assert result.roi_area_after != pytest.approx(result.roi_area_before), (
        "luật đổi connector từ multi_pin sang connector_rows mà cổng thấy y "
        "nguyên thì nó không đo được gì"
    )


def test_hand_pads_as_lead_source_finally_reach_the_ic_branch() -> None:
    """``--leads truth``: nguồn chân thứ hai, và là thứ mở được nhánh ``ic``.

    Đo được, trên chính fixture này:

    ==========================  ==========  ==========
    nguồn chân                  ngoài thân  trong thân
    ==========================  ==========  ==========
    pass 2 (lead detector)      42          **18**
    pad đếm tay của fixture     24          **4**
    ==========================  ==========  ==========

    ``_edge_of`` đòi TÂM chân nằm ngoài hộp thân, nên chân của pass 2 gần như
    không đóng góp cạnh nào và nhánh ``ic`` **chưa từng chạy tới** — suốt thời
    gian đó ai cũng tưởng nguyên nhân là quy ước box cũ của fixture. Không
    phải: pad đếm tay trên **cùng những hộp thân đó** lại nằm ngoài 24/28.
    Nguyên nhân là bước 2 cắt một cửa sổ QUANH linh kiện rồi tìm mối hàn bên
    trong, nên mối hàn nó trả về nằm đè lên thân.

    Chế độ này vì thế đo **LOGIC của luật với bằng chứng chân hoàn hảo**, đúng
    kiểu ``--families truth`` đo luật tách khỏi lỗi 6.1. Nó **không** thay được
    ``--leads model`` cho quyết định bật luật.
    """

    truth = evaluate_board(FIXTURES / "board_smd_00001.json", "truth", None,
                           "truth")
    assert truth.leads_found == 28, "phải dùng đúng 28 pad đếm tay làm chân"
    assert truth.leads_inside_body < 10, (
        f"pad đếm tay nằm trong thân {truth.leads_inside_body}/28 — cao thế "
        "này thì nhánh `ic` vẫn không chạy được, và fixture cần khoanh lại"
    )
    assert any("§8.2" in reason for reason in truth.abstain_reasons), (
        "phải có ít nhất một con ic đi tới được nhánh hai-cạnh-đối rồi bị chốt "
        f"§8.2 chặn; hiện có: {dict(truth.abstain_reasons)}"
    )


def test_the_gate_survives_a_console_that_cannot_encode_vietnamese() -> None:
    """Script phải tự cấu hình stdout, không dựa vào tác dụng phụ của ONNX.

    Console Windows mặc định cp1252. Trước đây script vẫn chạy — nhưng chỉ vì
    nạp ONNX Runtime cấu hình lại stdout hộ. Nên ngay khi thêm ``--leads truth``
    (không nạp model nào) thì nó đổ ở **dòng in đầu tiên**, trước cả khi đo gì.

    Bản đầu của test này là GIẢ: nó assert ``sys.stdout.encoding`` của chính
    tiến trình pytest, mà pytest đã bắt sẵn stdout thành UTF-8 — nên nó xanh dù
    script có reconfigure hay không. Bản này chạy script trong tiến trình RIÊNG
    với ``PYTHONIOENCODING=cp1252``, tức dựng lại đúng cái console đã làm nó đổ.
    """

    import os
    import subprocess
    import sys

    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    done = subprocess.run(
        [sys.executable, "scripts/evaluate_package_rule_gate.py",
         str(FIXTURES), "--leads", "truth"],
        cwd=Path(__file__).resolve().parents[1],
        env=env, capture_output=True, text=True, encoding="utf-8",
        errors="replace",
    )
    assert "UnicodeEncodeError" not in done.stderr, (
        "console cp1252 làm script đổ; nó phải tự reconfigure stdout sang UTF-8 "
        f"thay vì dựa vào việc nạp ONNX. stderr: {done.stderr[-800:]}"
    )
    assert done.returncode == 0, done.stderr[-800:]
