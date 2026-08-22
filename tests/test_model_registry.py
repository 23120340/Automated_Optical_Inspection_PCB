"""Phân biệt được model nào với model nào.

Mọi artifact trong dự án đều tên `best.onnx` — quy ước của Ultralytics và của
các notebook. Nên tên file không nói được gì, và bộ chọn từng hiện

    classifier/best.onnx (đang dùng)
    classifier_v2/best.onnx (của bạn)

cho hai model khác hẳn nhau về kiến trúc, ngày và điểm số. Thông tin ấy nằm sẵn
trong `model_manifest.json`; việc của registry là đọc ra.

Chỗ khó: bốn artifact của dự án dùng **bốn schema khác nhau**. `model.architecture`
với `base_model`, `created_at` với `created_at_utc`, và điểm đầu bảng nằm ở
`metrics.*`, `metrics.val.*` hoặc `training.*` tuỳ file. Một model tải từ ngoài
về sẽ dùng cách thứ năm. Nên phải đọc theo danh sách ứng viên, không ép một
schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aoi_pipeline.model_registry import (
    ModelEntry,
    ModelSummary,
    discover_models,
    find_active,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _entry(tmp_path: Path, manifest: dict | None, *, folder: str = "m") -> ModelEntry:
    directory = tmp_path / folder
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / "best.onnx"
    model.write_bytes(b"not a real onnx")
    manifest_path = None
    if manifest is not None:
        manifest_path = directory / "model_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return ModelEntry(name=f"{folder}/best.onnx", kind="classifier",
                      model_path=model, manifest_path=manifest_path, origin="library")


# --------------------------------------------------------------------------
# Đọc được cả bốn schema
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "manifest, architecture, created, metric",
    [
        pytest.param(
            {"model": {"architecture": "convnext_base"}, "created_at": "2026-08-22T01:01:15",
             "training": {"test_accuracy": 0.9539}},
            "convnext_base", "2026-08-22", "acc 0.954",
            id="classifier-v2",
        ),
        pytest.param(
            {"model": {"architecture": "efficientnet_b0"}, "created_at": "2026-08-18T06:22:11",
             "metrics": {"accuracy": 0.958}},
            "efficientnet_b0", "2026-08-18", "acc 0.958",
            id="classifier-cu",
        ),
        pytest.param(
            {"model": {"architecture": "yolo26s.pt"}, "created_at": "2026-08-20T23:08:28",
             "metrics": {"map50": 0.5052}},
            "yolo26s", "2026-08-20", "mAP50 0.505",
            id="detector-v2",
        ),
        pytest.param(
            {"base_model": "yolo26s.pt", "created_at_utc": "2026-08-17T06:54:32",
             "metrics": {"val": {"map50": 0.5788}}},
            "yolo26s", "2026-08-17", "mAP50 0.579",
            id="detector-ver1-schema-cu",
        ),
    ],
)
def test_the_summary_reads_every_schema_the_project_has(
    tmp_path, manifest, architecture, created, metric
) -> None:
    summary = _entry(tmp_path, manifest).summary()
    assert summary.architecture == architecture
    assert summary.created == created
    assert summary.metric == metric


def test_the_pt_suffix_is_stripped_from_the_architecture(tmp_path) -> None:
    """`yolo26s.pt` là tên file trọng số gốc, không phải tên kiến trúc."""

    summary = _entry(tmp_path, {"model": {"architecture": "yolo26s.pt"}}).summary()
    assert summary.architecture == "yolo26s"


def test_a_manifest_with_nothing_useful_degrades_quietly(tmp_path) -> None:
    """Model lạ vẫn phải hiện trong bộ chọn, chỉ là không có gì để khoe."""

    summary = _entry(tmp_path, {"schema_version": "gi-do/1.0"}).summary()
    assert summary.architecture is None
    assert summary.metric is None
    assert summary.as_line() == ""


def test_a_corrupt_manifest_does_not_take_the_picker_down(tmp_path) -> None:
    directory = tmp_path / "hong"
    directory.mkdir()
    (directory / "best.onnx").write_bytes(b"x")
    (directory / "model_manifest.json").write_text("{ khong phai json", encoding="utf-8")
    entry = ModelEntry(name="hong/best.onnx", kind="classifier",
                       model_path=directory / "best.onnx",
                       manifest_path=directory / "model_manifest.json",
                       origin="library")
    assert entry.manifest() is None
    assert entry.summary() == ModelSummary()


# --------------------------------------------------------------------------
# Nhãn hiển thị
# --------------------------------------------------------------------------


def test_the_label_says_more_than_best_onnx(tmp_path) -> None:
    """Test này tồn tại vì nhãn cũ là `<thư mục>/best.onnx (đang dùng)`, giống
    hệt nhau cho mọi model."""

    entry = _entry(tmp_path, {
        "model": {"architecture": "convnext_base"},
        "created_at": "2026-08-22T01:01:15",
        "training": {"test_accuracy": 0.9539},
    }, folder="classifier-convnext_base-20260822")

    label = entry.label
    assert "convnext_base" in label
    assert "2026-08-22" in label
    assert "0.954" in label
    assert "của bạn" in label
    assert "best.onnx" not in label, "tên file không phân biệt được gì, đừng chiếm chỗ"


def test_two_models_of_the_same_stage_get_different_labels() -> None:
    """Điều thực sự quan trọng: bộ chọn phải phân biệt được chúng."""

    labels = [entry.label for entry in discover_models("classifier")]
    assert len(labels) == len(set(labels)), f"nhãn trùng nhau: {labels}"


# --------------------------------------------------------------------------
# Lọc theo loại -- lỗi thật, sửa 2026-08-22
# --------------------------------------------------------------------------


def test_a_picker_only_offers_models_for_its_own_stage() -> None:
    """Bộ lọc từng chỉ áp cho `active/`, nên bộ chọn model mối hàn 6.2 chào cả
    classifier lẫn detector. Nạp vào thì hỏng ở một chỗ chẳng liên quan gì tới
    nguyên nhân."""

    for kind in ("detector", "classifier", "solder"):
        for entry in discover_models(kind):
            assert entry.kind in (kind, "unknown"), (
                f"bộ chọn {kind} chào một model loại {entry.kind}: {entry.name}"
            )


def test_the_stage_is_taken_from_the_manifest_not_the_folder_name(tmp_path) -> None:
    """Tên thư mục do người đặt và có thể sai; `task` do notebook sinh ra cùng
    lúc với trọng số."""

    from aoi_pipeline.model_registry import _kind_of

    assert _kind_of({"task": "solder_defect_classification"}, "ten-lung-tung") == "solder"
    assert _kind_of({"task": "component_family_classification"}, "abc") == "classifier"
    assert _kind_of({"task": "detect"}, "abc") == "detector"
    # Không có task thì mới nhìn tên thư mục.
    assert _kind_of(None, "detector-yolo26s-20260817") == "detector"
    assert _kind_of(None, "khong-goi-y-gi") == "unknown"


# --------------------------------------------------------------------------
# Những gì thực sự nằm trên đĩa
# --------------------------------------------------------------------------


def test_every_shipped_model_can_describe_itself() -> None:
    """Không model nào trong repo được hiện lên như một dòng trống."""

    entries = discover_models()
    assert entries, "không tìm thấy model nào"
    for entry in entries:
        summary = entry.summary()
        assert summary.architecture, f"{entry.name} không khai kiến trúc"
        assert summary.created, f"{entry.name} không khai ngày tạo"


def test_the_default_for_each_stage_is_still_found() -> None:
    """`active/<bước>/best.onnx` là đường app tìm mặc định. Đổi tên các thư mục
    đó sẽ làm app không tìm thấy gì, và test này là thứ báo."""

    for kind in ("detector", "classifier", "solder"):
        entry = find_active(kind)
        assert entry is not None, f"không thấy model mặc định cho {kind}"
        assert entry.model_path.is_file()


# --------------------------------------------------------------------------
# `active/` phải thật sự tự nạp
#
# `find_active()` được viết ra cho việc này nhưng suốt một thời gian không ai
# gọi, nên mọi phiên mở ra với bộ chọn ở "— không dùng —" và bước 4 báo đang
# chạy CV demo — người dùng phải chọn tay đúng cái model đã được đặt làm mặc
# định. `models/README.md` thì ghi "active/ = model app tự nạp", tức tài liệu
# nói một đằng, code làm một nẻo.
# --------------------------------------------------------------------------


def test_a_fresh_session_already_has_the_active_models_loaded() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=180)
    app.session_state["workspace_mode"] = "pipeline_lab"
    app.run()

    assert not app.exception
    for key, folder in (
        ("component_model_name", "detector"),
        ("classifier_model_name", "classifier"),
        ("solder_model_name", "solder"),
    ):
        assert app.session_state[key], f"{key} vẫn trống sau khi khởi tạo"
        assert folder in app.session_state[key]


def test_the_picker_shows_the_active_model_as_chosen_not_as_unused() -> None:
    """Ô chọn ghi "— không dùng —" trong khi model đang được nạp là nói dối
    người dùng về trạng thái của hệ thống."""

    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=180)
    app.session_state["workspace_mode"] = "pipeline_lab"
    app.run()

    choices = {s.key: s.value for s in app.sidebar.selectbox}
    for key in ("component_model_choice", "classifier_model_choice",
                "solder_model_choice"):
        assert key in choices, f"thiếu bộ chọn {key}"
        assert "đang dùng" in choices[key], (
            f"{key} đang là {choices[key]!r}, phải là model trong active/"
        )


def test_a_model_the_user_picked_is_not_reset_on_the_next_rerun() -> None:
    """Tự điền chỉ được phép điền vào ô TRỐNG. Giật lựa chọn của người dùng về
    mặc định ở mỗi lần bấm nút thì tính năng chọn model thành vô dụng."""

    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=180)
    app.session_state["workspace_mode"] = "pipeline_lab"
    app.run()

    app.session_state["classifier_model_path"] = "/mot/duong/dan/khac.onnx"
    app.session_state["classifier_model_name"] = "cua-toi/best.onnx"
    app.run()

    assert app.session_state["classifier_model_name"] == "cua-toi/best.onnx"


def test_uploading_a_model_survives_the_next_rerun() -> None:
    """Kịch bản thật đứng sau bản sửa: ô chọn nhớ giá trị của nó qua các lần
    chạy lại và bỏ qua ``index``. Nếu nó cứ thấy lệch với session là áp đặt,
    thì file vừa tải lên bị nó giật về model cũ ngay lần rerun kế tiếp — mất
    im lặng, và người dùng chỉ thấy "sao model của tôi không có tác dụng".
    """

    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=180)
    app.session_state["workspace_mode"] = "pipeline_lab"
    app.run()
    assert app.session_state["classifier_model_name"]        # đã tự nạp

    # Giả lập _set_model: một nguồn KHÁC ô chọn đặt lại đường dẫn.
    app.session_state["classifier_model_path"] = "/tmp/toi-vua-tai-len.onnx"
    app.session_state["classifier_model_name"] = "toi-vua-tai-len.onnx"
    app.run()

    assert app.session_state["classifier_model_name"] == "toi-vua-tai-len.onnx", (
        "ô chọn đã ghi đè lên model vừa tải lên"
    )


def test_removing_a_model_sticks() -> None:
    """Đây chính là lý do lần trước việc gán sẵn model bị bỏ đi — xem
    ``test_no_model_artifact_is_seeded_from_disk``. Nạp mặc định ở MỌI lần chạy
    lại thì nút "Gỡ model" trông như hỏng: bấm xong nó quay lại ngay.

    Nên phải nạp một lần cho mỗi phiên, không phải mỗi lần rerun.
    """

    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=180)
    app.session_state["workspace_mode"] = "pipeline_lab"
    app.run()
    assert app.session_state["component_model_name"]

    app.session_state["component_model_path"] = None
    app.session_state["component_model_name"] = None
    app.run()

    assert app.session_state["component_model_name"] is None, (
        "model đã gỡ lại được nạp về ở lần chạy lại kế tiếp"
    )
