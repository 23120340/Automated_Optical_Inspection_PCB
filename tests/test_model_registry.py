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
