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

import aoi_pipeline.modelops.model_registry as model_registry
from aoi_pipeline.modelops.model_registry import (
    ModelEntry,
    ModelFolderRenameError,
    ModelSummary,
    discover_models,
    find_active,
    rename_model_folder,
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
        pytest.param(
            {
                "model": {"architecture": "yolov8m-seg"},
                "created_at": "2026-08-24T01:59:35+00:00",
                "reported_metrics": {"map50_box": 0.56836, "map50_mask": 0.5573},
            },
            "yolov8m-seg", "2026-08-24", "mask mAP50 0.557",
            id="solder-detector-segmentation",
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

    for kind in ("detector", "classifier"):
        for entry in discover_models(kind):
            assert entry.kind in (kind, "unknown"), (
                f"bộ chọn {kind} chào một model loại {entry.kind}: {entry.name}"
            )
    for kind in ("solder_classifier", "lead_detector"):
        for entry in discover_models(kind):
            assert entry.kind == kind, (
                f"bộ chọn {kind} chào một model loại {entry.kind}: {entry.name}"
            )


def test_the_stage_is_taken_from_the_manifest_not_the_folder_name(tmp_path) -> None:
    """Tên thư mục do người đặt và có thể sai; `task` do notebook sinh ra cùng
    lúc với trọng số."""

    from aoi_pipeline.modelops.model_registry import _kind_of

    assert _kind_of(
        {"task": "solder_defect_classification"}, "ten-lung-tung"
    ) == "solder_classifier"
    assert _kind_of(
        {"task": "solder_defect_instance_segmentation"}, "ten-lung-tung"
    ) == "solder_segmenter"
    assert _kind_of({"task": "component_family_classification"}, "abc") == "classifier"
    assert _kind_of({"task": "detect"}, "abc") == "detector"
    # Không có task thì mới nhìn tên thư mục.
    assert _kind_of(None, "detector-yolo26s-20260817") == "detector"
    assert _kind_of(None, "solder/classifier") == "solder_classifier"
    assert _kind_of(None, "solder\\detector") == "solder_segmenter"
    assert _kind_of(None, "solder/classifier/detector") == "unknown"
    assert _kind_of(None, "khong-goi-y-gi") == "unknown"


def test_solder_schema_fallbacks_are_role_specific() -> None:
    from aoi_pipeline.modelops.model_registry import _kind_of

    assert _kind_of(
        {"schema_version": "pcb-solder-defect-classifier/1.0"}, "khong-goi-y"
    ) == "solder_classifier"
    assert _kind_of(
        {"schema_version": "aoi-external-yolo-segmentation/1.0"}, "khong-goi-y"
    ) == "solder_segmenter"
    assert _kind_of(
        {"schema_version": "pcb-solder/1.0"}, "khong-goi-y"
    ) == "unknown"


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

    for kind in ("detector", "classifier", "solder_classifier", "lead_detector"):
        entry = find_active(kind)
        assert entry is not None, f"không thấy model mặc định cho {kind}"
        assert entry.model_path.is_file()


def test_legacy_solder_alias_resolves_only_to_the_classifier() -> None:
    legacy = find_active("solder")
    classifier = find_active("solder_classifier")

    assert legacy == classifier
    assert legacy is not None
    assert legacy.kind == "solder_classifier"
    assert legacy.name == "solder_classifier/best.onnx"
    assert discover_models("solder") == discover_models("solder_classifier")


def test_solder_pickers_exclude_the_other_role_and_unknown_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = {
        "active": tmp_path / "active",
        "library": tmp_path / "library",
        "archive": tmp_path / "archive",
    }
    for root in roots.values():
        root.mkdir()
    monkeypatch.setattr(model_registry, "ACTIVE_ROOT", roots["active"])
    monkeypatch.setattr(model_registry, "LIBRARY_ROOT", roots["library"])
    monkeypatch.setattr(model_registry, "ARCHIVE_ROOT", roots["archive"])

    for folder, task in (
        ("ten-goi-nhu-detector", "solder_defect_classification"),
        ("ten-goi-nhu-classifier", "solder_defect_instance_segmentation"),
        ("khong-ro-loai", "mot_task_khong_biet"),
    ):
        directory = roots["library"] / folder
        directory.mkdir()
        (directory / "best.onnx").write_bytes(b"x")
        (directory / "model_manifest.json").write_text(
            json.dumps({"task": task}), encoding="utf-8"
        )

    classifier_entries = discover_models("solder_classifier")
    detector_entries = discover_models("solder_segmenter")

    assert [entry.name for entry in classifier_entries] == [
        "ten-goi-nhu-detector/best.onnx"
    ]
    assert [entry.kind for entry in classifier_entries] == ["solder_classifier"]
    assert [entry.name for entry in detector_entries] == [
        "ten-goi-nhu-classifier/best.onnx"
    ]
    assert [entry.kind for entry in detector_entries] == ["solder_segmenter"]


def test_find_active_rejects_a_manifest_from_the_other_solder_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    active = tmp_path / "active"
    directory = active / "solder" / "detector"
    directory.mkdir(parents=True)
    (directory / "best.onnx").write_bytes(b"x")
    (directory / "model_manifest.json").write_text(
        json.dumps({"task": "solder_defect_classification"}), encoding="utf-8"
    )
    monkeypatch.setattr(model_registry, "ACTIVE_ROOT", active)

    assert find_active("solder_segmenter") is None


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
    app.run()

    assert not app.exception
    for key, folder in (
        ("component_model_name", "detector"),
        ("classifier_model_name", "classifier"),
        ("solder_model_name", "solder_classifier"),
    ):
        assert app.session_state[key], f"{key} vẫn trống sau khi khởi tạo"
        assert folder in app.session_state[key]


def test_the_picker_shows_the_active_model_as_chosen_not_as_unused() -> None:
    """Ô chọn ghi "— không dùng —" trong khi model đang được nạp là nói dối
    người dùng về trạng thái của hệ thống."""

    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=180)
    app.run()

    choices = {s.key: s.value for s in app.sidebar.selectbox}
    for key in (
        "component_model_choice",
        "classifier_model_choice",
        "solder_model_choice",
    ):
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
    app.run()
    assert app.session_state["classifier_model_name"]        # đã tự nạp

    # Giả lập _set_model: một nguồn KHÁC ô chọn đặt lại đường dẫn.
    app.session_state["classifier_model_path"] = "/tmp/toi-vua-tai-len.onnx"
    app.session_state["classifier_model_name"] = "toi-vua-tai-len.onnx"
    app.run()

    assert app.session_state["classifier_model_name"] == "toi-vua-tai-len.onnx", (
        "ô chọn đã ghi đè lên model vừa tải lên"
    )


# --------------------------------------------------------------------------
# Đổi tên thư mục model từ bộ chọn
# --------------------------------------------------------------------------


@pytest.fixture
def isolated_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    roots = {
        "active": tmp_path / "active",
        "archive": tmp_path / "archive",
        "library": tmp_path / "library",
    }
    for root in roots.values():
        root.mkdir()
    monkeypatch.setattr(model_registry, "ACTIVE_ROOT", roots["active"])
    monkeypatch.setattr(model_registry, "ARCHIVE_ROOT", roots["archive"])
    monkeypatch.setattr(model_registry, "LIBRARY_ROOT", roots["library"])
    return roots


def _registry_entry(root: Path, *, origin: str, folder: str = "ten-cu") -> ModelEntry:
    directory = root / folder
    directory.mkdir(parents=True)
    model = directory / "best.onnx"
    model.write_bytes(b"onnx-payload")
    manifest = directory / "model_manifest.json"
    manifest.write_text(
        json.dumps({"task": "solder_defect_classification", "model": {"architecture": "yolo"}}),
        encoding="utf-8",
    )
    (directory / "ghi-chu.txt").write_text("giữ nguyên", encoding="utf-8")
    return ModelEntry(
        name=f"{folder}/best.onnx",
        kind="solder_classifier",
        model_path=model,
        manifest_path=manifest,
        origin=origin,
    )


@pytest.mark.parametrize("origin", ["library", "archive"])
def test_renaming_a_model_moves_the_whole_folder_and_rediscovers_it(
    isolated_registry: dict[str, Path], origin: str
) -> None:
    entry = _registry_entry(isolated_registry[origin], origin=origin)

    renamed = rename_model_folder(entry, "Model mối hàn tốt")

    assert not (isolated_registry[origin] / "ten-cu").exists()
    destination = isolated_registry[origin] / "Model mối hàn tốt"
    assert renamed.model_path == destination / "best.onnx"
    assert renamed.manifest_path == destination / "model_manifest.json"
    assert renamed.name == "Model mối hàn tốt/best.onnx"
    assert renamed.model_path.read_bytes() == b"onnx-payload"
    assert (destination / "ghi-chu.txt").read_text(encoding="utf-8") == "giữ nguyên"

    discovered = discover_models("solder")
    assert [(item.origin, item.name) for item in discovered] == [
        (origin, "Model mối hàn tốt/best.onnx")
    ]


def test_renaming_never_overwrites_an_existing_folder(
    isolated_registry: dict[str, Path]
) -> None:
    root = isolated_registry["library"]
    entry = _registry_entry(root, origin="library")
    occupied = root / "da-co"
    occupied.mkdir()
    marker = occupied / "khong-duoc-mat.txt"
    marker.write_text("safe", encoding="utf-8")

    with pytest.raises(ModelFolderRenameError, match="đã tồn tại"):
        rename_model_folder(entry, "da-co")

    assert entry.model_path.is_file()
    assert marker.read_text(encoding="utf-8") == "safe"


def test_case_only_rename_is_supported(isolated_registry: dict[str, Path]) -> None:
    root = isolated_registry["library"]
    entry = _registry_entry(root, origin="library", folder="Solder-AOI")

    renamed = rename_model_folder(entry, "solder-aoi")

    assert renamed.model_path == root / "solder-aoi" / "best.onnx"
    assert renamed.model_path.is_file()
    assert [child.name for child in root.iterdir()] == ["solder-aoi"]


@pytest.mark.parametrize(
    "bad_name",
    [
        "",
        " ",
        ".",
        "..",
        "../thoat-ra",
        "mot/hai",
        "mot\\hai",
        "C:\\model",
        "sai|ten",
        "sai*ten",
        "CON",
        "nul.json",
        "COM1",
        "LPT9.txt",
        "CONIN$",
        "conout$.txt",
        "co-dau-cham.",
        "co-khoang-trang ",
        "\x00",
    ],
)
def test_invalid_or_ambiguous_folder_names_are_rejected(
    isolated_registry: dict[str, Path], bad_name: str
) -> None:
    entry = _registry_entry(isolated_registry["library"], origin="library")

    with pytest.raises(ModelFolderRenameError):
        rename_model_folder(entry, bad_name)

    assert entry.model_path.is_file()


def test_active_model_folders_cannot_be_renamed(isolated_registry: dict[str, Path]) -> None:
    entry = _registry_entry(isolated_registry["active"], origin="active", folder="solder")

    with pytest.raises(ModelFolderRenameError, match="active"):
        rename_model_folder(entry, "ten-moi")

    assert entry.model_path.is_file()


def test_a_forged_entry_outside_the_claimed_registry_is_rejected(
    isolated_registry: dict[str, Path], tmp_path: Path
) -> None:
    entry = _registry_entry(tmp_path / "ngoai", origin="library")

    with pytest.raises(ModelFolderRenameError, match="ngoài registry"):
        rename_model_folder(entry, "ten-moi")

    assert entry.model_path.is_file()


def test_a_root_level_model_cannot_rename_the_registry_itself(
    isolated_registry: dict[str, Path]
) -> None:
    root = isolated_registry["library"]
    model = root / "best.onnx"
    model.write_bytes(b"x")
    entry = ModelEntry(
        name="best.onnx",
        kind="unknown",
        model_path=model,
        manifest_path=None,
        origin="library",
    )

    with pytest.raises(ModelFolderRenameError, match="chính thư mục"):
        rename_model_folder(entry, "khong-duoc")

    assert model.is_file()


def test_removing_a_model_sticks() -> None:
    """Đây chính là lý do lần trước việc gán sẵn model bị bỏ đi — xem
    ``test_no_model_artifact_is_seeded_from_disk``. Nạp mặc định ở MỌI lần chạy
    lại thì nút "Gỡ model" trông như hỏng: bấm xong nó quay lại ngay.

    Nên phải nạp một lần cho mỗi phiên, không phải mỗi lần rerun.
    """

    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=180)
    app.run()
    assert app.session_state["component_model_name"]

    app.session_state["component_model_path"] = None
    app.session_state["component_model_name"] = None
    app.run()

    assert app.session_state["component_model_name"] is None, (
        "model đã gỡ lại được nạp về ở lần chạy lại kế tiếp"
    )


# --------------------------------------------------------------------------- #
# Bước 6.2 nhận được hai hình thái model, và ô chứa chúng chỉ có một tên
# --------------------------------------------------------------------------- #


def test_the_score_of_a_single_head_detector_is_read(tmp_path: Path) -> None:
    """Model 6.2 chỉ có một đầu ra nên ghi ``map50`` trần, không hậu tố.

    Thiếu đường đọc đó thì bảng in "—" cho model ĐANG CHẠY trong khi bản cũ kém
    hơn lại khoe được điểm, và người chọn tưởng bản mới chưa từng được đo.
    """

    manifest = {
        "task": "solder_defect_detection",
        "model": {"architecture": "yolo11s"},
        "created_at": "2026-08-26T00:00:00+00:00",
        "reported_metrics": {"map50": 0.8561, "map50_95": 0.4036},
    }
    summary = _entry(tmp_path, manifest).summary()
    assert summary.metric == "mAP50 0.856"


def test_a_mask_score_still_wins_over_the_plain_one(tmp_path: Path) -> None:
    """Bản segment ghi cả ba khoá; ``map50`` trần không được che mất mask."""

    manifest = {
        "model": {"architecture": "yolov8m-seg"},
        "reported_metrics": {"map50": 0.60, "map50_box": 0.568, "map50_mask": 0.5573},
    }
    assert _entry(tmp_path, manifest).summary().metric == "mask mAP50 0.557"


def test_the_summary_carries_the_task_so_two_shapes_can_be_told_apart(
    tmp_path: Path,
) -> None:
    """Ô ``solder_segmenter`` nhận cả detect lẫn segment kể từ khi hợp đồng được
    mở, và tên ô không nói được đang là cái nào -- nhưng đổi giữa hai thứ đó đổi
    luôn hành vi của bước 6.2."""

    detect = _entry(tmp_path, {"task": "solder_defect_detection"}, folder="d").summary()
    segment = _entry(
        tmp_path, {"task": "solder_defect_instance_segmentation"}, folder="s"
    ).summary()
    assert detect.task == "solder_defect_detection"
    assert segment.task == "solder_defect_instance_segmentation"
    assert detect.task != segment.task


def test_the_task_stays_out_of_the_picker_label(tmp_path: Path) -> None:
    """Cùng lý do với ``sha256``: nhãn bộ chọn không được dài thêm."""

    summary = _entry(tmp_path, {"task": "solder_defect_detection",
                                "model": {"architecture": "yolo11s"}}).summary()
    assert "solder_defect_detection" not in summary.as_line()


def test_an_empty_summary_still_compares_equal() -> None:
    """``task`` phải có mặc định, nếu không mọi so sánh cũ đều gãy."""

    assert ModelSummary() == ModelSummary()


@pytest.mark.parametrize(
    "hint, expected",
    [
        # Tên thư mục do chính registry đặt phải đọc ngược được
        ("solder/segmenter", "solder_segmenter"),
        ("solder/classifier", "solder_classifier"),
        # Task do notebook của dự án sinh ra
        ("solder_defect_detection", "solder_segmenter"),
        ("solder_defect_instance_segmentation", "solder_segmenter"),
        ("solder_defect_classification", "solder_classifier"),
        # Schema là hợp đồng, nhận diện được kể cả khi nằm ngoài thư mục solder
        ("aoi-solder-defect-detection/1.0", "solder_segmenter"),
        # Không có gợi ý nào thì phải im lặng, không đoán
        ("solder", None),
        ("classifier", None),
    ],
)
def test_the_solder_role_resolver_reads_its_own_vocabulary(
    hint: str, expected: str | None
) -> None:
    """``segmenter`` là tên thư mục của chính dự án và từng KHÔNG có trong danh
    sách token, nên resolver không đọc ngược được đường dẫn nó tự ghi ra."""

    assert model_registry._solder_role_from_hint(hint) == expected


def test_adding_defect_as_a_token_would_break_the_classifier() -> None:
    """Ghi lại một cám dỗ đã bị bác: ``defect`` có mặt trong CẢ HAI task, nên
    thêm nó vào danh sách detector sẽ làm hai hint cùng đúng và resolver phải
    im lặng ở đúng vai trò nó đang giải đúng."""

    assert "defect" in "solder_defect_classification"
    assert model_registry._solder_role_from_hint("solder_defect_classification") == (
        "solder_classifier"
    )


def test_every_task_the_project_emits_maps_to_a_slot() -> None:
    """Một task thiếu ở đây nghĩa là một model thả vào ``models/library`` không
    được phân loại -- đúng luồng "của bạn" mà bộ chọn quảng cáo."""

    emitted = {
        "component_family_classification",
        "solder_defect_classification",
        "solder_defect_detection",
        "solder_defect_instance_segmentation",
        "component_detection",
        "component_and_lead_detection",
    }
    missing = emitted - set(model_registry._TASK_TO_KIND)
    assert not missing, f"task chưa có trong _TASK_TO_KIND: {sorted(missing)}"


def test_the_pass_two_lead_detector_has_a_slot_of_its_own() -> None:
    """Ba ô model cũ đều trả lời câu khác. ``detector`` nhìn cả board và học
    thân linh kiện; ``solder_segmenter`` khoanh LỖI và không có lớp nào cho mối
    hàn lành. Lượt 2 cần model khoanh MỌI mối hàn, kể cả lành, nên nó là ô riêng
    -- và nếu không có ô, model chỉ có thể bị nhét vào một ô sai."""

    assert model_registry.STAGE_FOLDERS["lead_detector"] == "lead_detector"
    assert model_registry._TASK_TO_KIND["solder_joint_localization"] == "lead_detector"


def test_a_joint_locator_is_not_mistaken_for_a_solder_role() -> None:
    """``solder_joint_localization`` chứa chữ "solder", nên nó đi ngang qua
    ``_solder_role_from_hint``. Hàm đó phải TRỪU chứ không được đoán, nếu không
    model lượt 2 sẽ hiện trong bộ chọn của bước 6.2."""

    assert model_registry._solder_role_from_hint("solder_joint_localization") is None


@pytest.mark.skipif(
    not (PROJECT_ROOT / "models/active/lead_detector/model_manifest.json").is_file(),
    reason="chưa cài model lượt 2",
)
def test_the_installed_lead_detector_admits_it_is_not_production_ready() -> None:
    """Đo trên board thật: hình học 5.5 đạt 0/28 pad bỏ sót, bật model này lên
    thành 2/28. Manifest phải nói ra điều đó, vì bảng chọn chỉ hiện mAP50 0.871
    và con số đó một mình sẽ mời người ta bật nó lên."""

    manifest = json.loads(
        (PROJECT_ROOT / "models/active/lead_detector/model_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["aoi_compatibility"]["safe_to_enable_in_production"] is False
    validation = manifest["on_board_validation"]
    # Bản thứ hai không còn làm mất pad nào (bản đầu mất 2), nên lý do "chưa bật
    # được" đổi: không phải lọt lưới nữa mà là ĐỘ PHỦ tụt và biên còn mỏng.
    assert validation["escapes_introduced"] == 0
    assert validation["previous_model_escapes"] == 2
    assert (
        validation["coverage_median_with_model"]
        < validation["coverage_median_geometry_only"]
    ), "số đo phải cho thấy vì sao chưa bật mặc định"
    assert validation["coverage_min_with_model"] > validation["coverage_gate"], (
        "nếu tụt dưới ngưỡng thì đây là ca lọt lưới, phải ghi là escapes"
    )
    assert validation["what_is_NOT_settled"], (
        "một phép đo có phần chưa ngã ngũ phải nói ra phần đó"
    )


def test_installing_a_lead_detector_does_not_switch_it_on() -> None:
    """Có mặt trong models/active KHÔNG được đồng nghĩa với đang chạy. Lượt 2 là
    opt-in qua lead_detection.model_path, và đó là thứ giữ cho pipeline không
    đổi hành vi khi một model chưa đạt được cài vào để thử."""

    from aoi_pipeline.config import PipelineConfig

    assert PipelineConfig().lead_detection.model_path is None
