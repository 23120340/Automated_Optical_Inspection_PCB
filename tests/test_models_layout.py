"""Bố cục ``models/`` phải khớp với thứ registry thật sự nạp.

Vì sao cần file này: trước khi có nó, ba thứ đã trôi khỏi nhau mà không gì báo.

* ``models/active/solder/defect/`` chứa một model mà **không ô nào nạp** — nằm
  trong thư mục có nghĩa là "app tự nạp".
* ``STAGE_FOLDERS["package_classifier"]`` trỏ vào ``active/package``, mà thư
  mục đó đã bị xoá ở một commit khác.
* ``models/README.md`` mô tả ``active/solder/detector/`` và khẳng định
  "Runtime chỉ dùng cặp ONNX + manifest" ở đó — sai cả tên lẫn nội dung, vì
  runtime không dùng gì cả.

Ba cái đó đều là tài liệu/cấu hình nói một đằng, đĩa một nẻo. Test đọc đĩa.
"""

from __future__ import annotations

from aoi_pipeline.modelops.model_registry import (
    ACTIVE_ROOT,
    ARCHIVE_ROOT,
    STAGE_FOLDERS,
    discover_models,
)


def test_every_declared_slot_has_a_folder() -> None:
    """Ô khai trong ``STAGE_FOLDERS`` mà không có thư mục là ô chết.

    Người dùng đọc tài liệu, mở đường dẫn, không thấy gì, và không có cách nào
    biết là mình sai hay repo sai.
    """

    missing = {
        kind: folder
        for kind, folder in STAGE_FOLDERS.items()
        if not (ACTIVE_ROOT / folder).is_dir()
    }
    assert not missing, f"ô khai nhưng không có thư mục: {missing}"


def test_the_folder_name_is_the_role_name() -> None:
    """Một cấp, và tên thư mục = tên vai trò.

    Trước đây ``solder_classifier`` nằm ở ``solder/classifier`` còn
    ``package_classifier`` ở ``package``. Hai quy ước khác nhau trong cùng một
    thư mục là lý do người đọc phải mở registry mới biết cái nào là cái nào.
    """

    mismatched = {
        kind: folder for kind, folder in STAGE_FOLDERS.items() if folder != kind
    }
    assert not mismatched, f"tên thư mục lệch tên vai trò: {mismatched}"

    nested = {
        kind: folder for kind, folder in STAGE_FOLDERS.items() if "/" in folder
    }
    assert not nested, f"ô lồng nhiều cấp: {nested}"


def test_nothing_sits_in_active_without_a_slot_to_load_it() -> None:
    """``active/`` nghĩa là *app tự nạp*. Một artifact không ô nào nạp mà nằm
    đó thì cái tên đang nói dối — và đúng chuyện đó đã xảy ra với model dò lỗi
    toàn board, nó ở lại ``active/solder/defect/`` sau khi ô của nó bị gỡ."""

    declared = {folder.split("/")[0] for folder in STAGE_FOLDERS.values()}
    stray = [
        path.relative_to(ACTIVE_ROOT).as_posix()
        for path in ACTIVE_ROOT.rglob("*.onnx")
        if path.relative_to(ACTIVE_ROOT).parts[0] not in declared
    ]
    assert not stray, (
        f"artifact trong active/ mà không ô nào nạp: {stray}. "
        "Chuyển sang models/archive/ chứ đừng để trong active/."
    )


def test_the_whole_board_defect_models_are_archived_not_active() -> None:
    """Hướng dò lỗi toàn board đã bị gỡ khỏi pipeline và app.

    ``pipeline.py`` và ``app/`` không gọi ``defect_detection`` ở đâu cả, nên
    hai artifact này phải nằm ở ``archive/`` — nơi định nghĩa là "không bao giờ
    tự nạp".
    """

    segmenters = list(discover_models("solder_segmenter"))
    assert segmenters, "không thấy artifact toàn board nào; nếu đã xoá thì bỏ test này"
    origins = {entry.origin for entry in segmenters}
    assert origins == {"archive"}, (
        f"artifact dò lỗi toàn board đang ở {origins}, phải ở archive"
    )


def test_each_active_model_ships_with_its_manifest() -> None:
    """Thiếu manifest thì ô đó từ chối nạp, nên một ONNX trần trong ``active/``
    là một ô hỏng chứ không phải một ô sẵn sàng."""

    orphans = [
        path.relative_to(ACTIVE_ROOT).as_posix()
        for path in ACTIVE_ROOT.rglob("*.onnx")
        if not (path.parent / "model_manifest.json").is_file()
    ]
    assert not orphans, f"ONNX không có manifest bên cạnh: {orphans}"


def test_archive_folders_follow_one_naming_convention() -> None:
    """``models/README.md`` khai quy ước ``<vai-trò>-<kiến-trúc>[-<nguồn>]-ver<N>``.

    Một thư mục đặt theo kiểu khác (``solder_segmenter_yolov8m_20260824``) không
    làm hỏng gì — bộ chọn đọc manifest chứ không đọc tên — nhưng nó làm người
    mở File Explorer tưởng đó là loại artifact khác.
    """

    offenders = [
        path.name
        for path in ARCHIVE_ROOT.iterdir()
        if path.is_dir() and not path.name.split("-")[-1].startswith("ver")
    ]
    assert not offenders, (
        f"thư mục archive không theo quy ước -ver<N>: {offenders}"
    )
