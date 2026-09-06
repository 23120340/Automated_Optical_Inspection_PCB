"""Ảnh lỗi phải là ảnh của ĐÚNG chỗ bị lỗi.

Kho đã ghi được vị trí (``test_storage.py``), nhưng yêu cầu chốt với dây chuyền
là *vị trí kèm ảnh*. Mảnh nối hai thứ đó là ``storage/capture.py``, và nó có một
cách hỏng mà không gì báo: toạ độ trong kết quả kiểm nằm trong hệ
``golden_board_pixels`` — tức hệ của ảnh **đã nắn** — nên cắt từ ảnh test **gốc**
bằng chính những toạ độ ấy vẫn ra một tấm ảnh hợp lệ, chỉ là ảnh của chỗ khác.

Nên bài test chính ở đây cố tình cho ảnh test **lệch** so với Golden, để hai
nguồn cắt cho ra hai kết quả khác nhau, rồi đòi đúng một trong hai.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from aoi_pipeline.golden.inspector import InspectionConfig
from aoi_pipeline.models import BoundingBox
from aoi_pipeline.storage import (
    DEFAULT_CROP_MARGIN,
    DefectRecord,
    InspectionStore,
    crop_defect_images,
    record_inspection,
)

from test_inspection import _inspector, _production_recipe, _shift_slot

SHIFT = (5.0, 3.0)


def _shifted_board(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Cả tấm ảnh lệch đi — đúng thứ bước nắn sinh ra để bù lại."""

    return cv2.warpAffine(
        image,
        np.float32([[1, 0, dx], [0, 1, dy]]),
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _padded(box, shape, margin: float = DEFAULT_CROP_MARGIN):
    b = BoundingBox(*box)
    pad_x, pad_y = b.width * margin, b.height * margin
    return (
        BoundingBox(b.x1 - pad_x, b.y1 - pad_y, b.x2 + pad_x, b.y2 + pad_y)
        .clamp(shape[1], shape[0])
        .to_int()
    )


def _decode(path: Path) -> np.ndarray:
    return cv2.imdecode(
        np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR
    )


def _ng_run(tmp_path: Path):
    """Một lần chạy NG thật, trên ảnh test lệch so với Golden."""

    golden, recipe, candidate = _production_recipe(tmp_path)
    slot = recipe.slots[0]
    defective = _shift_slot(golden, slot.expected_bbox_xyxy, 4.0, 0.0)
    test_image = _shifted_board(defective, *SHIFT)
    run = _inspector([candidate], config=InspectionConfig()).inspect(
        test_image, recipe, tmp_path
    )
    return recipe, test_image, run


def test_the_defect_crop_comes_from_the_aligned_image_not_the_raw_one(
    tmp_path: Path,
) -> None:
    """Bài test chính: cắt nhầm nguồn thì ảnh vẫn có, chỉ là sai chỗ.

    Ảnh test bị dịch cả tấm so với Golden, nên cắt từ ảnh gốc và cắt từ ảnh đã
    nắn cho ra hai vùng khác nhau. Ảnh lưu trong kho phải khớp bản **đã nắn**.
    """

    recipe, test_image, run = _ng_run(tmp_path)
    assert run.status == "ng", "fixture phải sinh ra một lỗi thật để mà chụp"
    aligned = run.alignment.image
    assert aligned is not None

    with InspectionStore(tmp_path / "kho") as store:
        board = store.ensure_board(serial="SN-CROP-1")
        inspection = store.load_inspection(
            record_inspection(store, run, recipe, board_id=board, event_id="EV-1")
        )
        defect = inspection.defects[0]
        assert defect["image_sha256"], "lỗi có toạ độ thì phải có ảnh"
        stored = _decode(store.image_file(defect["image_sha256"]))

    box = (defect["x1"], defect["y1"], defect["x2"], defect["y2"])
    x1, y1, x2, y2 = _padded(box, aligned.shape)
    from_aligned = aligned[y1:y2, x1:x2]
    from_raw = test_image[y1:y2, x1:x2]

    assert np.array_equal(stored, from_aligned)
    # Nếu hai nguồn tình cờ giống nhau thì bài test không chứng minh được gì.
    assert not np.array_equal(from_aligned, from_raw), (
        "ảnh test phải lệch đủ để phân biệt được hai nguồn cắt"
    )


def test_the_stored_box_and_the_stored_picture_describe_the_same_place(
    tmp_path: Path,
) -> None:
    """Hộp trong bảng và ảnh trong kho phải nói về một chỗ.

    Hai bên tự tính hộp riêng là hai bên sẽ lệch nhau, và khi lệch thì người xem
    mở ảnh ra thấy một nơi trong khi số đo chỉ một nơi khác — không ai phát hiện
    được bằng mắt.
    """

    recipe, _, run = _ng_run(tmp_path)
    with InspectionStore(tmp_path / "kho") as store:
        board = store.ensure_board()
        stored = store.load_inspection(
            record_inspection(store, run, recipe, board_id=board, event_id="EV-1")
        )
        defect = stored.defects[0]
        image = _decode(store.image_file(defect["image_sha256"]))
    box = (defect["x1"], defect["y1"], defect["x2"], defect["y2"])
    x1, y1, x2, y2 = _padded(box, run.alignment.image.shape)
    assert image.shape[:2] == (y2 - y1, x2 - x1)
    assert defect["coordinate_space"] == run.coordinate_space


def test_a_passing_board_is_stored_too_with_no_defect_rows(tmp_path: Path) -> None:
    """Bo ĐẠT cũng vào kho (§6.4).

    Chỉ lưu bo hỏng thì tỷ lệ đạt lần đầu không tính được: mẫu số biến mất.
    """

    golden, recipe, candidate = _production_recipe(tmp_path)
    run = _inspector([candidate], config=InspectionConfig()).inspect(
        golden, recipe, tmp_path
    )
    assert run.status == "pass"
    with InspectionStore(tmp_path / "kho") as store:
        board = store.ensure_board()
        stored = store.load_inspection(
            record_inspection(store, run, recipe, board_id=board, event_id="EV-OK")
        )
        assert stored.status == "pass"
        assert stored.defects == ()


def test_an_invalid_run_is_stored_even_though_there_is_no_image_to_crop(
    tmp_path: Path,
) -> None:
    """Nắn hỏng thì không có ảnh đã nắn — vẫn phải lưu được lần chạy.

    Một lần ``invalid`` bị bỏ qua là một lần bo đi tiếp mà không ai biết máy đã
    không kiểm được nó.
    """

    _, recipe, candidate = _production_recipe(tmp_path)
    noise = np.random.default_rng(7).integers(
        0, 255, size=(180, 240, 3), dtype=np.uint8
    )
    run = _inspector([candidate], config=InspectionConfig()).inspect(
        noise, recipe, tmp_path
    )
    assert run.status == "invalid" and run.alignment.image is None
    with InspectionStore(tmp_path / "kho") as store:
        board = store.ensure_board()
        stored = store.load_inspection(
            record_inspection(store, run, recipe, board_id=board, event_id="EV-BAD")
        )
        assert stored.status == "invalid"


def test_sending_the_same_event_twice_does_not_duplicate_images(
    tmp_path: Path,
) -> None:
    """Gửi lại một sự kiện: một bản ghi, một ảnh — không phải hai."""

    recipe, _, run = _ng_run(tmp_path)
    with InspectionStore(tmp_path / "kho") as store:
        board = store.ensure_board()
        first = record_inspection(store, run, recipe, board_id=board, event_id="EV-1")
        second = record_inspection(store, run, recipe, board_id=board, event_id="EV-1")
        assert first == second
        assets = store.connection.execute(
            "SELECT COUNT(*) AS n FROM image_asset"
        ).fetchone()["n"]
        assert assets == 1


def test_an_unfamiliar_coordinate_space_drops_the_image_instead_of_guessing(
    tmp_path: Path,
) -> None:
    """Hệ quy chiếu lạ thì không cắt bừa.

    Mất ảnh còn sửa được; ảnh sai chỗ thì không ai biết mà sửa.
    """

    recipe, _, run = _ng_run(tmp_path)
    odd = replace(run, coordinate_space="mot_he_khac")
    with InspectionStore(tmp_path / "kho") as store:
        board = store.ensure_board()
        stored = store.load_inspection(
            record_inspection(store, odd, recipe, board_id=board, event_id="EV-1")
        )
        assert stored.defects, "vẫn phải có hàng lỗi"
        assert not stored.defects[0]["image_sha256"], "nhưng không có ảnh đoán mò"


# ------------------------------------------- các ca biên của riêng phép cắt


def test_a_defect_with_no_position_keeps_its_row_and_loses_only_the_image(
    tmp_path: Path,
) -> None:
    """Không định vị được vẫn là một lỗi. Bỏ nó đi là nói dối."""

    image = np.full((40, 60, 3), 128, np.uint8)
    records = crop_defect_images(
        [DefectRecord(slot_id="U1", status="ng", reason="khong_do_duoc")],
        image,
        tmp_path / "crops",
    )
    assert len(records) == 1 and records[0].image_path is None


def test_a_box_outside_the_frame_keeps_its_row_and_loses_only_the_image(
    tmp_path: Path,
) -> None:
    image = np.full((40, 60, 3), 128, np.uint8)
    records = crop_defect_images(
        [
            DefectRecord(
                slot_id="U1",
                status="ng",
                bbox=(200.0, 200.0, 240.0, 240.0),
                coordinate_space="golden_board_pixels",
            )
        ],
        image,
        tmp_path / "crops",
    )
    assert len(records) == 1 and records[0].image_path is None


def test_two_slots_whose_names_collide_after_cleaning_still_get_two_files(
    tmp_path: Path,
) -> None:
    """``U/1`` và ``U:1`` cùng thành ``U_1``; ghi đè nhau thì một lỗi mất ảnh."""

    image = np.full((40, 60, 3), 128, np.uint8)
    box = (5.0, 5.0, 15.0, 15.0)
    records = crop_defect_images(
        [
            DefectRecord(
                slot_id="U/1", status="ng", bbox=box,
                coordinate_space="golden_board_pixels",
            ),
            DefectRecord(
                slot_id="U:1", status="ng", bbox=box,
                coordinate_space="golden_board_pixels",
            ),
        ],
        image,
        tmp_path / "crops",
    )
    paths = {record.image_path for record in records}
    assert len(paths) == 2


def test_a_negative_margin_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        crop_defect_images([], np.zeros((4, 4, 3), np.uint8), tmp_path, margin=-0.1)
