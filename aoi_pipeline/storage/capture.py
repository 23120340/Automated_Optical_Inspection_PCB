"""Chạy xong một lần kiểm → một bản ghi trong kho, kèm ảnh của từng lỗi.

Đây là mảnh còn thiếu giữa ``golden/inspector.py`` và ``storage/repository.py``:
``defects_from_run`` cho *vị trí* lỗi, ``save_inspection`` nhận *vị trí kèm ảnh*,
nhưng **chưa ai cắt ảnh**. Yêu cầu chốt với dây chuyền là "lưu các vị trí bị lỗi
kèm ảnh lỗi", nên thiếu mảnh này thì kho chạy được mà mọi bản ghi đều không có
ảnh — hỏng im lặng đúng nghĩa.

**Cắt từ ảnh nào là chỗ dễ sai nhất.** Detector chạy trên ``alignment.image`` —
ảnh test đã nắn về hệ ``golden_board_pixels`` — nên hộp của ứng viên và hộp
``expected_bbox_xyxy`` của recipe đều nằm trong hệ đó. Cắt từ ảnh test **gốc**
bằng chính những toạ độ ấy thì vẫn ra một tấm ảnh, vẫn lưu được, chỉ là ảnh của
chỗ khác. Không có gì báo lỗi. Vì vậy ở đây chỉ cắt khi hệ quy chiếu của lần chạy
đúng là hệ của ảnh đã nắn, còn không thì **bỏ ảnh chứ không cắt bừa**: mất ảnh
còn sửa được, ảnh sai chỗ thì không ai biết mà sửa.

Vị trí thì lấy nguyên từ ``defects_from_run``, không tự tính lại. Hai chỗ cùng
tính một hộp là hai chỗ sẽ lệch nhau, và khi lệch thì bbox lưu trong bảng trỏ một
nơi còn ảnh chụp một nơi khác.
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..golden.recipe import GOLDEN_COORDINATE_SPACE
from ..imaging.image_io import InvalidImageError, encode_image
from ..models import BoundingBox
from .bridge import PASSING_SLOT_STATUSES, defects_from_run
from .repository import DefectRecord, InspectionStore

__all__ = [
    "DEFAULT_CROP_MARGIN",
    "crop_defect_images",
    "derived_event_id",
    "record_inspection",
]

#: Nới hộp thêm bao nhiêu phần chiều rộng/cao mỗi bên khi cắt ảnh lỗi. Một hộp
#: cắt sát mép không đọc được: người xem cần thấy linh kiện bên cạnh mới biết
#: đang nhìn chỗ nào trên bo.
DEFAULT_CROP_MARGIN = 0.35


def _aligned_image(run: Any) -> np.ndarray | None:
    """Ảnh đã nắn của lần chạy, hoặc ``None`` nếu không dùng được để cắt."""

    if str(getattr(run, "coordinate_space", "") or "") != GOLDEN_COORDINATE_SPACE:
        # Hệ quy chiếu lạ: không biết ảnh nào khớp toạ độ, nên không cắt.
        return None
    alignment = getattr(run, "alignment", None)
    image = getattr(alignment, "image", None)
    return image if isinstance(image, np.ndarray) and image.size else None


def crop_defect_images(
    records: Sequence[DefectRecord],
    image: np.ndarray,
    out_dir: str | Path,
    *,
    margin: float = DEFAULT_CROP_MARGIN,
) -> list[DefectRecord]:
    """Cắt một ảnh cho mỗi bản ghi **có toạ độ**, trả về bản ghi đã gắn ảnh.

    Bản ghi không có toạ độ, hoặc có toạ độ nhưng nằm trọn ngoài khung ảnh, được
    trả về **nguyên vẹn không kèm ảnh**. Bỏ hẳn nó đi thì báo cáo "không có lỗi"
    thành lời nói dối, mà bịa một ảnh rỗng cho nó cũng không hơn.
    """

    if margin < 0.0:
        raise ValueError("margin không được âm")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    height, width = image.shape[:2]

    filled: list[DefectRecord] = []
    for index, record in enumerate(records):
        if record.bbox is None:
            filled.append(record)
            continue
        box = BoundingBox(*record.bbox)
        pad_x, pad_y = box.width * margin, box.height * margin
        x1, y1, x2, y2 = (
            BoundingBox(box.x1 - pad_x, box.y1 - pad_y, box.x2 + pad_x, box.y2 + pad_y)
            .clamp(width, height)
            .to_int()
        )
        if x2 <= x1 or y2 <= y1:
            # Hộp nằm ngoài khung: vẫn giữ bản ghi, chỉ là không có ảnh.
            filled.append(record)
            continue
        # Tên file mang cả số thứ tự vì `slot_id` không chắc là tên file hợp lệ,
        # và hai slot khác nhau vẫn có thể trùng tên sau khi lọc ký tự.
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in record.slot_id)
        path = out_dir / f"{index:04d}_{safe or 'slot'}.png"
        try:
            path.write_bytes(encode_image(image[y1:y2, x1:x2]))
        except InvalidImageError:
            filled.append(record)
            continue
        filled.append(replace(record, image_path=path))
    return filled


def derived_event_id(run: Any, *, board_id: str) -> str:
    """``event_id`` suy ra từ chính lần chạy — **chỉ dùng khi dây chuyền chưa cấp**.

    ``save_inspection`` bất biến theo ``event_id``, và điều đó chỉ có nghĩa nếu
    hai lần gửi cùng một sự kiện sản xuất mang cùng một mã. Mã đó **phải** do dây
    chuyền cấp: chỉ nó biết hai lần chạy có phải cùng một bo đi qua một lần hay
    không.

    Khi chưa có nguồn đó — trong workbench, hay khi chạy lại một ảnh đã lưu — thì
    mã ở đây khoá theo ``started_at`` của lần chạy, nên bấm Lưu hai lần cho cùng
    một kết quả không sinh thêm bản ghi, còn hai lần *kiểm* khác nhau vẫn là hai
    sự kiện. Nó **không** thay được mã của dây chuyền: hai máy trạm cùng kiểm một
    bo vẫn ra hai mã khác nhau, và đó là giới hạn thật chứ không phải lỗi.
    """

    material = "|".join((
        str(board_id),
        str(getattr(run, "started_at", "") or ""),
        str(getattr(run, "side", "") or ""),
        str(getattr(run, "recipe_sha256", "") or ""),
    ))
    return f"LOCAL-{hashlib.sha256(material.encode('utf-8')).hexdigest()[:16]}"


def record_inspection(
    store: InspectionStore,
    run: Any,
    recipe: Any = None,
    *,
    board_id: str,
    event_id: str,
    session_id: str | None = None,
    margin: float = DEFAULT_CROP_MARGIN,
    passing: frozenset[str] = PASSING_SLOT_STATUSES,
) -> str:
    """Lưu một lần chạy cùng vị trí và ảnh của từng lỗi. Trả về ``inspection_id``.

    Lưu **mọi** trạng thái bo — ``pass``, ``ng``, ``review``, ``invalid`` — chứ
    không chỉ bo hỏng (§6.4 và giai đoạn 3 của kế hoạch): tỷ lệ đạt lần đầu không
    tính được nếu mẫu số biến mất, và một lần ``invalid`` bị bỏ qua là một lần bo
    đi tiếp mà không ai biết máy đã không kiểm được nó.

    Bất biến theo ``event_id`` vì ``save_inspection`` bất biến: gọi lại cùng một
    sự kiện trả về đúng bản ghi cũ, không sinh thêm ảnh trong kho.
    """

    records = defects_from_run(run, recipe, passing=passing)
    image = _aligned_image(run)
    if image is None or not records:
        return store.save_inspection(
            run, board_id=board_id, event_id=event_id,
            defects=records, session_id=session_id,
        )
    # Cắt vào thư mục tạm rồi để kho tự chép vào kho địa chỉ-theo-nội dung: ảnh
    # trùng nội dung chỉ nằm một bản, và nếu lần lưu này là sự kiện lặp thì không
    # còn file thừa nào sót lại ngoài kho.
    with tempfile.TemporaryDirectory(prefix="aoi-defect-") as scratch:
        filled = crop_defect_images(records, image, scratch, margin=margin)
        return store.save_inspection(
            run, board_id=board_id, event_id=event_id,
            defects=filled, session_id=session_id,
        )
