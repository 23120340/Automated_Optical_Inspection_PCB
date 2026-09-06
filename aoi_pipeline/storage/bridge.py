"""Chuyển kết quả kiểm tra thành bản ghi lưu trữ.

Tách riêng khỏi ``repository.py`` vì đây là **phép dịch giữa hai miền**, không
phải chuyện lưu trữ: bên trái là ``InspectionRun`` của bước 3.5, bên phải là
``DefectRecord`` của kho. Trộn chung thì kho phải biết về recipe, và mỗi lần đổi
kết quả kiểm lại phải sửa cả tầng lưu trữ.

**Vì sao hàm này cần cả ``recipe``.** Yêu cầu của dây chuyền là *lưu vị trí lỗi*,
mà lỗi quan trọng nhất — **thiếu linh kiện** — lại là lỗi **không có vị trí**
trong kết quả chạy: ``SlotInspectionResult.candidate`` là ``None`` vì không tìm
thấy gì, còn ``PositionResult`` chỉ mang *độ lệch* chứ không mang toạ độ tuyệt
đối. Vị trí duy nhất đúng cho ca đó là **chỗ linh kiện LẼ RA phải nằm**, và chỗ
đó chỉ recipe biết.

Bỏ qua chi tiết này thì kho vẫn chạy, vẫn có bản ghi, chỉ là mọi lỗi thiếu linh
kiện đều không mở được ảnh — thứ hỏng im lặng đúng nghĩa.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .repository import DefectRecord

__all__ = ["PASSING_SLOT_STATUSES", "defects_from_run"]

#: Trạng thái slot **không** coi là lỗi. Mọi trạng thái khác đều được lưu, kể cả
#: ``review`` và các trạng thái "không đo được": một slot không kết luận được
#: cũng là thứ người vận hành phải nhìn, và bỏ nó ra khỏi kho là làm nó biến mất.
PASSING_SLOT_STATUSES = frozenset({"pass"})


def defects_from_run(
    run: Any,
    recipe: Any = None,
    *,
    images: Mapping[str, str | Path] | None = None,
    passing: frozenset[str] = PASSING_SLOT_STATUSES,
) -> list[DefectRecord]:
    """Lấy ra các slot **không đạt**, kèm vị trí và ảnh của từng cái.

    Thứ tự lấy vị trí, mạnh trước:

    1. Hộp của ứng viên tìm được — chỗ linh kiện **thật sự đang** nằm;
    2. ``expected_bbox_xyxy`` của recipe — chỗ nó **lẽ ra** phải nằm, dùng khi
       không tìm thấy ứng viên nào (thiếu linh kiện);
    3. không có vị trí — chỉ khi cả hai nguồn trên đều không có.

    Trường hợp 3 vẫn tạo bản ghi, **không** bỏ qua slot đó: một lỗi không định vị
    được vẫn là một lỗi, và im lặng bỏ nó đi thì báo cáo "không có lỗi" trở thành
    lời nói dối.

    ``images`` khoá theo ``slot_id``. Thiếu ảnh cho một slot thì bản ghi vẫn được
    tạo, chỉ là không có ảnh — mất ảnh không được phép làm mất cả vị trí.
    """

    by_slot = {}
    if recipe is not None:
        by_slot = {slot.slot_id: slot for slot in getattr(recipe, "slots", ())}
    space = str(getattr(run, "coordinate_space", "") or "")
    images = images or {}

    records: list[DefectRecord] = []
    for slot in getattr(run, "slots", ()):
        if slot.status in passing:
            continue
        bbox = None
        if slot.candidate is not None:
            bbox = tuple(float(value) for value in slot.candidate.bbox_xyxy)
        else:
            expected = by_slot.get(slot.slot_id)
            if expected is not None:
                bbox = tuple(
                    float(value) for value in expected.expected_bbox_xyxy.as_xyxy()
                )
        records.append(
            DefectRecord(
                slot_id=slot.slot_id,
                status=slot.status,
                reason=slot.reason,
                bbox=bbox,
                # Toạ độ và hệ quy chiếu đi cùng nhau hoặc cùng vắng: một bbox
                # không kèm hệ là con số không đọc lại được (kế hoạch §6.2).
                coordinate_space=space if bbox is not None and space else None,
                image_path=images.get(slot.slot_id),
            )
        )
    return records
