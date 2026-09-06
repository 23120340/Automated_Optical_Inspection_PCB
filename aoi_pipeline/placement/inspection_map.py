"""Bản đồ kiểm tra: linh kiện nào nằm đâu, và phải chụp mấy khung để phủ hết.

Board thật lớn hơn trường nhìn của camera ở độ phân giải cần cho kiểm tra
fillet. Đo được: ở 25 µm/px, một cảm biến 20 MP nhìn được ~137 × 91 mm, trong
khi board 200 × 150 mm — tức phải chụp nhiều khung rồi ghép lại
(`docs/design/yeu_cau_phan_cung_camera.md`).

Module này trả lời ba câu, theo thứ tự:

1. **Có những linh kiện nào và nằm đâu** — từ CAD / pick-and-place, hoặc từ
   BOM có toạ độ.
2. **Phải chụp mấy khung, mỗi khung ở đâu** — chia lưới theo trường nhìn, có
   chồng lấn.
3. **Mỗi khung chụp chứa những linh kiện nào** — để sau khi chụp thì cắt đúng
   crop mà không phải dò lại.

Nguồn nào đủ điều kiện dựng bản đồ, và vì sao:

| Nguồn | Dựng được? |
|---|---|
| **CAD / pick-and-place** | **Có.** Cho thẳng designator + toạ độ mm + góc xoay, và kích thước khi file có |
| **BOM có toạ độ** | **Có.** Tương đương, chỉ thường thiếu góc xoay |
| BOM không toạ độ | Không. Nó nói board có *những gì*, không nói *ở đâu* |
| Golden image | Không trực tiếp. Nó là ảnh, muốn ra toạ độ vẫn phải detect — mà detect chính là thứ ta đang muốn dẫn đường |

**Điều module này KHÔNG làm: điều khiển camera hay bàn máy.** Nó sinh ra *kế
hoạch chụp* — toạ độ mm của từng khung. Việc đưa camera tới đó là giao tiếp với
phần cứng cụ thể, và viết mã cho một thiết bị chưa từng chạy thử là cách chắc
chắn nhất để giao một thứ trông như đã xong mà không ai biết nó sai ở đâu.
:class:`CaptureRegion` là chỗ nối: đọc `center_mm` rồi ra lệnh cho bàn máy của
bạn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from ..exceptions import AOIPipelineError
from ..models import BoundingBox

__all__ = [
    "CaptureRegion",
    "InspectionMap",
    "MapComponent",
    "MapError",
    "build_from_bom",
    "build_from_cad",
    "plan_capture_regions",
]


class MapError(AOIPipelineError):
    """Không dựng được bản đồ từ nguồn đã cho."""


@dataclass(frozen=True, slots=True)
class MapComponent:
    """Một linh kiện cần kiểm, ở toạ độ board (mm)."""

    designator: str
    x: float
    y: float
    width: float = 0.0
    height: float = 0.0
    rotation: float = 0.0
    part_class: str | None = None
    side: str = "top"

    @property
    def has_size(self) -> bool:
        return self.width > 0.0 and self.height > 0.0

    def extent_mm(self, default: float) -> tuple[float, float]:
        """Bao nhiêu mm quanh tâm phải nằm trong khung chụp.

        Không có kích thước thì dùng ``default``: một linh kiện bị cắt đôi ở
        mép khung là một linh kiện không kiểm được, nên thà giả định to hơn
        thật còn hơn nhỏ hơn thật.
        """

        if not self.has_size:
            return default, default
        # Xoay 90° thì chiều dài đổi trục. Lấy bao ngoài của hình đã xoay thay
        # vì đúng hình chữ nhật xoay: dư một chút, và dư thì an toàn.
        angle = math.radians(self.rotation)
        cos, sin = abs(math.cos(angle)), abs(math.sin(angle))
        return (
            (self.width * cos + self.height * sin) / 2.0,
            (self.width * sin + self.height * cos) / 2.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "designator": self.designator,
            "x_mm": round(self.x, 3), "y_mm": round(self.y, 3),
            "width_mm": round(self.width, 3), "height_mm": round(self.height, 3),
            "rotation_deg": self.rotation,
            "part_class": self.part_class, "side": self.side,
        }


@dataclass(slots=True)
class InspectionMap:
    """Toàn bộ linh kiện cần kiểm trên một mặt board."""

    components: list[MapComponent] = field(default_factory=list)
    source: str = ""
    units: str = "mm"
    side: str = "top"

    def __len__(self) -> int:
        return len(self.components)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Khung bao của mọi linh kiện, tính bằng mm (x1, y1, x2, y2)."""

        if not self.components:
            return (0.0, 0.0, 0.0, 0.0)
        xs, ys = [], []
        for item in self.components:
            half_w, half_h = item.extent_mm(0.0)
            xs.extend((item.x - half_w, item.x + half_w))
            ys.extend((item.y - half_h, item.y + half_h))
        return (min(xs), min(ys), max(xs), max(ys))

    def with_size(self) -> int:
        return sum(1 for item in self.components if item.has_size)

    def to_dict(self) -> dict[str, Any]:
        x1, y1, x2, y2 = self.extent
        return {
            "source": self.source, "units": self.units, "side": self.side,
            "count": len(self.components),
            "with_size": self.with_size(),
            "extent_mm": [round(v, 3) for v in (x1, y1, x2, y2)],
            "span_mm": [round(x2 - x1, 3), round(y2 - y1, 3)],
            "components": [item.to_dict() for item in self.components],
        }


@dataclass(frozen=True, slots=True)
class CaptureRegion:
    """Một khung cần chụp, và những linh kiện nó phủ.

    ``center_mm`` là chỗ nối với phần cứng: đưa camera (hoặc bàn máy) tới toạ
    độ này rồi chụp. Module này cố tình không tự làm việc đó.
    """

    index: int
    row: int
    column: int
    center_x: float
    center_y: float
    width: float
    height: float
    designators: tuple[str, ...] = ()

    @property
    def center_mm(self) -> tuple[float, float]:
        return (self.center_x, self.center_y)

    @property
    def bounds_mm(self) -> tuple[float, float, float, float]:
        return (
            self.center_x - self.width / 2.0, self.center_y - self.height / 2.0,
            self.center_x + self.width / 2.0, self.center_y + self.height / 2.0,
        )

    def to_dict(self) -> dict[str, Any]:
        x1, y1, x2, y2 = self.bounds_mm
        return {
            "index": self.index, "row": self.row, "column": self.column,
            "center_mm": [round(self.center_x, 3), round(self.center_y, 3)],
            "bounds_mm": [round(v, 3) for v in (x1, y1, x2, y2)],
            "size_mm": [round(self.width, 3), round(self.height, 3)],
            "component_count": len(self.designators),
            "designators": list(self.designators),
        }


# --------------------------------------------------------------------------
# Dựng bản đồ
# --------------------------------------------------------------------------


def build_from_cad(board: Any, *, side: str | None = "top") -> InspectionMap:
    """Bản đồ từ :class:`~aoi_pipeline.solder.cad.BoardCad`.

    Đây là nguồn tốt nhất: nó cho designator, toạ độ, góc xoay, và kích thước
    khi định dạng file có mang.
    """

    components = []
    for item in getattr(board, "components", []) or []:
        if side is not None and getattr(item, "side", "top") != side:
            continue
        components.append(MapComponent(
            designator=item.designator, x=float(item.x), y=float(item.y),
            width=float(getattr(item, "width", 0.0) or 0.0),
            height=float(getattr(item, "height", 0.0) or 0.0),
            rotation=float(getattr(item, "rotation", 0.0) or 0.0),
            part_class=getattr(item, "part_class", None),
            side=getattr(item, "side", "top"),
        ))
    if not components:
        raise MapError(
            f"CAD không có linh kiện nào ở mặt {side!r}. "
            "Kiểm lại cột `side`/`layer` của file, hoặc đặt side=None để lấy tất cả."
        )
    return InspectionMap(components=components,
                         source=str(getattr(board, "source", "cad")),
                         side=side or "all")


def build_from_bom(bom: Any, *, side: str | None = "top") -> InspectionMap:
    """Bản đồ từ BOM — chỉ dùng được khi BOM có toạ độ.

    BOM dạng mua hàng (một dòng mỗi loại, không toạ độ) nói board có *những
    gì* chứ không nói *ở đâu*, nên nó không dựng được bản đồ. Từ chối rõ ràng
    còn hơn trả về một bản đồ mà mọi linh kiện chồng lên nhau ở gốc toạ độ.
    """

    entries = [item for item in getattr(bom, "entries", []) if item.has_position]
    if side is not None:
        entries = [item for item in entries if getattr(item, "side", "top") == side]
    if not entries:
        raise MapError(
            "BOM không có toạ độ nên không dựng được bản đồ. Cần file dạng "
            "một dòng mỗi linh kiện kèm cột X/Y, hoặc dùng CAD/pick-and-place."
        )
    return InspectionMap(
        components=[
            MapComponent(
                designator=item.designator, x=float(item.x), y=float(item.y),
                width=float(item.width or 0.0), height=float(item.height or 0.0),
                rotation=float(item.rotation or 0.0),
                part_class=item.part_class, side=getattr(item, "side", "top"),
            )
            for item in entries
        ],
        source=str(getattr(bom, "source", "bom")),
        side=side or "all",
    )


# --------------------------------------------------------------------------
# Kế hoạch chụp
# --------------------------------------------------------------------------


def plan_capture_regions(
    inspection_map: InspectionMap,
    fov_width_mm: float,
    fov_height_mm: float,
    *,
    overlap: float = 0.15,
    default_component_mm: float = 5.0,
    margin_mm: float = 2.0,
) -> list[CaptureRegion]:
    """Chia board thành các khung chụp, và gán linh kiện vào từng khung.

    ``overlap`` là phần chồng lấn giữa hai khung kề nhau. Nó không phải để cho
    đẹp: không chồng lấn thì một linh kiện nằm đúng đường ranh sẽ bị cắt đôi ở
    cả hai khung và **không khung nào kiểm được nó**. Chồng lấn cũng là thứ cho
    phép ghép ảnh sau này.

    Một linh kiện được gán vào khung khi **toàn bộ** nó nằm trong khung, không
    phải chỉ tâm. Gán theo tâm thì linh kiện ở mép bị cắt mà vẫn coi là đã kiểm.
    Nhờ chồng lấn, một linh kiện ở ranh giới thường nằm trọn trong khung kế bên.
    """

    if not inspection_map.components:
        raise MapError("Bản đồ rỗng, không có gì để lập kế hoạch chụp.")
    if fov_width_mm <= 0 or fov_height_mm <= 0:
        raise MapError("Trường nhìn phải lớn hơn 0.")
    if not 0.0 <= overlap < 0.9:
        raise MapError("Chồng lấn phải trong khoảng [0, 0.9).")

    x1, y1, x2, y2 = inspection_map.extent
    x1, y1 = x1 - margin_mm, y1 - margin_mm
    x2, y2 = x2 + margin_mm, y2 + margin_mm
    span_x, span_y = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)

    step_x = fov_width_mm * (1.0 - overlap)
    step_y = fov_height_mm * (1.0 - overlap)
    columns = max(1, math.ceil((span_x - fov_width_mm) / step_x) + 1) if span_x > fov_width_mm else 1
    rows = max(1, math.ceil((span_y - fov_height_mm) / step_y) + 1) if span_y > fov_height_mm else 1

    # Căn lưới vào giữa vùng cần phủ: dồn về một góc thì khung cuối thừa ra
    # ngoài board trong khi khung đầu sát mép, và linh kiện ở mép đầu dễ lọt.
    used_x = fov_width_mm + step_x * (columns - 1)
    used_y = fov_height_mm + step_y * (rows - 1)
    origin_x = x1 - (used_x - span_x) / 2.0
    origin_y = y1 - (used_y - span_y) / 2.0

    regions: list[CaptureRegion] = []
    index = 0
    for row in range(rows):
        for column in range(columns):
            cx = origin_x + fov_width_mm / 2.0 + step_x * column
            cy = origin_y + fov_height_mm / 2.0 + step_y * row
            left, right = cx - fov_width_mm / 2.0, cx + fov_width_mm / 2.0
            top, bottom = cy - fov_height_mm / 2.0, cy + fov_height_mm / 2.0
            inside = []
            for item in inspection_map.components:
                half_w, half_h = item.extent_mm(default_component_mm / 2.0)
                if (left <= item.x - half_w and item.x + half_w <= right
                        and top <= item.y - half_h and item.y + half_h <= bottom):
                    inside.append(item.designator)
            regions.append(CaptureRegion(
                index=index, row=row, column=column,
                center_x=cx, center_y=cy,
                width=fov_width_mm, height=fov_height_mm,
                designators=tuple(inside),
            ))
            index += 1
    return regions


def uncovered(
    inspection_map: InspectionMap, regions: Sequence[CaptureRegion]
) -> list[str]:
    """Linh kiện không nằm trọn trong bất kỳ khung nào.

    Đây là con số phải nhìn trước khi tin vào một kế hoạch chụp: một linh kiện
    không khung nào phủ trọn là một linh kiện **sẽ không được kiểm**, và nếu
    không ai đếm thì nó trôi qua trong im lặng.
    """

    covered = {name for region in regions for name in region.designators}
    return [item.designator for item in inspection_map.components
            if item.designator not in covered]


def components_in_capture(
    inspection_map: InspectionMap,
    region: CaptureRegion,
) -> list[MapComponent]:
    """Các linh kiện của một khung, kèm dữ liệu đầy đủ (không chỉ designator)."""

    wanted = set(region.designators)
    return [item for item in inspection_map.components if item.designator in wanted]


def crop_boxes_for_capture(
    inspection_map: InspectionMap,
    region: CaptureRegion,
    image_width: int,
    image_height: int,
    *,
    default_component_mm: float = 5.0,
    padding: float = 0.25,
) -> dict[str, BoundingBox]:
    """Toạ độ crop từng linh kiện **trong ảnh vừa chụp của khung này**.

    Ánh xạ thẳng: khung chụp phủ ``region.width × region.height`` mm lên
    ``image_width × image_height`` pixel, nên mm → pixel là một phép tỉ lệ.
    Không cần dò lại linh kiện trên ảnh — bản đồ đã biết chúng ở đâu.

    Giả định: ảnh khớp đúng khung đã lập kế hoạch. Bàn máy lệch thì crop lệch
    theo, và không có gì trong ảnh phát hiện ra điều đó — nên hãy căn bằng
    fiducial trước khi tin vào crop ở mức pixel.
    """

    left, top, right, bottom = region.bounds_mm
    scale_x = image_width / max(right - left, 1e-6)
    scale_y = image_height / max(bottom - top, 1e-6)

    boxes: dict[str, BoundingBox] = {}
    for item in components_in_capture(inspection_map, region):
        half_w, half_h = item.extent_mm(default_component_mm / 2.0)
        half_w *= 1.0 + padding
        half_h *= 1.0 + padding
        boxes[item.designator] = BoundingBox(
            (item.x - half_w - left) * scale_x,
            (item.y - half_h - top) * scale_y,
            (item.x + half_w - left) * scale_x,
            (item.y + half_h - top) * scale_y,
        ).clamp(image_width, image_height)
    return boxes
