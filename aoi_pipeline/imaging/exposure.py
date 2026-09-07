"""Tìm vùng **cháy sáng** — chỗ ảnh đã mất hẳn thông tin, không cứu được.

Một vùng cháy khác một vùng sáng. Connector nhựa kem trên bo dự án đo được
gần 250 mức xám mà vẫn thấy rõ từng gân nhựa: nó sáng, không cháy, và dùng để
huấn luyện vẫn tốt. Ngược lại, vệt loá trên soldermask đen cũng đo được ngần ấy
mức xám nhưng phẳng lì — chỗ đó không còn linh kiện, không còn mối hàn, không
còn gì.

Nên điều kiện ở đây là **ba vế cùng lúc**:

1. **gần bão hoà** — kênh màu sáng nhất chạm trần (``saturation_level``);
2. **sáng trắng** — độ sáng tổng cũng phải cao (``luminance_level``);
3. **mất chi tiết cục bộ** — độ lệch chuẩn trong cửa sổ nhỏ dưới ngưỡng
   (``flat_level``).

Thiếu vế 1 hoặc 3 thì mọi thứ màu sáng đều bị coi là hỏng, kể cả connector còn
nguyên chi tiết và **pad mạ vàng** — mà pad mạ vàng lộ thiên đúng là lớp bo dự án
đang cần dạy cho model. Loại chúng đi là tự tay bỏ mất thứ cần học.

**Vế 2 thêm vào sau khi đo hụt.** Bản đầu chỉ có vế 1 và 3, và nó chấm một tile
là "cháy 16%" trong khi vùng đó là **tản nhiệt nhôm anod xanh**: kênh lam chạm
trần trên mặt phẳng lì, còn hai kênh kia thấp. Mặt màu bão hoà **không phải** mặt
cháy — thông tin vẫn còn nguyên, chỉ là một màu. Cháy thật thì bạc màu về trắng,
nên độ sáng tổng phải cao chứ không riêng một kênh.

Ngưỡng mặc định đo trên 31 ảnh bo dự án (`real_pcb/phone`, 2026-09-07) và đã xem
tận mắt bằng ảnh chồng mặt nạ: chúng bắt nhãn giấy trắng, vệt loá và mặt kim
loại phản chiếu, mà **không** bắt connector còn vân hay soldermask thường.
"""

from __future__ import annotations

import cv2
import numpy as np

from .image_io import ensure_bgr

__all__ = [
    "DEFAULT_FLAT_LEVEL",
    "DEFAULT_LUMINANCE_LEVEL",
    "DEFAULT_SATURATION_LEVEL",
    "DEFAULT_WINDOW",
    "blown_fraction",
    "blown_highlight_mask",
]

#: Kênh sáng nhất từ mức này trở lên thì coi là chạm trần.
DEFAULT_SATURATION_LEVEL = 245
#: Độ sáng tổng phải từ mức này trở lên. Chặn mặt màu bão hoà một kênh: tản
#: nhiệt xanh đo được độ sáng ~100, pad mạ vàng ~188, còn vệt loá trắng ~250.
DEFAULT_LUMINANCE_LEVEL = 200
#: Độ lệch chuẩn cục bộ dưới mức này thì coi là đã mất chi tiết.
DEFAULT_FLAT_LEVEL = 6.0
#: Cạnh cửa sổ đo độ lệch chuẩn, tính bằng pixel. Lẻ để có tâm.
DEFAULT_WINDOW = 9


def blown_highlight_mask(
    image: np.ndarray,
    *,
    saturation_level: int = DEFAULT_SATURATION_LEVEL,
    luminance_level: int = DEFAULT_LUMINANCE_LEVEL,
    flat_level: float = DEFAULT_FLAT_LEVEL,
    window: int = DEFAULT_WINDOW,
    close_px: int = 7,
) -> np.ndarray:
    """Mặt nạ ``uint8`` (0/1) của các pixel đã cháy.

    ``close_px`` khép các lỗ nhỏ bên trong một vệt loá: giữa một vùng cháy
    thường còn vài pixel lệch ngưỡng, và đếm chúng là "còn chi tiết" thì một vệt
    liền mạch bị tính thành hàng chục mảnh vụn.
    """

    if window < 3 or window % 2 == 0:
        raise ValueError("window phải là số lẻ >= 3")
    if not 0 <= saturation_level <= 255:
        raise ValueError("saturation_level phải nằm trong 0..255")
    if not 0 <= luminance_level <= 255:
        raise ValueError("luminance_level phải nằm trong 0..255")
    if flat_level < 0:
        raise ValueError("flat_level không được âm")

    bgr = ensure_bgr(image)
    near_ceiling = bgr.max(axis=2) >= saturation_level

    grey_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    washed_out = grey_u8 >= luminance_level
    grey = grey_u8.astype(np.float32)
    mean = cv2.blur(grey, (window, window))
    mean_square = cv2.blur(grey * grey, (window, window))
    # max(...,0) vì sai số dấu phẩy động có thể cho phương sai âm rất nhỏ.
    local_std = np.sqrt(np.maximum(mean_square - mean * mean, 0.0))

    mask = (near_ceiling & washed_out & (local_std < flat_level)).astype(np.uint8)
    if close_px > 1:
        kernel = np.ones((close_px, close_px), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def blown_fraction(image: np.ndarray, **kwargs: object) -> float:
    """Tỉ lệ pixel đã cháy, trong khoảng 0..1."""

    return float(blown_highlight_mask(image, **kwargs).mean())  # type: ignore[arg-type]
