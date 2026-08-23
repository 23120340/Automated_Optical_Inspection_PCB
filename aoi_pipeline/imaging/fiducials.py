"""Dò fiducial mark để biết board nằm đâu, không cần ảnh golden.

Bước 3 hiện khoanh board bằng contour: tìm mảng lớn nhất trông giống hình chữ
nhật. Cách đó hỏng ở đúng những lúc hay gặp — nền cùng màu với board, board bị
che một phần, hoặc ảnh có nhiều board. Và nó cho một hình chữ nhật *bao quanh*
chứ không cho một hệ toạ độ.

Fiducial thì cho cả hai. Chúng là mốc chuẩn hoá theo IPC-7351: một pad đồng
tròn, thường đường kính **1 mm**, với cửa sổ solder-mask rộng gấp 2–3 lần nên
xung quanh là vành trống. Đồng trần phản xạ mạnh hơn mask xanh, nên chúng là
những đốm tròn sáng, cô lập, và **vị trí của chúng là toạ độ đã biết trên bản
vẽ** — tức chúng neo được ảnh vào hệ toạ độ của board.

Vì sao không dùng `cv2.HoughCircles`: nó nhạy với `param1`/`param2` tới mức
mỗi board lại phải chỉnh, và khi trượt thì trượt im lặng. Lọc theo thành phần
liên thông + độ tròn thì mỗi tiêu chí là một con số đọc được, đo được, và giải
thích được khi nó loại nhầm.

**Cái này KHÔNG làm được:** fiducial nằm *bên trong* mép board (thường lùi vào
5 mm), nên bao lồi của chúng nhỏ hơn board thật. Phải nới ra, và phần nới là
ước lượng chứ không phải đo. Nó cho một hệ toạ độ đáng tin, không cho một
đường viền board đáng tin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..config import FiducialConfig
from .image_io import ensure_bgr

__all__ = ["Fiducial", "FiducialResult", "find_fiducials"]


@dataclass(frozen=True, slots=True)
class Fiducial:
    """Một đốm tròn sáng đủ tiêu chuẩn làm mốc."""

    x: float
    y: float
    radius: float
    circularity: float
    contrast: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": round(self.x, 2), "y": round(self.y, 2),
            "radius": round(self.radius, 2),
            "circularity": round(self.circularity, 3),
            "contrast": round(self.contrast, 3),
            "score": round(self.score, 3),
        }


@dataclass(slots=True)
class FiducialResult:
    """Những mốc tìm được, và vì sao các ứng viên khác bị loại."""

    fiducials: list[Fiducial] = field(default_factory=list)
    rejected: dict[str, int] = field(default_factory=dict)
    threshold: int = 0

    @property
    def usable(self) -> bool:
        """Ba mốc là tối thiểu để chốt được xoay + tịnh tiến + tỉ lệ.

        Hai mốc cũng giải được một phép biến đổi tương tự, nhưng không phát
        hiện được khi một trong hai là dương tính giả — và một mốc sai thì
        toàn bộ hệ toạ độ sai theo, im lặng.
        """

        return len(self.fiducials) >= 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": len(self.fiducials),
            "usable": self.usable,
            "threshold": int(self.threshold),
            "rejected": dict(self.rejected),
            "fiducials": [item.to_dict() for item in self.fiducials],
        }


def _spread(points: np.ndarray, width: int, height: int) -> float:
    """Các mốc trải rộng tới đâu so với khung ảnh.

    Ba đốm sáng chụm vào một góc thường là một linh kiện bóng, không phải ba
    fiducial: fiducial được đặt xa nhau có chủ đích để chốt được góc xoay.
    """

    if len(points) < 2:
        return 0.0
    span = points.max(axis=0) - points.min(axis=0)
    return float(min(span[0] / max(width, 1), span[1] / max(height, 1)))


def find_fiducials(
    image: np.ndarray,
    config: FiducialConfig | None = None,
) -> FiducialResult:
    """Tìm các fiducial mark trong ảnh.

    Trả về cả những gì bị loại và vì sao. Một bộ dò chỉ nói "không thấy gì" thì
    không sửa được: người dùng không biết nên nới ngưỡng nào.
    """

    config = config or FiducialConfig()
    gray = cv2.cvtColor(ensure_bgr(image), cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]

    # So với nền CỤC BỘ, không phải ngưỡng toàn cục.
    #
    # Đo trên board thật: vành fiducial sáng khoảng 150-180 trong khi phân vị
    # 99 của cả ảnh là 239 (biến áp, tụ trắng). Một ngưỡng toàn cục bỏ qua
    # fiducial hoàn toàn — nó sáng so với vùng quanh nó, không so với cả ảnh.
    background = cv2.medianBlur(gray, config.background_kernel | 1)
    lift = cv2.subtract(gray, background)
    threshold = int(max(1, round(config.local_lift * 255)))
    mask = (lift >= threshold).astype(np.uint8) * 255
    if config.close_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.close_kernel, config.close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    rejected: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    candidates: list[Fiducial] = []

    for index in range(1, count):
        area = float(stats[index, cv2.CC_STAT_AREA])
        box_w = float(stats[index, cv2.CC_STAT_WIDTH])
        box_h = float(stats[index, cv2.CC_STAT_HEIGHT])
        # Lọc theo ĐƯỜNG KÍNH khung bao, không theo số pixel sáng: một vành
        # mỏng có ít pixel hơn hẳn một đĩa cùng đường kính, nên lọc theo diện
        # tích sẽ loại đúng thứ cần tìm.
        span = max(box_w, box_h) / 2.0
        if not config.min_radius_px <= span <= config.max_radius_px:
            reject("kích thước")
            continue
        aspect = max(box_w, box_h) / max(min(box_w, box_h), 1.0)
        if aspect > config.max_aspect:
            reject("không tròn (tỉ lệ cạnh)")
            continue

        component = (labels[
            stats[index, cv2.CC_STAT_TOP]:stats[index, cv2.CC_STAT_TOP] + int(box_h),
            stats[index, cv2.CC_STAT_LEFT]:stats[index, cv2.CC_STAT_LEFT] + int(box_w),
        ] == index).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            reject("không lấy được biên")
            continue
        perimeter = cv2.arcLength(contours[0], True)
        if perimeter <= 0:
            reject("không lấy được biên")
            continue
        # Diện tích của biên ĐÃ LẤP, không phải số pixel sáng.
        #
        # Fiducial ở đây là một VÀNH sáng quanh lõi tối, không phải đĩa đặc.
        # Lấy số pixel của vành chia cho chu vi ngoài thì độ tròn luôn thấp —
        # đo được cao nhất 0.60 trên board thật, dưới mọi ngưỡng hợp lý, nên
        # bộ dò trượt sạch. Diện tích đã lấp trả hình vành về đúng hình tròn.
        filled_area = float(abs(cv2.contourArea(contours[0])))
        if filled_area <= 0:
            reject("không lấy được biên")
            continue
        # 4πA/P² = 1 với hình tròn hoàn hảo, nhỏ dần khi biên răng cưa.
        circularity = float(4.0 * np.pi * filled_area / (perimeter * perimeter))
        if circularity < config.min_circularity:
            reject("không tròn (độ tròn)")
            continue

        cx, cy = float(centroids[index][0]), float(centroids[index][1])
        radius = float(np.sqrt(filled_area / np.pi))

        # PHÉP THỬ VÀNH — thứ thật sự phân biệt fiducial với một đốm loé.
        #
        # Đo trên board thật: fiducial là vành đồng sáng bao quanh lõi TỐI, còn
        # đốm loé trên linh kiện thì sáng đặc ở giữa. Phép đo tương phản đầu
        # tiên của tôi đòi tâm SÁNG, tức đòi đúng thứ ngược lại, nên nó nhận
        # toàn đốm loé và bỏ sạch fiducial thật.
        yy, xx = np.ogrid[
            int(max(0, cy - radius * 2)):int(min(height, cy + radius * 2)),
            int(max(0, cx - radius * 2)):int(min(width, cx + radius * 2)),
        ]
        distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        window = gray[
            int(max(0, cy - radius * 2)):int(min(height, cy + radius * 2)),
            int(max(0, cx - radius * 2)):int(min(width, cx + radius * 2)),
        ].astype(np.float32)
        if window.size == 0 or window.shape != distance.shape:
            reject("sát mép ảnh")
            continue
        core = window[distance <= radius * 0.45]
        annulus = window[(distance >= radius * 0.6) & (distance <= radius * 1.15)]
        if core.size == 0 or annulus.size == 0:
            reject("quá nhỏ để đo vành")
            continue
        ring_lift = float(annulus.mean() - core.mean()) / 255.0
        if config.require_dark_core and ring_lift < config.min_ring_lift:
            reject("không có vành quanh lõi tối")
            continue

        # Vành sáng so với NỀN BOARD quanh nó — đây là thứ phân biệt fiducial
        # với một mảng đồng lớn cũng sáng.
        #
        # Đo trên VÀNH, không đo ở tâm: tâm của fiducial vốn tối, nên một phép
        # đo "tâm sáng hơn xung quanh" sẽ loại đúng thứ cần tìm. Đó chính là
        # lỗi của bản đầu tiên.
        outside = window[(distance >= radius * config.ring_ratio * 0.5)
                         & (distance <= radius * config.ring_ratio)]
        if outside.size == 0:
            outside = window[distance > radius * 1.15]
        if outside.size == 0:
            reject("sát mép ảnh")
            continue
        contrast = float(annulus.mean() - np.median(outside)) / 255.0
        if contrast < config.min_contrast:
            reject("không nổi so với xung quanh")
            continue

        candidates.append(Fiducial(
            x=cx, y=cy, radius=radius, circularity=circularity,
            contrast=contrast, score=circularity * (0.5 + contrast),
        ))

    candidates.sort(key=lambda item: item.score, reverse=True)
    chosen = candidates[:config.max_count]

    if len(chosen) >= 2:
        points = np.array([[item.x, item.y] for item in chosen], dtype=np.float64)
        if _spread(points, width, height) < config.min_spread:
            reject("chụm vào một chỗ")
            chosen = []

    return FiducialResult(fiducials=chosen, rejected=rejected, threshold=threshold)
