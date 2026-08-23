"""Step 3: contour-based PCB localization with an explicit full-frame fallback."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from typing import Sequence

from ..config import BoardConfig, FiducialConfig
from ..exceptions import AOIPipelineError
from .image_io import ensure_bgr
from ..models import BoardRegion, BoundingBox


@dataclass(slots=True)
class _Candidate:
    contour: np.ndarray
    mask: np.ndarray
    mask_name: str
    area_ratio: float
    rectangularity: float
    score: float


class PCBLocalizer:
    def __init__(
        self,
        config: BoardConfig | None = None,
        fiducial_config: FiducialConfig | None = None,
    ) -> None:
        self.config = config or BoardConfig()
        self.fiducial_config = fiducial_config or FiducialConfig()

    def locate(
        self,
        image: np.ndarray,
        fiducials: Sequence[tuple[float, float]] | None = None,
    ) -> BoardRegion:
        """Khoanh vùng board.

        ``fiducials`` là toạ độ các mốc trong pixel ảnh, nếu có. Ba mốc trở lên
        cho một vùng board **chắc chắn hơn contour**: contour đi tìm mảng lớn
        nhất trông giống hình chữ nhật, và nó hỏng khi nền cùng màu với board,
        khi board bị che một phần, hoặc khi trong ảnh có nhiều board.

        Fiducial nằm LÙI VÀO so với mép board nên bao lồi của chúng nhỏ hơn
        board thật; phần nới ra là ước lượng, không phải đo — xem
        ``FiducialConfig.board_margin_ratio``.
        """

        bgr = ensure_bgr(image)
        height, width = bgr.shape[:2]
        if fiducials and len(fiducials) >= 3:
            return _region_from_fiducials(fiducials, width, height, self.fiducial_config)
        image_area = float(height * width)
        masks = _candidate_masks(bgr, self.config.morphology_kernel_ratio)
        candidates: list[_Candidate] = []

        for mask_name, mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = abs(float(cv2.contourArea(contour)))
                area_ratio = area / image_area
                if not self.config.min_area_ratio <= area_ratio <= self.config.max_area_ratio:
                    continue
                x, y, box_width, box_height = cv2.boundingRect(contour)
                touches_every_edge = (
                    x <= 1
                    and y <= 1
                    and x + box_width >= width - 1
                    and y + box_height >= height - 1
                )
                # Otsu on a flat frame produces a white, full-canvas contour.
                # Treat it as "board unknown/full ROI" instead of a confident detection.
                if touches_every_edge and area_ratio > 0.90:
                    continue
                rectangle = cv2.minAreaRect(contour)
                rectangle_area = max(float(rectangle[1][0] * rectangle[1][1]), 1.0)
                rectangularity = min(area / rectangle_area, 1.0)
                if rectangularity < self.config.min_rectangularity:
                    continue
                score = _candidate_score(contour, area_ratio, rectangularity, width, height)
                candidates.append(
                    _Candidate(contour, mask, mask_name, area_ratio, rectangularity, score)
                )

        if not candidates:
            if not self.config.fallback_to_full_image:
                raise AOIPipelineError("No PCB-like contour was found in the image")
            return _full_image_region(width, height)

        best = max(candidates, key=lambda candidate: candidate.score)
        rectangle = cv2.minAreaRect(best.contour)
        polygon_array = cv2.boxPoints(rectangle)
        polygon = [(float(point[0]), float(point[1])) for point in polygon_array]
        x, y, box_width, box_height = cv2.boundingRect(best.contour)
        padding = int(round(max(width, height) * max(0.0, self.config.padding_ratio)))
        bbox = BoundingBox(x - padding, y - padding, x + box_width + padding, y + box_height + padding)
        bbox = bbox.clamp(width, height)

        board_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(board_mask, [best.contour], -1, 255, thickness=cv2.FILLED)

        return BoardRegion(
            bbox=bbox,
            polygon=polygon,
            confidence=float(np.clip(best.score, 0.0, 0.99)),
            method=f"contour:{best.mask_name}",
            mask=board_mask,
            metadata={
                "area_ratio": best.area_ratio,
                "rectangularity": best.rectangularity,
                "candidate_count": len(candidates),
            },
        )


def _candidate_masks(image: np.ndarray, kernel_ratio: float) -> list[tuple[str, np.ndarray]]:
    height, width = image.shape[:2]
    kernel_size = max(3, int(round(min(height, width) * max(0.002, kernel_ratio))))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
    _, saturation_mask = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bright_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_mask = cv2.bitwise_not(bright_mask)

    median = float(np.median(blurred))
    lower = int(max(0.0, 0.66 * median))
    upper = int(min(255.0, max(lower + 1, 1.33 * median)))
    edge_mask = cv2.Canny(blurred, lower, upper)
    edge_mask = cv2.dilate(edge_mask, small_kernel, iterations=1)

    output: list[tuple[str, np.ndarray]] = []
    for name, mask in (
        ("saturation", saturation_mask),
        ("brightness", bright_mask),
        ("darkness", dark_mask),
        ("edges", edge_mask),
    ):
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, small_kernel, iterations=1)
        output.append((name, cleaned))
    return output


def _candidate_score(
    contour: np.ndarray,
    area_ratio: float,
    rectangularity: float,
    width: int,
    height: int,
) -> float:
    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-6:
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]
    else:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        center_x, center_y = x + box_width / 2.0, y + box_height / 2.0
    normalized_distance = np.hypot(
        (center_x - width / 2.0) / max(width / 2.0, 1.0),
        (center_y - height / 2.0) / max(height / 2.0, 1.0),
    )
    center_score = float(np.clip(1.0 - normalized_distance / np.sqrt(2.0), 0.0, 1.0))
    area_score = float(np.clip(area_ratio / 0.45, 0.0, 1.0))

    x, y, box_width, box_height = cv2.boundingRect(contour)
    touches = sum(
        (
            x <= 1,
            y <= 1,
            x + box_width >= width - 1,
            y + box_height >= height - 1,
        )
    )
    border_score = 1.0 - 0.12 * touches
    return float(
        np.clip(
            0.38 * area_score
            + 0.34 * rectangularity
            + 0.18 * center_score
            + 0.10 * border_score,
            0.0,
            1.0,
        )
    )


def _full_image_region(width: int, height: int) -> BoardRegion:
    return BoardRegion(
        bbox=BoundingBox(0.0, 0.0, float(width), float(height)),
        polygon=[(0.0, 0.0), (float(width), 0.0), (float(width), float(height)), (0.0, float(height))],
        confidence=0.10,
        method="full_image_fallback",
        mask=np.full((height, width), 255, dtype=np.uint8),
        metadata={"reason": "No contour passed the PCB geometry filters"},
    )


def _region_from_fiducials(
    points: Sequence[tuple[float, float]],
    width: int,
    height: int,
    config: FiducialConfig,
) -> BoardRegion:
    """Vùng board suy từ các mốc.

    Bao lồi của các mốc, nới ra một biên. Phần nới là **ước lượng**: fiducial
    được đặt lùi vào trong mép board (thường 5 mm) và bản vẽ không nằm trong
    ảnh, nên không có cách nào đo phần lùi đó từ chính tấm ảnh. `confidence`
    vì thế cố tình không phải 1.0.
    """

    array = np.array(points, dtype=np.float64).reshape(-1, 2)
    x1, y1 = array.min(axis=0)
    x2, y2 = array.max(axis=0)
    margin = max(width, height) * max(0.0, config.board_margin_ratio)
    bbox = BoundingBox(
        float(x1 - margin), float(y1 - margin),
        float(x2 + margin), float(y2 + margin),
    ).clamp(width, height)

    mask = np.zeros((height, width), dtype=np.uint8)
    corners = np.array([
        [bbox.x1, bbox.y1], [bbox.x2, bbox.y1],
        [bbox.x2, bbox.y2], [bbox.x1, bbox.y2],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [corners], 255)

    return BoardRegion(
        bbox=bbox,
        polygon=[(float(px), float(py)) for px, py in corners],
        # Không phải 1.0 có chủ ý: các mốc chốt được hệ toạ độ, nhưng biên nới
        # ra tới mép board là ước lượng.
        confidence=0.90,
        method=f"fiducial:{len(array)}",
        mask=mask,
        metadata={
            "fiducial_count": int(len(array)),
            "margin_px": round(float(margin), 1),
            "note": "Biên nới ra là ước lượng; fiducial nằm lùi vào trong mép board.",
        },
    )
