"""Fiducial mark: neo vùng board mà không cần ảnh golden.

Bước 3 vốn khoanh board bằng contour — tìm mảng lớn nhất trông giống hình chữ
nhật. Cách đó hỏng ở đúng những lúc hay gặp: nền cùng màu board, board bị che
một phần, hoặc nhiều board trong một khung. Fiducial thì cho một hệ toạ độ chứ
không chỉ một hình chữ nhật bao quanh.

Điều quan trọng nhất được kiểm ở đây không phải "tìm được bao nhiêu" mà là
**không nhận nhầm**: một mốc sai thì cả hệ toạ độ sai theo, và sai im lặng.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from aoi_pipeline.config import BoardConfig, FiducialConfig
from aoi_pipeline.imaging.board import PCBLocalizer
from aoi_pipeline.imaging.fiducials import Fiducial, find_fiducials


def _board(width: int = 800, height: int = 600) -> np.ndarray:
    """Nền xanh mask như board thật, có nhiễu nhẹ."""

    rng = np.random.default_rng(4)
    image = np.zeros((height, width, 3), np.uint8)
    image[:, :] = (40, 70, 45)
    noise = rng.integers(-6, 6, image.shape, dtype=np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _draw_ring(image: np.ndarray, centre, outer: int = 10) -> None:
    """Fiducial thật: VÀNH đồng sáng quanh lõi tối.

    Đây là hình dạng đo được trên board của dự án (nhãn FIDU4/FIDU6), không
    phải một đĩa sáng đặc — và sự khác biệt đó là lý do bộ dò đầu tiên trượt.
    """

    cv2.circle(image, centre, outer, (205, 210, 215), -1)
    cv2.circle(image, centre, max(2, outer // 2), (25, 30, 28), -1)


def _draw_disc(image: np.ndarray, centre, radius: int = 10) -> None:
    """Đốm loé trên linh kiện: sáng ĐẶC ở giữa."""

    cv2.circle(image, centre, radius, (235, 240, 245), -1)


# --------------------------------------------------------------------------
# Nhận đúng thứ cần nhận
# --------------------------------------------------------------------------


def test_three_ring_marks_are_found_and_usable() -> None:
    image = _board()
    for centre in ((120, 90), (680, 100), (140, 500)):
        _draw_ring(image, centre)

    result = find_fiducials(image)

    assert result.usable, f"chỉ tìm được {len(result.fiducials)}: {result.rejected}"
    found = sorted((round(f.x), round(f.y)) for f in result.fiducials)
    for expected in ((120, 90), (140, 500), (680, 100)):
        assert any(abs(x - expected[0]) <= 3 and abs(y - expected[1]) <= 3
                   for x, y in found), f"không thấy mốc ở {expected}: {found}"


def test_a_solid_bright_disc_is_not_mistaken_for_a_fiducial() -> None:
    """Phép thử quyết định. Fiducial là vành sáng quanh lõi TỐI; đốm loé trên
    linh kiện thì sáng đặc ở giữa. Bỏ phép thử này đi thì bộ dò nhận toàn đốm
    loé — đã đo trên board thật: 12 dương tính giả."""

    image = _board()
    for centre in ((200, 150), (600, 160), (220, 450)):
        _draw_disc(image, centre)

    result = find_fiducials(image)

    assert not result.usable
    assert result.rejected.get("không có vành quanh lõi tối", 0) >= 3


def test_marks_clustered_in_one_corner_are_refused() -> None:
    """Ba đốm chụm một chỗ thường là một linh kiện bóng. Fiducial được đặt xa
    nhau có chủ đích — đó là thứ chốt được góc xoay."""

    image = _board()
    for centre in ((100, 100), (130, 110), (115, 135)):
        _draw_ring(image, centre)

    result = find_fiducials(image)

    assert not result.usable
    assert "chụm vào một chỗ" in result.rejected


def test_two_marks_are_not_enough() -> None:
    """Hai mốc giải được một phép biến đổi tương tự, nhưng không phát hiện
    được khi một trong hai là dương tính giả."""

    image = _board()
    _draw_ring(image, (120, 90))
    _draw_ring(image, (680, 500))

    result = find_fiducials(image)
    assert len(result.fiducials) <= 2
    assert not result.usable


def test_it_says_why_candidates_were_rejected() -> None:
    """Một bộ dò chỉ nói "không thấy gì" thì không sửa được: người dùng không
    biết nên nới ngưỡng nào."""

    result = find_fiducials(_board())
    assert isinstance(result.rejected, dict)
    assert result.to_dict()["usable"] is False


# --------------------------------------------------------------------------
# Vùng board suy từ mốc
# --------------------------------------------------------------------------


def test_the_board_region_comes_from_the_marks_when_there_are_three() -> None:
    localizer = PCBLocalizer()
    region = localizer.locate(_board(), [(100, 80), (700, 90), (120, 520)])

    assert region.method == "fiducial:3"
    x1, y1, x2, y2 = region.bbox.to_int()
    # Bao lồi của mốc, nới ra một biên — nên rộng hơn chính các mốc.
    assert x1 < 100 and y1 < 80 and x2 > 700 and y2 > 520
    assert region.metadata["fiducial_count"] == 3


def test_the_region_confidence_is_not_certainty() -> None:
    """Các mốc chốt được hệ toạ độ, nhưng biên nới ra tới mép board là ước
    lượng: fiducial nằm lùi vào trong mép, và bản vẽ không có trong ảnh."""

    region = PCBLocalizer().locate(_board(), [(100, 80), (700, 90), (120, 520)])
    assert region.confidence < 1.0
    assert "ước lượng" in region.metadata["note"]


def test_fewer_than_three_marks_falls_back_to_the_old_path() -> None:
    """Đừng gửi đi một nửa dữ kiện: dưới 3 mốc thì không chốt được xoay."""

    localizer = PCBLocalizer()
    for marks in (None, [], [(100, 80)], [(100, 80), (700, 90)]):
        region = localizer.locate(_board(), marks)
        assert not region.method.startswith("fiducial"), marks


def test_the_pipeline_facade_passes_the_marks_through() -> None:
    from aoi_pipeline.pipeline import AOIPipeline

    region = AOIPipeline().detect_board(_board(), [(100, 80), (700, 90), (120, 520)])
    assert region.method == "fiducial:3"


# --------------------------------------------------------------------------
# Cấu hình
# --------------------------------------------------------------------------


def test_the_radius_band_covers_the_camera_range_the_project_plans_for() -> None:
    """Fiducial IPC-7351 đường kính ~1 mm. Ở 46 µm/px (thang đo được hiện nay)
    là ~22 px đường kính; ở 25 µm/px là ~40 px; ở 92 µm/px là ~11 px. Dải mặc
    định phải phủ cả ba, nếu không thì mỗi lần đổi ống kính là một lần dò hụt."""

    config = FiducialConfig()
    for micron_per_px, diameter_mm in ((25, 1.0), (46, 1.0), (92, 1.0)):
        radius_px = (diameter_mm * 1000 / micron_per_px) / 2
        assert config.min_radius_px <= radius_px <= config.max_radius_px, (
            f"{micron_per_px} µm/px cho bán kính {radius_px:.1f} px, ngoài dải "
            f"{config.min_radius_px}–{config.max_radius_px}"
        )


def test_the_dark_core_test_can_be_switched_off_deliberately() -> None:
    """Board dùng fiducial dạng đĩa đặc thì tắt được — nhưng phải tắt tường
    minh, không phải mặc định."""

    assert FiducialConfig().require_dark_core is True

    image = _board()
    for centre in ((200, 150), (600, 160), (220, 450)):
        _draw_disc(image, centre)
    relaxed = find_fiducials(image, FiducialConfig(require_dark_core=False))
    assert relaxed.usable
