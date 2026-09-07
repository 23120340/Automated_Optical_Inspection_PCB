"""Cháy sáng phải tách được khỏi *sáng*.

Phép đo này quyết định tile nào bị loại khỏi bộ khoanh nhãn, nên hai kiểu sai
của nó có giá khác hẳn nhau:

* bắt nhầm vùng còn chi tiết → ném đi ảnh dùng được, và tệ nhất là ném đúng
  **pad mạ vàng lộ thiên** — lớp mà bo dự án đang cần dạy cho model;
* bỏ sót vùng đã cháy → đưa vào bộ nhãn một khoảng trắng không có gì để khoanh.

Nên các bài dưới đây đều bám vào một câu: *chỗ này còn thông tin hay không*.
"""

from __future__ import annotations

import numpy as np
import pytest

from aoi_pipeline.imaging.exposure import (
    blown_fraction,
    blown_highlight_mask,
)


def _canvas(value: int = 40, size: int = 120) -> np.ndarray:
    return np.full((size, size, 3), value, np.uint8)


def test_a_flat_white_patch_is_blown() -> None:
    """Trắng phẳng: không còn gì để khoanh."""

    image = _canvas()
    image[20:60, 20:60] = 255
    mask = blown_highlight_mask(image)
    assert mask[35:45, 35:45].all(), "giữa mảng trắng phẳng phải bị coi là cháy"


def test_a_bright_patch_that_still_has_texture_is_not_blown() -> None:
    """Sáng mà còn vân thì vẫn dùng được — connector nhựa kem là ca thật.

    Đây là bài quan trọng nhất: chỉ xét độ sáng thì ca này bị loại oan.
    """

    image = _canvas()
    patch = np.full((40, 40, 3), 250, np.uint8)
    patch[::2] = 180  # vân sọc: sáng nhưng còn chi tiết
    image[20:60, 20:60] = patch
    assert blown_fraction(image) == 0.0


def test_a_gold_ring_around_a_via_survives() -> None:
    """Pad mạ vàng lộ thiên KHÔNG được coi là cháy hết.

    Đó đúng là lớp thị giác bo dự án đang sai (32% box lượt 1 là pad tròn), nên
    loại nó khỏi bộ nhãn là tự tay bỏ mất thứ duy nhất dạy được model.
    """

    import cv2

    image = _canvas(size=200)
    # Pad mạ vàng: vành sáng, lỗ tối ở giữa — còn chi tiết.
    cv2.circle(image, (60, 100), 22, (60, 190, 235), -1)
    cv2.circle(image, (60, 100), 11, (25, 25, 25), -1)
    # Vệt loá trắng phẳng ở nửa kia, để bài test phải PHÂN BIỆT chứ không chỉ
    # cần một mặt nạ trống là qua.
    image[70:130, 130:190] = 255

    mask = blown_highlight_mask(image).astype(bool)
    ring = np.zeros(image.shape[:2], np.uint8)
    cv2.circle(ring, (60, 100), 22, 1, -1)
    glare = np.zeros(image.shape[:2], bool)
    glare[80:120, 140:180] = True

    assert mask[glare].mean() > 0.9, "vệt loá phẳng phải bị bắt"
    assert mask[ring.astype(bool)].mean() < 0.5, (
        "vành pad vàng còn chi tiết, không được coi là cháy hết"
    )


def test_a_dark_board_has_nothing_blown() -> None:
    rng = np.random.default_rng(11)
    board = rng.integers(20, 70, size=(120, 120, 3), dtype=np.uint8)
    assert blown_fraction(board) == 0.0


def test_the_fraction_is_the_share_of_blown_pixels() -> None:
    image = _canvas(size=100)
    image[0:50, :] = 255
    fraction = blown_fraction(image)
    # Mép giữa vùng trắng và nền tối còn chi tiết nên không tính là cháy; phần
    # lõi thì có. Kiểm khoảng chứ không kiểm một con số chính xác giả.
    assert 0.40 < fraction < 0.50


def test_a_saturated_but_noisy_patch_is_not_blown() -> None:
    """Nhiễu ở mức gần trần vẫn là thông tin — chưa clip hẳn."""

    rng = np.random.default_rng(3)
    image = _canvas()
    image[20:60, 20:60] = rng.integers(200, 256, size=(40, 40, 3), dtype=np.uint8)
    assert blown_fraction(image) < 0.02


@pytest.mark.parametrize("window", [2, 4, 8])
def test_an_even_window_is_refused(window: int) -> None:
    """Cửa sổ chẵn không có tâm, nên mặt nạ lệch đi nửa pixel."""

    with pytest.raises(ValueError):
        blown_highlight_mask(_canvas(), window=window)


def test_a_negative_flat_level_is_refused() -> None:
    with pytest.raises(ValueError):
        blown_highlight_mask(_canvas(), flat_level=-1.0)


def test_a_flat_saturated_blue_surface_is_not_blown() -> None:
    """Tản nhiệt nhôm anod xanh: kênh lam chạm trần, mặt phẳng lì — nhưng KHÔNG cháy.

    Ca thật, và là lý do vế "sáng trắng" được thêm vào. Bản đầu chỉ xét kênh sáng
    nhất cộng độ phẳng, và nó chấm một tile bo dự án là "cháy 16%" trong khi phần
    đó là tản nhiệt: thông tin còn nguyên, chỉ là một màu. Loại tile ấy đi thì mất
    luôn nửa dưới của nó — chỗ có BGA và pad xuyên lỗ mạ vàng.
    """

    image = np.zeros((120, 120, 3), np.uint8)
    image[:, :] = (255, 90, 35)  # BGR: lam chạm trần, lục/đỏ thấp
    assert blown_fraction(image) == 0.0


def test_a_flat_saturated_red_surface_is_not_blown() -> None:
    """Cùng lý do, phía màu còn lại — để bài trên không chỉ đúng với riêng màu lam."""

    image = np.zeros((120, 120, 3), np.uint8)
    image[:, :] = (30, 40, 255)
    assert blown_fraction(image) == 0.0


def test_a_luminance_level_outside_the_byte_range_is_refused() -> None:
    with pytest.raises(ValueError):
        blown_highlight_mask(_canvas(), luminance_level=300)
