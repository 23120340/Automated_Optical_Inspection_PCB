"""Bộ kiểm ảnh chụp phải ĐẠT trên bộ tốt và BẮT được từng lỗi một.

Cả hai nửa đều cần. Một bộ kiểm luôn báo lỗi thì người dùng sẽ bỏ qua nó, và
một bộ kiểm luôn ĐẠT thì vô dụng — bản đầu của script này rơi vào vế thứ nhất:
nó so độ sáng trên CẢ KHUNG nên báo lỗi ngay trên bốn ô cắt từ cùng một tấm
ảnh, vì vùng nhiều linh kiện sáng hơn vùng board trống.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.check_capture_set import MIN_OVERLAP, check

TILES = Path(__file__).resolve().parents[1] / "datasets" / "test_images" / "tiles_1024"


def _source() -> np.ndarray:
    """Một ảnh mạch thật, đủ lớn để cắt thành nhiều khung chồng nhau."""

    candidates = sorted(TILES.glob("*.png"))
    if not candidates:
        pytest.skip("không có tile nào để dựng bộ ảnh thử")
    image = cv2.imread(str(candidates[0]))
    if image is None:
        pytest.skip("không đọc được tile")
    # Phóng to để bốn khung con vẫn đủ chi tiết cho ORB.
    return cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)


def _write_set(folder: Path, damage: dict[int, str] | None = None) -> list[Path]:
    """Bốn khung phủ ảnh nguồn theo lưới 2x2, chồng biên ~33%."""

    source = _source()
    height, width = source.shape[:2]
    tile_w, tile_h = int(width * 0.6), int(height * 0.6)
    paths = []
    for index, (fx, fy) in enumerate([(0, 0), (0.4, 0), (0, 0.4), (0.4, 0.4)]):
        x, y = int(fx * width), int(fy * height)
        patch = source[y:y + tile_h, x:x + tile_w].copy()
        kind = (damage or {}).get(index)
        if kind == "bright":
            patch = cv2.convertScaleAbs(patch, alpha=1.0, beta=45)
        elif kind == "blur":
            patch = cv2.GaussianBlur(patch, (15, 15), 0)
        elif kind == "resize":
            patch = cv2.resize(patch, None, fx=0.8, fy=0.8)
        path = folder / f"frame_{index}.png"
        cv2.imwrite(str(path), patch)
        paths.append(path)
    return paths


def test_a_clean_set_passes(tmp_path: Path) -> None:
    """Bốn ô cắt từ CÙNG một tấm thì mọi thiết lập máy giống hệt nhau.

    Bộ kiểm phải ĐẠT. Nếu nó báo lỗi ở đây thì nó đang đo khác biệt nội dung
    chứ không đo khác biệt máy.
    """

    report = check(_write_set(tmp_path))
    assert not report.problems, report.problems
    assert len(report.frames) == 4


def test_the_overlap_is_measured_not_assumed(tmp_path: Path) -> None:
    report = check(_write_set(tmp_path))
    strong = [p for p in report.pairs if p.overlap >= MIN_OVERLAP]
    assert strong, "lưới 2x2 chồng 33% lẽ ra phải có cặp đạt ngưỡng"
    assert all(p.inliers > 0 for p in strong)


def test_an_unlocked_exposure_is_caught(tmp_path: Path) -> None:
    """Đây là lỗi tốn kém nhất: 6.2 chấm mối hàn bằng ngưỡng ảnh, nên hai khung
    lệch sáng cho hai phán quyết khác nhau trên CÙNG một mối hàn."""

    report = check(_write_set(tmp_path, {1: "bright"}))
    assert any("mức sáng" in p for p in report.problems), report.problems


def test_a_soft_frame_is_caught(tmp_path: Path) -> None:
    report = check(_write_set(tmp_path, {2: "blur"}))
    assert any("độ nét" in p for p in report.problems), report.problems


def test_a_changed_zoom_is_caught(tmp_path: Path) -> None:
    """Đổi kích thước ảnh giữa các khung nghĩa là đã đổi zoom hoặc đổi máy —
    tỉ lệ mm/px sẽ khác nhau giữa các vùng của sơ đồ."""

    report = check(_write_set(tmp_path, {3: "resize"}))
    assert any("KHÔNG cùng kích thước" in p for p in report.problems), report.problems


def test_two_unrelated_photos_are_reported_as_unconnected(tmp_path: Path) -> None:
    source = _source()
    height, width = source.shape[:2]
    cv2.imwrite(str(tmp_path / "a.png"), source[: height // 3, : width // 3])
    cv2.imwrite(str(tmp_path / "b.png"), source[-height // 3:, -width // 3:])
    report = check(sorted(tmp_path.glob("*.png")))
    assert any("không chồng với khung nào" in p for p in report.problems), (
        report.problems
    )


def test_a_single_photo_cannot_be_merged(tmp_path: Path) -> None:
    source = _source()
    cv2.imwrite(str(tmp_path / "only.png"), source[:400, :400])
    report = check(sorted(tmp_path.glob("*.png")))
    assert any("ít nhất 2" in p for p in report.problems), report.problems


def test_the_two_things_a_machine_cannot_check_are_always_said(tmp_path: Path) -> None:
    """Vật chuẩn dài và độ vuông góc — máy không kiểm được, nên phải nhắc mọi
    lần, kể cả khi ĐẠT. Bộ ảnh ĐẠT mà thiếu thước thì sơ đồ chỉ có đơn vị px."""

    report = check(_write_set(tmp_path))
    assert len(report.notes) == 2
    assert any("vật chuẩn" in n for n in report.notes)
    assert any("vuông góc" in n for n in report.notes)


def test_focus_is_measured_on_native_pixels_not_on_the_warped_copy(
    tmp_path: Path,
) -> None:
    """Chốt riêng cho một lỗi đã xảy ra thật, và ngưỡng chung không bắt được.

    Bản đầu đo độ nét trên ảnh B **đã warp**. ``warpPerspective`` nội suy song
    tuyến nên tự làm mềm B, và mọi cặp đều ra tỉ số < 1 kể cả khi hai khung nét
    như nhau — đo được 0,667–0,832 trên bốn ô cắt từ CÙNG một tấm.

    ``MIN_FOCUS_RATIO`` là 0,6 nên nó **không** báo lỗi, tức bug đi qua im
    lặng và chỉ làm số liệu sai. Vì vậy phải canh bằng một ngưỡng chặt riêng
    chứ không dựa vào ngưỡng cảnh báo chung.
    """

    report = check(_write_set(tmp_path))
    clean = [p.focus_ratio for p in report.pairs if p.overlap >= MIN_OVERLAP]
    assert clean, "cần ít nhất một cặp chồng nhiều để đo"
    assert min(clean) > 0.95, (
        f"cặp cắt từ cùng một tấm phải ~1.0, đo được {min(clean):.3f}. "
        "Dấu hiệu độ nét đang bị đo trên bản đã warp."
    )
