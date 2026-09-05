"""Kiểm một bộ ảnh chụp nhiều khung NGAY TẠI CHỖ, trước khi rời khỏi bo.

Phát hiện bộ ảnh hỏng lúc còn đứng cạnh bo thì chụp lại được trong năm phút.
Về tới nhà mới biết thì mất cả buổi đi lại — đó là lý do script này tồn tại
thay vì một mục hướng dẫn trong tài liệu.

Kiểm sáu điều kiện ở ``Docs/ke_hoach/ke_hoach_golden_ghep_so_do.md`` §8.4, trừ
điều 6 (vật chuẩn dài) vì máy không tự nhận ra cái thước trong ảnh.

**Mọi phép so sáng/màu/nét đều chạy trên VÙNG CHỒNG, không trên cả khung.**
Bản đầu của script này so trên cả khung và báo sai ngay trên một bộ ảnh cắt từ
CÙNG một tấm: vùng nhiều linh kiện thì sáng hơn và nét hơn vùng board trống,
nên nó đo khác biệt *nội dung* rồi gọi đó là khác biệt *phơi sáng*. Chỉ khi so
đúng một mảnh board vật lý nhìn từ hai khung thì chênh lệch mới là do máy.

    python scripts/check_capture_set.py <thư mục ảnh>

Mã thoát khác 0 nghĩa là **chụp lại**.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
#: Chồng biên tối thiểu với **ít nhất một** khung kề. Không đòi mọi cặp chồng
#: nhau: trong lưới 2x2 thì hai khung chéo góc chồng rất ít, và đó là bình
#: thường. 15% là con số tài liệu phần cứng dùng để tính số khung.
MIN_OVERLAP = 0.15
#: Dưới mức này coi là hai khung KHÔNG kề nhau, không phải chồng biên thiếu.
NEIGHBOUR_FLOOR = 0.03
#: Số điểm khớp nội tại tối thiểu. Ba điểm đủ cho affine; đòi nhiều hơn vì ảnh
#: mạch có tỉ lệ khớp nhầm cao (chữ in lặp lại, pad giống hệt nhau).
MIN_INLIERS = 12
#: Lệch độ sáng CÙNG MỘT MẢNG BOARD nhìn từ hai khung, mức xám 0-255. Bước 6.2
#: chấm mối hàn bằng ngưỡng ảnh, nên hai khung lệch sáng cho hai phán quyết
#: khác nhau trên cùng một mối hàn.
MAX_BRIGHTNESS_DELTA = 8.0
#: Nét: tỉ số phương sai Laplacian trên cùng vùng chồng. Dưới mức này nghĩa là
#: một trong hai khung mất nét thật, không phải do nội dung khác nhau.
MIN_FOCUS_RATIO = 0.6


@dataclass(slots=True)
class Frame:
    path: Path
    width: int
    height: int


@dataclass(slots=True)
class Pair:
    a: str
    b: str
    overlap: float
    inliers: int
    brightness_delta: float = 0.0
    colour_delta: float = 0.0
    focus_ratio: float = 1.0


@dataclass(slots=True)
class Report:
    frames: list[Frame] = field(default_factory=list)
    pairs: list[Pair] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _prepared(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Ảnh màu đã thu nhỏ, và bản xám của nó. Thu nhỏ cho nhanh."""

    image = cv2.imread(str(path))
    if image is None:
        return None, None
    scale = 1200.0 / max(image.shape[:2])
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    return image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _compare(path_a: Path, path_b: Path) -> Pair | None:
    """So hai khung TRÊN VÙNG CHỒNG của chúng."""

    colour_a, grey_a = _prepared(path_a)
    colour_b, grey_b = _prepared(path_b)
    if grey_a is None or grey_b is None:
        return None

    orb = cv2.ORB_create(nfeatures=4000)
    ka, da = orb.detectAndCompute(grey_a, None)
    kb, db = orb.detectAndCompute(grey_b, None)
    if da is None or db is None or len(ka) < MIN_INLIERS or len(kb) < MIN_INLIERS:
        return None

    raw = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(db, da, k=2)
    good = [m for pair in raw if len(pair) == 2
            for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < MIN_INLIERS:
        return None

    src = np.float32([kb[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ka[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if matrix is None or mask is None or int(mask.sum()) < MIN_INLIERS:
        return None
    inliers = int(mask.sum())

    height_a, width_a = grey_a.shape
    height_b, width_b = grey_b.shape
    warped = cv2.warpPerspective(colour_b, matrix, (width_a, height_a))
    valid = cv2.warpPerspective(
        np.full((height_b, width_b), 255, np.uint8), matrix, (width_a, height_a)
    )
    # Bào mép: pixel sát biên vùng warp bị nội suy lẫn với nền đen.
    valid = cv2.erode(valid, np.ones((9, 9), np.uint8))
    overlap = float((valid > 0).mean())
    pair = Pair(a=path_a.name, b=path_b.name, overlap=overlap, inliers=inliers)
    if overlap <= NEIGHBOUR_FLOOR:
        return pair

    region = valid > 0
    grey_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    pair.brightness_delta = abs(
        float(np.median(grey_a[region])) - float(np.median(grey_warped[region]))
    )
    pair.colour_delta = max(
        abs(float(colour_a[:, :, i][region].mean())
            - float(warped[:, :, i][region].mean()))
        for i in range(3)
    )
    # Nét phải đo trên PIXEL GỐC của cả hai ảnh, không trên ảnh đã warp.
    # ``warpPerspective`` nội suy song tuyến nên tự làm mềm ảnh B: đo trên bản
    # warp thì mọi cặp đều ra tỉ số < 1 kể cả khi hai khung nét như nhau. Đã
    # thấy đúng chuyện đó — bộ ảnh cắt từ CÙNG một tấm vẫn báo lệch nét 40%.
    #
    # Nên: đưa mặt nạ vùng chồng NGƯỢC về hệ toạ độ của B, rồi đo B trên chính
    # pixel của nó.
    inverse = np.linalg.inv(matrix)
    region_b = cv2.warpPerspective(valid, inverse, (width_b, height_b)) > 0
    focus_a = float(cv2.Laplacian(grey_a, cv2.CV_64F)[region].var())
    focus_b = (float(cv2.Laplacian(grey_b, cv2.CV_64F)[region_b].var())
               if region_b.any() else 0.0)
    pair.focus_ratio = (min(focus_a, focus_b) / max(focus_a, focus_b)
                        if max(focus_a, focus_b) > 0 else 1.0)
    return pair


def check(paths: list[Path]) -> Report:
    report = Report()
    for path in sorted(paths):
        image = cv2.imread(str(path))
        if image is None:
            report.problems.append(f"không đọc được ảnh: {path.name}")
            continue
        report.frames.append(Frame(path, image.shape[1], image.shape[0]))

    if len(report.frames) < 2:
        report.problems.append(
            f"chỉ có {len(report.frames)} ảnh đọc được; cần ít nhất 2 để ghép"
        )
        return report

    sizes = {(f.width, f.height) for f in report.frames}
    if len(sizes) > 1:
        report.problems.append(
            f"ảnh KHÔNG cùng kích thước: {sorted(sizes)}. Đổi kích thước giữa "
            "các khung thường là đã đổi zoom hoặc đổi máy — tỉ lệ mm/px sẽ "
            "khác nhau giữa các vùng của sơ đồ."
        )

    neighbours: dict[str, set[str]] = {f.path.name: set() for f in report.frames}
    strong: dict[str, set[str]] = {f.path.name: set() for f in report.frames}
    for a, b in combinations(report.frames, 2):
        pair = _compare(a.path, b.path)
        if pair is None or pair.overlap <= NEIGHBOUR_FLOOR:
            continue
        report.pairs.append(pair)
        neighbours[pair.a].add(pair.b)
        neighbours[pair.b].add(pair.a)
        if pair.overlap >= MIN_OVERLAP:
            strong[pair.a].add(pair.b)
            strong[pair.b].add(pair.a)

        if pair.brightness_delta > MAX_BRIGHTNESS_DELTA:
            report.problems.append(
                f"{pair.a} ↔ {pair.b}: CÙNG một mảng board mà lệch "
                f"{pair.brightness_delta:.1f} mức sáng. Khoá phơi sáng (AE lock) "
                "rồi chụp lại — 6.2 chấm mối hàn bằng ngưỡng ảnh."
            )
        elif pair.colour_delta > MAX_BRIGHTNESS_DELTA:
            report.problems.append(
                f"{pair.a} ↔ {pair.b}: cùng một mảng board mà lệch màu "
                f"{pair.colour_delta:.1f} mức. Khoá cân bằng trắng."
            )
        # Chỉ xét nét trên cặp CHỒNG NHIỀU: vùng chồng bé thì ước lượng
        # phương sai nhiễu, và một cặp chéo góc 11% không nói được gì về tiêu cự.
        if pair.overlap >= MIN_OVERLAP and pair.focus_ratio < MIN_FOCUS_RATIO:
            report.problems.append(
                f"{pair.a} ↔ {pair.b}: cùng một mảng board mà độ nét chênh "
                f"{pair.focus_ratio:.0%}. Một trong hai khung mất nét — lấy nét "
                "lại rồi chụp lại."
            )

    for frame in report.frames:
        name = frame.path.name
        if not neighbours[name]:
            report.problems.append(
                f"{name} không chồng với khung nào. Hoặc nó chụp vùng khác hẳn, "
                "hoặc không đủ điểm nhận dạng chung với khung kề."
            )
        elif not strong[name]:
            best = max((p.overlap for p in report.pairs
                        if name in (p.a, p.b)), default=0.0)
            report.problems.append(
                f"{name} chồng nhiều nhất chỉ {best:.0%} với một khung "
                f"(cần ≥ {MIN_OVERLAP:.0%} với ít nhất một khung kề). Dịch bo "
                "ít hơn quanh khung này."
            )

    if report.frames and all(neighbours.values()):
        start = report.frames[0].path.name
        reached, stack = {start}, [start]
        while stack:
            for nxt in neighbours[stack.pop()]:
                if nxt not in reached:
                    reached.add(nxt)
                    stack.append(nxt)
        if len(reached) < len(report.frames):
            report.problems.append(
                f"bộ ảnh tách thành nhiều cụm rời (nối được {len(reached)}/"
                f"{len(report.frames)}). Cần thêm khung bắc cầu giữa các cụm."
            )

    report.notes.append(
        "KHÔNG kiểm được bằng máy: có vật chuẩn dài đã biết trong ít nhất một "
        "khung không (§8.4 điều 6). Thiếu nó thì sơ đồ ra chỉ có đơn vị pixel, "
        "không có mm."
    )
    report.notes.append(
        "KHÔNG kiểm được bằng máy: máy có vuông góc với mặt bo không. Nghiêng "
        "máy làm phối cảnh tệ hơn — xem §8.2."
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    args = parser.parse_args(argv)

    folder = Path(args.folder)
    if not folder.is_absolute():
        folder = (PROJECT_ROOT / folder).resolve()
    if not folder.is_dir():
        raise SystemExit(f"không thấy thư mục {folder}")
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUFFIXES)
    if not paths:
        raise SystemExit(f"không có ảnh nào trong {folder}")

    report = check(paths)
    print(f"{len(report.frames)} ảnh trong {folder.name}\n")

    if report.pairs:
        print(f"{'cặp khung (so trên VÙNG CHỒNG)':52s} {'chồng':>6s} "
              f"{'Δsáng':>6s} {'Δmàu':>6s} {'nét':>6s} {'điểm':>6s}")
        for pair in sorted(report.pairs, key=lambda p: -p.overlap):
            print(f"{pair.a[:24]:24s} ↔ {pair.b[:24]:24s} {pair.overlap:6.0%} "
                  f"{pair.brightness_delta:6.1f} {pair.colour_delta:6.1f} "
                  f"{pair.focus_ratio:6.0%} {pair.inliers:6d}")

    print()
    for note in report.notes:
        print(f"  ⚠️  {note}")

    if report.problems:
        print(f"\nCHỤP LẠI — {len(report.problems)} vấn đề:")
        for problem in report.problems:
            print(f"  ✗ {problem}")
        return 1
    print("\nĐẠT — bộ ảnh này ghép được (trừ hai điều máy không kiểm được ở trên).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
