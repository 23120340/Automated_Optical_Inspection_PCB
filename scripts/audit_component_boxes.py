"""Soi chất lượng box THÂN linh kiện do người duyệt vẽ, bằng số chứ không bằng cảm nhận.

Ba câu hỏi, ba phép đo khác nhau:

1. **Có box nào hỏng về hình học không?** trùng nhau, bé quá mức dùng được, vượt
   biên tile, tỉ lệ cạnh bất thường. Đây là phép đếm thuần tuý.

2. **Mép box có bám đúng mép thân không?** Phình/thu hộp từng pixel rồi đo
   gradient trung bình dọc viền; viền nào gradient mạnh nhất là mép thật.
   Lệch âm = mép thật nằm trong box (box quá to), lệch dương = ngược lại.

3. **Con số đó tốt hay tệ?** Không tự trả lời được. Nên cùng thước đo ấy chạy
   luôn trên một dataset công khai đã gán nhãn kiểu vỏ (Winnies) để lấy mốc.
   Thiếu mốc thì "45% bám đúng" là một con số không đọc được.

**Cạm bẫy đã đo được, đọc kỹ trước khi tin cột "quá bé":** thứ nằm ngay ngoài
thân linh kiện là *pad và chân* — cạnh sắc nét nhất trong cả vùng. Thước đo bị
chúng hút, nên một box ôm thân ĐÚNG quy ước vẫn bị chấm là "quá bé". Cột
``chạm trần`` trong báo cáo là để phát hiện đúng chuyện đó: đỉnh còn ở xa hơn
tầm quét thì nó không phải mép thân.

    python scripts/audit_component_boxes.py \\
        datasets/labelling/component_bodies \\
        --boxes "~/Downloads/joint_boxes (3).json"
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics as st
import sys
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

WINNIES_ZIP = (
    PROJECT_ROOT / "datasets" / "public" / "pcb_packages_winnies" / "export_yolov8_v3.zip"
)
#: Dưới ngưỡng này viền chỉ còn vài pixel, gradient đo ra là nhiễu chứ không phải mép.
MIN_SHORT_SIDE_PX = 12
#: Đỉnh phải nhô hơn nền bằng này thì mới coi là "có mép", không thì bỏ qua.
PEAK_OVER_BASELINE = 1.15


def _gradient(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.magnitude(
        cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3),
    )


def edge_offset(
    mag: np.ndarray, x: float, y: float, w: float, h: float
) -> tuple[int, int] | None:
    """Lệch giữa mép box và mép sáng nhất, kèm tầm quét đã dùng.

    Tầm quét co theo cỡ linh kiện: quét ±8 px quanh một con trở 14 px là quét
    quá nửa linh kiện, và khi đó phép đo trả về hàng xóm chứ không phải mép.
    """

    height, width = mag.shape
    reach = int(max(2, min(8, round(0.35 * min(w, h)))))
    if x - reach < 0 or y - reach < 0 or x + w + reach > width or y + h + reach > height:
        return None
    profile: list[float] = []
    for offset in range(-reach, reach + 1):
        x0, y0 = int(round(x - offset)), int(round(y - offset))
        x1, y1 = int(round(x + w + offset)), int(round(y + h + offset))
        if x1 - x0 < 4 or y1 - y0 < 4:
            profile.append(float("nan"))
            continue
        ring = np.concatenate(
            [mag[y0, x0:x1], mag[y1 - 1, x0:x1], mag[y0:y1, x0], mag[y0:y1, x1 - 1]]
        )
        profile.append(float(np.mean(ring)))
    values = np.array(profile, dtype=np.float64)
    if np.all(np.isnan(values)):
        return None
    peak, baseline = np.nanmax(values), np.nanmedian(values)
    if not np.isfinite(peak) or not np.isfinite(baseline):
        return None
    if peak <= baseline * PEAK_OVER_BASELINE:
        return None
    return int(np.nanargmax(values)) - reach, reach


def _band(short: float) -> str:
    if short < 20:
        return "nhỏ <20px"
    return "vừa 20-60px" if short < 60 else "lớn >=60px"


def _read_verified(crop_root: Path, boxes_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(boxes_path.read_text(encoding="utf-8"))
    crops_dir = crop_root / "crops"
    rows: list[dict[str, Any]] = []
    for name, record in sorted(payload.get("crops", {}).items()):
        if record.get("status") != "verified":
            continue
        image = cv2.imread(str(crops_dir / name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        height, width = image.shape
        mag = _gradient(image)
        for index, box in enumerate(record.get("boxes", [])):
            w, h = abs(float(box["w"])), abs(float(box["h"]))
            rows.append(
                {
                    "tile": name, "index": index,
                    "x": float(box["x"]), "y": float(box["y"]), "w": w, "h": h,
                    "short": min(w, h), "long": max(w, h),
                    "tile_w": width, "tile_h": height, "mag": mag,
                }
            )
    return rows


def _offsets(rows: list[dict[str, Any]]) -> tuple[list[tuple[int, int, str]], int]:
    measured: list[tuple[int, int, str]] = []
    skipped = 0
    for row in rows:
        if row["short"] < MIN_SHORT_SIDE_PX:
            continue
        found = edge_offset(row["mag"], row["x"], row["y"], row["w"], row["h"])
        if found is None:
            skipped += 1
            continue
        measured.append((found[0], found[1], _band(row["short"])))
    return measured, skipped


def _winnies_offsets(limit_scenes: int = 60) -> tuple[list[tuple[int, int, str]], int]:
    if not WINNIES_ZIP.is_file():
        return [], 0
    measured: list[tuple[int, int, str]] = []
    skipped = 0
    seen: set[str] = set()
    with zipfile.ZipFile(WINNIES_ZIP) as archive:
        names = set(archive.namelist())
        images = sorted(
            n for n in names
            if "/images/" in n and n.lower().endswith((".jpg", ".png"))
        )
        for member in images:
            # Roboflow sinh nhiều bản augment cho một ảnh; chỉ lấy một bản.
            stem = re.sub(r"\.rf\.[0-9a-f]+.*$", "", Path(member).name)
            if stem in seen:
                continue
            seen.add(stem)
            if len(seen) > limit_scenes:
                break
            label = member.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            if label not in names:
                continue
            buffer = np.frombuffer(archive.read(member), np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            height, width = image.shape
            mag = _gradient(image)
            for line in archive.read(label).decode().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                w, h = bw * width, bh * height
                if min(w, h) < MIN_SHORT_SIDE_PX:
                    continue
                found = edge_offset(mag, cx * width - w / 2, cy * height - h / 2, w, h)
                if found is None:
                    skipped += 1
                    continue
                measured.append((found[0], found[1], _band(min(w, h))))
    return measured, skipped


def _fit_line(label: str, measured: list[tuple[int, int, str]], skipped: int) -> None:
    if not measured:
        print(f"  {label}: không đo được box nào")
        return
    offsets = [d for d, _r, _b in measured]
    total = len(offsets)
    tight = sum(1 for d in offsets if abs(d) <= 1)
    too_big = sum(1 for d in offsets if d <= -2)
    too_small = sum(1 for d in offsets if d >= 2)
    print(f"  {label}")
    print(f"      n={total}, bỏ {skipped} box không có mép nổi rõ")
    print(
        f"      trung vị {st.median(offsets):+.1f}px | "
        f"bám đúng {100 * tight / total:.1f}% | "
        f"quá to {100 * too_big / total:.1f}% | quá bé {100 * too_small / total:.1f}%"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("crop_dir", type=Path)
    parser.add_argument("--boxes", type=Path, required=True)
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="bỏ qua phép đo mốc trên Winnies (nhanh hơn, nhưng số đo mất ý nghĩa)",
    )
    args = parser.parse_args(argv)

    crop_root = args.crop_dir.expanduser().resolve()
    boxes_path = args.boxes.expanduser()
    rows = _read_verified(crop_root, boxes_path)
    if not rows:
        raise SystemExit("không có box nào ở trạng thái 'verified'")

    tiles = len({row["tile"] for row in rows})
    print(f"1. HÌNH HỌC — {len(rows)} box trên {tiles} tile đã duyệt\n")

    shorts = [row["short"] for row in rows]
    longs = [row["long"] for row in rows]
    ratios = [row["long"] / max(row["short"], 1e-6) for row in rows]

    def q(values: list[float], p: float) -> float:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(p * len(ordered)))]

    print(f"   cạnh ngắn: p05={q(shorts,.05):.0f} trung vị={st.median(shorts):.0f} "
          f"p90={q(shorts,.90):.0f} max={max(shorts):.0f} px")
    print(f"   cạnh dài : p05={q(longs,.05):.0f} trung vị={st.median(longs):.0f} "
          f"p90={q(longs,.90):.0f} max={max(longs):.0f} px")
    print(f"   tỉ lệ cạnh: trung vị={st.median(ratios):.2f} p99={q(ratios,.99):.2f}")

    tiny = [r for r in rows if r["short"] < 6]
    out_of_bounds = [
        r for r in rows
        if r["x"] < 0 or r["y"] < 0
        or r["x"] + r["w"] > r["tile_w"] or r["y"] + r["h"] > r["tile_h"]
    ]
    elongated = [r for r, ar in zip(rows, ratios) if ar > 3.0]

    by_tile: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_tile[row["tile"]].append(row)

    def iou(a: dict[str, Any], b: dict[str, Any]) -> float:
        ix = max(0.0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
        iy = max(0.0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
        inter = ix * iy
        if inter <= 0:
            return 0.0
        return inter / (a["w"] * a["h"] + b["w"] * b["h"] - inter)

    duplicates = 0
    for group in by_tile.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if iou(group[i], group[j]) > 0.5:
                    duplicates += 1

    print(f"\n   box trùng nhau (IoU>0.5) : {duplicates}")
    print(f"   cạnh ngắn <6px            : {len(tiny)}")
    print(f"   vượt biên tile            : {len(out_of_bounds)}  "
          "(hợp lệ — chỉ clip lúc xuất YOLO)")
    print(f"   tỉ lệ cạnh >3             : {len(elongated)}  "
          "(connector/pin header thường thật sự dài)")

    print("\n2. MÉP BOX CÓ BÁM ĐÚNG MÉP THÂN KHÔNG\n")
    measured, skipped = _offsets(rows)
    _fit_line("BOX CỦA BẠN", measured, skipped)

    if not args.skip_baseline:
        baseline, baseline_skipped = _winnies_offsets()
        _fit_line("WINNIES v3 (dataset công khai — MỐC SO SÁNH)", baseline, baseline_skipped)

    print("\n   theo cỡ linh kiện:")
    for band in ("nhỏ <20px", "vừa 20-60px", "lớn >=60px"):
        subset = [(d, r) for d, r, b in measured if b == band]
        if not subset:
            continue
        offsets = [d for d, _r in subset]
        tight = sum(1 for d in offsets if abs(d) <= 1)
        outside = [(d, r) for d, r in subset if d >= 2]
        at_limit = sum(1 for d, r in outside if d == r)
        note = ""
        if outside:
            note = (f" | trong số 'quá bé', {100 * at_limit / len(outside):.0f}% "
                    "chạm trần tầm quét ⇒ là PAD/CHÂN, không phải mép thân")
        print(f"      {band:12s} n={len(offsets):4d} trung vị {st.median(offsets):+.1f} "
              f"bám đúng {100 * tight / len(offsets):.1f}%{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
