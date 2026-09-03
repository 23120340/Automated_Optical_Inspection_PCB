"""Cắt tile các ảnh train quá lớn, để linh kiện nhỏ không bị letterbox teo đi.

**Vấn đề đo được.** Trong gói `component_detect_v1`, ở `imgsz=1536`:

    nguồn      box      <8px    trung vị
    rf100    36.673    29,7%     10,8 px
    local     9.486     2,7%     21,0 px
    winnies   7.095      0,0%     26,2 px

Nhãn của người trong dự án hoàn toàn khoẻ. Vấn đề nằm ở **RF100**: ảnh gốc rộng
504–5985 px, bị letterbox về 1536 nên có ảnh co tới 3,9 lần — một linh kiện
30 px thành 7,7 px, tức nhỏ hơn một ô lưới P3 (stride 8) và gần như không học
được.

**Cách sửa gốc là cắt tile, không phải nâng imgsz.** Nâng 1536→1792 chỉ bớt
được ~6 điểm phần trăm mà tốn thêm 36% compute. Cắt ảnh 5985 px thành các tile
1024 giữ nguyên độ phân giải gốc, nên linh kiện giữ nguyên kích thước pixel.

**Chỉ đụng vào `train`.** `valid`/`test` là tile local đã khoá theo bo — sửa
chúng là làm hỏng đúng thứ dùng để chấm điểm. Dữ liệu công khai vốn cũng chỉ
nằm ở train, nên ràng buộc này không mất gì.

    python scripts/tile_packed_dataset.py \\
        datasets/train/component_detect_v1 \\
        --output datasets/train/component_detect_v1_tiled
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
#: Box nhỏ hơn mức này sau khi cắt là mảnh vụn do đường cắt tạo ra, không phải
#: linh kiện. Giữ lại chúng là dạy model học mảnh vụn.
MIN_BOX_PX = 4.0
#: Box được giữ theo TÂM. Lưu ý: tile CHỒNG NHAU (stride < tile), nên một tâm
#: nằm trong vùng chồng lấn thuộc về NHIỀU tile và linh kiện đó xuất hiện ở cả
#: hai — đo được 51.314 -> 94.905 box, tức ~1,85 lần. Đây là hành vi đúng của
#: tiling chồng lấn, không phải lỗi: linh kiện bị đường cắt xén ở tile này thì
#: còn nguyên vẹn ở tile kia. Tất cả đều nằm trong `train` nên không rò rỉ sang
#: thước đo.
def _tiles_for(width: int, height: int, tile: int, stride: int) -> list[tuple[int, int]]:
    xs = list(range(0, max(1, width - tile + 1), stride))
    ys = list(range(0, max(1, height - tile + 1), stride))
    if xs[-1] + tile < width:
        xs.append(max(0, width - tile))
    if ys[-1] + tile < height:
        ys.append(max(0, height - tile))
    return [(x, y) for y in ys for x in xs]


def _read_labels(path: Path) -> list[tuple[float, float, float, float]]:
    out = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            out.append(tuple(float(v) for v in parts[1:]))  # cx cy w h (chuẩn hoá)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=768, help="chồng lấn = tile - stride")
    parser.add_argument(
        "--min-side", type=int, default=1400,
        help="chỉ cắt ảnh có cạnh dài hơn mức này; nhỏ hơn thì chép nguyên",
    )
    args = parser.parse_args(argv)

    src = args.source.expanduser().resolve()
    dst = args.output.expanduser().resolve()
    if dst.exists():
        raise SystemExit(f"đã có {dst}; không ghi đè")
    if not (src / "data.yaml").is_file():
        raise SystemExit(f"không thấy {src / 'data.yaml'}")

    staging = Path(tempfile.mkdtemp(prefix=f".{dst.name}.staging-", dir=dst.parent))
    stats = {s: {"images_in": 0, "images_out": 0, "boxes_in": 0, "boxes_out": 0,
                 "tiled": 0, "empty_tiles_dropped": 0, "fragments_dropped": 0}
             for s in SPLITS}
    try:
        for split in SPLITS:
            (staging / split / "images").mkdir(parents=True)
            (staging / split / "labels").mkdir(parents=True)
            images = sorted(
                p for p in (src / split / "images").glob("*")
                if p.suffix.lower() in IMAGE_SUFFIXES
            )
            for path in images:
                labels = _read_labels(src / split / "labels" / f"{path.stem}.txt")
                stats[split]["images_in"] += 1
                stats[split]["boxes_in"] += len(labels)

                image = cv2.imread(str(path))
                if image is None:
                    raise SystemExit(f"không đọc được {path}")
                height, width = image.shape[:2]

                # valid/test giữ NGUYÊN: chúng là thước đo, không phải dữ liệu học.
                if split != "train" or max(width, height) <= args.min_side:
                    shutil.copy2(path, staging / split / "images" / path.name)
                    body = "\n".join(
                        f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in labels
                    )
                    (staging / split / "labels" / f"{path.stem}.txt").write_text(
                        body + ("\n" if labels else ""), encoding="utf-8")
                    stats[split]["images_out"] += 1
                    stats[split]["boxes_out"] += len(labels)
                    continue

                stats[split]["tiled"] += 1
                pixels = [(cx * width, cy * height, w * width, h * height)
                          for cx, cy, w, h in labels]
                for row, (x0, y0) in enumerate(_tiles_for(width, height, args.tile, args.stride)):
                    x1, y1 = min(x0 + args.tile, width), min(y0 + args.tile, height)
                    tw, th = x1 - x0, y1 - y0
                    kept = []
                    for cx, cy, bw, bh in pixels:
                        if not (x0 <= cx < x1 and y0 <= cy < y1):
                            continue  # tâm thuộc tile khác
                        left = max(cx - bw / 2, x0)
                        top = max(cy - bh / 2, y0)
                        right = min(cx + bw / 2, x1)
                        bottom = min(cy + bh / 2, y1)
                        nw, nh = right - left, bottom - top
                        if nw < MIN_BOX_PX or nh < MIN_BOX_PX:
                            stats[split]["fragments_dropped"] += 1
                            continue
                        kept.append((
                            ((left + right) / 2 - x0) / tw,
                            ((top + bottom) / 2 - y0) / th,
                            nw / tw, nh / th,
                        ))
                    if not kept:
                        # Ảnh công khai có thể gán nhãn KHÔNG đầy đủ, nên một tile
                        # "rỗng" chưa chắc là nền thật. Bỏ đi an toàn hơn là dạy
                        # model rằng vùng đó không có linh kiện nào.
                        stats[split]["empty_tiles_dropped"] += 1
                        continue
                    stem = f"{path.stem}__t{row:03d}"
                    cv2.imwrite(str(staging / split / "images" / f"{stem}.png"),
                                image[y0:y1, x0:x1])
                    (staging / split / "labels" / f"{stem}.txt").write_text(
                        "\n".join(f"0 {a:.6f} {b:.6f} {c:.6f} {d:.6f}" for a, b, c, d in kept)
                        + "\n", encoding="utf-8")
                    stats[split]["images_out"] += 1
                    stats[split]["boxes_out"] += len(kept)

        shutil.copy2(src / "data.yaml", staging / "data.yaml")
        parent_manifest = src / "pack_manifest.json"
        manifest = json.loads(parent_manifest.read_text(encoding="utf-8")) if parent_manifest.is_file() else {}
        manifest["tiling"] = {
            "source_pack": str(src.name),
            "tile": args.tile, "stride": args.stride, "min_side": args.min_side,
            "min_box_px": MIN_BOX_PX,
            "box_assignment": "theo TÂM; tile chồng lấn nên một linh kiện có thể xuất hiện ở nhiều tile (đo: x1,85 số box)",
            "splits_touched": ["train"],
            "why_not_valid_test": "valid/test là thước đo và đã khoá theo bo vật lý",
            "stats": stats,
        }
        (staging / "pack_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        staging.replace(dst)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    for split in SPLITS:
        s = stats[split]
        print(f"{split:6s} {s['images_in']:5d} -> {s['images_out']:5d} ảnh  "
              f"({s['tiled']} ảnh được cắt)   "
              f"{s['boxes_in']:6d} -> {s['boxes_out']:6d} box")
        if s["empty_tiles_dropped"] or s["fragments_dropped"]:
            print(f"       bỏ {s['empty_tiles_dropped']} tile rỗng, "
                  f"{s['fragments_dropped']} mảnh vụn <{MIN_BOX_PX:g}px")
    print(f"\nghi -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
