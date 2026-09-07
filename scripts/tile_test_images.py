"""Cut whole-board test photographs into tiles that show components clearly.

A 16 MP photograph of a whole board is the right input for steps 0-3 and the
wrong one for looking at a component: the board is one object in a big frame and
everything on it is small. This produces the other view -- the one the project's
own reference image is, ``00001__1024__1648___4120.png``: a 1024 px window onto
part of a board with individual parts readable.

The scale is not adjusted, and that is deliberate. Measured with the shipped
detector:

    00001__1024__1648___4120   39 components, median short side 22 px
    pcb_dslr_001__rec1        176 components, median short side 39 px  (1.78x)
    pcb31__rec1               172 components, median short side 54 px  (2.43x)

These photographs are already finer than the reference, so a native-resolution
tile lands *above* it -- larger parts, easier to see. Downscaling toward 22 px
would throw that away.

**Which tiles are kept is decided by the detector, not by a pixel statistic.**
Brightness and saturation were tried first and do not separate cleanly: on
``pcb31`` the background strip reads 133 brightness and the darkest board tile
reads 34. Component count does separate, and it is the property that matters --
a tile with nothing on it cannot test component detection no matter how
photogenic it is.

    python scripts/tile_test_images.py --limit 40

Tiles are cut from the **original** file, not the preprocessed frame, so the
pipeline still does its own step 1 on them. Detections come from the analysis
frame and are scaled back up.

``components`` in the manifest is therefore **what the whole-board pass found**,
not what a tile contains. Running the detector on a tile finds considerably more,
because the part is bigger and there is no competing context -- measured on two
tiles: 52 -> 96 and 11 -> 120. That gap is the reason these exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

#: Cùng quy ước tên với ảnh mẫu của dự án: ``{stem}__{tile}__{x}___{y}.png``.
#: Ba dấu gạch dưới trước ``y`` là của bộ Consolidated, giữ nguyên để một người
#: quen bộ đó đọc được ngay tên file này.
NAME = "{stem}__{tile}__{x}___{y}.png"


def tile_one(
    path: Path,
    detector,
    *,
    tile: int,
    stride: int,
    min_components: int,
    output: Path,
    dry_run: bool,
    max_dark: float = 1.0,
    dark_level: int = 40,
    max_blown: float = 1.0,
) -> list[dict[str, object]]:
    import cv2
    import numpy as np

    from aoi_pipeline.config import PreprocessConfig
    from aoi_pipeline.imaging.exposure import blown_fraction
    from aoi_pipeline.imaging.preprocessing import ImagePreprocessor

    original = cv2.imread(str(path))
    if original is None:
        return []
    height, width = original.shape[:2]

    analysis = ImagePreprocessor(PreprocessConfig()).process(original).image
    detections = detector.detect(analysis)
    # The analysis frame is capped at max_side, so it is usually smaller than the
    # file. One factor maps every box back; the aspect ratio is preserved by the
    # resize, so a single scalar is enough.
    scale = width / analysis.shape[1]
    boxes = [
        (
            d.bbox.x1 * scale, d.bbox.y1 * scale,
            d.bbox.x2 * scale, d.bbox.y2 * scale,
            d.label,
        )
        for d in detections
    ]

    rows: list[dict[str, object]] = []
    for top in range(0, max(1, height - tile + 1), stride):
        for left in range(0, max(1, width - tile + 1), stride):
            right, bottom = left + tile, top + tile
            # A component counts for this tile only if its CENTRE is inside it.
            # Overlap would count the same part for every tile it grazes and let
            # a tile of bare laminate qualify on a neighbour's connector.
            inside = [
                b for b in boxes
                if left <= (b[0] + b[2]) / 2 < right and top <= (b[1] + b[3]) / 2 < bottom
            ]
            if len(inside) < min_components:
                continue
            patch = original[top:bottom, left:right]
            # Tỉ lệ pixel tối: các board này được chụp trên vải đen, nên một tile
            # ở mép board vẫn đủ 6 linh kiện dồn vào một góc mà 80% khung là nền.
            # Nó KHÔNG bị loại -- mép board là thứ có thật và đáng test -- nhưng
            # con số phải nằm trong manifest, nếu không muốn lọc lại phải đo lại.
            dark = float(
                (cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) < dark_level).mean()
            )
            if dark > max_dark:
                continue
            # Tỉ lệ pixel CHÁY: vùng vừa chạm trần vừa mất chi tiết cục bộ. Sáng
            # thôi thì không tính -- connector nhựa kem và pad mạ vàng vẫn còn
            # chi tiết, mà pad mạ vàng lộ thiên đúng là lớp cần dạy cho model.
            blown = blown_fraction(patch)
            if blown > max_blown:
                continue
            name = NAME.format(stem=path.stem, tile=tile, x=left, y=top)
            if not dry_run:
                cv2.imwrite(str(output / name), patch)
            labels: dict[str, int] = {}
            for b in inside:
                labels[b[4]] = labels.get(b[4], 0) + 1
            rows.append({
                "file": name,
                "source": path.name,
                "x": left,
                "y": top,
                "tile": tile,
                "components": len(inside),
                "dark_fraction": round(dark, 3),
                "blown_fraction": round(blown, 4),
                "labels": dict(sorted(labels.items(), key=lambda kv: -kv[1])),
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path,
                        default=PROJECT_ROOT / "datasets" / "test_images")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "datasets" / "test_images" / "tiles_1024")
    parser.add_argument("--model", type=Path,
                        default=PROJECT_ROOT / "models" / "active" / "detector" / "best.onnx")
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=768,
                        help="bước trượt; nhỏ hơn tile để linh kiện ở mép vẫn "
                             "vào trọn trong một tile nào đó")
    parser.add_argument("--min-components", type=int, default=4,
                        help="tile có ít hơn ngần này linh kiện thì bỏ — nó không "
                             "test được gì cho bước 4")
    parser.add_argument("--max-dark-fraction", type=float, default=1.0,
                        help="bỏ tile có tỉ lệ pixel tối vượt ngưỡng (1.0 = giữ hết). "
                             "Đo trên bộ hiện tại: trung vị 30%%, 28%% số tile trên "
                             "50%%, 8%% trên 70%%")
    parser.add_argument("--max-blown-fraction", type=float, default=1.0,
                        help="bỏ tile có tỉ lệ pixel CHÁY SÁNG vượt ngưỡng "
                             "(1.0 = giữ hết). Cháy = vừa chạm trần vừa mất chi "
                             "tiết cục bộ; sáng mà còn vân thì không tính")
    parser.add_argument("--limit", type=int, default=0,
                        help="chỉ xử lý ngần này ảnh nguồn (0 = tất cả)")
    parser.add_argument("--pattern", default="*__rec1.jpg",
                        help="ảnh nguồn nào được cắt; mặc định một bản ghi mỗi board")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    from aoi_pipeline.config import ModelDetectorConfig
    from aoi_pipeline.detection.detectors import create_detector

    sources = sorted(args.source.rglob(args.pattern))
    if not sources:
        raise SystemExit(f"không thấy ảnh nào khớp {args.pattern!r} dưới {args.source}")
    if args.limit:
        sources = sources[: args.limit]
    output = args.output.resolve()
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)

    record = output / "tiles_manifest.json"
    previous: dict[str, list[dict[str, object]]] = {}
    if record.is_file():
        for row in json.loads(record.read_text(encoding="utf-8")):
            previous.setdefault(str(row.get("source")), []).append(row)

    detector = create_detector(str(args.model),
                               model_config=ModelDetectorConfig(confidence=0.25))
    rows: list[dict[str, object]] = []
    for index, path in enumerate(sources, start=1):
        # Nối tiếp được, nhưng CHỈ khi manifest cũ còn số đo của ảnh đó. Bỏ qua
        # một ảnh mà không mang theo số linh kiện mỗi tile thì manifest mới sẽ
        # thiếu đúng thứ nó sinh ra để ghi lại, và không ai biết là thiếu.
        done = previous.get(path.name)
        if done and not args.dry_run and all(
            (output / str(r["file"])).exists() for r in done
        ):
            print(f"  [{index}/{len(sources)}] {path.name}: bỏ qua, đã có {len(done)} tile")
            rows.extend(done)
            continue
        produced = tile_one(
            path, detector, tile=args.tile, stride=args.stride,
            min_components=args.min_components, output=output, dry_run=args.dry_run,
            max_dark=args.max_dark_fraction,
            max_blown=args.max_blown_fraction,
        )
        rows.extend(produced)
        print(f"  [{index}/{len(sources)}] {path.name}: {len(produced)} tile")

    if rows and not args.dry_run:
        record.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    counts = [int(r["components"]) for r in rows]
    print(f"\n{len(rows)} tile từ {len(sources)} ảnh → {output}")
    if counts:
        counts.sort()
        print(f"  linh kiện mỗi tile: trung vị {counts[len(counts)//2]}, "
              f"ít nhất {counts[0]}, nhiều nhất {counts[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
