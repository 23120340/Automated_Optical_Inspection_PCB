"""Turn board tiles into a labelling folder for retraining the PASS-1 detector.

Pass 1 finds component *bodies* on a board, and the shipped model does that
badly on large fine-pitch packages: measured on a tile of pcb7, its largest box
is 231x219 at confidence 0.25 and 251x250 even at 0.10, while the QFPs there are
about 350 px. So it never boxes them as a package -- it boxes a lead comb
(aspect ratio 4.5 and 5.6 observed) or the printed text area, and step 5.5 then
derives ROIs from a box that is not the part.

No public set fixes this. FPIC carries 2,263 IC boxes but only 86 with a short
side over 250 px; the two IC-focused Roboflow sets that do have large boxes
(``ic-hpvk3``, ``integrated-circuit-ic``) turned out to be isolated chips on a
tray and on white paper, which is the wrong problem -- pass 1 has to find a part
*among* other parts on a board.

What does have it is already on disk: ``datasets/test_images/tiles_1024``, cut
from 16 MP whole-board photographs and full of QFP and PLCC packages at 300-400
px. This prepares those tiles for the box tool that already exists.

**Box the BODY, not the leads.** That is not a preference, it is what step 5.5
assumes: ``lead_inner_ratio`` reaches 0.14 of the short side back INTO the box
and ``lead_outer_ratio`` reaches 0.26 OUT, so the lead band straddles the box
edge. Measured on the ten hand-checked parts in ``tests/data``, pads sit 42%
inside the detector box and 58% outside, and only 2 of 28 fall wholly inside. A
box drawn around the leads would push that band 0.26 x span past them, onto bare
laminate.

    python scripts/prepare_component_labelling.py
    python scripts/build_joint_box_app.py datasets/labelling/component_bodies \\
        --classes component
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def prepare(
    tiles: Path,
    output: Path,
    *,
    limit: int = 0,
    max_dark: float = 0.6,
    dry_run: bool = False,
) -> int:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None

    manifest_path = tiles / "tiles_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"không thấy {manifest_path}; chạy scripts/tile_test_images.py trước")
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Tile phần lớn là nền không đáng gán nhãn: người duyệt trả tiền bằng thời
    # gian cho mỗi ảnh, và một ảnh 80% vải đen dạy được rất ít. Ngưỡng đo được
    # trên bộ hiện tại: trung vị 30%, 28% số tile trên 50%.
    kept = [r for r in rows if float(r.get("dark_fraction", 0.0)) <= max_dark]
    # Nhiều linh kiện trước: mỗi ảnh trả về nhiều box hơn cho cùng công sức.
    kept.sort(key=lambda r: -int(r.get("components", 0)))
    if limit:
        kept = kept[:limit]

    crops = output / "crops"
    if not dry_run:
        crops.mkdir(parents=True, exist_ok=True)

    out_rows: list[dict[str, object]] = []
    for row in kept:
        source = tiles / str(row["file"])
        if not source.is_file():
            continue
        with Image.open(source) as handle:
            width, height = handle.size
        if not dry_run:
            shutil.copy2(source, crops / source.name)
        out_rows.append({
            "crop_path": source.name,
            # Ô này là cả một mảng board, không phải một linh kiện, nên không có
            # "lớp linh kiện" để hiện. Ghi số linh kiện lượt 1 tìm được để người
            # duyệt biết ảnh này dày hay thưa trước khi mở.
            "component_class": f"tile ~{row.get('components', '?')} linh kiện",
            "dataset_source": "tiles_1024",
            # Cảnh = ảnh board gốc. Chia tập theo nó, không theo tile: hai tile
            # cạnh nhau chồng lấn 256 px nên chia theo tile là rò rỉ.
            "scene_id": str(row.get("source", source.name)).rsplit(".", 1)[0],
            "crop_w": width,
            "crop_h": height,
            # Không có khung gợi ý: thứ cần khoanh là MỌI thân linh kiện trong
            # ảnh, không phải một cái đã biết trước. Số 0 tắt phần vẽ gợi ý.
            "body_x": 0, "body_y": 0, "body_w": 0, "body_h": 0,
            "roi_kind": "board_tile",
            "label_status": "", "reviewer_id": "", "notes": "",
        })

    if dry_run:
        print(f"[dry-run] {len(out_rows)} tile sẽ được chuẩn bị")
        return 0

    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    (output / "provenance.json").write_text(json.dumps({
        "source": str(tiles),
        "tiles_available": len(rows),
        "tiles_prepared": len(out_rows),
        "max_dark_fraction": max_dark,
        "purpose": "gán nhãn THÂN linh kiện để train lại detector lượt 1",
        "box_convention": (
            "Khoanh THÂN (gói đen / thân gốm / vỏ can), KHÔNG bao chân. Bước 5.5 "
            "đặt dải chân vắt qua mép hộp (lead_inner_ratio 0.14 vào trong, "
            "lead_outer_ratio 0.26 ra ngoài). Đo trên 10 linh kiện kiểm tay: pad "
            "nằm 42% trong hộp, 58% ngoài."
        ),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"đã chuẩn bị {len(out_rows)} tile → {output}")
    print(f"  bỏ {len(rows) - len(kept)} tile có >{max_dark:.0%} pixel nền")
    counts = sorted((int(r.get("components", 0)) for r in kept), reverse=True)
    if counts:
        print(f"  linh kiện/tile: nhiều nhất {counts[0]}, trung vị {counts[len(counts)//2]}")
    print(f"\nBước tiếp: python scripts/build_joint_box_app.py {output} --classes component")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tiles", type=Path,
                        default=PROJECT_ROOT / "datasets" / "test_images" / "tiles_1024")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "datasets" / "labelling" / "component_bodies")
    parser.add_argument("--limit", type=int, default=120,
                        help="số tile chuẩn bị, ưu tiên tile nhiều linh kiện (0 = tất cả)")
    parser.add_argument("--max-dark-fraction", type=float, default=0.6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return prepare(args.tiles.resolve(), args.output.resolve(),
                   limit=args.limit, max_dark=args.max_dark_fraction,
                   dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
