"""Bỏ những box đã soi bằng mắt và kết luận là sai, ra một checkpoint mới.

Vì sao tách thành file quyết định + script áp dụng, thay vì sửa tay checkpoint:

* checkpoint là **bằng chứng người duyệt đã vẽ gì**. Sửa nó tại chỗ là làm mất
  dấu vết, và không ai kiểm lại được là đã bỏ những gì, vì lý do gì.
* danh sách loại nằm trong `box_exclusions.json`, đi cùng repo, đọc được bằng
  mắt và review được như code.
* script này chỉ *áp dụng*, và **từ chối chạy** nếu checkpoint không khớp bản đã
  soi — box được chỉ theo chỉ số, mà chỉ số chỉ đúng với đúng file đó.

    python scripts/apply_box_exclusions.py \\
        datasets/labelling/component_bodies_round2_20260830 \\
        --boxes "~/Downloads/joint_boxes (11).json" \\
        --output "~/Downloads/joint_boxes_cleaned.json"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--boxes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclusions", type=Path, default=None,
        help="mặc định: <workspace>/box_exclusions.json",
    )
    args = parser.parse_args(argv)

    root = args.workspace.expanduser().resolve()
    listing = (args.exclusions or (root / "box_exclusions.json")).expanduser()
    payload: dict[str, Any] = json.loads(args.boxes.expanduser().read_text(encoding="utf-8"))
    rules = json.loads(listing.read_text(encoding="utf-8"))

    if rules.get("schema") != "aoi-box-exclusions/1.0":
        raise SystemExit(f"{listing}: schema lạ {rules.get('schema')!r}")
    expected = rules.get("source_exported_at")
    if expected and payload.get("exported_at") != expected:
        raise SystemExit(
            f"checkpoint này xuất lúc {payload.get('exported_at')!r} nhưng danh sách "
            f"loại được soi trên bản {expected!r}.\nBox được chỉ theo CHỈ SỐ, mà chỉ "
            "số chỉ đúng với đúng file đã soi — soát lại rồi cập nhật "
            "box_exclusions.json trước khi chạy."
        )

    # Gom theo tile rồi xoá từ chỉ số LỚN xuống, để việc xoá không làm lệch các
    # chỉ số chưa xoá trong cùng tile.
    by_tile: dict[str, list[dict[str, Any]]] = {}
    for entry in rules["excluded"]:
        by_tile.setdefault(str(entry["tile"]), []).append(entry)

    removed = 0
    for tile, entries in sorted(by_tile.items()):
        record = payload["crops"].get(tile)
        if record is None:
            raise SystemExit(f"checkpoint không có tile {tile!r}")
        boxes = record.get("boxes", [])
        for entry in sorted(entries, key=lambda e: -int(e["index"])):
            index = int(entry["index"])
            if index >= len(boxes):
                raise SystemExit(f"{tile}: không có box #{index}")
            box = boxes[index]
            actual = f"{abs(float(box['w'])):.0f}x{abs(float(box['h'])):.0f}"
            if entry.get("size") and actual != entry["size"]:
                raise SystemExit(
                    f"{tile} #{index}: kích thước {actual} khác {entry['size']} ghi "
                    "trong danh sách loại — checkpoint đã đổi, soát lại."
                )
            del boxes[index]
            removed += 1
            print(f"  bỏ {tile} #{index} ({actual}) — {entry['reason']}")

    payload.setdefault("curation", {})
    payload["curation"] = {
        "exclusions_applied": str(listing.name),
        "boxes_removed": removed,
        "reviewed_at": rules.get("reviewed_at"),
        "reviewer": rules.get("reviewer"),
    }

    out = args.output.expanduser()
    if out.exists():
        raise SystemExit(f"đã có {out}; không ghi đè")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    verified = [v for v in payload["crops"].values() if v.get("status") == "verified"]
    total = sum(len(v.get("boxes", [])) for v in verified)
    print(f"\nđã bỏ {removed} box; còn {total} box trên {len(verified)} tile đã duyệt")
    print(f"ghi -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
