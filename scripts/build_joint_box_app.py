"""Wrap a component-crop folder in an offline app for drawing joint-defect boxes.

``build_solder_label_app.py`` collects one *class* per crop, which is what the
6.2 classifier notebook eats. The 6.2 **detector** needs boxes, and it has to
share a class list with the Roboflow set it is trained beside, so this is a
separate tool rather than a flag on that one.

The class list defaults to the two classes of
``datasets/public/roboflow_solder_leadjoints`` for a reason worth stating: a
class present in only one of the two sources teaches the detector to separate
*cameras* rather than *defects*, because every instance of it also carries that
source's lighting and optics. Adding a class means labelling it on both sides.

    python scripts/build_joint_box_app.py datasets/labelling/fpic_components

Writes ``label_boxes.html`` next to the crops. Everything is inlined and
relative, so the page opens from the filesystem with no server. Progress
autosaves to ``localStorage``; the durable output is the ``joint_boxes.json``
the page exports, which ``pack_joint_detection_dataset.py`` reads.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

TEMPLATE_PATH = Path(__file__).with_name("_joint_box_app_template.html")

#: Matches ``datasets/public/roboflow_solder_leadjoints`` exactly. ``Bad_podu``
#: (坡度) is a fillet that never wetted into a proper slope; ``Bad_qiaojiao``
#: (翘脚) is a lead standing off its pad.
DEFAULT_CLASSES = [
    {"name": "Bad_podu", "vn": "thấm thiếc kém / fillet dốc sai", "color": "#f85149"},
    {"name": "Bad_qiaojiao", "vn": "chân vênh, không chạm pad", "color": "#d29922"},
]

#: Carried into the page so the reviewer sees what they are judging.
ROW_FIELDS = (
    "crop_path", "component_class", "dataset_source", "scene_id",
    "crop_w", "crop_h", "body_x", "body_y", "body_w", "body_h",
)

#: The page does arithmetic with these, so they must not arrive as strings.
INT_FIELDS = frozenset({"crop_w", "crop_h", "body_x", "body_y", "body_w", "body_h"})


def load_rows(manifest: Path, crops: Path) -> list[dict[str, object]]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle))
    if not raw:
        raise SystemExit(f"{manifest} has no rows")

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for record in raw:
        name = record.get("crop_path", "")
        if not (crops / name).exists():
            missing.append(name)
            continue
        row: dict[str, object] = {}
        for field in ROW_FIELDS:
            value = record.get(field, "")
            row[field] = int(value) if field in INT_FIELDS and value != "" else value
        rows.append(row)
    if missing:
        raise SystemExit(
            f"{len(missing)} rows point at crops that are not on disk, first: {missing[:3]}"
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("crop_dir", type=Path,
                        help="directory written by crop_components_for_labelling.py")
    parser.add_argument("--classes", nargs="*", default=None,
                        help="override the defect class list; must match the class list of "
                             "every dataset it will be trained beside")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    root = args.crop_dir.resolve()
    manifest, crops = root / "manifest.csv", root / "crops"
    for path in (manifest, crops):
        if not path.exists():
            raise SystemExit(f"missing {path}; run crop_components_for_labelling.py first")

    if args.classes:
        palette = ["#f85149", "#d29922", "#bc8cff", "#58a6ff", "#3fb950",
                   "#ff7b72", "#79c0ff", "#ffa657", "#a5d6ff"]
        classes = [{"name": n, "vn": "", "color": palette[i % len(palette)]}
                   for i, n in enumerate(args.classes)]
    else:
        classes = DEFAULT_CLASSES

    rows = load_rows(manifest, crops)
    payload = {
        # Keyed on content: relabelling a regenerated crop set must not silently
        # inherit saved progress that was drawn on different pixels.
        "dataset_id": hashlib.sha256(
            f"{root.name}|{len(rows)}|{rows[0]['crop_path']}|"
            f"{','.join(c['name'] for c in classes)}".encode()
        ).hexdigest()[:16],
        "dataset_name": root.name,
        "crops_dir": "crops",
        "classes": classes,
        "rows": rows,
    }
    html = (
        TEMPLATE_PATH.read_text(encoding="utf-8")
        .replace("__DATA__", json.dumps(payload, ensure_ascii=False))
        .replace("__DATASET__", root.name)
    )
    out = args.output.resolve() if args.output else root / "label_boxes.html"
    out.write_text(html, encoding="utf-8")

    by_class: dict[str, int] = {}
    for row in rows:
        by_class[str(row["component_class"])] = by_class.get(str(row["component_class"]), 0) + 1
    scenes = {row["scene_id"] for row in rows}
    print(f"wrote {out}")
    print(f"  {len(rows)} crop, {len(scenes)} cảnh gốc, "
          f"lớp lỗi: {', '.join(c['name'] for c in classes)}")
    for name, count in sorted(by_class.items(), key=lambda kv: -kv[1])[:10]:
        print(f"    {name:<24}{count:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
