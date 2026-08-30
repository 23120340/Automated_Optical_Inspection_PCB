"""Wrap a component-crop folder in an offline app for drawing joint-defect boxes.

``build_solder_label_app.py`` collects one *class* per crop, which is what the
6.2 classifier notebook eats. This tool collects **boxes**, for the pass-2 model
that tells step 5.5 where the joints are, so it is a separate tool rather than a
flag on that one.

One class, ``solder_joint``, covering **every** joint including sound ones --
see ``DEFAULT_CLASSES`` for why the two-class default was dropped. Boxes should
span the pad and its fillet, not just the bright metal: the first labelling pass
drew them on the metal alone and the resulting model produced boxes 0.76x the
pad area, which cut off the very thing step 6.2 grades.

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
from typing import Any

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

TEMPLATE_PATH = Path(__file__).with_name("_joint_box_app_template.html")


def template_digest() -> str:
    """SHA-256 của template dựng trang, để trang tự khai nó sinh ra từ bản nào."""

    return hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest()

#: Một lớp, và tên nó nói đúng thứ được khoanh: **mọi** mối hàn, kể cả mối hàn
#: lành. Đó là bài toán *định vị*, không phải bài toán chấm lỗi.
#:
#: Mặc định cũ là hai lớp của ``roboflow_solder_leadjoints`` (``Bad_podu`` /
#: ``Bad_qiaojiao``). Bỏ đi vì hai lý do đo được:
#:
#: * Người gắn nhãn **chưa bao giờ phân biệt hai lớp đó** -- 9.283/9.283 box của
#:   phiên đầu đều mang đúng một lớp. Báo cáo chỉ số theo lớp cho một phân biệt
#:   chưa từng được vẽ là báo một con số vô nghĩa.
#: * Bộ Roboflow chỉ khoanh mối hàn **LỖI**; nhãn ở đây khoanh **mọi** mối hàn.
#:   Cùng một vật -- một mối hàn lành -- là *nền* ở bộ kia và *positive* ở bộ
#:   này, nên hai bộ không ghép được và việc chung lớp với chúng mất ý nghĩa.
#:
#: Tên ``solder_joint`` nằm trong ``aoi_pipeline.solder.leads.LEAD_CLASSES``.
#: Đổi nó mà không sửa set kia thì mọi detection của lượt 2 bị bỏ qua lặng lẽ.
DEFAULT_CLASSES = [
    {"name": "solder_joint", "vn": "mối hàn — khoanh cả pad và fillet",
     "color": "#3fb950"},
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


def dataset_id_for(
    root: Path,
    rows: list[dict[str, object]],
    class_names: list[str],
) -> str:
    """Return the identity used by the browser export/localStorage contract.

    Keep this calculation compatible with already exported checkpoints.  A new
    continuation folder gets a new ``root.name`` and therefore a fresh storage
    key, while old reviewed JSON remains verifiable by the packer.
    """

    if not rows:
        raise ValueError("cannot identify an empty crop set")
    return hashlib.sha256(
        f"{root.name}|{len(rows)}|{rows[0]['crop_path']}|"
        f"{','.join(class_names)}".encode()
    ).hexdigest()[:16]


def load_seed(
    path: Path,
    *,
    dataset_name: str,
    dataset_id: str,
    crop_names: set[str],
    class_names: list[str],
) -> dict[str, dict[str, object]]:
    """Validate a continuation draft before embedding it in the offline app."""

    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read seed JSON {path}: {exc}") from exc
    expected = {
        "schema": "aoi-joint-boxes/1.0",
        "dataset": dataset_name,
        "dataset_id": dataset_id,
        "coordinate_space": "crop_pixels_top_left_origin",
        "classes": class_names,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise SystemExit(
                f"{path}: {field} mismatch; expected {value!r}, got {payload.get(field)!r}"
            )
    records = payload.get("crops")
    if not isinstance(records, dict):
        raise SystemExit(f"{path}: crops must be an object")
    unknown = sorted(set(records) - crop_names)
    if unknown:
        raise SystemExit(f"{path}: seed has unknown crop paths: {unknown[:3]}")

    by_class = {name: index for index, name in enumerate(class_names)}
    allowed_statuses = {"", "verified", "skipped", "unusable"}
    state: dict[str, dict[str, object]] = {}
    for crop_name, record in records.items():
        if not isinstance(record, dict):
            raise SystemExit(f"{path}: {crop_name} record must be an object")
        status = record.get("status", "")
        if status not in allowed_statuses:
            raise SystemExit(f"{path}: {crop_name} has invalid status {status!r}")
        boxes = record.get("boxes", [])
        if not isinstance(boxes, list):
            raise SystemExit(f"{path}: {crop_name} boxes must be a list")
        converted: list[dict[str, object]] = []
        for box in boxes:
            if not isinstance(box, dict) or box.get("cls") not in by_class:
                raise SystemExit(f"{path}: {crop_name} has invalid box class")
            geometry: dict[str, float] = {}
            for field in ("x", "y", "w", "h"):
                value = box.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise SystemExit(
                        f"{path}: {crop_name} box has non-numeric {field}={value!r}"
                    )
                geometry[field] = float(value)
            converted.append({"cls": by_class[str(box["cls"])], **geometry})
        state[crop_name] = {
            "status": status,
            "notes": str(record.get("notes", "")),
            "boxes": converted,
        }
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("crop_dir", type=Path,
                        help="directory written by crop_components_for_labelling.py")
    parser.add_argument("--classes", nargs="*", default=None,
                        help="override the class list; every name must appear in "
                             "aoi_pipeline.solder.leads.LEAD_CLASSES or fusion drops it")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed-json", type=Path, default=None,
                        help="validated draft/checkpoint used only when this dataset has "
                             "no localStorage yet")
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
    dataset_id = dataset_id_for(root, rows, [str(c["name"]) for c in classes])
    initial_state: dict[str, dict[str, object]] = {}
    if args.seed_json is not None:
        initial_state = load_seed(
            args.seed_json.resolve(),
            dataset_name=root.name,
            dataset_id=dataset_id,
            crop_names={str(row["crop_path"]) for row in rows},
            class_names=[str(c["name"]) for c in classes],
        )
    payload = {
        # Keyed on content: relabelling a regenerated crop set must not silently
        # inherit saved progress that was drawn on different pixels.
        "dataset_id": dataset_id,
        "dataset_name": root.name,
        "crops_dir": "crops",
        "classes": classes,
        "rows": rows,
        "initial_state": initial_state,
        # Vân tay của template đã dựng ra trang này. Trang nằm trong .gitignore
        # (nó nhúng cả ảnh), nên sửa template xong mà quên dựng lại thì người
        # duyệt vẫn mở trang cũ và KHÔNG có gì báo. Đã xảy ra thật: bản vá chống
        # ghi đè khi nạp file có trong template nhưng vắng ở trang trên đĩa.
        "template_sha256": template_digest(),
    }
    component_body_task = [str(c["name"]) for c in classes] == ["component"]
    if component_body_task:
        wording = {
            "__PAGE_TITLE__": "Khoanh THÂN linh kiện",
            "__CLEAN_FILTER__": "không có thân",
            "__CLEAN_BUTTON__": "Không có thân",
            "__DECISION_NOTE__": (
                "<b>Enter</b> xác nhận mọi box thân linh kiện trên tile. <b>C</b> chỉ dùng "
                "khi tile thật sự không có thân nào; ảnh chưa duyệt bị loại hoàn toàn. "
                "Không chắc thì bấm <b>Bỏ qua</b>."
            ),
            "__HELP_INTRO__": (
                "Mỗi ảnh là một tile của bo mạch. Khoanh <b>MỌI THÂN linh kiện</b> "
                "nhìn thấy; box nháp của detector chỉ là gợi ý và phải được sửa trước "
                "khi nhấn Enter."
            ),
            "__SHORTCUT_C__": (
                "xác nhận tile <b>không có thân linh kiện</b> rồi sang ảnh sau"
            ),
            "__BOX_GUIDANCE__": (
                "Khoanh sát <b>thân/gói/vỏ linh kiện</b> (gói đen, thân gốm, vỏ can). "
                "<b>Không bao chân, pad hay vùng thiếc.</b> Mỗi thân = một box, kể cả "
                "linh kiện tốt; đây là bài toán định vị chứ không phải phân loại lỗi."
            ),
            "__CLASS_GUIDANCE__": (
                "Chỉ có một lớp <b>component</b> vì detector lượt 1 cần định vị mọi "
                "thân linh kiện. Kiểu gói và tình trạng lỗi được xử lý ở bước sau."
            ),
        }
    else:
        wording = {
            "__PAGE_TITLE__": "Khoanh mối hàn lỗi 6.2",
            "__CLEAN_FILTER__": "sạch",
            "__CLEAN_BUTTON__": "Sạch",
            "__DECISION_NOTE__": (
                "<b>Sạch</b> khác <b>chưa duyệt</b>. Ảnh đánh <b>Sạch</b> đi vào tập "
                "train làm ảnh nền — đó chính là thứ giảm báo nhầm. Ảnh chưa duyệt bị "
                "loại hoàn toàn. Không chắc thì bấm <b>Bỏ qua</b>, đừng bấm Sạch."
            ),
            "__HELP_INTRO__": (
                "Mỗi ảnh là một linh kiện đã cắt từ ảnh board. Việc của bạn là "
                "<b>khoanh những mối hàn LỖI</b>. Mối hàn tốt thì không khoanh gì cả."
            ),
            "__SHORTCUT_C__": "đánh dấu <b>sạch</b> (không lỗi) rồi sang ảnh sau",
            "__BOX_GUIDANCE__": (
                "Chỉ khoanh <b>vùng mối hàn</b>, không khoanh cả linh kiện. Box nên "
                "ôm phần thiếc và chân tiếp xúc, không lấn sang thân. Một chân lỗi = một box."
            ),
            "__CLASS_GUIDANCE__": (
                "Các lớp ở đây trùng đúng với tập đang dùng để train. Thêm một lớp mới "
                "mà chỉ ảnh của bạn có nhãn sẽ dạy model phân biệt <i>nguồn ảnh</i> chứ "
                "không phải phân biệt <i>lỗi</i>."
            ),
        }
    html = (
        TEMPLATE_PATH.read_text(encoding="utf-8")
        .replace("__DATA__", json.dumps(payload, ensure_ascii=False))
        .replace("__DATASET__", root.name)
    )
    for marker, text in wording.items():
        html = html.replace(marker, text)
    out = args.output.resolve() if args.output else root / "label_boxes.html"
    out.write_text(html, encoding="utf-8")

    by_class: dict[str, int] = {}
    for row in rows:
        by_class[str(row["component_class"])] = by_class.get(str(row["component_class"]), 0) + 1
    scenes = {row["scene_id"] for row in rows}
    print(f"wrote {out}")
    print(f"  {len(rows)} crop, {len(scenes)} cảnh gốc, "
          f"lớp: {', '.join(c['name'] for c in classes)}")
    if initial_state:
        seeded = sum(bool(record["status"]) for record in initial_state.values())
        print(f"  seed: {len(initial_state)} crop ({seeded} đã duyệt)")
    for name, count in sorted(by_class.items(), key=lambda kv: -kv[1])[:10]:
        print(f"    {name:<24}{count:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
