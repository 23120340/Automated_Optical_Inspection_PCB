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
UNKNOWN_CLASS_INDEX = -1
_MIGRATION_STRATEGY = "preserve_geometry_reset_box_classes_to_unknown"
PACKAGE_CLASSES = (
    "hai_chan",
    "tru_dung",
    "goi_nho",
    "ic_hai_ben",
    "ic_bon_ben",
    "ic_khong_chan",
    "connector",
)
PACKAGE_VI = {
    "hai_chan": "Hai chân",
    "tru_dung": "Trụ đứng",
    "goi_nho": "Gói nhỏ 3–5 chân",
    "ic_hai_ben": "IC chân hai bên",
    "ic_bon_ben": "IC chân bốn bên",
    "ic_khong_chan": "IC không thấy chân",
    "connector": "Connector / xuyên lỗ",
}


def template_digest() -> str:
    """SHA-256 của template dựng trang, để trang tự khai nó sinh ra từ bản nào.

    Băm nội dung đã **chuẩn hoá xuống dòng**, không băm byte thô. Git tự đổi
    LF/CRLF theo `core.autocrlf` mỗi lần checkout, nên vân tay theo byte thô
    đổi khi chỉ đổi nhánh — chốt sẽ báo "trang cũ" trong khi template y
    nguyên. Đã dính đúng một lần: 631 ký tự CR xuất hiện sau `git checkout`.
    """

    normalised = TEMPLATE_PATH.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(normalised).hexdigest()

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


def load_seed_contract(
    path: Path,
    *,
    dataset_name: str,
    dataset_id: str,
    crop_names: set[str],
    class_names: list[str],
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """Validate a continuation draft and its editor-only migration contract.

    A seed may declare one ``unknown_class`` sentinel outside the training
    taxonomy.  It is represented in browser state by ``-1`` and may only occur
    on an unverified record.  ``migration_aliases`` are also kept out of normal
    exports: they are instructions for carrying an older localStorage key into
    this page without reinterpreting its old numeric class indices.
    """

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
    unknown_class = payload.get("unknown_class")
    if unknown_class is not None:
        if not isinstance(unknown_class, str) or not unknown_class:
            raise SystemExit(f"{path}: unknown_class must be a non-empty string")
        if unknown_class in by_class:
            raise SystemExit(
                f"{path}: unknown_class is an editor sentinel and must not be in classes"
            )

    raw_aliases = payload.get("migration_aliases", [])
    if not isinstance(raw_aliases, list):
        raise SystemExit(f"{path}: migration_aliases must be a list")
    migration_aliases: list[dict[str, object]] = []
    seen_aliases: set[str] = set()
    for index, alias in enumerate(raw_aliases):
        if not isinstance(alias, dict):
            raise SystemExit(f"{path}: migration_aliases[{index}] must be an object")
        alias_id = alias.get("dataset_id")
        alias_classes = alias.get("classes")
        strategy = alias.get("strategy")
        if not isinstance(alias_id, str) or not alias_id or alias_id == dataset_id:
            raise SystemExit(f"{path}: migration_aliases[{index}] has invalid dataset_id")
        if alias_id in seen_aliases:
            raise SystemExit(f"{path}: duplicate migration alias {alias_id!r}")
        if (
            not isinstance(alias_classes, list)
            or not alias_classes
            or any(not isinstance(name, str) or not name for name in alias_classes)
        ):
            raise SystemExit(f"{path}: migration_aliases[{index}] has invalid classes")
        if strategy != _MIGRATION_STRATEGY:
            raise SystemExit(
                f"{path}: migration_aliases[{index}] uses unsupported strategy {strategy!r}"
            )
        for field in ("source_crops_semantic_sha256", "box_geometry_semantic_sha256"):
            digest = alias.get(field)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise SystemExit(f"{path}: migration_aliases[{index}] has invalid {field}")
        seen_aliases.add(alias_id)
        migration_aliases.append({
            "dataset_id": alias_id,
            "classes": list(alias_classes),
            "strategy": strategy,
            "source_crops_semantic_sha256": alias["source_crops_semantic_sha256"],
            "box_geometry_semantic_sha256": alias["box_geometry_semantic_sha256"],
        })

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
            if not isinstance(box, dict):
                raise SystemExit(f"{path}: {crop_name} has invalid box class")
            class_name = box.get("cls")
            is_unknown = unknown_class is not None and class_name == unknown_class
            if class_name not in by_class and not is_unknown:
                raise SystemExit(f"{path}: {crop_name} has invalid box class")
            geometry: dict[str, float] = {}
            for field in ("x", "y", "w", "h"):
                value = box.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise SystemExit(
                        f"{path}: {crop_name} box has non-numeric {field}={value!r}"
                    )
                geometry[field] = float(value)
            converted_box: dict[str, object] = {
                "cls": UNKNOWN_CLASS_INDEX if is_unknown else by_class[str(class_name)],
                **geometry,
            }
            for field in ("source_cls", "prelabel_reason", "needs_review"):
                if field in box:
                    converted_box[field] = box[field]
            converted.append(converted_box)
        if status == "verified" and any(
            box["cls"] == UNKNOWN_CLASS_INDEX for box in converted
        ):
            raise SystemExit(
                f"{path}: {crop_name} is verified but still contains {unknown_class!r}"
            )
        state_record: dict[str, object] = {
            "status": status,
            "notes": str(record.get("notes", "")),
            "boxes": converted,
        }
        for field in ("source_status", "needs_review"):
            if field in record:
                state_record[field] = record[field]
        state[crop_name] = state_record
    return state, {
        "unknown_class": unknown_class,
        "migration_aliases": migration_aliases,
    }


def load_seed(
    path: Path,
    *,
    dataset_name: str,
    dataset_id: str,
    crop_names: set[str],
    class_names: list[str],
) -> dict[str, dict[str, object]]:
    """Validate a seed and return browser state (backwards-compatible API)."""

    state, _ = load_seed_contract(
        path,
        dataset_name=dataset_name,
        dataset_id=dataset_id,
        crop_names=crop_names,
        class_names=class_names,
    )
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
        classes = [{"name": n, "vn": PACKAGE_VI.get(n, ""), "color": palette[i % len(palette)]}
                   for i, n in enumerate(args.classes)]
    else:
        classes = DEFAULT_CLASSES

    rows = load_rows(manifest, crops)
    class_names = [str(c["name"]) for c in classes]
    dataset_id = dataset_id_for(root, rows, class_names)
    initial_state: dict[str, dict[str, object]] = {}
    seed_contract: dict[str, object] = {
        "unknown_class": None,
        "migration_aliases": [],
    }
    if args.seed_json is not None:
        initial_state, seed_contract = load_seed_contract(
            args.seed_json.resolve(),
            dataset_name=root.name,
            dataset_id=dataset_id,
            crop_names={str(row["crop_path"]) for row in rows},
            class_names=class_names,
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
        "unknown_class": seed_contract["unknown_class"],
        "migration_aliases": seed_contract["migration_aliases"],
        # Vân tay của template đã dựng ra trang này. Trang nằm trong .gitignore
        # (nó nhúng cả ảnh), nên sửa template xong mà quên dựng lại thì người
        # duyệt vẫn mở trang cũ và KHÔNG có gì báo. Đã xảy ra thật: bản vá chống
        # ghi đè khi nạp file có trong template nhưng vắng ở trang trên đĩa.
        "template_sha256": template_digest(),
    }
    component_body_task = class_names == ["component"]
    package_task = tuple(class_names) == PACKAGE_CLASSES
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
    elif package_task:
        wording = {
            "__PAGE_TITLE__": "Gán PACKAGE cho thân linh kiện",
            "__CLEAN_FILTER__": "không có linh kiện",
            "__CLEAN_BUTTON__": "Không có linh kiện",
            "__DECISION_NOTE__": (
                "Chọn từng box rồi bấm <b>1–7</b>. <b>Enter</b> chỉ xác nhận khi mọi box "
                "đã có một trong bảy package; box <b>unknown</b> màu hồng sẽ chặn xác nhận "
                "và chặn export."
            ),
            "__HELP_INTRO__": (
                "Box thân linh kiện đã được giữ nguyên từ lượt duyệt trước. Việc cần làm là "
                "<b>chọn box và bấm 1–7 để gán package</b>; chỉ sửa hình học nếu box thân "
                "thật sự sai."
            ),
            "__SHORTCUT_C__": (
                "xác nhận tile thật sự <b>không có linh kiện</b> rồi sang ảnh sau"
            ),
            "__BOX_GUIDANCE__": (
                "Giữ box ôm sát <b>thân/gói/vỏ linh kiện</b>, không bao chân, pad hay thiếc. "
                "Các box đã duyệt được migration nguyên tọa độ; package chỉ thay nhãn lớp."
            ),
            "__CLASS_GUIDANCE__": (
                "Bảy phím tương ứng: 1 hai chân; 2 trụ đứng; 3 gói nhỏ 3–5 chân; "
                "4 IC chân hai bên; 5 IC chân bốn bên; 6 IC không thấy chân; "
                "7 connector/xuyên lỗ. Không chắc thì để unknown, không xác nhận bừa."
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
