"""Convert an existing component-body draft into a seven-package label draft.

The input is evidence: this tool never rewrites it and never changes, drops or
reorders a box.  It only replaces the old box class with a conservative package
prelabel.  Ambiguous boxes use the ``unknown`` sentinel; that sentinel is *not*
part of the seven-class training taxonomy and must be resolved by a reviewer.

Typical round-2 use::

    python scripts/prepare_package_labelling.py \
        datasets/labelling/component_bodies_round2_20260830/draft_boxes.json

The default output is ``draft_package_boxes.json`` beside the source draft.
The write is atomic and refuses to overwrite an existing path.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import unicodedata
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


PACKAGE_CLASSES: tuple[str, ...] = (
    "hai_chan",
    "tru_dung",
    "goi_nho",
    "ic_hai_ben",
    "ic_bon_ben",
    "ic_khong_chan",
    "connector",
)
UNKNOWN_CLASS = "unknown"
ALLOWED_STATUSES = frozenset({"", "verified", "skipped", "unusable"})
COORDINATE_SPACE = "crop_pixels_top_left_origin"

# A label is trusted only when it names a visible package/family unambiguously.
# In particular, generic ``ic``/``transistor`` labels are deliberately absent:
# box aspect and area cannot tell a leaded IC from QFN/BGA safely.
_LABEL_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("tru_dung", (
        "electrolytic", "aluminium can", "aluminum can", "radial capacitor",
        "vertical capacitor", "tu hoa", "tru dung",
    )),
    ("ic_khong_chan", (
        "qfn", "dfn", "bga", "lga", "wlcsp", "wafer level", "chip scale package",
    )),
    ("ic_bon_ben", (
        "qfp", "tqfp", "lqfp", "pqfp", "cqfp", "plcc",
    )),
    ("ic_hai_ben", (
        "soic", "ssop", "tssop", "msop", "sop", "small outline", "dip",
    )),
    ("goi_nho", (
        "sot", "to 92", "to92", "mosfet 2", "mosfet2",
    )),
    ("connector", (
        "connector", "pin header", "header", "terminal block", "socket", "receptacle",
        "usb", "rj45",
    )),
    ("hai_chan", (
        "resistor", "capacitor", "diode", "inductor", "polyfuse", "fuse", "ferrite",
        "feriet", "melf", "sod", "led", "0201", "0402", "0603", "0805", "1206",
        "1210", "1812", "2512",
    )),
)

_LABEL_FIELDS = (
    "package", "package_class", "footprint", "label", "source_label",
    "component_class", "class_name",
)


def semantic_sha256(value: object) -> str:
    """Hash JSON meaning rather than whitespace or dictionary insertion order."""

    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _geometry_projection(crops: Mapping[str, object]) -> dict[str, list[dict[str, object]]]:
    """Return the exact ordered xywh payload used by the no-box-loss invariant."""

    projection: dict[str, list[dict[str, object]]] = {}
    for crop_name in sorted(crops):
        record = crops[crop_name]
        if not isinstance(record, Mapping):
            raise SystemExit(f"{crop_name}: crop record must be an object")
        boxes = record.get("boxes", [])
        if not isinstance(boxes, list):
            raise SystemExit(f"{crop_name}: boxes must be a list")
        projected: list[dict[str, object]] = []
        for index, box in enumerate(boxes):
            if not isinstance(box, Mapping):
                raise SystemExit(f"{crop_name}: box {index} must be an object")
            geometry: dict[str, object] = {}
            for field in ("x", "y", "w", "h"):
                value = box.get(field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise SystemExit(
                        f"{crop_name}: box {index} has invalid {field}={value!r}"
                    )
                geometry[field] = value
            projected.append(geometry)
        projection[crop_name] = projected
    return projection


def _normalise_label(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").strip().lower())
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _package_from_label(value: object) -> tuple[str, str] | None:
    normalised = _normalise_label(value)
    if not normalised or normalised in {"component", "component candidate", "ic", "transistor"}:
        return None

    slug = normalised.replace(" ", "_")
    if slug in PACKAGE_CLASSES:
        return slug, f"explicit_package:{normalised}"

    padded = f" {normalised} "
    for package_class, patterns in _LABEL_PATTERNS:
        for pattern in patterns:
            token = _normalise_label(pattern)
            if f" {token} " in padded or normalised.startswith(f"{token} "):
                return package_class, f"source_label:{normalised}"
    return None


def _candidate_labels(
    box: Mapping[str, object],
    row: Mapping[str, object],
    *,
    boxes_in_crop: int,
) -> list[object]:
    values = [box[field] for field in _LABEL_FIELDS if box.get(field) not in (None, "")]
    old_class = box.get("cls")
    if old_class not in (None, "", "component"):
        values.append(old_class)

    # A one-component crop may legitimately carry its family in the manifest.
    # A board tile cannot: ``tile ~24 linh kiện`` describes density, not any one
    # box, so applying it to every box would manufacture labels.
    row_label = row.get("component_class", "")
    if boxes_in_crop == 1 and not _normalise_label(row_label).startswith("tile "):
        values.append(row_label)
    return values


def _geometry_prelabel(
    box: Mapping[str, object], row: Mapping[str, object],
) -> tuple[str, str] | None:
    """Không đoán package từ hình học nữa. Luôn trả ``None``.

    Bản trước gán ``hai_chan`` khi ``aspect >= 3.2`` với lý lẽ "thân rất nhỏ và
    thuôn dài gần như chắc chắn là linh kiện hai chân". Lý lẽ đó **ngược**: chip
    hai chân thật (0402/0603/0805) có tỉ lệ thân khoảng 1,5–2,5, và trên chính
    hàng đợi này trung vị tỉ lệ đo được là 1,38. Ngưỡng ``>= 3.2`` vì thế chọn
    đúng nhóm *ít khả năng là hai chân nhất*.

    Đã kiểm bằng ảnh trên cả 8 box mà luật bắn trúng (tỉ lệ 4,7 · 17,9 · 3,3 ·
    8,3 · 9,6 · 3,4 · 4,5 · 3,5): ít nhất ba cái là **connector / hàng chân /
    mép IC**, không phải linh kiện hai chân. Xem
    ``Docs/danh_gia/danh_gia_khoanh_box_than_linh_kien.md``.

    Luật chỉ bắn 8/3855 box (0,2%) nên không tiết kiệm được công đáng kể, trong
    khi mỗi lần bắn sai lại tạo ra một box *trông như đã xong* — thứ người duyệt
    dễ bấm qua theo phản xạ. Đổi lấy: không điền sẵn gì cả, mọi box về
    ``unknown`` và người duyệt tự chọn.

    Muốn điền sẵn lại thì đo trước: gán tay vài trăm box, tính tỉ lệ đúng của
    quy tắc ứng viên, rồi mới bật. Đừng suy từ hình học ra nhãn bằng trực giác.
    """

    return None


def _prelabel(
    box: Mapping[str, object],
    row: Mapping[str, object],
    *,
    boxes_in_crop: int,
) -> tuple[str, str]:
    for label in _candidate_labels(box, row, boxes_in_crop=boxes_in_crop):
        mapped = _package_from_label(label)
        if mapped is not None:
            return mapped
    geometric = _geometry_prelabel(box, row)
    if geometric is not None:
        return geometric
    return UNKNOWN_CLASS, "review:label_or_geometry_ambiguous"


def _read_payload(path: Path) -> dict[str, Any]:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read source draft {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: root must be an object")
    return payload


def _write_json_atomic_no_overwrite(path: Path, payload: Mapping[str, object]) -> None:
    """Publish a complete JSON file atomically without a replace race.

    Linking a fully flushed same-directory temporary file gives both properties:
    readers never see a partial document and ``os.link`` fails if another writer
    created the destination after our initial check.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=1, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise SystemExit(f"output already exists; refusing to overwrite: {path}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare(
    source: Path,
    output: Path,
    *,
    dataset_root: Path | None = None,
    dry_run: bool = False,
) -> int:
    """Prepare a package draft while preserving the source's box geometry."""

    from scripts.build_joint_box_app import dataset_id_for, load_rows

    source = source.resolve()
    output = output.resolve()
    root = (dataset_root or source.parent).resolve()
    if source == output:
        raise SystemExit("input and output must be different; source drafts are immutable evidence")
    if not source.is_file():
        raise SystemExit(f"source draft does not exist: {source}")
    if output.exists():
        raise SystemExit(f"output already exists; refusing to overwrite: {output}")

    manifest, crop_dir = root / "manifest.csv", root / "crops"
    for required in (manifest, crop_dir):
        if not required.exists():
            raise SystemExit(f"missing {required}")
    rows = load_rows(manifest, crop_dir)
    rows_by_name = {str(row["crop_path"]): row for row in rows}

    source_payload = _read_payload(source)
    expected_header = {
        "schema": "aoi-joint-boxes/1.0",
        "dataset": root.name,
        "coordinate_space": COORDINATE_SPACE,
    }
    for field, expected in expected_header.items():
        if source_payload.get(field) != expected:
            raise SystemExit(
                f"{source}: {field} mismatch; expected {expected!r}, "
                f"got {source_payload.get(field)!r}"
            )
    source_classes = source_payload.get("classes")
    if (
        not isinstance(source_classes, list)
        or not source_classes
        or any(not isinstance(name, str) or not name for name in source_classes)
    ):
        raise SystemExit(f"{source}: classes must be a non-empty string list")
    if source_classes == list(PACKAGE_CLASSES):
        raise SystemExit(f"{source}: already uses the seven package classes")

    expected_source_id = dataset_id_for(root, rows, source_classes)
    if source_payload.get("dataset_id") != expected_source_id:
        raise SystemExit(
            f"{source}: dataset_id mismatch; expected {expected_source_id!r}, "
            f"got {source_payload.get('dataset_id')!r}"
        )
    source_dataset_id = expected_source_id
    package_dataset_id = dataset_id_for(root, rows, list(PACKAGE_CLASSES))

    source_crops = source_payload.get("crops")
    if not isinstance(source_crops, dict):
        raise SystemExit(f"{source}: crops must be an object keyed by crop_path")
    missing = sorted(set(rows_by_name) - set(source_crops))
    extra = sorted(set(source_crops) - set(rows_by_name))
    if missing or extra:
        raise SystemExit(
            f"{source}: crops do not match manifest; missing={missing[:3]}, extra={extra[:3]}"
        )

    source_geometry = _geometry_projection(source_crops)
    source_geometry_sha256 = semantic_sha256(source_geometry)
    source_crops_sha256 = semantic_sha256(source_crops)

    converted_crops: dict[str, dict[str, object]] = {}
    counts = {name: 0 for name in (*PACKAGE_CLASSES, UNKNOWN_CLASS)}
    source_verified = 0
    for crop_name, source_record in source_crops.items():
        if not isinstance(source_record, dict):
            raise SystemExit(f"{source}: {crop_name} record must be an object")
        source_status = source_record.get("status", "")
        if source_status not in ALLOWED_STATUSES:
            raise SystemExit(f"{source}: {crop_name} has invalid status {source_status!r}")
        if source_status == "verified":
            source_verified += 1
        source_boxes = source_record.get("boxes", [])
        if not isinstance(source_boxes, list):
            raise SystemExit(f"{source}: {crop_name} boxes must be a list")

        record = deepcopy(source_record)
        converted_boxes: list[dict[str, object]] = []
        for index, source_box in enumerate(source_boxes):
            if not isinstance(source_box, dict):
                raise SystemExit(f"{source}: {crop_name} box {index} must be an object")
            old_class = source_box.get("cls")
            if old_class not in source_classes:
                raise SystemExit(
                    f"{source}: {crop_name} box {index} has class {old_class!r} "
                    f"outside source classes"
                )
            package_class, reason = _prelabel(
                source_box, rows_by_name[crop_name], boxes_in_crop=len(source_boxes),
            )
            box = deepcopy(source_box)
            box["source_cls"] = old_class
            box["cls"] = package_class
            box["needs_review"] = package_class == UNKNOWN_CLASS
            box["prelabel_reason"] = reason
            converted_boxes.append(box)
            counts[package_class] += 1
        record["boxes"] = converted_boxes
        if source_status:
            record["source_status"] = source_status
        # A verified component-body outline is not a verified package label.
        # Preserve skip/unusable actions, and preserve verified only for an
        # explicitly empty tile where there is no package decision to make.
        if source_boxes and source_status not in {"skipped", "unusable"}:
            record["status"] = ""
            record["needs_review"] = True
        converted_crops[crop_name] = record

    converted_geometry = _geometry_projection(converted_crops)
    converted_geometry_sha256 = semantic_sha256(converted_geometry)
    if converted_geometry != source_geometry or converted_geometry_sha256 != source_geometry_sha256:
        raise RuntimeError("internal error: package conversion changed box geometry")

    controlled = {
        "dataset_id", "classes", "crops", "unknown_class", "migration_aliases",
        "source_dataset_id", "source_classes", "source_crops_semantic_sha256",
        "package_crops_semantic_sha256", "box_geometry_semantic_sha256",
        "carried_verified_semantic_sha256",
    }
    result: dict[str, object] = {
        key: deepcopy(value)
        for key, value in source_payload.items()
        if key not in controlled
    }
    if source_payload.get("carried_verified_semantic_sha256"):
        result["source_carried_verified_semantic_sha256"] = source_payload[
            "carried_verified_semantic_sha256"
        ]
    result.update({
        "schema": "aoi-joint-boxes/1.0",
        "dataset_id": package_dataset_id,
        "dataset": root.name,
        "coordinate_space": COORDINATE_SPACE,
        "classes": list(PACKAGE_CLASSES),
        # ``unknown`` is an editor sentinel, never an eighth training class.
        # The app builder requires this explicit declaration before accepting
        # such a box, so a typo in an ordinary dataset cannot silently become
        # an unresolved package label.
        "unknown_class": UNKNOWN_CLASS,
        "source_dataset_id": source_dataset_id,
        "source_classes": list(source_classes),
        "source_crops_semantic_sha256": source_crops_sha256,
        "box_geometry_semantic_sha256": source_geometry_sha256,
        "migration_aliases": [{
            "dataset_id": source_dataset_id,
            "classes": list(source_classes),
            "source_crops_semantic_sha256": source_crops_sha256,
            "box_geometry_semantic_sha256": source_geometry_sha256,
            "strategy": "preserve_geometry_reset_box_classes_to_unknown",
        }],
        "package_prelabel_summary": {
            "source_verified_tiles": source_verified,
            "box_count": sum(counts.values()),
            "by_class": counts,
            "policy": (
                "Only explicit family/package labels and extremely small elongated bodies "
                "are prelabelled; ambiguous boxes remain unknown and block verification."
            ),
        },
        "note": (
            "Mọi xywh được giữ nguyên từ source draft. Package chỉ là prelabel; "
            "unknown không thuộc taxonomy train và phải được người gán nhãn chọn 1–7."
        ),
        "crops": converted_crops,
    })
    result["package_crops_semantic_sha256"] = semantic_sha256(converted_crops)

    summary = result["package_prelabel_summary"]
    assert isinstance(summary, dict)
    if dry_run:
        print(f"[dry-run] validated {len(converted_crops)} crops; would write {output}")
    else:
        _write_json_atomic_no_overwrite(output, result)
        print(f"wrote {output}")
    print(
        f"  dataset_id: {source_dataset_id} -> {package_dataset_id}; "
        f"geometry sha256: {source_geometry_sha256}"
    )
    print(
        f"  {summary['box_count']} boxes: {counts[UNKNOWN_CLASS]} unknown cần chọn; "
        f"{summary['box_count'] - counts[UNKNOWN_CLASS]} prelabel bảo thủ"
    )
    if not dry_run:
        class_args = " ".join(PACKAGE_CLASSES)
        print("\nBước tiếp:")
        print(f"  python scripts/build_joint_box_app.py {root} \\")
        print(f"      --classes {class_args} \\")
        print(f"      --seed-json {output} --output {root / 'label_packages.html'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="existing draft_boxes.json")
    parser.add_argument(
        "--output", type=Path,
        help="default: <source folder>/draft_package_boxes.json; never overwritten",
    )
    parser.add_argument(
        "--dataset-root", type=Path,
        help="folder containing manifest.csv and crops/ (default: source parent)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    source = args.source.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else source.with_name("draft_package_boxes.json")
    )
    return prepare(
        source,
        output,
        dataset_root=args.dataset_root.resolve() if args.dataset_root else None,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
