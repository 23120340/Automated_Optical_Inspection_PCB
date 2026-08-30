"""Pack reviewed component-body tiles with leakage-safe public bootstrap data.

The target task has one class: ``component`` means the visible component body,
not its leads, pads, test points, or a loose component photographed off-board.
This command joins three inputs without modifying any of them:

* a ``component_bodies`` labelling folder plus an exported
  ``aoi-joint-boxes/1.0`` checkpoint;
* RF100 ``printed-circuit-board`` v4 (the local folder is historically, and
  incorrectly, named ``fpic_boards_rf100``);
* Winnies ``pcb-components-wc8ms`` v3.

The rules live here instead of in a notebook so a training run cannot bypass
them accidentally:

* only locally ``verified`` tiles are ground truth;
* exact duplicate local pixels are packed once and conflicting verified copies
  fail closed;
* Roboflow variants are grouped by their pre-``.rf.`` source scene and only one
  deterministic representative is retained;
* all recordings and aliases of one physical PCB-DSLR board share one group;
* public data is train-only; validation and test contain reviewed target tiles;
* a public image of a target board held out for validation/test is excluded;
* ambiguous RF100 scenes are quarantined rather than turned into false
  background labels.

Audit the current checkpoint without writing anything::

    python scripts/pack_component_detection_dataset.py \
        datasets/labelling/component_bodies \
        --boxes ~/Downloads/joint_boxes.json \
        --pcb-dslr-reference-root datasets/reference_sets/pcb_dslr_30_diverse \
        --require-ic-audit-pass --audit-only

Build after enough distinct target boards have been reviewed::

    python scripts/pack_component_detection_dataset.py \
        datasets/labelling/component_bodies \
        --boxes ~/Downloads/joint_boxes.json \
        --output datasets/train/component_detect_v2
"""

from __future__ import annotations

import argparse
import collections
import csv
from dataclasses import dataclass, field
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import sys
import tempfile
from typing import Any, Iterable, Sequence
import zipfile

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]{32}", re.IGNORECASE)
SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
PCB_DSLR_BOARD = re.compile(
    r"^pcb(?:_dslr_)?0*(?P<board>[0-9]+)(?:(?:__)?rec[0-9]+)",
    re.IGNORECASE,
)
PCB_DSLR_BOARD_ONLY = re.compile(
    r"^pcb(?:_dslr_)?0*(?P<board>[0-9]+)(?:$|__|_|-)",
    re.IGNORECASE,
)

RF100_TAG = "rf100_printed_circuit_board_v4"
CONSOLIDATED_TAG = "pcb_component_detection_consolidated_v1"
WINNIES_TAG = "winnies_pcb_components_v3"
LOCAL_TAG = "local_component_bodies"
CHECKPOINT_STATUSES = frozenset({"verified", "skipped", "unusable"})

RF100_CLASSES = (
    "Button",
    "Capacitor Jumper",
    "Capacitor",
    "Clock",
    "Connector",
    "Diode",
    "EM",
    "Electrolytic Capacitor",
    "Ferrite Bead",
    "IC",
    "Inductor",
    "Jumper",
    "Led",
    "Pads",
    "Pins",
    "Resistor Jumper",
    "Resistor Network",
    "Resistor",
    "Switch",
    "Test Point",
    "Transistor",
    "Unknown Unlabeled",
    "iC",
)
RF100_BODY_CLASSES = frozenset(
    {
        "Button",
        "Capacitor Jumper",
        "Capacitor",
        "Clock",
        "Connector",
        "Diode",
        "Electrolytic Capacitor",
        "Ferrite Bead",
        "IC",
        "Inductor",
        "Jumper",
        "Led",
        "Resistor Jumper",
        "Resistor Network",
        "Resistor",
        "Switch",
        "Transistor",
        "iC",
    }
)
RF100_NON_BODY_CLASSES = frozenset({"Pads", "Pins", "Test Point"})
RF100_AMBIGUOUS_CLASSES = frozenset({"EM", "Unknown Unlabeled"})

WINNIES_CLASSES = (
    "CHIP",
    "LED",
    "MOSFET",
    "MOSFET-2",
    "Polyfuse_GR",
    "Polyfuse_Z",
    "Resistor rond",
    "SOD123",
    "SOD128",
    "SOD323",
    "SOIC-12",
    "SOIC-14",
    "SOIC-16",
    "SOT143",
    "SOT223",
    "SOT23",
    "SOT457",
    "SOT753",
    "SOT96",
    "TSSOP-14",
    "TSSOP-16",
    "capacitor",
    "feriet kraal",
    "resistor",
)

#: Thứ tự đúng như `components_data_uncropped/data.yaml`; chỉ số phải khớp
#: `class_map` trong models/active/detector/model_manifest.json, vì đây chính
#: là bộ đã train ra detector đang chạy.
CONSOLIDATED_CLASSES = (
    "battery", "button", "buzzer", "capacitor", "clock", "connector", "diode",
    "display", "fuse", "heatsink", "ic", "inductor", "led", "pads", "pins",
    "potentiometer", "relay", "resistor", "switch", "transducer", "transformer",
    "transistor",
)
#: `pads`/`pins` là VÙNG HÀN, không phải thân linh kiện. Giữ chúng lại trong một
#: detector một lớp `component` là dạy model gọi mối hàn là linh kiện.
CONSOLIDATED_NON_BODY_CLASSES = frozenset({"pads", "pins"})

PUBLIC_SPECS = {
    RF100_TAG: {
        "workspace": "roboflow-100",
        "project": "printed-circuit-board",
        "version": "4",
        "license": "CC BY 4.0",
        "url": "https://universe.roboflow.com/roboflow-100/printed-circuit-board/dataset/4",
        "classes": RF100_CLASSES,
        "non_body_classes": RF100_NON_BODY_CLASSES,
        "ambiguous_classes": RF100_AMBIGUOUS_CLASSES,
        "roboflow_metadata": True,
        "provenance_caveat": (
            "Some numbered boards overlap TU Wien PCB-DSLR, whose official terms "
            "limit use to noncommercial research; downstream CC BY metadata does "
            "not by itself clear commercial reuse."
        ),
    },
    WINNIES_TAG: {
        "workspace": "winnies-workspace-0yaec",
        "project": "pcb-components-wc8ms",
        "version": "3",
        "license": "CC BY 4.0",
        "url": "https://universe.roboflow.com/winnies-workspace-0yaec/pcb-components-wc8ms/dataset/3",
        "classes": WINNIES_CLASSES,
        "non_body_classes": frozenset(),
        "ambiguous_classes": frozenset(),
        "roboflow_metadata": True,
        "provenance_caveat": (
            "Community-origin dataset; retain attribution and keep locked "
            "target validation/test independent."
        ),
    },
    CONSOLIDATED_TAG: {
        "workspace": "aryanstein",
        "project": "pcb-component-detection-consolidated-dataset",
        "version": "1",
        "license": "Apache-2.0 (khai bởi người đăng Kaggle)",
        "url": "https://www.kaggle.com/datasets/aryanstein/pcb-component-detection-consolidated-dataset",
        "classes": CONSOLIDATED_CLASSES,
        "non_body_classes": CONSOLIDATED_NON_BODY_CLASSES,
        "ambiguous_classes": frozenset(),
        # Gói Kaggle, không có khối `roboflow:` trong data.yaml.
        "roboflow_metadata": False,
        "provenance_caveat": (
            "Hợp nhất từ WACV/FICS-PCB/PCB-Vision/CompDetect; Apache-2.0 là khai "
            "báo của người đăng, giấy phép từng nguồn thành phần phải tự kiểm. "
            "Ảnh của nó CHỒNG với kho tile của dự án -- xem fixture_leaks trong "
            "báo cáo audit."
        ),
    },
}


for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


@dataclass(frozen=True, slots=True)
class YoloBox:
    """One normalized YOLO box after clipping to the unit square."""

    cx: float
    cy: float
    width: float
    height: float

    def line(self) -> str:
        return (
            f"0 {self.cx:.6f} {self.cy:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )


@dataclass(slots=True)
class PackedImage:
    source_tag: str
    source_scene: str
    group_id: str
    boxes: list[YoloBox]
    width: int
    height: int
    original_name: str
    pixel_sha256: str
    local_path: Path | None = None
    archive_path: Path | None = None
    archive_member: str | None = None
    split: str = "train"

    def read_bytes(self) -> bytes:
        if self.local_path is not None:
            return self.local_path.read_bytes()
        if self.archive_path is None or self.archive_member is None:
            raise RuntimeError(f"{self.original_name}: no image source")
        with zipfile.ZipFile(self.archive_path) as archive:
            return archive.read(self.archive_member)


@dataclass(slots=True)
class PublicCandidate:
    source_tag: str
    scene: str
    group_id: str
    image_member: str
    label_member: str
    width: int
    height: int
    boxes: list[YoloBox]
    original_classes: collections.Counter[str]


@dataclass(slots=True)
class PackPlan:
    local: list[PackedImage]
    public: list[PackedImage]
    split_of_group: dict[str, str]
    report: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    """Return provenance without leaking an absolute workstation path."""

    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.name


def source_scene(name: str) -> str:
    """Source-scene stem behind a Roboflow ``.rf.<hash>`` variant."""

    basename = Path(name).name
    without_rf = RF_SUFFIX.sub("", basename)
    return Path(without_rf).stem


def canonical_board_id(value: str, source_tag: str) -> str:
    """Canonical physical-board group, shared across known PCB-DSLR aliases.

    ``pcb7rec1_jpg``, ``pcb7__rec5`` and ``pcb_dslr_007__rec1`` all map to
    ``pcb_dslr:007``.  The recording number is intentionally discarded: five
    captures of one board are not independent ML samples. Unknown naming
    schemes remain namespaced by source so unrelated datasets cannot collide.
    """

    scene = source_scene(value)
    match = PCB_DSLR_BOARD.match(scene) or PCB_DSLR_BOARD_ONLY.match(scene)
    if match:
        return f"pcb_dslr:{int(match.group('board')):03d}"
    normalized = SAFE_NAME.sub("_", scene.strip()).strip("._").lower() or "scene"
    return f"{source_tag}:{normalized}"


def read_class_names(text: str) -> list[str]:
    """Read an inline or block-form ``names`` list without adding PyYAML."""

    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("names:"):
            continue
        inline = stripped.split("names:", 1)[1].strip()
        if inline.startswith("["):
            if "]" not in inline:
                raise SystemExit("data.yaml has an unterminated inline names list")
            body = inline[1 : inline.rindex("]")]
            return [item.strip().strip("'\"") for item in body.split(",") if item.strip()]
        names: list[str] = []
        for follow in lines[index + 1 :]:
            item = follow.strip()
            if item.startswith("- "):
                names.append(item[2:].strip().strip("'\""))
            elif item and not item.startswith("#"):
                break
        return names
    raise SystemExit("data.yaml has no names list")


def read_roboflow_metadata(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    in_block = False
    for line in text.splitlines():
        if line.strip() == "roboflow:":
            in_block = True
            continue
        if not in_block:
            continue
        if line and not line[0].isspace():
            break
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        metadata[key.strip()] = value.strip().strip("'\"")
    return metadata


def read_boxes(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    if payload.get("schema") != "aoi-joint-boxes/1.0":
        raise SystemExit(f"{path}: unexpected schema {payload.get('schema')!r}")
    if payload.get("coordinate_space") != "crop_pixels_top_left_origin":
        raise SystemExit(
            f"{path}: unexpected coordinate space "
            f"{payload.get('coordinate_space')!r}"
        )
    if payload.get("classes") != ["component"]:
        raise SystemExit(
            f"{path}: expected exactly ['component'], got {payload.get('classes')!r}"
        )
    if not isinstance(payload.get("crops"), dict):
        raise SystemExit(f"{path}: crops must be an object")
    return payload


def _expected_checkpoint_dataset_id(
    crop_root: Path,
    manifest_rows: Sequence[dict[str, str]],
) -> str:
    """Reproduce the identity embedded by ``build_joint_box_app.py``.

    The browser app keys both localStorage and its JSON export to this value.
    Keeping the calculation byte-for-byte compatible makes a checkpoint from
    another generated crop set fail closed instead of being joined by filename.
    """

    if not manifest_rows:
        raise SystemExit(f"{crop_root / 'manifest.csv'} has no rows")
    first_crop = manifest_rows[0].get("crop_path", "")
    if not first_crop:
        raise SystemExit(f"{crop_root / 'manifest.csv'}: first row has no crop_path")
    identity = f"{crop_root.name}|{len(manifest_rows)}|{first_crop}|component"
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


def assign_splits(
    groups: Sequence[str],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, str]:
    """Assign stable per-board hash buckets.

    The result for one group depends only on ``seed`` and that group ID, never
    on how many other boards have been reviewed.  Sorting then shuffling the
    whole current set looks deterministic but is unsafe for an active labelling
    project: adding board 11 can move board 7 from test back into train.
    """

    if any(not math.isfinite(value) or value < 0 for value in ratios):
        raise ValueError("split ratios must be finite and non-negative")
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError("split ratios must sum to 1")
    result: dict[str, str] = {}
    test_boundary = ratios[2]
    valid_boundary = test_boundary + ratios[1]
    for group in sorted(set(groups)):
        digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
        score = int.from_bytes(digest[:8], "big") / float(1 << 64)
        if score < test_boundary:
            result[group] = "test"
        elif score < valid_boundary:
            result[group] = "valid"
        else:
            result[group] = "train"
    return result


def _pixel_identity_from_bytes(blob: bytes) -> tuple[int, int, str]:
    try:
        with Image.open(io.BytesIO(blob)) as handle:
            rgb = handle.convert("RGB")
            width, height = rgb.size
            pixel_hash = _pixel_sha256(rgb)
    except Exception as exc:
        raise SystemExit(f"cannot decode image: {exc}") from exc
    return width, height, pixel_hash


def _pixel_sha256(rgb: Image.Image) -> str:
    """Hash decoded RGB pixels, independent of PNG/JPEG container metadata."""

    converted = rgb if rgb.mode == "RGB" else rgb.convert("RGB")
    width, height = converted.size
    digest = hashlib.sha256()
    digest.update(f"RGB:{width}x{height}:".encode("ascii"))
    digest.update(converted.tobytes())
    return digest.hexdigest()


def _pixel_identity(path: Path) -> tuple[int, int, str]:
    try:
        return _pixel_identity_from_bytes(path.read_bytes())
    except OSError as exc:
        raise SystemExit(f"cannot read image {path}: {exc}") from exc


def _normalise_xywh(
    cx: float,
    cy: float,
    width: float,
    height: float,
) -> YoloBox | None:
    values = (cx, cy, width, height)
    if not all(math.isfinite(value) for value in values):
        return None
    x0 = max(0.0, min(1.0, cx - width / 2.0))
    y0 = max(0.0, min(1.0, cy - height / 2.0))
    x1 = max(0.0, min(1.0, cx + width / 2.0))
    y1 = max(0.0, min(1.0, cy + height / 2.0))
    clipped_width, clipped_height = x1 - x0, y1 - y0
    if clipped_width <= 1e-9 or clipped_height <= 1e-9:
        return None
    return YoloBox(
        (x0 + x1) / 2.0,
        (y0 + y1) / 2.0,
        clipped_width,
        clipped_height,
    )


def _local_box(box: dict[str, Any], width: int, height: int, crop_name: str) -> YoloBox | None:
    if box.get("cls") != "component":
        raise SystemExit(f"{crop_name}: unexpected class {box.get('cls')!r}")
    try:
        x = float(box["x"])
        y = float(box["y"])
        box_width = float(box["w"])
        box_height = float(box["h"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"{crop_name}: malformed box {box!r}") from exc
    return _normalise_xywh(
        (x + box_width / 2.0) / width,
        (y + box_height / 2.0) / height,
        box_width / width,
        box_height / height,
    )


def _box_signature(boxes: Iterable[YoloBox]) -> tuple[tuple[float, ...], ...]:
    return tuple(
        sorted(
            (round(box.cx, 8), round(box.cy, 8), round(box.width, 8), round(box.height, 8))
            for box in boxes
        )
    )


def _load_local(
    crop_root: Path,
    boxes_path: Path,
) -> tuple[list[PackedImage], dict[str, Any]]:
    manifest_path = crop_root / "manifest.csv"
    crops_dir = crop_root / "crops"
    for required in (manifest_path, crops_dir, boxes_path):
        if not required.exists():
            raise SystemExit(f"missing {required}")
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{manifest_path} has no rows")
    manifest = {row.get("crop_path", ""): row for row in rows}
    if "" in manifest:
        raise SystemExit(f"{manifest_path}: a row has no crop_path")
    payload = read_boxes(boxes_path)
    expected_dataset = crop_root.name
    if payload.get("dataset") != expected_dataset:
        raise SystemExit(
            f"{boxes_path}: checkpoint dataset mismatch; expected "
            f"{expected_dataset!r}, got {payload.get('dataset')!r}"
        )
    expected_dataset_id = _expected_checkpoint_dataset_id(crop_root, rows)
    if payload.get("dataset_id") != expected_dataset_id:
        raise SystemExit(
            f"{boxes_path}: checkpoint dataset_id mismatch; expected "
            f"{expected_dataset_id!r}, got {payload.get('dataset_id')!r}"
        )
    labelled: dict[str, dict[str, Any]] = payload["crops"]
    unknown = sorted(set(labelled) - set(manifest))
    if unknown:
        raise SystemExit(
            f"{len(unknown)} labelled crops are from a different crop set; "
            f"first: {unknown[:3]}"
        )

    status_counts: collections.Counter[str] = collections.Counter()
    hash_records: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    verified: list[PackedImage] = []
    dropped_degenerate = 0
    for name, record in sorted(labelled.items()):
        if not isinstance(record, dict):
            raise SystemExit(f"{name}: checkpoint record must be an object")
        status = record.get("status")
        if status not in CHECKPOINT_STATUSES:
            #: `""` không phải một status lạ mà là dấu hiệu của một nhầm lẫn cụ
            #: thể: đưa `draft_boxes.json` (bản nháp để seed app, mọi record còn
            #: trống) thay cho file app xuất ra. Gọi tên nhầm lẫn đó thay vì
            #: liệt kê các giá trị hợp lệ.
            if status == "":
                raise SystemExit(
                    f"{name}: status rỗng — đây là bản NHÁP (draft_boxes.json), "
                    "chưa ai duyệt. --boxes cần file do app gán nhãn xuất ra "
                    "(nút 'Xuất JSON', thường là ~/Downloads/joint_boxes (N).json)."
                )
            raise SystemExit(
                f"{name}: unsupported checkpoint status {status!r}; expected one of "
                f"{sorted(CHECKPOINT_STATUSES)}"
            )
        status = str(status)
        status_counts[status] += 1
        image_path = crops_dir / name
        if not image_path.is_file():
            raise SystemExit(f"missing labelled crop {image_path}")
        width, height, pixel_hash = _pixel_identity(image_path)
        row = manifest[name]
        expected = (int(row["crop_w"]), int(row["crop_h"]))
        if (width, height) != expected:
            raise SystemExit(
                f"{name}: pixels are {width}x{height}, manifest says "
                f"{expected[0]}x{expected[1]}"
            )
        hash_records[pixel_hash].append({"name": name, "status": status})
        if status != "verified":
            continue
        boxes: list[YoloBox] = []
        raw_boxes = record.get("boxes", [])
        if not isinstance(raw_boxes, list):
            raise SystemExit(f"{name}: boxes must be a list")
        for raw_box in raw_boxes:
            if not isinstance(raw_box, dict):
                raise SystemExit(f"{name}: malformed box {raw_box!r}")
            box = _local_box(raw_box, width, height, name)
            if box is None:
                dropped_degenerate += 1
            else:
                boxes.append(box)
        scene = str(row.get("scene_id") or name)
        verified.append(
            PackedImage(
                source_tag=LOCAL_TAG,
                source_scene=scene,
                group_id=canonical_board_id(scene, LOCAL_TAG),
                boxes=boxes,
                width=width,
                height=height,
                original_name=name,
                pixel_sha256=pixel_hash,
                local_path=image_path,
            )
        )

    by_hash: dict[str, list[PackedImage]] = collections.defaultdict(list)
    for image in verified:
        by_hash[image.pixel_sha256].append(image)
    kept: list[PackedImage] = []
    duplicate_verified_dropped: list[str] = []
    for pixel_hash, images in sorted(by_hash.items()):
        ordered = sorted(images, key=lambda item: item.original_name)
        signatures = {_box_signature(item.boxes) for item in ordered}
        if len(signatures) > 1:
            names = [item.original_name for item in ordered]
            raise SystemExit(
                "conflicting verified labels for exact duplicate pixels: "
                f"{names}. Resolve the duplicate before packing."
            )
        kept.append(ordered[0])
        duplicate_verified_dropped.extend(item.original_name for item in ordered[1:])

    duplicate_groups = [
        {
            "pixel_sha256": pixel_hash,
            "records": records,
            "verified_kept": next(
                (
                    item.original_name
                    for item in kept
                    if item.pixel_sha256 == pixel_hash
                ),
                None,
            ),
        }
        for pixel_hash, records in sorted(hash_records.items())
        if len(records) > 1
    ]
    unreviewed = len(rows) - len(labelled)
    #: Mọi board có tile trong crop set, kể cả board chưa ai duyệt. `verified`
    #: chỉ cho biết board nào ĐÃ là target group; cái người dùng cần biết khi
    #: một bucket trống là board nào SẼ rơi vào bucket đó nếu duyệt tiếp.
    candidate_groups: dict[str, dict[str, int]] = {}
    for row in rows:
        scene = str(row.get("scene_id") or row.get("crop_path") or "")
        if not scene:
            continue
        group = canonical_board_id(scene, LOCAL_TAG)
        entry = candidate_groups.setdefault(group, {"tiles": 0, "verified_tiles": 0})
        entry["tiles"] += 1
        record = labelled.get(str(row.get("crop_path", "")))
        if isinstance(record, dict) and record.get("status") == "verified":
            entry["verified_tiles"] += 1
    report = {
        "crop_set": _portable_path(crop_root),
        "candidate_groups": dict(sorted(candidate_groups.items())),
        "manifest_sha256": _sha256_file(manifest_path),
        "boxes_file": _portable_path(boxes_path),
        "boxes_sha256": _sha256_file(boxes_path),
        "dataset_id": payload.get("dataset_id", ""),
        "reviewer_id": payload.get("reviewer_id", ""),
        "exported_at": payload.get("exported_at", ""),
        "manifest_tiles": len(rows),
        "exported_records": len(labelled),
        "unreviewed_not_in_export": unreviewed,
        "status_counts": dict(sorted(status_counts.items())),
        "verified_before_exact_dedup": len(verified),
        "verified_after_exact_dedup": len(kept),
        "verified_boxes": sum(len(item.boxes) for item in kept),
        "dropped_degenerate_boxes": dropped_degenerate,
        "duplicate_verified_dropped": duplicate_verified_dropped,
        "duplicate_pixel_groups_in_export": duplicate_groups,
    }
    return sorted(kept, key=lambda item: item.original_name), report


def _zip_yaml(archive: zipfile.ZipFile) -> tuple[str, str]:
    candidates = [name for name in archive.namelist() if Path(name).name == "data.yaml"]
    if not candidates:
        raise SystemExit(f"{archive.filename}: no data.yaml")
    contents = {
        archive.read(name).decode("utf-8", errors="replace") for name in candidates
    }
    if len(contents) != 1:
        raise SystemExit(f"{archive.filename}: conflicting data.yaml files")
    return sorted(candidates, key=lambda name: (name.count("/"), name))[0], contents.pop()


def _label_member(image_member: str, names: set[str]) -> str | None:
    normalized = image_member.replace("\\", "/")
    marker = "/images/"
    if marker not in normalized:
        return None
    candidate = normalized.replace(marker, "/labels/", 1)
    candidate = str(Path(candidate).with_suffix(".txt")).replace("\\", "/")
    return candidate if candidate in names else None


def _parse_public_labels(
    raw: str,
    classes: Sequence[str],
) -> tuple[list[tuple[str, YoloBox]], int]:
    parsed: list[tuple[str, YoloBox]] = []
    dropped = 0
    for line_number, line in enumerate(raw.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            raise SystemExit(f"malformed YOLO label at line {line_number}: {line!r}")
        try:
            raw_index = float(parts[0])
            values = [float(value) for value in parts[1:5]]
        except ValueError as exc:
            raise SystemExit(
                f"malformed YOLO label at line {line_number}: {line!r}"
            ) from exc
        if not math.isfinite(raw_index) or not raw_index.is_integer():
            raise SystemExit(
                f"non-integer class index at line {line_number}: {parts[0]!r}"
            )
        index = int(raw_index)
        if not 0 <= index < len(classes):
            raise SystemExit(f"class index {index} is outside 0..{len(classes) - 1}")
        box = _normalise_xywh(*values)
        if box is None:
            dropped += 1
            continue
        parsed.append((classes[index], box))
    return parsed, dropped


def _public_candidate(
    archive: zipfile.ZipFile,
    image_member: str,
    label_member: str,
    classes: Sequence[str],
    source_tag: str,
) -> tuple[PublicCandidate, int]:
    try:
        with archive.open(image_member) as stream, Image.open(stream) as handle:
            width, height = handle.size
    except Exception as exc:
        raise SystemExit(f"{archive.filename}:{image_member}: cannot decode: {exc}") from exc
    parsed, dropped = _parse_public_labels(
        archive.read(label_member).decode("utf-8", errors="replace"), classes
    )
    original = collections.Counter(name for name, _box in parsed)
    non_body = PUBLIC_SPECS[source_tag].get("non_body_classes", frozenset())
    boxes = [box for name, box in parsed if name not in non_body]
    scene = source_scene(image_member)
    return (
        PublicCandidate(
            source_tag=source_tag,
            scene=scene,
            group_id=canonical_board_id(scene, source_tag),
            image_member=image_member,
            label_member=label_member,
            width=width,
            height=height,
            boxes=boxes,
            original_classes=original,
        ),
        dropped,
    )


def _validate_public_contract(
    archive_path: Path,
    source_tag: str,
    yaml_text: str,
) -> tuple[list[str], dict[str, str]]:
    spec = PUBLIC_SPECS[source_tag]
    classes = read_class_names(yaml_text)
    if tuple(classes) != tuple(spec["classes"]):
        raise SystemExit(
            f"{archive_path}: taxonomy drift for {source_tag}; expected "
            f"{list(spec['classes'])}, got {classes}"
        )
    if not spec.get("roboflow_metadata", True):
        # Nguồn ngoài Roboflow không có khối `roboflow:`; danh sách lớp ở trên
        # vẫn là hợp đồng, chỉ metadata là lấy từ spec.
        return classes, {
            key: str(spec[key]) for key in ("workspace", "project", "version", "license")
        }
    metadata = read_roboflow_metadata(yaml_text)
    for key in ("workspace", "project", "version", "license"):
        expected = str(spec[key])
        if metadata.get(key) != expected:
            raise SystemExit(
                f"{archive_path}: {key} is {metadata.get(key)!r}, expected {expected!r}"
            )
    return classes, metadata


def _load_public_archive(
    archive_path: Path,
    source_tag: str,
) -> tuple[list[PackedImage], dict[str, Any]]:
    if not archive_path.is_file():
        raise SystemExit(f"missing public archive {archive_path}")
    Image.MAX_IMAGE_PIXELS = None
    spec = PUBLIC_SPECS[source_tag]
    with zipfile.ZipFile(archive_path) as archive:
        _yaml_name, yaml_text = _zip_yaml(archive)
        classes, metadata = _validate_public_contract(
            archive_path, source_tag, yaml_text
        )
        names = set(archive.namelist())
        image_members = sorted(
            name
            for name in names
            if Path(name).suffix.lower() in IMAGE_SUFFIXES and "/images/" in name
        )
        by_scene: dict[str, list[PublicCandidate]] = collections.defaultdict(list)
        dropped_degenerate = 0
        missing_labels = 0
        raw_variant_classes: collections.Counter[str] = collections.Counter()
        for image_member in image_members:
            label_member = _label_member(image_member, names)
            if label_member is None:
                missing_labels += 1
                continue
            candidate, dropped = _public_candidate(
                archive,
                image_member,
                label_member,
                classes,
                source_tag,
            )
            dropped_degenerate += dropped
            raw_variant_classes.update(candidate.original_classes)
            by_scene[candidate.scene].append(candidate)

        quarantined: dict[str, dict[str, int]] = {}
        selected: list[PublicCandidate] = []
        selected_classes: collections.Counter[str] = collections.Counter()
        variants_discarded = 0
        for scene, variants in sorted(by_scene.items()):
            ambiguous_names = PUBLIC_SPECS[source_tag].get(
                "ambiguous_classes", frozenset()
            )
            if ambiguous_names:
                ambiguous = collections.Counter()
                for candidate in variants:
                    for name in ambiguous_names:
                        ambiguous[name] += candidate.original_classes.get(name, 0)
                ambiguous = +ambiguous
                if ambiguous:
                    quarantined[scene] = dict(sorted(ambiguous.items()))
                    variants_discarded += len(variants)
                    continue
            representative = sorted(
                variants,
                key=lambda item: (
                    -len(item.boxes),
                    -(item.width * item.height),
                    item.image_member,
                ),
            )[0]
            selected.append(representative)
            selected_classes.update(representative.original_classes)
            variants_discarded += len(variants) - 1

        packed: list[PackedImage] = []
        for candidate in selected:
            blob = archive.read(candidate.image_member)
            width, height, pixel_hash = _pixel_identity_from_bytes(blob)
            if (width, height) != (candidate.width, candidate.height):
                raise SystemExit(
                    f"{archive_path}:{candidate.image_member}: image changed while reading"
                )
            packed.append(
                PackedImage(
                    source_tag=source_tag,
                    source_scene=candidate.scene,
                    group_id=candidate.group_id,
                    boxes=candidate.boxes,
                    width=width,
                    height=height,
                    original_name=Path(candidate.image_member).name,
                    pixel_sha256=pixel_hash,
                    archive_path=archive_path,
                    archive_member=candidate.image_member,
                )
            )

    dropped_non_body = {
        name: selected_classes.get(name, 0)
        for name in sorted(spec.get("non_body_classes", frozenset()))
        if selected_classes.get(name, 0)
    }
    report = {
        "tag": source_tag,
        "archive": _portable_path(archive_path),
        "archive_sha256": _sha256_file(archive_path),
        "workspace": metadata["workspace"],
        "project": metadata["project"],
        "version": int(metadata["version"]),
        "declared_license": metadata["license"],
        "url": str(spec["url"]),
        "provenance_caveat": str(spec["provenance_caveat"]),
        "class_contract": list(classes),
        "image_variants": len(image_members),
        "source_scenes": len(by_scene),
        "selected_scenes_before_exact_dedup": len(packed),
        "variants_discarded_or_quarantined": variants_discarded,
        "missing_label_files": missing_labels,
        "quarantined_scenes": len(quarantined),
        "quarantined_by_ambiguous_class": dict(
            collections.Counter(
                name
                for values in quarantined.values()
                for name in values
            )
        ),
        "quarantined_scene_ids": sorted(quarantined),
        "raw_variant_boxes_by_class": dict(raw_variant_classes.most_common()),
        "selected_boxes_by_original_class": dict(selected_classes.most_common()),
        "dropped_non_body_boxes": dropped_non_body,
        "dropped_degenerate_boxes": dropped_degenerate,
        "component_boxes_after_mapping": sum(len(item.boxes) for item in packed),
        "dedupe_policy": (
            "group by pre-.rf. source stem; quarantine ambiguous scenes; choose "
            "most component-body boxes, then greatest pixel area, then member name"
        ),
    }
    return packed, report


def _dedupe_public_exact(
    sources: Sequence[PackedImage],
) -> tuple[list[PackedImage], list[dict[str, Any]]]:
    by_hash: dict[str, list[PackedImage]] = collections.defaultdict(list)
    for image in sources:
        by_hash[image.pixel_sha256].append(image)
    kept: list[PackedImage] = []
    duplicates: list[dict[str, Any]] = []
    for pixel_hash, images in sorted(by_hash.items()):
        ordered = sorted(
            images,
            # Exact public copies can carry different annotation completeness.
            # Prefer the copy with more body boxes; only use provenance/name as
            # deterministic tie breakers.  Unlike two human-verified local
            # copies this is not a conflict: community exports commonly omit a
            # subset of labels in one downstream version.
            key=lambda item: (
                -len(item.boxes),
                item.source_tag,
                item.source_scene,
                item.original_name,
            ),
        )
        kept.append(ordered[0])
        if len(ordered) > 1:
            duplicates.append(
                {
                    "pixel_sha256": pixel_hash,
                    "kept": f"{ordered[0].source_tag}:{ordered[0].source_scene}",
                    "kept_boxes": len(ordered[0].boxes),
                    "dropped": [
                        {
                            "source": f"{item.source_tag}:{item.source_scene}",
                            "boxes": len(item.boxes),
                        }
                        for item in ordered[1:]
                    ],
                }
            )
    return kept, duplicates


def _reference_member(root: Path, relative: str, *, field: str) -> Path:
    """Resolve a reference manifest member without allowing path traversal."""

    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"reference {field} escapes its dataset root: {relative!r}") from exc
    if not candidate.is_file():
        raise SystemExit(f"missing reference {field}: {candidate}")
    return candidate


def _recording_id(value: str) -> str | None:
    match = re.search(
        r"(?:^|__)rec(?P<recording>[0-9]+)(?:__|_|$)", source_scene(value), re.I
    )
    return f"rec{int(match.group('recording'))}" if match else None


def _intersection_over_smaller(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    denominator = min(first_area, second_area)
    return intersection / denominator if denominator > 0 else 0.0


def _area_ratio(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    smaller = min(first_area, second_area)
    return max(first_area, second_area) / smaller if smaller > 0 else math.inf


def _audit_pcb_dslr_ic_completeness(
    local: Sequence[PackedImage],
    reference_root: Path,
    tile_manifest_path: Path,
    *,
    ios_threshold: float = 0.75,
    max_area_ratio: float = 4.0,
) -> dict[str, Any]:
    """Check reviewed boxes against trusted IC-only upstream annotations.

    This is a completeness gate, never a label importer.  Only upstream IC
    rectangles fully contained by a reviewed tile are considered.  A smaller
    body-only human box may sit inside the upstream rotated-rectangle AABB, so
    matching uses intersection-over-smaller plus an area-ratio guard instead
    of IoU.  Every decoded tile must equal the corresponding crop of the
    manifest-pinned reference image before coordinates are trusted.
    """

    root = reference_root.expanduser().resolve()
    source_manifest_path = root / "manifest.json"
    if not source_manifest_path.is_file():
        raise SystemExit(f"missing PCB-DSLR reference manifest {source_manifest_path}")
    tile_manifest_path = tile_manifest_path.expanduser().resolve()
    if not tile_manifest_path.is_file():
        raise SystemExit(f"missing tile manifest {tile_manifest_path}")
    try:
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        tile_rows = json.loads(tile_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read PCB-DSLR audit manifests: {exc}") from exc
    if source_manifest.get("schema_version") != "aoi-reference-source-set/2.0":
        raise SystemExit(
            f"{source_manifest_path}: unexpected schema "
            f"{source_manifest.get('schema_version')!r}"
        )
    if not isinstance(source_manifest.get("files"), list) or not isinstance(
        tile_rows, list
    ):
        raise SystemExit("PCB-DSLR audit manifests have invalid record lists")
    references = {
        canonical_board_id(str(record.get("board_id", "")), LOCAL_TAG): record
        for record in source_manifest["files"]
        if isinstance(record, dict) and record.get("board_id")
    }
    tiles = {
        str(record.get("file", "")): record
        for record in tile_rows
        if isinstance(record, dict) and record.get("file")
    }

    # The parser is already the source of truth used to build these references.
    # Import lazily so ordinary public-dataset packing remains a light PIL-only
    # command and does not load OpenCV unless this optional audit is requested.
    from scripts.build_diverse_reference_bootstrap import (  # noqa: PLC0415
        BootstrapError,
        parse_upstream_ic_annotations,
    )

    per_tile: list[dict[str, Any]] = []
    match_scores: list[float] = []
    audited_instances = 0
    matched_instances = 0
    unique_instances: set[tuple[str, int]] = set()
    missing: list[dict[str, Any]] = []
    large = {
        threshold: {"audited": 0, "matched": 0, "missing": 0}
        for threshold in (150, 250, 350)
    }
    unassessed: list[dict[str, str]] = []

    by_group: dict[str, list[PackedImage]] = collections.defaultdict(list)
    for image in local:
        by_group[image.group_id].append(image)
    for group_id, group_images in sorted(by_group.items()):
        reference = references.get(group_id)
        if reference is None:
            for image in sorted(group_images, key=lambda item: item.original_name):
                reason = "no PCB-DSLR reference coverage for this board"
                unassessed.append({"crop": image.original_name, "reason": reason})
                per_tile.append(
                    {"crop": image.original_name, "status": "unassessed", "reason": reason}
                )
            continue
        expected_recording = str(reference.get("recording_id", ""))
        image_path = _reference_member(
            root, str(reference.get("image_path", "")), field="image_path"
        )
        annotation_path = _reference_member(
            root,
            str(reference.get("annotation_path", "")),
            field="annotation_path",
        )
        expected_image_sha = str(reference.get("image_sha256", ""))
        expected_annotation_sha = str(reference.get("annotation_sha256", ""))
        if _sha256_file(image_path) != expected_image_sha:
            raise SystemExit(f"PCB-DSLR reference image hash mismatch: {image_path}")
        if _sha256_file(annotation_path) != expected_annotation_sha:
            raise SystemExit(
                f"PCB-DSLR reference annotation hash mismatch: {annotation_path}"
            )
        try:
            with Image.open(image_path) as handle:
                board_rgb = handle.convert("RGB")
        except Exception as exc:
            raise SystemExit(f"cannot decode PCB-DSLR reference {image_path}: {exc}") from exc
        board_width, board_height = board_rgb.size
        if (board_width, board_height) != (
            int(reference.get("width", -1)),
            int(reference.get("height", -1)),
        ):
            raise SystemExit(f"PCB-DSLR reference dimensions mismatch: {image_path}")
        try:
            annotations = parse_upstream_ic_annotations(
                annotation_path.read_text(encoding="utf-8"),
                image_width=board_width,
                image_height=board_height,
            )
        except (OSError, BootstrapError) as exc:
            raise SystemExit(f"cannot parse {annotation_path}: {exc}") from exc

        for image in sorted(group_images, key=lambda item: item.original_name):
            tile = tiles.get(image.original_name)
            if tile is None:
                reason = "crop is absent from the supplied tile manifest"
                unassessed.append({"crop": image.original_name, "reason": reason})
                per_tile.append(
                    {"crop": image.original_name, "status": "unassessed", "reason": reason}
                )
                continue
            tile_group = canonical_board_id(str(tile.get("source", "")), LOCAL_TAG)
            tile_recording = _recording_id(str(tile.get("source", "")))
            if tile_group != group_id or tile_recording != expected_recording:
                raise SystemExit(
                    f"{image.original_name}: tile source identity does not match "
                    f"reference {group_id}/{expected_recording}"
                )
            try:
                tile_x, tile_y = int(tile["x"]), int(tile["y"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit(f"{image.original_name}: invalid tile coordinates") from exc
            tile_right, tile_bottom = tile_x + image.width, tile_y + image.height
            if (
                tile_x < 0
                or tile_y < 0
                or tile_right > board_width
                or tile_bottom > board_height
            ):
                raise SystemExit(f"{image.original_name}: tile lies outside reference image")
            reference_crop = board_rgb.crop((tile_x, tile_y, tile_right, tile_bottom))
            if _pixel_sha256(reference_crop) != image.pixel_sha256:
                raise SystemExit(
                    f"{image.original_name}: tile pixels do not match the pinned "
                    "PCB-DSLR reference crop"
                )

            contained: list[tuple[int, tuple[float, float, float, float], float]] = []
            for annotation_index, annotation in enumerate(annotations):
                box = annotation.bbox
                if (
                    box.x1 < tile_x
                    or box.y1 < tile_y
                    or box.x2 > tile_right
                    or box.y2 > tile_bottom
                ):
                    continue
                local_box = (
                    float(box.x1 - tile_x),
                    float(box.y1 - tile_y),
                    float(box.x2 - tile_x),
                    float(box.y2 - tile_y),
                )
                short_side = min(local_box[2] - local_box[0], local_box[3] - local_box[1])
                contained.append((annotation_index, local_box, short_side))
                unique_instances.add((group_id, annotation_index))
            human = [
                (
                    (box.cx - box.width / 2.0) * image.width,
                    (box.cy - box.height / 2.0) * image.height,
                    (box.cx + box.width / 2.0) * image.width,
                    (box.cy + box.height / 2.0) * image.height,
                )
                for box in image.boxes
            ]
            candidates: list[tuple[float, float, int, int]] = []
            for upstream_index, (_annotation_index, upstream_box, _short) in enumerate(
                contained
            ):
                for human_index, human_box in enumerate(human):
                    score = _intersection_over_smaller(upstream_box, human_box)
                    ratio = _area_ratio(upstream_box, human_box)
                    if score >= ios_threshold and ratio <= max_area_ratio:
                        candidates.append((score, ratio, upstream_index, human_index))
            matched_upstream: dict[int, float] = {}
            matched_human: set[int] = set()
            for score, _ratio, upstream_index, human_index in sorted(
                candidates, key=lambda item: (-item[0], item[1], item[2], item[3])
            ):
                if upstream_index in matched_upstream or human_index in matched_human:
                    continue
                matched_upstream[upstream_index] = score
                matched_human.add(human_index)

            audited_instances += len(contained)
            matched_instances += len(matched_upstream)
            match_scores.extend(matched_upstream.values())
            tile_missing = 0
            for upstream_index, (annotation_index, upstream_box, short_side) in enumerate(
                contained
            ):
                is_matched = upstream_index in matched_upstream
                for threshold, counts in large.items():
                    if short_side > threshold:
                        counts["audited"] += 1
                        counts["matched" if is_matched else "missing"] += 1
                if not is_matched:
                    tile_missing += 1
                    missing.append(
                        {
                            "crop": image.original_name,
                            "board_group": group_id,
                            "annotation_index": annotation_index,
                            "bbox_xyxy": [round(value, 3) for value in upstream_box],
                            "short_side_px": round(short_side, 3),
                        }
                    )
            per_tile.append(
                {
                    "crop": image.original_name,
                    "status": "pass" if tile_missing == 0 else "missing",
                    "audited_ic_instances": len(contained),
                    "matched": len(matched_upstream),
                    "missing": tile_missing,
                }
            )

    audited_tiles = sum(item["status"] != "unassessed" for item in per_tile)
    return {
        "mode": "read_only_completeness_audit; upstream boxes are never imported",
        "reference_root": _portable_path(root),
        "reference_manifest_sha256": _sha256_file(source_manifest_path),
        "tile_manifest": _portable_path(tile_manifest_path),
        "tile_manifest_sha256": _sha256_file(tile_manifest_path),
        "source_annotation_scope": "IC only",
        "match_policy": {
            "metric": "intersection_over_smaller",
            "minimum": ios_threshold,
            "maximum_area_ratio": max_area_ratio,
            "one_to_one": True,
            "fully_contained_upstream_boxes_only": True,
        },
        "reviewed_tiles": len(local),
        "audited_tiles": audited_tiles,
        "unassessed_tiles": len(unassessed),
        "unassessed": unassessed,
        "audited_ic_instances": audited_instances,
        "unique_physical_ic_instances": len(unique_instances),
        "matched": matched_instances,
        "missing": audited_instances - matched_instances,
        "available_coverage_pass": audited_tiles > 0 and audited_instances == matched_instances,
        "match_ios": {
            "minimum": min(match_scores) if match_scores else None,
            "median": statistics.median(match_scores) if match_scores else None,
            "maximum": max(match_scores) if match_scores else None,
        },
        "large_ic_instances": {f"short_side_gt_{key}": value for key, value in large.items()},
        "missing_instances": missing,
        "per_tile": per_tile,
    }


def _validation_fixture_hashes() -> set[str]:
    """Vân tay pixel của các board dùng làm cổng nghiệm thu trên board thật.

    Không phải nhãn, nên chúng không nằm trong split nào -- và vì thế phép khử
    trùng theo board group không nhìn thấy chúng.
    """

    root = PROJECT_ROOT / "tests" / "data" / "solder_geometry"
    hashes: set[str] = set()
    if not root.is_dir():
        return hashes
    for path in sorted(root.glob("*.png")):
        try:
            _w, _h, digest = _pixel_identity(path)
        except SystemExit:
            continue
        hashes.add(digest)
    return hashes


def build_plan(
    crop_root: Path,
    boxes_path: Path,
    rf100_path: Path,
    winnies_path: Path,
    *,
    consolidated_path: Path | None = None,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    min_target_groups: int,
    audit_only: bool,
    pcb_dslr_reference_root: Path | None = None,
    tile_manifest_path: Path | None = None,
    require_ic_audit_pass: bool = False,
) -> PackPlan:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise SystemExit("val/test ratios must be non-negative and sum to less than 1")
    local, local_report = _load_local(crop_root, boxes_path)
    if not local:
        raise SystemExit("no crop reached status 'verified'; nothing to audit or pack")
    if require_ic_audit_pass and pcb_dslr_reference_root is None:
        raise SystemExit(
            "--require-ic-audit-pass needs --pcb-dslr-reference-root"
        )
    if pcb_dslr_reference_root is not None:
        effective_tile_manifest = tile_manifest_path or (
            PROJECT_ROOT / "datasets" / "test_images" / "tiles_1024" / "tiles_manifest.json"
        )
        ic_audit = _audit_pcb_dslr_ic_completeness(
            local,
            pcb_dslr_reference_root,
            effective_tile_manifest,
        )
        if require_ic_audit_pass and not ic_audit["available_coverage_pass"]:
            raise SystemExit(
                "PCB-DSLR IC completeness audit failed: "
                f"{ic_audit['missing']} missing across "
                f"{ic_audit['audited_ic_instances']} auditable instances"
            )
    else:
        ic_audit = {
            "mode": "not_requested",
            "note": (
                "Pass --pcb-dslr-reference-root to compare verified tiles with "
                "the official IC-only annotations without importing them."
            ),
        }
    rf100, rf100_report = _load_public_archive(rf100_path, RF100_TAG)
    winnies, winnies_report = _load_public_archive(winnies_path, WINNIES_TAG)
    public_reports = [rf100_report, winnies_report]
    consolidated: list[PackedImage] = []
    if consolidated_path is not None:
        consolidated, consolidated_report = _load_public_archive(
            consolidated_path, CONSOLIDATED_TAG
        )
        public_reports.append(consolidated_report)
    # Ảnh công khai trùng PIXEL với fixture nghiệm thu phải bị loại. Đo được:
    # `components_data_uncropped/train/images/00001__1024__1648___4120.png` của
    # Consolidated trùng từng pixel với `tests/data/solder_geometry/
    # board_smd_00001.png` -- board dùng để chấm 28 pad của bước 5.5. Train lên
    # chính ảnh dùng để chấm thì con số nghiệm thu không còn nghĩa gì.
    fixture_hashes = _validation_fixture_hashes()
    fixture_leaks = [
        {"tag": item.source_tag, "scene": item.source_scene, "pixel_sha256": item.pixel_sha256}
        for item in [*rf100, *winnies, *consolidated]
        if item.pixel_sha256 in fixture_hashes
    ]
    leaked = {entry["pixel_sha256"] for entry in fixture_leaks}
    keep = [
        item for item in [*rf100, *winnies, *consolidated]
        if item.pixel_sha256 not in leaked
    ]
    public, public_exact_duplicates = _dedupe_public_exact(keep)

    target_groups = sorted({item.group_id for item in local})
    split_of_group = assign_splits(
        target_groups,
        (1.0 - val_ratio - test_ratio, val_ratio, test_ratio),
        seed,
    )
    target_groups_by_split = {
        split: sorted(
            group for group, assigned in split_of_group.items() if assigned == split
        )
        for split in SPLITS
    }
    missing_target_splits = [
        split for split in SPLITS if not target_groups_by_split[split]
    ]
    ready = (
        len(target_groups) >= min_target_groups and not missing_target_splits
    )
    #: "thiếu bucket valid" một mình là chẩn đoán chứ không phải việc làm được:
    #: bucket là hàm băm của board id, nên người duyệt không thể đoán board nào
    #: rơi vào đó. Chạy chính phép gán ấy trên MỌI board ứng viên rồi nêu tên
    #: board cần duyệt — cùng một seed, cùng một công thức, nên lời khuyên này
    #: đúng theo định nghĩa chứ không phải phỏng đoán.
    candidate_groups: dict[str, dict[str, int]] = local_report.get(
        "candidate_groups", {}
    )
    split_of_candidate = assign_splits(
        sorted(candidate_groups),
        (1.0 - val_ratio - test_ratio, val_ratio, test_ratio),
        seed,
    )
    boards_that_would_fill = {
        split: [
            {
                "group_id": group,
                "tiles": candidate_groups[group]["tiles"],
                "verified_tiles": candidate_groups[group]["verified_tiles"],
            }
            for group in sorted(candidate_groups)
            if split_of_candidate.get(group) == split
            and candidate_groups[group]["verified_tiles"] == 0
        ]
        for split in missing_target_splits
    }
    if not audit_only and not ready:
        if len(target_groups) < min_target_groups:
            reason = (
                f"only {len(target_groups)} verified target board groups; need at least "
                f"{min_target_groups}"
            )
        else:
            reason = (
                "stable hash assignment has no target board in split(s): "
                + ", ".join(missing_target_splits)
            )
        raise SystemExit(
            f"{reason} before creating stable train/valid/test. "
            "Keep labelling or use --audit-only."
        )

    for image in local:
        image.split = split_of_group[image.group_id]
    excluded_public: collections.Counter[str] = collections.Counter()
    included_public: list[PackedImage] = []
    for image in public:
        target_split = split_of_group.get(image.group_id)
        if target_split in {"valid", "test"}:
            excluded_public[target_split] += 1
            continue
        image.split = "train"
        included_public.append(image)

    group_sets = {
        split: {
            image.group_id
            for image in [*local, *included_public]
            if image.split == split
        }
        for split in SPLITS
    }
    group_intersections = {
        f"{left}_{right}": sorted(group_sets[left] & group_sets[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    if any(group_intersections.values()):
        raise SystemExit(f"group leakage remains after split: {group_intersections}")

    pixel_sets = {
        split: {
            image.pixel_sha256
            for image in [*local, *included_public]
            if image.split == split
        }
        for split in SPLITS
    }
    pixel_intersections = {
        f"{left}_{right}": sorted(pixel_sets[left] & pixel_sets[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    if any(pixel_intersections.values()):
        raise SystemExit(f"exact-pixel leakage remains after split: {pixel_intersections}")

    all_images = [*local, *included_public]
    report = {
        "schema_version": "aoi-component-detection-pack/1.0",
        "class_names": ["component"],
        "box_convention": "visible component body only; exclude leads, pads and test points",
        "local": local_report,
        "pcb_dslr_ic_completeness": ic_audit,
        "public_sources": public_reports,
        "validation_fixture_leaks": fixture_leaks,
        "public_exact_duplicate_groups": public_exact_duplicates,
        "split_policy": {
            "unit": "canonical physical board",
            "assignment": "sha256(seed:canonical_group) stable threshold buckets",
            "stable_when_new_groups_are_added": True,
            "target_domain": "whole reviewed board groups split into train/valid/test",
            "public_domain": "train only",
            "public_overlap_with_target_holdout": "excluded from the pack",
            "seed": seed,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "minimum_target_groups": min_target_groups,
        },
        "readiness": {
            "ready_to_pack": ready,
            "verified_target_groups": len(target_groups),
            "minimum_target_groups": min_target_groups,
            "missing_target_splits": missing_target_splits,
            "boards_that_would_fill": boards_that_would_fill,
        },
        "target_group_split": target_groups_by_split,
        "public_overlap_excluded": dict(sorted(excluded_public.items())),
        "images_per_split": {
            split: sum(item.split == split for item in all_images) for split in SPLITS
        },
        "boxes_per_split": {
            split: sum(len(item.boxes) for item in all_images if item.split == split)
            for split in SPLITS
        },
        "source_images_included": dict(
            collections.Counter(item.source_tag for item in all_images)
        ),
        "leakage_audit": {
            "group_intersections": group_intersections,
            "pixel_sha256_intersections": pixel_intersections,
        },
    }
    return PackPlan(local, included_public, split_of_group, report)


def _safe_output_stem(image: PackedImage, index: int) -> str:
    scene = SAFE_NAME.sub("_", image.source_scene).strip("._") or "scene"
    original = SAFE_NAME.sub("_", Path(image.original_name).stem).strip("._") or "image"
    return f"{image.source_tag}__{scene}__{index:04d}__{original}"


def write_pack(plan: PackPlan, output: Path) -> Path:
    destination = output.expanduser().resolve()
    if destination.exists():
        raise SystemExit(f"output already exists; refusing to overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        for split in SPLITS:
            (staging / split / "images").mkdir(parents=True)
            (staging / split / "labels").mkdir(parents=True)
        ordered = sorted(
            [*plan.local, *plan.public],
            key=lambda item: (
                item.split,
                item.source_tag,
                item.group_id,
                item.source_scene,
                item.original_name,
            ),
        )
        written_names: set[str] = set()
        for index, image in enumerate(ordered):
            stem = _safe_output_stem(image, index)
            if stem in written_names:
                raise RuntimeError(f"generated duplicate output name {stem}")
            written_names.add(stem)
            suffix = Path(image.original_name).suffix.lower()
            if suffix not in IMAGE_SUFFIXES:
                suffix = ".png"
            image_path = staging / image.split / "images" / f"{stem}{suffix}"
            label_path = staging / image.split / "labels" / f"{stem}.txt"
            image_path.write_bytes(image.read_bytes())
            label_path.write_text(
                "\n".join(box.line() for box in image.boxes)
                + ("\n" if image.boxes else ""),
                encoding="utf-8",
            )
        (staging / "data.yaml").write_text(
            "train: train/images\n"
            "val: valid/images\n"
            "test: test/images\n"
            "nc: 1\n"
            "names: ['component']\n",
            encoding="utf-8",
        )
        (staging / "pack_manifest.json").write_text(
            json.dumps(plan.report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def _print_summary(plan: PackPlan, *, audit_only: bool) -> None:
    local = plan.report["local"]
    readiness = plan.report["readiness"]
    prefix = "audit" if audit_only else "pack"
    print(
        f"{prefix}: {local['verified_after_exact_dedup']} verified tile, "
        f"{local['verified_boxes']} box, "
        f"{readiness['verified_target_groups']} target board group"
    )
    print(
        "  trạng thái export: "
        + ", ".join(
            f"{name}={count}" for name, count in local["status_counts"].items()
        )
        + f", chưa export={local['unreviewed_not_in_export']}"
    )
    #: Trình duyệt đặt tên mọi lần xuất là `joint_boxes (N).json`, nên chỉ một
    #: chữ số sai là audit đúng định dạng nhưng sai checkpoint — im lặng, vì
    #: file cũ vẫn hợp lệ. In ngày xuất và vân tay để cái nhầm ấy lộ ra ngay.
    print(
        f"  checkpoint: {Path(local['boxes_file']).name}"
        f" · xuất lúc {local['exported_at'] or 'không ghi'}"
        f" · sha256 {local['boxes_sha256'][:12]}"
    )
    for source in plan.report["public_sources"]:
        print(
            f"  {source['tag']}: {source['image_variants']} variant -> "
            f"{source['selected_scenes_before_exact_dedup']} scene, "
            f"{source['component_boxes_after_mapping']} component box, "
            f"quarantine {source['quarantined_scenes']} scene"
        )
    leaks = plan.report.get("validation_fixture_leaks") or []
    if leaks:
        print(
            f"  loại {len(leaks)} ảnh công khai trùng pixel với fixture nghiệm thu: "
            + ", ".join(sorted({str(entry["scene"]) for entry in leaks})[:3])
        )
    ic_audit = plan.report["pcb_dslr_ic_completeness"]
    if ic_audit.get("mode") != "not_requested":
        print(
            "  PCB-DSLR IC audit: "
            f"{ic_audit['matched']}/{ic_audit['audited_ic_instances']} khớp, "
            f"{ic_audit['unassessed_tiles']} tile chưa có coverage"
        )
    for split in SPLITS:
        print(
            f"  {split:<5}: {plan.report['images_per_split'][split]:>4} ảnh, "
            f"{plan.report['boxes_per_split'][split]:>6} box"
        )
    if not readiness["ready_to_pack"]:
        detail = (
            f"{readiness['verified_target_groups']}/"
            f"{readiness['minimum_target_groups']} board đích"
        )
        if readiness["missing_target_splits"]:
            detail += ", thiếu bucket " + ", ".join(
                readiness["missing_target_splits"]
            )
        print(f"  CHƯA ĐỦ để khóa benchmark: {detail}")
        for split, boards in readiness.get("boards_that_would_fill", {}).items():
            if not boards:
                print(
                    f"    bucket {split}: KHÔNG board ứng viên nào rơi vào đây — "
                    "cần thêm tile của board mới vào crop set, duyệt thêm không đủ"
                )
                continue
            named = ", ".join(
                f"{board['group_id']} ({board['tiles']} tile)" for board in boards[:6]
            )
            more = f" và {len(boards) - 6} board nữa" if len(boards) > 6 else ""
            print(f"    duyệt 1 tile của board sau để lấp bucket {split}: {named}{more}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("crop_dir", type=Path)
    parser.add_argument("--boxes", type=Path, required=True)
    parser.add_argument(
        "--rf100",
        type=Path,
        default=PROJECT_ROOT
        / "datasets"
        / "public"
        / "fpic_boards_rf100"
        / "export_yolov8_v4.zip",
        help="RF100 printed-circuit-board v4 zip (legacy local folder name is accepted)",
    )
    parser.add_argument(
        "--winnies",
        type=Path,
        default=PROJECT_ROOT
        / "datasets"
        / "public"
        / "pcb_packages_winnies"
        / "export_yolov8_v3.zip",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--audit-report",
        type=Path,
        help="optional JSON report; refuses to overwrite an existing file",
    )
    parser.add_argument(
        "--pcb-dslr-reference-root",
        type=Path,
        help=(
            "optional aoi-reference-source-set/2.0 root; official IC-only boxes "
            "are used for a read-only completeness audit, never imported"
        ),
    )
    parser.add_argument(
        "--tile-manifest",
        type=Path,
        help=(
            "tiles_manifest.json mapping reviewed crops to source coordinates "
            "(default: the project's datasets/test_images/tiles_1024 manifest)"
        ),
    )
    parser.add_argument(
        "--require-ic-audit-pass",
        action="store_true",
        help="refuse packing if any fully-contained auditable upstream IC is missing",
    )
    parser.add_argument(
        "--consolidated",
        type=Path,
        default=None,
        help="zip PCB Component Detection Consolidated (Kaggle). Không bắt buộc; "
        "bỏ qua thì pack chỉ dùng RF100 + Winnies.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--min-target-groups",
        type=int,
        default=10,
        help="minimum distinct reviewed physical boards required before writing a pack",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.audit_only and args.output is not None:
        raise SystemExit("--audit-only does not write --output")
    if not args.audit_only and args.output is None:
        raise SystemExit("--output is required unless --audit-only is used")
    if args.min_target_groups < 3:
        raise SystemExit("--min-target-groups must be at least 3 for train/valid/test")
    output_path = (
        None if args.output is None else args.output.expanduser().resolve()
    )
    if output_path is not None and output_path.exists():
        raise SystemExit(
            f"output already exists; refusing to overwrite: {output_path}"
        )

    plan = build_plan(
        args.crop_dir.expanduser().resolve(),
        args.boxes.expanduser().resolve(),
        args.rf100.expanduser().resolve(),
        args.winnies.expanduser().resolve(),
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        min_target_groups=args.min_target_groups,
        consolidated_path=args.consolidated,
        audit_only=args.audit_only,
        pcb_dslr_reference_root=(
            None
            if args.pcb_dslr_reference_root is None
            else args.pcb_dslr_reference_root.expanduser().resolve()
        ),
        tile_manifest_path=(
            None
            if args.tile_manifest is None
            else args.tile_manifest.expanduser().resolve()
        ),
        require_ic_audit_pass=args.require_ic_audit_pass,
    )
    if args.audit_report is not None:
        report_path = args.audit_report.expanduser().resolve()
        if report_path.exists():
            raise SystemExit(f"audit report already exists; refusing to overwrite: {report_path}")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(plan.report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if args.audit_only:
        _print_summary(plan, audit_only=True)
        return 0
    assert output_path is not None
    destination = write_pack(plan, output_path)
    _print_summary(plan, audit_only=False)
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
