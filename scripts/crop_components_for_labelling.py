"""Cut per-component crops out of whole-board datasets so joints can be labelled.

Step 6.2 works on a component crop, not on a whole board, so a public set is only
useful here if a component in it is big enough for a fillet to be *visible*. That
is an absolute-pixel question and it is the one this command enforces: a crop is
emitted only when the component's short side clears ``--min-short-side``. At the
default 48 px each of the two joints is roughly 24 px, which is the pad size
measured on this project's own board -- below it there are pixels showing that
solder exists but nothing showing whether the fillet is sound, and a label
guessed from that is worse than no label.

The measured cost of the gate on the two accepted sources:

    RF100 printed-circuit-board 134,047 boxes -> 12.1% pass (IC 68%, Connector 77%)
    pcb_packages_winnies 16,632 boxes -> 20.9% pass   (SOT/SOD packages 35-100%)

The RF100 archive currently lives under the legacy directory name
``fpic_boards_rf100``. It is not FPIC; keep the path only for compatibility.

Capacitors and resistors are the bulk of both sets and almost all fail, because a
0402 chip part at these optical scales is 12-17 px wide. That is a property of
the camera, not of the gate.

Crops carry a margin past the annotated body so the fillets, which sit outside
the component outline, are inside the picture. The margin is proportional per
axis, so a two-terminal part gets most of its padding along the axis its
terminals lie on.

    python scripts/crop_components_for_labelling.py datasets/public/fpic_boards_rf100 \
        --output datasets/labelling/rf100_components \
        --dataset-source rf100_printed_circuit_board

The output is a crops/ folder plus manifest.csv, which is what
``build_joint_box_app.py`` reads. No label is invented: ``label_status`` is
empty on every row and stays that way until a person fills it.
"""

from __future__ import annotations

import argparse
import collections
import csv
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

#: Classes that are not a soldered component and so carry no joint to grade.
#: ``Test Point`` is a bare exposed pad; ``Unknown Unlabeled`` is the RF100
#: export's marker for an instance the annotators declined to name.
DEFAULT_DENY = ("unknown unlabeled", "unknown", "test point", "pcb", "board", "text")

#: Roboflow appends ``.rf.<hash>`` to every augmented copy but keeps the original
#: stem in front of it, so the real scene survives the augmentation.
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]{32}", re.I)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass(frozen=True)
class Box:
    cls: str
    cx: float
    cy: float
    w: float
    h: float


def scene_of(name: str) -> str:
    """The source photograph behind a possibly-augmented file name."""
    stem = Path(name).name
    stem = RF_SUFFIX.sub("", stem)
    return Path(stem).stem


def read_class_names(text: str) -> list[str]:
    """Pull ``names:`` out of a Roboflow data.yaml without taking a YAML dependency."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("names:"):
            continue
        inline = stripped.split("names:", 1)[1].strip()
        if inline.startswith("["):
            body = inline[1 : inline.rindex("]")] if "]" in inline else inline[1:]
            return [item.strip().strip("'\"") for item in body.split(",") if item.strip()]
        # block form: subsequent "- name" lines
        names = []
        for follow in lines[index + 1 :]:
            item = follow.strip()
            if item.startswith("- "):
                names.append(item[2:].strip().strip("'\""))
            elif item and not item.startswith("#"):
                break
        return names
    return []


class Source:
    """A YOLO dataset held either as a directory or as an unextracted zip.

    Reading the zip in place matters: these exports are hundreds of megabytes and
    extracting them doubles that on disk for no gain, since every image is opened
    exactly once.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._zip: zipfile.ZipFile | None = None
        self._name_set: set[str] | None = None
        if path.is_dir():
            zips = sorted(path.glob("*.zip"))
            if zips and not (path / "data.yaml").exists():
                self._zip = zipfile.ZipFile(zips[0])
                self.label = path.name
            else:
                self.label = path.name
        elif path.suffix.lower() == ".zip":
            self._zip = zipfile.ZipFile(path)
            self.label = path.stem
        else:
            raise SystemExit(f"{path} is neither a directory nor a .zip")

    def _names(self) -> list[str]:
        if self._zip is not None:
            return self._zip.namelist()
        return [str(p.relative_to(self.path)).replace("\\", "/") for p in self.path.rglob("*")]

    def read(self, name: str) -> bytes:
        if self._zip is not None:
            return self._zip.read(name)
        return (self.path / name).read_bytes()

    def classes(self) -> list[str]:
        for name in self._names():
            if name.endswith("data.yaml"):
                return read_class_names(self.read(name).decode("utf-8", errors="replace"))
        return []

    def images(self) -> list[str]:
        return [n for n in self._names() if n.lower().endswith(IMAGE_SUFFIXES)]

    def label_for(self, image_name: str) -> str | None:
        if self._name_set is None:
            self._name_set = set(self._names())
        candidate = image_name.replace("/images/", "/labels/")
        candidate = str(Path(candidate).with_suffix(".txt")).replace("\\", "/")
        return candidate if candidate in self._name_set else None


def parse_labels(raw: str, classes: list[str]) -> list[Box]:
    boxes = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            index = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        name = classes[index] if 0 <= index < len(classes) else str(index)
        boxes.append(Box(name, cx, cy, w, h))
    return boxes


def crop_box(
    box: Box, width: int, height: int, margin: float, min_pad: int
) -> tuple[int, int, int, int]:
    """Body box grown by a proportional margin, clamped to the image."""
    bw, bh = box.w * width, box.h * height
    pad_x = max(margin * bw, min_pad)
    pad_y = max(margin * bh, min_pad)
    left = int(round((box.cx - box.w / 2) * width - pad_x))
    top = int(round((box.cy - box.h / 2) * height - pad_y))
    right = int(round((box.cx + box.w / 2) * width + pad_x))
    bottom = int(round((box.cy + box.h / 2) * height + pad_y))
    return (max(0, left), max(0, top), min(width, right), min(height, bottom))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="YOLO dataset directory or .zip")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--min-short-side",
        type=int,
        default=48,
        help="reject a component whose short side is below this, in source pixels "
        "(default 48: each joint is then about 24 px, the pad size on this project's board)",
    )
    parser.add_argument("--margin", type=float, default=0.30,
                        help="padding past the body box, as a fraction of each axis (default 0.30)")
    parser.add_argument("--min-pad", type=int, default=8)
    parser.add_argument("--max-per-class", type=int, default=0,
                        help="cap crops per class so one class cannot swamp the labelling queue "
                        "(0 = no cap)")
    parser.add_argument("--deny", nargs="*", default=list(DEFAULT_DENY),
                        help="class names to skip, case-insensitive")
    parser.add_argument("--allow", nargs="*", default=None,
                        help="if given, keep only these class names")
    parser.add_argument("--dataset-source", default="",
                        help="provenance tag written to every row (default: source folder name)")
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="count what would be written without decoding pixels or saving crops")
    args = parser.parse_args(argv)

    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None

    source = Source(args.source)
    classes = source.classes()
    if not classes:
        raise SystemExit(f"no data.yaml with names: found in {args.source}")
    deny = {d.strip().lower() for d in args.deny}
    allow = {a.strip().lower() for a in args.allow} if args.allow else None
    tag = args.dataset_source or source.label

    # One file per scene. The augmented copies are the same photograph, so
    # cropping all of them would hand the labeller the same component repeatedly
    # and then leak it across the train/val split.
    by_scene: dict[str, str] = {}
    for name in sorted(source.images()):
        by_scene.setdefault(scene_of(name), name)
    scenes = sorted(by_scene)
    if args.limit_scenes:
        scenes = scenes[: args.limit_scenes]

    out = args.output.resolve()
    crops_dir = out / "crops"
    if not args.dry_run:
        crops_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    per_class: collections.Counter[str] = collections.Counter()
    skipped_small: collections.Counter[str] = collections.Counter()
    skipped_class: collections.Counter[str] = collections.Counter()
    no_labels = 0

    for scene in scenes:
        image_name = by_scene[scene]
        label_name = source.label_for(image_name)
        if label_name is None:
            no_labels += 1
            continue
        boxes = parse_labels(source.read(label_name).decode("utf-8", errors="replace"), classes)
        if not boxes:
            continue

        blob: bytes | None = None
        image = None
        size: tuple[int, int] | None = None
        for order, box in enumerate(boxes):
            key = box.cls.strip().lower()
            if key in deny or (allow is not None and key not in allow):
                skipped_class[box.cls] += 1
                continue
            if size is None:
                blob = source.read(image_name)
                with Image.open(io.BytesIO(blob)) as handle:
                    size = handle.size
            width, height = size
            bw, bh = box.w * width, box.h * height
            if min(bw, bh) < args.min_short_side:
                skipped_small[box.cls] += 1
                continue
            if args.max_per_class and per_class[box.cls] >= args.max_per_class:
                continue

            left, top, right, bottom = crop_box(box, width, height, args.margin, args.min_pad)
            if right - left < 8 or bottom - top < 8:
                continue
            if image is None and not args.dry_run:
                with Image.open(io.BytesIO(blob)) as handle:
                    image = handle.convert("RGB")
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{scene}__{order:03d}__{box.cls}")
            crop_name = f"{safe}.png"
            if not args.dry_run:
                image.crop((left, top, right, bottom)).save(crops_dir / crop_name)

            rows.append({
                "crop_path": crop_name,
                "dataset_source": tag,
                "source_image": image_name,
                "scene_id": scene,
                "component_class": box.cls,
                "crop_w": right - left,
                "crop_h": bottom - top,
                # Where the annotated body sits inside the padded crop, so the
                # labelling app can show it without re-deriving the geometry.
                "body_x": int(round((box.cx - box.w / 2) * width)) - left,
                "body_y": int(round((box.cy - box.h / 2) * height)) - top,
                "body_w": int(round(bw)),
                "body_h": int(round(bh)),
                "roi_kind": "component",
                "label_status": "",
                "reviewer_id": "",
                "notes": "",
            })
            per_class[box.cls] += 1

    if args.dry_run:
        print(f"[dry-run] {len(rows)} crops from {len(scenes)} scenes")
    else:
        manifest = out / "manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else
                                    ["crop_path", "dataset_source", "source_image", "scene_id",
                                     "component_class", "crop_w", "crop_h", "body_x", "body_y",
                                     "body_w", "body_h", "roi_kind", "label_status",
                                     "reviewer_id", "notes"])
            writer.writeheader()
            writer.writerows(rows)
        (out / "provenance.json").write_text(json.dumps({
            "source": str(args.source),
            "source_sha256": hashlib.sha256(
                args.source.read_bytes()).hexdigest() if args.source.is_file() else None,
            "dataset_source": tag,
            "classes": classes,
            "scenes_available": len(by_scene),
            "scenes_used": len(scenes),
            "min_short_side_px": args.min_short_side,
            "margin": args.margin,
            "crops": len(rows),
            "per_class": dict(per_class.most_common()),
            "rejected_too_small": dict(skipped_small.most_common()),
            "rejected_by_class_filter": dict(skipped_class.most_common()),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {out}")
        print(f"  {len(rows)} crops from {len(scenes)} scenes -> {crops_dir}")

    total_small = sum(skipped_small.values())
    print(f"  bỏ vì nhỏ hơn {args.min_short_side} px: {total_small}")
    if no_labels:
        print(f"  {no_labels} ảnh không có file nhãn đi kèm")
    for name, count in per_class.most_common(20):
        print(f"    {name:<24}{count:>7}   (bỏ {skipped_small.get(name, 0)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
