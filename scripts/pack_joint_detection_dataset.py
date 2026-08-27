"""Turn labelled component crops into a YOLO detection set, optionally merged.

Reads the ``joint_boxes.json`` the labelling page exports, joins it back to the
crop manifest, and writes a YOLO detection dataset that trains beside
``datasets/public/roboflow_solder_leadjoints``.

Three rules are enforced here rather than left to the notebook, because by the
time the notebook applies them a Kaggle run has already been spent:

* **The split is by scene, never by crop.** Twenty crops cut from one board
  photograph share its lighting, its soldermask and its focus. Splitting them at
  random puts near-copies on both sides and reports a validation score that
  measures memorisation.
* **Only ``verified`` crops are written.** A crop nobody looked at also has no
  boxes, and YOLO reads an empty label file as *this image contains no defects* --
  a confident negative. Silence must not be able to impersonate a judgement.
* **Class names must match across sources.** A class only one source labels is a
  class whose every instance also carries that source's camera, so the detector
  can score well on it by recognising the camera. The merge refuses rather than
  quietly taking the union.

    python scripts/pack_joint_detection_dataset.py datasets/labelling/fpic_components \\
        --boxes ~/Downloads/joint_boxes.json \\
        --merge datasets/public/roboflow_solder_leadjoints \\
        --output datasets/train/solder_detect_v2
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

SPLITS = ("train", "valid", "test")
USABLE_STATUS = {"verified"}


def read_boxes(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema != "aoi-joint-boxes/1.0":
        raise SystemExit(f"{path}: unexpected schema {schema!r}")
    if payload.get("coordinate_space") != "crop_pixels_top_left_origin":
        raise SystemExit(f"{path}: unexpected coordinate space "
                         f"{payload.get('coordinate_space')!r}")
    return payload


def read_class_names(data_yaml: Path) -> list[str]:
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            inline = stripped.split("names:", 1)[1].strip()
            if inline.startswith("["):
                body = inline[1 : inline.rindex("]")]
                return [item.strip().strip("'\"") for item in body.split(",") if item.strip()]
    raise SystemExit(f"{data_yaml}: no inline names: list")


def assign_splits(scenes: list[str], ratios: tuple[float, float, float], seed: int
                  ) -> dict[str, str]:
    """Deal whole scenes into splits, deterministically for a given seed."""
    ordered = sorted(scenes)
    random.Random(seed).shuffle(ordered)
    n = len(ordered)
    n_valid = max(1, round(n * ratios[1])) if n > 2 else 0
    n_test = max(1, round(n * ratios[2])) if n > 2 else 0
    if n_valid + n_test >= n:
        n_valid, n_test = (1, 1) if n > 2 else (0, 0)
    out: dict[str, str] = {}
    for index, scene in enumerate(ordered):
        if index < n_test:
            out[scene] = "test"
        elif index < n_test + n_valid:
            out[scene] = "valid"
        else:
            out[scene] = "train"
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("crop_dir", type=Path)
    parser.add_argument("--boxes", type=Path, required=True,
                        help="joint_boxes.json exported by the labelling page")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merge", type=Path, nargs="*", default=[],
                        help="existing YOLO datasets to fold in; class names must match")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--min-clean-ratio", type=float, default=0.0,
                        help="fail if fewer than this share of written crops are clean negatives; "
                             "the point of this source is negatives, so an accidental "
                             "positives-only pack should not be packaged silently")
    args = parser.parse_args(argv)

    root = args.crop_dir.resolve()
    manifest_path, crops_dir = root / "manifest.csv", root / "crops"
    for path in (manifest_path, crops_dir, args.boxes):
        if not path.exists():
            raise SystemExit(f"missing {path}")

    payload = read_boxes(args.boxes)
    labelled = payload["crops"]
    classes: list[str] = list(payload["classes"])

    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest = {row["crop_path"]: row for row in csv.DictReader(handle)}

    unknown = sorted(set(labelled) - set(manifest))
    if unknown:
        raise SystemExit(
            f"{len(unknown)} labelled crops are not in {manifest_path.name}, first: {unknown[:3]}. "
            "The export belongs to a different crop set."
        )

    # every dataset folded in must speak the same class list, in the same order
    for merge_root in args.merge:
        names = read_class_names(Path(merge_root) / "data.yaml")
        if names != classes:
            raise SystemExit(
                f"{merge_root} has classes {names}, the labels have {classes}. "
                "Reconcile them before merging; taking the union would create a class "
                "that only one camera ever labels."
            )

    usable = {name: rec for name, rec in labelled.items()
              if rec.get("status") in USABLE_STATUS}
    if not usable:
        raise SystemExit("no crop reached status 'verified'; nothing to pack")

    scenes = sorted({manifest[name]["scene_id"] for name in usable})
    split_of = assign_splits(scenes, (1 - args.val_ratio - args.test_ratio,
                                      args.val_ratio, args.test_ratio), args.seed)

    out = args.output.resolve()
    for split in SPLITS:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    counts = {split: collections.Counter() for split in SPLITS}
    clean = {split: 0 for split in SPLITS}
    written = {split: 0 for split in SPLITS}
    dropped_boxes = 0

    for name, record in sorted(usable.items()):
        row = manifest[name]
        split = split_of[row["scene_id"]]
        width, height = int(row["crop_w"]), int(row["crop_h"])
        lines: list[str] = []
        for box in record.get("boxes", []):
            if box["cls"] not in classes:
                raise SystemExit(f"{name}: class {box['cls']!r} is not in {classes}")
            index = classes.index(box["cls"])
            # clamp before normalising: a box dragged past the edge is a real
            # thing a reviewer does, and YOLO rejects coordinates outside [0,1]
            x0 = max(0, min(width, box["x"]))
            y0 = max(0, min(height, box["y"]))
            x1 = max(0, min(width, box["x"] + box["w"]))
            y1 = max(0, min(height, box["y"] + box["h"]))
            if x1 - x0 < 2 or y1 - y0 < 2:
                dropped_boxes += 1
                continue
            lines.append(
                f"{index} {(x0 + x1) / 2 / width:.6f} {(y0 + y1) / 2 / height:.6f} "
                f"{(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}"
            )
            counts[split][box["cls"]] += 1

        stem = Path(name).stem
        shutil.copy2(crops_dir / name, out / split / "images" / name)
        (out / split / "labels" / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written[split] += 1
        if not lines:
            clean[split] += 1

    merged_images = 0
    for merge_root in args.merge:
        merge_root = Path(merge_root).resolve()
        for split in SPLITS:
            images = merge_root / split / "images"
            if not images.is_dir():
                continue
            for image in sorted(images.iterdir()):
                label = merge_root / split / "labels" / f"{image.stem}.txt"
                if not label.exists():
                    continue
                # prefix so two sources cannot collide on a stem, and so the
                # provenance of any single file stays readable on disk
                tag = f"{merge_root.name}__{image.name}"
                shutil.copy2(image, out / split / "images" / tag)
                shutil.copy2(label, out / split / "labels" / f"{Path(tag).stem}.txt")
                merged_images += 1
                for line in label.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if parts:
                        counts[split][classes[int(parts[0])]] += 1

    for split in SPLITS:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
    (out / "data.yaml").write_text(
        "train: ../train/images\nval: ../valid/images\ntest: ../test/images\n"
        f"nc: {len(classes)}\nnames: {classes!r}\n".replace("'", "'"),
        encoding="utf-8")

    total_written = sum(written.values())
    clean_ratio = sum(clean.values()) / total_written if total_written else 0.0
    provenance = {
        "schema_version": "aoi-joint-detection-pack/1.0",
        "crop_dir": str(root),
        "boxes_file": str(args.boxes),
        "boxes_sha256": hashlib.sha256(args.boxes.read_bytes()).hexdigest(),
        "reviewer_id": payload.get("reviewer_id", ""),
        "exported_at": payload.get("exported_at", ""),
        "classes": classes,
        "split_by": "scene_id",
        "seed": args.seed,
        "scenes": {split: sorted(s for s, v in split_of.items() if v == split)
                   for split in SPLITS},
        "crops_written": written,
        "clean_negatives": clean,
        "clean_ratio": round(clean_ratio, 4),
        "boxes_per_split": {split: dict(counts[split]) for split in SPLITS},
        "merged_from": [str(Path(m).resolve()) for m in args.merge],
        "merged_images": merged_images,
        "dropped_degenerate_boxes": dropped_boxes,
        "labelled_but_unused": {
            name: rec.get("status") for name, rec in labelled.items()
            if rec.get("status") not in USABLE_STATUS
        },
    }
    (out / "pack_manifest.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote {out}")
    for split in SPLITS:
        print(f"  {split:<6}{written[split]:>6} crop  ({clean[split]} sạch)  "
              f"box: {dict(counts[split])}")
    if merged_images:
        print(f"  ghép thêm {merged_images} ảnh từ {len(args.merge)} bộ")
    print(f"  ảnh nền (không lỗi): {clean_ratio:.0%} của phần gắn nhãn")
    if dropped_boxes:
        print(f"  bỏ {dropped_boxes} box nhỏ hơn 2 px sau khi cắt về biên")
    if clean_ratio < args.min_clean_ratio:
        raise SystemExit(
            f"chỉ {clean_ratio:.0%} là ảnh nền, dưới ngưỡng {args.min_clean_ratio:.0%}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
