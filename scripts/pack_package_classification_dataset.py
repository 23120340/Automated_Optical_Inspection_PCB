"""Pack reviewed seven-class package boxes into a board-grouped ImageFolder ZIP.

This is deliberately a post-labelling command.  It rejects draft sentinels,
unreviewed/missing crops, class-order drift, board leakage and a split missing
any class.  Package geometry can remove solder ROIs, so silently training on a
partial queue is worse than having no model.

Example::

    python scripts/pack_package_classification_dataset.py \
      datasets/labelling/component_bodies_round2_20260830/package_boxes.json
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aoi_pipeline.config import PACKAGE_CLASSES  # noqa: E402


SCHEMA = "aoi-package-imagefolder/1.0"
LABEL_SCHEMA = "aoi-joint-boxes/1.0"
COORDINATE_SPACE = "crop_pixels_top_left_origin"
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Sample:
    crop_path: str
    scene_id: str
    package_class: str
    box_index: int
    bbox: tuple[float, float, float, float]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: root must be an object")
    return payload


def _read_manifest(path: Path) -> dict[str, dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    if not rows or not all(row.get("crop_path") for row in rows):
        raise SystemExit(f"{path}: empty or missing crop_path")
    by_path = {str(row["crop_path"]): row for row in rows}
    if len(by_path) != len(rows):
        raise SystemExit(f"{path}: duplicate crop_path")
    return by_path


def _finite_box(box: Mapping[str, object], context: str) -> tuple[float, float, float, float]:
    values: list[float] = []
    for field in ("x", "y", "w", "h"):
        value = box.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise SystemExit(f"{context}: invalid {field}={value!r}")
        values.append(float(value))
    if values[2] <= 0 or values[3] <= 0:
        raise SystemExit(f"{context}: box width/height must be positive")
    return tuple(values)  # type: ignore[return-value]


def load_samples(label_path: Path, root: Path) -> tuple[list[Sample], dict[str, Any]]:
    payload = _read_json(label_path)
    expected = {
        "schema": LABEL_SCHEMA,
        "dataset": root.name,
        "coordinate_space": COORDINATE_SPACE,
        "classes": list(PACKAGE_CLASSES),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise SystemExit(
                f"{label_path}: {field} mismatch; expected {value!r}, "
                f"got {payload.get(field)!r}"
            )
    if payload.get("unknown_class") is not None:
        raise SystemExit(
            f"{label_path}: this is a draft with unknown_class; export the fully "
            "reviewed package page before packing"
        )
    records = payload.get("crops")
    if not isinstance(records, dict):
        raise SystemExit(f"{label_path}: crops must be an object")
    manifest = _read_manifest(root / "manifest.csv")
    first_crop = next(iter(manifest))
    expected_dataset_id = hashlib.sha256(
        f"{root.name}|{len(manifest)}|{first_crop}|{','.join(PACKAGE_CLASSES)}".encode()
    ).hexdigest()[:16]
    if payload.get("dataset_id") != expected_dataset_id:
        raise SystemExit(
            f"{label_path}: dataset_id mismatch; expected {expected_dataset_id!r}, "
            f"got {payload.get('dataset_id')!r}"
        )
    missing = sorted(set(manifest) - set(records))
    extra = sorted(set(records) - set(manifest))
    if missing or extra:
        raise SystemExit(
            "review checkpoint must account for every manifest crop; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )

    samples: list[Sample] = []
    dispositions = {"verified": 0, "skipped": 0, "unusable": 0}
    for crop_path, record in records.items():
        if not isinstance(record, Mapping):
            raise SystemExit(f"{crop_path}: record must be an object")
        status = record.get("status")
        if status not in dispositions:
            raise SystemExit(
                f"{crop_path}: status must be verified/skipped/unusable, got {status!r}"
            )
        dispositions[str(status)] += 1
        boxes = record.get("boxes", [])
        if not isinstance(boxes, list):
            raise SystemExit(f"{crop_path}: boxes must be a list")
        if status != "verified":
            if boxes:
                raise SystemExit(f"{crop_path}: {status} crop must not carry boxes")
            continue
        scene_id = str(manifest[crop_path].get("scene_id") or "").strip()
        if not scene_id:
            raise SystemExit(f"{crop_path}: manifest scene_id is required for board split")
        for index, box in enumerate(boxes):
            if not isinstance(box, Mapping):
                raise SystemExit(f"{crop_path} box {index}: must be an object")
            package_class = box.get("cls")
            if package_class not in PACKAGE_CLASSES:
                raise SystemExit(
                    f"{crop_path} box {index}: unresolved/invalid class {package_class!r}"
                )
            samples.append(Sample(
                crop_path=crop_path,
                scene_id=scene_id,
                package_class=str(package_class),
                box_index=index,
                bbox=_finite_box(box, f"{crop_path} box {index}"),
            ))
    if not samples:
        raise SystemExit("no verified package boxes to pack")
    counts = {name: sum(item.package_class == name for item in samples) for name in PACKAGE_CLASSES}
    absent = [name for name, count in counts.items() if not count]
    if absent:
        raise SystemExit(f"dataset has no reviewed sample for classes: {absent}")
    return samples, {
        "source_dataset_id": payload.get("dataset_id"),
        "reviewer_id": payload.get("reviewer_id"),
        "dispositions": dispositions,
        "class_counts": counts,
    }


def split_groups(
    samples: Sequence[Sample], *, seed: int, val_ratio: float, test_ratio: float,
) -> dict[str, str]:
    groups = sorted({item.scene_id for item in samples})
    if len(groups) < 3:
        raise SystemExit("need at least three source boards/scenes for train/val/test")
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(),
    )
    test_count = max(1, round(len(groups) * test_ratio))
    val_count = max(1, round(len(groups) * val_ratio))
    if test_count + val_count >= len(groups):
        test_count = val_count = 1
    assignments = {group: "test" for group in ordered[:test_count]}
    assignments.update({
        group: "val" for group in ordered[test_count:test_count + val_count]
    })
    assignments.update({
        group: "train" for group in ordered[test_count + val_count:]
    })
    return assignments


def _validate_split_coverage(
    samples: Sequence[Sample], assignments: Mapping[str, str],
) -> dict[str, dict[str, int]]:
    counts = {
        split: {name: 0 for name in PACKAGE_CLASSES}
        for split in SPLITS
    }
    for item in samples:
        counts[assignments[item.scene_id]][item.package_class] += 1
    missing = {
        split: [name for name, count in by_class.items() if count == 0]
        for split, by_class in counts.items()
    }
    missing = {split: names for split, names in missing.items() if names}
    if missing:
        raise SystemExit(
            "board-grouped split lacks package classes; label more independent "
            f"boards instead of leaking crops across splits: {missing}"
        )
    return counts


def _crop_image(image, box: tuple[float, float, float, float], padding_ratio: float):
    height, width = image.shape[:2]
    x, y, w, h = box
    pad_x, pad_y = w * padding_ratio, h * padding_ratio
    x1 = max(0, int(math.floor(x - pad_x)))
    y1 = max(0, int(math.floor(y - pad_y)))
    x2 = min(width, int(math.ceil(x + w + pad_x)))
    y2 = min(height, int(math.ceil(y + h + pad_y)))
    if x2 <= x1 or y2 <= y1:
        raise SystemExit(f"box is outside image bounds: {box} vs {width}x{height}")
    return image[y1:y2, x1:x2]


def pack(
    label_path: Path,
    output: Path,
    *,
    dataset_root: Path | None = None,
    seed: int = 52026,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    padding_ratio: float = 0.06,
) -> int:
    label_path = label_path.resolve()
    root = (dataset_root or label_path.parent).resolve()
    output = output.resolve()
    if output.exists():
        raise SystemExit(f"output already exists; refusing to overwrite: {output}")
    if not 0 <= padding_ratio <= 0.5:
        raise SystemExit("padding_ratio must be between 0 and 0.5")
    if not 0 < val_ratio < 0.5 or not 0 < test_ratio < 0.5:
        raise SystemExit("val/test ratios must be between 0 and 0.5")

    samples, source = load_samples(label_path, root)
    assignments = split_groups(
        samples, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio,
    )
    split_counts = _validate_split_coverage(samples, assignments)
    group_sets = {
        split: sorted(group for group, assigned in assignments.items() if assigned == split)
        for split in SPLITS
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "task": "component_package_classification",
        "class_names": list(PACKAGE_CLASSES),
        "input_semantics": "RGB body crop; runtime letterboxes to 128x128",
        "split_unit": "board_scene_id",
        "seed": seed,
        "source": source,
        "split_groups": group_sets,
        "split_class_counts": split_counts,
        "padding_ratio": padding_ratio,
        "samples": [],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    current_crop_path: str | None = None
    current_image: Any = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for ordinal, item in enumerate(samples):
                if item.crop_path != current_crop_path:
                    image_path = root / "crops" / item.crop_path
                    current_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    current_crop_path = item.crop_path
                    if current_image is None:
                        raise SystemExit(f"cannot decode crop image: {image_path}")
                patch = _crop_image(current_image, item.bbox, padding_ratio)
                ok, encoded = cv2.imencode(".png", patch)
                if not ok:
                    raise SystemExit(f"cannot encode {item.crop_path} box {item.box_index}")
                split = assignments[item.scene_id]
                stem = hashlib.sha256(
                    f"{item.crop_path}:{item.box_index}:{item.bbox}".encode()
                ).hexdigest()[:16]
                member = f"{split}/{item.package_class}/{ordinal:06d}_{stem}.png"
                archive.writestr(member, encoded.tobytes())
                manifest["samples"].append({
                    "path": member,
                    "source_crop": item.crop_path,
                    "scene_id": item.scene_id,
                    "box_index": item.box_index,
                    "bbox_xywh": list(item.bbox),
                    "class": item.package_class,
                    "split": split,
                })
            archive.writestr(
                "dataset_manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            )
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise SystemExit(f"output already exists; refusing to overwrite: {output}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    print(f"wrote {output}: {len(samples)} package crops")
    for split in SPLITS:
        total = sum(split_counts[split].values())
        print(f"  {split}: {total} samples / {len(group_sets[split])} boards")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=52026)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--padding-ratio", type=float, default=0.06)
    args = parser.parse_args(argv)
    output = args.output or args.labels.with_name("package_classification_board_split.zip")
    return pack(
        args.labels, output,
        dataset_root=args.dataset_root,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        padding_ratio=args.padding_ratio,
    )


if __name__ == "__main__":
    raise SystemExit(main())
