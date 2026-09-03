#!/usr/bin/env python3
"""Build and merge contact sheets for a model-free package-label review.

The ``sheets`` command only crops, resizes, and annotates source pixels.  It does
not load or run a detector/classifier.  Reviewers write one compact token per
numbered cell, then ``merge`` applies those visual decisions to a copy of the
package checkpoint while preserving every box coordinate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import OrderedDict
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


PACKAGE_CLASSES = (
    "hai_chan",
    "tru_dung",
    "goi_nho",
    "ic_hai_ben",
    "ic_bon_ben",
    "ic_khong_chan",
    "connector",
)
TOKEN_TO_CLASS = {str(index): name for index, name in enumerate(PACKAGE_CLASSES, 1)}
UNKNOWN_TOKENS = {"u", "unknown"}


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scene_crops(root: Path, checkpoint: dict[str, Any]) -> OrderedDict[str, list[str]]:
    by_scene: OrderedDict[str, list[str]] = OrderedDict()
    with (root / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            crop_path = row["crop_path"]
            if crop_path not in checkpoint["crops"]:
                continue
            by_scene.setdefault(row["scene_id"], []).append(crop_path)

    missing = set(checkpoint["crops"]) - {
        crop_path for crop_paths in by_scene.values() for crop_path in crop_paths
    }
    if missing:
        raise ValueError(f"manifest.csv is missing {len(missing)} checkpoint crops")
    return by_scene


def _balanced_assignments(
    scenes: OrderedDict[str, list[str]], checkpoint: dict[str, Any], part_count: int
) -> list[list[str]]:
    weighted = []
    for order, (scene_id, crop_paths) in enumerate(scenes.items()):
        count = sum(len(checkpoint["crops"][path]["boxes"]) for path in crop_paths)
        weighted.append((scene_id, count, order))

    parts: list[list[str]] = [[] for _ in range(part_count)]
    totals = [0] * part_count
    for scene_id, count, _ in sorted(weighted, key=lambda item: (-item[1], item[2])):
        target = min(range(part_count), key=lambda index: (totals[index], index))
        parts[target].append(scene_id)
        totals[target] += count

    scene_order = {scene_id: order for order, scene_id in enumerate(scenes)}
    for part in parts:
        part.sort(key=scene_order.__getitem__)
    return parts


def _fit_context(
    image: np.ndarray,
    box: dict[str, Any],
    output_width: int,
    output_height: int,
    context_scale: float,
) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    x, y, width, height = (int(box[key]) for key in ("x", "y", "w", "h"))
    center_x = x + width / 2.0
    center_y = y + height / 2.0

    target_aspect = output_width / output_height
    context_width = max(width * context_scale, height * context_scale * target_aspect, 36.0)
    context_height = context_width / target_aspect
    if context_height < max(height * context_scale, 36.0):
        context_height = max(height * context_scale, 36.0)
        context_width = context_height * target_aspect

    context_width = min(float(image_width), context_width)
    context_height = min(float(image_height), context_height)
    left = max(0.0, min(center_x - context_width / 2.0, image_width - context_width))
    top = max(0.0, min(center_y - context_height / 2.0, image_height - context_height))
    right = min(float(image_width), left + context_width)
    bottom = min(float(image_height), top + context_height)

    left_i, top_i = int(math.floor(left)), int(math.floor(top))
    right_i, bottom_i = int(math.ceil(right)), int(math.ceil(bottom))
    context = image[top_i:bottom_i, left_i:right_i].copy()
    if context.size == 0:
        raise ValueError(f"empty context for box {(x, y, width, height)}")

    scale_x = output_width / context.shape[1]
    scale_y = output_height / context.shape[0]
    interpolation = cv2.INTER_CUBIC if min(scale_x, scale_y) > 1.0 else cv2.INTER_AREA
    context = cv2.resize(context, (output_width, output_height), interpolation=interpolation)

    x1 = int(round((x - left_i) * scale_x))
    y1 = int(round((y - top_i) * scale_y))
    x2 = int(round((x + width - left_i) * scale_x))
    y2 = int(round((y + height - top_i) * scale_y))
    x1, x2 = sorted((max(0, min(output_width - 1, x1)), max(0, min(output_width - 1, x2))))
    y1, y2 = sorted((max(0, min(output_height - 1, y1)), max(0, min(output_height - 1, y2))))
    thickness = max(2, int(round(min(scale_x, scale_y))))
    cv2.rectangle(context, (x1, y1), (x2, y2), (255, 255, 0), thickness)
    return context


def _sheet_image(
    root: Path,
    entries: list[dict[str, Any]],
    scene_id: str,
    page_index: int,
    columns: int,
    rows: int,
    cell_width: int,
    cell_height: int,
    context_scale: float,
) -> np.ndarray:
    legend_height = 72
    cell_header = 30
    canvas = np.full(
        (legend_height + rows * cell_height, columns * cell_width, 3),
        (24, 24, 24),
        dtype=np.uint8,
    )
    cv2.putText(
        canvas,
        f"{scene_id} / page {page_index + 1} | LABEL FROM PIXELS ONLY",
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "1 hai_chan   2 tru_dung   3 goi_nho   4 ic_hai_ben   "
        "5 ic_bon_ben   6 ic_khong_chan   7 connector   suffix ? = uncertain",
        (12, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    image_cache: dict[str, np.ndarray] = {}
    content_width = cell_width - 10
    content_height = cell_height - cell_header - 8
    for ordinal, entry in enumerate(entries):
        row, column = divmod(ordinal, columns)
        origin_x = column * cell_width
        origin_y = legend_height + row * cell_height
        crop_path = entry["crop_path"]
        image = image_cache.get(crop_path)
        if image is None:
            image = cv2.imread(str(root / "crops" / crop_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(root / "crops" / crop_path)
            image_cache[crop_path] = image

        context = _fit_context(
            image,
            entry["bbox"],
            content_width,
            content_height,
            context_scale,
        )
        canvas[
            origin_y + cell_header : origin_y + cell_header + content_height,
            origin_x + 5 : origin_x + 5 + content_width,
        ] = context
        header = (
            f"{ordinal:02d} | T{entry['tile_index']:02d}:B{entry['box_index']:02d} | "
            f"{entry['bbox']['w']}x{entry['bbox']['h']}"
        )
        cv2.putText(
            canvas,
            header,
            (origin_x + 7, origin_y + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (80, 220, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            canvas,
            (origin_x, origin_y),
            (origin_x + cell_width - 1, origin_y + cell_height - 1),
            (80, 80, 80),
            1,
        )
    return canvas


def build_sheets(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    seed_path = args.seed_json.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = _json_load(seed_path)
    if tuple(checkpoint.get("classes", ())) != PACKAGE_CLASSES:
        raise ValueError("checkpoint does not use the fixed seven-class package taxonomy")
    scenes = _scene_crops(root, checkpoint)
    parts = _balanced_assignments(scenes, checkpoint, args.parts)
    page_size = args.columns * args.rows

    index_payload: dict[str, Any] = {
        "schema": "aoi.package_visual_contact_sheets/v1",
        "source_seed": str(seed_path),
        "source_seed_sha256": _file_sha256(seed_path),
        "root": str(root),
        "class_tokens": TOKEN_TO_CLASS,
        "unknown_tokens": sorted(UNKNOWN_TOKENS),
        "context_scale": args.context_scale,
        "parts": [],
        "sheets": {},
    }
    total_boxes = 0
    total_sheets = 0

    for part_index, scene_ids in enumerate(parts):
        part_name = f"part_{part_index}"
        part_dir = output_dir / part_name
        part_dir.mkdir()
        label_lines = [
            "# One token per numbered cell, in order. 1..7; append ? when uncertain.",
            "# u means unresolved. Do not use a model to fill this file.",
        ]
        part_box_count = 0
        part_sheet_names: list[str] = []
        for scene_id in scene_ids:
            entries: list[dict[str, Any]] = []
            for tile_index, crop_path in enumerate(scenes[scene_id]):
                for box_index, box in enumerate(checkpoint["crops"][crop_path]["boxes"]):
                    entries.append(
                        {
                            "scene_id": scene_id,
                            "crop_path": crop_path,
                            "tile_index": tile_index,
                            "box_index": box_index,
                            "bbox": {key: int(box[key]) for key in ("x", "y", "w", "h")},
                        }
                    )

            safe_scene = "".join(character if character.isalnum() else "_" for character in scene_id)
            for page_index, start in enumerate(range(0, len(entries), page_size)):
                page_entries = entries[start : start + page_size]
                sheet_name = f"{safe_scene}__p{page_index:02d}.png"
                relative_name = f"{part_name}/{sheet_name}"
                sheet = _sheet_image(
                    root,
                    page_entries,
                    scene_id,
                    page_index,
                    args.columns,
                    args.rows,
                    args.cell_width,
                    args.cell_height,
                    args.context_scale,
                )
                if not cv2.imwrite(str(part_dir / sheet_name), sheet, [cv2.IMWRITE_PNG_COMPRESSION, 2]):
                    raise OSError(f"could not write {part_dir / sheet_name}")
                index_payload["sheets"][relative_name] = page_entries
                part_sheet_names.append(relative_name)
                label_lines.append(f"{relative_name} |")
                total_sheets += 1

            part_box_count += len(entries)
            total_boxes += len(entries)

        template_name = f"{part_name}.labels"
        (output_dir / template_name).write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        index_payload["parts"].append(
            {
                "part": part_index,
                "scenes": scene_ids,
                "box_count": part_box_count,
                "sheet_count": len(part_sheet_names),
                "label_file": template_name,
                "sheets": part_sheet_names,
            }
        )

    index_payload["total_boxes"] = total_boxes
    index_payload["total_sheets"] = total_sheets
    _json_dump(output_dir / "index.json", index_payload)
    print(json.dumps({
        "output_dir": str(output_dir),
        "total_boxes": total_boxes,
        "total_sheets": total_sheets,
        "parts": index_payload["parts"],
    }, ensure_ascii=False, indent=2))


def _iter_label_rows(path: Path) -> Iterable[tuple[str, list[str], int]]:
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" not in line:
            raise ValueError(f"{path}:{line_number}: expected '<sheet> | <tokens>'")
        sheet_name, raw_tokens = line.split("|", 1)
        tokens = raw_tokens.split()
        yield sheet_name.strip().replace("\\", "/"), tokens, line_number


def merge_labels(args: argparse.Namespace) -> None:
    index = _json_load(args.index.resolve())
    seed_path = args.seed_json.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {output_path}")
    if _file_sha256(seed_path) != index["source_seed_sha256"]:
        raise ValueError("seed checkpoint differs from the one used to build contact sheets")

    checkpoint = _json_load(seed_path)
    output = deepcopy(checkpoint)
    seen_sheets: set[str] = set()
    uncertain = 0
    unresolved = 0
    labelled = 0
    class_counts = {name: 0 for name in PACKAGE_CLASSES}

    for label_path in args.labels:
        resolved_label_path = label_path.resolve()
        for sheet_name, tokens, line_number in _iter_label_rows(resolved_label_path):
            if sheet_name in seen_sheets:
                raise ValueError(f"sheet labelled more than once: {sheet_name}")
            if sheet_name not in index["sheets"]:
                raise ValueError(f"{resolved_label_path}:{line_number}: unknown sheet {sheet_name}")
            entries = index["sheets"][sheet_name]
            if len(tokens) != len(entries):
                raise ValueError(
                    f"{resolved_label_path}:{line_number}: {sheet_name} has {len(entries)} "
                    f"cells but {len(tokens)} tokens"
                )
            seen_sheets.add(sheet_name)

            for token, entry in zip(tokens, entries):
                token = token.lower()
                is_uncertain = token.endswith("?")
                bare_token = token[:-1] if is_uncertain else token
                if bare_token in TOKEN_TO_CLASS:
                    package_class = TOKEN_TO_CLASS[bare_token]
                    class_counts[package_class] += 1
                    labelled += 1
                elif bare_token in UNKNOWN_TOKENS:
                    package_class = "unknown"
                    unresolved += 1
                    is_uncertain = True
                else:
                    raise ValueError(
                        f"{resolved_label_path}:{line_number}: invalid token {token!r}"
                    )
                if is_uncertain:
                    uncertain += 1

                target = output["crops"][entry["crop_path"]]["boxes"][entry["box_index"]]
                source = checkpoint["crops"][entry["crop_path"]]["boxes"][entry["box_index"]]
                if any(int(target[key]) != int(source[key]) for key in ("x", "y", "w", "h")):
                    raise AssertionError("box geometry changed before visual labels were applied")
                target["cls"] = package_class
                target["needs_review"] = True
                target["prelabel_reason"] = (
                    "manual_visual_first_pass:unresolved"
                    if package_class == "unknown"
                    else "manual_visual_first_pass:uncertain"
                    if is_uncertain
                    else "manual_visual_first_pass"
                )

    expected_sheets = set(index["sheets"])
    missing_sheets = expected_sheets - seen_sheets
    extra_sheets = seen_sheets - expected_sheets
    if missing_sheets or extra_sheets:
        raise ValueError(
            f"incomplete labels: missing {len(missing_sheets)} sheets; extra {len(extra_sheets)}"
        )

    for record in output["crops"].values():
        record["status"] = ""
        record["needs_review"] = True
    output["reviewer_id"] = args.reviewer_id
    output["note"] = (
        "First package-label pass made by direct visual inspection of every numbered box; "
        "no local detector/classifier/model was used. All crops remain unverified for the "
        "owner's second review."
    )
    output["visual_labelling_pass"] = {
        "schema": "aoi.package_visual_labelling_pass/v1",
        "date": args.review_date,
        "method": "direct_pixel_inspection_of_numbered_contact_sheets",
        "local_model_used": False,
        "source_seed_sha256": index["source_seed_sha256"],
        "reviewer_id": args.reviewer_id,
        "box_count": labelled + unresolved,
        "labelled_count": labelled,
        "unresolved_count": unresolved,
        "uncertain_count": uncertain,
        "class_counts": class_counts,
        "review_status": "owner_second_pass_required",
    }
    _json_dump(output_path, output)
    print(json.dumps(output["visual_labelling_pass"], ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sheets = subparsers.add_parser("sheets", help="render model-free numbered contact sheets")
    sheets.add_argument("root", type=Path)
    sheets.add_argument("--seed-json", type=Path, required=True)
    sheets.add_argument("--output-dir", type=Path, required=True)
    sheets.add_argument("--parts", type=int, default=4)
    sheets.add_argument("--columns", type=int, default=8)
    sheets.add_argument("--rows", type=int, default=4)
    sheets.add_argument("--cell-width", type=int, default=300)
    sheets.add_argument("--cell-height", type=int, default=290)
    sheets.add_argument("--context-scale", type=float, default=2.15)
    sheets.set_defaults(func=build_sheets)

    merge = subparsers.add_parser("merge", help="apply reviewed sheet tokens to a new checkpoint")
    merge.add_argument("--index", type=Path, required=True)
    merge.add_argument("--seed-json", type=Path, required=True)
    merge.add_argument("--labels", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--reviewer-id", default="codex_visual_pass_20260903")
    merge.add_argument("--review-date", default=date.today().isoformat())
    merge.set_defaults(func=merge_labels)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
