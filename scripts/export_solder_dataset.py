"""Build a step-6.2 solder-defect dataset from a folder of board images.

Runs steps 0-5.5 over every board, derives the solder-joint ROIs and writes a
flat crop folder plus one CSV whose ``defect_class`` column is left empty for
the annotator. Labelling then becomes a per-row verdict instead of a boxing
job, because the geometry is already resolved.

    python scripts/export_solder_dataset.py boards/ --output datasets/solder_v1 ^
        --model models/detector/best.onnx --overlays

The crops carry the illumination they were captured under. A solder fillet is
readable only when the light encodes its slope, so review ``overlays/`` and a
sample of ``crops/`` before labelling a whole batch: if good and cold joints
look identical there, the fix is the lighting, not the model.
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

from aoi_pipeline import (  # noqa: E402
    AOIPipeline,
    AOIPipelineError,
    PipelineConfig,
    encode_image,
    load_image,
)
from aoi_pipeline.solder import render_solder_overlay  # noqa: E402

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}

CSV_COLUMNS = (
    "crop_path",
    "source_image",
    "joint_id",
    "detection_id",
    "component_label",
    "kind",
    "position",
    "pin_index",
    "terminal_geometry",
    "angle",
    "x1",
    "y1",
    "x2",
    "y2",
    "detector_confidence",
    "roi_width_px",
    "roi_height_px",
    "defect_class",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive solder-joint crops from board images for step 6.2 labelling."
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Image files, directories, or glob patterns holding board images.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Dataset directory to create. Existing crops with the same name are overwritten.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Detector weights (.onnx preferred, .pt from trusted sources only). "
        "Without it the OpenCV candidate demo runs and the labels are not real classes.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence (default: 0.25).")
    parser.add_argument("--imgsz", type=int, default=1280, help="Detector input size (default: 1280).")
    parser.add_argument(
        "--joint-size",
        type=int,
        default=128,
        help="Square crop size in pixels; 0 keeps the native ROI size (default: 128).",
    )
    parser.add_argument(
        "--split-pins",
        action="store_true",
        help="Cut multi-pin bands into one ROI per lead. Bands stay whole where the "
        "lead row cannot be read reliably.",
    )
    parser.add_argument(
        "--no-body-view",
        action="store_true",
        help="Skip the per-component view that shows the body together with its joints.",
    )
    parser.add_argument(
        "--joints-only",
        action="store_true",
        help="Write only joint ROIs to the CSV, excluding body views.",
    )
    parser.add_argument(
        "--overlays",
        action="store_true",
        help="Also write an annotated board per image so the ROI geometry can be checked.",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Stop after N images (0 = no limit)."
    )
    return parser


def resolve_images(patterns: list[str]) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            found.extend(
                item for item in sorted(path.rglob("*")) if item.suffix.lower() in IMAGE_EXTENSIONS
            )
        elif path.is_file():
            found.append(path)
        else:
            found.extend(
                Path(item)
                for item in sorted(glob.glob(pattern))
                if Path(item).suffix.lower() in IMAGE_EXTENSIONS
            )
    unique: list[Path] = []
    seen: set[Path] = set()
    for item in found:
        resolved = item.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def build_config(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig()
    config.model_detector.confidence = args.conf
    config.model_detector.image_size = args.imgsz
    # Match the Kaggle recipe: one-to-many head with external NMS.
    config.model_detector.end2end = False
    config.solder.enabled = True
    config.solder.split_pins = args.split_pins
    config.solder.include_body_view = not args.no_body_view
    config.solder.target_size = (
        (args.joint_size, args.joint_size) if args.joint_size > 0 else None
    )
    return config


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    images = resolve_images(args.images)
    if not images:
        print("No board images matched the given paths.", file=sys.stderr)
        return 2
    if args.limit > 0:
        images = images[: args.limit]

    output = Path(args.output).expanduser().resolve()
    crops_dir = output / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = output / "overlays"
    if args.overlays:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    try:
        pipeline = AOIPipeline(build_config(args), model_path=args.model)
    except AOIPipelineError as exc:
        print(f"Could not build the pipeline: {exc}", file=sys.stderr)
        return 1
    if args.model is None:
        print(
            "WARNING: no detector model given; component labels come from the "
            "OpenCV candidate demo and the derived ROI topology is a guess.",
            file=sys.stderr,
        )

    rows: list[dict[str, object]] = []
    failures = 0
    for index, image_path in enumerate(images, start=1):
        try:
            run = pipeline.run(load_image(image_path), source_name=image_path.name)
        except AOIPipelineError as exc:
            failures += 1
            print(f"[{index}/{len(images)}] {image_path.name}: FAILED — {exc}", file=sys.stderr)
            continue

        stem = image_path.stem
        written = 0
        for crop in run.solder_crops:
            joint = crop.joint
            if args.joints_only and joint.kind != "joint":
                continue
            filename = f"{stem}__{crop.filename}"
            (crops_dir / filename).write_bytes(encode_image(crop.image, ".png"))
            written += 1
            rows.append(
                {
                    "crop_path": f"crops/{filename}",
                    "source_image": image_path.name,
                    "joint_id": joint.joint_id,
                    "detection_id": joint.detection_id,
                    "component_label": joint.label,
                    "kind": joint.kind,
                    "position": joint.position,
                    "pin_index": "" if joint.pin_index is None else joint.pin_index,
                    "terminal_geometry": joint.terminal_geometry,
                    "angle": f"{joint.angle:.2f}",
                    "x1": f"{joint.bbox.x1:.2f}",
                    "y1": f"{joint.bbox.y1:.2f}",
                    "x2": f"{joint.bbox.x2:.2f}",
                    "y2": f"{joint.bbox.y2:.2f}",
                    "detector_confidence": f"{float(joint.metadata.get('detector_confidence', 0.0)):.4f}",
                    "roi_width_px": int(round(joint.bbox.width)),
                    "roi_height_px": int(round(joint.bbox.height)),
                    "defect_class": "",
                }
            )

        if args.overlays:
            overlay = render_solder_overlay(
                run.final_image, [crop.joint for crop in run.solder_crops]
            )
            (overlays_dir / f"{stem}.png").write_bytes(encode_image(overlay, ".png"))

        print(
            f"[{index}/{len(images)}] {image_path.name}: "
            f"{len(run.detections)} detections -> {written} ROI crops"
        )

    manifest = output / "solder_dataset.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    joints = sum(1 for row in rows if row["kind"] == "joint")
    print(f"\nWrote {len(rows)} crops ({joints} joints) to {crops_dir}")
    print(f"Label the empty 'defect_class' column in {manifest}")
    _report_small_rois(rows)
    if failures:
        print(f"{failures} image(s) failed; see the messages above.", file=sys.stderr)
    return 1 if failures and not rows else 0


def _report_small_rois(rows: list[dict[str, object]], threshold: int = 24) -> None:
    """Flag ROIs too small to judge, instead of letting them into training silently.

    A fillet needs roughly ten pixels across it before its shape is readable at
    all. ROIs under ``threshold`` px usually mean the capture resolution, not
    the geometry, is the limiting factor.
    """

    small = [
        row
        for row in rows
        if row["kind"] == "joint"
        and min(int(row["roi_width_px"]), int(row["roi_height_px"])) < threshold
    ]
    if not small:
        return
    share = 100.0 * len(small) / max(1, sum(1 for row in rows if row["kind"] == "joint"))
    print(
        f"WARNING: {len(small)} joint ROIs ({share:.1f}%) are under {threshold}px on "
        "their short side. At that scale a fillet cannot be graded; capture at a "
        "higher resolution or a smaller field of view before labelling these.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
