"""Measure known-good boards and propose step-6.2 thresholds from them.

The defaults shipped in :class:`SolderGradingConfig` are placeholders. They have
to be, because "how much of this ROI should be solder" depends on the lens, the
lighting, the land geometry and how generously step 5.5 sized the ROI -- none of
which this repository can know in advance. Running the shipped numbers on a new
line typically flags most of the board.

So measure instead of guess. Point this at boards you have already accepted,
and it reports the distribution of every feature and proposes thresholds at
percentiles of it. A joint below the 1st percentile of your own good boards is
a genuine outlier; a joint below a number invented here is not evidence of
anything.

    python scripts/calibrate_solder_thresholds.py good_boards/ ^
        --model models/active/detector/best.onnx ^
        --output config/solder_thresholds.json

The output is a JSON fragment that drops straight into the ``solder_grading``
section of the pipeline config. Review it before use: if the boards fed in were
not actually all good, the thresholds inherit their defects.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from aoi_pipeline import (  # noqa: E402
    AOIPipeline,
    AOIPipelineError,
    PipelineConfig,
    load_image,
)

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}

FEATURE_COLUMNS = (
    "solder_ratio",
    "span_ratio",
    "width_ratio",
    "centroid_offset_ratio",
    "specular_ratio",
    "contrast",
    "uniformity",
    "edge_density",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Propose step-6.2 thresholds from boards you have accepted."
    )
    parser.add_argument("images", nargs="+", help="Known-good board images, folders or globs.")
    parser.add_argument("--output", required=True, help="Where to write the JSON fragment.")
    parser.add_argument("--model", default=None, help="Detector weights (.onnx preferred).")
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence.")
    parser.add_argument("--cad", default=None, help="Optional board CAD file for step 5.5.")
    parser.add_argument(
        "--low-percentile",
        type=float,
        default=1.0,
        help="Percentile that becomes the lower bound, e.g. insufficient (default: 1).",
    )
    parser.add_argument(
        "--high-percentile",
        type=float,
        default=99.0,
        help="Percentile that becomes the upper bound, e.g. excess (default: 99).",
    )
    parser.add_argument(
        "--dump-features",
        default=None,
        help="Also write every measured ROI to this CSV, for plotting the distribution.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Stop after N images.")
    return parser


def resolve_images(patterns: list[str]) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            found.extend(
                item for item in sorted(path.rglob("*"))
                if item.suffix.lower() in IMAGE_EXTENSIONS
            )
        elif path.is_file():
            found.append(path)
        else:
            found.extend(
                Path(item) for item in sorted(glob.glob(pattern))
                if Path(item).suffix.lower() in IMAGE_EXTENSIONS
            )
    seen: set[Path] = set()
    unique: list[Path] = []
    for item in found:
        resolved = item.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    images = resolve_images(args.images)
    if not images:
        print("No board images matched the given paths.", file=sys.stderr)
        return 2
    if args.limit > 0:
        images = images[: args.limit]

    config = PipelineConfig()
    config.model_detector.confidence = args.conf
    config.model_detector.end2end = False
    config.cad.path = args.cad
    try:
        pipeline = AOIPipeline(config, model_path=args.model)
    except AOIPipelineError as exc:
        print(f"Could not build the pipeline: {exc}", file=sys.stderr)
        return 1
    if args.model is None:
        print(
            "WARNING: no detector model given; ROIs come from the OpenCV candidate "
            "demo and the thresholds will describe those, not real components.",
            file=sys.stderr,
        )

    rows: list[dict[str, object]] = []
    for index, path in enumerate(images, start=1):
        try:
            run = pipeline.run(load_image(path), source_name=path.name)
        except AOIPipelineError as exc:
            print(f"[{index}/{len(images)}] {path.name}: FAILED — {exc}", file=sys.stderr)
            continue
        joints = [v for v in run.solder_verdicts if v.scope == "joint" and v.features]
        for verdict in joints:
            row = {"source_image": path.name, "joint_id": verdict.joint_id,
                   "component_label": verdict.component_label}
            row.update(
                {name: getattr(verdict.features, name) for name in FEATURE_COLUMNS}
            )
            rows.append(row)
        print(f"[{index}/{len(images)}] {path.name}: {len(joints)} joint ROIs measured")

    if not rows:
        print("No joint ROIs were measured; nothing to calibrate.", file=sys.stderr)
        return 1

    summary = _summarize(rows, args.low_percentile, args.high_percentile)
    proposal = _propose(summary, args.low_percentile, args.high_percentile, len(rows), len(images))

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(proposal, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.dump_features:
        dump = Path(args.dump_features).expanduser().resolve()
        dump.parent.mkdir(parents=True, exist_ok=True)
        with dump.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["source_image", "joint_id", "component_label", *FEATURE_COLUMNS]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPer-ROI features: {dump}")

    _report(summary, rows)
    print(f"\nProposed thresholds: {output}")
    print(
        "Paste the 'solder_grading' object into your pipeline config. Review it "
        "first: these numbers describe the boards you supplied, so any defect "
        "among them widens the thresholds that were meant to catch it."
    )
    return 0


def _summarize(
    rows: list[dict[str, object]], low: float, high: float
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for name in FEATURE_COLUMNS:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        summary[name] = {
            "count": int(values.size),
            "min": float(values.min()),
            "p_low": float(np.percentile(values, low)),
            "median": float(np.median(values)),
            "p_high": float(np.percentile(values, high)),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    return summary


def _propose(
    summary: dict[str, dict[str, float]],
    low: float,
    high: float,
    roi_count: int,
    image_count: int,
) -> dict[str, object]:
    solder = summary["solder_ratio"]
    span = summary["span_ratio"]
    specular = summary["specular_ratio"]
    contrast = summary["contrast"]
    offset = summary["centroid_offset_ratio"]

    # Sit the "insufficient" line just under the worst good joint seen, and the
    # "missing" line well under that: a land with no solder should be nowhere
    # near the low tail of a healthy distribution.
    insufficient = max(0.01, solder["p_low"] * 0.9)
    missing = max(0.002, insufficient * 0.25)
    return {
        "_meta": {
            "generated_from_images": image_count,
            "measured_joint_rois": roi_count,
            "low_percentile": low,
            "high_percentile": high,
            "note": (
                "Đo từ board đã được chấp nhận. Ngưỡng chỉ đúng với đúng camera, "
                "ống kính, ánh sáng và recipe ROI đã dùng khi đo."
            ),
        },
        "solder_grading": {
            "missing_solder_ratio": round(missing, 4),
            "insufficient_solder_ratio": round(insufficient, 4),
            "excess_solder_ratio": round(min(0.98, solder["p_high"] * 1.1), 4),
            "insufficient_span_ratio": round(max(0.05, span["p_low"] * 0.9), 4),
            "cold_specular_ratio": round(max(0.0005, specular["p_low"] * 0.8), 5),
            "cold_contrast": round(max(0.005, contrast["p_low"] * 0.8), 4),
            "max_centroid_offset_ratio": round(min(0.95, offset["p_high"] * 1.15), 4),
            "escape_guard_solder_ratio": round(max(0.005, missing * 1.5), 4),
        },
        "distribution": summary,
    }


def _report(summary: dict[str, dict[str, float]], rows: list[dict[str, object]]) -> None:
    print(f"\nMeasured {len(rows)} joint ROIs")
    print(f"{'feature':24s} {'min':>8s} {'p_low':>8s} {'median':>8s} {'p_high':>8s} {'max':>8s}")
    for name in FEATURE_COLUMNS:
        stats = summary[name]
        print(
            f"{name:24s} {stats['min']:8.4f} {stats['p_low']:8.4f} "
            f"{stats['median']:8.4f} {stats['p_high']:8.4f} {stats['max']:8.4f}"
        )
    spread = summary["solder_ratio"]
    if spread["std"] > spread["mean"]:
        print(
            "\nWARNING: solder_ratio varies more than its own mean. One threshold "
            "will not fit every package on this board; consider calibrating per "
            "component class, or tightening the step-5.5 ROI geometry first.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    raise SystemExit(main())
