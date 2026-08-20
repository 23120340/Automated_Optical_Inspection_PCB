"""Measure whether step-1 preprocessing helps or hurts your trained detector.

Every training notebook in this repo (detector, classifier, solder) fits on
raw dataset images plus standard augmentation. It never sees the step-1 chain
that :class:`PreprocessConfig` runs by default in the app -- denoise,
gray-world white balance, CLAHE, percentile luminance normalization, and an
unsharp mask. That is a real train/inference domain gap, not a hypothetical
one, but *which way it cuts* depends on your camera and lighting and cannot be
argued from first principles. Measure it.

This script runs your trained detector twice on the same board photos: once
through the exact preprocessing your app applies by default, once "raw" (only
resize, nothing else), then matches boxes between the two runs by IoU+class
and reports what changed -- detections gained/lost and the confidence shift on
boxes both runs agree on. Add ``--isolate`` to flip each of the five steps on
one at a time against the raw baseline, so a regression can be pinned on one
step instead of "preprocessing" as an undifferentiated block.

    python scripts/compare_preprocessing_ab.py board_photos/ ^
        --model models/detector/kaggle/best.onnx --isolate

Needs real board photos, not synthetic fixtures: denoise and white-balance
only do real work against real sensor noise and real color cast.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import replace
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from aoi_pipeline import AOIPipeline, AOIPipelineError, PipelineConfig, load_image  # noqa: E402
from aoi_pipeline.config import PreprocessConfig  # noqa: E402
from aoi_pipeline.core.models import Detection  # noqa: E402
from aoi_pipeline.imaging.preprocessing import ImagePreprocessor  # noqa: E402

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
STEPS = ("denoise", "white_balance", "clahe", "normalize", "sharpen")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A/B the step-1 preprocessing chain against your trained detector."
    )
    parser.add_argument("images", nargs="+", help="Board photos, folders or globs.")
    parser.add_argument("--model", required=True, help="Detector weights (.onnx preferred).")
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence.")
    parser.add_argument(
        "--iou-match",
        type=float,
        default=0.5,
        help="IoU above which two boxes from different runs count as the same detection.",
    )
    parser.add_argument(
        "--isolate",
        action="store_true",
        help="Also flip each preprocessing step on individually against the raw "
        "baseline, to find which one moves the numbers.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Stop after N images.")
    parser.add_argument("--output", default=None, help="Optional path to dump the full JSON report.")
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


def _raw_preprocess() -> PreprocessConfig:
    """Only resize; every enhancement step off."""

    return PreprocessConfig(
        denoise=False, white_balance=False, clahe=False, normalize=False, sharpen=False
    )


def _iou(a: Detection, b: Detection) -> float:
    x1, y1 = max(a.bbox.x1, b.bbox.x1), max(a.bbox.y1, b.bbox.y1)
    x2, y2 = min(a.bbox.x2, b.bbox.x2), min(a.bbox.y2, b.bbox.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.bbox.area + b.bbox.area - intersection
    return intersection / union if union > 0 else 0.0


def _match(
    baseline: list[Detection], variant: list[Detection], threshold: float
) -> tuple[list[tuple[Detection, Detection]], list[Detection], list[Detection]]:
    """Greedy same-class IoU matching between two detection sets."""

    pairs: list[tuple[float, Detection, Detection]] = []
    for a in baseline:
        for b in variant:
            if a.label != b.label:
                continue
            iou = _iou(a, b)
            if iou >= threshold:
                pairs.append((iou, a, b))
    pairs.sort(key=lambda item: item[0], reverse=True)

    matched: list[tuple[Detection, Detection]] = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    for _, a, b in pairs:
        if id(a) in used_a or id(b) in used_b:
            continue
        used_a.add(id(a))
        used_b.add(id(b))
        matched.append((a, b))
    only_baseline = [a for a in baseline if id(a) not in used_a]
    only_variant = [b for b in variant if id(b) not in used_b]
    return matched, only_baseline, only_variant


def _run(pipeline: AOIPipeline, preprocess: PreprocessConfig, path: Path) -> list[Detection]:
    image = load_image(path)
    preprocessed = ImagePreprocessor(preprocess).process(image)
    board = pipeline.detect_board(preprocessed.image)
    return pipeline.detect_components(preprocessed.image, board)


def _summarize_pair(
    label: str, baseline: list[Detection], variant: list[Detection], iou_match: float
) -> dict[str, object]:
    matched, only_baseline, only_variant = _match(baseline, variant, iou_match)
    deltas = [b.confidence - a.confidence for a, b in matched]
    return {
        "variant": label,
        "baseline_count": len(baseline),
        "variant_count": len(variant),
        "matched": len(matched),
        "lost_vs_baseline": len(only_baseline),
        "gained_vs_baseline": len(only_variant),
        "lost_labels": sorted({d.label for d in only_baseline}),
        "gained_labels": sorted({d.label for d in only_variant}),
        "mean_confidence_delta": float(np.mean(deltas)) if deltas else 0.0,
        "confidence_delta_std": float(np.std(deltas)) if deltas else 0.0,
    }


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
    try:
        pipeline = AOIPipeline(config, model_path=args.model)
    except AOIPipelineError as exc:
        print(f"Could not build the pipeline: {exc}", file=sys.stderr)
        return 1

    raw_config = _raw_preprocess()
    default_config = PreprocessConfig()

    variant_configs: dict[str, PreprocessConfig] = {"full_default": default_config}
    if args.isolate:
        for step in STEPS:
            variant_configs[f"only_{step}"] = replace(_raw_preprocess(), **{step: True})

    per_image: list[dict[str, object]] = []
    aggregate: dict[str, list[dict[str, object]]] = {name: [] for name in variant_configs}

    for index, path in enumerate(images, start=1):
        try:
            baseline = _run(pipeline, raw_config, path)
        except AOIPipelineError as exc:
            print(f"[{index}/{len(images)}] {path.name}: FAILED — {exc}", file=sys.stderr)
            continue
        image_report: dict[str, object] = {"source_image": path.name, "raw_detections": len(baseline)}
        for name, variant_config in variant_configs.items():
            variant = _run(pipeline, variant_config, path)
            summary = _summarize_pair(name, baseline, variant, args.iou_match)
            image_report[name] = summary
            aggregate[name].append(summary)
        per_image.append(image_report)
        print(f"[{index}/{len(images)}] {path.name}: raw={len(baseline)} boxes")

    if not per_image:
        print("No images produced detections; nothing to compare.", file=sys.stderr)
        return 1

    report = {"per_image": per_image, "overall": _overall(aggregate)}
    _print_report(report["overall"])

    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nFull report: {output}")
    return 0


def _overall(aggregate: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    overall: dict[str, object] = {}
    for name, summaries in aggregate.items():
        if not summaries:
            continue
        overall[name] = {
            "images": len(summaries),
            "total_lost": sum(s["lost_vs_baseline"] for s in summaries),
            "total_gained": sum(s["gained_vs_baseline"] for s in summaries),
            "mean_confidence_delta": float(
                np.mean([s["mean_confidence_delta"] for s in summaries])
            ),
        }
    return overall


def _print_report(overall: dict[str, object]) -> None:
    print("\n=== raw (chỉ resize) so với các biến thể ===")
    print(f"{'variant':16s} {'mất':>6s} {'thêm':>6s} {'Δ conf (mean)':>16s}")
    for name, stats in overall.items():
        print(
            f"{name:16s} {stats['total_lost']:6d} {stats['total_gained']:6d} "
            f"{stats['mean_confidence_delta']:16.4f}"
        )
    default = overall.get("full_default")
    if default is None:
        return
    print()
    if default["total_lost"] > 0 and default["mean_confidence_delta"] < -0.02:
        print(
            "Tiền xử lý mặc định đang MẤT detection và giảm confidence so với ảnh thô "
            "trên chính model đã train của bạn. Cân nhắc tắt bớt (denoise/CLAHE/sharpen) "
            "cho tới khi bạn train lại model trên ảnh đã qua tiền xử lý -- hoặc tắt hẳn "
            "tiền xử lý trước detector."
        )
    elif default["total_gained"] > 0 and default["mean_confidence_delta"] > 0.02:
        print(
            "Tiền xử lý mặc định đang giúp: nhiều detection hơn và confidence cao hơn "
            "ảnh thô. Giữ nguyên."
        )
    else:
        print(
            "Không có khác biệt lớn trên bộ ảnh này. Chạy thêm ảnh (đặc biệt ảnh có "
            "board bẩn/mờ/lệch sáng, nơi denoise và CLAHE có việc thật để làm) trước "
            "khi kết luận."
        )
    print(
        "Dùng --isolate để biết bước nào (denoise/white_balance/clahe/normalize/sharpen) "
        "là nguyên nhân chính nếu có khác biệt."
    )


if __name__ == "__main__":
    raise SystemExit(main())
