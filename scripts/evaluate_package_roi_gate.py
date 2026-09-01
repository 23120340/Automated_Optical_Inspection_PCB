"""Gate a trained step-5.2 package artifact against labels and real ROI geometry.

This script intentionally runs *after* training.  A package model is eligible
for manual promotion only when all three plan gates pass:

* test macro recall is at least 0.85 on a board-grouped split;
* test confusion ``ic_hai_ben <-> ic_khong_chan`` is exactly zero;
* enabling accepted step-5.2 predictions does not reduce coverage of the 28
  hand-measured pads in ``tests/data/solder_geometry``.

It never copies an artifact into ``models/active`` and never changes the model
registry.  Promotion remains an explicit human action.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aoi_pipeline.classification.package import (  # noqa: E402
    ONNXPackageClassifier,
    PACKAGE_CLASS_NAMES,
)
from aoi_pipeline.config import PipelineConfig  # noqa: E402
from aoi_pipeline.models import BoundingBox, Detection, SolderJoint  # noqa: E402
from aoi_pipeline.pipeline import AOIPipeline  # noqa: E402
from aoi_pipeline.solder.geometry import deconflict_joint_rois  # noqa: E402


DEFAULT_TRUTH = (
    PROJECT_ROOT / "tests" / "data" / "solder_geometry" / "board_smd_00001.json"
)
MIN_MACRO_RECALL = 0.85
MIN_PAD_COVERAGE = 0.50
EXPECTED_PAD_COUNT = 28


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _evaluation_metrics(manifest: Mapping[str, Any]) -> tuple[float, int, str]:
    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("manifest.evaluation is required for package promotion")
    split_unit = str(evaluation.get("split_unit") or "").strip().lower()
    if split_unit != "board":
        raise ValueError("manifest.evaluation.split_unit must be 'board'")
    macro_recall = evaluation.get("test_macro_recall")
    if (
        isinstance(macro_recall, bool)
        or not isinstance(macro_recall, (int, float))
        or not 0.0 <= float(macro_recall) <= 1.0
    ):
        raise ValueError("manifest.evaluation.test_macro_recall must be in [0, 1]")

    confusion = evaluation.get("test_confusion_matrix")
    if not isinstance(confusion, Mapping):
        raise ValueError("manifest.evaluation.test_confusion_matrix is required")
    labels = confusion.get("class_names")
    matrix = confusion.get("matrix")
    if labels != list(PACKAGE_CLASS_NAMES):
        raise ValueError("test confusion class_names must match the seven-class order")
    if (
        not isinstance(matrix, list)
        or len(matrix) != len(PACKAGE_CLASS_NAMES)
        or any(not isinstance(row, list) or len(row) != len(labels) for row in matrix)
    ):
        raise ValueError("test confusion matrix must be a square 7x7 list")
    dual = labels.index("ic_hai_ben")
    hidden = labels.index("ic_khong_chan")
    dangerous = _nonnegative_count(matrix[dual][hidden]) + _nonnegative_count(
        matrix[hidden][dual]
    )
    return float(macro_recall), dangerous, split_unit


def _nonnegative_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("confusion matrix values must be non-negative integers")
    result = int(value)
    if result != value or result < 0:
        raise ValueError("confusion matrix values must be non-negative integers")
    return result


def _coverage(roi: BoundingBox, pad: Sequence[int]) -> float:
    px1, py1, px2, py2 = (float(value) for value in pad)
    intersection_width = max(0.0, min(roi.x2, px2) - max(roi.x1, px1))
    intersection_height = max(0.0, min(roi.y2, py2) - max(roi.y1, py1))
    pad_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    return intersection_width * intersection_height / pad_area if pad_area else 0.0


def _coverage_summary(
    truth: Mapping[str, Any],
    baseline: Mapping[int, Sequence[SolderJoint]],
    candidate: Mapping[int, Sequence[SolderJoint]],
) -> dict[str, Any]:
    components = truth.get("components")
    if not isinstance(components, Mapping):
        raise ValueError("ground truth components must be an object")
    rows: list[dict[str, Any]] = []
    for designator, raw_entry in components.items():
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"ground truth component {designator} must be an object")
        detection_index = raw_entry.get("detection_index")
        pads = raw_entry.get("pads")
        if isinstance(detection_index, bool) or not isinstance(detection_index, int):
            raise ValueError(f"{designator}: invalid detection_index")
        if not isinstance(pads, list):
            raise ValueError(f"{designator}: pads must be a list")
        for pad_index, pad in enumerate(pads):
            if not isinstance(pad, list) or len(pad) != 4:
                raise ValueError(f"{designator}: pad {pad_index} must be xyxy")
            baseline_best = max(
                (_coverage(joint.bbox, pad) for joint in baseline.get(detection_index, ())),
                default=0.0,
            )
            candidate_best = max(
                (_coverage(joint.bbox, pad) for joint in candidate.get(detection_index, ())),
                default=0.0,
            )
            rows.append(
                {
                    "component": str(designator),
                    "pad_index": pad_index,
                    "baseline_coverage": round(baseline_best, 6),
                    "candidate_coverage": round(candidate_best, 6),
                    "baseline_covered": baseline_best >= MIN_PAD_COVERAGE,
                    "candidate_covered": candidate_best >= MIN_PAD_COVERAGE,
                }
            )
    baseline_count = sum(bool(row["baseline_covered"]) for row in rows)
    candidate_count = sum(bool(row["candidate_covered"]) for row in rows)
    return {
        "pad_count": len(rows),
        "minimum_coverage": MIN_PAD_COVERAGE,
        "baseline_covered": baseline_count,
        "candidate_covered": candidate_count,
        "coverage_not_reduced": candidate_count >= baseline_count,
        "pads": rows,
    }


def _joints_by_detection(
    image,
    detections: Sequence[Detection],
    pipeline: AOIPipeline,
) -> dict[int, list[SolderJoint]]:
    joints = pipeline.solder_cropper.derive(image, detections)
    joints = deconflict_joint_rois(joints, detections, pipeline.config.solder)
    index_by_id = {item.detection_id: index for index, item in enumerate(detections)}
    grouped: dict[int, list[SolderJoint]] = {
        index: [] for index in range(len(detections))
    }
    for joint in joints:
        if joint.kind == "joint":
            grouped[index_by_id[joint.detection_id]].append(joint)
    return grouped


def evaluate(
    model_path: Path,
    manifest_path: Path,
    *,
    truth_path: Path = DEFAULT_TRUTH,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    macro_recall, dangerous_confusions, split_unit = _evaluation_metrics(manifest)
    truth = _load_json(truth_path)
    image_path = truth_path.parent / str(truth.get("image") or "")
    raw = cv2.imread(str(image_path))
    if raw is None:
        raise ValueError(f"Cannot decode gate image: {image_path}")

    classifier = ONNXPackageClassifier(model_path, manifest_path)
    pipeline = AOIPipeline(PipelineConfig(), package_classifier=classifier)
    image = pipeline.preprocess(raw).image
    detections = [
        Detection(
            str(item["label"]),
            float(item["confidence"]),
            BoundingBox(*item["box"]),
            detection_id=f"gate_det_{index:04d}",
        )
        for index, item in enumerate(truth.get("detections", []))
    ]
    baseline = _joints_by_detection(image, detections, pipeline)
    crops = pipeline.make_crops(image, detections)
    predictions = pipeline.classify_packages(crops)
    annotated = pipeline.apply_package_classifications(detections, predictions)
    candidate = _joints_by_detection(image, annotated, pipeline)
    coverage = _coverage_summary(truth, baseline, candidate)
    if coverage["pad_count"] != EXPECTED_PAD_COUNT:
        raise ValueError(
            f"ROI gate requires exactly {EXPECTED_PAD_COUNT} hand-measured pads; "
            f"found {coverage['pad_count']}"
        )

    gates = {
        "board_grouped_split": split_unit == "board",
        "macro_recall_at_least_0_85": macro_recall >= MIN_MACRO_RECALL,
        "dangerous_confusions_zero": dangerous_confusions == 0,
        "real_pad_coverage_not_reduced": bool(coverage["coverage_not_reduced"]),
    }
    return {
        "schema": "pcb-package-promotion-gate/1.0",
        "model": str(model_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "truth": str(truth_path.resolve()),
        "test_macro_recall": macro_recall,
        "dangerous_ic_hai_ben_ic_khong_chan_confusions": dangerous_confusions,
        "package_decisions": {
            decision: sum(item.decision == decision for item in predictions)
            for decision in ("accept", "review", "unknown")
        },
        "roi_coverage": coverage,
        "gates": gates,
        "passed": all(gates.values()),
        "promotion": "manual_only",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path, help="trained package best.onnx")
    parser.add_argument("manifest", type=Path, help="matching model_manifest.json")
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args(argv)
    try:
        report = evaluate(
            args.model.resolve(),
            args.manifest.resolve(),
            truth_path=args.truth.resolve(),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
