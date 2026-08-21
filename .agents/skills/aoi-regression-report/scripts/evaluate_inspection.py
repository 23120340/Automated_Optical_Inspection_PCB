"""Automated Regression Evaluation and Report Generator for AOI Inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import cv2
import numpy as np

from aoi_pipeline.inspection import AOIInspector
from aoi_pipeline.recipe import load_recipe


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AOI Inspection Regression.")
    parser.add_argument("--recipe-dir", type=str, required=True, help="Recipe output directory")
    parser.add_argument("--test-dir", type=str, required=True, help="Synthetic test dataset directory")
    parser.add_argument("--report-out", type=str, default="inspection_report.md", help="Output markdown path")
    args = parser.parse_args()

    recipe_dir = Path(args.recipe_dir)
    test_dir = Path(args.test_dir)
    gt_file = test_dir / "dataset_gt.json"

    if not gt_file.exists():
        print(f"Ground truth file not found: {gt_file}")
        return

    recipe = load_recipe(recipe_dir / "recipe.json", recipe_dir)
    inspector = AOIInspector()

    with open(gt_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    dx_errors = []
    dy_errors = []
    angle_errors = []
    tp, fp, tn, fn = 0, 0, 0, 0

    for sample in samples:
        img_path = test_dir / sample["image"]
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        result = inspector.inspect(image, recipe)
        gt_dict = sample["ground_truth"]

        for slot_result in result.slots:
            slot_id = slot_result.slot_id
            if slot_id not in gt_dict:
                continue

            gt = gt_dict[slot_id]
            true_dx = gt["true_dx_px"]
            true_dy = gt["true_dy_px"]
            true_angle = gt["true_angle_deg"]

            if slot_result.position is not None and slot_result.position.status == "PASS":
                meas_dx = slot_result.position.dx_px
                meas_dy = slot_result.position.dy_px
                meas_angle = slot_result.position.angle_deg or 0.0

                dx_errors.append(abs(meas_dx - true_dx))
                dy_errors.append(abs(meas_dy - true_dy))
                angle_errors.append(abs(meas_angle - true_angle))

            # Defect classification metric
            is_defect_true = gt["defect_type"] != "ok"
            is_defect_pred = slot_result.status.value != "PASS"

            if is_defect_true and is_defect_pred:
                tp += 1
            elif not is_defect_true and is_defect_pred:
                fp += 1
            elif not is_defect_true and not is_defect_pred:
                tn += 1
            else:
                fn += 1

    mae_dx = float(np.mean(dx_errors)) if dx_errors else 0.0
    mae_dy = float(np.mean(dy_errors)) if dy_errors else 0.0
    mae_angle = float(np.mean(angle_errors)) if angle_errors else 0.0

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)

    report_content = f"""# AOI Inspection Regression Evaluation Report

## Metrology Accuracy
- **MAE dx (px)**: `{mae_dx:.4f}`
- **MAE dy (px)**: `{mae_dy:.4f}`
- **MAE Angle (°)**: `{mae_angle:.4f}`

## Defect Detection Classification Metrics
- **True Positives (TP)**: `{tp}`
- **False Positives (FP)**: `{fp}`
- **True Negatives (TN)**: `{tn}`
- **False Negatives (FN)**: `{fn}`
- **Precision**: `{precision:.4f}`
- **Recall**: `{recall:.4f}`
- **F1 Score**: `{f1:.4f}`
"""

    report_path = Path(args.report_out)
    report_path.write_text(report_content, encoding="utf-8")
    print(f"Report generated successfully at: {report_path}")


if __name__ == "__main__":
    main()
