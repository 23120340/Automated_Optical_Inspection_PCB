---
name: aoi-regression-report
description: Runs automated verification and benchmark reporting on AOI inspection datasets. Computes position errors (MAE dx, dy, angle), defect classification Precision/Recall/F1, and generates markdown summaries.
---

# AOI Regression Report Skill

Use this skill after making changes to core AOI algorithms (`position.py`, `alignment.py`, `golden_compare.py`, `inspection.py`) to automatically evaluate accuracy against a ground-truth dataset and generate an inspection regression report.

## Usage

Run evaluation via Python:

```bash
python .agents/skills/aoi-regression-report/scripts/evaluate_inspection.py \
    --recipe-dir tests/data/sample_recipe \
    --test-dir tmp/synthetic_tests \
    --report-out tmp/inspection_regression_report.md
```

## Evaluated Metrics

1. **Alignment Verification**:
   - Residual error (px), Inlier ratio, Fail-closed rate on invalid images.
2. **Position Check Metrology Accuracy**:
   - **MAE dx (px / mm)**: Mean Absolute Error between measured `dx` and ground-truth `dx`.
   - **MAE dy (px / mm)**: Mean Absolute Error between measured `dy` and ground-truth `dy`.
   - **MAE Angle (°)**: Mean Absolute Error of sub-pixel rotation.
3. **Defect Classification Performance**:
   - **Confusion Matrix**: True Positive (TP), False Positive (FP), True Negative (TN), False Negative (FN).
   - **Precision**: `TP / (TP + FP)`
   - **Recall**: `TP / (TP + FN)`
   - **F1-Score**: `2 * (Precision * Recall) / (Precision + Recall)`
