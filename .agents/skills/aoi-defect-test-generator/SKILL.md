---
name: aoi-defect-test-generator
description: Generates synthetic PCB test images with controlled component shifts (dx/dy), rotations, missing components, or appearance defects alongside ground-truth JSON for regression testing.
---

# AOI Defect Test Generator

Use this skill to generate synthetic test datasets from real Golden PCB images to verify position measurement, rotation refinement, and Golden compare logic under controlled conditions.

## Usage

Run the generator script with Python:

```bash
python .agents/skills/aoi-defect-test-generator/scripts/generate_synthetic_defects.py \
    --recipe-dir tests/data/sample_recipe \
    --output-dir tmp/synthetic_tests \
    --num-samples 10
```

## Generated Defect Types

1. **Shifted Only (`shifted`)**: Component translated by `dx_px`, `dy_px` (from `0.25px` up to `25px`).
   - Expected Result: `position_status = PASS` (if within tolerance) or `POSITION_NG`, `appearance_status = PASS`.
2. **Rotated Only (`rotated`)**: Component rotated by `angle_deg` (`-15°` to `+15°`).
   - Expected Result: `position_status = PASS/NG`, `appearance_status = PASS` (after pose compensation).
3. **Shifted + Rotated (`pose_anomaly`)**: Combined translation + rotation.
4. **Missing Component (`missing`)**: Component area replaced with background texture / inpainting.
   - Expected Result: `position_status = UNMEASURABLE`, `appearance_status = DEFECT_MISSING`.
5. **Appearance Anomaly (`appearance`)**: Simulated scratch, stain, or wrong component soldered in ROI.
   - Expected Result: `position_status = PASS`, `appearance_status = DEFECT_APPEARANCE`.

## Ground Truth JSON Format

The generator outputs `ground_truth.json` in the target directory:

```json
{
  "test_image": "test_001.png",
  "slots": {
    "slot_0001": {
      "true_dx_px": 12.5,
      "true_dy_px": -3.2,
      "true_angle_deg": 4.5,
      "expected_position_status": "PASS",
      "expected_appearance_status": "PASS"
    }
  }
}
```
