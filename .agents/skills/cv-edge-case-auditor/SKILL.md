---
name: cv-edge-case-auditor
description: Review checklist for OpenCV and metrology code in AOI pipelines. Scans for sub-pixel rounding errors, ROI truncation/out-of-bounds, mask alignment, and numeric overflow.
---

# CV Edge Case Auditor Checklist

Use this skill whenever reviewing code diffs or debugging issues in OpenCV image processing pipelines.

## Audit Checklist

### 1. ROI Slicing & Boundary Clipping
- [ ] **Shifted ROI Slicing**: When cropping an image at offset `(x + dx, y + dy)`, are `x + dx` and `y + dy` clamped or padded?
- [ ] **Shape Equivalence**: Does `observed_roi.shape == template.shape` hold under all shift conditions, including image boundary hits?
- [ ] **Empty ROI Protection**: Are empty ROIs (`width <= 0` or `height <= 0`) explicitly checked and rejected with a clear status string?

### 2. Sub-pixel & Coordinate Conversions
- [ ] **Integer Truncation vs Rounding**: Are pixel coordinates rounded with `int(round(val))` instead of plain `int(val)` truncation?
- [ ] **Base vs Residual Separation**: When combining coarse translation `(dx_int, dy_int)` and sub-pixel refinement `(sub_dx, sub_dy)`, is the base shift explicitly added back to the final result?
- [ ] **Center Offset Reference**: Is `center_px` defined relative to the Golden template origin `(expected_center - roi_top_left)` without double-counting translation offsets?
- [ ] **Angle Periodicity**: Is component symmetry (`180°` vs `360°`) normalized using periodic angle mapping (`(angle + period/2) % period - period/2`)?

### 3. Mask & Array Operations
- [ ] **Mask Alignment**: Is the mask in the correct coordinate space (Golden space vs Observed space)?
- [ ] **Mask Dilation Radius**: Is `ecc_mask` or search mask dilation scaled appropriately to cover expected component displacement without incorporating static PCB background?
- [ ] **Boolean vs Uint8 Indexing**: Are masks converted to boolean masks `mask > 0` before array indexing (e.g., `image[mask > 0]`)?
- [ ] **Valid Overlap Ratio**: Is the ratio of valid overlapping pixels calculated against `np.sum(component_mask > 0)` to fail-closed when components are partially out-of-frame?

### 4. Data Types & Overflow Prevention
- [ ] **Color Space Conversion**: Are images cast to `np.float32` after `cv2.cvtColor(img, cv2.COLOR_BGR2LAB)` before computing absolute differences?
- [ ] **Division by Zero**: Are denominators in ratio metrics protected with `max(1e-6, denominator)`?
- [ ] **OpenCV Matrix Dtype**: Are transformation matrices passed to `cv2.warpAffine` or `cv2.findTransformECC` formatted as `np.float32` or `np.float64`?
