---
name: cv-pipeline-guardrails
description: Enforces strict computer vision, metrology, and architectural guardrails for the AOI PCB pipeline when editing alignment.py, position.py, golden_compare.py, or recipe.py.
---

# CV Pipeline Guardrails for AOI PCB Inspection

Use this skill whenever modifying core computer vision logic in `aoi_pipeline/` (`alignment.py`, `position.py`, `golden_compare.py`, `inspection.py`, `recipe.py`).

## Core Architecture Guardrails

### 1. Image Domain Separation
- **Measurement Domain**: Never resize, letterbox, or warp measurement images arbitrarily. Native resolution BGR `uint8` images must be preserved for all alignment, position measurement, and Golden compare steps.
- **Inference Domain**: Object detection models (e.g., `best.onnx`) run on letterboxed/resized crops only for generating candidate proposals or class hints. **Never** use model bounding box centers as final metrology measurements.

### 2. Coordinate System Conventions
- **Canonical Coordinate Space**: All recipe bounding boxes, anchor points, and slot centers MUST be defined in `golden_board_pixels`.
- **Bounding Box Convention**: Use `xyxy` format `(x1, y1, x2, y2)` with right and bottom edges **exclusive**.
- **Sign Convention**:
  - `dx_px = observed_center_x - expected_center_x` (positive when component shifted right).
  - `dy_px = observed_center_y - expected_center_y` (positive when component shifted down).
  - `angle_deg`: Angle in degrees. Account for periodic symmetry (`180°` vs `360°`).

### 3. Alignment Guardrails (`alignment.py`)
- **Fail-Closed Policy**: If anchors are insufficient (`< min_anchors`), RANSAC residual exceeds `max_residual_px`, or affine scale/rotation exceeds quality gates: immediately return `INVALID` alignment status and abort inspection.
- **No Resize Fallback**: Production inspection must NEVER fall back to scaling or arbitrary resizes to force a PASS decision.
- **Model**: Partial affine transformation (`estimateAffinePartial2D`) only (translation + rotation + uniform scale).

### 4. Position Check Guardrails (`position.py`)
- **Coarse Match + Refinement**: First find coarse integer translation `(dx, dy)` using template matching (`cv2.matchTemplate`), then refine sub-pixel position and Euclidean pose (`cv2.findTransformECC`).
- **No Scale/Shear**: Local pose refinement MUST be Euclidean (translation + rotation only). Do not allow affine or homography scaling in local position measurement.
- **Translated ROI Cropping**: When extracting `observed_roi` for sub-pixel rotation refinement, ALWAYS shift the crop window by integer `(dx, dy)` so the component is not truncated at the ROI boundaries.
- **Zero Padding**: If a translated ROI extends beyond the image border, zero-pad the ROI to maintain identical shape with `template`.

### 5. Golden Compare Guardrails (`golden_compare.py`)
- **Pose Compensation**: Warping MUST occur after position check. Compensate local pose using `cv2.warpAffine` with `WARP_INVERSE_MAP` to bring the test ROI back to Golden coordinates.
- **Multi-Metric Evaluation**: Do not decide appearance defect based solely on a single global SSIM score. Combine Luminance-normalized SSIM, LAB color diff ratio, edge diff ratio, and connected component blob analysis.
- **Mask Compliance**: Always apply `compare_mask`, `ignore_mask`, and `valid_overlap_mask`. Ignore masked pixels in all metrics.

### 6. Dependency & Model Policy
- Reuse existing OpenCV (`cv2`) and NumPy utilities. Do not add heavy external vision dependencies.
- Keep `best.onnx` detector adapter (`UltralyticsDetector`) decoupled via dependency injection.
