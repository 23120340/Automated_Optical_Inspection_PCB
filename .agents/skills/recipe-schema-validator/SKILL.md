---
name: recipe-schema-validator
description: Validates Golden inspection recipe JSON schemas, asset hashes (SHA-256), metrology calibration, and alignment quality gates before running pipeline inspection.
---

# Recipe Schema Validator

Use this skill whenever modifying `recipe.py`, creating new enrollment scripts, or loading recipe files.

## Schema Validation Criteria (`recipe.py`)

1. **Schema Version & Coordinate Space**:
   - `schema_version` MUST be `aoi-inspection-recipe/1.1`.
   - `coordinate_space` MUST be `golden_board_pixels`.

2. **Asset Integrity & SHA-256 Hashes**:
   - `golden_asset_path` (e.g., `golden.png`) and all template/mask asset paths must exist.
   - All assets MUST be stored in lossless format (`.png`). Never use `.jpg` or `.jpeg`.
   - `asset_sha256` dictionary must contain an exact 64-character hex digest matching every referenced asset.

3. **Deterministic Slot Identification**:
   - `slot_id` values MUST follow sequential pattern: `slot_0001`, `slot_0002`, ..., `slot_N`.
   - Bounding boxes MUST be valid non-empty `xyxy` coordinates within image dimensions `(width, height)`.

4. **Metrology & Production Eligibility**:
   - `metrology.verified` must be `True` for production eligibility.
   - `alignment.anchors` count MUST satisfy `min_anchors >= 2` (default 3+).
   - No `opencv_candidate` sources allowed in production recipes.
   - `component_detector` model identifier MUST be present in `model_identifiers`.

## Validation Execution

Validate a recipe file programmatically via Python:

```python
from pathlib import Path
from aoi_pipeline.recipe import load_recipe, validate_recipe_assets

recipe_dir = Path("path/to/recipe")
recipe = load_recipe(recipe_dir / "recipe.json", recipe_dir)
validate_recipe_assets(recipe, recipe_dir)
print("Recipe schema and asset validation PASSED successfully.")
```
