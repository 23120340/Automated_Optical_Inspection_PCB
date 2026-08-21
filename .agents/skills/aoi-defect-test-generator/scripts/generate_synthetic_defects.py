"""Synthetic PCB Defect Image Generator for AOI Regression Testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import cv2
import numpy as np


def warp_component(
    image: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    dx_px: float,
    dy_px: float,
    angle_deg: float,
) -> np.ndarray:
    """Warp a component region with translation and rotation into a copy of image."""
    x1, y1, x2, y2 = bbox_xyxy
    h, w = image.shape[:2]
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    # Extract component patch
    pad = 10
    px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
    px2, py2 = min(w, x2 + pad), min(h, y2 + pad)
    patch = image[py1:py2, px1:px2].copy()

    # Rotation + translation matrix around patch center
    patch_center = (center_x - px1, center_y - py1)
    M = cv2.getRotationMatrix2D(patch_center, angle_deg, 1.0)
    M[0, 2] += dx_px
    M[1, 2] += dy_px

    warped_patch = cv2.warpAffine(
        patch, M, (px2 - px1, py2 - py1), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )

    out = image.copy()
    mask = np.zeros((py2 - py1, px2 - px1), dtype=np.uint8)
    cv2.rectangle(mask, (x1 - px1, y1 - py1), (x2 - px1, y2 - py1), 255, -1)
    warped_mask = cv2.warpAffine(mask, M, (px2 - px1, py2 - py1), flags=cv2.INTER_NEAREST)

    # Inpaint or cover original position
    cv2.rectangle(out, (x1, y1), (x2, y2), (128, 128, 128), -1)

    # Overlay warped component
    roi = out[py1:py2, px1:px2]
    idx = warped_mask > 0
    roi[idx] = warped_patch[idx]
    out[py1:py2, px1:px2] = roi
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic AOI test dataset.")
    parser.add_argument("--recipe-dir", type=str, required=True, help="Path to recipe folder")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save generated dataset")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of synthetic samples")
    args = parser.parse_args()

    recipe_path = Path(args.recipe_dir) / "recipe.json"
    if not recipe_path.exists():
        print(f"Recipe file not found at {recipe_path}")
        return

    with open(recipe_path, "r", encoding="utf-8") as f:
        recipe_data = json.load(f)

    golden_img_path = Path(args.recipe_dir) / recipe_data["golden_asset_path"]
    golden_img = cv2.imread(str(golden_img_path))
    if golden_img is None:
        print(f"Golden image not found at {golden_img_path}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slots = recipe_data["slots"]
    dataset_info = []

    for idx in range(1, args.num_samples + 1):
        test_img = golden_img.copy()
        ground_truth = {}

        for slot in slots:
            slot_id = slot["slot_id"]
            bbox = slot["expected_bbox_xyxy"]
            x1, y1, x2, y2 = [int(v) for v in bbox]

            defect_type = random.choice(["ok", "shifted", "rotated", "shifted_rotated"])
            if defect_type == "ok":
                dx, dy, angle = 0.0, 0.0, 0.0
            elif defect_type == "shifted":
                dx = random.uniform(-15.0, 15.0)
                dy = random.uniform(-15.0, 15.0)
                angle = 0.0
            elif defect_type == "rotated":
                dx, dy = 0.0, 0.0
                angle = random.uniform(-10.0, 10.0)
            else:
                dx = random.uniform(-12.0, 12.0)
                dy = random.uniform(-12.0, 12.0)
                angle = random.uniform(-8.0, 8.0)

            if dx != 0 or dy != 0 or angle != 0:
                test_img = warp_component(test_img, (x1, y1, x2, y2), dx, dy, angle)

            ground_truth[slot_id] = {
                "true_dx_px": dx,
                "true_dy_px": dy,
                "true_angle_deg": angle,
                "defect_type": defect_type,
            }

        img_filename = f"test_sample_{idx:03d}.png"
        cv2.imwrite(str(output_dir / img_filename), test_img)

        dataset_info.append({"image": img_filename, "ground_truth": ground_truth})

    with open(output_dir / "dataset_gt.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Successfully generated {args.num_samples} synthetic test images in {output_dir}")


if __name__ == "__main__":
    main()
