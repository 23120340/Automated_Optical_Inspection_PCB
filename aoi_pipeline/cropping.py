"""Step 5: padded component crop extraction and letterbox normalization."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .config import CropConfig
from .image_io import encode_image, ensure_bgr
from .models import BoundingBox, ComponentCrop, Detection


class ComponentCropper:
    def __init__(self, config: CropConfig | None = None) -> None:
        self.config = config or CropConfig()

    def extract(
        self,
        image: np.ndarray,
        detections: Sequence[Detection],
        output_dir: str | Path | None = None,
    ) -> list[ComponentCrop]:
        bgr = ensure_bgr(image)
        height, width = bgr.shape[:2]
        destination = Path(output_dir).expanduser().resolve() if output_dir is not None else None
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=True)

        crops: list[ComponentCrop] = []
        for index, detection in enumerate(detections):
            source_bbox = detection.bbox.clamp(width, height)
            crop_bbox = _expanded_box(source_bbox, width, height, self.config)
            x1, y1, x2, y2 = crop_bbox.to_int()
            raw_crop = bgr[y1:y2, x1:x2]
            if raw_crop.size == 0:
                continue
            normalized = _normalize_crop(raw_crop, self.config)
            safe_label = _safe_filename_part(detection.label)
            safe_detection_id = _safe_filename_part(detection.detection_id)
            filename = (
                f"{index:04d}_{safe_label}_{safe_detection_id}{self.config.image_extension}"
            )
            file_path: Path | None = None
            if destination is not None:
                file_path = destination / filename
                file_path.write_bytes(
                    encode_image(
                        normalized,
                        self.config.image_extension,
                        jpeg_quality=self.config.jpeg_quality,
                    )
                )
            crops.append(
                ComponentCrop(
                    image=normalized,
                    detection_id=detection.detection_id,
                    label=detection.label,
                    confidence=detection.confidence,
                    source_bbox=source_bbox,
                    crop_bbox=crop_bbox,
                    filename=filename,
                    path=file_path,
                    metadata={
                        "raw_shape": {
                            "height": int(raw_crop.shape[0]),
                            "width": int(raw_crop.shape[1]),
                        },
                        "normalized": self.config.target_size is not None,
                        "letterboxed": self.config.target_size is not None,
                    },
                )
            )
        return crops


def _expanded_box(
    bbox: BoundingBox,
    image_width: int,
    image_height: int,
    config: CropConfig,
) -> BoundingBox:
    pad = max(config.padding_pixels, int(round(max(bbox.width, bbox.height) * config.padding_ratio)))
    width = bbox.width + 2 * pad
    height = bbox.height + 2 * pad
    if config.square:
        width = height = max(width, height)
    center_x = (bbox.x1 + bbox.x2) / 2.0
    center_y = (bbox.y1 + bbox.y2) / 2.0
    expanded = BoundingBox(
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
    )
    return expanded.clamp(image_width, image_height)


def _normalize_crop(crop: np.ndarray, config: CropConfig) -> np.ndarray:
    if config.target_size is None:
        return np.ascontiguousarray(crop.copy())
    target_width, target_height = (int(value) for value in config.target_size)
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Crop target_size values must be positive")
    source_height, source_width = crop.shape[:2]
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(crop, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.full(
        (target_height, target_width, 3),
        tuple(int(np.clip(value, 0, 255)) for value in config.letterbox_color),
        dtype=np.uint8,
    )
    x = (target_width - resized_width) // 2
    y = (target_height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def _safe_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._")[:64] or "component"
