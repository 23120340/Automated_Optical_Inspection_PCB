"""Step 5: padded component crop extraction and letterbox normalization."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import CropConfig, terminal_geometry
from ..core.image_io import encode_image, ensure_bgr, letterbox_normalize
from ..core.models import BoundingBox, ComponentCrop, Detection


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
            crop_bbox = _expanded_box(
                source_bbox, width, height, self.config, detection.label
            )
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
    label: str = "",
) -> BoundingBox:
    if config.solder_aware_padding:
        pad_x, pad_y = _axis_padding(bbox, config, label)
    else:
        pad_x = pad_y = max(
            config.padding_pixels,
            int(round(max(bbox.width, bbox.height) * config.padding_ratio)),
        )
    width = bbox.width + 2 * pad_x
    height = bbox.height + 2 * pad_y
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


def _axis_padding(
    bbox: BoundingBox, config: CropConfig, label: str
) -> tuple[int, int]:
    """Per-axis margin so a crop can contain the terminals, not just the body.

    The long axis carries the terminals of a two-lead part, so it needs a much
    larger margin than the short axis. ``max(w, h)`` scaling is deliberately
    not used here: it over-pads the short side of a long thin chip and pulls in
    the neighbouring components.
    """

    profile = config.pad_profiles.get(terminal_geometry(label))
    if profile is None:
        pad = max(
            config.padding_pixels,
            int(round(max(bbox.width, bbox.height) * config.padding_ratio)),
        )
        return (pad, pad)
    horizontal_is_long = bbox.width >= bbox.height
    ratio_x = profile.long_axis if horizontal_is_long else profile.short_axis
    ratio_y = profile.short_axis if horizontal_is_long else profile.long_axis
    pad_x = max(config.padding_pixels, int(round(bbox.width * ratio_x)))
    pad_y = max(config.padding_pixels, int(round(bbox.height * ratio_y)))
    return (pad_x, pad_y)


def _normalize_crop(crop: np.ndarray, config: CropConfig) -> np.ndarray:
    return letterbox_normalize(crop, config.target_size, config.letterbox_color)


def _safe_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._")[:64] or "component"
