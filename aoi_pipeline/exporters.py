"""JSON and ZIP exporters for a completed steps 0-5 run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np

from .exceptions import ExportError
from .image_io import encode_image
from .models import PipelineRun


def export_json(run: PipelineRun, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.write_text(_manifest_json(run), encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        raise ExportError(f"Could not export JSON to {destination}: {exc}") from exc
    return destination


def export_zip(
    run: PipelineRun,
    path: str | Path,
    *,
    include_input: bool = True,
    include_intermediate: bool = True,
    include_crops: bool = True,
) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.suffix.lower() != ".zip":
        destination = destination.with_suffix(".zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr("manifest.json", _manifest_json(run))
            if include_input:
                archive.writestr("images/00_input.png", encode_image(run.input_image))
            if include_intermediate:
                archive.writestr(
                    "images/01_preprocessed.png", encode_image(run.preprocess_result.image)
                )
                archive.writestr("images/02_aligned.png", encode_image(run.alignment_result.image))
                archive.writestr("images/03_annotated.png", encode_image(render_annotations(run)))
                if run.board_region.mask is not None:
                    mask_bgr = cv2.cvtColor(run.board_region.mask, cv2.COLOR_GRAY2BGR)
                    archive.writestr("images/03_board_mask.png", encode_image(mask_bgr))
            if include_crops:
                for crop in run.crops:
                    extension = Path(crop.filename).suffix or ".png"
                    archive.writestr(
                        f"crops/{crop.filename}", encode_image(crop.image, extension)
                    )
    except (OSError, TypeError, ValueError) as exc:
        raise ExportError(f"Could not export ZIP to {destination}: {exc}") from exc
    return destination


def render_annotations(run: PipelineRun) -> np.ndarray:
    """Render board and component boxes without mutating the run image."""

    canvas = run.final_image.copy()
    board_points = np.asarray(run.board_region.polygon, dtype=np.int32).reshape(-1, 1, 2)
    if len(board_points) >= 3:
        cv2.polylines(canvas, [board_points], True, (0, 220, 255), 2, cv2.LINE_AA)
    for detection in run.detections:
        x1, y1, x2, y2 = detection.bbox.to_int()
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 220, 60), 2)
        caption = f"{detection.label} {detection.confidence:.2f}"
        text_y = max(14, y1 - 5)
        cv2.putText(
            canvas,
            caption,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (60, 220, 60),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _manifest_json(run: PipelineRun) -> str:
    return json.dumps(
        run.to_dict(),
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
