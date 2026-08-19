"""JSON and ZIP exporters for a completed steps 0-6.1 run."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np

from ..core.exceptions import ExportError
from ..core.image_io import encode_image
from ..core.models import PipelineRun
from .overlays import render_annotations, render_solder_overlay


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
    include_solder_crops: bool = True,
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
                if run.solder_crops:
                    archive.writestr(
                        "images/05_solder_rois.png",
                        encode_image(
                            render_solder_overlay(
                                run.final_image,
                                [crop.joint for crop in run.solder_crops],
                            )
                        ),
                    )
                if run.board_region.mask is not None:
                    mask_bgr = cv2.cvtColor(run.board_region.mask, cv2.COLOR_GRAY2BGR)
                    archive.writestr("images/03_board_mask.png", encode_image(mask_bgr))
            if include_crops:
                for crop in run.crops:
                    extension = Path(crop.filename).suffix or ".png"
                    archive.writestr(
                        f"crops/{crop.filename}", encode_image(crop.image, extension)
                    )
            if include_solder_crops and run.solder_crops:
                for crop in run.solder_crops:
                    extension = Path(crop.filename).suffix or ".png"
                    folder = "body_views" if crop.joint.kind == "body" else "joints"
                    archive.writestr(
                        f"solder_joints/{folder}/{crop.filename}",
                        encode_image(crop.image, extension),
                    )
                archive.writestr(
                    "solder_joints/solder_joints.csv", solder_joints_csv(run)
                )
            if run.fusion is not None and getattr(run.fusion, "used_cad", False):
                archive.writestr("cad/cad_findings.csv", cad_findings_csv(run))
                archive.writestr(
                    "cad/registration.json",
                    json.dumps(
                        run.fusion.to_dict(), ensure_ascii=False, indent=2,
                        default=_json_default,
                    ),
                )
    except (OSError, TypeError, ValueError) as exc:
        raise ExportError(f"Could not export ZIP to {destination}: {exc}") from exc
    return destination


SOLDER_CSV_COLUMNS = (
    "joint_id",
    "detection_id",
    "label",
    "kind",
    "position",
    "pin_index",
    "terminal_geometry",
    "angle",
    "x1",
    "y1",
    "x2",
    "y2",
    "detector_confidence",
    # Provenance, empty until a CAD board is loaded. ``source`` says whether the
    # ROI came from the detector geometry, registered CAD lands, or both
    # agreeing, which is what lets a training set be filtered by trust.
    "source",
    "designator",
    "pin",
    "net",
    "filename",
    "defect_class",
)

CAD_FINDING_COLUMNS = (
    "kind",
    "severity",
    "designator",
    "detection_id",
    "expected_class",
    "observed_class",
    "shift_mm",
    "x1",
    "y1",
    "x2",
    "y2",
    "message",
)


def solder_joints_csv(run: PipelineRun) -> str:
    """One row per derived ROI, with an empty ``defect_class`` to fill in.

    This is the labelling sheet for step 6.2: the geometry is already resolved,
    so annotation is a per-row verdict rather than a boxing job.
    """

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(SOLDER_CSV_COLUMNS)
    for crop in run.solder_crops:
        joint = crop.joint
        folder = "body_views" if joint.kind == "body" else "joints"
        writer.writerow(
            [
                joint.joint_id,
                joint.detection_id,
                joint.label,
                joint.kind,
                joint.position,
                "" if joint.pin_index is None else joint.pin_index,
                joint.terminal_geometry,
                f"{joint.angle:.2f}",
                f"{joint.bbox.x1:.2f}",
                f"{joint.bbox.y1:.2f}",
                f"{joint.bbox.x2:.2f}",
                f"{joint.bbox.y2:.2f}",
                f"{float(joint.metadata.get('detector_confidence', 0.0)):.4f}",
                joint.source,
                joint.designator or "",
                joint.pin or "",
                joint.net or "",
                f"{folder}/{crop.filename}",
                "",
            ]
        )
    return buffer.getvalue()


def cad_findings_csv(run: PipelineRun) -> str:
    """Board-level disagreements between CAD and what the camera saw.

    Missing, shifted and unexpected components are defects in their own right,
    found by comparison rather than by any model, so they ship next to the ROI
    table instead of inside it.
    """

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CAD_FINDING_COLUMNS)
    findings = getattr(run.fusion, "findings", None) or []
    for finding in findings:
        bbox = finding.bbox
        writer.writerow(
            [
                finding.kind,
                finding.severity,
                finding.designator or "",
                finding.detection_id or "",
                finding.expected_class or "",
                finding.observed_class or "",
                "" if finding.shift_mm is None else f"{finding.shift_mm:.3f}",
                "" if bbox is None else f"{bbox.x1:.2f}",
                "" if bbox is None else f"{bbox.y1:.2f}",
                "" if bbox is None else f"{bbox.x2:.2f}",
                "" if bbox is None else f"{bbox.y2:.2f}",
                finding.message,
            ]
        )
    return buffer.getvalue()


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
