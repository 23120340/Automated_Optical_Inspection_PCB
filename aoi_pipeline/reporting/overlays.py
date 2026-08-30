"""Overlay renderers shared by the ZIP export, the CLI and the app.

Every one of these draws onto a copy: an overlay that mutated the analysis
image would corrupt the very pixels the next stage measures.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from ..imaging.image_io import ensure_bgr
from ..models import PipelineRun, SolderJoint

__all__ = [
    "SOLDER_OVERLAY_COLORS",
    "render_annotations",
    "render_solder_overlay",
    "render_verdict_overlay",
]


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


SOLDER_OVERLAY_COLORS = {
    "joint": (0, 200, 255),  # BGR amber
    "body": (255, 170, 0),   # BGR blue
}


def render_solder_overlay(
    image: np.ndarray,
    joints: Sequence[SolderJoint],
    draw_body: bool = True,
) -> np.ndarray:
    """Draw derived ROIs so the geometry can be checked by eye before labelling."""

    canvas = ensure_bgr(image).copy()
    for joint in joints:
        if joint.kind == "body" and not draw_body:
            continue
        x1, y1, x2, y2 = joint.bbox.to_int()
        color = SOLDER_OVERLAY_COLORS.get(joint.kind, (200, 200, 200))
        thickness = 1 if joint.kind == "body" else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    return canvas


VERDICT_OVERLAY_COLORS = {
    "accept": (80, 200, 80),    # BGR green
    "review": (0, 170, 255),    # BGR orange
    "reject": (40, 40, 230),    # BGR red
}


def render_verdict_overlay(run: PipelineRun, draw_component: bool = False) -> np.ndarray:
    """Colour every graded ROI by its step-6.2 decision.

    Component-scope ROIs are off by default: they enclose their own joints, so
    drawing both leaves the joint verdicts buried inside a box of a different
    colour.
    """

    canvas = ensure_bgr(run.final_image).copy()
    for verdict in run.solder_verdicts:
        if verdict.scope == "component" and not draw_component:
            continue
        box = verdict.metadata.get("bbox")
        if box is None:
            continue
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        color = VERDICT_OVERLAY_COLORS.get(verdict.decision, (200, 200, 200))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        if verdict.decision != "accept":
            cv2.putText(
                canvas,
                verdict.label,
                (x1, max(11, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                color,
                1,
                cv2.LINE_AA,
            )
    return canvas
