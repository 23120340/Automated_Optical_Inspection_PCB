"""Pass 2: detect leads and pads *inside* each component box.

Pass 1 finds component bodies on the whole board. That is the only thing the
public datasets can teach it -- ``pads`` scores 0.072 recall there, because only
30 of 670 training images contain the class at all. So step 5.5 currently
*derives* the joint ROIs from the box plus the class topology.

Derivation is geometry, and geometry does not know what is actually on the
board. It cannot tell which axis of a square-ish part holds the terminals, it
places ROIs on the lead-free sides of a SOT-23, and it lets one component's ROI
reach onto its neighbour. Every one of those was measured on a real board.

A second detector run *inside* the component crop answers the same question
from pixels instead. This module is the plumbing for that: crop with margin,
run whatever detector is configured, and translate what it finds back into the
board's coordinate system. It deliberately owns no model. With none configured
it returns nothing and step 5.5 keeps its derived geometry unchanged, so the
pipeline behaves exactly as before until a model exists.

The detections it returns carry the ``pads``/``pins`` labels that
:mod:`aoi_pipeline.solder.leads` already knows how to fuse, per terminal rather than
per component.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from ..config import LeadDetectionConfig
from ..imaging.image_io import ensure_bgr
from ..models import BoundingBox, Detection

__all__ = [
    "LeadDetector",
    "component_crop_window",
    "detect_leads_in_components",
    "to_board_coordinates",
]


class LeadDetector(Protocol):
    """Anything that can find leads in one component crop.

    Deliberately structural: the Ultralytics wrapper already satisfies it, and
    so does a stub in a test, without either having to know about this module.

    ``detect_batch`` is optional. A detector that offers it is handed several
    crops at once; one that does not is called per crop, which costs about 1.47x
    more but works. Both paths must return one sequence per input crop, in the
    same order -- pass 2 matches results to components positionally, and a
    detector that silently drops an empty result would shift every lead after it
    onto the wrong component.
    """

    def detect(self, image: np.ndarray) -> Sequence[Detection]:  # pragma: no cover
        ...


def _detect_in_batches(
    detector: LeadDetector,
    crops: list[np.ndarray],
    batch_size: int,
) -> list[Sequence[Detection] | None]:
    """Results for each crop, ``None`` where the detector failed.

    One unreadable part is not a bad board: pass 1 already produced a usable
    box and step 5.5 can still derive that component's geometry. So a failure
    is recorded against its own crops and the rest go on.
    """

    batched = getattr(detector, "detect_batch", None)
    results: list[Sequence[Detection] | None] = []
    for start in range(0, len(crops), max(1, batch_size)):
        chunk = crops[start:start + max(1, batch_size)]
        if batched is not None:
            try:
                found = list(batched(chunk))
            except Exception:  # noqa: BLE001
                found = [None] * len(chunk)
            if len(found) != len(chunk):
                # A detector that returns a different number of results has
                # broken the positional contract. Guessing which crop each
                # result belongs to would put leads on the wrong component, so
                # fall back to one-at-a-time rather than align by luck.
                found = [None] * len(chunk)
                batched = None
        if batched is None:
            found = []
            for crop in chunk:
                try:
                    found.append(detector.detect(crop))
                except Exception:  # noqa: BLE001
                    found.append(None)
        results.extend(found)
    return results


def component_crop_window(
    box: BoundingBox,
    image_width: int,
    image_height: int,
    config: LeadDetectionConfig,
) -> tuple[int, int, int, int]:
    """The crop pass 2 should look at: the body box plus room for its fillets.

    The margin is a fraction of the box's own longer side, so it holds across an
    0402 chip and a connector. Without it the crop stops at the component
    silhouette and the fillet -- the whole point of the exercise -- lies outside
    the frame the detector is given.
    """

    x1, y1, x2, y2 = box.clamp(image_width, image_height).to_int()
    margin = int(round(config.crop_margin_ratio * max(x2 - x1, y2 - y1)))
    margin = max(margin, config.crop_margin_min_px)
    return (
        max(0, x1 - margin),
        max(0, y1 - margin),
        min(image_width, x2 + margin),
        min(image_height, y2 + margin),
    )


def to_board_coordinates(
    detection: Detection,
    window: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> BoundingBox:
    """Translate a crop-local box onto the board.

    The whole coordinate question is this one addition. The crop is an
    axis-aligned window cut straight out of the analysis frame with no resize,
    so the mapping is a translation by the window's origin and nothing else.
    Any scaling the detector does internally is its own business -- it reports
    back in the pixel space it was handed.
    """

    offset_x, offset_y = window[0], window[1]
    box = detection.bbox
    return BoundingBox(
        float(box.x1) + offset_x,
        float(box.y1) + offset_y,
        float(box.x2) + offset_x,
        float(box.y2) + offset_y,
    ).clamp(image_width, image_height)


def detect_leads_in_components(
    image: np.ndarray,
    detections: Sequence[Detection],
    detector: LeadDetector | None,
    config: LeadDetectionConfig | None = None,
) -> list[Detection]:
    """Run pass 2 over every component and return leads in board coordinates.

    Returns an empty list when there is no detector or the stage is switched
    off, which is what keeps this a no-op until a model exists.

    A failure on one component is recorded on that component and skipped rather
    than taken as a failure of the board: pass 1 already produced usable boxes,
    and step 5.5 can still derive geometry for the ones pass 2 could not read.
    """

    config = config or LeadDetectionConfig()
    if detector is None or not config.enabled or not detections:
        return []

    frame = ensure_bgr(image)
    height, width = frame.shape[:2]
    leads: list[Detection] = []

    # Cut every crop first, then run them in batches. The crops are views'
    # worth of pixels -- ~48x48 each, a few MB for a whole board -- so holding
    # them briefly costs far less than the per-call setup they save.
    windows: list[tuple[int, int, int, int]] = []
    crops: list[np.ndarray] = []
    kept: list[Detection] = []
    for detection in detections:
        window = component_crop_window(detection.bbox, width, height, config)
        x1, y1, x2, y2 = window
        if (x2 - x1) < config.min_crop_px or (y2 - y1) < config.min_crop_px:
            continue
        windows.append(window)
        crops.append(np.ascontiguousarray(frame[y1:y2, x1:x2]))
        kept.append(detection)

    if not crops:
        return []

    outcomes = _detect_in_batches(detector, crops, config.batch_size)
    del crops

    for detection, window, found in zip(kept, windows, outcomes):
        if found is None:
            continue
        for index, lead in enumerate(found or []):
            if float(lead.confidence) < config.confidence:
                continue
            bbox = to_board_coordinates(lead, window, width, height)
            if bbox.width < config.min_lead_px or bbox.height < config.min_lead_px:
                continue
            leads.append(
                Detection(
                    label=str(lead.label),
                    confidence=float(lead.confidence),
                    bbox=bbox,
                    class_id=lead.class_id,
                    source="lead_pass2",
                    detection_id=f"{detection.detection_id}_lead{index:03d}",
                    metadata={
                        "parent_detection_id": detection.detection_id,
                        "parent_label": detection.label,
                        "crop_window": list(window),
                        "bbox_in_crop": lead.bbox.to_dict(),
                    },
                )
            )
    return leads
