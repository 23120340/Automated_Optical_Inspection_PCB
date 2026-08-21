"""Step 5.5: derive solder-joint inspection ROIs from component detections.

The component detector is trained on datasets that label component *bodies*.
A solder fillet lies outside that silhouette, so no bounding box the detector
produces -- and no amount of retraining it on the same labels -- will contain
the joint. Asking a detector to find the joints directly is not an option
either: the shipped model scores ``pads`` at 0.0 recall on both val and test.

So the joints are *derived*. Given a box and the class's terminal topology, the
location of every joint is a geometry problem, which is how window-based AOI
has always placed its inspection ROIs. This module is the fallback for boards
with no CAD data; when Gerber/pick-and-place is available, projecting the real
land coordinates through the step-2 homography is strictly better.

Coordinates follow the rest of the pipeline: pixels of the preprocessed/aligned
analysis image, not the raw input frame.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .config import SolderJointConfig, terminal_geometry
from .image_io import encode_image, ensure_bgr, letterbox_normalize
from .models import BoundingBox, Detection, SolderJoint, SolderJointCrop
from .grading.features import segment_solder

__all__ = [
    "ComponentFrame",
    "deconflict_joint_rois",
    "refine_joint_to_metal",
    "SolderJointCropper",
    "derive_solder_joints",
    "estimate_component_angle",
]


@dataclass(frozen=True, slots=True)
class _LocalRect:
    """A ROI in the component-local frame; local +x runs along the long axis."""

    cx: float
    cy: float
    width: float
    height: float
    position: str
    kind: str = "joint"
    pin_index: int | None = None


@dataclass(frozen=True, slots=True)
class ComponentFrame:
    """The component's own coordinate frame inside the analysis image.

    ``angle`` rotates local +x, which runs along ``length``, off the image x
    axis. CAD fusion builds this frame from registered land coordinates instead
    of from the detector box, which is why it is part of the public surface.
    """

    center_x: float
    center_y: float
    angle: float
    length: float
    span: float


def derive_solder_joints(
    detection: Detection,
    image_width: int,
    image_height: int,
    config: SolderJointConfig | None = None,
    image: np.ndarray | None = None,
    frame: ComponentFrame | None = None,
    geometry: str | None = None,
) -> list[SolderJoint]:
    """Return the inspection ROIs for one detection.

    ``image`` is optional. Without it the perimeter-band energy filter, the
    per-pin split and orientation estimation are all skipped and every
    candidate ROI is kept, because dropping one on no evidence would silently
    lose joints.

    ``frame`` and ``geometry`` let a caller that knows better override what the
    box alone can say. CAD fusion uses them to keep this exact ROI geometry
    while anchoring it on a registered placement and a real pad count.
    """

    config = config or SolderJointConfig()
    box = detection.bbox.clamp(image_width, image_height)
    if box.width <= 0 or box.height <= 0:
        return []

    geometry = geometry or terminal_geometry(detection.label)
    frame = frame or _component_frame(box, config, image)

    if geometry == "two_terminal":
        rects = _resolve_two_terminal_rects(
            frame, config, image, image_width, image_height
        )
    elif geometry == "pad_only":
        rects = _pad_only_rects(frame, config)
    else:
        rects = _multi_pin_rects(frame, config)

    if geometry == "multi_pin" and image is not None:
        rects = _filter_bands_by_energy(
            rects, image, frame, image_width, image_height, config
        )
        if config.split_pins:
            rects = _split_bands_into_pins(
                rects, image, frame, image_width, image_height, config
            )

    if config.include_body_view and rects:
        rects = [*rects, _body_rect(rects, frame)]

    joints: list[SolderJoint] = []
    for index, rect in enumerate(rects):
        bbox = _local_rect_to_bbox(rect, frame, image_width, image_height)
        if bbox is None:
            continue
        if bbox.width < config.min_roi_pixels or bbox.height < config.min_roi_pixels:
            continue
        joints.append(
            SolderJoint(
                detection_id=detection.detection_id,
                joint_id=f"{detection.detection_id}_{rect.kind}{index:02d}",
                label=detection.label,
                kind=rect.kind,
                bbox=bbox,
                terminal_geometry=geometry,
                position=rect.position,
                angle=float(frame.angle),
                pin_index=rect.pin_index,
                metadata={
                    "source_bbox": box.to_dict(),
                    "local_size": {
                        "width": float(rect.width),
                        "height": float(rect.height),
                    },
                    "detector_confidence": float(detection.confidence),
                },
            )
        )
    return joints


# --------------------------------------------------------------------------- #
# Component frame
# --------------------------------------------------------------------------- #


def _component_frame(
    box: BoundingBox,
    config: SolderJointConfig,
    image: np.ndarray | None,
) -> ComponentFrame:
    """Pick the frame whose +x axis runs along the component's long side.

    The axis-aligned default already covers the usual 0/90 degree placement.
    Orientation estimation only replaces it when the measured axis is both
    plausible and close enough to the box to be believable.
    """

    center_x = (box.x1 + box.x2) / 2.0
    center_y = (box.y1 + box.y2) / 2.0
    length = max(box.width, box.height)
    span = min(box.width, box.height)
    fallback = ComponentFrame(
        center_x,
        center_y,
        0.0 if box.width >= box.height else 90.0,
        length,
        span,
    )
    if not config.estimate_orientation or image is None:
        return fallback

    measured = _measure_min_area_rect(image, box)
    if measured is None:
        return fallback
    (rect_cx, rect_cy), rect_angle, rect_length, rect_span = measured

    diagonal = math.hypot(box.width, box.height)
    if not 0.4 * length <= rect_length <= 1.15 * diagonal:
        return fallback
    if rect_span < 3.0:
        return fallback
    deviation = _angle_difference(rect_angle, fallback.angle)
    if abs(deviation) < 5.0 or abs(deviation) > config.orientation_max_angle:
        return fallback
    return ComponentFrame(rect_cx, rect_cy, rect_angle, rect_length, rect_span)


def _measure_min_area_rect(
    image: np.ndarray, box: BoundingBox
) -> tuple[tuple[float, float], float, float, float] | None:
    """Fit a rotated rect to the component body inside ``box``."""

    x1, y1, x2, y2 = box.to_int()
    patch = ensure_bgr(image)[y1:y2, x1:x2]
    if patch.size == 0 or min(patch.shape[:2]) < 8:
        return None
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # The body may be brighter or darker than the board; keep whichever
    # polarity puts mass at the patch center.
    height, width = mask.shape[:2]
    if mask[height // 2, width // 2] == 0:
        mask = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 0.20 * width * height:
        return None
    (rect_cx, rect_cy), (rect_width, rect_height), rect_angle = cv2.minAreaRect(contour)
    # ``minAreaRect`` measures +x along ``rect_width``. Rotating by +90 maps
    # local +x onto the ``rect_height`` direction, matching ``boxPoints``.
    if rect_width >= rect_height:
        angle, roi_length, roi_span = rect_angle, rect_width, rect_height
    else:
        angle, roi_length, roi_span = rect_angle + 90.0, rect_height, rect_width
    return (
        (rect_cx + x1, rect_cy + y1),
        _wrap_angle(angle),
        float(roi_length),
        float(roi_span),
    )


def estimate_component_angle(
    image: np.ndarray,
    box: BoundingBox,
    max_angle: float = 30.0,
) -> float:
    """Best-effort long-axis angle of the component in ``box``, in degrees.

    Falls back to the box's own axis (0 or 90) whenever the estimate is not
    trustworthy. A wrong angle mis-places every ROI derived from it, so an
    abstention is cheaper than a guess.
    """

    fallback = 0.0 if box.width >= box.height else 90.0
    measured = _measure_min_area_rect(image, box)
    if measured is None:
        return fallback
    _, angle, _, _ = measured
    deviation = _angle_difference(angle, fallback)
    if abs(deviation) < 5.0 or abs(deviation) > max_angle:
        return fallback
    return angle


def _wrap_angle(angle: float) -> float:
    """Normalize an undirected axis angle into ``[-90, 90)`` degrees."""

    return ((float(angle) + 90.0) % 180.0) - 90.0


def _angle_difference(angle: float, reference: float) -> float:
    return _wrap_angle(angle - reference)


# --------------------------------------------------------------------------- #
# Local-frame geometry
# --------------------------------------------------------------------------- #


def _two_terminal_rects(
    frame: ComponentFrame,
    config: SolderJointConfig,
    *,
    along_long_axis: bool = True,
) -> list[_LocalRect]:
    """The two terminal ROIs, on the long axis by default.

    ``along_long_axis=False`` places them on the short axis instead. A box only
    names its long axis when it is actually longer one way; for a square-ish
    part the caller has to decide from something other than the box, and then
    needs a way to say so.
    """

    length = frame.length if along_long_axis else frame.span
    span = frame.span if along_long_axis else frame.length
    inner = config.terminal_inner_ratio * length
    outer = config.terminal_outer_ratio * length
    side = config.terminal_side_ratio * span
    roi_length = inner + outer
    roi_span = span + 2.0 * side
    offset = length / 2.0 + (outer - inner) / 2.0
    if along_long_axis:
        return [
            _LocalRect(-offset, 0.0, roi_length, roi_span, "terminal_a"),
            _LocalRect(offset, 0.0, roi_length, roi_span, "terminal_b"),
        ]
    return [
        _LocalRect(0.0, -offset, roi_span, roi_length, "terminal_a"),
        _LocalRect(0.0, offset, roi_span, roi_length, "terminal_b"),
    ]


def _outer_strip(
    frame: ComponentFrame, config: SolderJointConfig, *, along_long_axis: bool
) -> list[_LocalRect]:
    """The two bands just *outside* the body on one candidate terminal axis.

    Only outside: the land a fillet sits on sticks out past the component
    silhouette on the terminal axis, and on the other axis there is bare
    laminate or silkscreen. Inside the body both axes can look metallic -- the
    top of an electrolytic can is metal all over -- so the inside says nothing.
    """

    length = frame.length if along_long_axis else frame.span
    span = frame.span if along_long_axis else frame.length
    outer = config.terminal_outer_ratio * length
    side = config.terminal_side_ratio * span
    offset = length / 2.0 + outer / 2.0
    width, height = outer, span + 2.0 * side
    if along_long_axis:
        return [
            _LocalRect(-offset, 0.0, width, height, "probe_a"),
            _LocalRect(offset, 0.0, width, height, "probe_b"),
        ]
    return [
        _LocalRect(0.0, -offset, height, width, "probe_a"),
        _LocalRect(0.0, offset, height, width, "probe_b"),
    ]


def _axis_metal_score(
    frame: ComponentFrame,
    config: SolderJointConfig,
    image: np.ndarray,
    image_width: int,
    image_height: int,
    *,
    along_long_axis: bool,
) -> float:
    """Mean metal fraction in the two outward bands of one candidate axis."""

    scores: list[float] = []
    for rect in _outer_strip(frame, config, along_long_axis=along_long_axis):
        bbox = _local_rect_to_bbox(rect, frame, image_width, image_height)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox.to_int()
        patch = image[y1:y2, x1:x2]
        if patch.size == 0 or min(patch.shape[:2]) < 3:
            continue
        mask = segment_solder(patch, saturation_max=config.saturation_max)
        if mask.size == 0:
            continue
        scores.append(float(np.count_nonzero(mask)) / float(mask.size))
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def _resolve_two_terminal_rects(
    frame: ComponentFrame,
    config: SolderJointConfig,
    image: np.ndarray | None,
    image_width: int,
    image_height: int,
) -> list[_LocalRect]:
    """Place the two terminal ROIs, deciding the axis when the box cannot.

    A box whose sides are within ``terminal_axis_min_aspect`` of each other
    does not know where its terminals are: at 40x40 against 40x41 the ROIs flip
    by 90 degrees on one pixel. So the axis is measured from the metal outside
    the body, and when that measurement is not decisive both axes are emitted.
    Four reviewable ROIs cost an operator seconds; two ROIs on the wrong two
    sides inspect bare laminate and pass every defect on the real ones.
    """

    span = max(frame.span, 1e-6)
    if frame.length / span >= config.terminal_axis_min_aspect:
        return _two_terminal_rects(frame, config, along_long_axis=True)

    if image is not None:
        long_score = _axis_metal_score(
            frame, config, image, image_width, image_height, along_long_axis=True
        )
        short_score = _axis_metal_score(
            frame, config, image, image_width, image_height, along_long_axis=False
        )
        margin = config.terminal_axis_decision_margin
        if long_score > margin * short_score:
            return _two_terminal_rects(frame, config, along_long_axis=True)
        if short_score > margin * long_score:
            return _two_terminal_rects(frame, config, along_long_axis=False)

    # Undecidable: keep both, but name them so a reviewer can see that the pair
    # on one axis is the alternative hypothesis, not four separate terminals.
    return [
        *_two_terminal_rects(frame, config, along_long_axis=True),
        *(
            replace(rect, position=f"{rect.position}_cross")
            for rect in _two_terminal_rects(frame, config, along_long_axis=False)
        ),
    ]


def _pad_only_rects(frame: ComponentFrame, config: SolderJointConfig) -> list[_LocalRect]:
    margin = config.pad_margin_ratio * frame.span
    return [
        _LocalRect(
            0.0,
            0.0,
            frame.length + 2.0 * margin,
            frame.span + 2.0 * margin,
            "pad",
        )
    ]


def _multi_pin_rects(frame: ComponentFrame, config: SolderJointConfig) -> list[_LocalRect]:
    inner = config.lead_inner_ratio * frame.span
    outer = config.lead_outer_ratio * frame.span
    depth = inner + outer
    # Bands run past the body corners so corner pins are not clipped.
    offset_x = frame.length / 2.0 + (outer - inner) / 2.0
    offset_y = frame.span / 2.0 + (outer - inner) / 2.0
    return [
        _LocalRect(-offset_x, 0.0, depth, frame.span + 2.0 * outer, "lead_left"),
        _LocalRect(offset_x, 0.0, depth, frame.span + 2.0 * outer, "lead_right"),
        _LocalRect(0.0, -offset_y, frame.length + 2.0 * outer, depth, "lead_top"),
        _LocalRect(0.0, offset_y, frame.length + 2.0 * outer, depth, "lead_bottom"),
    ]


def _body_rect(rects: Sequence[_LocalRect], frame: ComponentFrame) -> _LocalRect:
    """Smallest local rect covering the body and every derived joint ROI."""

    half_length = frame.length / 2.0
    half_span = frame.span / 2.0
    left = min([-half_length, *(rect.cx - rect.width / 2.0 for rect in rects)])
    right = max([half_length, *(rect.cx + rect.width / 2.0 for rect in rects)])
    top = min([-half_span, *(rect.cy - rect.height / 2.0 for rect in rects)])
    bottom = max([half_span, *(rect.cy + rect.height / 2.0 for rect in rects)])
    return _LocalRect(
        (left + right) / 2.0,
        (top + bottom) / 2.0,
        right - left,
        bottom - top,
        "body",
        kind="body",
    )


def _to_image(points: np.ndarray, frame: ComponentFrame) -> np.ndarray:
    """Map local-frame points to image pixels.

    Uses the same rotation convention as ``cv2.boxPoints`` in the y-down image
    frame, so a frame angle taken from ``minAreaRect`` stays consistent.
    """

    mapped = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if abs(frame.angle) > 1e-9:
        radians = math.radians(frame.angle)
        cos_a, sin_a = math.cos(radians), math.sin(radians)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
        mapped = mapped @ rotation.T
    mapped[:, 0] += frame.center_x
    mapped[:, 1] += frame.center_y
    return mapped


def _local_rect_corners(rect: _LocalRect, frame: ComponentFrame) -> np.ndarray:
    half_width = rect.width / 2.0
    half_height = rect.height / 2.0
    corners = np.array(
        [
            [rect.cx - half_width, rect.cy - half_height],
            [rect.cx + half_width, rect.cy - half_height],
            [rect.cx + half_width, rect.cy + half_height],
            [rect.cx - half_width, rect.cy + half_height],
        ],
        dtype=np.float64,
    )
    return _to_image(corners, frame)


def _local_rect_to_bbox(
    rect: _LocalRect,
    frame: ComponentFrame,
    image_width: int,
    image_height: int,
) -> BoundingBox | None:
    corners = _local_rect_corners(rect, frame)
    bbox = BoundingBox(
        float(corners[:, 0].min()),
        float(corners[:, 1].min()),
        float(corners[:, 0].max()),
        float(corners[:, 1].max()),
    ).clamp(image_width, image_height)
    if bbox.width <= 0 or bbox.height <= 0:
        return None
    return bbox


# --------------------------------------------------------------------------- #
# Image-driven refinement
# --------------------------------------------------------------------------- #


def _band_patch(
    rect: _LocalRect,
    image: np.ndarray,
    frame: ComponentFrame,
    image_width: int,
    image_height: int,
) -> np.ndarray | None:
    bbox = _local_rect_to_bbox(rect, frame, image_width, image_height)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox.to_int()
    patch = image[y1:y2, x1:x2]
    return patch if patch.size else None


def _filter_bands_by_energy(
    rects: Sequence[_LocalRect],
    image: np.ndarray,
    frame: ComponentFrame,
    image_width: int,
    image_height: int,
    config: SolderJointConfig,
) -> list[_LocalRect]:
    """Drop perimeter bands with no lead metal, e.g. the pin-free sides of a SOIC.

    The threshold is relative to the strongest band of the same component, so it
    adapts to exposure instead of assuming an absolute brightness.
    """

    if config.lead_band_energy_ratio is None:
        return list(rects)
    bgr = ensure_bgr(image)
    energies: list[float] = []
    for rect in rects:
        patch = _band_patch(rect, bgr, frame, image_width, image_height)
        if patch is None:
            energies.append(0.0)
            continue
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        energies.append(float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_32F)))))
    peak = max(energies, default=0.0)
    if peak <= 1e-6:
        return list(rects)
    threshold = config.lead_band_energy_ratio * peak
    kept = [rect for rect, energy in zip(rects, energies) if energy >= threshold]
    # Never return nothing: a component with no surviving band would silently
    # disappear from inspection.
    return kept or list(rects)


def _split_bands_into_pins(
    rects: Sequence[_LocalRect],
    image: np.ndarray,
    frame: ComponentFrame,
    image_width: int,
    image_height: int,
    config: SolderJointConfig,
) -> list[_LocalRect]:
    """Cut each lead band into one ROI per pin, or keep the band if unsure.

    Only attempted on axis-aligned frames: for any other angle the patch is the
    band's enclosing box rather than the band, so the 1-D profile would mix in
    board pixels from the corners.
    """

    if _axis_offset(frame.angle) > 1.0:
        return list(rects)
    bgr = ensure_bgr(image)
    result: list[_LocalRect] = []
    for rect in rects:
        split = _split_one_band(rect, bgr, frame, image_width, image_height, config)
        result.extend(split if split else [rect])
    return result


def _split_one_band(
    rect: _LocalRect,
    image: np.ndarray,
    frame: ComponentFrame,
    image_width: int,
    image_height: int,
    config: SolderJointConfig,
) -> list[_LocalRect] | None:
    patch = _band_patch(rect, image, frame, image_width, image_height)
    if patch is None:
        return None

    # Take the band direction from which edge it was built on, not from its
    # aspect ratio: on a small part the band depth can exceed its length.
    if rect.position in {"lead_left", "lead_right"}:
        along_local_y = True
    elif rect.position in {"lead_top", "lead_bottom"}:
        along_local_y = False
    else:
        return None
    extent = rect.height if along_local_y else rect.width
    start_scalar = (rect.cy if along_local_y else rect.cx) - extent / 2.0
    if along_local_y:
        endpoints = np.array(
            [[rect.cx, start_scalar], [rect.cx, start_scalar + extent]], dtype=np.float64
        )
    else:
        endpoints = np.array(
            [[start_scalar, rect.cy], [start_scalar + extent, rect.cy]], dtype=np.float64
        )
    mapped = _to_image(endpoints, frame)
    delta = mapped[1] - mapped[0]
    axis = 1 if abs(delta[1]) >= abs(delta[0]) else 0
    flipped = delta[axis] < 0.0

    # The band deliberately runs past the body corners so corner pins are not
    # clipped, but no lead lives out there. Ignoring the overhang keeps the
    # detrended profile from turning the bare-board margin into a fake pin.
    body_extent = frame.span if along_local_y else frame.length
    margin = max(0.0, (extent - body_extent) / (2.0 * extent)) if extent > 0 else 0.0
    runs = _find_pin_runs(
        patch, vertical=axis == 1, config=config, margin_fraction=margin
    )
    if runs is None:
        return None

    # ``axis`` is an image axis (0=x, 1=y); NumPy indexes rows first.
    patch_length = patch.shape[0] if axis == 1 else patch.shape[1]
    if patch_length <= 0:
        return None
    centers = [(start + stop) / 2.0 for start, stop in runs]
    pitch = float(np.median(np.diff(centers))) if len(centers) > 1 else 0.0
    growth = config.pin_padding_ratio * pitch

    rects: list[_LocalRect] = []
    for pin_index, (start, stop) in enumerate(runs):
        fraction = (start + stop) / 2.0 / patch_length
        if flipped:
            fraction = 1.0 - fraction
        position = start_scalar + fraction * extent
        size = ((stop - start) + 2.0 * growth) / patch_length * extent
        rects.append(
            _LocalRect(
                rect.cx if along_local_y else position,
                position if along_local_y else rect.cy,
                rect.width if along_local_y else size,
                size if along_local_y else rect.height,
                f"{rect.position}_pin{pin_index:02d}",
                pin_index=pin_index,
            )
        )
    # Keep pin order stable along the band regardless of the image-space flip.
    if flipped:
        rects = [
            _LocalRect(
                item.cx,
                item.cy,
                item.width,
                item.height,
                f"{rect.position}_pin{index:02d}",
                pin_index=index,
            )
            for index, item in enumerate(reversed(rects))
        ]
    return rects


def _find_pin_runs(
    patch: np.ndarray,
    vertical: bool,
    config: SolderJointConfig,
    margin_fraction: float = 0.0,
) -> list[tuple[int, int]] | None:
    """Locate individual leads along a band, or ``None`` if the split is unsafe.

    Leads are bright and evenly pitched. The profile is detrended against a wide
    moving average so uneven illumination across the band does not bias it, and
    the result is accepted only when the run count and pitch regularity look
    like a real lead row -- a bad split would scatter one pin's defect across
    two training samples.
    """

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    profile = gray.mean(axis=1) if vertical else gray.mean(axis=0)
    if profile.size < 3 * config.min_pins_per_band:
        return None
    window = max(3, (profile.size // max(config.min_pins_per_band, 1)) | 1)
    baseline = cv2.blur(
        profile.reshape(-1, 1), (1, window), borderType=cv2.BORDER_REPLICATE
    ).ravel()
    above = (profile - baseline) > 0.0

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(above):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, int(above.size)))

    minimum_width = max(2, profile.size // (config.max_pins_per_band * 2))
    runs = [run for run in runs if run[1] - run[0] >= minimum_width]
    if margin_fraction > 0.0:
        low = margin_fraction * profile.size
        high = (1.0 - margin_fraction) * profile.size
        runs = [run for run in runs if low <= (run[0] + run[1]) / 2.0 <= high]
    if not config.min_pins_per_band <= len(runs) <= config.max_pins_per_band:
        return None
    centers = np.array([(start + stop) / 2.0 for start, stop in runs])
    pitches = np.diff(centers)
    if pitches.size == 0 or float(pitches.mean()) <= 0.0:
        return None
    if float(pitches.std() / pitches.mean()) > 0.35:
        return None
    return runs


def _axis_offset(angle: float) -> float:
    """Degrees between ``angle`` and the nearest image axis."""

    remainder = abs(float(angle)) % 90.0
    return min(remainder, 90.0 - remainder)


# --------------------------------------------------------------------------- #
# Crop extraction
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Neighbour de-confliction
# --------------------------------------------------------------------------- #


def _area(box: BoundingBox) -> float:
    return max(0.0, box.width) * max(0.0, box.height)


def _intersects(a: BoundingBox, b: BoundingBox) -> bool:
    return a.x1 < b.x2 and b.x1 < a.x2 and a.y1 < b.y2 and b.y1 < a.y2


def _cut_out_of(box: BoundingBox, obstacle: BoundingBox) -> BoundingBox:
    """Trim ``box`` back out of ``obstacle``, keeping as much of it as possible.

    A rectangle minus a rectangle is not a rectangle, so this keeps the largest
    single remaining slab rather than inventing a shape the rest of the
    pipeline cannot crop.
    """

    if not _intersects(box, obstacle):
        return box
    options: list[BoundingBox] = []
    if obstacle.x1 > box.x1:
        options.append(BoundingBox(box.x1, box.y1, obstacle.x1, box.y2))
    if obstacle.x2 < box.x2:
        options.append(BoundingBox(obstacle.x2, box.y1, box.x2, box.y2))
    if obstacle.y1 > box.y1:
        options.append(BoundingBox(box.x1, box.y1, box.x2, obstacle.y1))
    if obstacle.y2 < box.y2:
        options.append(BoundingBox(box.x1, obstacle.y2, box.x2, box.y2))
    if not options:
        return box
    return max(options, key=_area)


def _split_between(a: BoundingBox, b: BoundingBox) -> tuple[BoundingBox, BoundingBox]:
    """Give each box the half of their overlap that is nearer to it.

    Split on the shallower axis: that is the direction the two ROIs actually
    grew into each other, and cutting there costs both of them the least.
    """

    if not _intersects(a, b):
        return a, b
    overlap_x = min(a.x2, b.x2) - max(a.x1, b.x1)
    overlap_y = min(a.y2, b.y2) - max(a.y1, b.y1)
    if overlap_x <= overlap_y:
        middle = (max(a.x1, b.x1) + min(a.x2, b.x2)) / 2.0
        a_first = (a.x1 + a.x2) <= (b.x1 + b.x2)
        first, second = (a, b) if a_first else (b, a)
        near = BoundingBox(first.x1, first.y1, middle, first.y2)
        far = BoundingBox(middle, second.y1, second.x2, second.y2)
        return (near, far) if a_first else (far, near)
    middle = (max(a.y1, b.y1) + min(a.y2, b.y2)) / 2.0
    a_first = (a.y1 + a.y2) <= (b.y1 + b.y2)
    first, second = (a, b) if a_first else (b, a)
    near = BoundingBox(first.x1, first.y1, first.x2, middle)
    far = BoundingBox(second.x1, middle, second.x2, second.y2)
    return (near, far) if a_first else (far, near)


def _accept_cut(
    original: BoundingBox, cut: BoundingBox, config: SolderJointConfig
) -> BoundingBox | None:
    """Take a trimmed ROI only when enough of it survives to inspect."""

    if cut.width < config.min_roi_pixels or cut.height < config.min_roi_pixels:
        return None
    if _area(cut) < config.deconflict_min_area_fraction * _area(original):
        return None
    return cut


def _with_bbox(
    joint: SolderJoint, bbox: BoundingBox, unresolved: bool, against: str
) -> SolderJoint:
    metadata = dict(joint.metadata)
    if bbox is not joint.bbox:
        metadata.setdefault("roi_before_deconflict", joint.bbox.to_dict())
        metadata["deconflicted_against"] = against
    if unresolved:
        metadata["overlap_unresolved"] = True
    return SolderJoint(
        detection_id=joint.detection_id,
        joint_id=joint.joint_id,
        label=joint.label,
        kind=joint.kind,
        bbox=bbox,
        terminal_geometry=joint.terminal_geometry,
        position=joint.position,
        angle=joint.angle,
        pin_index=joint.pin_index,
        source=joint.source,
        designator=joint.designator,
        pin=joint.pin,
        net=joint.net,
        metadata=metadata,
    )


def deconflict_joint_rois(
    joints: Sequence[SolderJoint],
    detections: Sequence[Detection],
    config: SolderJointConfig,
) -> list[SolderJoint]:
    """Stop one component's ROIs from reaching onto its neighbours.

    Every ROI is derived from its own box in isolation, and the outward reach
    is a fixed fraction of that box. On a dense board that reach lands on the
    part next door: two chips 10 px apart produced facing ROIs overlapping 97%,
    so step 6.2 measured the same pixels twice, ``bridge`` lost its meaning,
    and ``refine_to_metal`` snapped both ROIs onto the same blob of solder.

    Two rules, in order. A joint ROI may not cover another component's body --
    whatever is under there, it is not this component's fillet. Then any pair
    of joint ROIs belonging to different components splits the ground between
    them. Body views are left alone: they exist to show the whole part and
    nothing is measured on them.

    A cut that would leave too little to inspect is refused and the ROI is
    marked ``overlap_unresolved`` instead. An ROI swallowed whole by a
    neighbour means the detector merged or duplicated a box, and quietly
    deleting the ROI would hide that.
    """

    if not config.deconflict_neighbours:
        return list(joints)

    bodies = {str(detection.detection_id): detection.bbox for detection in detections}
    resolved = list(joints)

    for index, joint in enumerate(resolved):
        if joint.kind != "joint":
            continue
        box = joint.bbox
        unresolved = False
        for detection_id, body in bodies.items():
            if detection_id == joint.detection_id or not _intersects(box, body):
                continue
            cut = _cut_out_of(box, body)
            # ``_cut_out_of`` hands back the box untouched when there is no
            # slab left to keep -- an ROI wholly inside a neighbour's body. That
            # is not a successful cut, so check the overlap is really gone
            # rather than trusting the box it returned.
            if _intersects(cut, body):
                unresolved = True
                continue
            accepted = _accept_cut(joint.bbox, cut, config)
            if accepted is None:
                unresolved = True
                continue
            box = accepted
        if box is not joint.bbox or unresolved:
            resolved[index] = _with_bbox(joint, box, unresolved, "body")

    for i in range(len(resolved)):
        if resolved[i].kind != "joint":
            continue
        for j in range(i + 1, len(resolved)):
            if resolved[j].kind != "joint":
                continue
            first, second = resolved[i], resolved[j]
            if first.detection_id == second.detection_id:
                continue
            if not _intersects(first.bbox, second.bbox):
                continue
            cut_a, cut_b = _split_between(first.bbox, second.bbox)
            kept_a = _accept_cut(first.bbox, cut_a, config)
            kept_b = _accept_cut(second.bbox, cut_b, config)
            if kept_a is None or kept_b is None:
                resolved[i] = _with_bbox(first, first.bbox, True, "neighbour")
                resolved[j] = _with_bbox(second, second.bbox, True, "neighbour")
                continue
            resolved[i] = _with_bbox(first, kept_a, False, "neighbour")
            resolved[j] = _with_bbox(second, kept_b, False, "neighbour")

    return resolved


def refine_joint_to_metal(
    joint: SolderJoint,
    image: np.ndarray,
    config: SolderJointConfig,
) -> SolderJoint:
    """Shrink a derived ROI onto the metal that is really inside it.

    The expansion ratios that placed the ROI are a guess about land size; the
    pixels are not. Refining raised mean IoU against known pad rectangles from
    0.24 to 0.70 on a synthetic benchmark and tightened 20 of 21 ROIs on a real
    board photo.

    Only the ROI's own pixels are searched. Widening the search to the whole
    component neighbourhood scored identically on the synthetic board and
    visibly worse on the real one, where it latched onto copper traces and a
    neighbouring header's pads. Geometry decides where to look; pixels only
    decide how far the box extends.

    Returns the joint unchanged whenever the evidence is too weak to act on --
    a ``body`` view, a ROI with no metal, or a blob so small that shrinking onto
    it would hide the very emptiness step 6.2 needs to see.
    """

    if joint.kind != "joint":
        return joint
    height, width = image.shape[:2]
    x1, y1, x2, y2 = joint.bbox.clamp(width, height).to_int()
    region = image[y1:y2, x1:x2]
    if region.size == 0 or min(region.shape[:2]) < 3:
        return joint

    mask = segment_solder(region, saturation_max=config.saturation_max)
    if mask.size == 0:
        return joint
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    area = float(region.shape[0] * region.shape[1])
    best, best_area = None, 0.0
    for index in range(1, count):
        left, top, blob_width, blob_height, blob_area = stats[index]
        if blob_area > best_area:
            best, best_area = (left, top, blob_width, blob_height), float(blob_area)
    if best is None or best_area < config.refine_min_metal_fraction * area:
        return joint

    left, top, blob_width, blob_height = best
    if blob_width * blob_height < config.refine_min_area_fraction * area:
        return joint

    refined = BoundingBox(
        float(x1 + left), float(y1 + top),
        float(x1 + left + blob_width), float(y1 + top + blob_height),
    ).clamp(width, height)
    if refined.width <= 0 or refined.height <= 0:
        return joint

    metadata = dict(joint.metadata)
    metadata["refined_to_metal"] = True
    metadata["roi_before_refine"] = joint.bbox.to_dict()
    return SolderJoint(
        detection_id=joint.detection_id,
        joint_id=joint.joint_id,
        label=joint.label,
        kind=joint.kind,
        bbox=refined,
        terminal_geometry=joint.terminal_geometry,
        position=joint.position,
        angle=joint.angle,
        pin_index=joint.pin_index,
        source=joint.source,
        designator=joint.designator,
        pin=joint.pin,
        net=joint.net,
        metadata=metadata,
    )


class SolderJointCropper:
    """Turn detections into label-ready solder-joint crops for step 6.2."""

    def __init__(self, config: SolderJointConfig | None = None) -> None:
        self.config = config or SolderJointConfig()

    def derive(
        self, image: np.ndarray, detections: Sequence[Detection]
    ) -> list[SolderJoint]:
        bgr = ensure_bgr(image)
        height, width = bgr.shape[:2]
        joints: list[SolderJoint] = []
        for detection in detections:
            joints.extend(
                derive_solder_joints(detection, width, height, self.config, bgr)
            )
        # Before refining: two ROIs overlapping 97% would otherwise both snap
        # onto the same blob of metal and read as a confident agreement.
        joints = deconflict_joint_rois(joints, detections, self.config)
        if self.config.refine_to_metal:
            joints = [refine_joint_to_metal(joint, bgr, self.config) for joint in joints]
        return joints

    def extract(
        self,
        image: np.ndarray,
        detections: Sequence[Detection],
        output_dir: str | Path | None = None,
    ) -> list[SolderJointCrop]:
        if not self.config.enabled:
            return []
        return self.extract_joints(image, self.derive(image, detections), output_dir)

    def extract_joints(
        self,
        image: np.ndarray,
        joints: Sequence[SolderJoint],
        output_dir: str | Path | None = None,
    ) -> list[SolderJointCrop]:
        """Cut out ROIs that were already resolved.

        Split from :meth:`extract` so CAD fusion can rewrite the ROI set between
        derivation and cropping without the cropper knowing where a ROI came
        from.
        """

        if not self.config.enabled:
            return []
        bgr = ensure_bgr(image)
        destination = (
            Path(output_dir).expanduser().resolve() if output_dir is not None else None
        )
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=True)

        crops: list[SolderJointCrop] = []
        for index, joint in enumerate(joints):
            x1, y1, x2, y2 = joint.bbox.to_int()
            raw = bgr[y1:y2, x1:x2]
            if raw.size == 0:
                continue
            normalized = letterbox_normalize(
                raw, self.config.target_size, self.config.letterbox_color
            )
            designator = f"{_safe(joint.designator)}_" if joint.designator else ""
            filename = (
                f"{index:05d}_{_safe(joint.label)}_{designator}{joint.kind}_"
                f"{_safe(joint.position)}_{_safe(joint.joint_id)}"
                f"{self.config.image_extension}"
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
                SolderJointCrop(
                    image=normalized,
                    joint=joint,
                    filename=filename,
                    path=file_path,
                    metadata={
                        "raw_shape": {
                            "height": int(raw.shape[0]),
                            "width": int(raw.shape[1]),
                        },
                        "normalized": self.config.target_size is not None,
                        "source": joint.source,
                    },
                )
            )
        return crops


def _safe(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned.strip("._")[:48] or "joint"
