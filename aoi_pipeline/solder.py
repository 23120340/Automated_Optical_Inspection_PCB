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
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .config import SolderJointConfig, terminal_geometry
from .image_io import encode_image, ensure_bgr
from .models import BoundingBox, Detection, SolderJoint, SolderJointCrop

__all__ = [
    "SolderJointCropper",
    "derive_solder_joints",
    "estimate_component_angle",
    "letterbox_normalize",
    "render_solder_overlay",
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
class _Frame:
    """The component's own coordinate frame inside the analysis image."""

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
) -> list[SolderJoint]:
    """Return the inspection ROIs for one detection.

    ``image`` is optional. Without it the perimeter-band energy filter, the
    per-pin split and orientation estimation are all skipped and every
    candidate ROI is kept, because dropping one on no evidence would silently
    lose joints.
    """

    config = config or SolderJointConfig()
    box = detection.bbox.clamp(image_width, image_height)
    if box.width <= 0 or box.height <= 0:
        return []

    geometry = terminal_geometry(detection.label)
    frame = _component_frame(box, config, image)

    if geometry == "two_terminal":
        rects = _two_terminal_rects(frame, config)
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
) -> _Frame:
    """Pick the frame whose +x axis runs along the component's long side.

    The axis-aligned default already covers the usual 0/90 degree placement.
    Orientation estimation only replaces it when the measured axis is both
    plausible and close enough to the box to be believable.
    """

    center_x = (box.x1 + box.x2) / 2.0
    center_y = (box.y1 + box.y2) / 2.0
    length = max(box.width, box.height)
    span = min(box.width, box.height)
    fallback = _Frame(
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
    return _Frame(rect_cx, rect_cy, rect_angle, rect_length, rect_span)


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


def _two_terminal_rects(frame: _Frame, config: SolderJointConfig) -> list[_LocalRect]:
    inner = config.terminal_inner_ratio * frame.length
    outer = config.terminal_outer_ratio * frame.length
    side = config.terminal_side_ratio * frame.span
    roi_length = inner + outer
    roi_span = frame.span + 2.0 * side
    offset = frame.length / 2.0 + (outer - inner) / 2.0
    return [
        _LocalRect(-offset, 0.0, roi_length, roi_span, "terminal_a"),
        _LocalRect(offset, 0.0, roi_length, roi_span, "terminal_b"),
    ]


def _pad_only_rects(frame: _Frame, config: SolderJointConfig) -> list[_LocalRect]:
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


def _multi_pin_rects(frame: _Frame, config: SolderJointConfig) -> list[_LocalRect]:
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


def _body_rect(rects: Sequence[_LocalRect], frame: _Frame) -> _LocalRect:
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


def _to_image(points: np.ndarray, frame: _Frame) -> np.ndarray:
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


def _local_rect_corners(rect: _LocalRect, frame: _Frame) -> np.ndarray:
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
    frame: _Frame,
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
    frame: _Frame,
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
    frame: _Frame,
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
    frame: _Frame,
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
    frame: _Frame,
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
        return joints

    def extract(
        self,
        image: np.ndarray,
        detections: Sequence[Detection],
        output_dir: str | Path | None = None,
    ) -> list[SolderJointCrop]:
        if not self.config.enabled:
            return []
        bgr = ensure_bgr(image)
        destination = (
            Path(output_dir).expanduser().resolve() if output_dir is not None else None
        )
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=True)

        crops: list[SolderJointCrop] = []
        for index, joint in enumerate(self.derive(bgr, detections)):
            x1, y1, x2, y2 = joint.bbox.to_int()
            raw = bgr[y1:y2, x1:x2]
            if raw.size == 0:
                continue
            normalized = letterbox_normalize(
                raw, self.config.target_size, self.config.letterbox_color
            )
            filename = (
                f"{index:05d}_{_safe(joint.label)}_{joint.kind}_"
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
                    },
                )
            )
        return crops


def letterbox_normalize(
    crop: np.ndarray,
    target_size: tuple[int, int] | None,
    letterbox_color: tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """Resize preserving aspect ratio and pad to ``target_size``."""

    if target_size is None:
        return np.ascontiguousarray(crop.copy())
    target_width, target_height = (int(value) for value in target_size)
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target_size values must be positive")
    source_height, source_width = crop.shape[:2]
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(
        crop, (resized_width, resized_height), interpolation=interpolation
    )
    canvas = np.full(
        (target_height, target_width, 3),
        tuple(int(np.clip(value, 0, 255)) for value in letterbox_color),
        dtype=np.uint8,
    )
    x = (target_width - resized_width) // 2
    y = (target_height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
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


def _safe(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned.strip("._")[:48] or "joint"
