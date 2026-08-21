"""Layer A of step 6.2: measure a solder ROI without any trained model.

Everything here is a deterministic measurement, so it works on the first board
of a new line before a single label exists. That matters three times over:

* it inspects from day one, with thresholds an engineer can argue about;
* it pre-labels the dataset, so annotation becomes confirm-or-correct rather
  than label-from-scratch;
* it stays on afterwards as the guard rail under the network. A model that
  calls a joint good while the measured solder area is four percent must not
  be able to pass it silently.

The measurements are ratios wherever possible, so a threshold tuned on one lens
still means something on another. What they cannot be is lighting-independent:
solder is found by being bright and unsaturated, which assumes the fillet
returns more light than the mask around it. Under flat white light a cold joint
and a good one measure nearly the same -- that is a property of the optics, not
of this code, and no threshold here can recover it.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..imaging.image_io import ensure_bgr
from ..models import SolderFeatures

__all__ = [
    "measure_solder",
    "segment_solder",
]


def segment_solder(
    image: np.ndarray,
    saturation_max: int = 110,
    min_value: int = 40,
    min_dynamic_range: int = 35,
    max_bright_fraction: float = 0.90,
) -> np.ndarray:
    """Binary mask of the solder/metal inside one ROI.

    Solder is specular and close to grey: high value, low saturation. Green
    mask, dark component bodies and silkscreen all fail one of those. The
    brightness cut is found by Otsu on the ROI itself rather than fixed, so
    exposure differences between boards do not move it.
    """

    # Checked before ensure_bgr, which rejects empty arrays outright. A ROI
    # clipped entirely off the frame edge has to measure as nothing, not take
    # the whole grading stage down with it.
    if not isinstance(image, np.ndarray) or image.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    bgr = ensure_bgr(image)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    threshold, bright = cv2.threshold(
        value, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # Otsu on a ROI that is almost entirely solder splits the solder itself.
    # Keep it honest by refusing a cut below the plausible floor.
    if threshold < min_value:
        bright = (value >= min_value).astype(np.uint8) * 255
    mask = ((bright > 0) & (saturation <= saturation_max)).astype(np.uint8) * 255

    if int(np.count_nonzero(mask)) == 0:
        # Nothing passed the colour test. That is usually the honest answer --
        # there is no metal in this ROI -- but it is also what happens when the
        # lighting makes solder read as saturated, so fall back to brightness
        # alone *only* where there is something bright to find.
        #
        # The guards matter: Otsu on a flat ROI returns a threshold of zero and
        # marks every pixel bright, so an unguarded fallback turns a bare green
        # land into a fully flooded one and reports excess solder on a joint
        # that has none. That is the worst possible direction to be wrong in.
        span = int(value.max()) - int(value.min())
        bright_fraction = float(np.count_nonzero(bright)) / bright.size
        if span >= min_dynamic_range and bright_fraction <= max_bright_fraction:
            mask = bright.astype(np.uint8)
        else:
            return np.zeros_like(mask)

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Opening exists to drop speckle, not the signal. On a small ROI the whole
    # joint can be a few dozen scattered pixels, and letting the filter erase
    # them reports exactly 0.0 -- indistinguishable from a land with no solder
    # on it at all, which is the most severe call this stage makes.
    if int(np.count_nonzero(cleaned)) == 0:
        return mask
    return cleaned


def measure_solder(
    image: np.ndarray,
    along_axis: str = "auto",
    saturation_max: int = 110,
    specular_percentile: float = 99.0,
) -> SolderFeatures:
    """Measure one ROI.

    ``along_axis`` is the direction the joint runs in: ``"x"``, ``"y"``, or
    ``"auto"`` to take the ROI's longer side. Spans and edge contacts are
    reported along that axis, which is what makes them comparable between a
    horizontal chip terminal and a vertical one.
    """

    if not isinstance(image, np.ndarray) or image.size == 0:
        return _empty_features()
    bgr = ensure_bgr(image)
    if min(bgr.shape[:2]) < 2:
        return _empty_features()

    height, width = bgr.shape[:2]
    axis_is_x = width >= height if along_axis == "auto" else along_axis == "x"

    mask = segment_solder(bgr, saturation_max=saturation_max)
    solder = mask > 0
    total = float(height * width)
    area = int(np.count_nonzero(solder))
    solder_ratio = area / total if total else 0.0

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if area == 0:
        return SolderFeatures(
            solder_ratio=0.0,
            solder_area_px=0,
            span_ratio=0.0,
            width_ratio=0.0,
            centroid_offset_ratio=0.0,
            specular_ratio=0.0,
            edge_density=float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_32F)))) / 255.0,
            contrast=0.0,
            uniformity=0.0,
            edge_contact_start=0.0,
            edge_contact_end=0.0,
            mean_value=float(value.mean()) / 255.0,
            mean_saturation=float(saturation.mean()) / 255.0,
        )

    rows = np.any(solder, axis=1)
    columns = np.any(solder, axis=0)
    row_indices = np.flatnonzero(rows)
    column_indices = np.flatnonzero(columns)
    vertical_span = int(row_indices[-1] - row_indices[0] + 1)
    horizontal_span = int(column_indices[-1] - column_indices[0] + 1)

    if axis_is_x:
        span_ratio = horizontal_span / width
        width_ratio = vertical_span / height
        profile = solder.sum(axis=0).astype(np.float32) / max(1, height)
        length = width
    else:
        span_ratio = vertical_span / height
        width_ratio = horizontal_span / width
        profile = solder.sum(axis=1).astype(np.float32) / max(1, width)
        length = height

    # How far the solder sits from the ROI centre, as a fraction of half the
    # ROI. A well-formed joint is centred on its land; a shifted part or a
    # dragged fillet is not.
    moments = cv2.moments(mask, binaryImage=True)
    centre_x = moments["m10"] / moments["m00"]
    centre_y = moments["m01"] / moments["m00"]
    offset_x = abs(centre_x - width / 2.0) / max(1.0, width / 2.0)
    offset_y = abs(centre_y - height / 2.0) / max(1.0, height / 2.0)
    centroid_offset = float(offset_x if axis_is_x else offset_y)

    # A good fillet throws a crisp specular streak; a cold joint scatters and
    # stays dull even when it covers the same area.
    solder_values = value[solder]
    cutoff = float(np.percentile(value, specular_percentile))
    specular_ratio = float(np.count_nonzero(solder_values >= max(cutoff, 200.0))) / area

    background = value[~solder]
    contrast = (
        float(solder_values.mean() - background.mean()) / 255.0
        if background.size
        else 0.0
    )

    # Coverage evenness along the joint. A bridge is uniformly full; a partial
    # or dewetted joint is patchy.
    uniformity = 1.0 - float(np.std(profile) / max(1e-6, np.mean(profile) + 1e-6))
    uniformity = float(np.clip(uniformity, 0.0, 1.0))

    edge_start, edge_end = _edge_contact(solder, axis_is_x)

    return SolderFeatures(
        solder_ratio=float(solder_ratio),
        solder_area_px=area,
        span_ratio=float(np.clip(span_ratio, 0.0, 1.0)),
        width_ratio=float(np.clip(width_ratio, 0.0, 1.0)),
        centroid_offset_ratio=float(np.clip(centroid_offset, 0.0, 1.0)),
        specular_ratio=float(np.clip(specular_ratio, 0.0, 1.0)),
        edge_density=float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_32F)))) / 255.0,
        contrast=float(np.clip(contrast, -1.0, 1.0)),
        uniformity=uniformity,
        edge_contact_start=edge_start,
        edge_contact_end=edge_end,
        mean_value=float(value.mean()) / 255.0,
        mean_saturation=float(saturation.mean()) / 255.0,
    )


def _edge_contact(solder: np.ndarray, axis_is_x: bool) -> tuple[float, float]:
    """Solder coverage on the two ROI borders across the joint's own axis.

    Both borders covered is the signature of solder running out of its own cell
    into the neighbouring pin, which is what a bridge is. Reported raw so the
    rule layer, which can see the neighbour, decides what it means.
    """

    if solder.size == 0:
        return (0.0, 0.0)
    if axis_is_x:
        start = solder[:, 0]
        end = solder[:, -1]
    else:
        start = solder[0, :]
        end = solder[-1, :]
    return (
        float(np.count_nonzero(start)) / max(1, start.size),
        float(np.count_nonzero(end)) / max(1, end.size),
    )


def _empty_features() -> SolderFeatures:
    return SolderFeatures(
        solder_ratio=0.0,
        solder_area_px=0,
        span_ratio=0.0,
        width_ratio=0.0,
        centroid_offset_ratio=0.0,
        specular_ratio=0.0,
        edge_density=0.0,
        contrast=0.0,
        uniformity=0.0,
        edge_contact_start=0.0,
        edge_contact_end=0.0,
        mean_value=0.0,
        mean_saturation=0.0,
    )
