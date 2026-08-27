"""Step 5.5 regression: derived ROIs must land on the real joints of a real board.

The synthetic boards in :mod:`tests.inspection.test_solder_joints` check the
geometry rules in isolation. They cannot catch the failures that only appear on
a photograph: a square-ish body whose terminal axis the box cannot name, an
outward reach measured against the long side of a part that has no long side,
or a three-lead package whose lone lead sits on the edge the band filter throws
away.

So this module runs the same derivation against one real board crop with
hand-measured, visually verified pad rectangles, and asserts the two properties
that matter for step 6.2:

* **coverage** -- every real joint is inside some ROI, because a joint with no
  ROI is never inspected and ships;
* **usefulness** -- no ROI is a sliver, and ROIs that sit on nothing are
  bounded, because every one of them costs an operator a decision.

All 39 detections are frozen from a real run of the shipped detector rather
than recomputed, so the test stays fast, offline and deterministic while still
exercising the boxes the pipeline actually receives -- and so neighbour
deconfliction, which trims one part's ROIs off the part next door, sees the same
crowded board it sees in production.

The image is run through step 1 first, because that is the frame step 5.5 is
handed and the two are not interchangeable: white balance alone moves the mean
saturation of this board from 144 to 91, and C231's top tab from S=118/V=124 to
S=64/V=171. An earlier version of this file measured on the raw file, passed
everything, and hid the fact that the shipped pipeline had barely changed.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from aoi_pipeline import BoundingBox, Detection, SolderJointConfig, derive_solder_joints
from aoi_pipeline.config import PreprocessConfig
from aoi_pipeline.imaging.preprocessing import ImagePreprocessor
from aoi_pipeline.solder.geometry import deconflict_joint_rois

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "solder_geometry"
GROUND_TRUTH = DATA_DIR / "board_smd_00001.json"

# A joint is "covered" when this much of its pad area sits inside one ROI. Half
# is deliberate: the ROI has to hold enough of the fillet to grade it, but a
# fillet that overhangs its land slightly must not fail the check.
MIN_PAD_COVERAGE = 0.50
# Below this a crop cannot be graded no matter how good the model is; the
# exporter already warns about the same threshold at 24px for the short side.
MIN_ROI_SHORT_SIDE_PX = 12
# Coverage alone passes a box several times too big or visibly off to one side,
# which is what a reviewer notices first and what coverage cannot see. Both
# numbers come off this board rather than off taste: the parts nobody
# complained about measured 1.4..2.8 area and 0.04..0.37 offset, while the three
# V-chip electrolytics that drew the complaint measured 14..28 and up to 0.60.
# The gap is wide enough that these gates sit in open ground.
MAX_ROI_TO_PAD_AREA = 6.0
MAX_CENTRE_OFFSET_IN_PAD_SIZES = 0.5


@pytest.fixture(scope="module")
def board() -> tuple[np.ndarray, dict]:
    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    raw = cv2.imread(str(DATA_DIR / truth["image"]))
    assert raw is not None, f"missing test asset {truth['image']}"
    # Step 1, with the shipped defaults, exactly as ``AOIPipeline.run`` does it.
    # Photometric only at these settings, so the hand-measured pad rectangles
    # still land where they were measured.
    processed = ImagePreprocessor(PreprocessConfig()).process(raw).image
    assert processed.shape == raw.shape, "step 1 moved the geometry; pads no longer align"
    return processed, truth


def _coverage(roi: BoundingBox, pad: list[int]) -> float:
    px1, py1, px2, py2 = pad
    ix = max(0.0, min(roi.x2, px2) - max(roi.x1, px1))
    iy = max(0.0, min(roi.y2, py2) - max(roi.y1, py1))
    area = float((px2 - px1) * (py2 - py1))
    return (ix * iy) / area if area > 0 else 0.0


def _all_joints(image: np.ndarray, truth: dict) -> dict[int, list]:
    """Derive every ROI on the board, then deconflict, exactly as step 5.5 does."""

    height, width = image.shape[:2]
    config = SolderJointConfig()
    detections = [
        Detection(d["label"], d["confidence"], BoundingBox(*d["box"]))
        for d in truth["detections"]
    ]
    index_of = {d.detection_id: i for i, d in enumerate(detections)}

    joints: list = []
    for detection in detections:
        joints.extend(
            derive_solder_joints(detection, width, height, config=config, image=image)
        )

    per_detection: dict[int, list] = {i: [] for i in range(len(detections))}
    for joint in deconflict_joint_rois(joints, detections, config):
        if joint.kind != "joint":
            continue
        per_detection[index_of[joint.detection_id]].append(joint)
    return per_detection


@pytest.fixture(scope="module")
def joints_by_detection(board) -> dict[int, list]:
    image, truth = board
    return _all_joints(image, truth)


def _joints_for(entry: dict, joints_by_detection: dict[int, list]) -> list:
    return joints_by_detection[entry["detection_index"]]


def _cases() -> list[str]:
    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    return sorted(truth["components"])


@pytest.mark.parametrize("name", _cases())
def test_every_real_joint_is_covered_by_an_roi(name: str, board, joints_by_detection) -> None:
    """No pad may be left without an ROI: an uninspected joint is an escape."""

    _, truth = board
    entry = truth["components"][name]
    joints = _joints_for(entry, joints_by_detection)

    uncovered = []
    for index, pad in enumerate(entry["pads"]):
        best = max((_coverage(j.bbox, pad) for j in joints), default=0.0)
        if best < MIN_PAD_COVERAGE:
            uncovered.append((index, pad, round(best, 3)))

    assert not uncovered, (
        f"{name} ({entry['package']}): {len(uncovered)} of {len(entry['pads'])} pads "
        f"are not covered to {MIN_PAD_COVERAGE:.0%}; worst offenders {uncovered}. "
        f"ROIs were {[tuple(int(v) for v in j.bbox.to_int()) for j in joints]}"
    )


@pytest.mark.parametrize("name", _cases())
def test_the_roi_is_about_the_size_of_the_joint_it_inspects(
    name: str, board, joints_by_detection
) -> None:
    """A box swallowing the whole component is not inspecting one joint."""

    _, truth = board
    entry = truth["components"][name]
    joints = _joints_for(entry, joints_by_detection)
    # A part may carry a tighter cap of its own. The global gate is set where it
    # is so it can be applied to every package on the board, which makes it too
    # loose to catch a regression on a part already measured well inside it --
    # D201's lone lead went from 2.65 to 1.20 without the global gate moving.
    limit = float(entry.get("max_roi_to_pad_area", MAX_ROI_TO_PAD_AREA))

    oversized = []
    for index, pad in enumerate(entry["pads"]):
        best = max(joints, key=lambda j: _coverage(j.bbox, pad), default=None)
        if best is None:
            continue
        pad_area = float((pad[2] - pad[0]) * (pad[3] - pad[1]))
        ratio = (best.bbox.width * best.bbox.height) / max(pad_area, 1.0)
        if ratio > limit:
            oversized.append((index, best.position, round(ratio, 1)))

    assert not oversized, (
        f"{name} ({entry['package']}): ROI is over {limit:g}x the pad "
        f"area for {oversized}"
    )


@pytest.mark.parametrize("name", _cases())
def test_the_roi_is_centred_on_the_joint(name: str, board, joints_by_detection) -> None:
    """An ROI offset by most of a pad grades the neighbouring copper too."""

    _, truth = board
    entry = truth["components"][name]
    joints = _joints_for(entry, joints_by_detection)

    off_centre = []
    for index, pad in enumerate(entry["pads"]):
        best = max(joints, key=lambda j: _coverage(j.bbox, pad), default=None)
        if best is None:
            continue
        pad_cx, pad_cy = (pad[0] + pad[2]) / 2.0, (pad[1] + pad[3]) / 2.0
        roi_cx = (best.bbox.x1 + best.bbox.x2) / 2.0
        roi_cy = (best.bbox.y1 + best.bbox.y2) / 2.0
        pad_size = max(pad[2] - pad[0], pad[3] - pad[1])
        offset = np.hypot(roi_cx - pad_cx, roi_cy - pad_cy) / max(pad_size, 1.0)
        if offset > MAX_CENTRE_OFFSET_IN_PAD_SIZES:
            off_centre.append((index, best.position, round(float(offset), 2)))

    assert not off_centre, (
        f"{name} ({entry['package']}): ROI centre is more than "
        f"{MAX_CENTRE_OFFSET_IN_PAD_SIZES:g} pad-widths off for {off_centre}"
    )


@pytest.mark.parametrize("name", _cases())
def test_no_roi_is_a_sliver(name: str, board, joints_by_detection) -> None:
    """A few-pixel-wide ROI cannot be graded and only costs review time."""

    _, truth = board
    entry = truth["components"][name]
    joints = _joints_for(entry, joints_by_detection)

    slivers = [
        (j.position, tuple(int(v) for v in j.bbox.to_int()))
        for j in joints
        if min(j.bbox.width, j.bbox.height) < MIN_ROI_SHORT_SIDE_PX
    ]
    assert not slivers, (
        f"{name} ({entry['package']}): {len(slivers)} ROIs are thinner than "
        f"{MIN_ROI_SHORT_SIDE_PX}px on their short side: {slivers}"
    )


@pytest.mark.parametrize("name", _cases())
def test_rois_that_sit_on_no_joint_are_bounded(name: str, board, joints_by_detection) -> None:
    """Every ROI that covers nothing is a decision an operator pays for."""

    _, truth = board
    entry = truth["components"][name]
    joints = _joints_for(entry, joints_by_detection)
    pads = entry["pads"]

    stray = [
        (j.position, tuple(int(v) for v in j.bbox.to_int()))
        for j in joints
        # ``*_cross`` is not a stray. It is the second terminal axis, emitted on
        # purpose and named so a reviewer can see it is the alternative
        # hypothesis rather than a third and fourth terminal. On this board the
        # two axes of C211 and C231 measure 0.284 against 0.295 and 0.325
        # against 0.326 once step 1 has run, and a rule that called either of
        # those a decision would be inventing confidence it does not have. The
        # cost of abstaining is these two reviewable ROIs; the cost of guessing
        # was C231 shipping with both of its joints uninspected.
        if not j.position.endswith("_cross")
        and max((_coverage(j.bbox, pad) for pad in pads), default=0.0) < 0.10
    ]
    # One spare ROI per part is tolerable; a part whose ROIs are mostly on bare
    # laminate is not, and that is the state this test was written to end.
    assert len(stray) <= 1, (
        f"{name} ({entry['package']}): {len(stray)} of {len(joints)} ROIs cover no "
        f"real joint: {stray}"
    )
