"""Step 5.5: the footprint-symmetry prior on a row of leads.

A surface-mount footprint places its lands symmetrically about the body centre
line. SOT-23, SOT-223, SOD, SOIC and QFP all do, which makes two things true
that no amount of looking at one band can establish on its own:

* a row holding an odd number of leads has one of them exactly on the centre
  line, so a **lone** bright run that is centred is a lead and one that is not
  is legend or a via -- this is what makes splitting a one-lead edge safe, and
  ``min_pins_per_band`` existed only because that evidence was missing;
* a row of any size mirrors onto itself, which is the **only** regularity check
  that constrains a two-lead edge. ``np.diff`` of two centres yields a single
  pitch whose standard deviation is zero by construction, so the pitch test in
  ``_find_pin_runs`` has never been able to reject a SOT-23 edge.

Measured on the real board in ``tests/data``, the residual is 0.006 for D201's
and D202's two-lead rows, 0.011 for U201's four-lead rows, and 0.000 and 0.051
for the two lone leads -- while legend in a band corner sits at 0.2 to 0.6.

Rejecting a split is safe by construction: it falls back to one ROI for the
whole band, which is what the code did before. What it cannot do is lose a
joint.
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from aoi_pipeline import BoundingBox, Detection, SolderJointConfig, derive_solder_joints
from aoi_pipeline.solder.geometry import _row_symmetry_residual

BOARD_SIZE = (300, 300)
BODY = (110, 120, 190, 175)  # x1, y1, x2, y2 -- wider than tall, long axis is x


def _sot23(
    *,
    lone_offset: int = 0,
    pair_shift: int = 0,
) -> tuple[np.ndarray, Detection]:
    """A SOT-23: two leads on the bottom edge, one on the top.

    ``lone_offset`` slides the single top lead off the centre line.
    ``pair_shift`` slides *both* bottom leads the same way, which is how to make
    that row asymmetric while keeping each lead inside the band -- moving only
    one of them pushes it past the body, where the band's margin filter drops it
    and the row is left with a single run, testing the lone-lead path over
    again instead of the pair.
    """

    image = np.full((*BOARD_SIZE, 3), (40, 90, 40), np.uint8)
    x1, y1, x2, y2 = BODY
    centre_x = (x1 + x2) // 2
    # lands first, then the body over them, so the leads emerge from under it
    cv2.rectangle(
        image,
        (centre_x - 10 + lone_offset, y1 - 18),
        (centre_x + 10 + lone_offset, y1 + 4),
        (215, 215, 215),
        -1,
    )
    for sign in (-1, 1):
        cx = centre_x + sign * 22 + pair_shift
        cv2.rectangle(image, (cx - 10, y2 - 4), (cx + 10, y2 + 18), (215, 215, 215), -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), (22, 22, 22), -1)
    return image, Detection("ic", 0.9, BoundingBox(*BODY))


def _derive(image, detection, config=None):
    return derive_solder_joints(
        detection,
        BOARD_SIZE[1],
        BOARD_SIZE[0],
        config or SolderJointConfig(),
        image=image,
    )


def _positions(joints) -> set[str]:
    return {j.position for j in joints if j.kind == "joint"}


def _row(joints, edge: str) -> list:
    return [j for j in joints if j.kind == "joint" and j.position.startswith(edge)]


# --------------------------------------------------------------------------- #
# The residual itself
# --------------------------------------------------------------------------- #


def test_a_centred_lone_run_has_no_residual() -> None:
    assert _row_symmetry_residual(np.array([50.0]), 100.0) == pytest.approx(0.0)


def test_a_lone_run_off_the_centre_line_is_penalised_by_twice_its_offset() -> None:
    # reflection sends it to the far side, so a 10-unit offset reads as 20
    assert _row_symmetry_residual(np.array([40.0]), 100.0) == pytest.approx(0.20)


def test_a_mirrored_pair_has_no_residual() -> None:
    assert _row_symmetry_residual(np.array([20.0, 80.0]), 100.0) == pytest.approx(0.0)


def test_a_lopsided_pair_is_penalised() -> None:
    # 20 and 60 reflect to 80 and 40; the nearest real centres are 20 away
    assert _row_symmetry_residual(np.array([20.0, 60.0]), 100.0) == pytest.approx(0.20)


def test_an_odd_row_mirrors_its_middle_lead_onto_itself() -> None:
    assert _row_symmetry_residual(np.array([20.0, 50.0, 80.0]), 100.0) == pytest.approx(0.0)


def test_an_empty_row_is_not_symmetric_by_default() -> None:
    """Vacuous truth here would let a band with nothing in it claim a split."""

    assert _row_symmetry_residual(np.array([]), 100.0) == float("inf")


# --------------------------------------------------------------------------- #
# What it changes on a component
# --------------------------------------------------------------------------- #


def test_the_lone_lead_of_a_sot23_gets_its_own_roi() -> None:
    """Before the symmetry prior this edge could not be split at all: the count
    floor is two, so one lead meant one ROI spanning the whole edge."""

    image, detection = _sot23()
    joints = _derive(image, detection)
    top = _row(joints, "lead_top")
    assert len(top) == 1
    assert top[0].position.endswith("_pin00"), (
        f"the lone lead is still an unsplit band: {top[0].position}"
    )


def test_that_roi_is_centred_on_the_lead_and_not_on_the_whole_edge() -> None:
    """The point of splitting it: an edge-wide ROI is mostly bare laminate.

    Measured on the real board, this took D201's lone lead from 2.65x its pad
    area down to 1.20x with its pad coverage unchanged.
    """

    image, detection = _sot23()
    top = _row(_derive(image, detection), "lead_top")[0]
    x1, _, x2, _ = BODY
    body_width = x2 - x1
    roi_centre = (top.bbox.x1 + top.bbox.x2) / 2.0
    assert abs(roi_centre - (x1 + x2) / 2.0) <= 0.10 * body_width
    assert top.bbox.width < 0.75 * body_width, (
        f"ROI is {top.bbox.width:.0f}px across a {body_width}px edge; it has not "
        "been narrowed to the lead"
    )


def test_the_lone_roi_still_reaches_past_the_metal_to_hold_the_fillet() -> None:
    """A lone lead has no pitch, so a naive growth of zero clips the ROI to the
    bright metal and cuts away the thing being graded."""

    image, detection = _sot23()
    top = _row(_derive(image, detection), "lead_top")[0]
    assert top.bbox.width > 20, f"ROI {top.bbox.width:.0f}px is no wider than the 20px land"


def test_an_off_centre_lone_run_is_refused_and_the_band_is_kept() -> None:
    """Legend and vias are what the count floor was guarding against, and this
    is the guard that replaces it. Refusing costs the coarser whole-band ROI;
    accepting would put an ROI on a scrap of silkscreen."""

    image, detection = _sot23(lone_offset=26)
    top = _row(_derive(image, detection), "lead_top")
    assert len(top) == 1
    assert not top[0].position.endswith("_pin00"), (
        "an off-centre run was accepted as a centred lone lead"
    )


def test_a_lopsided_pair_is_refused_where_the_pitch_test_cannot_see_it() -> None:
    """Two runs give ``np.diff`` one pitch and a standard deviation of zero, so
    before this rule every two-run edge passed regardless of where the runs sat.

    Both leads are still found here -- the band profile holds runs at 34.5 and
    78.5 of a 92px band, a residual of 0.228 -- so the refusal comes from the
    symmetry gate and not from one of them having gone missing.
    """

    image, detection = _sot23(pair_shift=10)
    bottom = _row(_derive(image, detection), "lead_bottom")
    assert not any(j.position.endswith(("_pin00", "_pin01")) for j in bottom), (
        "an asymmetric pair was split; the pitch check cannot reject two runs"
    )


def test_a_mirrored_pair_is_still_split() -> None:
    """The guard must not cost the case it was built around."""

    image, detection = _sot23()
    bottom = _row(_derive(image, detection), "lead_bottom")
    assert len(bottom) == 2
    assert {j.pin_index for j in bottom} == {0, 1}


def test_refusing_a_split_never_drops_a_joint() -> None:
    """The safety property the whole rule rests on: the fallback is a coarser
    ROI over the same edge, never no ROI.

    Both asymmetries are checked, because they refuse on different paths -- one
    through the lone-lead branch and one through the pair.
    """

    accepted_edges = {p.split("_pin")[0] for p in _positions(_derive(*_sot23()))}
    assert accepted_edges >= {"lead_top", "lead_bottom"}

    for label, kwargs in (("lone", {"lone_offset": 26}), ("pair", {"pair_shift": 10})):
        joints = _derive(*_sot23(**kwargs))
        refused_edges = {p.split("_pin")[0] for p in _positions(joints)}
        assert refused_edges == accepted_edges, (
            f"the {label} asymmetry changed which edges carry an ROI: "
            f"{sorted(refused_edges)} against {sorted(accepted_edges)}"
        )
        for edge in accepted_edges:
            assert _row(joints, edge), f"{edge} lost its ROI entirely on the {label} case"


def test_turning_the_lone_pin_rule_off_restores_the_previous_behaviour() -> None:
    """The rule is a config flag, so a caller who distrusts it can have the old
    conservative geometry back without editing code."""

    image, detection = _sot23()
    config = replace(SolderJointConfig(), split_lone_centred_pin=False)
    top = _row(_derive(image, detection, config), "lead_top")
    assert len(top) == 1
    assert not top[0].position.endswith("_pin00")


def test_a_tighter_tolerance_rejects_a_real_row(monkeypatch) -> None:
    """Guards the tolerance itself: if the gate did nothing, shrinking it to
    zero would change nothing either, and the test above would prove nothing."""

    image, detection = _sot23()
    config = replace(SolderJointConfig(), pin_row_symmetry_max_residual=0.0)
    bottom = _row(_derive(image, detection, config), "lead_bottom")
    assert not any(j.position.endswith("_pin00") for j in bottom)
