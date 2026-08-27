"""Every CSV the pipeline writes must neutralise spreadsheet formulas.

The escaping was already correct in the Streamlit exporter, in ``digitizer`` and
in ``scripts/build_reference_bundle`` -- three private copies of the same four
lines -- while ``aoi_pipeline/exporters.py``, whose entire job is writing these
files, had none. That is the wrong way round: the exporter's output is what a
technician opens in Excel, and ``designator``/``net`` reach it straight from a
user-supplied CAD or pick-and-place file via ``aoi_pipeline.solder.cad``.

So this module checks the behaviour at the seam that matters -- the rendered CSV
text -- rather than the helper in isolation, and it checks every exporter rather
than the one that happened to be remembered.
"""

from __future__ import annotations

import csv
import io
from types import SimpleNamespace

import pytest

from aoi_pipeline.exporters import (
    cad_findings_csv,
    csv_cell,
    solder_joints_csv,
    solder_verdicts_csv,
)
from aoi_pipeline.models import BoundingBox, SolderJoint, SolderJointCrop

#: The four leaders Excel, LibreOffice and Sheets treat as a formula, plus the
#: two whitespace characters that let a payload hide behind one.
DANGEROUS = ('=HYPERLINK("http://evil","x")', "+1+1", "-2+3", "@SUM(A1)", "\tcmd", "\rcmd")


def _cells(row: str) -> list[str]:
    return next(csv.reader(io.StringIO(row)))


@pytest.mark.parametrize("payload", DANGEROUS)
def test_the_helper_prefixes_every_formula_leader(payload: str) -> None:
    assert csv_cell(payload) == f"'{payload}"


@pytest.mark.parametrize("payload", ["R1", "capacitor", "", "1.5", "a=b"])
def test_the_helper_leaves_ordinary_text_alone(payload: str) -> None:
    """An apostrophe on every cell would be its own kind of corruption."""

    assert csv_cell(payload) == payload


def _joint(**overrides) -> SolderJoint:
    fields = dict(
        detection_id="d0",
        joint_id="d0_joint00",
        label="capacitor",
        kind="joint",
        bbox=BoundingBox(1, 2, 3, 4),
        terminal_geometry="two_terminal",
        position="terminal_a",
        angle=0.0,
    )
    fields.update(overrides)
    return SolderJoint(**fields)


@pytest.mark.parametrize("payload", DANGEROUS)
def test_solder_joints_csv_escapes_cad_supplied_text(payload: str) -> None:
    """``designator`` and ``net`` are read verbatim out of the user's CAD file."""

    crop = SolderJointCrop(
        joint=_joint(designator=payload, net=payload, label=payload),
        image=None,
        filename="x.png",
    )
    row = solder_joints_csv(SimpleNamespace(solder_crops=[crop])).splitlines()[1]
    assert not [c for c in _cells(row) if c.startswith(("=", "+", "-", "@", "\t", "\r"))]


@pytest.mark.parametrize("payload", DANGEROUS)
def test_solder_verdicts_csv_escapes_labels_and_designators(payload: str) -> None:
    verdict = SimpleNamespace(
        joint_id="j0", detection_id="d0", designator=payload, pin=payload,
        component_label=payload, scope="joint", label=payload, decision="review",
        source="rules", probability=0.5, rule_label=payload, model_label=payload,
        model_probability=None, features=None, metadata={},
        model_version=payload, reasons=[payload],
    )
    row = solder_verdicts_csv(SimpleNamespace(solder_verdicts=[verdict])).splitlines()[1]
    assert not [c for c in _cells(row) if c.startswith(("=", "+", "-", "@", "\t", "\r"))]


@pytest.mark.parametrize("payload", DANGEROUS)
def test_cad_findings_csv_escapes_every_text_column(payload: str) -> None:
    finding = SimpleNamespace(
        kind=payload, severity=payload, designator=payload, detection_id=payload,
        expected_class=payload, observed_class=payload, shift_mm=None,
        bbox=None, message=payload,
    )
    run = SimpleNamespace(fusion=SimpleNamespace(findings=[finding]))
    row = cad_findings_csv(run).splitlines()[1]
    assert not [c for c in _cells(row) if c.startswith(("=", "+", "-", "@", "\t", "\r"))]


def test_the_escaped_value_still_round_trips_to_the_original() -> None:
    """Escaping must be reversible, or the export stops being a record."""

    payload = '=HYPERLINK("http://evil","x")'
    crop = SolderJointCrop(joint=_joint(designator=payload), image=None, filename="x.png")
    row = solder_joints_csv(SimpleNamespace(solder_crops=[crop])).splitlines()[1]
    designator = next(c for c in _cells(row) if "HYPERLINK" in c)
    assert designator == f"'{payload}"
    assert designator.removeprefix("'") == payload


def test_numeric_columns_are_not_quoted_into_text() -> None:
    """A negative coordinate starts with '-'. Escaping it would turn a number
    into a string and break every downstream reader of this file."""

    crop = SolderJointCrop(
        joint=_joint(bbox=BoundingBox(-5, -6, 3, 4), angle=-12.5),
        image=None,
        filename="x.png",
    )
    row = solder_joints_csv(SimpleNamespace(solder_crops=[crop])).splitlines()[1]
    cells = _cells(row)
    assert "-12.50" in cells, "angle was escaped and is no longer a number"
    assert "-5.00" in cells and "-6.00" in cells


def test_all_four_call_sites_share_one_implementation() -> None:
    """Three private copies is how they drift apart silently."""

    import aoi_pipeline.digitizer as digitizer
    import scripts.build_reference_bundle as bundle

    assert digitizer._safe_csv_text is csv_cell
    assert bundle._safe_csv_text is csv_cell
