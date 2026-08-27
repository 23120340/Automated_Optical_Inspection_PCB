"""The step-6.2 labelling tools must agree with the notebook that consumes them.

``pack_solder_dataset.py`` exists to fail locally in the same places
``pcb_solder_defect_v2_kaggle.py`` fails on Kaggle, so the rules are asserted
here rather than trusted. A rule that drifts from the notebook is worse than no
rule: it packages work that training then discards without anyone watching.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import cv2
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pack_solder_dataset as packer  # noqa: E402
import build_solder_label_app as builder  # noqa: E402

FIELDS = [
    "crop_path", "defect_class", "board_id", "capture_id", "dataset_source",
    "roi_kind", "label_status", "reviewer_id", "source_image", "label_scope",
    "split", "notes",
]


def _export(tmp_path: Path, rows: list[dict], *, fields: list[str] | None = None) -> Path:
    export = tmp_path / "export"
    (export / "crops").mkdir(parents=True)
    for row in rows:
        cv2.imwrite(str(export / "crops" / row["crop_path"]), np.zeros((32, 32, 3), np.uint8))
    with (export / "solder_dataset.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return export


def _row(name: str, label: str, board: str = "b1", **over) -> dict:
    row = {f: "" for f in FIELDS}
    row.update(
        crop_path=name, defect_class=label, board_id=board, capture_id="c1",
        dataset_source="local_camera", roi_kind="joint",
        label_status="verified" if label else "",
        label_scope=("not_a_joint" if label == "unknown" else "joint") if label else "",
    )
    row.update(over)
    return row


@pytest.mark.parametrize(
    "label,scope,reason",
    [
        ("bridge", "joint", "pair rule"),
        ("shift_component", "joint", "component-placement"),
        ("unknown", "joint", "non-joint label_scope"),
        ("wobbly", "joint", "unrecognised"),
    ],
)
def test_labels_the_notebook_would_discard_are_rejected_here(
    label: str, scope: str, reason: str, tmp_path: Path
) -> None:
    """Each of these has a named rejection in the notebook; none may slip through."""

    export = _export(tmp_path, [_row("a.png", label, label_scope=scope)])
    rows = list(csv.DictReader((export / "solder_dataset.csv").open(encoding="utf-8")))
    usable, rejected = packer.audit(rows, export / "crops")

    assert not usable, f"{label!r} should not be usable"
    assert reason in rejected[0][1], f"{label!r} rejected for the wrong reason: {rejected[0][1]}"


def test_an_unverified_label_is_not_used(tmp_path: Path) -> None:
    """The reviewer has to have signed the row, not merely typed in it."""

    export = _export(tmp_path, [_row("a.png", "good", label_status="")])
    rows = list(csv.DictReader((export / "solder_dataset.csv").open(encoding="utf-8")))
    usable, rejected = packer.audit(rows, export / "crops")
    assert not usable
    assert "not verified" in rejected[0][1]


def test_an_unlabelled_row_is_skipped_without_being_an_error(tmp_path: Path) -> None:
    """Blank means "I did not decide", which is a legitimate answer."""

    export = _export(tmp_path, [_row("a.png", "")])
    rows = list(csv.DictReader((export / "solder_dataset.csv").open(encoding="utf-8")))
    usable, rejected = packer.audit(rows, export / "crops")
    assert not usable
    assert "not an error" in rejected[0][1]


def test_the_five_joint_labels_survive(tmp_path: Path) -> None:
    labels = ["good", "insufficient", "excess", "cold", "missing_solder"]
    rows = [_row(f"{name}.png", name) for name in labels]
    export = _export(tmp_path, rows)
    parsed = list(csv.DictReader((export / "solder_dataset.csv").open(encoding="utf-8")))
    usable, rejected = packer.audit(parsed, export / "crops")
    assert len(usable) == len(labels), rejected


def test_packaging_refuses_a_single_board(tmp_path: Path) -> None:
    """One board cannot be split into train and test without leaking."""

    rows = [_row(f"{i}.png", "good" if i % 2 else "cold", board="only") for i in range(60)]
    export = _export(tmp_path, rows)
    code = packer.main([str(export), "--output", str(tmp_path / "out.zip")])
    assert code == 1
    assert not (tmp_path / "out.zip").exists(), "a blocked dataset must not be written"


def test_packaging_writes_a_zip_when_the_data_holds_up(tmp_path: Path) -> None:
    rows = [
        _row(f"{i}.png", "good" if i % 3 else "insufficient", board=f"b{i % 2}")
        for i in range(60)
    ]
    export = _export(tmp_path, rows)
    out = tmp_path / "out.zip"
    assert packer.main([str(export), "--output", str(out)]) == 0
    assert out.exists()


def test_a_hand_edited_step_5_5_manifest_is_refused(tmp_path: Path) -> None:
    """The step-5.5 CSV lacks the provenance columns training needs."""

    export = _export(
        tmp_path,
        [{"crop_path": "a.png", "defect_class": "good"}],
        fields=["crop_path", "defect_class"],
    )
    with pytest.raises(SystemExit) as excinfo:
        packer.main([str(export)])
    assert "missing required columns" in str(excinfo.value)


def test_the_app_offers_only_labels_the_notebook_accepts() -> None:
    """Offering a label that training discards collects wasted work."""

    template = (PROJECT_ROOT / "scripts" / "_solder_label_app_template.html").read_text(
        encoding="utf-8"
    )
    offered = set(packer.JOINT_LABELS) | {"unknown"}
    for name in offered:
        assert f'name:"{name}"' in template, f"{name} missing from the app"
    for banned in ("bridge", "shift_component", "tombstone", "wrong_polarity"):
        assert f'name:"{banned}"' not in template, f"{banned} must not be offered"


def test_the_app_builds_from_a_real_export(tmp_path: Path) -> None:
    rows = [_row(f"{i}.png", "") for i in range(4)]
    export = _export(tmp_path, rows)
    assert builder.main([str(export), "--board-id", "b1"]) == 0
    html = (export / "label_app.html").read_text(encoding="utf-8")
    assert "__DATA__" not in html and "__DATASET__" not in html
    payload = json.loads(html.split("const DATA = ", 1)[1].split(";\n", 1)[0])
    assert len(payload["rows"]) == 4
    assert all(not row["defect_class"] for row in payload["rows"]), "rows must start unlabelled"


def test_body_views_are_excluded_unless_asked_for(tmp_path: Path) -> None:
    """A whole-component view is not a joint; training routes it out."""

    rows = [_row("j.png", ""), _row("b.png", "", roi_kind="body")]
    export = _export(tmp_path, rows)
    built = builder.build_rows(
        export / "solder_dataset.csv", "b1", "", "local_camera", joints_only=True
    )
    assert [r["crop_path"] for r in built] == ["j.png"]
