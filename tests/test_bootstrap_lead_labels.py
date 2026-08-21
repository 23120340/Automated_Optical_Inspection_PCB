"""Exporting geometry-derived lead boxes as a correctable YOLO dataset.

The point of this export is to break the data constraint behind the detector's
``pads`` recall of 0.072: only 30 of 670 public training images contain the
class at all. These tests pin the properties that make the export usable --
correct coordinate frame, no body boxes, valid YOLO, and an output that says
out loud it is not ground truth.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import bootstrap_lead_labels as bootstrap  # noqa: E402

from aoi_pipeline.models import BoundingBox, SolderJoint  # noqa: E402


def _board(path: Path, width: int = 700, height: int = 500) -> None:
    image = np.full((height, width, 3), (45, 95, 45), np.uint8)
    for x, y in ((120, 120), (300, 200)):
        cv2.rectangle(image, (x, y), (x + 70, y + 30), (35, 35, 35), -1)
        cv2.rectangle(image, (x - 14, y), (x, y + 30), (205, 205, 205), -1)
        cv2.rectangle(image, (x + 70, y), (x + 84, y + 30), (205, 205, 205), -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _joint(kind: str, geometry: str) -> SolderJoint:
    return SolderJoint(
        detection_id="d0",
        joint_id="d0_j0",
        label="resistor",
        kind=kind,
        bbox=BoundingBox(10, 10, 30, 30),
        terminal_geometry=geometry,
        position="terminal_a",
    )


# --------------------------------------------------------------------------- #
# Class proposal
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "geometry,expected",
    [("two_terminal", "pads"), ("pad_only", "pads"), ("multi_pin", "pins")],
)
def test_terminal_topology_decides_the_proposed_class(geometry, expected) -> None:
    assert bootstrap.lead_class_for(_joint("joint", geometry)) == expected


def test_a_body_view_is_never_proposed_as_a_lead() -> None:
    """The body view covers the whole component. Writing it as a ``pads`` box
    would teach the detector that a resistor is a pad."""

    assert bootstrap.lead_class_for(_joint("body", "two_terminal")) is None


def test_an_unknown_topology_is_skipped_rather_than_guessed() -> None:
    assert bootstrap.lead_class_for(_joint("joint", "something_new")) is None


# --------------------------------------------------------------------------- #
# YOLO encoding
# --------------------------------------------------------------------------- #


def test_yolo_line_is_normalised_against_the_image_it_will_ship_with() -> None:
    line = bootstrap.to_yolo_line(1, BoundingBox(100, 50, 300, 150), 1000, 500)
    index, cx, cy, width, height = line.split()
    assert index == "1"
    assert float(cx) == pytest.approx(0.2)
    assert float(cy) == pytest.approx(0.2)
    assert float(width) == pytest.approx(0.2)
    assert float(height) == pytest.approx(0.2)


def test_a_degenerate_box_produces_no_line() -> None:
    assert bootstrap.to_yolo_line(0, BoundingBox(10, 10, 10, 40), 100, 100) is None


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


def test_export_writes_a_loadable_yolo_dataset(tmp_path: Path) -> None:
    boards = tmp_path / "boards"
    _board(boards / "b0.png")
    _board(boards / "b1.png")
    output = tmp_path / "out"

    exit_code = bootstrap.main(
        [str(boards), "--output", str(output), "--overlays"]
    )
    assert exit_code == 0

    assert (output / "data.yaml").is_file()
    yaml_text = (output / "data.yaml").read_text(encoding="utf-8")
    for name in bootstrap.LEAD_CLASS_NAMES:
        assert f"- {name}" in yaml_text

    for stem in ("b0", "b1"):
        assert (output / "images" / f"{stem}.png").is_file()
        assert (output / "labels" / f"{stem}.txt").is_file()
        assert (output / "overlays" / f"{stem}.png").is_file()

    # Every line is valid YOLO: a known class index and four values in [0, 1].
    for label_file in (output / "labels").glob("*.txt"):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            assert len(parts) == 5
            assert 0 <= int(parts[0]) < len(bootstrap.LEAD_CLASS_NAMES)
            assert all(0.0 <= float(value) <= 1.0 for value in parts[1:])


def test_the_written_image_is_the_frame_the_boxes_were_measured_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ROIs live in the aligned/preprocessed analysis frame. Shipping the
    original file instead would offset every box by the preprocessing scale --
    silently, because the labels would still look well-formed.

    Step 1 only rescales above ``PreprocessConfig.max_side``; rather than build
    a 4096-wide fixture to cross that line, the limit is lowered so a small
    board crosses it instead. Same code path, a fraction of the memory.
    """

    original = bootstrap.build_config

    def shrinking_config(args):
        config = original(args)
        config.preprocess.max_side = 320
        return config

    monkeypatch.setattr(bootstrap, "build_config", shrinking_config)

    boards = tmp_path / "boards"
    _board(boards / "wide.png", width=700, height=500)
    output = tmp_path / "out"

    assert bootstrap.main([str(boards), "--output", str(output)]) == 0

    written = cv2.imread(str(output / "images" / "wide.png"))
    assert written is not None
    assert max(written.shape[:2]) == 320, (
        "the analysis frame was rescaled but the original was written instead"
    )


def test_the_manifest_says_the_labels_are_not_ground_truth(tmp_path: Path) -> None:
    """Training on uncorrected pseudo-labels only re-learns the geometry that
    produced them. The output has to carry that warning with it."""

    boards = tmp_path / "boards"
    _board(boards / "b0.png")
    output = tmp_path / "out"
    assert bootstrap.main([str(boards), "--output", str(output)]) == 0

    manifest = json.loads((output / "bootstrap_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PSEUDO_LABELS_NEED_REVIEW"
    assert "class_counts" in manifest and "roi_source_counts" in manifest

    readme = (output / "README_FIRST.md").read_text(encoding="utf-8")
    assert "không phải ground truth" in readme.lower()


def test_no_boards_is_an_error_not_an_empty_dataset(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert bootstrap.main([str(empty), "--output", str(tmp_path / "out")]) == 2
