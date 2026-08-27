"""Tests for the component cropper that feeds step-6.2 joint labelling.

The property that matters most here is that ``body_x``/``body_y`` really locate
the annotated component inside the padded crop. The labelling app draws its hint
rectangle from those two numbers, so an off-by-one in the padding arithmetic
would put the hint somewhere plausible-looking and quietly mislead the reviewer
on every crop.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import sys
import zipfile

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.crop_components_for_labelling import (  # noqa: E402
    Box,
    crop_box,
    main,
    read_class_names,
    scene_of,
)


def test_read_class_names_inline() -> None:
    text = "train: ../train/images\nnc: 3\nnames: ['IC', 'Capacitor', 'Test Point']\n"
    assert read_class_names(text) == ["IC", "Capacitor", "Test Point"]


def test_read_class_names_block_form() -> None:
    text = "nc: 2\nnames:\n  - IC\n  - Capacitor\ntrain: x\n"
    assert read_class_names(text) == ["IC", "Capacitor"]


def test_scene_of_strips_roboflow_augmentation_suffix() -> None:
    augmented = "train/images/Arty_Top_jpg.rf.5d15b4645b32647b6439efa7fe4e3942.jpg"
    plain = "train/images/Arty_Top_jpg.jpg"
    assert scene_of(augmented) == scene_of(plain) == "Arty_Top_jpg"


def test_crop_box_pads_each_axis_in_proportion() -> None:
    # a two-terminal part twice as wide as it is tall: the terminals lie on the
    # wide axis, so that axis must receive the larger pad
    box = Box("Resistor", cx=0.5, cy=0.5, w=0.2, h=0.1)
    left, top, right, bottom = crop_box(box, 1000, 1000, margin=0.25, min_pad=0)
    assert (right - left) - 200 == pytest.approx(2 * 0.25 * 200, abs=1)
    assert (bottom - top) - 100 == pytest.approx(2 * 0.25 * 100, abs=1)


def test_crop_box_clamps_at_the_image_edge() -> None:
    box = Box("IC", cx=0.02, cy=0.02, w=0.04, h=0.04)
    left, top, right, bottom = crop_box(box, 500, 500, margin=0.5, min_pad=8)
    assert (left, top) == (0, 0)
    assert right <= 500 and bottom <= 500


def _write_dataset(path: Path) -> None:
    """A two-scene dataset: one scene stored twice as Roboflow augmentations."""
    names = ["IC", "Capacitor", "Test Point"]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data.yaml", f"nc: 3\nnames: {names!r}\n".replace("'", "'"))
        # 400x400 board. IC is 100x80 at centre -> passes a 48 px gate.
        # Capacitor is 20x20 -> fails. Test Point is 60x60 but denied by class.
        labels = "\n".join([
            "0 0.5 0.5 0.25 0.2",
            "1 0.1 0.1 0.05 0.05",
            "2 0.8 0.8 0.15 0.15",
        ])
        for split, stem in (("train", "boardA_jpg.rf." + "a" * 32),
                            ("valid", "boardA_jpg.rf." + "b" * 32),
                            ("train", "boardB_jpg.rf." + "c" * 32)):
            buffer = io.BytesIO()
            Image.new("RGB", (400, 400), (12, 90, 40)).save(buffer, format="PNG")
            archive.writestr(f"{split}/images/{stem}.png", buffer.getvalue())
            archive.writestr(f"{split}/labels/{stem}.txt", labels)


def test_end_to_end_gate_dedupe_and_body_coordinates(tmp_path: Path) -> None:
    source = tmp_path / "src.zip"
    _write_dataset(source)
    out = tmp_path / "out"
    assert main([str(source), "--output", str(out), "--min-short-side", "48"]) == 0

    rows = list(csv.DictReader((out / "manifest.csv").open(encoding="utf-8")))
    # two scenes survive the augmentation dedupe, one usable component each
    assert sorted(r["scene_id"] for r in rows) == ["boardA_jpg", "boardB_jpg"]
    assert {r["component_class"] for r in rows} == {"IC"}

    provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["scenes_available"] == 2
    assert provenance["rejected_too_small"] == {"Capacitor": 2}
    assert provenance["rejected_by_class_filter"] == {"Test Point": 2}

    for row in rows:
        crop = Image.open(out / "crops" / row["crop_path"])
        assert crop.size == (int(row["crop_w"]), int(row["crop_h"]))
        # the body must sit wholly inside the crop, and the padding on each side
        # must match what was asked for
        bx, by = int(row["body_x"]), int(row["body_y"])
        bw, bh = int(row["body_w"]), int(row["body_h"])
        assert (bw, bh) == (100, 80)
        assert bx >= 0 and by >= 0
        assert bx + bw <= crop.size[0]
        assert by + bh <= crop.size[1]
        assert bx == pytest.approx(0.30 * bw, abs=1)
        assert by == pytest.approx(0.30 * bh, abs=1)


def test_body_coordinates_survive_clamping_at_the_edge(tmp_path: Path) -> None:
    """A component near the border loses padding on one side only.

    Clamping is where a naive implementation breaks: it shrinks the crop but
    forgets that the body then sits at a different offset inside it.
    """
    source = tmp_path / "edge.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("data.yaml", "nc: 1\nnames: ['IC']\n")
        buffer = io.BytesIO()
        Image.new("RGB", (400, 400), (0, 0, 0)).save(buffer, format="PNG")
        stem = "edge_jpg"
        archive.writestr(f"train/images/{stem}.png", buffer.getvalue())
        # body flush against the left edge: left pad has nowhere to go
        archive.writestr(f"train/labels/{stem}.txt", "0 0.125 0.5 0.25 0.25")

    out = tmp_path / "out"
    assert main([str(source), "--output", str(out), "--min-short-side", "48"]) == 0
    row = next(iter(csv.DictReader((out / "manifest.csv").open(encoding="utf-8"))))
    assert int(row["body_x"]) == 0, "body starts at the crop edge when padding is clamped away"
    assert int(row["body_y"]) == pytest.approx(0.30 * 100, abs=1)
    crop = Image.open(out / "crops" / row["crop_path"])
    assert crop.size[0] == int(row["body_w"]) + int(round(0.30 * 100))


def test_no_label_is_ever_invented(tmp_path: Path) -> None:
    source = tmp_path / "src.zip"
    _write_dataset(source)
    out = tmp_path / "out"
    main([str(source), "--output", str(out)])
    rows = list(csv.DictReader((out / "manifest.csv").open(encoding="utf-8")))
    assert rows
    assert all(row["label_status"] == "" and row["reviewer_id"] == "" for row in rows)
    assert all(row["roi_kind"] == "component" for row in rows)


def test_max_per_class_caps_the_labelling_queue(tmp_path: Path) -> None:
    source = tmp_path / "src.zip"
    _write_dataset(source)
    out = tmp_path / "out"
    main([str(source), "--output", str(out), "--max-per-class", "1"])
    rows = list(csv.DictReader((out / "manifest.csv").open(encoding="utf-8")))
    assert len(rows) == 1
