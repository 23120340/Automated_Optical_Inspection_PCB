"""Merging heterogeneous public datasets into the step-6.2 taxonomy."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from aoi_pipeline.grading.datasets import (
    BARE_BOARD_DATASETS,
    SOURCES,
    DatasetSource,
    coverage_report,
    load_source,
    merge_sources,
    probe_layout,
)
from aoi_pipeline.grading.rules import JOINT_CLASSES

TAXONOMY = list(JOINT_CLASSES) + ["shift_component", "missing_component", "tombstone"]


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((24, 24, 3), 120, np.uint8))


# --------------------------------------------------------------------------- #
# Layout probing
# --------------------------------------------------------------------------- #


def test_folder_per_class_is_recognised(tmp_path: Path) -> None:
    for label in ("good", "insufficient", "excess"):
        for index in range(2):
            _image(tmp_path / label / f"{label}_{index}.png")
    probe = probe_layout(tmp_path)
    assert probe.layout == "folder_per_class"
    assert {p.name for p in probe.class_dirs} == {"good", "insufficient", "excess"}


def test_csv_manifest_is_recognised(tmp_path: Path) -> None:
    _image(tmp_path / "crops" / "a.png")
    (tmp_path / "solder_dataset.csv").write_text(
        "crop_path,source_image,defect_class\ncrops/a.png,board1.png,good\n",
        encoding="utf-8",
    )
    assert probe_layout(tmp_path).layout == "csv"


def test_coco_is_recognised(tmp_path: Path) -> None:
    _image(tmp_path / "images" / "b.png")
    (tmp_path / "annotations.json").write_text(
        json.dumps({
            "images": [{"id": 1, "file_name": "images/b.png"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [1, 1, 8, 8]}],
            "categories": [{"id": 7, "name": "misalignment"}],
        }),
        encoding="utf-8",
    )
    assert probe_layout(tmp_path).layout == "coco"


def test_an_unreadable_layout_is_reported_not_guessed(tmp_path: Path) -> None:
    """Guessing a layout would attach labels that were never in the data."""

    for index in range(5):
        _image(tmp_path / f"loose_{index}.png")
    probe = probe_layout(tmp_path)
    assert probe.layout == "unknown"

    records, report = load_source(SOURCES["local_export"], tmp_path, probe)
    assert records == []
    assert "error" in report
    assert "nothing is guessed" in report["error"]


# --------------------------------------------------------------------------- #
# Label mapping
# --------------------------------------------------------------------------- #


def test_labels_are_mapped_into_the_taxonomy(tmp_path: Path) -> None:
    for label in ("misalignment", "excessive_solder", "good"):
        _image(tmp_path / label / f"{label}.png")
        _image(tmp_path / label / f"{label}_2.png")
    records, report = load_source(SOURCES["soldef_ai"], tmp_path)
    labels = sorted({record.label for record in records})
    assert labels == ["excess", "good", "shift_component"]
    assert report["per_class"]["shift_component"] == 2


def test_an_unmapped_label_is_skipped_and_reported(tmp_path: Path) -> None:
    """Folding an unknown defect into the nearest class hides it behind a pass."""

    for label in ("good", "insufficient", "some_new_defect"):
        _image(tmp_path / label / "a.png")
        _image(tmp_path / label / "b.png")
    records, report = load_source(SOURCES["soldef_ai"], tmp_path)
    assert all(record.label != "some_new_defect" for record in records)
    assert report["unmapped_labels"] == {"some_new_defect": 2}
    assert "warning" in report
    assert "do not fold them into the nearest class" in report["warning"]


def test_deliberately_ignored_labels_are_separated_from_unmapped_ones(tmp_path: Path) -> None:
    for label in ("good", "solder_ball", "cold_solder"):
        _image(tmp_path / label / "a.png")
        _image(tmp_path / label / "b.png")
    _, report = load_source(SOURCES["roboflow_soldering"], tmp_path)
    assert report["ignored_on_purpose"] == {"solder_ball": 2}
    assert report["unmapped_labels"] == {}


def test_label_matching_tolerates_spacing_and_case(tmp_path: Path) -> None:
    for name in ("Cold Solder", "INSUFFICIENT-SOLDER"):
        _image(tmp_path / name / "a.png")
        _image(tmp_path / name / "b.png")
    _image(tmp_path / "good" / "a.png")
    _image(tmp_path / "good" / "b.png")
    records, _ = load_source(SOURCES["roboflow_soldering"], tmp_path)
    assert sorted({r.label for r in records}) == ["cold", "good", "insufficient"]


# --------------------------------------------------------------------------- #
# Grouping and merging
# --------------------------------------------------------------------------- #


def test_groups_come_from_the_source_image_so_boards_stay_together(tmp_path: Path) -> None:
    _image(tmp_path / "crops" / "x.png")
    _image(tmp_path / "crops" / "y.png")
    (tmp_path / "solder_dataset.csv").write_text(
        "crop_path,source_image,defect_class\n"
        "crops/x.png,boardA.png,good\n"
        "crops/y.png,boardA.png,cold\n",
        encoding="utf-8",
    )
    records, report = load_source(SOURCES["local_export"], tmp_path)
    assert len(records) == 2
    # Both crops came off one board, so they must land in one group.
    assert report["groups"] == 1


def test_merging_namespaces_groups_so_two_sources_cannot_collide(tmp_path: Path) -> None:
    """Two datasets both naming a board 'board1' must not share a group."""

    first = tmp_path / "one"
    second = tmp_path / "two"
    # "board1__" is what marks the parent board, so both crops group together.
    for root in (first, second):
        for label in ("good", "insufficient", "excess"):
            _image(root / label / "board1__a.png")
            _image(root / label / "board1__b.png")

    import dataclasses

    loaded = [
        load_source(SOURCES["soldef_ai"], first),
        load_source(dataclasses.replace(SOURCES["soldef_ai"], name="other"), second),
    ]
    merged, summary = merge_sources(loaded)
    # Both sources call their board "board1"; without namespacing they would
    # collapse into one group and leak across the train/validation split.
    assert summary["groups"] == 2
    assert sorted({r.group for r in merged}) == ["other/board1", "soldef_ai/board1"]
    assert set(summary["per_source"]) == {"soldef_ai", "other"}
    assert all("/" in record.group for record in merged)


# --------------------------------------------------------------------------- #
# Coverage honesty
# --------------------------------------------------------------------------- #


def test_coverage_names_the_gaps_rather_than_hiding_them(tmp_path: Path) -> None:
    for label in ("good", "insufficient"):
        for index in range(40):
            _image(tmp_path / label / f"{index}.png")
    records, _ = load_source(SOURCES["soldef_ai"], tmp_path)
    coverage = coverage_report(records, TAXONOMY, minimum=30)

    assert set(coverage["covered"]) == {"good", "insufficient"}
    assert "bridge" in coverage["missing"]
    assert "cold" in coverage["missing"]
    assert "shift_component" in coverage["missing"]
    assert any("No data at all" in note for note in coverage["advice"])


def test_a_thin_class_is_called_out_separately_from_a_missing_one(tmp_path: Path) -> None:
    for index in range(40):
        _image(tmp_path / "good" / f"{index}.png")
    for index in range(3):
        _image(tmp_path / "excess" / f"{index}.png")
    _image(tmp_path / "insufficient" / "a.png")
    records, _ = load_source(SOURCES["soldef_ai"], tmp_path)
    coverage = coverage_report(records, TAXONOMY, minimum=30)
    assert coverage["thin"]["excess"] == 3
    assert "bridge" in coverage["missing"]
    assert any("Very few samples" in note for note in coverage["advice"])


def test_single_source_classes_are_flagged(tmp_path: Path) -> None:
    for label in ("good", "insufficient", "excess"):
        for index in range(40):
            _image(tmp_path / label / f"{index}.png")
    records, _ = load_source(SOURCES["soldef_ai"], tmp_path)
    coverage = coverage_report(records, TAXONOMY, minimum=30)
    assert set(coverage["single_source_classes"]) == {"good", "insufficient", "excess"}
    assert any("one source only" in note for note in coverage["advice"])


# --------------------------------------------------------------------------- #
# Registry hygiene
# --------------------------------------------------------------------------- #


def test_every_registered_source_maps_only_into_the_taxonomy() -> None:
    """A map that emits an unknown class would train a head the app cannot read."""

    allowed = set(TAXONOMY)
    for name, source in SOURCES.items():
        targets = set(source.label_map.values())
        assert targets <= allowed, f"{name} maps to unknown classes: {targets - allowed}"


def test_ignored_labels_all_carry_a_reason() -> None:
    for name, source in SOURCES.items():
        for label, reason in source.ignore.items():
            assert reason.strip(), f"{name}.{label} is ignored with no reason given"


def test_bare_board_datasets_are_listed_as_rejected() -> None:
    """DeepPCB-style sets are a different problem and get mis-cited constantly."""

    assert "deeppcb" in BARE_BOARD_DATASETS
    assert "hripcb" in BARE_BOARD_DATASETS
    assert "dspcbsd" in BARE_BOARD_DATASETS
    assert all(reason.strip() for reason in BARE_BOARD_DATASETS.values())
    # And none of them is wired in as a usable source.
    assert not set(BARE_BOARD_DATASETS) & set(SOURCES)


def test_unverified_sources_say_so() -> None:
    """Whether a layout was confirmed by inspection decides if it gets probed."""

    assert SOURCES["soldef_ai"].verified is False
    assert "probed at load time" in SOURCES["soldef_ai"].notes
