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
    source_group,
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


def _yolo_pair(root: Path, split: str, name: str, class_index: int, box: tuple[float, float, float, float]) -> None:
    _image(root / split / "images" / f"{name}.jpg")
    labels_dir = root / split / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    cx, cy, w, h = box
    (labels_dir / f"{name}.txt").write_text(f"{class_index} {cx} {cy} {w} {h}\n", encoding="utf-8")


def test_yolo_is_recognised_and_names_list_style_resolves_real_labels(tmp_path: Path) -> None:
    """Classic Roboflow YOLOv5/v8 export: ``names: ['a', 'b']`` flow-style list."""

    _yolo_pair(tmp_path, "train", "img1", 0, (0.5, 0.5, 0.2, 0.2))
    _yolo_pair(tmp_path, "train", "img2", 1, (0.5, 0.5, 0.3, 0.3))
    (tmp_path / "data.yaml").write_text(
        "train: train/images\nval: train/images\nnc: 2\n"
        "names: ['bridge', 'good']\n",
        encoding="utf-8",
    )
    probe = probe_layout(tmp_path)
    assert probe.layout == "yolo"

    records, report = load_source(SOURCES["roboflow_soldering"], tmp_path)
    labels = sorted(r.label for r in records)
    assert labels == ["bridge", "good"]
    assert report["unmapped_labels"] == {}


def test_yolo_names_block_sequence_style_resolves_real_labels(tmp_path: Path) -> None:
    """This is the exact bug found on a real Kaggle run: a Roboflow "yolo26"
    export wrote ``names`` as a YAML block sequence (``- bridge`` per line)
    instead of the classic flow-style list. Verified directly: the old
    hand-rolled regex (``names\\s*:\\s*\\[(.*?)\\]`` then a digit:value line
    scan) returns no match at all for this shape, so every label silently
    fell back to its raw numeric class index ("7", "4", ...) -- none of which
    matched anything in LABEL_MAPS, so an entire 11832-annotation dataset was
    reported as 0 kept / all unmapped, with no error. A real YAML parser
    resolves this shape without having to special-case it."""

    _yolo_pair(tmp_path, "train", "img1", 0, (0.5, 0.5, 0.2, 0.2))
    _yolo_pair(tmp_path, "train", "img2", 1, (0.5, 0.5, 0.3, 0.3))
    (tmp_path / "data.yaml").write_text(
        "train: train/images\nval: train/images\nnc: 2\n"
        "names:\n  - bridge\n  - good\n",
        encoding="utf-8",
    )
    probe = probe_layout(tmp_path)
    assert probe.layout == "yolo"

    records, report = load_source(SOURCES["roboflow_soldering"], tmp_path)
    labels = sorted(r.label for r in records)
    assert labels == ["bridge", "good"]
    assert report["unmapped_labels"] == {}


def test_the_largest_roboflow_bucket_maps_to_shift_component(tmp_path: Path) -> None:
    """``component misalignment`` (with a space) is 4192 instances on the real
    export -- the biggest single bucket in the whole merge, and the only
    substantial source of ``shift_component`` found anywhere. It was silently
    dropped as unmapped until the label map covered it."""

    _yolo_pair(tmp_path, "train", "img1", 0, (0.5, 0.5, 0.2, 0.2))
    (tmp_path / "data.yaml").write_text(
        "train: train/images\nnc: 1\nnames:\n  - component misalignment\n",
        encoding="utf-8",
    )
    records, report = load_source(SOURCES["roboflow_soldering"], tmp_path)
    assert [r.label for r in records] == ["shift_component"]
    assert report["unmapped_labels"] == {}


@pytest.mark.parametrize("raw", ["solder residue", "charred solder"])
def test_defects_with_no_taxonomy_class_are_ignored_with_a_reason(
    tmp_path: Path, raw: str
) -> None:
    """Charred solder is an OVER-heating defect; the nearest-looking class,
    cold, is the UNDER-heating one. Mapping it there would teach the opposite
    physical cause, so it is refused rather than approximated."""

    _yolo_pair(tmp_path, "train", "img1", 0, (0.5, 0.5, 0.2, 0.2))
    (tmp_path / "data.yaml").write_text(
        f"train: train/images\nnc: 1\nnames:\n  - {raw}\n", encoding="utf-8"
    )
    records, report = load_source(SOURCES["roboflow_soldering"], tmp_path)
    assert records == []
    # Skipped on purpose with a recorded reason, not silently unmapped.
    assert report["ignored_on_purpose"] == {raw: 1}
    assert report["unmapped_labels"] == {}


# --------------------------------------------------------------------------- #
# Roboflow augmentation grouping
# --------------------------------------------------------------------------- #


def test_roboflow_augmented_copies_collapse_onto_their_source_photo() -> None:
    """Roboflow renames exports to ``<original>_<ext>.rf.<md5>`` and emits one
    per augmented copy. Left as separate groups, copy 1 can train while copy 2
    validates -- the exact leak group-splitting exists to prevent, and it
    inflates every reported number without leaving a trace."""

    stems = [
        "WIN_20221023_15_16_54_Pro_jpg.rf.5c7b7f809b7434e086988928701e5ada",
        "WIN_20221023_15_16_54_Pro_jpg.rf.444d180c9112bd3168330f0b318ac6b3",
        "WIN_20221023_15_16_54_Pro_jpg.rf.51fb873b2a6ccb8601389d91f7c75b8a",
    ]
    groups = {source_group(stem) for stem in stems}
    assert groups == {"WIN_20221023_15_16_54_Pro"}


def test_a_name_that_is_not_a_roboflow_export_is_left_alone() -> None:
    """Inert for every other source -- collapsing unrelated names would merge
    genuinely different scenes into one group and shrink the split."""

    for stem in ("board_017", "WIN_20220329_15_23_17_Pro", "crop__0001"):
        assert source_group(stem) == stem


def test_augmented_copies_of_one_photo_stay_on_the_same_side_of_the_split(
    tmp_path: Path,
) -> None:
    """The property that actually matters, measured through the real reader."""

    for index, digest in enumerate(("a" * 32, "b" * 32, "c" * 32)):
        _yolo_pair(
            tmp_path, "train", f"shot1_jpg.rf.{digest}", 0, (0.5, 0.5, 0.2, 0.2)
        )
    (tmp_path / "data.yaml").write_text(
        "train: train/images\nnc: 1\nnames:\n  - bridge\n", encoding="utf-8"
    )
    records, _ = load_source(SOURCES["roboflow_soldering"], tmp_path)

    assert len(records) == 3
    assert {r.group for r in records} == {"shot1"}, (
        "three augmented copies of one photo must form one group, not three"
    )


def test_yolo_without_any_names_yaml_reports_numeric_labels_visibly(tmp_path: Path) -> None:
    """No data.yaml at all: the fallback to raw class index is still allowed
    (better than dropping the annotations outright), but it must show up in
    unmapped_labels as digit-strings rather than disappear silently -- that
    visibility is what let this exact failure mode get caught and fixed."""

    _yolo_pair(tmp_path, "train", "img1", 0, (0.5, 0.5, 0.2, 0.2))
    records, report = load_source(SOURCES["roboflow_soldering"], tmp_path)
    assert records == []
    assert report["unmapped_labels"] == {"0": 1}


def _labelme_pair(directory: Path, name: str, shapes: list[dict]) -> None:
    """One LabelMe-style image + JSON sidecar, matching SolDef_AI's real layout
    (`Labeled/<name>.jpg` + `<name>.json`, confirmed by inspecting a run)."""

    _image(directory / f"{name}.jpg")
    payload = {
        "version": "5.3.1", "flags": {}, "shapes": shapes,
        "imagePath": f"{name}.jpg", "imageData": None,
        "imageHeight": 24, "imageWidth": 24,
    }
    (directory / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_labelme_sidecars_are_recognised_not_folder_per_class(tmp_path: Path) -> None:
    """This is the exact bug found in a real Kaggle run: a flat folder of
    image+json pairs was misread as folder_per_class (0 class subfolders found)
    and fell through to 'unknown', silently dropping all 428 images to 0
    records. Confirm the layout is identified as its own kind now."""

    labeled = tmp_path / "SolDef_AI" / "Labeled"
    _labelme_pair(labeled, "a", [{"label": "good", "points": [[2, 2], [20, 20]], "shape_type": "rectangle"}])
    _labelme_pair(labeled, "b", [{"label": "misalignment", "points": [[2, 2], [20, 20]], "shape_type": "rectangle"}])
    # The dataset's second top-level folder, unexplored -- must not confuse
    # detection of the Labeled/ layout.
    (tmp_path / "SolDef_AI" / "Dataset" / "CS1").mkdir(parents=True)

    probe = probe_layout(tmp_path / "SolDef_AI")
    assert probe.layout == "labelme"
    assert probe.image_count == 2
    assert len(probe.annotation_files) == 2


def test_labelme_rectangle_and_polygon_shapes_become_bboxes(tmp_path: Path) -> None:
    _labelme_pair(tmp_path, "rect", [
        {"label": "good", "points": [[10, 10], [50, 40]], "shape_type": "rectangle"},
    ])
    _labelme_pair(tmp_path, "poly", [
        {"label": "misalignment", "points": [[5, 5], [30, 5], [32, 45], [3, 48]], "shape_type": "polygon"},
    ])
    records, report = load_source(SOURCES["soldef_ai"], tmp_path)
    by_group = {record.group: record for record in records}

    assert by_group["rect"].label == "good"
    assert by_group["rect"].bbox == (10, 10, 50, 40)

    # A polygon's box is its own extent (min/max over its points), the same
    # approximation every other reader in this module makes for a shape.
    assert by_group["poly"].label == "shift_component"
    assert by_group["poly"].bbox == (3, 5, 32, 48)
    assert report["layout"] == "labelme"


def test_labelme_circle_reconstructs_a_bbox_from_centre_and_radius(tmp_path: Path) -> None:
    """A LabelMe circle stores [centre, one point on the rim] -- NOT two
    corners -- so treating it like a rectangle would collapse it to a sliver."""

    _labelme_pair(tmp_path, "c", [
        {"label": "good", "points": [[50, 50], [80, 50]], "shape_type": "circle"},
    ])
    records, _ = load_source(SOURCES["soldef_ai"], tmp_path)
    assert len(records) == 1
    assert records[0].bbox == (20, 20, 80, 80)


def test_labelme_falls_back_to_the_sibling_image_when_imagepath_is_stale(tmp_path: Path) -> None:
    """LabelMe stores imagePath relative to wherever the annotator's own
    machine had the file, which is routinely wrong once the export moves."""

    _image(tmp_path / "x.jpg")
    payload = {
        "shapes": [{"label": "good", "points": [[1, 1], [10, 10]], "shape_type": "rectangle"}],
        "imagePath": "C:\\Users\\someone\\Desktop\\old_location\\x.jpg",
        "imageHeight": 24, "imageWidth": 24,
    }
    (tmp_path / "x.json").write_text(json.dumps(payload), encoding="utf-8")
    records, _ = load_source(SOURCES["soldef_ai"], tmp_path)
    assert len(records) == 1
    assert records[0].image_path.name == "x.jpg"


def test_labelme_json_with_no_shapes_key_is_not_mistaken_for_labelme(tmp_path: Path) -> None:
    """A JSON that happens to sit next to an image but is not a LabelMe file
    (no 'shapes') must not be swept into this reader."""

    _image(tmp_path / "y.jpg")
    (tmp_path / "y.json").write_text(json.dumps({"note": "unrelated metadata"}), encoding="utf-8")
    assert probe_layout(tmp_path).layout == "unknown"


def test_labelme_unmapped_shape_labels_are_reported_not_guessed(tmp_path: Path) -> None:
    _labelme_pair(tmp_path, "a", [
        {"label": "good", "points": [[1, 1], [10, 10]], "shape_type": "rectangle"},
        {"label": "some_new_defect_nobody_mapped", "points": [[1, 1], [10, 10]], "shape_type": "rectangle"},
    ])
    records, report = load_source(SOURCES["soldef_ai"], tmp_path)
    assert len(records) == 1
    assert report["unmapped_labels"] == {"some_new_defect_nobody_mapped": 1}


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
