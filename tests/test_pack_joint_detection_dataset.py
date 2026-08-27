"""Tests for packing labelled crops into a YOLO detection set.

The two failures worth guarding are silent ones. A crop-level split reports a
validation score that measures memorisation of a board photograph, and an
unreviewed crop written out with an empty label file reads to YOLO as a
confident "no defects here". Neither raises anything at training time.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import zipfile

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.crop_components_for_labelling import main as crop_main  # noqa: E402
from scripts.pack_joint_detection_dataset import assign_splits, main  # noqa: E402

CLASSES = ["Bad_podu", "Bad_qiaojiao"]


@pytest.fixture
def crops(tmp_path: Path) -> Path:
    source = tmp_path / "src.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("data.yaml", "nc: 1\nnames: ['IC']\n")
        for stem in [f"board{i}_jpg" for i in range(8)]:
            buffer = io.BytesIO()
            Image.new("RGB", (400, 400), (10, 80, 40)).save(buffer, format="PNG")
            archive.writestr(f"train/images/{stem}.png", buffer.getvalue())
            archive.writestr(f"train/labels/{stem}.txt",
                             "0 0.5 0.5 0.25 0.2\n0 0.25 0.75 0.2 0.2\n")
    out = tmp_path / "crops_out"
    assert crop_main([str(source), "--output", str(out)]) == 0
    return out


def write_boxes(path: Path, crops_dir: Path, *, verified: int, clean: int,
                skipped: int = 0) -> None:
    import csv

    rows = list(csv.DictReader((crops_dir / "manifest.csv").open(encoding="utf-8")))
    payload = {
        "schema": "aoi-joint-boxes/1.0",
        "dataset_id": "test",
        "reviewer_id": "qnn",
        "exported_at": "2026-08-26T00:00:00Z",
        "coordinate_space": "crop_pixels_top_left_origin",
        "classes": CLASSES,
        "crops": {},
    }
    cursor = 0
    for _ in range(verified):
        row = rows[cursor]; cursor += 1
        payload["crops"][row["crop_path"]] = {
            "status": "verified", "notes": "",
            "boxes": [{"cls": "Bad_podu", "x": 5, "y": 5, "w": 20, "h": 12}],
        }
    for _ in range(clean):
        row = rows[cursor]; cursor += 1
        payload["crops"][row["crop_path"]] = {"status": "verified", "notes": "", "boxes": []}
    for _ in range(skipped):
        row = rows[cursor]; cursor += 1
        payload["crops"][row["crop_path"]] = {"status": "skipped", "notes": "", "boxes": []}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_assign_splits_never_puts_a_scene_in_two_places() -> None:
    scenes = [f"board{i}" for i in range(20)]
    mapping = assign_splits(scenes, (0.7, 0.15, 0.15), seed=17)
    assert set(mapping) == set(scenes)
    assert set(mapping.values()) == {"train", "valid", "test"}
    assert assign_splits(scenes, (0.7, 0.15, 0.15), seed=17) == mapping, "not deterministic"


def test_crops_from_one_scene_stay_on_one_side(tmp_path: Path, crops: Path) -> None:
    boxes = tmp_path / "joint_boxes.json"
    write_boxes(boxes, crops, verified=8, clean=8)
    out = tmp_path / "pack"
    assert main([str(crops), "--boxes", str(boxes), "--output", str(out)]) == 0

    manifest = json.loads((out / "pack_manifest.json").read_text(encoding="utf-8"))
    assert manifest["split_by"] == "scene_id"
    seen: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        for image in (out / split / "images").iterdir():
            scene = image.name.split("__")[0]
            assert seen.setdefault(scene, split) == split, (
                f"scene {scene} appears in both {seen[scene]} and {split}")


def test_unreviewed_crops_are_never_written(tmp_path: Path, crops: Path) -> None:
    boxes = tmp_path / "joint_boxes.json"
    write_boxes(boxes, crops, verified=2, clean=2, skipped=3)
    out = tmp_path / "pack"
    main([str(crops), "--boxes", str(boxes), "--output", str(out)])

    written = {p.name for split in ("train", "valid", "test")
               for p in (out / split / "images").iterdir()}
    assert len(written) == 4, "only verified crops belong in the pack"
    manifest = json.loads((out / "pack_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["labelled_but_unused"]) == 3
    assert set(manifest["labelled_but_unused"].values()) == {"skipped"}


def test_clean_crops_become_empty_label_files(tmp_path: Path, crops: Path) -> None:
    boxes = tmp_path / "joint_boxes.json"
    write_boxes(boxes, crops, verified=0, clean=6)
    out = tmp_path / "pack"
    main([str(crops), "--boxes", str(boxes), "--output", str(out)])
    labels = [p for split in ("train", "valid", "test")
              for p in (out / split / "labels").iterdir()]
    assert labels and all(p.read_text(encoding="utf-8") == "" for p in labels)
    manifest = json.loads((out / "pack_manifest.json").read_text(encoding="utf-8"))
    assert manifest["clean_ratio"] == 1.0


def test_boxes_are_normalised_inside_the_unit_square(tmp_path: Path, crops: Path) -> None:
    import csv

    rows = list(csv.DictReader((crops / "manifest.csv").open(encoding="utf-8")))
    row = rows[0]
    boxes = tmp_path / "joint_boxes.json"
    # deliberately dragged past the right and bottom edges
    boxes.write_text(json.dumps({
        "schema": "aoi-joint-boxes/1.0", "classes": CLASSES,
        "coordinate_space": "crop_pixels_top_left_origin",
        "crops": {row["crop_path"]: {"status": "verified", "boxes": [
            {"cls": "Bad_qiaojiao", "x": -5, "y": -5,
             "w": int(row["crop_w"]) + 40, "h": int(row["crop_h"]) + 40}]}},
    }), encoding="utf-8")
    out = tmp_path / "pack"
    main([str(crops), "--boxes", str(boxes), "--output", str(out)])
    text = next(p for split in ("train", "valid", "test")
                for p in (out / split / "labels").iterdir()).read_text(encoding="utf-8")
    index, cx, cy, w, h = text.split()
    assert index == "1"
    assert all(0.0 <= float(v) <= 1.0 for v in (cx, cy, w, h))
    assert float(w) == pytest.approx(1.0) and float(h) == pytest.approx(1.0)


def test_merge_refuses_a_mismatched_class_list(tmp_path: Path, crops: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (other / "data.yaml").write_text("nc: 1\nnames: ['extra__solder']\n", encoding="utf-8")
    boxes = tmp_path / "joint_boxes.json"
    write_boxes(boxes, crops, verified=2, clean=2)
    with pytest.raises(SystemExit) as excinfo:
        main([str(crops), "--boxes", str(boxes), "--output", str(tmp_path / "pack"),
              "--merge", str(other)])
    assert "Reconcile" in str(excinfo.value)


def test_merge_carries_the_other_dataset_in_with_a_source_prefix(
    tmp_path: Path, crops: Path
) -> None:
    other = tmp_path / "roboflow_like"
    (other / "train" / "images").mkdir(parents=True)
    (other / "train" / "labels").mkdir(parents=True)
    (other / "data.yaml").write_text(f"nc: 2\nnames: {CLASSES!r}\n", encoding="utf-8")
    Image.new("RGB", (64, 64)).save(other / "train" / "images" / "shot1.jpg")
    (other / "train" / "labels" / "shot1.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    boxes = tmp_path / "joint_boxes.json"
    write_boxes(boxes, crops, verified=4, clean=4)
    out = tmp_path / "pack"
    main([str(crops), "--boxes", str(boxes), "--output", str(out), "--merge", str(other)])

    names = {p.name for p in (out / "train" / "images").iterdir()}
    assert "roboflow_like__shot1.jpg" in names, "merged files must stay attributable"
    manifest = json.loads((out / "pack_manifest.json").read_text(encoding="utf-8"))
    assert manifest["merged_images"] == 1


def test_rejects_an_export_from_a_different_crop_set(tmp_path: Path, crops: Path) -> None:
    boxes = tmp_path / "joint_boxes.json"
    boxes.write_text(json.dumps({
        "schema": "aoi-joint-boxes/1.0", "classes": CLASSES,
        "coordinate_space": "crop_pixels_top_left_origin",
        "crops": {"not_from_here.png": {"status": "verified", "boxes": []}},
    }), encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        main([str(crops), "--boxes", str(boxes), "--output", str(tmp_path / "pack")])
    assert "different crop set" in str(excinfo.value)
