"""Continuation safety for the component-body labelling queue.

The reviewed checkpoint is evidence, not a cache: successor queues may replace
duplicate/unusable images, but they must keep every verified pixel and record
exactly.  These tests exercise that contract on small synthetic tile aliases.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aoi_pipeline.models import BoundingBox  # noqa: E402
from scripts.build_joint_box_app import dataset_id_for, load_rows  # noqa: E402
from scripts.pack_component_detection_dataset import (  # noqa: E402
    _expected_checkpoint_dataset_id,
)
from scripts.prepare_component_labelling import prepare  # noqa: E402
from scripts.prelabel_component_bodies import prelabel  # noqa: E402


MANIFEST_FIELDS = [
    "crop_path", "component_class", "dataset_source", "scene_id",
    "crop_w", "crop_h", "body_x", "body_y", "body_w", "body_h",
    "roi_kind", "label_status", "reviewer_id", "notes",
]


def _write_rgb(path: Path, colour: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), colour).save(path)


def _manifest_row(name: str, source: str, components: int) -> dict[str, object]:
    return {
        "file": name,
        "source": f"{source}.jpg",
        "x": 0,
        "y": 0,
        "tile": 32,
        "components": components,
        "labels": {"ic": components},
        "dark_fraction": 0.1,
    }


def _old_csv_row(name: str, scene: str) -> dict[str, object]:
    return {
        "crop_path": name,
        "component_class": "tile ~8 linh kiện",
        "dataset_source": "tiles_1024",
        "scene_id": scene,
        "crop_w": 32,
        "crop_h": 24,
        "body_x": 0,
        "body_y": 0,
        "body_w": 0,
        "body_h": 0,
        "roi_kind": "board_tile",
        "label_status": "",
        "reviewer_id": "",
        "notes": "",
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _pixel_hash(path: Path) -> str:
    with Image.open(path) as handle:
        rgb = handle.convert("RGB")
    digest = hashlib.sha256()
    digest.update(rgb.width.to_bytes(8, "big"))
    digest.update(rgb.height.to_bytes(8, "big"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _queue_fixture(tmp_path: Path) -> dict[str, object]:
    tiles = tmp_path / "tiles"
    old = tmp_path / "component_bodies"
    names = {
        "verified": "pcb1__rec1__1024__0___0.png",
        "verified_alias": "pcb_dslr_001__rec1__1024__0___0.png",
        "bad": "pcb2__rec1__1024__0___0.png",
        "bad_alias": "pcb_dslr_002__rec1__1024__0___0.png",
        "old": "pcb3__rec1__1024__0___0.png",
        "old_alias": "pcb_dslr_003__rec1__1024__0___0.png",
        "new4": "pcb4__rec1__1024__0___0.png",
        "new5": "pcb5__rec1__1024__0___0.png",
        "new6": "pcb6__rec1__1024__0___0.png",
    }
    rows = [
        _manifest_row(names["verified"], "pcb1__rec1", 12),
        _manifest_row(names["verified_alias"], "pcb_dslr_001__rec1", 12),
        _manifest_row(names["bad"], "pcb2__rec1", 11),
        _manifest_row(names["bad_alias"], "pcb_dslr_002__rec1", 11),
        _manifest_row(names["old"], "pcb3__rec1", 10),
        _manifest_row(names["old_alias"], "pcb_dslr_003__rec1", 10),
        _manifest_row(names["new4"], "pcb4__rec1", 9),
        _manifest_row(names["new5"], "pcb5__rec1", 8),
        _manifest_row(names["new6"], "pcb6__rec1", 7),
    ]
    colours = {
        names["verified"]: (10, 20, 30),
        names["verified_alias"]: (10, 20, 30),
        names["bad"]: (40, 50, 60),
        names["bad_alias"]: (40, 50, 60),
        names["old"]: (70, 80, 90),
        names["old_alias"]: (70, 80, 90),
        names["new4"]: (100, 110, 120),
        names["new5"]: (130, 140, 150),
        names["new6"]: (160, 170, 180),
    }
    for name, colour in colours.items():
        _write_rgb(tiles / name, colour)
    (tiles / "tiles_manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8"
    )

    old_names = [
        names["verified"], names["verified_alias"], names["bad"],
        names["bad_alias"], names["old"], names["old_alias"],
    ]
    old_rows = [_old_csv_row(name, name.split("__1024", 1)[0]) for name in old_names]
    _write_csv(old / "manifest.csv", old_rows)
    for name in old_names:
        target = old / "crops" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((tiles / name).read_bytes())

    verified_record = {
        "status": "verified",
        "notes": "human reviewed",
        "boxes": [{"cls": "component", "x": -2, "y": 3, "w": 12, "h": 9}],
    }
    checkpoint_payload = {
        "schema": "aoi-joint-boxes/1.0",
        "dataset": old.name,
        "dataset_id": _expected_checkpoint_dataset_id(old, old_rows),
        "reviewer_id": "reviewer",
        "exported_at": "2026-08-30T00:00:00Z",
        "coordinate_space": "crop_pixels_top_left_origin",
        "classes": ["component"],
        "crops": {
            names["verified"]: verified_record,
            names["verified_alias"]: {
                "status": "unusable", "notes": "duplicate", "boxes": [],
            },
            names["bad"]: {"status": "unusable", "notes": "bad", "boxes": []},
        },
    }
    checkpoint = tmp_path / "joint_boxes.json"
    checkpoint.write_text(json.dumps(checkpoint_payload), encoding="utf-8")

    base_draft = old / "draft_boxes.json"
    base_draft.write_text(json.dumps({
        "schema": "aoi-joint-boxes/1.0",
        "dataset": old.name,
        "coordinate_space": "crop_pixels_top_left_origin",
        "classes": ["component"],
        "crops": {
            name: {
                "status": "", "notes": "old detector draft",
                "boxes": [{"cls": "component", "x": 2, "y": 2, "w": 5, "h": 5}],
            }
            for name in old_names
        },
    }), encoding="utf-8")
    return {
        "tiles": tiles,
        "old": old,
        "checkpoint": checkpoint,
        "checkpoint_payload": checkpoint_payload,
        "verified_record": verified_record,
        "base_draft": base_draft,
        "names": names,
    }


def test_successor_queue_deduplicates_rejects_and_replaces_without_losing_verified(
    tmp_path: Path,
) -> None:
    fixture = _queue_fixture(tmp_path)
    output = tmp_path / "component_bodies_round2"
    assert prepare(
        fixture["tiles"], output, limit=4, max_dark=0.6,
        checkpoint=fixture["checkpoint"], checkpoint_root=fixture["old"],
        max_per_board=2,
    ) == 0

    rows = list(csv.DictReader((output / "manifest.csv").open(encoding="utf-8")))
    selected = {row["crop_path"] for row in rows}
    names = fixture["names"]
    assert len(rows) == 4
    assert names["verified"] in selected
    assert names["verified_alias"] not in selected
    assert names["bad"] not in selected and names["bad_alias"] not in selected
    hashes = [_pixel_hash(output / "crops" / name) for name in selected]
    assert len(hashes) == len(set(hashes)) == 4

    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["checkpoint"]["verified_carried"] == 1
    assert provenance["checkpoint"]["verified_boxes_carried"] == 1
    assert provenance["unique_pixel_groups_prepared"] == 4
    assert provenance["new_tiles_from_source_pool"] == 2


def test_continuation_draft_copies_verified_record_and_only_infers_new_tiles(
    tmp_path: Path, monkeypatch,
) -> None:
    fixture = _queue_fixture(tmp_path)
    output = tmp_path / "component_bodies_round2"
    prepare(
        fixture["tiles"], output, limit=4, max_dark=0.6,
        checkpoint=fixture["checkpoint"], checkpoint_root=fixture["old"],
        max_per_board=2,
    )

    calls: list[int] = []

    class Detector:
        def detect(self, image):
            calls.append(1)
            return [SimpleNamespace(bbox=BoundingBox(1, 1, 9, 7))]

    monkeypatch.setattr(
        "aoi_pipeline.detection.detectors.create_detector",
        lambda *_args, **_kwargs: Detector(),
    )
    assert prelabel(
        output, tmp_path / "unused.onnx", confidence=0.3, max_aspect=3.0,
        dry_run=False, checkpoint=fixture["checkpoint"],
        previous_folder=fixture["old"], base_draft=fixture["base_draft"],
    ) == 0

    result = json.loads((output / "draft_boxes.json").read_text(encoding="utf-8"))
    verified_name = fixture["names"]["verified"]
    assert result["crops"][verified_name] == fixture["verified_record"]
    assert {record["status"] for name, record in result["crops"].items()
            if name != verified_name} == {""}
    # One retained old draft plus two genuinely new tiles; only the latter infer.
    assert len(calls) == 2
    app_rows = load_rows(output / "manifest.csv", output / "crops")
    assert result["dataset_id"] == dataset_id_for(output, app_rows, ["component"])

    semantic = json.dumps(
        {verified_name: fixture["verified_record"]},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    assert result["carried_verified_semantic_sha256"] == hashlib.sha256(semantic).hexdigest()


def test_prepare_never_overwrites_an_existing_successor_folder(tmp_path: Path) -> None:
    fixture = _queue_fixture(tmp_path)
    output = tmp_path / "component_bodies_round2"
    output.mkdir()
    sentinel = output / "reviewed.txt"
    sentinel.write_text("keep", encoding="utf-8")
    try:
        prepare(fixture["tiles"], output, limit=4)
    except SystemExit as exc:
        assert "không ghi đè" in str(exc)
    else:
        raise AssertionError("existing output was accepted")
    assert sentinel.read_text(encoding="utf-8") == "keep"

