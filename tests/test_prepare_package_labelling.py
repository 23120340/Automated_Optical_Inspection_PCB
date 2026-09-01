"""Package-label bootstrap must preserve evidence and isolate old class indices."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_joint_box_app import dataset_id_for, load_rows, main as build_app  # noqa: E402
from scripts.prepare_package_labelling import (  # noqa: E402
    PACKAGE_CLASSES,
    prepare,
    semantic_sha256,
)


REPO = Path(__file__).resolve().parents[1]
PACKAGE_SMOKE = REPO / "tests" / "js" / "package_label_app_smoke.mjs"


@pytest.fixture
def source_dataset(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "component_bodies_round2"
    crops = root / "crops"
    crops.mkdir(parents=True)
    rows = [
        {
            "crop_path": "board_a__tile_000.png", "component_class": "tile ~2 linh kiện",
            "dataset_source": "fixture", "scene_id": "board_a", "crop_w": 200,
            "crop_h": 100, "body_x": 0, "body_y": 0, "body_w": 200, "body_h": 100,
        },
        {
            "crop_path": "board_b__u1.png", "component_class": "IC",
            "dataset_source": "fixture", "scene_id": "board_b", "crop_w": 80,
            "crop_h": 60, "body_x": 2, "body_y": 3, "body_w": 70, "body_h": 50,
        },
        {
            "crop_path": "board_c__empty.png", "component_class": "tile empty",
            "dataset_source": "fixture", "scene_id": "board_c", "crop_w": 64,
            "crop_h": 64, "body_x": 0, "body_y": 0, "body_w": 64, "body_h": 64,
        },
    ]
    for row in rows:
        (crops / str(row["crop_path"])).write_bytes(b"fixture")
    with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    loaded = load_rows(root / "manifest.csv", crops)
    source_classes = ["component"]
    source_id = dataset_id_for(root, loaded, source_classes)
    source = root / "draft_boxes.json"
    source.write_text(json.dumps({
        "schema": "aoi-joint-boxes/1.0",
        "dataset_id": source_id,
        "dataset": root.name,
        "reviewer_id": "fixture-reviewer",
        "coordinate_space": "crop_pixels_top_left_origin",
        "classes": source_classes,
        "crops": {
            rows[0]["crop_path"]: {
                "status": "verified", "notes": "generic boxes",
                "boxes": [
                    {"cls": "component", "x": 1.25, "y": 2.5, "w": 30.75, "h": 20.125},
                    {"cls": "component", "package": "SOIC-16",
                     "x": 50, "y": 10, "w": 40, "h": 18},
                ],
            },
            rows[1]["crop_path"]: {
                "status": "verified", "notes": "explicit package",
                "boxes": [{"cls": "component", "package_class": "QFN-32",
                           "x": 4, "y": 5, "w": 42, "h": 40}],
            },
            rows[2]["crop_path"]: {
                "status": "verified", "notes": "explicitly empty", "boxes": [],
            },
        },
    }, ensure_ascii=False), encoding="utf-8")
    return root, source


def test_prepare_preserves_order_and_exact_geometry(source_dataset: tuple[Path, Path]) -> None:
    root, source = source_dataset
    output = root / "draft_package_boxes.json"

    assert prepare(source, output) == 0
    old = json.loads(source.read_text(encoding="utf-8"))
    new = json.loads(output.read_text(encoding="utf-8"))

    assert new["classes"] == list(PACKAGE_CLASSES)
    assert new["unknown_class"] == "unknown"
    assert new["dataset_id"] != old["dataset_id"]
    assert new["migration_aliases"] == [{
        "dataset_id": old["dataset_id"],
        "classes": old["classes"],
        "source_crops_semantic_sha256": semantic_sha256(old["crops"]),
        "box_geometry_semantic_sha256": new["box_geometry_semantic_sha256"],
        "strategy": "preserve_geometry_reset_box_classes_to_unknown",
    }]

    for path, old_record in old["crops"].items():
        new_record = new["crops"][path]
        assert [tuple(box[field] for field in ("x", "y", "w", "h"))
                for box in new_record["boxes"]] == [
            tuple(box[field] for field in ("x", "y", "w", "h"))
            for box in old_record["boxes"]
        ]
    first = new["crops"]["board_a__tile_000.png"]
    assert first["status"] == ""
    assert first["source_status"] == "verified"
    assert [box["cls"] for box in first["boxes"]] == ["unknown", "ic_hai_ben"]
    assert all(box["source_cls"] == "component" for box in first["boxes"])
    assert new["crops"]["board_b__u1.png"]["boxes"][0]["cls"] == "ic_khong_chan"
    assert new["crops"]["board_c__empty.png"]["status"] == "verified"


def test_prepare_is_no_overwrite_and_dry_run_writes_nothing(
    source_dataset: tuple[Path, Path],
) -> None:
    root, source = source_dataset
    output = root / "draft_package_boxes.json"
    assert prepare(source, output, dry_run=True) == 0
    assert not output.exists()
    assert prepare(source, output) == 0
    original = output.read_bytes()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        prepare(source, output)
    assert output.read_bytes() == original


def test_builder_rejects_a_verified_unknown_package(
    source_dataset: tuple[Path, Path],
) -> None:
    root, source = source_dataset
    output = root / "draft_package_boxes.json"
    prepare(source, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["crops"]["board_a__tile_000.png"]["status"] = "verified"
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit, match="verified but still contains"):
        build_app([str(root), "--classes", *PACKAGE_CLASSES, "--seed-json", str(output)])


@pytest.mark.skipif(shutil.which("node") is None, reason="needs a Node runtime")
def test_package_page_migrates_old_browser_state_without_reusing_classes(
    source_dataset: tuple[Path, Path],
) -> None:
    root, source = source_dataset
    seed = root / "draft_package_boxes.json"
    page = root / "label_packages.html"
    prepare(source, seed)
    assert build_app([
        str(root), "--classes", *PACKAGE_CLASSES,
        "--seed-json", str(seed), "--output", str(page),
    ]) == 0

    result = subprocess.run(
        ["node", str(PACKAGE_SMOKE), str(page)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")
    assert "ok: package migration" in result.stdout
