"""The package trainer may consume only complete, board-separated labels."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aoi_pipeline.config import PACKAGE_CLASSES  # noqa: E402
from scripts.pack_package_classification_dataset import pack  # noqa: E402


def _dataset(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "package_labels"
    (root / "crops").mkdir(parents=True)
    rows: list[dict[str, object]] = []
    records: dict[str, object] = {}
    for board_index in range(3):
        crop_path = f"board_{board_index}.png"
        image = np.full((24, 112, 3), 80 + board_index * 20, dtype=np.uint8)
        assert cv2.imwrite(str(root / "crops" / crop_path), image)
        rows.append({
            "crop_path": crop_path,
            "component_class": "tile ~7 linh kiện",
            "dataset_source": "fixture",
            "scene_id": f"board_{board_index}",
            "crop_w": 112, "crop_h": 24,
            "body_x": 0, "body_y": 0, "body_w": 112, "body_h": 24,
        })
        records[crop_path] = {
            "status": "verified",
            "notes": "",
            "boxes": [
                {"cls": name, "x": 2 + index * 15, "y": 5, "w": 12, "h": 12}
                for index, name in enumerate(PACKAGE_CLASSES)
            ],
        }
    with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dataset_id = hashlib.sha256(
        f"{root.name}|{len(rows)}|{rows[0]['crop_path']}|{','.join(PACKAGE_CLASSES)}".encode()
    ).hexdigest()[:16]
    labels = root / "package_boxes.json"
    labels.write_text(json.dumps({
        "schema": "aoi-joint-boxes/1.0",
        "dataset_id": dataset_id,
        "dataset": root.name,
        "reviewer_id": "fixture",
        "coordinate_space": "crop_pixels_top_left_origin",
        "classes": list(PACKAGE_CLASSES),
        "crops": records,
    }), encoding="utf-8")
    return root, labels


def test_pack_writes_leak_free_imagefolder_and_auditable_manifest(tmp_path: Path) -> None:
    root, labels = _dataset(tmp_path)
    output = root / "package.zip"
    assert pack(labels, output) == 0

    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("dataset_manifest.json"))
        members = set(archive.namelist())
    assert manifest["class_names"] == list(PACKAGE_CLASSES)
    assert manifest["split_unit"] == "board_scene_id"
    groups = [set(manifest["split_groups"][split]) for split in ("train", "val", "test")]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert len(manifest["samples"]) == 21
    for split in ("train", "val", "test"):
        assert set(manifest["split_class_counts"][split]) == set(PACKAGE_CLASSES)
        assert all(manifest["split_class_counts"][split].values())
    assert all(sample["path"] in members for sample in manifest["samples"])


def test_pack_rejects_editor_unknown_and_never_overwrites(tmp_path: Path) -> None:
    root, labels = _dataset(tmp_path)
    payload = json.loads(labels.read_text(encoding="utf-8"))
    payload["unknown_class"] = "unknown"
    labels.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="draft with unknown_class"):
        pack(labels, root / "package.zip")

    payload.pop("unknown_class")
    labels.write_text(json.dumps(payload), encoding="utf-8")
    output = root / "package.zip"
    pack(labels, output)
    original = output.read_bytes()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        pack(labels, output)
    assert output.read_bytes() == original


def test_pack_requires_every_manifest_crop_to_have_a_review_disposition(
    tmp_path: Path,
) -> None:
    root, labels = _dataset(tmp_path)
    payload = json.loads(labels.read_text(encoding="utf-8"))
    payload["crops"].pop("board_2.png")
    labels.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="account for every manifest crop"):
        pack(labels, root / "package.zip")
