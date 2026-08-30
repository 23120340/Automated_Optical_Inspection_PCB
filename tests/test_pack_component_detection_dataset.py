"""Contract tests for the component-body detection dataset packer.

The fixtures are deliberately tiny, but preserve the two input contracts that
matter in production: the browser checkpoint is immutable reviewed evidence,
and the public archives must keep their exact Roboflow metadata/taxonomy.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pack_component_detection_dataset import (  # noqa: E402
    LOCAL_TAG,
    RF100_AMBIGUOUS_CLASSES,
    RF100_CLASSES,
    RF100_TAG,
    WINNIES_CLASSES,
    WINNIES_TAG,
    _load_local,
    _load_public_archive,
    assign_splits,
    build_plan,
    canonical_board_id,
    main,
)


RF100_EXPECTED_CLASSES = (
    "Button",
    "Capacitor Jumper",
    "Capacitor",
    "Clock",
    "Connector",
    "Diode",
    "EM",
    "Electrolytic Capacitor",
    "Ferrite Bead",
    "IC",
    "Inductor",
    "Jumper",
    "Led",
    "Pads",
    "Pins",
    "Resistor Jumper",
    "Resistor Network",
    "Resistor",
    "Switch",
    "Test Point",
    "Transistor",
    "Unknown Unlabeled",
    "iC",
)
WINNIES_EXPECTED_CLASSES = (
    "CHIP",
    "LED",
    "MOSFET",
    "MOSFET-2",
    "Polyfuse_GR",
    "Polyfuse_Z",
    "Resistor rond",
    "SOD123",
    "SOD128",
    "SOD323",
    "SOIC-12",
    "SOIC-14",
    "SOIC-16",
    "SOT143",
    "SOT223",
    "SOT23",
    "SOT457",
    "SOT753",
    "SOT96",
    "TSSOP-14",
    "TSSOP-16",
    "capacitor",
    "feriet kraal",
    "resistor",
)
PUBLIC_METADATA = {
    RF100_TAG: {
        "workspace": "roboflow-100",
        "project": "printed-circuit-board",
        "version": "4",
        "license": "CC BY 4.0",
        "classes": RF100_EXPECTED_CLASSES,
    },
    WINNIES_TAG: {
        "workspace": "winnies-workspace-0yaec",
        "project": "pcb-components-wc8ms",
        "version": "3",
        "license": "CC BY 4.0",
        "classes": WINNIES_EXPECTED_CLASSES,
    },
}
HASHES = tuple(character * 32 for character in "abcdef0123456789")


def _image_bytes(
    size: tuple[int, int], color: tuple[int, int, int]
) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG", quality=100)
    return buffer.getvalue()


def _write_public_archive(
    path: Path,
    source_tag: str,
    images: list[
        tuple[
            str,
            tuple[int, int],
            tuple[int, int, int],
            list[tuple[str, float, float, float, float]],
        ]
    ],
) -> Path:
    contract = PUBLIC_METADATA[source_tag]
    classes = contract["classes"]
    yaml = (
        f"names: {list(classes)!r}\n"
        f"nc: {len(classes)}\n"
        "roboflow:\n"
        f"  workspace: {contract['workspace']}\n"
        f"  project: {contract['project']}\n"
        f"  version: {contract['version']}\n"
        f"  license: {contract['license']}\n"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data.yaml", yaml)
        for member, size, color, labels in images:
            archive.writestr(member, _image_bytes(size, color))
            label_member = member.replace("/images/", "/labels/")
            label_member = str(Path(label_member).with_suffix(".txt")).replace(
                "\\", "/"
            )
            lines = [
                f"{classes.index(name)} {cx} {cy} {width} {height}"
                for name, cx, cy, width, height in labels
            ]
            archive.writestr(label_member, "\n".join(lines) + "\n")
    return path


def _minimal_public_archives(tmp_path: Path) -> tuple[Path, Path]:
    rf100 = _write_public_archive(
        tmp_path / "rf100.zip",
        RF100_TAG,
        [
            (
                f"train/images/rf_seed.rf.{HASHES[0]}.jpg",
                (28, 24),
                (210, 20, 20),
                [("IC", 0.5, 0.5, 0.4, 0.4)],
            )
        ],
    )
    winnies = _write_public_archive(
        tmp_path / "winnies.zip",
        WINNIES_TAG,
        [
            (
                f"train/images/winnie_seed.rf.{HASHES[1]}.jpg",
                (30, 26),
                (20, 210, 20),
                [("CHIP", 0.5, 0.5, 0.3, 0.3)],
            )
        ],
    )
    return rf100, winnies


def _write_local_checkpoint(
    root: Path,
    records: list[dict[str, object]],
) -> tuple[Path, Path]:
    crop_root = root / "component_bodies"
    crops_dir = crop_root / "crops"
    crops_dir.mkdir(parents=True)
    manifest_rows: list[dict[str, object]] = []
    exported: dict[str, object] = {}
    for index, record in enumerate(records):
        name = str(record.get("name", f"tile_{index}.jpg"))
        size = tuple(record.get("size", (36, 28)))
        blob = record.get("blob")
        if blob is None:
            color = tuple(record.get("color", (20 + index * 17, 40, 70)))
            blob = _image_bytes(size, color)
        else:
            with Image.open(io.BytesIO(blob)) as image:
                size = image.size
        (crops_dir / name).write_bytes(blob)
        manifest_rows.append(
            {
                "crop_path": name,
                "scene_id": str(record.get("scene", f"board_{index}")),
                "crop_w": size[0],
                "crop_h": size[1],
            }
        )
        status = record.get("status")
        if status is not None:
            exported[name] = {
                "status": status,
                "notes": str(record.get("notes", "kept from reviewer")),
                "boxes": record.get(
                    "boxes",
                    [
                        {
                            "cls": "component",
                            "x": 4,
                            "y": 3,
                            "w": 10,
                            "h": 8,
                        }
                    ],
                ),
            }
    manifest = crop_root / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("crop_path", "scene_id", "crop_w", "crop_h")
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    boxes = root / "joint_boxes.json"
    dataset_id = hashlib.sha256(
        f"{crop_root.name}|{len(manifest_rows)}|"
        f"{manifest_rows[0]['crop_path']}|component".encode()
    ).hexdigest()[:16]
    boxes.write_text(
        json.dumps(
            {
                "schema": "aoi-joint-boxes/1.0",
                "dataset_id": dataset_id,
                "dataset": crop_root.name,
                "reviewer_id": "fixture-reviewer",
                "exported_at": "2026-08-29T05:29:17.157Z",
                "coordinate_space": "crop_pixels_top_left_origin",
                "classes": ["component"],
                "crops": exported,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return crop_root, boxes


def _verified_boards(count: int) -> list[dict[str, object]]:
    return [
        {
            "name": f"pcb_dslr_{board:03d}__rec1__tile.jpg",
            "scene": f"pcb_dslr_{board:03d}__rec1",
            "color": (20 + board * 13, 30 + board, 80),
            "status": "verified",
        }
        for board in range(1, count + 1)
    ]


def _snapshot_files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _pcb_dslr_audit_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    reference_root = tmp_path / "reference"
    board_dir = reference_root / "boards" / "pcb_dslr_001"
    board_dir.mkdir(parents=True)
    image_path = board_dir / "rec1.jpg"
    Image.new("RGB", (80, 60), (30, 110, 60)).save(
        image_path, format="JPEG", quality=100
    )
    with Image.open(image_path) as board:
        tile_buffer = io.BytesIO()
        board.convert("RGB").crop((10, 10, 50, 40)).save(tile_buffer, format="PNG")
    annotation_path = board_dir / "rec1-annot.txt"
    # Full-board coordinates; local tile AABB is [8, 6, 24, 20].
    annotation_path.write_text("26 23 16 14 0 TEST_IC\n", encoding="utf-8")
    manifest = {
        "schema_version": "aoi-reference-source-set/2.0",
        "source_dataset": "TU Wien PCB DSLR",
        "files": [
            {
                "board_id": "pcb_dslr_001",
                "recording_id": "rec1",
                "image_path": "boards/pcb_dslr_001/rec1.jpg",
                "annotation_path": "boards/pcb_dslr_001/rec1-annot.txt",
                "width": 80,
                "height": 60,
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "annotation_sha256": hashlib.sha256(
                    annotation_path.read_bytes()
                ).hexdigest(),
            }
        ],
    }
    (reference_root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    tile_name = "pcb_dslr_001__rec1__40__10___10.png"
    crop_root, boxes = _write_local_checkpoint(
        tmp_path / "local",
        [
            {
                "name": tile_name,
                "scene": "pcb_dslr_001__rec1",
                "blob": tile_buffer.getvalue(),
                "status": "verified",
                # Tighter body-only box inside the official rotated-rect AABB.
                "boxes": [
                    {"cls": "component", "x": 10, "y": 8, "w": 12, "h": 10}
                ],
            }
        ],
    )
    tile_manifest = tmp_path / "tiles_manifest.json"
    tile_manifest.write_text(
        json.dumps(
            [
                {
                    "file": tile_name,
                    "source": "pcb1__rec1.jpg",
                    "x": 10,
                    "y": 10,
                    "tile": 40,
                }
            ]
        ),
        encoding="utf-8",
    )
    rf100, winnies = _minimal_public_archives(tmp_path)
    return crop_root, boxes, rf100, winnies, reference_root, tile_manifest


@pytest.mark.parametrize(
    ("name", "source_tag"),
    [
        (f"pcb7rec1_jpg.rf.{HASHES[0]}.jpg", RF100_TAG),
        ("pcb7__rec5__x0_y0.png", LOCAL_TAG),
        ("pcb_dslr_007__rec1__x768_y0.jpg", LOCAL_TAG),
    ],
)
def test_canonical_pcb_dslr_aliases_share_one_physical_board(
    name: str, source_tag: str
) -> None:
    assert canonical_board_id(name, source_tag) == "pcb_dslr:007"


def test_existing_board_split_never_changes_when_new_groups_are_added() -> None:
    existing = [f"pcb_dslr:{board:03d}" for board in range(1, 11)]
    before = assign_splits(existing, (0.7, 0.15, 0.15), seed=17)
    after = assign_splits(
        [*existing, "pcb_dslr:011"], (0.7, 0.15, 0.15), seed=17
    )
    assert {group: after[group] for group in existing} == before


def test_public_taxonomies_and_metadata_match_the_expected_exports(
    tmp_path: Path,
) -> None:
    assert RF100_CLASSES == RF100_EXPECTED_CLASSES
    assert WINNIES_CLASSES == WINNIES_EXPECTED_CLASSES
    rf100, winnies = _minimal_public_archives(tmp_path)

    _rf_images, rf_report = _load_public_archive(rf100, RF100_TAG)
    _win_images, win_report = _load_public_archive(winnies, WINNIES_TAG)

    assert rf_report["class_contract"] == list(RF100_EXPECTED_CLASSES)
    assert (rf_report["workspace"], rf_report["project"], rf_report["version"]) == (
        "roboflow-100",
        "printed-circuit-board",
        4,
    )
    assert rf_report["declared_license"] == "CC BY 4.0"
    assert win_report["class_contract"] == list(WINNIES_EXPECTED_CLASSES)
    assert (win_report["workspace"], win_report["project"], win_report["version"]) == (
        "winnies-workspace-0yaec",
        "pcb-components-wc8ms",
        3,
    )
    assert win_report["declared_license"] == "CC BY 4.0"


def test_local_loader_uses_verified_only_without_mutating_reviewed_inputs(
    tmp_path: Path,
) -> None:
    crop_root, boxes = _write_local_checkpoint(
        tmp_path,
        [
            {"scene": "pcb_dslr_001__rec1", "status": "verified"},
            {"scene": "pcb_dslr_002__rec1", "status": "unusable"},
            {"scene": "pcb_dslr_003__rec1", "status": "skipped"},
            {"scene": "pcb_dslr_004__rec1", "status": None},
        ],
    )
    before = _snapshot_files(tmp_path)

    images, report = _load_local(crop_root, boxes)

    assert [image.group_id for image in images] == ["pcb_dslr:001"]
    assert report["status_counts"] == {
        "skipped": 1,
        "unusable": 1,
        "verified": 1,
    }
    assert report["unreviewed_not_in_export"] == 1
    assert report["verified_before_exact_dedup"] == 1
    assert report["verified_after_exact_dedup"] == 1
    assert _snapshot_files(tmp_path) == before


def test_rf100_drops_non_bodies_quarantines_ambiguity_and_selects_one_variant(
    tmp_path: Path,
) -> None:
    rf100 = _write_public_archive(
        tmp_path / "rf100.zip",
        RF100_TAG,
        [
            (
                f"train/images/body.rf.{HASHES[0]}.jpg",
                (50, 40),
                (180, 20, 20),
                [
                    ("IC", 0.5, 0.5, 0.3, 0.3),
                    ("Pads", 0.2, 0.2, 0.1, 0.1),
                    ("Pins", 0.3, 0.2, 0.1, 0.1),
                    ("Test Point", 0.4, 0.2, 0.1, 0.1),
                ],
            ),
            (
                f"train/images/ambiguous.rf.{HASHES[1]}.jpg",
                (44, 40),
                (20, 180, 20),
                [
                    ("IC", 0.5, 0.5, 0.3, 0.3),
                    (next(iter(RF100_AMBIGUOUS_CLASSES)), 0.2, 0.2, 0.1, 0.1),
                ],
            ),
            (
                f"train/images/variant.rf.{HASHES[2]}.jpg",
                (80, 80),
                (20, 20, 180),
                [("IC", 0.5, 0.5, 0.3, 0.3)],
            ),
            (
                f"valid/images/variant.rf.{HASHES[3]}.jpg",
                (20, 18),
                (40, 40, 190),
                [
                    ("IC", 0.3, 0.5, 0.2, 0.3),
                    ("Capacitor", 0.7, 0.5, 0.2, 0.3),
                ],
            ),
        ],
    )

    images, report = _load_public_archive(rf100, RF100_TAG)
    by_scene = {image.source_scene: image for image in images}

    assert set(by_scene) == {"body", "variant"}
    assert len(by_scene["body"].boxes) == 1
    assert report["dropped_non_body_boxes"] == {
        "Pads": 1,
        "Pins": 1,
        "Test Point": 1,
    }
    assert report["quarantined_scenes"] == 1
    assert report["quarantined_scene_ids"] == ["ambiguous"]
    assert by_scene["variant"].width == 20
    assert by_scene["variant"].height == 18
    assert len(by_scene["variant"].boxes) == 2
    assert HASHES[3] in by_scene["variant"].original_name


def test_public_aliases_of_local_holdouts_are_excluded_and_leakage_is_empty(
    tmp_path: Path,
) -> None:
    crop_root, boxes = _write_local_checkpoint(tmp_path, _verified_boards(6))
    rf_images = [
        (
            f"train/images/pcb{board}rec1_jpg.rf.{HASHES[board]}.jpg",
            (30 + board, 26),
            (100 + board * 9, 20, 20),
            [("IC", 0.5, 0.5, 0.4, 0.4)],
        )
        for board in range(1, 7)
    ]
    rf100 = _write_public_archive(tmp_path / "rf100.zip", RF100_TAG, rf_images)
    winnies = _write_public_archive(
        tmp_path / "winnies.zip",
        WINNIES_TAG,
        [
            (
                f"train/images/unrelated.rf.{HASHES[8]}.jpg",
                (33, 29),
                (20, 180, 180),
                [("CHIP", 0.5, 0.5, 0.4, 0.4)],
            )
        ],
    )

    plan = build_plan(
        crop_root,
        boxes,
        rf100,
        winnies,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=17,
        min_target_groups=3,
        audit_only=False,
    )

    holdouts = {
        group
        for split in ("valid", "test")
        for group in plan.report["target_group_split"][split]
    }
    assert len(holdouts) == 2
    assert not holdouts & {image.group_id for image in plan.public}
    assert all(image.split == "train" for image in plan.public)
    assert plan.report["public_overlap_excluded"] == {"test": 1, "valid": 1}
    assert plan.report["leakage_audit"] == {
        "group_intersections": {
            "train_valid": [],
            "train_test": [],
            "valid_test": [],
        },
        "pixel_sha256_intersections": {
            "train_valid": [],
            "train_test": [],
            "valid_test": [],
        },
    }


def test_insufficient_target_groups_fail_pack_but_allow_read_only_audit(
    tmp_path: Path,
) -> None:
    crop_root, boxes = _write_local_checkpoint(tmp_path, _verified_boards(2))
    rf100, winnies = _minimal_public_archives(tmp_path)
    output = tmp_path / "pack"
    common = [
        str(crop_root),
        "--boxes",
        str(boxes),
        "--rf100",
        str(rf100),
        "--winnies",
        str(winnies),
        "--min-target-groups",
        "3",
    ]

    with pytest.raises(SystemExit, match="only 2 verified target board groups"):
        main([*common, "--output", str(output)])
    assert not output.exists()
    assert main([*common, "--audit-only"]) == 0
    assert not output.exists()


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    crop_root, boxes = _write_local_checkpoint(tmp_path, _verified_boards(3))
    rf100, winnies = _minimal_public_archives(tmp_path)
    output = tmp_path / "pack"
    output.mkdir()
    sentinel = output / "reviewed-evidence.txt"
    sentinel.write_text("do not replace", encoding="utf-8")
    # Output ownership is checked before parsing checkpoints or constructing a
    # split plan. A malformed input must not mask the overwrite refusal.
    boxes.write_text("{malformed checkpoint", encoding="utf-8")

    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(
            [
                str(crop_root),
                "--boxes",
                str(boxes),
                "--rf100",
                str(rf100),
                "--winnies",
                str(winnies),
                "--min-target-groups",
                "3",
                "--output",
                str(output),
            ]
        )

    assert sentinel.read_text(encoding="utf-8") == "do not replace"
    assert list(output.iterdir()) == [sentinel]


@pytest.mark.parametrize(
    ("field", "wrong_value", "message"),
    [
        ("dataset", "different_crop_set", "checkpoint dataset mismatch"),
        ("dataset_id", "0000000000000000", "checkpoint dataset_id mismatch"),
    ],
)
def test_checkpoint_identity_must_match_generated_labelling_app(
    tmp_path: Path,
    field: str,
    wrong_value: str,
    message: str,
) -> None:
    crop_root, boxes = _write_local_checkpoint(
        tmp_path, [{"scene": "pcb_dslr_001__rec1", "status": "verified"}]
    )
    payload = json.loads(boxes.read_text(encoding="utf-8"))
    payload[field] = wrong_value
    boxes.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit, match=message):
        _load_local(crop_root, boxes)


def test_checkpoint_rejects_status_not_emitted_by_the_labelling_app(
    tmp_path: Path,
) -> None:
    crop_root, boxes = _write_local_checkpoint(
        tmp_path,
        [{"scene": "pcb_dslr_001__rec1", "status": "unreviewed"}],
    )

    with pytest.raises(SystemExit, match="unsupported checkpoint status"):
        _load_local(crop_root, boxes)


def test_conflicting_verified_labels_for_exact_pixels_fail_closed(
    tmp_path: Path,
) -> None:
    shared_pixels = _image_bytes((40, 30), (90, 80, 70))
    crop_root, boxes = _write_local_checkpoint(
        tmp_path,
        [
            {
                "name": "copy_a.jpg",
                "scene": "pcb1__rec1",
                "blob": shared_pixels,
                "status": "verified",
                "boxes": [
                    {"cls": "component", "x": 2, "y": 2, "w": 10, "h": 8}
                ],
            },
            {
                "name": "copy_b.jpg",
                "scene": "pcb_dslr_001__rec1",
                "blob": shared_pixels,
                "status": "verified",
                "boxes": [
                    {"cls": "component", "x": 4, "y": 2, "w": 10, "h": 8}
                ],
            },
        ],
    )

    with pytest.raises(SystemExit, match="conflicting verified labels"):
        _load_local(crop_root, boxes)


def test_official_ic_boxes_are_a_read_only_completeness_audit(
    tmp_path: Path,
) -> None:
    crop_root, boxes, rf100, winnies, reference_root, tile_manifest = (
        _pcb_dslr_audit_fixture(tmp_path)
    )
    before = _snapshot_files(tmp_path)

    plan = build_plan(
        crop_root,
        boxes,
        rf100,
        winnies,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=17,
        min_target_groups=3,
        audit_only=True,
        pcb_dslr_reference_root=reference_root,
        tile_manifest_path=tile_manifest,
        require_ic_audit_pass=True,
    )

    audit = plan.report["pcb_dslr_ic_completeness"]
    assert audit["source_annotation_scope"] == "IC only"
    assert audit["audited_ic_instances"] == audit["matched"] == 1
    assert audit["missing"] == 0
    assert audit["available_coverage_pass"] is True
    assert _snapshot_files(tmp_path) == before


def test_ic_audit_rejects_a_tile_that_does_not_match_reference_pixels(
    tmp_path: Path,
) -> None:
    crop_root, boxes, rf100, winnies, reference_root, tile_manifest = (
        _pcb_dslr_audit_fixture(tmp_path)
    )
    tile = next((crop_root / "crops").iterdir())
    Image.new("RGB", (40, 30), (200, 10, 10)).save(tile)

    with pytest.raises(SystemExit, match="tile pixels do not match"):
        build_plan(
            crop_root,
            boxes,
            rf100,
            winnies,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=17,
            min_target_groups=3,
            audit_only=True,
            pcb_dslr_reference_root=reference_root,
            tile_manifest_path=tile_manifest,
        )


def test_written_pack_is_a_valid_single_class_yolo_tree_with_no_split_leakage(
    tmp_path: Path,
) -> None:
    """Nửa GHI của packer trước đây không test nào chạy qua.

    Hai test có ``--output`` đều khẳng định packer TỪ CHỐI ghi, nên toàn bộ
    ``write_pack`` — layout images/labels, ``data.yaml``, chuẩn hoá toạ độ —
    chưa từng được thực thi. Một dataset ghi sai vẫn "pack thành công" và chỉ lộ
    ra sau nhiều giờ train.
    """

    # 9 board là số nhỏ nhất mà phép băm seed=17 phủ đủ cả ba bucket, nên đây là
    # lần đầu tiên đường ghi chạy trọn vẹn thay vì dừng ở cổng readiness.
    crop_root, boxes = _write_local_checkpoint(tmp_path, _verified_boards(9))
    rf100, winnies = _minimal_public_archives(tmp_path)
    output = tmp_path / "pack"

    assert (
        main(
            [
                str(crop_root),
                "--boxes",
                str(boxes),
                "--rf100",
                str(rf100),
                "--winnies",
                str(winnies),
                "--min-target-groups",
                "3",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    data_yaml = (output / "data.yaml").read_text(encoding="utf-8")
    assert "nc: 1" in data_yaml
    assert "names: ['component']" in data_yaml or 'names: ["component"]' in data_yaml

    seen_stems: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        images = sorted((output / split / "images").iterdir())
        labels = sorted((output / split / "labels").iterdir())
        assert images, f"{split} rỗng — cổng readiness lẽ ra đã chặn"
        assert [path.stem for path in images] == [path.stem for path in labels]
        for label in labels:
            for line in label.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                assert len(fields) == 5, f"{label}: {line!r} không phải 5 trường YOLO"
                assert fields[0] == "0", "một lớp duy nhất thì class index phải là 0"
                assert all(
                    0.0 <= float(value) <= 1.0 for value in fields[1:]
                ), f"{label}: toạ độ chưa chuẩn hoá: {line!r}"
        for image in images:
            # Một ảnh nằm ở hai split là rò rỉ, và rò rỉ chỉ hiện ra dưới dạng
            # điểm số đẹp giả — không có gì báo lỗi lúc train.
            assert image.stem not in seen_stems, (
                f"{image.stem} có mặt ở cả {seen_stems[image.stem]} và {split}"
            )
            seen_stems[image.stem] = split

    manifest = json.loads((output / "pack_manifest.json").read_text(encoding="utf-8"))
    assert manifest["readiness"]["ready_to_pack"] is True
    assert manifest["class_names"] == ["component"]


def test_an_empty_split_names_the_unreviewed_boards_that_would_fill_it(
    tmp_path: Path,
) -> None:
    """"Thiếu bucket valid" một mình không nói được phải làm gì tiếp.

    Bucket là hàm băm của board id, nên người duyệt không thể tự đoán duyệt
    board nào thì lấp được. Packer biết câu trả lời và phải nói ra.
    """

    # Chỉ board 001 được duyệt (rơi vào train ở seed 17), nên valid và test đều
    # trống. Board 002 (valid) và 009 (test) có tile trong manifest nhưng vắng
    # trong export — `status=None` — nên đó chính là hai board cần gọi tên.
    records = _verified_boards(1)
    records.extend(
        {
            "name": f"pcb_dslr_{board:03d}__rec1__tile.jpg",
            "scene": f"pcb_dslr_{board:03d}__rec1",
            "color": (90 + board, 60, 120),
            "status": None,
        }
        for board in (2, 9)
    )
    crop_root, boxes = _write_local_checkpoint(tmp_path, records)
    rf100, winnies = _minimal_public_archives(tmp_path)

    plan = build_plan(
        crop_root,
        boxes,
        rf100,
        winnies,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=17,
        min_target_groups=3,
        audit_only=True,
    )

    readiness = plan.report["readiness"]
    assert readiness["missing_target_splits"] == ["valid", "test"]
    assert [
        board["group_id"] for board in readiness["boards_that_would_fill"]["valid"]
    ] == ["pcb_dslr:002"]
    assert [
        board["group_id"] for board in readiness["boards_that_would_fill"]["test"]
    ] == ["pcb_dslr:009"]

    # Board đã duyệt không bao giờ được đề nghị duyệt lại.
    for boards in readiness["boards_that_would_fill"].values():
        assert all(board["verified_tiles"] == 0 for board in boards)
