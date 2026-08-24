from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_diverse_reference_bootstrap as bootstrap  # noqa: E402

from aoi_pipeline.models import BoundingBox, Detection  # noqa: E402


class _FakeRuntime:
    @property
    def identifier(self) -> str:
        return "fake.onnx:" + "a" * 64

    def detect(self, image: np.ndarray, board_bbox: BoundingBox) -> list[Detection]:
        assert board_bbox.width > 0
        return [
            Detection("ic", 0.8, BoundingBox(19, 14, 31, 26), source="ultralytics"),
            Detection("resistor", 0.9, BoundingBox(40, 30, 52, 36), source="ultralytics"),
            Detection("pads", 0.95, BoundingBox(2, 2, 8, 8), source="ultralytics"),
        ]


def _dataset(root: Path, count: int = 2) -> list[Path]:
    entries = []
    images = []
    for number in range(1, count + 1):
        board_id = f"pcb_dslr_{number:03d}"
        board_dir = root / "boards" / board_id
        board_dir.mkdir(parents=True)
        image = np.full((60, 80, 3), (20 + number, 80, 35), np.uint8)
        cv2.rectangle(image, (5, 5), (74, 54), (50, 150, 80), -1)
        mask = np.zeros((60, 80), np.uint8)
        cv2.rectangle(mask, (5, 5), (74, 54), 255, -1)
        image_path = board_dir / "rec1.jpg"
        mask_path = board_dir / "rec1-mask.png"
        annotation_path = board_dir / "rec1-annot.txt"
        assert cv2.imwrite(str(image_path), image)
        assert cv2.imwrite(str(mask_path), mask)
        annotation_path.write_text("25 20 12 12 -10.5 IC MARKING\n", encoding="utf-8")
        image_payload = image_path.read_bytes()
        entries.append(
            {
                "board_id": board_id,
                "upstream_board_id": f"pcb{number}",
                "image_path": f"boards/{board_id}/rec1.jpg",
                "mask_path": f"boards/{board_id}/rec1-mask.png",
                "annotation_path": f"boards/{board_id}/rec1-annot.txt",
                "image_sha256": sha256(image_payload).hexdigest(),
                "mask_sha256": sha256(mask_path.read_bytes()).hexdigest(),
                "annotation_sha256": sha256(annotation_path.read_bytes()).hexdigest(),
                "width": 80,
                "height": 60,
                "source_archive": "test.zip",
            }
        )
        images.append(image_path)
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps({"source_dataset": "test", "files": entries}),
        encoding="utf-8",
    )
    return images


def test_parse_upstream_ic_preserves_float_angle_and_free_text() -> None:
    parsed = bootstrap.parse_upstream_ic_annotations(
        "20.5 30.25 12.0 8.0 -89.1696853638 WORDCLASS MAINBOARD\n",
        image_width=100,
        image_height=80,
    )
    assert len(parsed) == 1
    assert parsed[0].center_x == pytest.approx(20.5)
    assert parsed[0].angle_deg == pytest.approx(-89.1696853638)
    assert parsed[0].text == "WORDCLASS MAINBOARD"
    assert parsed[0].bbox.width > 0


def test_merge_prefers_upstream_ic_and_excludes_pad_classes() -> None:
    upstream = bootstrap.parse_upstream_ic_annotations(
        "25 20 12 12 -10.5 IC MARKING\n",
        image_width=80,
        image_height=60,
    )
    proposals = bootstrap.merge_proposals(_FakeRuntime().detect(
        np.zeros((60, 80, 3), np.uint8), BoundingBox(0, 0, 80, 60)
    ), upstream)
    assert [item.label for item in proposals] == ["ic", "resistor"]
    assert proposals[0].proposal_source == "pcb_dslr_upstream_ic_annotation"
    assert proposals[0].rotation_deg == pytest.approx(-10.5)


def test_low_confidence_empty_result_fallback_remains_explicit_in_pnp() -> None:
    detection = Detection(
        "ic",
        0.12,
        BoundingBox(10, 10, 20, 20),
        source="ultralytics",
        metadata={"empty_result_fallback": True, "applied_confidence": 0.10},
    )
    proposals = bootstrap.merge_proposals([detection], [])
    assert proposals[0].proposal_source.endswith("low_confidence_fallback")
    rows = bootstrap._proposal_rows("pcb_dslr_021", proposals)
    assert "low_confidence_empty_result_fallback" in rows[0]["review_reasons"]


def test_build_creates_separate_review_only_reference_and_pixel_pnp(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    images = _dataset(root)
    output = root / "bootstrap"
    result = bootstrap.build_bootstrap(
        bootstrap.BootstrapConfig(
            dataset_root=root,
            output_dir=output,
            model_path=tmp_path / "unused.onnx",
            expected_count=2,
            preview_max_side=300,
        ),
        runtime=_FakeRuntime(),
    )
    assert result == output.resolve()
    manifest = json.loads(
        (output / "bootstrap_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["board_count"] == 2
    assert manifest["production_eligible"] is False
    assert "one source board" in manifest["distinct_layout_policy"]
    assert len({item["board_id"] for item in manifest["boards"]}) == 2
    assert (output / "contact_sheet.jpg").is_file()

    first = output / "references" / "pcb_dslr_001"
    reference = json.loads((first / "golden_candidate.json").read_text(encoding="utf-8"))
    assert reference["status"] == bootstrap.REFERENCE_STATUS
    assert reference["production_eligible"] is False
    assert reference["metrology"]["verified"] is False
    assert not Path(reference["image"]["path"]).is_absolute()
    assert reference["image"]["sha256"] == sha256(images[0].read_bytes()).hexdigest()

    with (first / "pnp_pixels_NEEDS_REVIEW.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["expected_class"] for row in rows] == ["ic", "resistor"]
    assert all(row["coordinate_space"] == "source_image_pixels" for row in rows)
    assert all(row["status"] == "NEEDS_REVIEW" for row in rows)
    assert all("Mid X" not in row and "Mid Y" not in row for row in rows)
    assert rows[0]["designator"].startswith("U_AUTO_")
    assert rows[0]["rotation_source"].startswith("upstream_opencv")
    assert rows[1]["rotation_deg_observed"] == ""


def test_source_image_tamper_and_existing_output_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    images = _dataset(root, count=1)
    images[0].write_bytes(images[0].read_bytes() + b"tamper")
    with pytest.raises(bootstrap.BootstrapError, match="failed SHA-256"):
        bootstrap.load_source_boards(root, expected_count=1)

    mask_root = tmp_path / "mask_tamper"
    _dataset(mask_root, count=1)
    mask_path = mask_root / "boards" / "pcb_dslr_001" / "rec1-mask.png"
    mask_path.write_bytes(mask_path.read_bytes() + b"tamper")
    with pytest.raises(bootstrap.BootstrapError, match="Mask failed SHA-256"):
        bootstrap.load_source_boards(mask_root, expected_count=1)

    root = tmp_path / "clean"
    _dataset(root, count=1)
    output = root / "bootstrap"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(bootstrap.BootstrapError, match="refusing to overwrite"):
        bootstrap.build_bootstrap(
            bootstrap.BootstrapConfig(
                dataset_root=root,
                output_dir=output,
                model_path=tmp_path / "unused.onnx",
                expected_count=1,
            ),
            runtime=_FakeRuntime(),
        )
    assert marker.read_text(encoding="utf-8") == "keep"
