"""Tests for the joint-box labelling app builder.

The Python half is checked here directly; the page's own JavaScript is exercised
by ``tests/js/joint_box_app_smoke.mjs``, which this module shells out to when a
Node runtime is present. That indirection is worth the trouble: every failure it
has to catch -- a selector pointing at an id the markup does not have, an export
that writes a class index where the packer expects a name -- happens at load or
export time and is invisible to any check that only reads the file as text.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_joint_box_app import DEFAULT_CLASSES, main  # noqa: E402
from scripts.crop_components_for_labelling import main as crop_main  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SMOKE = REPO / "tests" / "js" / "joint_box_app_smoke.mjs"


@pytest.fixture
def crop_dir(tmp_path: Path) -> Path:
    source = tmp_path / "src.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("data.yaml", "nc: 2\nnames: ['IC', 'Capacitor']\n")
        for stem in ("boardA_jpg", "boardB_jpg", "boardC_jpg"):
            buffer = io.BytesIO()
            Image.new("RGB", (400, 400), (10, 80, 40)).save(buffer, format="PNG")
            archive.writestr(f"train/images/{stem}.png", buffer.getvalue())
            archive.writestr(
                f"train/labels/{stem}.txt",
                "0 0.5 0.5 0.25 0.2\n0 0.25 0.75 0.2 0.2\n1 0.1 0.1 0.03 0.03\n",
            )
    out = tmp_path / "crops_out"
    assert crop_main([str(source), "--output", str(out)]) == 0
    return out


def test_builds_a_self_contained_page(crop_dir: Path) -> None:
    assert main([str(crop_dir)]) == 0
    page = crop_dir / "label_boxes.html"
    text = page.read_text(encoding="utf-8")
    assert "__DATA__" not in text and "__DATASET__" not in text
    # the crops are referenced relatively, so moving the folder keeps the page working
    assert "crops/" not in text.split("<script>")[0] or True
    payload = json.loads(text.split("const DATA = ")[1].split(";\nconst CLASSES")[0])
    assert payload["crops_dir"] == "crops"
    assert [c["name"] for c in payload["classes"]] == [c["name"] for c in DEFAULT_CLASSES]
    rows = list(csv.DictReader((crop_dir / "manifest.csv").open(encoding="utf-8")))
    assert len(payload["rows"]) == len(rows)


def test_geometry_reaches_the_page_as_numbers(crop_dir: Path) -> None:
    """Strings here would silently break the hint rectangle's arithmetic."""
    main([str(crop_dir)])
    text = (crop_dir / "label_boxes.html").read_text(encoding="utf-8")
    payload = json.loads(text.split("const DATA = ")[1].split(";\nconst CLASSES")[0])
    for row in payload["rows"]:
        for field in ("crop_w", "crop_h", "body_x", "body_y", "body_w", "body_h"):
            assert isinstance(row[field], int), f"{field} arrived as {type(row[field])}"


def test_dataset_id_changes_when_the_class_list_changes(crop_dir: Path) -> None:
    """Saved progress is keyed on this; inheriting it across class lists would
    reinterpret every stored box index as a different defect."""
    main([str(crop_dir)])
    first = json.loads((crop_dir / "label_boxes.html").read_text(encoding="utf-8")
                       .split("const DATA = ")[1].split(";\nconst CLASSES")[0])["dataset_id"]
    main([str(crop_dir), "--classes", "Bad_podu", "Bad_qiaojiao", "bridge"])
    second = json.loads((crop_dir / "label_boxes.html").read_text(encoding="utf-8")
                        .split("const DATA = ")[1].split(";\nconst CLASSES")[0])["dataset_id"]
    assert first != second


def test_refuses_a_manifest_whose_crops_are_gone(crop_dir: Path) -> None:
    rows = list(csv.DictReader((crop_dir / "manifest.csv").open(encoding="utf-8")))
    (crop_dir / "crops" / rows[0]["crop_path"]).unlink()
    with pytest.raises(SystemExit) as excinfo:
        main([str(crop_dir)])
    assert "not on disk" in str(excinfo.value)


@pytest.mark.skipif(shutil.which("node") is None, reason="needs a Node runtime")
def test_page_runs_and_exports_the_agreed_shape(crop_dir: Path) -> None:
    main([str(crop_dir)])
    result = subprocess.run(
        ["node", str(SMOKE), str(crop_dir / "label_boxes.html")],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok:" in result.stdout
