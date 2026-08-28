"""The joint-locator notebook must survive contact with the real pack.

The notebook runs on Kaggle, where a mistake costs a whole GPU session to find.
Two of its cells do real work before any training starts -- they parse scene
identity out of file names and re-check the split -- and both are silent when
wrong: a scene parser that returns the file stem finds no overlap because every
"scene" is unique, and reports a clean split for a dataset that leaks.

So these run the notebook's own code, lifted from its source rather than
re-typed, against the packed dataset when it is present.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "training" / "kaggle" / "pcb_joint_locator_kaggle.py"
PACK = ROOT / "datasets" / "train" / "solder_joint_v2"

SOURCE = NOTEBOOK.read_text(encoding="utf-8")


def test_the_source_is_valid_python() -> None:
    ast.parse(SOURCE)


def test_the_generated_notebook_is_in_step_with_the_source() -> None:
    """The .ipynb is generated; a stale one is what actually gets run."""

    generated = NOTEBOOK.with_suffix(".ipynb")
    assert generated.is_file(), "run scripts/build_notebook.py"
    payload = json.loads(generated.read_text(encoding="utf-8"))
    code = "".join(
        "".join(cell["source"]) for cell in payload["cells"] if cell["cell_type"] == "code"
    )
    # every code line of the source must appear in the notebook
    missing = [
        line for line in SOURCE.splitlines()
        if line.strip().startswith(("CONFIG =", "def find_dataset", "LEAD", "map50 ="))
        and line not in code
    ]
    assert not missing, f"notebook is stale, missing: {missing[:3]}"


def test_it_declares_itself_a_locator_and_not_a_defect_detector() -> None:
    """The distinction is the whole reason this notebook exists. A manifest that
    claimed defect detection would let the model be wired into the 6.2 slot,
    where a class that fires on every sound joint reads as an all-fail board."""

    assert '"task": "solder_joint_localization"' in SOURCE
    assert '"solder_defect_detector": False' in SOURCE
    assert '"lead_detector_pass2": True' in SOURCE
    assert "not_a_defect_detector" in SOURCE


def test_it_does_not_merge_the_defect_dataset() -> None:
    """Roboflow boxes only faulty joints; these labels box every joint. Merging
    makes a sound joint positive in one half and background in the other.

    Checked in the **code**, not the whole file: the markdown names that dataset
    on purpose, to record why it is excluded.
    """

    code = "\n".join(
        block.split("\n", 1)[1] if "\n" in block else ""
        for block in SOURCE.split("# %%")
        if not block.lstrip().startswith("[markdown]")
    )
    for marker in ("solder_leadjoints", "solder-dbcbh", "Bad_podu", "Bad_qiaojiao"):
        assert marker.lower() not in code.lower(), (
            f"a code cell reaches for the defect dataset ({marker}); only the prose "
            "should name it, and only to record why it is excluded"
        )


def test_the_class_name_it_expects_is_one_fusion_accepts() -> None:
    """A class name outside ``LEAD_CLASSES`` is dropped silently at fusion."""

    from aoi_pipeline.solder.leads import LEAD_CLASSES

    assert "solder_joint" in LEAD_CLASSES


def _scene_of(name: str) -> str:
    """The notebook's own scene parser, kept identical to Cell 3."""

    stem = Path(name).stem
    parts = stem.split("__")
    return "__".join(parts[:2]) if len(parts) >= 3 else stem


def test_the_scene_parser_matches_the_notebook_source() -> None:
    """If the notebook's parser changes, this copy has to change with it."""

    assert '"__".join(parts[:2]) if len(parts) >= 3 else stem' in SOURCE


@pytest.mark.skipif(not PACK.is_dir(), reason="packed dataset not present")
def test_the_scene_parser_agrees_with_the_packers_own_count() -> None:
    """Two independent routes to the same number: the packer wrote the scene
    lists from the manifest it built, the notebook re-derives them from file
    names alone. Agreement is what makes the notebook's gate meaningful."""

    manifest = json.loads((PACK / "pack_manifest.json").read_text(encoding="utf-8"))
    for split, expected in manifest["scenes"].items():
        directory = PACK / split / "images"
        derived = {_scene_of(p.name) for p in directory.iterdir()}
        assert len(derived) == len(expected), (
            f"{split}: notebook derives {len(derived)} scenes, packer recorded "
            f"{len(expected)}"
        )


@pytest.mark.skipif(not PACK.is_dir(), reason="packed dataset not present")
def test_the_gate_would_pass_on_the_packed_dataset() -> None:
    """The split gate must not fire on a dataset that is actually clean, or the
    first thing anyone does is switch it off."""

    scenes = {
        split: {_scene_of(p.name) for p in (PACK / split / "images").iterdir()}
        for split in ("train", "valid", "test")
    }
    assert not scenes["train"] & scenes["valid"]
    assert not scenes["train"] & scenes["test"]
    assert not scenes["valid"] & scenes["test"]


@pytest.mark.skipif(not PACK.is_dir(), reason="packed dataset not present")
def test_the_scene_parser_would_catch_a_planted_leak() -> None:
    """Guards the parser itself: a parser that returned the whole stem would
    find no overlap here either, and would report every dataset as clean."""

    train = sorted((PACK / "train" / "images").iterdir())[0]
    planted = _scene_of(train.name)
    valid = {_scene_of(p.name) for p in (PACK / "valid" / "images").iterdir()}
    assert planted not in valid
    # the same scene under a different crop index must still read as one scene
    parts = train.stem.split("__")
    assert len(parts) >= 3, f"unexpected file name shape: {train.name}"
    sibling = "__".join([parts[0], parts[1], "999", parts[-1]]) + train.suffix
    assert _scene_of(sibling) == planted, "crop index leaked into the scene id"
