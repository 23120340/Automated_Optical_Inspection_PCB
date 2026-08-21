"""The training notebook must cut crops the way inference cuts them.

The notebook runs on Kaggle, where this repository does not exist, so it carries
its own copy of ``component_crop_window``. A copy is a liability: the day the
library's margin changes and the notebook's does not, every crop the model was
trained on differs slightly from every crop it is asked to read, and nothing
reports it. The model just quietly gets worse.

So the copy is checked against the original here, on the values that matter.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from aoi_pipeline import BoundingBox, LeadDetectionConfig
from aoi_pipeline.lead_detection import component_crop_window

NOTEBOOK = (
    Path(__file__).resolve().parents[2]
    / "training" / "kaggle" / "pcb_lead_detector_kaggle.py"
)


@pytest.fixture(scope="module")
def notebook_source() -> str:
    assert NOTEBOOK.is_file(), f"thiếu notebook {NOTEBOOK}"
    return NOTEBOOK.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def notebook_config(notebook_source: str) -> dict:
    """The notebook's own CONFIG dict, read from its source."""

    match = re.search(r"^CONFIG = \{.*?^\}", notebook_source, re.S | re.M)
    assert match, "không tìm thấy CONFIG trong notebook"
    namespace: dict = {}
    exec(match.group(0), namespace)  # noqa: S102 - our own file, read from disk
    return namespace["CONFIG"]


@pytest.fixture(scope="module")
def notebook_crop_window(notebook_source: str, notebook_config: dict):
    """The notebook's copy of the crop-window function, pulled out and made
    callable so it can be compared against the real one."""

    match = re.search(
        r"^def component_crop_window\(.*?(?=^def |\Z)", notebook_source, re.S | re.M
    )
    assert match, "không tìm thấy component_crop_window trong notebook"
    namespace = {"CONFIG": notebook_config}
    exec(match.group(0), namespace)  # noqa: S102
    return namespace["component_crop_window"]


def test_the_notebook_margins_match_the_library_defaults(notebook_config) -> None:
    """A margin that differs between training and inference is a distribution
    shift nobody is told about."""

    defaults = LeadDetectionConfig()
    assert notebook_config["crop_margin_ratio"] == defaults.crop_margin_ratio
    assert notebook_config["crop_margin_min_px"] == defaults.crop_margin_min_px
    assert notebook_config["min_crop_px"] == defaults.min_crop_px


@pytest.mark.parametrize(
    "box",
    [
        (200, 150, 270, 190),   # 0805 chip, the common case
        (100, 100, 110, 108),   # tiny part where the floor decides
        (2, 2, 40, 30),         # against the frame corner
        (600, 340, 690, 396),   # against the far edge
        (333, 240, 464, 378),   # a large electrolytic
        (539, 643, 603, 690),   # SOT-23
    ],
)
def test_the_notebook_cuts_the_same_window_as_inference(
    notebook_crop_window, box
) -> None:
    width, height = 700, 400
    from_library = component_crop_window(
        BoundingBox(*box), width, height, LeadDetectionConfig()
    )
    from_notebook = notebook_crop_window(box, width, height)
    assert from_notebook == from_library, (
        f"box {box}: notebook cắt {from_notebook}, thư viện cắt {from_library}"
    )


def test_the_notebook_refuses_unreviewed_pseudo_labels(notebook_source: str) -> None:
    """Training on uncorrected bootstrap boxes teaches the model to reproduce
    the geometry that produced them -- including the geometry errors measured on
    a real board. The notebook has to stop, not warn."""

    assert "PSEUDO_LABELS_NEED_REVIEW" in notebook_source
    assert "allow_unreviewed_labels" in notebook_source
    assert "raise SystemExit" in notebook_source


def test_the_notebook_splits_by_board_not_by_crop(notebook_source: str) -> None:
    """Crops from one board share lighting, solder batch and part types. Random
    per-crop splitting puts siblings on both sides and inflates the score -- the
    exact mistake that made step 6.2 read 97.65% before it was corrected to
    89.9%."""

    assert "by_board" in notebook_source
    assert 'splits = {' in notebook_source
    assert "random.Random(CONFIG[\"seed\"]).shuffle(board_names)" in notebook_source
