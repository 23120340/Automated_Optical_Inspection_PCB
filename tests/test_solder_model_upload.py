"""Uploading the step-6.2 model must not break the app — or the other steps.

Two real failures are pinned here.

1. ``KeyError: 'solder_grading'``. The sidebar writes the uploaded artifact
   paths into ``st.session_state.config["solder_grading"]``, but
   ``_default_config`` never created that section, so the first upload killed
   the whole app before anything rendered.

2. A model uploaded before its manifest. ``create_solder_classifier`` refuses
   half a contract, and it refuses from inside ``AOIPipeline.__init__`` — so
   the bridge ends up with *no engine at all* and step 4 quietly drops back to
   the CV demo. The half-pair has to be held back until it is complete.

The upload path is exercised through a real Streamlit script run rather than by
calling the setters bare: ``st.session_state`` outside a run context is not the
thing that broke.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import app.streamlit_app as ui
from app.pipeline_bridge import PipelineBridge

SOLDER_MODEL = (
    ui.PROJECT_ROOT / "models" / "active" / "solder" / "classifier" / "best.onnx"
)
SOLDER_MANIFEST = (
    ui.PROJECT_ROOT
    / "models"
    / "active"
    / "solder"
    / "classifier"
    / "model_manifest.json"
)

needs_artifacts = pytest.mark.skipif(
    not (SOLDER_MODEL.is_file() and SOLDER_MANIFEST.is_file()),
    reason="step-6.2 artifacts are not checked into this working copy",
)

HARNESS = '''
import sys
from pathlib import Path

sys.path.insert(0, {root!r})

import streamlit as st

import app.streamlit_app as ui


class _Upload:
    def __init__(self, name: str, path: str) -> None:
        self.name = name
        self._data = Path(path).read_bytes()

    def getvalue(self) -> bytes:
        return self._data


ui._init_state()
if {with_model!r}:
    ui._set_solder_model(_Upload("best.onnx", {model!r}))
if {with_manifest!r}:
    ui._set_solder_manifest(_Upload("model_manifest.json", {manifest!r}))
st.write("done")
'''


def _run_upload(tmp_path: Path, *, with_model: bool, with_manifest: bool):
    from streamlit.testing.v1 import AppTest

    script = tmp_path / "upload_harness.py"
    script.write_text(
        HARNESS.format(
            root=str(ui.PROJECT_ROOT),
            model=str(SOLDER_MODEL),
            manifest=str(SOLDER_MANIFEST),
            with_model=with_model,
            with_manifest=with_manifest,
        ),
        encoding="utf-8",
    )
    harness = AppTest.from_file(str(script), default_timeout=120)
    harness.run()
    assert not harness.exception, [str(e.value) for e in harness.exception]
    return harness


def test_default_config_declares_every_section_the_sidebar_writes_into() -> None:
    """The KeyError, stated as a rule rather than as one missing key: any
    section the UI assigns into must already exist."""

    config = ui._default_config()
    for section in ("solder", "solder_grading"):
        assert section in config, f"_default_config thiếu section '{section}'"
        assert isinstance(config[section], dict)


@needs_artifacts
def test_uploading_the_model_records_its_path_without_crashing(tmp_path: Path) -> None:
    """The exact traceback from the running app: KeyError: 'solder_grading'
    raised by ``_set_solder_model`` on the very first upload."""

    harness = _run_upload(tmp_path, with_model=True, with_manifest=False)
    assert harness.session_state.solder_model_name == "best.onnx"
    assert harness.session_state.config["solder_grading"]["model_path"]


@needs_artifacts
def test_uploading_both_halves_records_both(tmp_path: Path) -> None:
    harness = _run_upload(tmp_path, with_model=True, with_manifest=True)
    grading = harness.session_state.config["solder_grading"]
    assert grading["model_path"] and grading["manifest_path"]


@needs_artifacts
def test_a_half_pair_is_held_back_so_the_rest_of_the_pipeline_survives() -> None:
    """Without the guard the engine fails to construct and every step falls
    back to the CV demo — a step-6.2 upload silently downgrading step 4."""

    config = ui._default_config()
    config["solder_grading"]["model_path"] = str(SOLDER_MODEL)

    assert ui._engine_config(config)["solder_grading"]["model_path"] is None

    bridge = PipelineBridge(config=ui._engine_config(config))
    assert bridge.engine is not None, bridge.engine_error
    assert bridge.engine.solder_inspector.has_model is False


@needs_artifacts
def test_a_complete_pair_reaches_the_inspector() -> None:
    config = ui._default_config()
    config["solder_grading"]["model_path"] = str(SOLDER_MODEL)
    config["solder_grading"]["manifest_path"] = str(SOLDER_MANIFEST)

    bridge = PipelineBridge(config=ui._engine_config(config))
    assert bridge.engine is not None, bridge.engine_error
    assert bridge.engine.solder_inspector.has_model is True


def test_step_5_5_settings_survive_the_trip_into_the_typed_config() -> None:
    """``_section`` falls back to the whole config when a section is absent, so
    a missing ``solder`` key fails silently rather than loudly. Assert the
    values actually arrive."""

    from aoi_pipeline.config import PipelineConfig

    config = ui._default_config()
    config["solder"].update({"target_size": 96, "split_pins": True})
    typed = PipelineConfig.from_mapping(config)
    assert typed.solder.target_size == (96, 96)
    assert typed.solder.split_pins is True
    assert typed.solder.terminal_outer_ratio == config["solder"]["terminal_outer_ratio"]


def test_the_manifest_schema_literals_still_match_the_core() -> None:
    """The UI validates manifests against string literals so it does not import
    the core at module load. That is only safe while the literals track the
    constants they mirror."""

    from aoi_pipeline.classification.family import MANIFEST_SCHEMA as CLASSIFIER_SCHEMA
    from aoi_pipeline.grading.classifier import MANIFEST_SCHEMA as SOLDER_SCHEMA

    assert ui.CLASSIFIER_MANIFEST_SCHEMA == CLASSIFIER_SCHEMA
    assert ui.SOLDER_MANIFEST_SCHEMA == SOLDER_SCHEMA


RENDER_HARNESS = '''
import sys
from pathlib import Path

sys.path.insert(0, {root!r})

import cv2
import numpy as np
import streamlit as st

import app.streamlit_app as ui
from app.pipeline_bridge import PipelineBridge

image = np.full((240, 340, 3), 24, dtype=np.uint8)
cv2.rectangle(image, (35, 30), (305, 210), (38, 118, 50), -1)
cv2.rectangle(image, (35, 30), (305, 210), (180, 220, 180), 3)
rng = np.random.default_rng(1234)
for index in range(28):
    x = int(rng.integers(48, 280))
    y = int(rng.integers(43, 190))
    width = int(rng.integers(7, 24))
    height = int(rng.integers(5, 17))
    color = (15, 15, 15) if index % 2 == 0 else (180, 180, 195)
    cv2.rectangle(image, (x, y), (min(x + width, 297), min(y + height, 202)), color, -1)
    cv2.circle(image, (x, y), 2, (220, 220, 220), -1)
cv2.putText(image, "AOI-103", (80, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 2)
image = cv2.resize(image, (1000, 700), interpolation=cv2.INTER_CUBIC)

ui._init_state()
if {with_model!r}:
    st.session_state.config["solder_grading"]["model_path"] = {model!r}
    st.session_state.config["solder_grading"]["manifest_path"] = {manifest!r}
else:
    # Từ 2026-08-22 `_init_state` nạp sẵn model trong models/active/, nên
    # "không có model" phải được dựng ra tường minh chứ không còn là mặc định.
    st.session_state.config["solder_grading"]["model_path"] = None
    st.session_state.config["solder_grading"]["manifest_path"] = None
    st.session_state.solder_model_path = None
    st.session_state.solder_manifest_path = None

bridge = PipelineBridge(config=ui._engine_config())
preprocessed = bridge.preprocess(image)
board = bridge.detect_board(preprocessed.image)
detections = bridge.detect_components(preprocessed.image, board)

st.session_state.input_image = image
st.session_state.preprocess_result = preprocessed
st.session_state.board_result = board
st.session_state.detection_result = detections
st.session_state.crops = bridge.make_crops(preprocessed.image, detections.detections)
st.session_state.solder_result = bridge.make_solder_crops(
    preprocessed.image, detections.detections
)

ui._render_step_seven()
st.write("VERDICTS", len(st.session_state.solder_result.verdicts))
st.write("GRADED_BY_MODEL", st.session_state.solder_result.graded_by_model)
'''


def _render_step_seven(tmp_path: Path, *, with_model: bool):
    from streamlit.testing.v1 import AppTest

    script = tmp_path / "render_step_seven.py"
    script.write_text(
        RENDER_HARNESS.format(
            root=str(ui.PROJECT_ROOT),
            model=str(SOLDER_MODEL),
            manifest=str(SOLDER_MANIFEST),
            with_model=with_model,
        ),
        encoding="utf-8",
    )
    harness = AppTest.from_file(str(script), default_timeout=300)
    harness.run()
    return harness


def _written(harness, prefix: str) -> str:
    return next(m.value for m in harness.markdown if m.value.startswith(prefix))


def test_step_seven_renders_real_verdicts_without_a_model(tmp_path: Path) -> None:
    """The rule layer alone must draw a complete panel.

    Two of the three functions this panel calls were dropped by a merge and the
    module still imported cleanly, because a NameError in a renderer only fires
    when that renderer runs. Nothing short of rendering it catches that.
    """

    harness = _render_step_seven(tmp_path, with_model=False)
    assert not harness.exception, [str(e.value) for e in harness.exception]
    # Từ 2026-08-22 app tự nạp model trong models/active/, nên "không có model"
    # phải được dựng ra một cách tường minh. Kịch bản vẫn đáng kiểm: lớp luật
    # vật lý là thứ đáng tin nhất ở 6.2 và phải đứng một mình được.
    assert _written(harness, "VERDICTS") != "VERDICTS `0`"
    assert "GRADED_BY_MODEL `False`" == _written(harness, "GRADED_BY_MODEL")
    labels = [metric.label for metric in harness.metric]
    assert "Cần kiểm tra" in labels


@needs_artifacts
def test_step_seven_renders_once_when_a_model_is_loaded(tmp_path: Path) -> None:
    """Step 6.2 was promoted out of step 4 into a section of its own, but the
    5.5 view kept its own copy of the grading tab. Both drew the same checkbox
    in one pass and Streamlit rejected the duplicate key, which took the entire
    step-7 view down whenever there were verdicts to show."""

    harness = _render_step_seven(tmp_path, with_model=True)
    assert not harness.exception, [str(e.value) for e in harness.exception]
    assert "GRADED_BY_MODEL `True`" == _written(harness, "GRADED_BY_MODEL")
    # One grading panel, not two: the four-metric verdict row appears once.
    assert [metric.label for metric in harness.metric].count("Đạt") == 1


def test_no_model_artifact_is_seeded_from_disk() -> None:
    """One way in for every model: the sidebar uploader.

    ``_init_state`` used to plant ``models/detector/best.onnx`` into
    ``component_model_path``. That path is the only one no uploader wrote, so it
    carried no digest, never ran the ``.pt`` trust check, could not be restored
    after "Gỡ model" — and named a file the repo stopped having once the
    detectors moved under ``kaggle/ver*``.
    """

    import inspect
    import re

    source = inspect.getsource(ui._init_state)
    seeded = [
        line.strip()
        for line in source.splitlines()
        if re.match(r'\s*"\w*_(model|manifest)_(path|name)":', line)
        and not line.strip().endswith("None,")
    ]
    assert seeded == [], f"artifact được gán sẵn thay vì nạp từ sidebar: {seeded}"
