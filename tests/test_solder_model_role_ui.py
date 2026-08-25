"""UI contract for keeping solder segmentation and ROI classification separate."""

from __future__ import annotations

import inspect
from pathlib import Path

import app.streamlit_app as ui


DETECTOR_MODEL = (
    ui.PROJECT_ROOT / "models" / "active" / "solder" / "segmenter" / "best.onnx"
)
DETECTOR_MANIFEST = DETECTOR_MODEL.with_name("model_manifest.json")
CLASSIFIER_MODEL = (
    ui.PROJECT_ROOT / "models" / "active" / "solder" / "classifier" / "best.onnx"
)
CLASSIFIER_MANIFEST = CLASSIFIER_MODEL.with_name("model_manifest.json")


def test_sidebar_names_both_solder_roles_and_both_contracts_explicitly() -> None:
    source = inspect.getsource(ui._render_sidebar)

    assert "Segmenter lỗi mối hàn · YOLO Segment" in source
    assert "Manifest segmenter (model_manifest.json)" in source
    assert "Classifier ROI mối hàn · raw logits" in source
    assert "Manifest classifier ROI (model_manifest.json)" in source
    assert '_render_model_picker("solder_segmenter")' in source
    assert '_render_model_picker("solder")' in source
    assert "Contract 6.2" not in source


def test_engine_holds_each_incomplete_solder_pair_back_independently() -> None:
    config = ui._default_config()
    config["solder_grading"].update(
        {
            "model_path": str(CLASSIFIER_MODEL),
            "manifest_path": str(CLASSIFIER_MANIFEST),
        }
    )
    config["solder_defect_detection"]["model_path"] = str(DETECTOR_MODEL)

    guarded = ui._engine_config(config)

    assert guarded["solder_grading"]["model_path"] == str(CLASSIFIER_MODEL)
    assert guarded["solder_grading"]["manifest_path"] == str(CLASSIFIER_MANIFEST)
    assert guarded["solder_defect_detection"]["model_path"] is None
    assert guarded["solder_defect_detection"]["manifest_path"] is None


ROLE_HARNESS = '''
import sys
from pathlib import Path

sys.path.insert(0, {root!r})

import streamlit as st
import app.streamlit_app as ui


class _Upload:
    def __init__(self, path: str) -> None:
        source = Path(path)
        self.name = source.name
        self._data = source.read_bytes()

    def getvalue(self) -> bytes:
        return self._data


ui._init_state()

# A fresh session adopts both role-specific active folders.
assert Path(st.session_state.solder_model_path) == Path({classifier_model!r})
assert Path(st.session_state.solder_manifest_path) == Path({classifier_manifest!r})
assert Path(st.session_state.solder_segmenter_model_path) == Path({detector_model!r})
assert Path(st.session_state.solder_segmenter_manifest_path) == Path({detector_manifest!r})

# Cross-wiring either manifest must fail before any state is changed.
classifier_before = st.session_state.solder_manifest_path
detector_before = st.session_state.solder_segmenter_manifest_path
try:
    ui._set_solder_segmenter_manifest(_Upload({classifier_manifest!r}))
except ValueError as exc:
    assert "detector/segment" in str(exc)
else:
    raise AssertionError("classifier manifest was accepted by detector slot")
assert st.session_state.solder_segmenter_manifest_path == detector_before

try:
    ui._set_solder_manifest(_Upload({detector_manifest!r}))
except ValueError as exc:
    assert "classifier ROI" in str(exc)
else:
    raise AssertionError("detector manifest was accepted by classifier slot")
assert st.session_state.solder_manifest_path == classifier_before

# Removing the detector must leave the classifier pair and grading config intact.
ui._remove_solder_segmenter()
assert st.session_state.solder_segmenter_model_path is None
assert st.session_state.solder_segmenter_manifest_path is None
assert st.session_state.solder_model_path == {classifier_model!r}
assert st.session_state.solder_manifest_path == {classifier_manifest!r}
assert st.session_state.config["solder_grading"]["model_path"] == {classifier_model!r}
assert st.session_state.config["solder_grading"]["manifest_path"] == {classifier_manifest!r}

st.write("ROLE_SPLIT_OK")
'''


def test_active_adoption_cross_manifest_guards_and_independent_remove(
    tmp_path: Path,
) -> None:
    from streamlit.testing.v1 import AppTest

    assert DETECTOR_MODEL.is_file() and DETECTOR_MANIFEST.is_file()
    assert CLASSIFIER_MODEL.is_file() and CLASSIFIER_MANIFEST.is_file()
    script = tmp_path / "solder_role_harness.py"
    script.write_text(
        ROLE_HARNESS.format(
            root=str(ui.PROJECT_ROOT),
            detector_model=str(DETECTOR_MODEL),
            detector_manifest=str(DETECTOR_MANIFEST),
            classifier_model=str(CLASSIFIER_MODEL),
            classifier_manifest=str(CLASSIFIER_MANIFEST),
        ),
        encoding="utf-8",
    )
    app = AppTest.from_file(str(script), default_timeout=120).run()

    assert not app.exception, [str(item.value) for item in app.exception]
    assert any("ROLE_SPLIT_OK" in item.value for item in app.markdown)

