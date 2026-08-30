"""Classifier decision thresholds can be overridden for controlled UI tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import aoi_pipeline.modelops.model_registry as model_registry
import app.streamlit_app as ui
from aoi_pipeline.config import PipelineConfig
from aoi_pipeline.modelops.model_registry import discover_models


def test_convnext_manifest_thresholds_are_available_to_the_ui() -> None:
    manifest = (
        ui.PROJECT_ROOT
        / "models"
        / "library"
        / "classifier-convnext_base-ver1"
        / "model_manifest.json"
    )
    if not manifest.is_file():
        pytest.skip("ConvNeXt v2 manifest is not present in this working copy")

    assert ui._classifier_manifest_thresholds(manifest) == pytest.approx((0.85, 0.50))


def test_classifier_threshold_override_reaches_the_typed_pipeline_config() -> None:
    config = ui._default_config()

    assert ui._set_classifier_threshold_override(
        config, enabled=True, accept=0.68, review=0.31
    )
    typed = PipelineConfig.from_mapping(config)
    assert typed.classification.accept_threshold == pytest.approx(0.68)
    assert typed.classification.review_threshold == pytest.approx(0.31)

    assert ui._set_classifier_threshold_override(config, enabled=False)
    typed = PipelineConfig.from_mapping(config)
    assert typed.classification.accept_threshold is None
    assert typed.classification.review_threshold is None


def test_classifier_threshold_order_is_fail_closed() -> None:
    config = ui._default_config()

    with pytest.raises(ValueError, match="review ≤ accept"):
        ui._set_classifier_threshold_override(
            config, enabled=True, accept=0.40, review=0.70
        )

    assert config["classification"]["accept_threshold"] is None
    assert config["classification"]["review_threshold"] is None


def test_sidebar_unlocks_convnext_thresholds_and_can_restore_manifest_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    roots = {
        "active": tmp_path / "active",
        "archive": tmp_path / "archive",
        "library": tmp_path / "library",
    }
    for root in roots.values():
        root.mkdir()
    monkeypatch.setattr(model_registry, "ACTIVE_ROOT", roots["active"])
    monkeypatch.setattr(model_registry, "ARCHIVE_ROOT", roots["archive"])
    monkeypatch.setattr(model_registry, "LIBRARY_ROOT", roots["library"])

    folder = roots["library"] / "classifier-convnext_base-ver2"
    folder.mkdir()
    model = folder / "best.onnx"
    model.write_bytes(b"onnx")
    manifest = folder / "model_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "pcb-component-classifier/1.0",
                "task": "component_family_classification",
                "created_at": "2026-08-24T00:00:00Z",
                "model": {"architecture": "convnext_base", "version": "v2"},
                "decision_thresholds": {
                    "accept": 0.85,
                    "review": 0.50,
                    "accept_by_class": {},
                },
            }
        ),
        encoding="utf-8",
    )

    app = AppTest.from_file(
        str(ui.PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=180
    )
    app.run()
    assert not app.exception

    entry = discover_models("classifier")[0]
    app.sidebar.selectbox(key="classifier_model_choice").set_value(entry.label).run()
    assert app.session_state["classifier_manifest_path"] == str(manifest)

    toggle = app.sidebar.toggle(key="classifier_threshold_override_enabled")
    assert toggle.value is False
    toggle.set_value(True).run()
    assert not app.exception

    manifest_id = hashlib.sha256(
        str(manifest.resolve(strict=False)).encode("utf-8")
    ).hexdigest()[:12]
    slider_key = f"classifier_threshold_range_{manifest_id}"
    slider = app.sidebar.slider(key=slider_key)
    assert tuple(slider.value) == pytest.approx((0.50, 0.85))

    app.session_state["classification_result"] = "stale-result"
    slider.set_value((0.30, 0.67)).run()
    classification = app.session_state.config["classification"]
    assert classification["review_threshold"] == pytest.approx(0.30)
    assert classification["accept_threshold"] == pytest.approx(0.67)
    assert app.session_state["classification_result"] is None

    app.sidebar.toggle(key="classifier_threshold_override_enabled").set_value(False).run()
    classification = app.session_state.config["classification"]
    assert classification["review_threshold"] is None
    assert classification["accept_threshold"] is None
    assert not app.exception
