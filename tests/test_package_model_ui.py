"""Package 5.2 must be discoverable but remain explicit and role-safe."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import app.streamlit_app as ui
from app.pipeline_bridge import CropRecord, PipelineBridge
from aoi_pipeline.classification.package import (
    MANIFEST_SCHEMA,
    MANIFEST_TASK,
    PACKAGE_CLASS_NAMES,
    PackageClassification,
)
from aoi_pipeline.modelops.model_registry import _kind_of
from aoi_pipeline.models import ClassProbability


ROOT = Path(__file__).resolve().parents[1]


def test_registry_and_ui_give_package_a_strict_separate_slot() -> None:
    assert ui.PACKAGE_MANIFEST_SCHEMA == MANIFEST_SCHEMA
    assert ui.PACKAGE_MANIFEST_TASK == MANIFEST_TASK
    assert ui.PACKAGE_CLASSES == PACKAGE_CLASS_NAMES
    assert ui._MODEL_SLOTS["package"][0] == "package_classifier"
    assert "package" in ui._NO_AUTO_ADOPT
    assert _kind_of({"task": MANIFEST_TASK}, "anything") == "package_classifier"
    assert _kind_of({"schema_version": MANIFEST_SCHEMA}, "anything") == (
        "package_classifier"
    )
    assert _kind_of(None, "package") == "package_classifier"


def test_default_config_is_operational_but_has_no_package_artifact() -> None:
    config = ui._default_config()
    assert config["package_classification"]["enabled"] is True
    assert config["package_classification"]["apply_to_solder_geometry"] is True
    source = inspect.getsource(ui._get_bridge)
    assert "package_ready" in source
    assert "if package_ready else None" in source
    assert '_render_model_picker("package")' in inspect.getsource(ui._render_sidebar)


def test_ui_rejects_family_manifest_in_the_package_slot() -> None:
    family = json.loads(
        (ROOT / "models" / "active" / "classifier" / "model_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    with pytest.raises(ValueError, match="contract package"):
        ui._validate_package_manifest(family)

    package = json.loads(
        (ROOT / "docs" / "thiet_ke" / "package_model_manifest_template.json").read_text(
            encoding="utf-8"
        )
    )
    ui._validate_package_manifest(package)


class _PackageEngine:
    def classify_packages(self, crops):
        crop = crops[0]
        return [PackageClassification(
            crop_id=crop.crop_id,
            detection_id=crop.detection_id,
            package_class="ic_hai_ben",
            probability=0.91,
            top_k=[ClassProbability("ic_hai_ben", 0.91)],
            unknown_score=0.09,
            decision="accept",
            model_version="fixture",
        )]


def test_bridge_exposes_real_package_results_without_family_fallback() -> None:
    bridge = PipelineBridge.__new__(PipelineBridge)
    bridge.package_model_path = "best.onnx"
    bridge.package_manifest_path = "model_manifest.json"
    bridge.engine = _PackageEngine()
    bridge.engine_error = None
    raw = SimpleNamespace(crop_id="crop_1", detection_id="det_1")
    crop = CropRecord(
        crop_id="crop_1", label="ic", image=np.zeros((16, 16, 3), np.uint8),
        bbox=(0, 0, 16, 16), confidence=0.8, source="fixture", raw=raw,
    )

    result = bridge.classify_packages([crop])

    assert result.classifications[0].package_class == "ic_hai_ben"
    assert result.classifications[0].decision == "accept"
    assert result.classifications[0].raw is not None


def test_bridge_refuses_a_half_package_contract() -> None:
    bridge = PipelineBridge.__new__(PipelineBridge)
    bridge.package_model_path = "best.onnx"
    bridge.package_manifest_path = None
    bridge.engine = _PackageEngine()
    bridge.engine_error = None
    with pytest.raises(RuntimeError, match="đủ best.onnx"):
        bridge.classify_packages([])
