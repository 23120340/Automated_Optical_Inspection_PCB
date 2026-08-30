"""UI contract for renaming a model folder from the sidebar picker."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import aoi_pipeline.modelops.model_registry as model_registry
from aoi_pipeline.modelops.model_registry import ModelEntry, discover_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_folder_rename_repoints_every_loaded_slot_and_solder_config(tmp_path: Path) -> None:
    from app.streamlit_app import _sync_renamed_model_folder

    old_folder = tmp_path / "ten-cu"
    new_folder = tmp_path / "Tên model dễ nhớ"
    old = ModelEntry(
        name="ten-cu/best.onnx",
        kind="solder",
        model_path=old_folder / "best.onnx",
        manifest_path=old_folder / "model_manifest.json",
        origin="library",
    )
    renamed = ModelEntry(
        name="Tên model dễ nhớ/best.onnx",
        kind="solder",
        model_path=new_folder / "best.onnx",
        manifest_path=new_folder / "model_manifest.json",
        origin="library",
    )
    outside = tmp_path / "khac" / "best.onnx"
    state = {
        "component_model_path": str(old_folder / "best.onnx"),
        "component_model_name": "ten-cu/best.onnx",
        "classifier_model_path": str(old_folder / "nested" / "classifier.onnx"),
        "classifier_model_name": "ten-cu/nested/classifier.onnx",
        "classifier_manifest_path": str(old_folder / "nested" / "model_manifest.json"),
        "classifier_manifest_name": "model_manifest.json",
        "solder_model_path": str(outside),
        "solder_model_name": "khac/best.onnx",
        "solder_manifest_path": None,
        "solder_manifest_name": None,
        "config": {
            "solder_grading": {
                "model_path": str(old_folder / "best.onnx"),
                "manifest_path": str(old_folder / "model_manifest.json"),
            }
        },
    }

    affected = _sync_renamed_model_folder(state, old, renamed)

    assert affected == {"component", "classifier", "solder"}
    assert state["component_model_path"] == str(new_folder / "best.onnx")
    assert state["component_model_name"] == "Tên model dễ nhớ/best.onnx"
    assert state["classifier_model_path"] == str(new_folder / "nested" / "classifier.onnx")
    assert state["classifier_model_name"] == "Tên model dễ nhớ/nested/classifier.onnx"
    assert state["classifier_manifest_path"] == str(
        new_folder / "nested" / "model_manifest.json"
    )
    assert state["solder_model_path"] == str(outside), "đường dẫn ngoài folder phải giữ nguyên"
    assert state["config"]["solder_grading"] == {
        "model_path": str(new_folder / "best.onnx"),
        "manifest_path": str(new_folder / "model_manifest.json"),
    }
    for slot in affected:
        assert state[f"{slot}_model_choice_reset"] is True


def test_folder_label_has_double_click_keyboard_and_safe_text_protocol() -> None:
    from app.streamlit_app import (
        _MODEL_FOLDER_COMPONENT_HTML,
        _MODEL_FOLDER_COMPONENT_JS,
    )

    assert "dblclick" in _MODEL_FOLDER_COMPONENT_JS
    assert "F2" in _MODEL_FOLDER_COMPONENT_JS
    assert "Enter" in _MODEL_FOLDER_COMPONENT_JS
    assert "textContent" in _MODEL_FOLDER_COMPONENT_JS
    assert "innerHTML" not in _MODEL_FOLDER_COMPONENT_JS
    assert "model_id" in _MODEL_FOLDER_COMPONENT_JS
    assert "Đổi tên thư mục model" in _MODEL_FOLDER_COMPONENT_HTML


def test_sidebar_rename_form_moves_the_selected_folder_and_refreshes_state(
    tmp_path: Path, monkeypatch
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

    folder = roots["library"] / "solder-kho-nho"
    folder.mkdir()
    (folder / "best.onnx").write_bytes(b"onnx")
    (folder / "model_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "pcb-solder-defect-classifier/1.0",
                "task": "solder_defect_classification",
                "model": {"architecture": "yolov8m"},
                "created_at": "2026-08-24T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    app = AppTest.from_file(
        str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=180
    )
    app.run()
    assert not app.exception

    entry = discover_models("solder")[0]
    app.sidebar.selectbox(key="solder_model_choice").set_value(entry.label).run()
    assert app.session_state["solder_model_path"] == str(entry.model_path)

    entry_id = hashlib.sha256(
        f"{entry.origin}\0{entry.model_path.resolve(strict=False)}".encode("utf-8")
    ).hexdigest()
    app.session_state["solder_model_rename_target"] = entry_id
    app.run()
    assert not app.exception

    input_key = f"solder_model_rename_name_{entry_id[:12]}"
    app.sidebar.text_input(key=input_key).set_value("Model mối hàn AOI")
    next(button for button in app.sidebar.button if button.label == "Lưu tên").click()
    app.run()

    destination = roots["library"] / "Model mối hàn AOI"
    assert not app.exception
    assert not folder.exists()
    assert (destination / "best.onnx").read_bytes() == b"onnx"
    assert app.session_state["solder_model_path"] == str(destination / "best.onnx")
    assert app.session_state["solder_manifest_path"] == str(
        destination / "model_manifest.json"
    )
    assert app.session_state["solder_model_name"] == "Model mối hàn AOI/best.onnx"
    assert app.sidebar.selectbox(key="solder_model_choice").value.startswith("Model mối hàn AOI")
