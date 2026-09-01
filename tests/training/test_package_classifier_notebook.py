"""Static contract checks for the post-labelling package training notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "training" / "kaggle" / "pcb_package_classification_kaggle.py"
NOTEBOOK = SOURCE.with_suffix(".ipynb")


def _notebook_code() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell["cell_type"] == "code"
    )


def test_package_notebook_is_present_and_all_code_cells_parse() -> None:
    assert SOURCE.is_file() and NOTEBOOK.is_file()
    code = _notebook_code()
    compile(code, str(NOTEBOOK), "exec")


def test_package_notebook_pins_the_runtime_contract_and_board_split() -> None:
    code = _notebook_code()
    for required in (
        'INPUT_SIZE = 128',
        'MANIFEST_SCHEMA = "pcb-package-classifier/1.0"',
        'TASK = "component_package_classification"',
        'dataset_manifest.get("split_unit") != "board_scene_id"',
        '"split_unit": "board"',
        'PACKAGE_CLASSES.index(row["class"])',
        '"color_space": "RGB"',
        '"resize_mode": "letterbox"',
        '"default_enabled": False',
        '"real_board_roi_gate_passed": False',
    ):
        assert required in code
    assert "ImageFolder(" not in code, "alphabetical folder order would scramble class ids"


def test_package_notebook_measures_both_required_classification_gates() -> None:
    code = _notebook_code()
    assert 'average="macro"' in code
    assert 'PACKAGE_CLASSES.index("ic_hai_ben")' in code
    assert 'PACKAGE_CLASSES.index("ic_khong_chan")' in code
    assert "dangerous_confusions == 0" in code
    assert "onnxruntime" in code and "max_abs_diff > 1e-4" in code


def test_package_notebook_documents_the_local_roi_gate_and_manual_promotion() -> None:
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert "evaluate_package_roi_gate.py" in text
    assert "models/active/package/" in text
    assert "không tự bật model" in text
