from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

NOTEBOOK_PATH = (
    Path(__file__).parents[1]
    / "training"
    / "kaggle"
    / "pcb_component_detection_kaggle.ipynb"
)


class _YamlStub:
    @staticmethod
    def safe_load(_: str) -> dict[str, object]:
        return {
            "train": "train/images",
            "val": "valid/images",
            "names": {0: "resistor"},
        }


def _resolver_namespace(
    input_root: Path,
    dataset_source: str | None = None,
) -> dict[str, object]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    resolver_source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "def select_dataset_yaml" in "".join(cell["source"])
    )
    tree = ast.parse(resolver_source)
    required_functions = {
        "load_yaml_if_yolo",
        "find_relative_input_yamls",
        "resolve_input_config_path",
        "select_dataset_yaml",
    }
    function_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in required_functions
    ]
    assert {node.name for node in function_nodes} == required_functions

    namespace: dict[str, object] = {
        "Path": Path,
        "yaml": _YamlStub,
        "INPUT_ROOT": input_root,
        "DATA_WORK_DIR": input_root.parent / "working" / "dataset",
        "CONFIG": {
            "dataset_source": dataset_source,
            "data_yaml": "components_data_uncropped/data.yaml",
            "max_extract_gb": 12,
        },
    }
    exec(compile(ast.Module(body=function_nodes, type_ignores=[]), str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace


def test_notebook_finds_relative_yaml_inside_kaggle_mount(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    expected = (
        input_root
        / "pcb-component-detection-consolidated-dataset"
        / "components_data_uncropped"
        / "data.yaml"
    )
    expected.parent.mkdir(parents=True)
    expected.write_text(
        "train: train/images\nval: valid/images\nnames:\n  0: resistor\n",
        encoding="utf-8",
    )

    namespace = _resolver_namespace(input_root)
    selected = namespace["select_dataset_yaml"]()

    assert selected == expected.resolve()


def test_notebook_recovers_when_configured_mount_name_is_stale(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    expected = input_root / "renamed-kaggle-mount" / "components_data_uncropped" / "data.yaml"
    expected.parent.mkdir(parents=True)
    expected.write_text(
        "train: train/images\nval: valid/images\nnames:\n  0: resistor\n",
        encoding="utf-8",
    )

    namespace = _resolver_namespace(
        input_root,
        dataset_source="/kaggle/input/old-mount-name",
    )

    assert namespace["select_dataset_yaml"]() == expected.resolve()


def test_notebook_explains_when_kaggle_input_is_not_attached(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    namespace = _resolver_namespace(
        input_root,
        dataset_source="/kaggle/input/missing-dataset",
    )

    with pytest.raises(FileNotFoundError, match="Add Input"):
        namespace["select_dataset_yaml"]()
