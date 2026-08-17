from __future__ import annotations

import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

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

    @staticmethod
    def safe_dump(value: object, **_: object) -> str:
        return json.dumps(value)


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


def _audit_namespace(tmp_path: Path) -> dict[str, object]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    audit_source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "def audit_dataset" in "".join(cell["source"])
    )
    tree = ast.parse(audit_source)
    required_functions = {
        "expand_split_entries",
        "label_path_for_image",
        "sha256_file",
        "add_issue",
        "audit_dataset",
    }
    function_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in required_functions
    ]
    assert {node.name for node in function_nodes} == required_functions

    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    namespace: dict[str, object] = {
        "Path": Path,
        "IMAGE_EXTS": {".png"},
        "hashlib": hashlib,
        "defaultdict": defaultdict,
        "Counter": Counter,
        "random": random,
        "Image": Image,
        "math": math,
        "np": np,
        "pd": pd,
        "datetime": datetime,
        "timezone": timezone,
        "yaml": _YamlStub,
        "REPORT_DIR": report_dir,
        "SOURCE_YAML": tmp_path / "source.yaml",
        "RESOLVED_YAML": tmp_path / "resolved.yaml",
        "missing_ratio_limit": 0.0,
        "CONFIG": {
            "decode_max_per_split": 500,
            "seed": 42,
            "hash_duplicates": True,
            "allow_edge_crossing_boxes": True,
            "deduplicate_across_splits": True,
            "absent_train_class_is_error": False,
            "allow_negative_images": False,
            "max_missing_label_ratio": 0.0,
            "imgsz": 1280,
        },
    }
    exec(compile(ast.Module(body=function_nodes, type_ignores=[]), str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace


def test_audit_accepts_partial_boxes_and_excludes_cross_split_duplicate(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    for split in ("train", "valid"):
        (dataset / split / "images").mkdir(parents=True)
        (dataset / split / "labels").mkdir(parents=True)

    duplicate_image = Image.new("RGB", (16, 16), (10, 20, 30))
    duplicate_image.save(dataset / "train" / "images" / "board.png")
    duplicate_image.save(dataset / "valid" / "images" / "board-copy.png")
    Image.new("RGB", (16, 16), (30, 20, 10)).save(dataset / "valid" / "images" / "unique.png")

    partial_row = "0 0.98 0.5 0.1 0.2"
    (dataset / "train" / "labels" / "board.txt").write_text(
        f"{partial_row}\n{partial_row}\n",
        encoding="utf-8",
    )
    (dataset / "valid" / "labels" / "board-copy.txt").write_text(
        f"{partial_row}\n",
        encoding="utf-8",
    )
    (dataset / "valid" / "labels" / "unique.txt").write_text(
        "0 0.5 0.5 0.25 0.25\n",
        encoding="utf-8",
    )

    resolved = {
        "path": str(dataset),
        "names": {0: "component"},
        "train": str(dataset / "train" / "images"),
        "val": str(dataset / "valid" / "images"),
    }
    namespace = _audit_namespace(tmp_path)
    report, split_images, _, invalid, _, training_yaml = namespace["audit_dataset"](
        resolved,
        ["component"],
    )

    assert report["error_count"] == 0
    assert report["edge_crossing_boxes"]["count"] == 2
    assert len(report["cross_split_excluded_images"]) == 1
    assert len(split_images["train"]) == 1
    assert len(split_images["val"]) == 1
    assert invalid["reason"].tolist() == ["duplicate_annotation"]
    assert training_yaml.is_file()


def test_ground_truth_renderer_builds_a_three_channel_rgb_color(tmp_path: Path) -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    preview_source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "def render_ground_truth" in "".join(cell["source"])
    )
    tree = ast.parse(preview_source)
    function_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"read_valid_boxes", "render_ground_truth"}
    ]

    class _ColorMaps:
        @staticmethod
        def tab20(_: int) -> tuple[float, float, float, float]:
            return 0.1, 0.2, 0.3, 1.0

    class _PlotStub:
        cm = _ColorMaps()

    image_path = tmp_path / "board.png"
    label_path = tmp_path / "board.txt"
    Image.new("RGB", (16, 16), "white").save(image_path)
    label_path.write_text("0 0.98 0.5 0.1 0.2\n", encoding="utf-8")
    namespace = {
        "Path": Path,
        "Image": Image,
        "ImageDraw": ImageDraw,
        "plt": _PlotStub(),
        "CLASS_NAMES": ["component"],
        "label_path_for_image": lambda _: label_path,
    }
    exec(compile(ast.Module(body=function_nodes, type_ignores=[]), str(NOTEBOOK_PATH), "exec"), namespace)

    rendered = namespace["render_ground_truth"](image_path)

    assert rendered.mode == "RGB"
    assert rendered.getpixel((15, 6)) == (26, 51, 76)
