"""Guards against the two ways this project has actually accumulated waste.

Both were found by auditing the tree, not by anyone noticing:

* four modules each carried their own copy of the same box-overlap arithmetic,
  and three files of tests were verbatim copies of tests that already existed
  elsewhere -- 16 of 449 collected tests were the same tests running twice;
* the copies all still agreed, checked on 4005 box pairs. That is the dangerous
  state, not the safe one: nothing fails while they agree, and nothing tells you
  when one of them stops.
"""

from __future__ import annotations

import ast
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tracked(suffix: str = ".py") -> list[str]:
    output = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [line for line in output.splitlines() if line.endswith(suffix)]


def _module_level_functions(path: Path):
    """(name, normalised body) for every top-level def, docstring stripped."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = [
            statement
            for statement in node.body
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            )
        ]
        if len(body) < 2:
            continue          # một dòng thì trùng ngẫu nhiên là chuyện thường
        yield node.name, ast.dump(ast.Module(body=body, type_ignores=[]))


def test_no_helper_is_copied_verbatim_into_two_core_modules() -> None:
    """``aoi_pipeline`` and ``app`` are one codebase; a helper belongs in one
    place there. Training notebooks are exempt: they run on Kaggle where this
    repository does not exist, so they must carry their own copies."""

    seen = defaultdict(list)
    for relative in _tracked():
        if not relative.startswith(("aoi_pipeline/", "app/")):
            continue
        path = ROOT / relative
        if not path.is_file():
            continue
        for name, body in _module_level_functions(path):
            seen[body].append((relative, name))

    duplicates = {
        body: places
        for body, places in seen.items()
        if len({relative for relative, _ in places}) > 1
    }
    assert duplicates == {}, "hàm bị chép nguyên văn sang module khác: " + str(
        [places for places in duplicates.values()]
    )


def test_no_test_is_defined_in_two_files() -> None:
    """A merge left ``tests/test_board_detector_crop.py`` holding 14 tests that
    all already existed elsewhere, plus one more in
    ``tests/imaging/test_preprocessing.py``. They cost runtime, and the moment
    one copy is updated the two quietly stop testing the same thing."""

    by_name = defaultdict(list)
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                by_name[node.name].append(path.relative_to(ROOT).as_posix())

    duplicates = {
        name: files for name, files in by_name.items() if len(set(files)) > 1
    }
    assert duplicates == {}, f"test trùng tên ở nhiều file: {duplicates}"


def test_the_shared_overlap_helpers_have_exactly_one_definition() -> None:
    """Named directly, because these are the ones that were copied four ways."""

    definitions = defaultdict(list)
    for relative in _tracked():
        if not relative.startswith(("aoi_pipeline/", "app/")):
            continue
        path = ROOT / relative
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in {
                "intersection_over_union",
                "intersection_over_smaller",
                "_intersection_over_union",
                "_intersection_over_smaller",
            }:
                definitions[node.name.lstrip("_")].append(relative)

    for name, files in definitions.items():
        assert files == ["aoi_pipeline/models.py"], (
            f"{name} phải chỉ được định nghĩa trong models.py, đang ở: {files}"
        )
