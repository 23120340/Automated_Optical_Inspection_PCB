"""``aoi_pipeline`` has one front door, and moving files must not narrow it.

The package was regrouped into ``imaging/``, ``detection/``, ``solder/``,
``golden/`` and ``grading/``. Every module moved; ``__init__.py`` is what keeps
that invisible to callers. This pins the surface so the next reshuffle cannot
quietly drop a name -- an import that disappears is only found by whoever
happens to import it.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
from pathlib import Path

import pytest

import aoi_pipeline

ROOT = Path(__file__).resolve().parents[1]

# Captured before the modules were regrouped.
EXPECTED_EXPORTS = 145


def test_every_declared_export_can_actually_be_imported() -> None:
    """``__all__`` is a promise. A name in it that does not resolve is a broken
    promise that only shows up at someone else's import."""

    missing = [name for name in aoi_pipeline.__all__ if not hasattr(aoi_pipeline, name)]
    assert missing == [], f"có trong __all__ nhưng không import được: {missing}"


def test_the_public_surface_did_not_shrink() -> None:
    assert len(aoi_pipeline.__all__) >= EXPECTED_EXPORTS, (
        f"__all__ còn {len(aoi_pipeline.__all__)} tên, trước khi gom nhóm là "
        f"{EXPECTED_EXPORTS}. Gom nhóm không được làm hẹp cửa trước."
    )


def test_no_name_is_exported_twice() -> None:
    """Two modules exporting the same name through ``__init__`` means one of
    them silently wins, and which one depends on import order."""

    seen = list(aoi_pipeline.__all__)
    duplicates = sorted({name for name in seen if seen.count(name) > 1})
    assert duplicates == [], f"tên bị export hai lần: {duplicates}"


def test_no_module_shares_its_name_with_its_own_package() -> None:
    """The collision that broke the package once already: a folder named
    ``inspection/`` next to a module named ``inspection.py``. Both resolve to
    ``aoi_pipeline.inspection`` and the import that loses is chosen by
    filesystem order."""

    package = ROOT / "aoi_pipeline"
    for directory in package.rglob("*"):
        if not directory.is_dir() or directory.name == "__pycache__":
            continue
        clash = directory.parent / f"{directory.name}.py"
        assert not clash.exists(), (
            f"{clash.relative_to(ROOT)} trùng tên với thư mục "
            f"{directory.relative_to(ROOT)}/"
        )
        inner = directory / f"{directory.name}.py"
        assert not inner.exists(), (
            f"{inner.relative_to(ROOT)} trùng tên với thư mục chứa nó"
        )


@pytest.mark.parametrize(
    "module",
    [
        "aoi_pipeline.imaging.image_io",
        "aoi_pipeline.imaging.preprocessing",
        "aoi_pipeline.imaging.calibration",
        "aoi_pipeline.imaging.alignment",
        "aoi_pipeline.imaging.board",
        "aoi_pipeline.detection.detectors",
        "aoi_pipeline.detection.tiling",
        "aoi_pipeline.detection.cropping",
        "aoi_pipeline.solder.geometry",
        "aoi_pipeline.solder.leads",
        "aoi_pipeline.solder.lead_detection",
        "aoi_pipeline.solder.cad",
        "aoi_pipeline.solder.cad_fusion",
        "aoi_pipeline.golden.recipe",
        "aoi_pipeline.golden.inspector",
        "aoi_pipeline.golden.position",
        "aoi_pipeline.golden.compare",
        "aoi_pipeline.grading.features",
        "aoi_pipeline.grading.rules",
        "aoi_pipeline.grading.classifier",
        "aoi_pipeline.grading.inspector",
        "aoi_pipeline.classification",
        "aoi_pipeline.pipeline",
    ],
)
def test_each_module_imports_on_its_own(module: str) -> None:
    """Importing one module must not require the package façade to have run
    first, or a cycle is hiding behind import order."""

    assert importlib.import_module(module) is not None


def test_no_source_file_still_points_at_a_pre_move_module_path() -> None:
    """Dotted paths in docs, scripts and notebooks go stale silently: nothing
    imports them, so nothing fails."""

    moved = (
        "aoi_pipeline.alignment", "aoi_pipeline.board", "aoi_pipeline.calibration",
        "aoi_pipeline.preprocessing", "aoi_pipeline.image_io",
        "aoi_pipeline.detectors", "aoi_pipeline.tiling", "aoi_pipeline.cropping",
        "aoi_pipeline.leads", "aoi_pipeline.lead_detection", "aoi_pipeline.cad_fusion",
        "aoi_pipeline.recipe", "aoi_pipeline.inspection", "aoi_pipeline.position",
        "aoi_pipeline.golden_compare",
    )
    tracked = subprocess.run(
        ["git", "ls-files", "*.py", "*.md"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()

    stale: list[str] = []
    for relative in tracked:
        path = ROOT / relative
        if not path.is_file() or relative.startswith("tests/test_public_api"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for name in moved:
            if f"{name} " in text or f"{name}\n" in text or f"{name}." in text:
                stale.append(f"{relative}: {name}")
    assert stale == [], "đường dẫn module cũ còn sót:\n  " + "\n  ".join(stale)
