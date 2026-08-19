"""Build a Kaggle notebook from its percent-format Python source.

The `.py` is the file that gets edited and reviewed; the `.ipynb` is generated
from it. Keeping the source as plain Python means diffs stay readable and the
cells can be imported and unit-tested, which is what `tests/training/` does.

    python scripts/build_notebook.py training/kaggle/pcb_solder_defect_kaggle.py
    python scripts/build_notebook.py --all

Supersedes build_classification_notebook.py, which hardcoded a single path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DIR = ROOT / "training" / "kaggle"


def flush(cells: list[dict], kind: str | None, lines: list[str]) -> None:
    if kind is None:
        return
    if kind == "markdown":
        source = []
        for line in lines:
            if line.startswith("# "):
                source.append(line[2:])
            elif line.startswith("#"):
                source.append(line[1:])
            else:
                source.append(line)
    else:
        source = lines
    cells.append(
        {
            "cell_type": kind,
            # nbformat 4.5 (nbformat_minor 5) requires a cell id. Derive it from
            # the index and content so rebuilding unchanged source produces a
            # byte-identical notebook instead of a diff full of fresh uuids.
            "id": _cell_id(len(cells), source),
            "metadata": {},
            "source": [f"{line}\n" for line in source],
            **({"outputs": [], "execution_count": None} if kind == "code" else {}),
        }
    )


def _cell_id(index: int, source: list[str]) -> str:
    digest = hashlib.sha256("\n".join(source).encode("utf-8")).hexdigest()
    return f"c{index:03d}{digest[:5]}"


def build(source: Path, accelerator: str = "gpu") -> Path:
    cells: list[dict] = []
    kind: str | None = None
    lines: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line == "# %% [markdown]" or line == "# %%":
            flush(cells, kind, lines)
            kind = "markdown" if "markdown" in line else "code"
            lines = []
        else:
            lines.append(line)
    flush(cells, kind, lines)

    if not cells:
        raise SystemExit(
            f"{source.name}: no '# %%' cell markers found; this is not a "
            "percent-format notebook source."
        )

    notebook = {
        "cells": cells,
        "metadata": {
            "kaggle": {"accelerator": accelerator, "dataSources": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination = source.with_suffix(".ipynb")
    destination.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"Wrote {destination.relative_to(ROOT)} with {len(cells)} cells")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sources", nargs="*", help="Percent-format .py sources to build.")
    parser.add_argument(
        "--all", action="store_true", help="Build every .py in training/kaggle/."
    )
    parser.add_argument("--accelerator", default="gpu", choices=["gpu", "tpu", "none"])
    args = parser.parse_args(argv)

    if args.all:
        sources = sorted(
            path
            for path in KAGGLE_DIR.glob("*.py")
            if not path.name.startswith("_")
        )
    else:
        sources = [Path(item).expanduser().resolve() for item in args.sources]
    if not sources:
        parser.error("Give at least one source, or --all.")

    for source in sources:
        if not source.is_file():
            print(f"Skipping missing source: {source}", file=sys.stderr)
            continue
        build(source, args.accelerator)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
