"""Build the checked-in classification notebook from its percent-format source."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "training" / "kaggle" / "pcb_component_classification_kaggle.py"
DESTINATION = SOURCE.with_suffix(".ipynb")


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
            "metadata": {},
            "source": [f"{line}\n" for line in source],
            **({"outputs": [], "execution_count": None} if kind == "code" else {}),
        }
    )


def main() -> None:
    cells: list[dict] = []
    kind: str | None = None
    lines: list[str] = []
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        if line == "# %% [markdown]" or line == "# %%":
            flush(cells, kind, lines)
            kind = "markdown" if "markdown" in line else "code"
            lines = []
        else:
            lines.append(line)
    flush(cells, kind, lines)
    notebook = {
        "cells": cells,
        "metadata": {
            "kaggle": {"accelerator": "gpu", "dataSources": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    DESTINATION.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {DESTINATION} with {len(cells)} cells")


if __name__ == "__main__":
    main()
