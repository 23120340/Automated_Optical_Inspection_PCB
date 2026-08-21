"""Find the models on disk instead of asking someone to upload them again.

Three places, and the difference between them is who owns the file:

``models/active/<kind>/``
    What the app loads by default. One folder per stage, each holding
    ``best.onnx`` and its ``model_manifest.json``. These are committed, so a
    fresh clone can inspect a board without anyone hunting for weights.

``models/archive/``
    Earlier versions kept for comparison. Never loaded automatically; a model
    that is no longer the best one should not be one careless click away from
    being used on a production board.

``models/library/``
    Yours. Drop anything here and it appears in the picker beside the active
    ones. Git ignores it, so it never fights a pull and never bloats the repo.

A model is only offered when its manifest sits beside it. Step 6.1 and 6.2 both
refuse half a contract at load time -- an ONNX whose class order is unknown
would have to be guessed at, and guessing wrong maps every defect onto a pass --
so offering the file alone would only produce a failure one click later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

__all__ = [
    "ACTIVE_ROOT",
    "ARCHIVE_ROOT",
    "LIBRARY_ROOT",
    "MODELS_ROOT",
    "ModelEntry",
    "discover_models",
    "find_active",
]

MODELS_ROOT = Path(__file__).resolve().parents[1] / "models"
ACTIVE_ROOT = MODELS_ROOT / "active"
ARCHIVE_ROOT = MODELS_ROOT / "archive"
LIBRARY_ROOT = MODELS_ROOT / "library"

#: Folder name under ``models/active`` for each pipeline stage.
STAGE_FOLDERS = {
    "detector": "detector",
    "classifier": "classifier",
    "solder": "solder",
}


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """One loadable model: the weights, its contract, and where it came from."""

    name: str
    kind: str
    model_path: Path
    manifest_path: Path | None
    origin: str          # "active" | "archive" | "library"

    @property
    def has_manifest(self) -> bool:
        return self.manifest_path is not None and self.manifest_path.is_file()

    @property
    def label(self) -> str:
        tag = {"active": "đang dùng", "archive": "bản cũ", "library": "của bạn"}
        return f"{self.name} ({tag.get(self.origin, self.origin)})"

    def manifest(self) -> dict | None:
        if not self.has_manifest:
            return None
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None


def _manifest_beside(model_path: Path) -> Path | None:
    """The contract that belongs to this file, if it is there.

    Checked by name first so ``best.onnx`` prefers ``best.manifest.json`` over a
    folder-wide ``model_manifest.json`` when a folder holds two models.
    """

    candidates = (
        model_path.with_suffix(".manifest.json"),
        model_path.parent / f"{model_path.stem}_manifest.json",
        model_path.parent / "model_manifest.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _scan(root: Path, origin: str, kind: str | None) -> Iterable[ModelEntry]:
    if not root.is_dir():
        return
    for model_path in sorted(root.rglob("*.onnx")):
        relative = model_path.relative_to(root)
        folder = relative.parts[0] if len(relative.parts) > 1 else ""
        entry_kind = folder if folder in STAGE_FOLDERS else (kind or "unknown")
        if kind is not None and entry_kind != kind and origin == "active":
            continue
        name = str(relative.parent) if relative.parent != Path(".") else model_path.stem
        yield ModelEntry(
            name=f"{name}/{model_path.name}" if name else model_path.name,
            kind=entry_kind,
            model_path=model_path,
            manifest_path=_manifest_beside(model_path),
            origin=origin,
        )


def discover_models(kind: str | None = None, *, require_manifest: bool = True) -> list[ModelEntry]:
    """Every model on disk that can actually be loaded, active ones first.

    ``.pt`` files are deliberately not listed. They carry pickle, the app blocks
    them until a person confirms the source, and a picker that offers one by
    default makes that confirmation a formality.
    """

    entries: list[ModelEntry] = []
    entries.extend(_scan(ACTIVE_ROOT, "active", kind))
    entries.extend(_scan(LIBRARY_ROOT, "library", kind))
    entries.extend(_scan(ARCHIVE_ROOT, "archive", kind))
    if require_manifest:
        entries = [entry for entry in entries if entry.has_manifest]
    return entries


def find_active(kind: str) -> ModelEntry | None:
    """The model this stage loads when nobody chooses otherwise."""

    folder = STAGE_FOLDERS.get(kind)
    if folder is None:
        return None
    model_path = ACTIVE_ROOT / folder / "best.onnx"
    if not model_path.is_file():
        return None
    return ModelEntry(
        name=f"{folder}/best.onnx",
        kind=kind,
        model_path=model_path,
        manifest_path=_manifest_beside(model_path),
        origin="active",
    )
