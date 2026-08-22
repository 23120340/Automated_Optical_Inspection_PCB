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
from typing import Any, Iterable, Mapping

__all__ = [
    "ACTIVE_ROOT",
    "ARCHIVE_ROOT",
    "LIBRARY_ROOT",
    "MODELS_ROOT",
    "ModelEntry",
    "ModelSummary",
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


#: Nơi từng schema cất cùng một thông tin. Bốn artifact của dự án dùng bốn
#: cách khác nhau, và một model tải từ ngoài về sẽ dùng cách thứ năm. Đọc theo
#: danh sách ứng viên thay vì ép một schema là cách duy nhất để bộ chọn không
#: hiện "unknown" ngay lần đầu ai đó thả file lạ vào ``models/library``.
_ARCHITECTURE_PATHS = (
    ("model", "architecture"),
    ("base_model",),
    ("model_family",),
    ("model", "name"),
    ("architecture",),
)
_CREATED_PATHS = (
    ("created_at",),
    ("created_at_utc",),
    ("model", "created_at"),
)
_VERSION_PATHS = (
    ("model", "version"),
    ("run_name",),
    ("version",),
)
#: Chỉ số đầu bảng của từng loại model, xếp theo thứ tự ưu tiên. Cặp
#: (đường dẫn, nhãn hiển thị).
_HEADLINE_METRICS = (
    (("metrics", "val", "map50"), "mAP50"),
    (("metrics", "map50"), "mAP50"),
    (("metrics", "accuracy"), "acc"),
    (("training", "test_accuracy"), "acc"),
    (("training", "test_macro_recall"), "macro recall"),
    (("metrics", "macro_f1"), "macro F1"),
    (("training", "best_macro_recall"), "macro recall"),
)


def _dig(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    """Đi theo một chuỗi khoá, trả None nếu hụt ở bất kỳ bậc nào."""

    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _first(payload: Mapping[str, Any], paths: tuple[tuple[str, ...], ...]) -> Any:
    for path in paths:
        value = _dig(payload, path)
        if value not in (None, "", {}, []):
            return value
    return None


@dataclass(frozen=True, slots=True)
class ModelSummary:
    """Những gì cần để phân biệt hai model với nhau, đọc từ manifest.

    Tên file không nói được điều này -- mọi artifact trong dự án đều tên
    ``best.onnx``. Manifest thì có, và nó không lệch khỏi file như tên thư mục
    có thể lệch.
    """

    architecture: str | None = None
    created: str | None = None      # YYYY-MM-DD
    version: str | None = None
    metric_name: str | None = None
    metric_value: float | None = None

    @property
    def metric(self) -> str | None:
        if self.metric_name is None or self.metric_value is None:
            return None
        return f"{self.metric_name} {self.metric_value:.3f}"

    def as_line(self) -> str:
        """Một dòng gọn cho bộ chọn: kiến trúc · ngày · điểm."""

        parts = [part for part in (self.architecture, self.created, self.metric) if part]
        return " · ".join(parts)


def _summarise(manifest: Mapping[str, Any] | None) -> ModelSummary:
    if not manifest:
        return ModelSummary()
    architecture = _first(manifest, _ARCHITECTURE_PATHS)
    if isinstance(architecture, str):
        # ``yolo26s.pt`` là tên file trọng số gốc, không phải kiến trúc.
        architecture = architecture.removesuffix(".pt")
    created = _first(manifest, _CREATED_PATHS)
    metric_name = metric_value = None
    for path, display in _HEADLINE_METRICS:
        value = _dig(manifest, path)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metric_name, metric_value = display, float(value)
            break
    return ModelSummary(
        architecture=str(architecture) if architecture else None,
        created=str(created)[:10] if created else None,
        version=(lambda v: str(v) if v else None)(_first(manifest, _VERSION_PATHS)),
        metric_name=metric_name,
        metric_value=metric_value,
    )


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
    def size_mb(self) -> float:
        try:
            return self.model_path.stat().st_size / 1e6
        except OSError:
            return 0.0

    def summary(self) -> ModelSummary:
        return _summarise(self.manifest())

    @property
    def label(self) -> str:
        """Nhãn cho bộ chọn.

        Trước đây chỉ có ``classifier/best.onnx (đang dùng)``, mà mọi artifact
        đều tên ``best.onnx`` -- nên hai bản khác hẳn nhau trông y hệt. Nay kèm
        kiến trúc, ngày tạo và chỉ số đầu bảng, tất cả lấy từ manifest.
        """

        tag = {"active": "đang dùng", "archive": "bản cũ", "library": "của bạn"}
        folder = self.name.rsplit("/", 1)[0] if "/" in self.name else self.name
        detail = self.summary().as_line()
        head = f"{folder} — {detail}" if detail else folder
        return f"{head}  ({tag.get(self.origin, self.origin)})"

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


#: ``task`` trong manifest -> bước nào của đường ống nạp nó. Đây là nguồn đáng
#: tin nhất: tên thư mục do người đặt và có thể sai, ``task`` do chính notebook
#: sinh ra cùng lúc với trọng số.
_TASK_TO_KIND = {
    "component_family_classification": "classifier",
    "solder_defect_classification": "solder",
    "component_and_lead_detection": "detector",
    "component_detection": "detector",
    "detect": "detector",
}


def _kind_of(manifest: Mapping[str, Any] | None, folder: str) -> str:
    """Model này thuộc bước nào.

    Ưu tiên ``task`` trong manifest, rồi mới đến tên thư mục. Một model tải từ
    ngoài về có thể nằm trong thư mục tên gì cũng được, nhưng nếu nó mang
    ``task`` thì ta biết chắc.
    """

    if manifest:
        task = manifest.get("task")
        if isinstance(task, str) and task in _TASK_TO_KIND:
            return _TASK_TO_KIND[task]
        schema = str(manifest.get("schema_version", ""))
        for token, mapped in (("detector", "detector"), ("solder", "solder"),
                              ("classifier", "classifier")):
            if token in schema:
                return mapped
    lowered = folder.lower()
    for token, mapped in (("detector", "detector"), ("solder", "solder"),
                          ("classifier", "classifier")):
        if token in lowered:
            return mapped
    return "unknown"


def _scan(root: Path, origin: str, kind: str | None) -> Iterable[ModelEntry]:
    if not root.is_dir():
        return
    for model_path in sorted(root.rglob("*.onnx")):
        relative = model_path.relative_to(root)
        folder = relative.parts[0] if len(relative.parts) > 1 else ""
        manifest_path = _manifest_beside(model_path)
        manifest = None
        if manifest_path is not None:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                manifest = None
        entry_kind = _kind_of(manifest, folder)

        # Lọc theo loại ở MỌI nguồn, không riêng ``active``. Trước đây bộ lọc
        # chỉ áp cho ``active``, nên bộ chọn model mối hàn 6.2 chào cả
        # classifier lẫn detector -- nạp vào thì hỏng ở một chỗ chẳng liên quan
        # gì tới nguyên nhân. Model không xác định được loại vẫn được chào ở
        # mọi bước, vì giấu hẳn nó đi thì người thả file vào không hiểu vì sao
        # nó biến mất.
        if kind is not None and entry_kind not in (kind, "unknown"):
            continue

        name = str(relative.parent) if relative.parent != Path(".") else model_path.stem
        yield ModelEntry(
            name=f"{name}/{model_path.name}" if name else model_path.name,
            kind=entry_kind,
            model_path=model_path,
            manifest_path=manifest_path,
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
