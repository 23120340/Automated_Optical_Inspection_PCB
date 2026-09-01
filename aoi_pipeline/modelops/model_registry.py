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

A model is only offered when its manifest sits beside it. Steps 5.2, 6.1 and
6.2 all refuse half a contract at load time -- an ONNX whose class order is
unknown would have to be guessed at, and guessing wrong can change the solder
ROI topology or map every defect onto a pass -- so offering the file alone
would only produce a failure one click later.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

__all__ = [
    "ACTIVE_ROOT",
    "ARCHIVE_ROOT",
    "LIBRARY_ROOT",
    "MODELS_ROOT",
    "ModelEntry",
    "ModelFolderRenameError",
    "ModelSummary",
    "discover_models",
    "find_active",
    "rename_model_folder",
]

MODELS_ROOT = Path(__file__).resolve().parents[2] / "models"
ACTIVE_ROOT = MODELS_ROOT / "active"
ARCHIVE_ROOT = MODELS_ROOT / "archive"
LIBRARY_ROOT = MODELS_ROOT / "library"

#: Folder name under ``models/active`` for each pipeline stage.
STAGE_FOLDERS = {
    "detector": "detector",
    "classifier": "classifier",
    # Step 5.2 package topology.  It remains opt-in in the UI even when an
    # artifact is installed here; see ``app.streamlit_app._NO_AUTO_ADOPT``.
    "package_classifier": "package",
    "solder_classifier": "solder/classifier",
    # ``solder_segmenter`` (detect lỗi trên toàn board) đã được GỠ khỏi app và
    # pipeline: dự án đi theo hướng lượt 2 (định vị mối hàn) + 6.2 (chấm từng
    # ROI), và lớp chẩn đoán toàn board chỉ là một đường thứ ba, ``diagnostic_only``
    # và train trên camera khác.
    #
    # Module ``aoi_pipeline/solder/defect_detection.py`` và model trong
    # ``models/active/solder/defect/`` vẫn còn trên đĩa; thiếu ở đây nghĩa là
    # không có ô nào tự nạp chúng nữa.
    # Lượt 2 của bước 5.5: tìm mối hàn BÊN TRONG crop linh kiện. Khác cả ba ô
    # trên. Không phải ``detector`` (ô đó nhìn cả board và học thân linh kiện),
    # không phải ``solder_segmenter`` (ô đó khoanh LỖI và không có lớp nào cho
    # mối hàn lành). Ô này cần model khoanh MỌI mối hàn, kể cả lành, vì thứ nó
    # cung cấp là *vị trí* để bước 6.2 chấm, không phải phán quyết.
    "lead_detector": "lead_detector",
}

# ``solder`` was the public name of the classifier slot before step 6.2 was
# split into two independent artifacts.  Keep accepting it at API boundaries
# so saved UI state and third-party scripts keep working, but never emit it as
# a ModelEntry.kind: entries always carry the unambiguous canonical role.
#
# ``solder_detector`` is the same story one step later.  The board-level
# yolov8m-seg artifact was called a "detector", which put it one word away from
# the two slots that genuinely localise things for inspection -- the pass-1
# component detector and the pass-2 lead detector -- and the three were read as
# interchangeable.
#
# ``solder_segmenter`` no longer has a folder under ``models/active``: the
# whole-board defect stage was removed from the app and the pipeline. The kind
# is still recognised so an artifact sitting in ``models/archive`` or
# ``models/library`` still classifies rather than showing up as "unknown", but
# nothing loads one any more.
_KIND_ALIASES = {
    "solder": "solder_classifier",
    "solder_detector": "solder_segmenter",
    "package": "package_classifier",
}
_SOLDER_KINDS = frozenset(("solder_classifier", "solder_segmenter"))
# Unknown artifacts must never leak into roles whose output changes inspection
# geometry.  A family classifier can safely stay discoverable as ``unknown``
# for backwards compatibility; package and localisation roles need a positive
# manifest/folder identity before the picker may offer them.
_STRICT_KINDS = _SOLDER_KINDS | frozenset(("package_classifier", "lead_detector"))


def _canonical_kind(kind: str | None) -> str | None:
    if kind is None:
        return None
    return _KIND_ALIASES.get(kind, kind)


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
#: Danh tính BỀN của một model. Tên file luôn là ``best.onnx``, tên thư mục do
#: người đặt và có thể lệch, đường dẫn thì riêng từng máy — chỉ sha256 gắn với
#: chính artifact. Đây là khoá để một bản ghi đánh giá còn ý nghĩa sau khi
#: model bị thay.
_SHA256_PATHS = (
    ("model", "sha256"),
    ("sha256",),
    ("model", "digest"),
    ("onnx", "sha256"),
)
#: Chỉ số đầu bảng của từng loại model, xếp theo thứ tự ưu tiên. Cặp
#: (đường dẫn, nhãn hiển thị).
_HEADLINE_METRICS = (
    (("metrics", "val", "map50"), "mAP50"),
    (("metrics", "map50"), "mAP50"),
    (("reported_metrics", "map50_mask"), "mask mAP50"),
    (("reported_metrics", "map50_box"), "box mAP50"),
    # Model phát hiện lỗi mối hàn của bước 6.2 chỉ có một đầu ra nên ghi
    # ``map50`` trần, không hậu tố. Thiếu dòng này thì bảng chọn model in "—"
    # cho model ĐANG CHẠY trong khi bản cũ kém hơn lại khoe được điểm của nó,
    # và người chọn tưởng bản mới chưa từng được đo.
    (("reported_metrics", "map50"), "mAP50"),
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
    #: Thêm sau cùng, có mặc định, để ``ModelSummary() == ModelSummary()`` giữ
    #: nguyên. KHÔNG đưa vào ``as_line()``: nhãn bộ chọn không được dài thêm.
    sha256: str | None = None
    #: Task khai trong manifest. Ô ``solder_segmenter`` nhận được cả model
    #: detect lẫn model segment kể từ khi hợp đồng được mở, và tên ô không nói
    #: được đang là cái nào -- nhưng đổi model sẽ đổi hành vi của pipeline. Đây
    #: là chỗ duy nhất đọc được điều đó, nên bảng liệt kê phải in nó ra.
    #: Cùng lý do với ``sha256``: đứng cuối, có mặc định, không vào ``as_line()``.
    task: str | None = None

    @property
    def metric(self) -> str | None:
        if self.metric_name is None or self.metric_value is None:
            return None
        return f"{self.metric_name} {self.metric_value:.3f}"

    def as_line(self) -> str:
        """Một dòng gọn cho bộ chọn: kiến trúc · ngày · điểm."""

        parts = [part for part in (self.architecture, self.created, self.metric) if part]
        return " · ".join(parts)


def _summarise(
    manifest: Mapping[str, Any] | None,
    filename: str | None = None,
) -> ModelSummary:
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
    # Schema cũ của detector ghi digest theo TÊN FILE dưới ``files``, không
    # phải ở một chỗ cố định. Thử đường đó trước khi rơi về các vị trí chung.
    paths = _SHA256_PATHS
    if filename:
        paths = (("files", filename, "sha256"),) + paths
    digest = _first(manifest, paths)
    task = _first(manifest, (("task",), ("aoi_compatibility", "required_ultralytics_task")))
    return ModelSummary(
        architecture=str(architecture) if architecture else None,
        created=str(created)[:10] if created else None,
        version=(lambda v: str(v) if v else None)(_first(manifest, _VERSION_PATHS)),
        metric_name=metric_name,
        metric_value=metric_value,
        sha256=str(digest) if digest else None,
        task=str(task) if task else None,
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
        return _summarise(self.manifest(), self.model_path.name)

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


class ModelFolderRenameError(ValueError):
    """The requested registry-folder rename is unsafe or cannot be completed."""


_WINDOWS_RESERVED_FOLDER_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "CONIN$",
    "CONOUT$",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def _rename_root(origin: str) -> Path:
    if origin == "library":
        return LIBRARY_ROOT
    if origin == "archive":
        return ARCHIVE_ROOT
    if origin == "active":
        raise ModelFolderRenameError(
            "Không thể đổi tên thư mục active vì ứng dụng dùng đường dẫn cố định này."
        )
    raise ModelFolderRenameError("Model này không nằm trong library hoặc archive.")


def _clean_folder_name(value: str) -> str:
    """Validate one portable folder component and return its trimmed form."""

    if not isinstance(value, str):
        raise ModelFolderRenameError("Tên thư mục phải là chuỗi ký tự.")
    name = value.strip()
    if not name:
        raise ModelFolderRenameError("Tên thư mục không được để trống.")
    if name != value:
        raise ModelFolderRenameError(
            "Tên thư mục không được bắt đầu hoặc kết thúc bằng khoảng trắng."
        )
    if name in {".", ".."}:
        raise ModelFolderRenameError("Tên thư mục không được là '.' hoặc '..'.")
    if len(name) > 255:
        raise ModelFolderRenameError("Tên thư mục không được dài quá 255 ký tự.")
    if any(character in name for character in '/\\<>:"|?*'):
        raise ModelFolderRenameError(
            "Tên thư mục không được chứa /, \\, <, >, :, |, ?, * hoặc dấu ngoặc kép."
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        raise ModelFolderRenameError("Tên thư mục không được chứa ký tự điều khiển.")
    if name.endswith((".", " ")):
        raise ModelFolderRenameError(
            "Tên thư mục không được kết thúc bằng dấu chấm hoặc khoảng trắng."
        )
    device_name = name.split(".", 1)[0].upper()
    if device_name in _WINDOWS_RESERVED_FOLDER_NAMES:
        raise ModelFolderRenameError(f"'{name}' là tên dành riêng của Windows.")
    return name


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def rename_model_folder(entry: ModelEntry, new_name: str) -> ModelEntry:
    """Rename the folder containing a library/archive model without overwriting.

    ``active`` stage folders are an application contract and therefore cannot be
    renamed.  The complete model folder is moved so manifests and any training
    sidecars stay beside their ONNX artifact.
    """

    folder_name = _clean_folder_name(new_name)
    root = _rename_root(entry.origin)
    try:
        root_resolved = root.resolve(strict=True)
        model_resolved = entry.model_path.resolve(strict=True)
        source = model_resolved.parent
    except OSError as exc:
        raise ModelFolderRenameError(
            f"Không tìm thấy thư mục model để đổi tên: {exc}"
        ) from exc

    if not model_resolved.is_file():
        raise ModelFolderRenameError("Artifact model không phải là một file hợp lệ.")
    if source == root_resolved:
        raise ModelFolderRenameError("Không thể đổi tên chính thư mục library/archive.")
    if not _inside(source, root_resolved):
        raise ModelFolderRenameError("Đường dẫn model nằm ngoài registry được phép đổi tên.")

    # Refuse a symlink/junction masquerading as the model folder. Resolving the
    # path above protects against escapes; this additionally makes the object
    # being renamed unambiguous on every supported platform.
    try:
        logical_source = entry.model_path.parent
        is_junction = getattr(logical_source, "is_junction", lambda: False)
        if logical_source.is_symlink() or is_junction():
            raise ModelFolderRenameError(
                "Không đổi tên thư mục model là symlink/junction."
            )
    except OSError as exc:
        raise ModelFolderRenameError(
            f"Không kiểm tra được thư mục model: {exc}"
        ) from exc

    if source.name == folder_name:
        raise ModelFolderRenameError("Tên mới trùng với tên hiện tại.")

    destination = source.parent / folder_name
    destination_parent = destination.parent.resolve(strict=True)
    if not _inside(destination_parent, root_resolved):
        raise ModelFolderRenameError("Tên mới sẽ đưa model ra ngoài registry.")

    model_relative = model_resolved.relative_to(source)
    manifest_relative: Path | None = None
    if entry.manifest_path is not None:
        try:
            manifest_resolved = entry.manifest_path.resolve(strict=False)
            manifest_relative = manifest_resolved.relative_to(source)
        except (OSError, ValueError) as exc:
            raise ModelFolderRenameError(
                "Manifest của model không nằm trong cùng thư mục với artifact."
            ) from exc

    case_only = source.name.casefold() == folder_name.casefold()
    if os.path.lexists(destination):
        try:
            destination_is_source = source.samefile(destination)
        except OSError:
            destination_is_source = False
        if not destination_is_source:
            raise ModelFolderRenameError(f"Thư mục '{folder_name}' đã tồn tại.")

    try:
        if case_only:
            # Windows treats differently-cased paths as the same destination.
            # A unique sibling hop makes the requested casing deterministic.
            intermediate = source.parent / f".{source.name}.rename-{uuid4().hex}"
            source.rename(intermediate)
            try:
                intermediate.rename(destination)
            except OSError:
                intermediate.rename(source)
                raise
        else:
            source.rename(destination)
    except OSError as exc:
        raise ModelFolderRenameError(f"Không thể đổi tên thư mục model: {exc}") from exc

    renamed_model = destination / model_relative
    renamed_manifest = (
        destination / manifest_relative if manifest_relative is not None else None
    )
    relative_model = renamed_model.relative_to(root_resolved)
    return ModelEntry(
        name=relative_model.as_posix(),
        kind=entry.kind,
        model_path=renamed_model,
        manifest_path=renamed_manifest,
        origin=entry.origin,
    )


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


def _read_manifest(manifest_path: Path | None) -> Mapping[str, Any] | None:
    if manifest_path is None:
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


#: ``task`` trong manifest -> bước nào của đường ống nạp nó. Đây là nguồn đáng
#: tin nhất: tên thư mục do người đặt và có thể sai, ``task`` do chính notebook
#: sinh ra cùng lúc với trọng số.
_TASK_TO_KIND = {
    "component_family_classification": "classifier",
    "component_package_classification": "package_classifier",
    "solder_defect_classification": "solder_classifier",
    "solder_defect_instance_segmentation": "solder_segmenter",
    # Task mà notebook 6.2 của chính dự án sinh ra
    # (``training/kaggle/pcb_solder_detector_kaggle.py``). Thiếu nó thì một bản
    # copy thả vào ``models/library`` không được phân loại, đúng luồng
    # "của bạn" mà bộ chọn model quảng cáo.
    "solder_defect_detection": "solder_segmenter",
    "component_and_lead_detection": "detector",
    "component_detection": "detector",
    "detect": "detector",
    # Task do ``training/kaggle/pcb_joint_locator_kaggle.py`` sinh ra. Tên nó
    # chứa "solder" nên phải nằm ở đây: ``_solder_role_from_hint`` sẽ trả None
    # cho nó (không có token detector nào), và thiếu dòng này thì một model
    # lượt 2 thả vào ``models/library`` không được phân loại.
    "solder_joint_localization": "lead_detector",
}


def _solder_role_from_hint(value: str) -> str | None:
    """Read a solder role from a schema or a possibly nested folder path.

    A bare ``solder`` is deliberately not enough here: it was historically the
    classifier folder, but it cannot distinguish a classifier from the new
    segmentation detector.  The caller handles that one legacy folder only
    after trying all explicit hints.
    """

    lowered = value.strip().lower().replace("\\", "/")
    if not lowered:
        return None

    # These two schemas are contracts, not user-controlled folder names, so
    # recognise them even if a future copy is stored outside ``solder/``.
    if "pcb-solder-defect-classifier" in lowered:
        return "solder_classifier"
    if "aoi-external-yolo-segmentation" in lowered:
        return "solder_segmenter"

    if "solder" not in lowered:
        return None
    classifier_hint = "classifier" in lowered or "classification" in lowered
    detector_hint = any(
        token in lowered
        for token in (
            "detector",
            "detection",
            "instance_segmentation",
            "segmentation",
            # ``segmenter`` is the project's OWN folder name for this slot
            # (``STAGE_FOLDERS["solder_segmenter"]``) and was not in this list,
            # so the resolver could not read back a path it had itself written.
            #
            # ``defect`` deliberately stays out: it appears in
            # ``solder_defect_classification`` too, so adding it would make both
            # hints fire on the classifier and the function would abstain on a
            # role it currently resolves correctly.
            "segmenter",
        )
    )
    if classifier_hint == detector_hint:
        # Neither hint, or contradictory hints: do not guess and accidentally
        # offer one artifact in both solder pickers.
        return None
    return "solder_classifier" if classifier_hint else "solder_segmenter"


def _kind_of(manifest: Mapping[str, Any] | None, folder: str) -> str:
    """Model này thuộc bước nào.

    Ưu tiên ``task`` trong manifest, rồi mới đến tên thư mục. Một model tải từ
    ngoài về có thể nằm trong thư mục tên gì cũng được, nhưng nếu nó mang
    ``task`` thì ta biết chắc.
    """

    if manifest:
        task = manifest.get("task")
        if isinstance(task, str):
            task_kind = _TASK_TO_KIND.get(task.strip().lower())
            if task_kind is not None:
                return task_kind
        schema = str(manifest.get("schema_version", "")).strip().lower()
        if "pcb-package-classifier" in schema:
            return "package_classifier"
        solder_role = _solder_role_from_hint(schema)
        if solder_role is not None:
            return solder_role
        if "solder" in schema:
            return "unknown"
        for token, mapped in (("detector", "detector"),
                              ("classifier", "classifier")):
            if token in schema:
                return mapped

    lowered = folder.strip().lower().replace("\\", "/")
    package_parts = tuple(part for part in lowered.split("/") if part)
    if package_parts and (
        package_parts[-1] == "package"
        or (
            "package" in package_parts[-1]
            and any(token in package_parts[-1] for token in ("classifier", "classification"))
        )
    ):
        return "package_classifier"
    solder_role = _solder_role_from_hint(lowered)
    if solder_role is not None:
        return solder_role
    # Compatibility for an old active/solder/best.onnx layout.  New folders
    # must say classifier or detector explicitly.
    if lowered.strip("/") == "solder":
        return "solder_classifier"
    if "solder" in lowered:
        return "unknown"
    for token, mapped in (("detector", "detector"),
                          ("classifier", "classifier")):
        if token in lowered:
            return mapped
    return "unknown"


def _scan(root: Path, origin: str, kind: str | None) -> Iterable[ModelEntry]:
    if not root.is_dir():
        return
    for model_path in sorted(root.rglob("*.onnx")):
        relative = model_path.relative_to(root)
        # Keep the complete nested path.  Looking only at the first component
        # turns both active/solder/classifier and active/solder/segmenter into
        # the same ambiguous hint: ``solder``.
        folder = relative.parent.as_posix() if relative.parent != Path(".") else ""
        manifest_path = _manifest_beside(model_path)
        manifest = _read_manifest(manifest_path)
        entry_kind = _kind_of(manifest, folder)

        # Lọc theo loại ở MỌI nguồn, không riêng ``active``. Trước đây bộ lọc
        # chỉ áp cho ``active``, nên bộ chọn model mối hàn 6.2 chào cả
        # classifier lẫn detector -- nạp vào thì hỏng ở một chỗ chẳng liên quan
        # gì tới nguyên nhân. Model không xác định được loại vẫn được chào ở
        # mọi bước, vì giấu hẳn nó đi thì người thả file vào không hiểu vì sao
        # nó biến mất.
        if kind is not None:
            if kind in _STRICT_KINDS:
                # A truly unknown model must not appear in both solder slots:
                # their output contracts (raw logits vs boxes+masks) are not
                # interchangeable.  Requiring an explicit role fails safely.
                if entry_kind != kind:
                    continue
            elif entry_kind not in (kind, "unknown"):
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

    canonical_kind = _canonical_kind(kind)
    entries: list[ModelEntry] = []
    entries.extend(_scan(ACTIVE_ROOT, "active", canonical_kind))
    entries.extend(_scan(LIBRARY_ROOT, "library", canonical_kind))
    entries.extend(_scan(ARCHIVE_ROOT, "archive", canonical_kind))
    if require_manifest:
        entries = [entry for entry in entries if entry.has_manifest]
    return entries


def find_active(kind: str) -> ModelEntry | None:
    """The model this stage loads when nobody chooses otherwise."""

    canonical_kind = _canonical_kind(kind)
    folder = STAGE_FOLDERS.get(canonical_kind)
    if folder is None:
        return None
    model_path = ACTIVE_ROOT / folder / "best.onnx"
    if not model_path.is_file():
        return None
    manifest_path = _manifest_beside(model_path)
    manifest = _read_manifest(manifest_path)
    if manifest is not None and _kind_of(manifest, folder) != canonical_kind:
        # The fixed active path is not enough evidence when its adjacent
        # contract explicitly says it belongs to another role.
        return None
    return ModelEntry(
        name=f"{folder}/best.onnx",
        kind=canonical_kind,
        model_path=model_path,
        manifest_path=manifest_path,
        origin="active",
    )
