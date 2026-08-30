"""Người vận hành ghi lại chỗ model làm sai — bằng toạ độ, không phải ảnh.

Mọi đánh giá model trong dự án đến giờ đều do một người đo trên **một board
chuẩn**. Người thật sự nhìn thấy model sai ở đâu là người đứng máy, và họ chưa
có chỗ nào để nói ra. Module này là chỗ đó.

Bản ghi giữ **toạ độ**, ảnh được cắt lại khi cần xem — cùng nguyên tắc và cùng
bộ máy với :mod:`aoi_pipeline.reporting.evidence`, chỉ khác chủ thể: `evidence` ghi lỗi
*của board*, còn đây ghi lỗi *của model*.

Ba điều quyết định thiết kế:

**Không có đường dẫn ảnh gốc để mà lưu.** App giải mã ảnh thẳng vào bộ nhớ
(`st.session_state.input_image`) và không bao giờ ghi nó ra đĩa. Nên bản ghi
giữ `source_name` + `source_sha256`, và dựng :class:`SourceFingerprint` lúc đọc,
trỏ vào chỗ file đang nằm *hôm nay*. Digest và kích thước khung vẫn là con số đã
ghi, nên `EvidenceViewer` vẫn từ chối đúng lúc cần từ chối. Thêm nữa file này
được commit, mà `AGENTS.md` cấm export đường dẫn tuyệt đối của máy.

**Danh tính model nằm trên TỪNG bản ghi, và là sha256.** Tên file luôn là
``best.onnx``, tên thư mục do người đặt, đường dẫn riêng từng máy — chỉ sha256
gắn với chính artifact. Không có nó thì đổi model một lần là toàn bộ đánh giá cũ
thành vô nghĩa; có nó thì hai model **so được với nhau trên cùng những lỗi đã
báo**, và đó mới là thứ đáng giá nhất ở đây.

**Ghi nối thêm từng dòng (JSON Lines), không phải một mảng JSON.** Hai phiên
Streamlit cùng ghi một mảng JSON là đọc-sửa-ghi cả file, và một người sẽ mất bản
ghi mà không ai biết. Một dòng ngắn ghi vào handle mở chế độ ``"a"`` được hệ
điều hành đặt ở cuối file ngay lúc ghi, nên hai phiên không chèn vào nhau được.
Kèm theo: hai người thêm hai dòng khác nhau thì git gộp được, còn một mảng JSON
thì dấu ``]`` là trạng thái chung và luôn xung đột.
"""

from __future__ import annotations

import json
import os
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

try:                                    # Windows
    import msvcrt
except ImportError:                     # pragma: no cover - POSIX
    msvcrt = None                       # type: ignore[assignment]
try:                                    # POSIX
    import fcntl
except ImportError:                     # pragma: no cover - Windows
    fcntl = None                        # type: ignore[assignment]

from ..reporting.evidence import DefectEvidence, EvidenceBundle, SourceFingerprint
from ..models import BoundingBox, utc_now_iso

__all__ = [
    "ERROR_KINDS",
    "FeedbackEntry",
    "FeedbackError",
    "FeedbackTargetRef",
    "MAX_COMMENT_CHARS",
    "ModelIdentity",
    "SCHEMA_VERSION",
    "append_feedback",
    "entries_for_source",
    "error_label",
    "evidence_bundle_for",
    "feedback_root",
    "group_by_model",
    "load_feedback",
    "preprocess_identity",
]

SCHEMA_VERSION = "aoi-model-feedback/1.0"

#: Bình luận là ghi chú, không phải chỗ dán ảnh. Chặn ở tầng lưu trữ chứ không
#: chỉ ở widget: giới hạn của widget là gợi ý, còn đây là bất biến của file.
MAX_COMMENT_CHARS = 1000
MAX_LINE_BYTES = 8 * 1024

DEFAULT_LOG_NAME = "model_feedback.jsonl"
_FEEDBACK_ENV = "AOI_FEEDBACK_DIR"

#: Từ vựng lỗi, riêng cho từng bước. Cố định trong mã chứ không để người dùng
#: gõ tự do: "sai nhãn" và "gán nhãn sai" mà thành hai loại khác nhau thì không
#: đếm được gì cả. Ghi chú tự do vẫn còn ở trường `comment`.
ERROR_KINDS: dict[str, tuple[tuple[str, str], ...]] = {
    "detection": (
        ("wrong_label", "Sai nhãn — khoanh đúng chỗ, gọi sai loại"),
        ("false_positive", "Thừa — khoanh vào chỗ không có linh kiện"),
        ("missed", "Bỏ sót — có linh kiện mà không khoanh"),
        ("bad_box", "Khung lệch — đúng linh kiện, khung sai chỗ hoặc sai cỡ"),
        ("duplicate", "Trùng lặp — một linh kiện bị khoanh nhiều lần"),
    ),
    "classification": (
        ("wrong_family", "Sai họ linh kiện"),
        ("should_be_unknown", "Đáng lẽ phải unknown — crop không rõ mà vẫn tự tin"),
        ("wrongly_unknown", "Unknown oan — crop rất rõ mà bị đẩy vào review"),
        ("bad_confidence", "Nhãn đúng nhưng confidence sai lệch"),
    ),
    "solder": (
        ("wrong_defect", "Sai loại khuyết tật"),
        ("false_call", "Báo nhầm — mối hàn tốt bị gọi là lỗi"),
        ("escape", "BỎ SÓT LỖI — mối hàn xấu được cho qua"),
        ("roi_misplaced", "ROI sai chỗ — không nằm trên mối hàn"),
        ("roi_missing", "Thiếu ROI — một chân không được khoanh"),
    ),
}

#: Các tham số tiền xử lý ĐỔI PIXEL. Giữ danh sách trắng thay vì cả dict vì
#: `calibration_profile` là nội tại camera của một máy cụ thể — không nên nằm
#: trong file commit. Bản sắc đầy đủ do `preprocess_sha256` giữ.
_PREPROCESS_KEYS = (
    "undistort", "undistort_alpha", "resize_enabled", "max_side",
    "denoise", "denoise_strength", "white_balance", "clahe", "clahe_clip",
    "normalize", "sharpen",
)

_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")


class FeedbackError(ValueError):
    """Bản ghi đánh giá không hợp lệ, hoặc không ghi/đọc được."""


def feedback_root() -> Path:
    """Thư mục chứa log. ``AOI_FEEDBACK_DIR`` thắng, để test không ghi vào repo.

    Đọc biến môi trường lúc GỌI chứ không lúc import: AppTest chạy trong cùng
    tiến trình, nên `monkeypatch.setenv` phải có tác dụng.
    """

    override = os.environ.get(_FEEDBACK_ENV)
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[2] / "feedback"


def error_label(stage: str, code: str) -> str:
    """Nhãn tiếng Việt của một mã lỗi, hoặc chính mã đó nếu không nhận ra."""

    for candidate, label in ERROR_KINDS.get(stage, ()):
        if candidate == code:
            return label
    return code


def preprocess_identity(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Phần tiền xử lý đáng ghi lại, kèm digest của toàn bộ cấu hình.

    Toạ độ chỉ có nghĩa cùng với ảnh mà nó đo trên. Bật CLAHE hay đổi
    `max_side` là đổi pixel dưới cùng một toạ độ. Danh sách trắng để người đọc
    hiểu được; digest để máy so được.
    """

    import hashlib

    config = config or {}
    kept = {key: config[key] for key in _PREPROCESS_KEYS if key in config}
    canonical = json.dumps(config, sort_keys=True, default=str, ensure_ascii=False)
    kept["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return kept


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Model nào đang bị đánh giá — đủ để nhận ra nó sau khi nó bị thay."""

    slot: str                       # component | classifier | solder
    kind: str                       # loại trong registry
    loaded: bool = False
    name: str | None = None
    origin: str | None = None       # active | library | archive | upload
    sha256: str | None = None
    version: str | None = None
    architecture: str | None = None
    created: str | None = None

    @property
    def compare_key(self) -> str:
        """Khoá gộp. sha256 trước, vì nó gắn với chính file trọng số."""

        return self.sha256 or self.version or self.name or "no-model"

    @property
    def display(self) -> str:
        if not self.loaded:
            return "chưa nạp model"
        parts = [part for part in (self.architecture, self.created) if part]
        head = self.name or self.architecture or "model"
        return f"{head} ({' · '.join(parts)})" if parts else head

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot, "kind": self.kind, "loaded": self.loaded,
            "name": self.name, "origin": self.origin, "sha256": self.sha256,
            "version": self.version, "architecture": self.architecture,
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ModelIdentity:
        return cls(
            slot=str(value.get("slot", "")), kind=str(value.get("kind", "")),
            loaded=bool(value.get("loaded", False)),
            name=value.get("name"), origin=value.get("origin"),
            sha256=value.get("sha256"), version=value.get("version"),
            architecture=value.get("architecture"), created=value.get("created"),
        )


@dataclass(frozen=True, slots=True)
class FeedbackTargetRef:
    """Mục nào trong kết quả bị báo sai, và model đã nói gì về nó lúc đó."""

    record_type: str | None = None     # detection | classification | solder_verdict
    record_id: str | None = None
    model_label: str | None = None
    model_decision: str | None = None
    model_probability: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.record_type, "record_id": self.record_id,
            "model_label": self.model_label, "model_decision": self.model_decision,
            "model_probability": self.model_probability,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> FeedbackTargetRef:
        value = value or {}
        probability = value.get("model_probability")
        return cls(
            record_type=value.get("record_type"), record_id=value.get("record_id"),
            model_label=value.get("model_label"),
            model_decision=value.get("model_decision"),
            model_probability=float(probability) if probability is not None else None,
        )


@dataclass(frozen=True, slots=True)
class FeedbackEntry:
    """Một lần người vận hành nói "model sai ở đây"."""

    stage: str                          # detection | classification | solder
    step: int                           # 4 | 6 | 7
    bbox: tuple[int, int, int, int]     # analysis_image_pixels, xyxy, gốc trên-trái
    error_kind: str
    model: ModelIdentity
    source_name: str
    source_sha256: str
    analysis_width: int
    analysis_height: int
    analysis_stage: int = 0             # 0 gốc · 1 tiền xử lý · 2 đã căn
    image_role: str = "input"
    preprocess: dict[str, Any] = field(default_factory=dict)
    origin: str = "result_row"          # result_row | magnifier
    # Cạnh ô người dùng chọn ở chế độ kính lúp, tính bằng pixel ảnh phân tích.
    # `bbox` cũng mang kích thước, NHƯNG nó bị cắt khi ô chạm mép ảnh -- lúc đó
    # kích thước thật đã mất. Giữ riêng để lượt train sau đọc được ý định của
    # người ghi, không phải phần còn lại sau khi cắt.
    box_size: int | None = None
    target: FeedbackTargetRef = field(default_factory=FeedbackTargetRef)
    expected_label: str | None = None
    comment: str = ""
    runtime_mode: str = ""
    entry_id: str = field(default_factory=lambda: uuid4().hex)
    recorded_at: str = field(default_factory=utc_now_iso)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.stage not in ERROR_KINDS:
            raise FeedbackError(f"Bước không hợp lệ: {self.stage!r}")
        codes = {code for code, _ in ERROR_KINDS[self.stage]}
        if self.error_kind not in codes:
            # Một lỗi gõ sai ở giao diện không được đẻ ra một loại không đếm được.
            raise FeedbackError(
                f"Loại lỗi {self.error_kind!r} không thuộc bước {self.stage!r}. "
                f"Chọn một trong: {sorted(codes)}"
            )
        x1, y1, x2, y2 = self.bbox
        if x2 <= x1 or y2 <= y1:
            raise FeedbackError(f"Khung rỗng hoặc lật ngược: {self.bbox}")
        if len(self.comment) > MAX_COMMENT_CHARS:
            raise FeedbackError(
                f"Ghi chú {len(self.comment)} ký tự, quá {MAX_COMMENT_CHARS}."
            )
        if not _HEX64.match(self.source_sha256 or ""):
            raise FeedbackError(f"source_sha256 không phải sha256: {self.source_sha256!r}")

    # -- serialise ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "entry_id": self.entry_id,
            "recorded_at": self.recorded_at,
            "stage": self.stage,
            "step": int(self.step),
            "error_kind": self.error_kind,
            "bbox": [int(value) for value in self.bbox],
            "coordinate_space": "analysis_image_pixels",
            "origin": self.origin,
            "box_size": self.box_size,
            "expected_label": self.expected_label,
            "comment": self.comment,
            "runtime_mode": self.runtime_mode,
            "model": self.model.to_dict(),
            "target": self.target.to_dict(),
            "source": {
                "name": self.source_name,
                "sha256": self.source_sha256,
                "analysis_width": int(self.analysis_width),
                "analysis_height": int(self.analysis_height),
                "analysis_stage": int(self.analysis_stage),
                "image_role": self.image_role,
                "preprocess": dict(self.preprocess),
            },
        }

    def to_json_line(self) -> str:
        """Một bản ghi, đúng MỘT dòng vật lý.

        ``ensure_ascii=False`` giữ dấu tiếng Việt đọc được; ``json.dumps``
        vẫn thoát ``\\n`` bên trong chuỗi, nên một ghi chú nhiều dòng không
        phá được cấu trúc file.
        """

        line = json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))
        if len(line.encode("utf-8")) > MAX_LINE_BYTES:
            raise FeedbackError(
                f"Bản ghi {len(line.encode('utf-8'))} byte, quá {MAX_LINE_BYTES}."
            )
        return line

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FeedbackEntry:
        version = str(value.get("schema_version", ""))
        if version != SCHEMA_VERSION:
            raise FeedbackError(
                f"Bản ghi dùng schema {version!r}, mã này đọc {SCHEMA_VERSION!r}."
            )
        source = value.get("source") or {}
        return cls(
            stage=str(value["stage"]),
            step=int(value.get("step", 0)),
            bbox=tuple(int(item) for item in value["bbox"][:4]),  # type: ignore[arg-type]
            error_kind=str(value["error_kind"]),
            model=ModelIdentity.from_dict(value.get("model") or {}),
            source_name=str(source.get("name", "")),
            source_sha256=str(source.get("sha256", "")),
            analysis_width=int(source.get("analysis_width", 0)),
            analysis_height=int(source.get("analysis_height", 0)),
            analysis_stage=int(source.get("analysis_stage", 0)),
            image_role=str(source.get("image_role", "input")),
            preprocess=dict(source.get("preprocess") or {}),
            origin=str(value.get("origin", "result_row")),
            box_size=(int(value["box_size"]) if value.get("box_size") else None),
            target=FeedbackTargetRef.from_dict(value.get("target")),
            expected_label=value.get("expected_label"),
            comment=str(value.get("comment", "")),
            runtime_mode=str(value.get("runtime_mode", "")),
            entry_id=str(value.get("entry_id", "")) or uuid4().hex,
            recorded_at=str(value.get("recorded_at", "")),
            schema_version=version,
        )

    # -- xem lại pixel -----------------------------------------------------

    def fingerprint(self, source_path: str | Path) -> SourceFingerprint:
        """Vân tay trỏ vào chỗ file đang nằm HÔM NAY.

        Bản ghi không giữ đường dẫn (xem docstring của module), nên người gọi
        phải nói file ở đâu. Digest và kích thước khung vẫn là con số đã ghi,
        nên `EvidenceViewer` vẫn từ chối một file đã đổi hoặc một khung dựng
        lại sai cỡ.
        """

        return SourceFingerprint(
            path=str(source_path),
            sha256=self.source_sha256,
            analysis_width=self.analysis_width,
            analysis_height=self.analysis_height,
            preprocess=dict(self.preprocess),
        )

    def clamped_bbox(self, width: int, height: int) -> tuple[int, int, int, int]:
        return BoundingBox(*(float(v) for v in self.bbox)).clamp(width, height).to_int()


# --------------------------------------------------------------------------
# Ghi
# --------------------------------------------------------------------------


#: Các phiên Streamlit là các LUỒNG trong cùng một tiến trình, nên đây là mức
#: tranh chấp thật sự hay gặp: hai tab trình duyệt cùng bấm "Ghi nhận".
_WRITE_LOCK = threading.Lock()


@contextmanager
def _locked(handle) -> Any:
    """Khoá file ở mức hệ điều hành, cho trường hợp hai tiến trình.

    Đã đo trên Windows: chế độ ``"a"`` và cả ``os.O_APPEND`` đều **không**
    nguyên tử — 8 luồng ghi 200 dòng chỉ còn 175 và 147. Giả định "hệ điều
    hành đặt con trỏ ở cuối file lúc ghi" chỉ đúng trên POSIX. Nên phải khoá
    tường minh chứ không dựa vào chế độ mở file.
    """

    if msvcrt is not None:                      # Windows
        handle.seek(0, os.SEEK_END)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        except OSError:
            # Hết lượt thử của LK_LOCK. Khoá trong tiến trình vẫn đang giữ,
            # nên ghi tiếp còn hơn là để mất bản ghi.
            yield
            return
        try:
            yield
        finally:
            handle.seek(0, os.SEEK_END)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
    elif fcntl is not None:                     # POSIX
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    else:
        yield


def append_feedback(entry: FeedbackEntry, *, path: Path | None = None) -> Path:
    """Nối một bản ghi vào cuối log.

    Ghi có khoá, hai tầng: một `threading.Lock` cho các phiên Streamlit (vốn là
    luồng trong cùng tiến trình) và một khoá file của hệ điều hành cho trường
    hợp hai tiến trình. Xem :func:`_locked` để biết vì sao không thể chỉ dựa
    vào chế độ mở file.

    Lỗi ghi (checkout chỉ đọc, file bị khoá) được để thoát ra ngoài — nuốt nó
    đi là để người dùng tưởng đã lưu trong khi không có gì được lưu.
    """

    line = entry.to_json_line()          # kiểm tra trước khi chạm vào đĩa
    destination = Path(path) if path is not None else feedback_root() / DEFAULT_LOG_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (line + "\n").encode("utf-8")
    with _WRITE_LOCK:
        with destination.open("ab") as handle:
            with _locked(handle):
                handle.seek(0, os.SEEK_END)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
    return destination


# --------------------------------------------------------------------------
# Đọc
# --------------------------------------------------------------------------


def load_feedback(
    *, root: Path | None = None,
) -> tuple[list[FeedbackEntry], list[str]]:
    """Mọi bản ghi trong thư mục, kèm danh sách dòng đọc không được.

    Một dòng hỏng — tiến trình bị giết giữa chừng, hoặc một lần merge vụng —
    **không được** làm cả lịch sử đánh giá thành không đọc được. Nên dòng hỏng
    bị bỏ qua và báo lại, chứ không ném lỗi.

    Trùng ``entry_id`` bị gộp: git merge kiểu union có thể nhân đôi dòng, và
    uuid4 đảm bảo hai người không bao giờ đụng id của nhau.
    """

    directory = Path(root) if root is not None else feedback_root()
    entries: list[FeedbackEntry] = []
    problems: list[str] = []
    seen: set[str] = set()
    if not directory.is_dir():
        return entries, problems

    for file in sorted(directory.glob("**/*.jsonl")):
        try:
            text = file.read_text(encoding="utf-8")
        except OSError as exc:
            problems.append(f"{file.name}: không đọc được ({exc})")
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = FeedbackEntry.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                problems.append(f"{file.name}:{number}: {exc}")
                continue
            if entry.entry_id in seen:
                continue
            seen.add(entry.entry_id)
            entries.append(entry)
    return entries, problems


def entries_for_source(
    entries: Iterable[FeedbackEntry],
    source_sha256: str,
    *,
    stage: str | None = None,
) -> list[FeedbackEntry]:
    """Bản ghi thuộc về đúng tấm ảnh này (và tuỳ chọn: đúng bước này)."""

    return [
        entry for entry in entries
        if entry.source_sha256 == source_sha256
        and (stage is None or entry.stage == stage)
    ]


def group_by_model(entries: Iterable[FeedbackEntry]) -> dict[str, list[FeedbackEntry]]:
    """Gộp theo model. Đây là thứ trả lời "model nào yếu ở đâu"."""

    grouped: dict[str, list[FeedbackEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.model.compare_key, []).append(entry)
    return grouped


def evidence_bundle_for(
    entries: Sequence[FeedbackEntry],
    source_path: str | Path,
) -> EvidenceBundle:
    """Đưa các bản ghi về :class:`EvidenceBundle` để dùng lại `EvidenceViewer`.

    Toàn bộ chuyện cắt lại pixel — giữ đúng một khung, kiểm digest, kiểm kích
    thước, và mọi thông báo từ chối bằng tiếng Việt — đã có sẵn và đã có test ở
    :mod:`aoi_pipeline.reporting.evidence`. Ở đây chỉ ánh xạ sang đúng hình dạng đó, nên
    không có bản sao thứ hai của logic ấy để mà lệch nhau.

    ``joint_id`` nhận ``entry_id``, nên gọi ``viewer.crop(bundle, entry.entry_id)``.
    """

    if not entries:
        raise FeedbackError("Không có bản ghi nào để dựng bundle.")
    digests = {entry.source_sha256 for entry in entries}
    if len(digests) > 1:
        raise FeedbackError(
            "Các bản ghi thuộc nhiều ảnh khác nhau; một bundle chỉ ứng với một ảnh."
        )
    first = entries[0]
    return EvidenceBundle(
        board_id=first.source_name,
        source=first.fingerprint(source_path),
        defects=[
            DefectEvidence(
                joint_id=entry.entry_id,
                detection_id=entry.target.record_id or "",
                label=entry.error_kind,
                decision="feedback",
                bbox=entry.bbox,
                component_label=entry.target.model_label or "",
                reasons=[entry.comment] if entry.comment else [],
                features={"model": entry.model.compare_key, "stage": entry.stage},
            )
            for entry in entries
        ],
    )
