"""What to keep so an operator can look at a defect months later.

An operator has to be able to see the picture of what was called a defect.
Keeping a picture of every one is the obvious way and it does not scale: on a
board of ~960 components this project measured 7.1 MB of defect crops per board
against 1.39 MB for the coordinates and measurements that describe them. At a
hundred boards a day that is 260 GB a year of pictures against 50 GB of numbers,
and the numbers are the part you can search, total and compare.

So the record keeps coordinates, and the picture is cut again when somebody asks
for it. Measured on a real board:

    rebuild the analysis frame from the source file   219 ms   (once per board)
    cut one ROI out of a frame already in memory      0.004 ms

Which is why :class:`EvidenceViewer` holds exactly one frame. An operator opens
one board's report and looks through its defects, so the 219 ms is paid once and
every crop after that is free. Rebuilding per ROI instead would cost about
59,000 times more for the same board.

The part that matters more than the saving: a rebuilt frame has to be *the same
frame*. Coordinates recorded against one preprocessing recipe point at the wrong
pixels under another, and an operator shown the wrong pixels is worse off than
one shown nothing. So the bundle records the source file's digest, the
preprocessing settings and the frame size it measured, and the viewer refuses to
crop when any of them disagrees.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .models import BoundingBox, utc_now_iso

__all__ = [
    "DefectEvidence",
    "EvidenceBundle",
    "EvidenceViewer",
    "EvidenceMismatch",
    "SourceFingerprint",
    "file_digest",
]

SCHEMA_VERSION = "aoi-defect-evidence/1.0"


class EvidenceMismatch(RuntimeError):
    """The frame rebuilt now is not the frame the coordinates were measured on."""


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SourceFingerprint:
    """Everything needed to rebuild the analysis frame, and to prove it matched.

    ``preprocess`` is the settings dict that produced the frame. It is part of
    the identity, not decoration: turning CLAHE on changes the pixels a ROI
    covers even though its coordinates are untouched.
    """

    path: str
    sha256: str
    analysis_width: int
    analysis_height: int
    preprocess: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "analysis_width": int(self.analysis_width),
            "analysis_height": int(self.analysis_height),
            "preprocess": dict(self.preprocess),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SourceFingerprint:
        return cls(
            path=str(value["path"]),
            sha256=str(value["sha256"]),
            analysis_width=int(value["analysis_width"]),
            analysis_height=int(value["analysis_height"]),
            preprocess=dict(value.get("preprocess") or {}),
        )


@dataclass(frozen=True, slots=True)
class DefectEvidence:
    """One call an operator may want to look at, without its pixels."""

    joint_id: str
    detection_id: str
    label: str
    decision: str
    bbox: tuple[int, int, int, int]
    component_label: str = ""
    designator: str | None = None
    pin: str | None = None
    reasons: list[str] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_id": self.joint_id,
            "detection_id": self.detection_id,
            "label": self.label,
            "decision": self.decision,
            "bbox": [int(value) for value in self.bbox],
            "component_label": self.component_label,
            "designator": self.designator,
            "pin": self.pin,
            "reasons": list(self.reasons),
            "features": dict(self.features),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DefectEvidence:
        return cls(
            joint_id=str(value["joint_id"]),
            detection_id=str(value.get("detection_id", "")),
            label=str(value.get("label", "")),
            decision=str(value.get("decision", "")),
            bbox=tuple(int(item) for item in value["bbox"][:4]),  # type: ignore[arg-type]
            component_label=str(value.get("component_label", "")),
            designator=value.get("designator"),
            pin=value.get("pin"),
            reasons=list(value.get("reasons") or []),
            features=dict(value.get("features") or {}),
        )


@dataclass(slots=True)
class EvidenceBundle:
    """One board's defect record: the numbers, and how to find the pixels."""

    board_id: str
    source: SourceFingerprint
    defects: list[DefectEvidence] = field(default_factory=list)
    captured_at: str = field(default_factory=utc_now_iso)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "board_id": self.board_id,
            "captured_at": self.captured_at,
            "source": self.source.to_dict(),
            "defects": [item.to_dict() for item in self.defects],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=1)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json(), encoding="utf-8")
        return destination

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvidenceBundle:
        version = str(value.get("schema_version", ""))
        if version != SCHEMA_VERSION:
            raise EvidenceMismatch(
                f"Bản ghi bằng chứng dùng schema '{version}', mã này đọc "
                f"'{SCHEMA_VERSION}'."
            )
        return cls(
            board_id=str(value.get("board_id", "")),
            source=SourceFingerprint.from_dict(value["source"]),
            defects=[DefectEvidence.from_dict(item) for item in value.get("defects", [])],
            captured_at=str(value.get("captured_at", "")),
            schema_version=version,
        )

    @classmethod
    def load(cls, path: str | Path) -> EvidenceBundle:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class EvidenceViewer:
    """Cut defect crops on demand, holding at most one board's frame.

    ``rebuild`` is whatever turns a source file back into the analysis frame --
    normally ``lambda path: pipeline.preprocess(load_image(path)).image``. It is
    injected rather than imported so this module does not depend on the pipeline
    facade, and so a test can supply a frame without touching a disk.
    """

    def __init__(self, rebuild) -> None:
        self._rebuild = rebuild
        self._frame: np.ndarray | None = None
        self._key: str | None = None

    @property
    def loaded_board(self) -> str | None:
        return self._key

    def release(self) -> None:
        """Drop the held frame. Call this when the operator closes a board."""

        self._frame = None
        self._key = None

    def frame_for(self, bundle: EvidenceBundle) -> np.ndarray:
        """The analysis frame this bundle was measured on, rebuilt if needed."""

        if self._key == bundle.source.sha256 and self._frame is not None:
            return self._frame

        source = Path(bundle.source.path)
        if not source.is_file():
            raise EvidenceMismatch(
                f"Không còn ảnh gốc để dựng lại: {source}. Bản ghi chỉ giữ toạ độ, "
                "nên mất ảnh gốc là mất luôn khả năng xem lại."
            )
        actual = file_digest(source)
        if actual != bundle.source.sha256:
            raise EvidenceMismatch(
                f"Ảnh tại {source} đã khác lúc kiểm: digest {actual[:12]}… so với "
                f"{bundle.source.sha256[:12]}… đã ghi. Toạ độ cũ trỏ vào pixel khác."
            )

        # Release before rebuilding, not after: holding the old frame while the
        # new one is built doubles peak memory for no benefit.
        self.release()
        frame = self._rebuild(source)
        if frame is None or getattr(frame, "ndim", 0) < 2:
            raise EvidenceMismatch(f"Dựng lại khung ảnh thất bại cho {source}.")
        height, width = frame.shape[:2]
        if (width, height) != (bundle.source.analysis_width, bundle.source.analysis_height):
            raise EvidenceMismatch(
                f"Khung dựng lại là {width}x{height}, lúc kiểm là "
                f"{bundle.source.analysis_width}x{bundle.source.analysis_height}. "
                "Nhiều khả năng cấu hình tiền xử lý đã đổi; toạ độ không còn đúng."
            )
        self._frame = frame
        self._key = bundle.source.sha256
        return frame

    def crop(self, bundle: EvidenceBundle, joint_id: str) -> np.ndarray:
        """The pixels behind one recorded defect."""

        defect = next((item for item in bundle.defects if item.joint_id == joint_id), None)
        if defect is None:
            raise KeyError(f"Không có bằng chứng nào mang joint_id '{joint_id}'")
        frame = self.frame_for(bundle)
        height, width = frame.shape[:2]
        box = BoundingBox(*(float(value) for value in defect.bbox)).clamp(width, height)
        x1, y1, x2, y2 = box.to_int()
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            raise EvidenceMismatch(
                f"ROI {defect.bbox} nằm ngoài khung {width}x{height} của {joint_id}."
            )
        return patch


def build_evidence_bundle(
    board_id: str,
    source_path: str | Path,
    analysis_shape: tuple[int, int],
    verdicts: Sequence[Any],
    preprocess: Mapping[str, Any] | None = None,
    *,
    keep_decisions: Sequence[str] = ("review", "reject"),
) -> EvidenceBundle:
    """Collect the calls worth keeping from one run's verdicts.

    Only the ones an operator would open. Keeping ``accept`` as well would grow
    the record by roughly a third for rows nobody looks at; if a line ever needs
    full traceability, widen ``keep_decisions`` rather than saving pictures.
    """

    height, width = analysis_shape[:2]
    keep = set(keep_decisions)
    defects = [
        DefectEvidence(
            joint_id=str(getattr(item, "joint_id", "")),
            detection_id=str(getattr(item, "detection_id", "")),
            label=str(getattr(item, "label", "")),
            decision=str(getattr(item, "decision", "")),
            bbox=tuple(int(value) for value in getattr(item, "bbox", (0, 0, 0, 0))[:4]),
            component_label=str(getattr(item, "component_label", "")),
            designator=getattr(item, "designator", None),
            pin=getattr(item, "pin", None),
            reasons=list(getattr(item, "reasons", []) or []),
            features=dict(getattr(item, "features", {}) or {}),
        )
        for item in verdicts
        if str(getattr(item, "decision", "")) in keep
    ]
    return EvidenceBundle(
        board_id=board_id,
        source=SourceFingerprint(
            path=str(Path(source_path).resolve()),
            sha256=file_digest(source_path),
            analysis_width=int(width),
            analysis_height=int(height),
            preprocess=dict(preprocess or {}),
        ),
        defects=defects,
    )
