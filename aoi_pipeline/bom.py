"""Bill of materials: what the board is *supposed* to carry.

CAD and BOM answer different questions, and the difference is the whole point
of this module.

A CAD/centroid export says *where the lands are*. It is routinely partial --
thermal pads, shields and mechanical lands get left out -- so
:mod:`aoi_pipeline.solder.cad_fusion` treats a detection with no CAD entry as
an observation worth noting, not a fault.

A BOM says *what parts exist*. It is the assembly contract: every part that
belongs on the board is on it, and nothing else is. Under that contract a
component found where the BOM lists nothing is **a defect** -- an extra part,
a part placed at the wrong site, or foreign material -- not a gap in the file.

So this module carries a completeness flag and refuses to guess it. A BOM
loaded from a file is complete by definition; anything else has to say so.

Two file shapes are accepted, because both get called "the BOM":

``one row per part instance``
    Designator, coordinates, rotation, size. Really a placement/centroid file.
    This is the shape that can be reconciled by *position*.

``one row per part type``
    Designator holds a list -- ``"R1, R2, R5"`` -- plus Quantity. This is the
    purchasing shape and carries no coordinates. It can still be reconciled by
    *designator and count*, which catches a missing or duplicated part even
    with no geometry at all.

Rows are expanded either way, so downstream code sees one entry per part.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .exceptions import AOIPipelineError
from .models import BoundingBox, Detection
from .solder.cad import (
    BoardCad,
    CadComponent,
    _map_columns,
    _normalize_side,
    _strip_comments,
    _to_float,
    _unit_scale,
    classes_agree,
    designator_to_class,
    is_informative_label,
)

__all__ = [
    "BillOfMaterials",
    "BomEntry",
    "BomError",
    "BomFinding",
    "BomReconciliation",
    "load_bom",
    "reconcile_bom",
]

#: Designator lists are written with commas, semicolons or spaces, and often
#: mix them: ``"C1, C2;C3 C4"``. Ranges like ``R1-R4`` are deliberately NOT
#: expanded -- ``R1-R4`` could mean four parts or a single part named that way,
#: and inventing three components that may not exist is worse than reporting
#: the one designator that was actually written.
_DESIGNATOR_SPLIT = re.compile(r"[,;]|\s{2,}")

_QUANTITY_ALIASES = ("quantity", "qty", "count", "amount")


class BomError(AOIPipelineError):
    """Raised when a BOM cannot be read or contradicts itself."""


@dataclass(frozen=True, slots=True)
class BomEntry:
    """One part the board is supposed to carry."""

    designator: str
    value: str | None = None
    footprint: str | None = None
    part_class: str | None = None
    # Position and size in mm. ``None`` means the BOM did not carry it, which
    # is normal for the purchasing shape -- do not substitute zero.
    x: float | None = None
    y: float | None = None
    rotation: float = 0.0
    width: float | None = None
    height: float | None = None
    side: str = "top"

    @property
    def has_position(self) -> bool:
        return self.x is not None and self.y is not None

    @property
    def has_size(self) -> bool:
        return bool(self.width and self.height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "designator": self.designator,
            "value": self.value,
            "footprint": self.footprint,
            "part_class": self.part_class,
            "x_mm": self.x,
            "y_mm": self.y,
            "rotation_deg": self.rotation,
            "width_mm": self.width,
            "height_mm": self.height,
            "side": self.side,
        }


@dataclass(slots=True)
class BillOfMaterials:
    """Every part on the board, one entry per instance."""

    entries: list[BomEntry] = field(default_factory=list)
    source: str | None = None
    units: str = "mm"
    #: A BOM read from a file lists the whole assembly. Anything assembled by
    #: hand from a partial source must set this to False, and reconciliation
    #: will then stop calling unlisted detections defects.
    complete: bool = True
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def has_positions(self) -> bool:
        return any(entry.has_position for entry in self.entries)

    def by_designator(self) -> dict[str, BomEntry]:
        return {entry.designator.upper(): entry for entry in self.entries}

    def side(self, side: str | None) -> BillOfMaterials:
        if side is None:
            return self
        wanted = _normalize_side(side)
        return BillOfMaterials(
            entries=[entry for entry in self.entries if entry.side == wanted],
            source=self.source,
            units=self.units,
            complete=self.complete,
            warnings=list(self.warnings),
        )

    def to_board_cad(self) -> BoardCad:
        """Reuse the CAD registration machinery on the positioned entries.

        Only entries that actually carry coordinates come across. Registration
        solves for a transform from point correspondences, and a pile of parts
        all sitting at (0, 0) would drag that solution somewhere meaningless.
        """

        components = [
            CadComponent(
                designator=entry.designator,
                x=float(entry.x),          # type: ignore[arg-type]
                y=float(entry.y),          # type: ignore[arg-type]
                rotation=entry.rotation,
                side=entry.side,
                footprint=entry.footprint,
                value=entry.value,
                part_class=entry.part_class,
                width=float(entry.width or 0.0),
                height=float(entry.height or 0.0),
            )
            for entry in self.entries
            if entry.has_position
        ]
        return BoardCad(
            components=components,
            source=self.source or "bom",
            source_format="bom",
            units=self.units,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "units": self.units,
            "complete": self.complete,
            "count": len(self.entries),
            "with_position": sum(1 for entry in self.entries if entry.has_position),
            "with_size": sum(1 for entry in self.entries if entry.has_size),
            "warnings": list(self.warnings),
            "entries": [entry.to_dict() for entry in self.entries],
        }


def _split_designators(raw: str) -> list[str]:
    """``"C1, C2;C3"`` -> ``["C1", "C2", "C3"]``.

    Single spaces are not separators. ``"CONN 1"`` is one designator with a
    space in it, and splitting it would invent a part called ``1``.
    """

    parts = [piece.strip() for piece in _DESIGNATOR_SPLIT.split(raw)]
    return [piece for piece in parts if piece]


def _quantity_column(fieldnames: Iterable[str]) -> str | None:
    for name in fieldnames:
        if name and re.sub(r"[^a-z0-9]", "", name.strip().lower()) in _QUANTITY_ALIASES:
            return name
    return None


def load_bom(
    path: str | Path,
    units: str = "mm",
    *,
    complete: bool = True,
) -> BillOfMaterials:
    """Read a BOM from CSV, expanding multi-designator rows into one entry each.

    ``complete`` stays True unless the caller knows the file only covers part
    of the board. It controls whether an unlisted detection is reported as a
    defect, so it is the one thing here worth being deliberate about.
    """

    source = Path(path)
    if not source.is_file():
        raise BomError(f"Không tìm thấy file BOM: {source}")

    scale = {"mm": 1.0, "cm": 10.0, "m": 1000.0,
             "mil": 0.0254, "in": 25.4, "inch": 25.4}.get(units.lower(), 1.0)

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(_strip_comments(handle))
        if reader.fieldnames is None:
            raise BomError(f"{source.name}: không có dòng tiêu đề")
        columns = _map_columns(reader.fieldnames)
        if "designator" not in columns:
            raise BomError(
                f"{source.name}: thiếu cột designator "
                f"(thấy {list(reader.fieldnames)})"
            )
        quantity_column = _quantity_column(reader.fieldnames)
        rows = list(reader)

    entries: list[BomEntry] = []
    warnings: list[str] = []
    seen: dict[str, int] = {}

    for line, row in enumerate(rows, start=2):
        raw = str(row.get(columns["designator"], "")).strip()
        if not raw:
            continue
        designators = _split_designators(raw)

        if quantity_column is not None:
            declared = _to_float(row.get(quantity_column), default=float(len(designators)))
            if declared and int(declared) != len(designators):
                # Worth saying out loud: a BOM whose Quantity disagrees with its
                # own designator list is wrong on paper, before any board is
                # inspected. Trust the designators -- they name real sites.
                warnings.append(
                    f"dòng {line}: Quantity ghi {int(declared)} nhưng liệt kê "
                    f"{len(designators)} designator ({raw})"
                )

        row_scale = scale
        x_raw = row.get(columns["x"]) if "x" in columns else None
        if x_raw is not None:
            row_scale = _unit_scale(x_raw, scale)

        def _optional(key: str) -> float | None:
            if key not in columns:
                return None
            text = str(row.get(columns[key], "")).strip()
            if not text:
                return None
            return _to_float(text) * row_scale

        x, y = _optional("x"), _optional("y")
        width, height = _optional("width"), _optional("height")
        value = str(row.get(columns.get("value", ""), "")).strip() or None
        footprint = str(row.get(columns.get("footprint", ""), "")).strip() or None
        rotation = _to_float(row.get(columns.get("rotation", "")))
        side = _normalize_side(row.get(columns.get("side", "")))

        for designator in designators:
            key = designator.upper()
            if key in seen:
                warnings.append(
                    f"designator {designator} xuất hiện lại ở dòng {line} "
                    f"(đã có ở dòng {seen[key]})"
                )
                continue
            seen[key] = line
            entries.append(
                BomEntry(
                    designator=designator,
                    value=value,
                    footprint=footprint,
                    part_class=designator_to_class(designator),
                    x=x, y=y, rotation=rotation,
                    width=width, height=height,
                    side=side,
                )
            )

    if not entries:
        raise BomError(f"{source.name}: không đọc được linh kiện nào")

    return BillOfMaterials(
        entries=entries,
        source=str(source),
        units="mm",
        complete=complete,
        warnings=warnings,
    )


@dataclass(frozen=True, slots=True)
class BomFinding:
    """One disagreement between the BOM and what the camera saw."""

    kind: str            # missing | unexpected | class_mismatch | shifted
    severity: str        # error | warning | info
    message: str
    designator: str | None = None
    detection_id: str | None = None
    expected_class: str | None = None
    observed_class: str | None = None
    bbox: BoundingBox | None = None
    offset_px: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "message": self.message,
            "designator": self.designator,
            "detection_id": self.detection_id,
            "expected_class": self.expected_class,
            "observed_class": self.observed_class,
            "bbox": self.bbox.to_dict() if self.bbox is not None else None,
            "offset_px": self.offset_px,
        }


@dataclass(slots=True)
class BomReconciliation:
    """What the board carries versus what the BOM says it should."""

    findings: list[BomFinding] = field(default_factory=list)
    matched: list[tuple[str, str]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[BomFinding]:
        return [item for item in self.findings if item.severity == "error"]

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "stats": self.stats,
            "matched": [{"designator": d, "detection_id": i} for d, i in self.matched],
            "findings": [item.to_dict() for item in self.findings],
        }


def reconcile_bom(
    bom: BillOfMaterials,
    detections: Sequence[Detection],
    project: Any | None = None,
    *,
    match_tolerance_px: float = 60.0,
    report_class_mismatch: bool = True,
) -> BomReconciliation:
    """Compare detections against the BOM.

    ``project`` maps a BOM entry's mm coordinates into analysis-image pixels --
    normally ``CadRegistration.project``. Without it, or when the BOM carries no
    coordinates, matching falls back to counting: how many parts of each class
    the BOM expects against how many were found. That is weaker, but it still
    catches a missing part and an extra one, which is most of what a BOM is for.

    An unlisted detection is an **error** when ``bom.complete``. That is the
    case the operator asked for: a component sitting where the BOM lists
    nothing is an extra part, a misplaced part or foreign material.
    """

    findings: list[BomFinding] = []
    matched: list[tuple[str, str]] = []
    unlisted_severity = "error" if bom.complete else "info"

    for warning in bom.warnings:
        findings.append(BomFinding(
            kind="bom_inconsistent", severity="warning",
            message=f"BOM tự mâu thuẫn: {warning}",
        ))

    positioned = [entry for entry in bom.entries if entry.has_position]
    can_match_by_position = bool(positioned) and project is not None

    if can_match_by_position:
        matched, unmatched_entries, unmatched_detections = _match_by_position(
            positioned, detections, project, match_tolerance_px
        )
        entry_by_designator = {entry.designator: entry for entry in positioned}
        detection_by_id = {d.detection_id: d for d in detections}

        for designator, detection_id in matched:
            entry = entry_by_designator[designator]
            detection = detection_by_id[detection_id]
            if not report_class_mismatch:
                continue
            if not (is_informative_label(entry.part_class)
                    and is_informative_label(detection.label)):
                continue
            if not classes_agree(entry.part_class, detection.label):
                findings.append(BomFinding(
                    kind="class_mismatch", severity="warning",
                    message=(
                        f"{designator}: BOM ghi {entry.part_class}, "
                        f"camera thấy {detection.label}"
                    ),
                    designator=designator, detection_id=detection_id,
                    expected_class=entry.part_class, observed_class=detection.label,
                    bbox=detection.bbox,
                ))

        for entry in unmatched_entries:
            findings.append(BomFinding(
                kind="missing", severity="error",
                message=(
                    f"{entry.designator} có trong BOM nhưng không thấy trên board "
                    f"(vị trí BOM {entry.x:.2f}, {entry.y:.2f} mm)"
                ),
                designator=entry.designator, expected_class=entry.part_class,
            ))

        for detection in unmatched_detections:
            findings.append(BomFinding(
                kind="unexpected", severity=unlisted_severity,
                message=(
                    f"Thấy {detection.label} tại "
                    f"({detection.bbox.center[0]:.0f}, {detection.bbox.center[1]:.0f}) px "
                    "nhưng BOM không có linh kiện nào ở toạ độ này"
                ),
                detection_id=detection.detection_id,
                observed_class=detection.label, bbox=detection.bbox,
            ))
        method = "position"
    else:
        findings.extend(_reconcile_by_count(bom, detections, unlisted_severity))
        unmatched_entries, unmatched_detections = [], []
        method = "count"

    stats = {
        "method": method,
        "bom_entries": len(bom.entries),
        "bom_complete": bom.complete,
        "detections": len(detections),
        "matched": len(matched),
        "missing": sum(1 for item in findings if item.kind == "missing"),
        "unexpected": sum(1 for item in findings if item.kind == "unexpected"),
        "class_mismatch": sum(1 for item in findings if item.kind == "class_mismatch"),
    }
    return BomReconciliation(findings=findings, matched=matched, stats=stats)


def _match_by_position(
    entries: Sequence[BomEntry],
    detections: Sequence[Detection],
    project: Any,
    tolerance_px: float,
) -> tuple[list[tuple[str, str]], list[BomEntry], list[Detection]]:
    """Nearest-neighbour pairing under a distance cap, closest pairs first.

    Greedy on sorted distance rather than per-entry: taking each entry's nearest
    free detection in list order lets an early entry claim a detection that sits
    much closer to a later one, and every pair after it shifts by one.
    """

    if not entries or not detections:
        return [], list(entries), list(detections)

    projected = project(np.array([[entry.x, entry.y] for entry in entries],
                                 dtype=np.float64))
    centres = np.array([detection.bbox.center for detection in detections],
                       dtype=np.float64)

    candidates: list[tuple[float, int, int]] = []
    for entry_index, point in enumerate(projected):
        distances = np.hypot(centres[:, 0] - point[0], centres[:, 1] - point[1])
        for detection_index in np.argsort(distances)[:8]:
            distance = float(distances[detection_index])
            if distance <= tolerance_px:
                candidates.append((distance, entry_index, int(detection_index)))
    candidates.sort()

    used_entries: set[int] = set()
    used_detections: set[int] = set()
    matched: list[tuple[str, str]] = []
    for _, entry_index, detection_index in candidates:
        if entry_index in used_entries or detection_index in used_detections:
            continue
        used_entries.add(entry_index)
        used_detections.add(detection_index)
        matched.append((entries[entry_index].designator,
                        detections[detection_index].detection_id))

    return (
        matched,
        [entry for index, entry in enumerate(entries) if index not in used_entries],
        [d for index, d in enumerate(detections) if index not in used_detections],
    )


def _reconcile_by_count(
    bom: BillOfMaterials,
    detections: Sequence[Detection],
    unlisted_severity: str,
) -> list[BomFinding]:
    """No coordinates: compare how many parts of each class were expected.

    Weaker than position matching and honest about it -- it cannot say *which*
    part is missing, only that one is. Still worth doing: a board short one
    resistor shows up here with no geometry at all.
    """

    findings: list[BomFinding] = []
    expected: dict[str, int] = {}
    for entry in bom.entries:
        if is_informative_label(entry.part_class):
            expected[str(entry.part_class)] = expected.get(str(entry.part_class), 0) + 1

    observed: dict[str, int] = {}
    for detection in detections:
        if is_informative_label(detection.label):
            observed[detection.label] = observed.get(detection.label, 0) + 1

    for part_class in sorted(set(expected) | set(observed)):
        want, got = expected.get(part_class, 0), observed.get(part_class, 0)
        if want == got:
            continue
        if got < want:
            findings.append(BomFinding(
                kind="missing", severity="error",
                message=(
                    f"BOM cần {want} linh kiện loại {part_class}, chỉ thấy {got}. "
                    "BOM không có toạ độ nên không chỉ được thiếu con nào."
                ),
                expected_class=part_class,
            ))
        else:
            findings.append(BomFinding(
                kind="unexpected", severity=unlisted_severity,
                message=(
                    f"Thấy {got} linh kiện loại {part_class} nhưng BOM chỉ có "
                    f"{want}. BOM không có toạ độ nên không chỉ được con thừa."
                ),
                observed_class=part_class,
            ))
    return findings
