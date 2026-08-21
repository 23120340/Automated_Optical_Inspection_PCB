"""CAD board data: a normalized model, format loaders and image registration.

This is the half of step 5.5 that knows where the lands *really* are. The
derived geometry in :mod:`aoi_pipeline.inspection.solder` infers ROI positions from a
detector box plus a per-class terminal topology; CAD replaces that inference
with measured coordinates. Neither one supersedes the other -- see
:mod:`aoi_pipeline.inspection.fusion`, which merges them and uses each to check the other.

Nothing here is required. With no CAD file the pipeline behaves exactly as it
did before, and every function in this module is simply never called.

Dropping a board in later means two things:

1. Put the file somewhere and load it with :func:`load_cad`. The format is
   sniffed from the extension and content; add new formats by registering a
   loader in :data:`CAD_LOADERS` rather than editing the caller.
2. Register it to the camera once per SKU with :func:`register_cad`, then save
   the result with :meth:`CadRegistration.to_dict` so later runs reuse it.

Coordinates: CAD lives in millimetres on the board; the registration maps them
into the pixels of the preprocessed/aligned analysis image, the same space the
rest of the pipeline uses.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .exceptions import AOIPipelineError

__all__ = [
    "BoardCad",
    "UNINFORMATIVE_LABELS",
    "CAD_LOADERS",
    "CadComponent",
    "CadError",
    "CadPad",
    "CadRegistration",
    "designator_to_class",
    "load_cad",
    "register_cad",
    "register_from_fiducials",
]


class CadError(AOIPipelineError):
    """Raised when a CAD file cannot be read or registered."""


# --------------------------------------------------------------------------- #
# Normalized model
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class CadPad:
    """One land in board millimetres."""

    designator: str
    pin: str
    x: float
    y: float
    width: float = 0.0
    height: float = 0.0
    rotation: float = 0.0
    shape: str = "unknown"
    net: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "designator": self.designator,
            "pin": self.pin,
            "x_mm": self.x,
            "y_mm": self.y,
            "width_mm": self.width,
            "height_mm": self.height,
            "rotation_deg": self.rotation,
            "shape": self.shape,
            "net": self.net,
        }


@dataclass(slots=True)
class CadComponent:
    """One placed part, with its lands when the source format carries them."""

    designator: str
    x: float
    y: float
    rotation: float = 0.0
    side: str = "top"
    footprint: str | None = None
    value: str | None = None
    part_class: str | None = None
    pads: list[CadPad] = field(default_factory=list)

    @property
    def has_pads(self) -> bool:
        return bool(self.pads)

    def pad_span_mm(self) -> float:
        """Largest distance between two lands, i.e. the part's true length."""

        if len(self.pads) < 2:
            return 0.0
        points = np.array([[pad.x, pad.y] for pad in self.pads], dtype=np.float64)
        spread = points.max(axis=0) - points.min(axis=0)
        return float(math.hypot(spread[0], spread[1]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "designator": self.designator,
            "x_mm": self.x,
            "y_mm": self.y,
            "rotation_deg": self.rotation,
            "side": self.side,
            "footprint": self.footprint,
            "value": self.value,
            "part_class": self.part_class,
            "pad_count": len(self.pads),
            "pads": [pad.to_dict() for pad in self.pads],
        }


@dataclass(slots=True)
class BoardCad:
    """Every part on one board side, in millimetres."""

    components: list[CadComponent] = field(default_factory=list)
    source: str = ""
    source_format: str = ""
    side: str = "top"
    units: str = "mm"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pad_count(self) -> int:
        return sum(len(component.pads) for component in self.components)

    def for_side(self, side: str) -> BoardCad:
        """Keep only the parts on ``side``; boards are inspected one side at a time."""

        wanted = side.strip().lower()
        return BoardCad(
            components=[
                component
                for component in self.components
                if component.side.lower() == wanted
            ],
            source=self.source,
            source_format=self.source_format,
            side=wanted,
            units=self.units,
            metadata=dict(self.metadata),
        )

    def extent(self) -> tuple[float, float, float, float] | None:
        """Bounding box of everything placed, as ``(x1, y1, x2, y2)`` in mm."""

        points: list[tuple[float, float]] = []
        for component in self.components:
            points.append((component.x, component.y))
            points.extend((pad.x, pad.y) for pad in component.pads)
        if not points:
            return None
        array = np.asarray(points, dtype=np.float64)
        return (
            float(array[:, 0].min()),
            float(array[:, 1].min()),
            float(array[:, 0].max()),
            float(array[:, 1].max()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_format": self.source_format,
            "side": self.side,
            "units": self.units,
            "component_count": len(self.components),
            "pad_count": self.pad_count,
            "components_with_pads": sum(
                1 for component in self.components if component.has_pads
            ),
            "metadata": self.metadata,
        }


# --------------------------------------------------------------------------- #
# Designator to detector class
# --------------------------------------------------------------------------- #

# Reference-designator prefixes follow IEEE 315 closely enough to be a reliable
# class prior, and the value is checked against the detector's own class rather
# than trusted outright.
DESIGNATOR_CLASSES: dict[str, str] = {
    "R": "resistor",
    "RN": "resistor",
    "RV": "potentiometer",
    "VR": "potentiometer",
    "POT": "potentiometer",
    "C": "capacitor",
    "CN": "connector",
    "L": "inductor",
    "FB": "inductor",
    "D": "diode",
    "DS": "led",
    "LED": "led",
    "U": "ic",
    "IC": "ic",
    "Q": "transistor",
    "F": "fuse",
    "J": "connector",
    "P": "connector",
    "JP": "connector",
    "CON": "connector",
    "SW": "switch",
    "S": "switch",
    "K": "relay",
    "RY": "relay",
    "Y": "clock",
    "X": "clock",
    "XTAL": "clock",
    "BT": "battery",
    "BAT": "battery",
    "LS": "buzzer",
    "SP": "buzzer",
    "BZ": "buzzer",
    "T": "transformer",
    "TR": "transformer",
    "TP": "pads",
    "HS": "heatsink",
    "M": "transducer",
}

#: Detector labels that name no class. The OpenCV proposal mode emits
#: ``component_candidate`` for everything, and treating that as a *contradiction*
#: of the CAD class would both bury the operator in false mismatches and remove
#: the only signal that separates a board from its own mirror image.
UNINFORMATIVE_LABELS: frozenset[str] = frozenset(
    {"", "component", "component_candidate", "candidate", "object", "unknown"}
)


def is_informative_label(label: str | None) -> bool:
    return str(label or "").strip().lower() not in UNINFORMATIVE_LABELS


def classes_agree(source_label: str | None, target_label: str | None) -> bool:
    """True unless both labels are informative and they disagree."""

    if not is_informative_label(source_label) or not is_informative_label(target_label):
        return True
    return str(source_label).strip().lower() == str(target_label).strip().lower()


_DESIGNATOR_PREFIX = re.compile(r"^([A-Za-z]+)")


def designator_to_class(designator: str) -> str | None:
    """Guess the detector class from a reference designator prefix."""

    match = _DESIGNATOR_PREFIX.match(str(designator).strip())
    if not match:
        return None
    prefix = match.group(1).upper()
    # Longest prefix wins so "LED3" is not read as "L" + "ED3".
    for length in range(len(prefix), 0, -1):
        found = DESIGNATOR_CLASSES.get(prefix[:length])
        if found is not None:
            return found
    return None


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

_UNIT_SCALES = {
    "mm": 1.0,
    "millimeter": 1.0,
    "millimetre": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "in": 25.4,
    "inch": 25.4,
    "mil": 0.0254,
    "thou": 0.0254,
    "um": 0.001,
}

# Header aliases, lower-cased and stripped of separators.
_ALIASES: dict[str, tuple[str, ...]] = {
    "designator": ("designator", "refdes", "ref", "reference", "part", "component", "name", "id"),
    "pin": ("pin", "pinnumber", "pinnum", "padnumber", "pad", "padname", "terminal"),
    "x": ("xmm", "x", "midx", "centerx", "centrex", "posx", "positionx", "refx", "xloc"),
    "y": ("ymm", "y", "midy", "centery", "centrey", "posy", "positiony", "refy", "yloc"),
    "width": ("widthmm", "width", "padwidth", "sizex", "w"),
    "height": ("heightmm", "height", "padheight", "sizey", "h"),
    "rotation": ("rotationdeg", "rotation", "rot", "angle", "orientation", "theta"),
    "shape": ("shape", "padshape", "type", "padtype"),
    "net": ("net", "netname", "signal"),
    "side": ("side", "layer", "tb", "topbottom", "mountingside"),
    "footprint": ("footprint", "package", "pattern", "fp"),
    "value": ("value", "comment", "val", "partnumber", "description"),
}


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).strip().lower())


def _map_columns(fieldnames: Iterable[str]) -> dict[str, str]:
    """Map canonical field names to the actual column names in a CSV."""

    normalized = {_normalize_key(name): name for name in fieldnames if name}
    mapping: dict[str, str] = {}
    for canonical, candidates in _ALIASES.items():
        for candidate in candidates:
            if candidate in normalized:
                mapping[canonical] = normalized[candidate]
                break
    return mapping


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip().replace(",", ".")
    if not text:
        return default
    match = re.match(r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if match is None:
        return default
    return float(match.group(0))


def _unit_scale(value: Any, fallback: float) -> float:
    """Read a trailing unit off a value like ``12.5mm`` or a units column."""

    text = str(value or "").strip().lower()
    match = re.search(r"([a-z]+)$", text)
    if match:
        scale = _UNIT_SCALES.get(match.group(1))
        if scale is not None:
            return scale
    return fallback


def _normalize_side(value: Any, default: str = "top") -> str:
    text = str(value or "").strip().lower()
    if text in {"top", "t", "1", "front", "topside", "top side", "smd_top"}:
        return "top"
    if text in {"bottom", "b", "2", "back", "bot", "bottomside", "smd_bottom"}:
        return "bottom"
    return default


def load_pads_csv(path: Path, units: str = "mm") -> BoardCad:
    """Read the canonical pad table.

    Required columns: ``designator``, ``pin``, ``x_mm``, ``y_mm``. Optional:
    ``width_mm``, ``height_mm``, ``rotation_deg``, ``shape``, ``net``, ``side``,
    ``footprint``, ``value``. Header names are matched loosely, so an Altium or
    KiCad pad report usually loads without editing.

    This is the format to convert to when nothing else fits: every EDA tool can
    produce a pad list, and a converter is a few lines.
    """

    scale = _UNIT_SCALES.get(units.lower(), 1.0)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(_strip_comments(handle))
        if reader.fieldnames is None:
            raise CadError(f"{path.name}: no header row")
        columns = _map_columns(reader.fieldnames)
        for required in ("designator", "x", "y"):
            if required not in columns:
                raise CadError(
                    f"{path.name}: missing a '{required}' column "
                    f"(saw {list(reader.fieldnames)})"
                )
        components: dict[str, CadComponent] = {}
        for index, row in enumerate(reader):
            designator = str(row.get(columns["designator"], "")).strip()
            if not designator:
                continue
            row_scale = _unit_scale(row.get(columns["x"]), scale)
            pad = CadPad(
                designator=designator,
                pin=str(row.get(columns.get("pin", ""), "") or index + 1).strip(),
                x=_to_float(row.get(columns["x"])) * row_scale,
                y=_to_float(row.get(columns["y"])) * row_scale,
                width=_to_float(row.get(columns.get("width", ""))) * row_scale,
                height=_to_float(row.get(columns.get("height", ""))) * row_scale,
                rotation=_to_float(row.get(columns.get("rotation", ""))),
                shape=str(row.get(columns.get("shape", ""), "") or "unknown").strip().lower(),
                net=(str(row.get(columns.get("net", ""), "")).strip() or None),
            )
            component = components.get(designator)
            if component is None:
                component = CadComponent(
                    designator=designator,
                    x=0.0,
                    y=0.0,
                    side=_normalize_side(row.get(columns.get("side", ""))),
                    footprint=(str(row.get(columns.get("footprint", ""), "")).strip() or None),
                    value=(str(row.get(columns.get("value", ""), "")).strip() or None),
                    part_class=designator_to_class(designator),
                )
                components[designator] = component
            component.pads.append(pad)

    for component in components.values():
        # The placement is the centroid of its own lands, which is what the
        # detector's box centre corresponds to.
        component.x = float(np.mean([pad.x for pad in component.pads]))
        component.y = float(np.mean([pad.y for pad in component.pads]))
        component.rotation = _rotation_from_pads(component.pads)
    return BoardCad(
        components=sorted(components.values(), key=lambda item: item.designator),
        source=str(path),
        source_format="pads_csv",
        units="mm",
    )


def load_placement_csv(path: Path, units: str = "mm") -> BoardCad:
    """Read a pick-and-place / centroid file: one row per placed part, no lands.

    This is the format most likely to be available from a contract manufacturer.
    It fixes each part's true centre, rotation and side, which is already enough
    to place ROIs far better than a detector box alone -- the terminal topology
    then still comes from the derived geometry. Fusion handles that mix.
    """

    scale = _UNIT_SCALES.get(units.lower(), 1.0)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(_strip_comments(handle))
        if reader.fieldnames is None:
            raise CadError(f"{path.name}: no header row")
        columns = _map_columns(reader.fieldnames)
        for required in ("designator", "x", "y"):
            if required not in columns:
                raise CadError(
                    f"{path.name}: missing a '{required}' column "
                    f"(saw {list(reader.fieldnames)})"
                )
        components: list[CadComponent] = []
        for row in reader:
            designator = str(row.get(columns["designator"], "")).strip()
            if not designator:
                continue
            row_scale = _unit_scale(row.get(columns["x"]), scale)
            components.append(
                CadComponent(
                    designator=designator,
                    x=_to_float(row.get(columns["x"])) * row_scale,
                    y=_to_float(row.get(columns["y"])) * row_scale,
                    rotation=_to_float(row.get(columns.get("rotation", ""))),
                    side=_normalize_side(row.get(columns.get("side", ""))),
                    footprint=(str(row.get(columns.get("footprint", ""), "")).strip() or None),
                    value=(str(row.get(columns.get("value", ""), "")).strip() or None),
                    part_class=designator_to_class(designator),
                )
            )
    return BoardCad(
        components=components,
        source=str(path),
        source_format="placement_csv",
        units="mm",
    )


# IPC-D-356A: fixed-column netlist test data. Record 317/327/367 lines carry a
# net name, a designator/pin and an X/Y location, which makes it one of the few
# widely-issued deliverables with real land coordinates.
_IPC_UNIT_SCALES = {
    "0": 0.00254,   # 0.0001 inch
    "1": 0.001,     # 0.001 mm
    "2": 0.000254,  # 0.00001 inch
    "3": 0.0001,    # 0.0001 mm
}


def load_ipc356(path: Path, units: str = "auto") -> BoardCad:
    """Read pad coordinates out of an IPC-D-356A netlist.

    Parsed by anchoring on the coordinate block rather than on fixed columns.
    The standard specifies columns, but files in the wild drift by a character
    or two depending on the tool that wrote them, and a column-exact reader
    silently returns nothing on those instead of failing loudly.
    """

    scale = _IPC_UNIT_SCALES["0"]
    components: dict[str, CadComponent] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if line[:1] == "P":
            match = re.search(r"UNITS\s+CUST\s*(\d)", line, re.IGNORECASE)
            if match:
                scale = _IPC_UNIT_SCALES.get(match.group(1), scale)
            continue
        if not re.match(r"^3(17|27|67)", line):
            continue
        location = re.search(r"X([-+]\d+)Y([-+]\d+)", line)
        if location is None:
            continue
        reference, pin = _ipc_reference(line, location.start())
        if not reference:
            continue

        x = int(location.group(1)) * scale
        y = int(location.group(2)) * scale
        size = re.match(r"X(\d+)Y(\d+)", line[location.end() :])
        width = int(size.group(1)) * scale if size else 0.0
        height = int(size.group(2)) * scale if size else 0.0
        component = components.get(reference)
        if component is None:
            component = CadComponent(
                designator=reference,
                x=0.0,
                y=0.0,
                side=_ipc_side(line[:location.start()]),
                part_class=designator_to_class(reference),
            )
            components[reference] = component
        component.pads.append(
            CadPad(
                designator=reference,
                pin=pin or str(len(component.pads) + 1),
                x=x,
                y=y,
                width=width,
                height=height,
                net=(line[3:17].strip() or None),
            )
        )

    if not components:
        raise CadError(f"{path.name}: no IPC-D-356A 3x7 pad records found")
    for component in components.values():
        component.x = float(np.mean([pad.x for pad in component.pads]))
        component.y = float(np.mean([pad.y for pad in component.pads]))
        component.rotation = _rotation_from_pads(component.pads)
    return BoardCad(
        components=sorted(components.values(), key=lambda item: item.designator),
        source=str(path),
        source_format="ipc356",
        units="mm",
    )


_IPC_REFERENCE = re.compile(r"([A-Za-z][A-Za-z0-9_.]*)\s*-\s*([A-Za-z0-9]+)")


def _ipc_reference(line: str, coordinate_start: int) -> tuple[str, str]:
    """Pull ``REFDES-PIN`` out of the field between the net name and the coords."""

    for begin in (17, 3):
        segment = line[begin:coordinate_start]
        match = _IPC_REFERENCE.search(segment)
        if match:
            return match.group(1), match.group(2)
    return ("", "")


def _ipc_side(prefix: str) -> str:
    """Board side from the access code: 00 (both) and 01 (top) are the top face."""

    match = re.search(r"A(\d{2})", prefix)
    if match is None:
        return "top"
    return "top" if match.group(1) in {"00", "01"} else "bottom"


def load_cad_json(path: Path, units: str = "mm") -> BoardCad:
    """Read a board previously saved by :func:`save_cad_json`."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    components: list[CadComponent] = []
    for entry in payload.get("components", []):
        components.append(
            CadComponent(
                designator=str(entry.get("designator", "")),
                x=float(entry.get("x_mm", 0.0)),
                y=float(entry.get("y_mm", 0.0)),
                rotation=float(entry.get("rotation_deg", 0.0)),
                side=_normalize_side(entry.get("side")),
                footprint=entry.get("footprint"),
                value=entry.get("value"),
                part_class=entry.get("part_class")
                or designator_to_class(str(entry.get("designator", ""))),
                pads=[
                    CadPad(
                        designator=str(entry.get("designator", "")),
                        pin=str(pad.get("pin", index + 1)),
                        x=float(pad.get("x_mm", 0.0)),
                        y=float(pad.get("y_mm", 0.0)),
                        width=float(pad.get("width_mm", 0.0)),
                        height=float(pad.get("height_mm", 0.0)),
                        rotation=float(pad.get("rotation_deg", 0.0)),
                        shape=str(pad.get("shape", "unknown")),
                        net=pad.get("net"),
                    )
                    for index, pad in enumerate(entry.get("pads", []))
                ],
            )
        )
    return BoardCad(
        components=components,
        source=str(path),
        source_format="cad_json",
        units="mm",
        metadata=dict(payload.get("metadata", {})),
    )


def save_cad_json(board: BoardCad, path: str | Path) -> Path:
    """Write a normalized board so a slow or lossy import runs only once."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "metadata": {**board.metadata, "source": board.source, "format": board.source_format},
        "components": [component.to_dict() for component in board.components],
    }
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination


#: Format name to loader. Register a new format here and both the pipeline and
#: the UI pick it up; no caller needs to change.
CAD_LOADERS: dict[str, Callable[[Path, str], BoardCad]] = {
    "pads_csv": load_pads_csv,
    "placement_csv": load_placement_csv,
    "ipc356": load_ipc356,
    "cad_json": load_cad_json,
}


def load_cad(
    path: str | Path,
    fmt: str = "auto",
    units: str = "mm",
    side: str | None = None,
) -> BoardCad:
    """Load a CAD file, sniffing the format unless ``fmt`` names one."""

    source = Path(path).expanduser()
    if not source.is_file():
        raise CadError(f"CAD file not found: {source}")
    chosen = fmt if fmt != "auto" else sniff_cad_format(source)
    loader = CAD_LOADERS.get(chosen)
    if loader is None:
        raise CadError(
            f"Unknown CAD format '{chosen}'. Known: {sorted(CAD_LOADERS)}. "
            "Convert to the pads_csv schema or register a loader in CAD_LOADERS."
        )
    try:
        board = loader(source, units)
    except CadError:
        raise
    except (OSError, ValueError, KeyError, IndexError) as exc:
        raise CadError(f"{source.name}: {type(exc).__name__}: {exc}") from exc
    if not board.components:
        raise CadError(f"{source.name}: parsed as {chosen} but held no components")
    return board.for_side(side) if side else board


def sniff_cad_format(path: Path) -> str:
    """Pick a loader from the extension plus a peek at the first lines."""

    suffix = path.suffix.lower()
    if suffix == ".json":
        return "cad_json"
    head = ""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            head = "".join(next(handle, "") for _ in range(40))
    except OSError as exc:
        raise CadError(f"{path.name}: {exc}") from exc
    if re.search(r"^P\s+(JOB|UNITS)|^3(17|27|67)", head, re.MULTILINE):
        return "ipc356"
    first_line = head.splitlines()[0] if head.splitlines() else ""
    columns = _map_columns(next(csv.reader([first_line]), []))
    if "pin" in columns:
        return "pads_csv"
    if "designator" in columns and "x" in columns:
        return "placement_csv"
    raise CadError(
        f"{path.name}: could not identify the CAD format. Pass fmt= explicitly "
        f"(one of {sorted(CAD_LOADERS)})."
    )


def _strip_comments(handle: Iterable[str]) -> Iterable[str]:
    """Drop ``#`` comment lines that CM-issued centroid files often carry."""

    for line in handle:
        if line.lstrip().startswith("#"):
            continue
        yield line


def _rotation_from_pads(pads: Sequence[CadPad]) -> float:
    """Part rotation from the axis its own lands lie on, in degrees."""

    if len(pads) < 2:
        return 0.0
    points = np.array([[pad.x, pad.y] for pad in pads], dtype=np.float64)
    centered = points - points.mean(axis=0)
    if not np.any(np.abs(centered) > 1e-9):
        return 0.0
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    axis = vectors[0]
    return float(math.degrees(math.atan2(axis[1], axis[0])))


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class CadRegistration:
    """Maps board millimetres onto analysis-image pixels.

    ``matrix`` is a 3x3 homography so both similarity and perspective fits share
    one representation. ``inlier_ratio`` and ``residual_px`` say how much to
    trust it -- fusion downgrades to derived-only geometry when they are poor
    rather than confidently placing ROIs in the wrong place.
    """

    matrix: np.ndarray
    method: str = "manual"
    inlier_ratio: float = 1.0
    residual_px: float = 0.0
    matched_points: int = 0
    # Matched pairs whose CAD part class and detector class agree. Guards
    # against a mirrored fit that scores well on distance alone.
    class_agreements: int = 0
    y_flipped: bool = False
    mirrored: bool = False
    # True when a substantially different transform scored just as well. A
    # symmetric layout inspected without class information has no unique
    # alignment, and the residual looks perfect either way.
    ambiguous: bool = False

    @property
    def scale_px_per_mm(self) -> float:
        """Average pixels per millimetre implied by the linear part."""

        linear = self.matrix[:2, :2]
        determinant = abs(float(np.linalg.det(linear)))
        return math.sqrt(determinant) if determinant > 0 else 0.0

    def is_usable(self, min_inlier_ratio: float, max_residual_px: float) -> bool:
        return (
            self.scale_px_per_mm > 0.0
            and self.inlier_ratio >= min_inlier_ratio
            and self.residual_px <= max_residual_px
        )

    def projected_points(self, points_mm: Sequence[Sequence[float]]) -> np.ndarray:
        return self.to_image(points_mm)

    def to_image(self, points_mm: Sequence[Sequence[float]]) -> np.ndarray:
        """Project ``(x, y)`` millimetre points into image pixels."""

        array = np.asarray(points_mm, dtype=np.float64).reshape(-1, 1, 2)
        if array.size == 0:
            return np.zeros((0, 2), dtype=np.float64)
        return cv2.perspectiveTransform(array, self.matrix).reshape(-1, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix": self.matrix.tolist(),
            "method": self.method,
            "inlier_ratio": float(self.inlier_ratio),
            "residual_px": float(self.residual_px),
            "matched_points": int(self.matched_points),
            "class_agreements": int(self.class_agreements),
            "ambiguous": bool(self.ambiguous),
            "scale_px_per_mm": self.scale_px_per_mm,
            "y_flipped": self.y_flipped,
            "mirrored": self.mirrored,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CadRegistration:
        matrix = np.asarray(payload["matrix"], dtype=np.float64)
        if matrix.shape != (3, 3):
            raise CadError("Registration matrix must be 3x3")
        return cls(
            matrix=matrix,
            method=str(payload.get("method", "manual")),
            inlier_ratio=float(payload.get("inlier_ratio", 1.0)),
            residual_px=float(payload.get("residual_px", 0.0)),
            matched_points=int(payload.get("matched_points", 0)),
            class_agreements=int(payload.get("class_agreements", 0)),
            ambiguous=bool(payload.get("ambiguous", False)),
            y_flipped=bool(payload.get("y_flipped", False)),
            mirrored=bool(payload.get("mirrored", False)),
        )


def register_from_fiducials(
    cad_points_mm: Sequence[Sequence[float]],
    image_points_px: Sequence[Sequence[float]],
    perspective: bool = False,
) -> CadRegistration:
    """Fit a transform from hand-picked correspondences.

    Three points give a full affine, which already absorbs the board's y-axis
    direction and a mirrored bottom side. Four or more with ``perspective`` fit
    a homography, for a camera that is not square to the board.
    """

    source = np.asarray(cad_points_mm, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(image_points_px, dtype=np.float64).reshape(-1, 2)
    if source.shape[0] != target.shape[0]:
        raise CadError("Fiducial lists must be the same length")
    if source.shape[0] < 2:
        raise CadError("At least two fiducials are needed")

    if perspective and source.shape[0] >= 4:
        matrix, _ = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
        method = "fiducial_homography"
    elif source.shape[0] >= 3:
        affine, _ = cv2.estimateAffine2D(source, target, method=cv2.LMEDS)
        matrix = _to_homography(affine)
        method = "fiducial_affine"
    else:
        affine, _ = cv2.estimateAffinePartial2D(source, target, method=cv2.LMEDS)
        matrix = _to_homography(affine)
        method = "fiducial_similarity"
    if matrix is None:
        raise CadError("Could not fit a transform to the given fiducials")

    registration = CadRegistration(matrix=np.asarray(matrix, dtype=np.float64), method=method)
    projected = registration.to_image(source)
    registration.residual_px = float(
        np.mean(np.linalg.norm(projected - target, axis=1))
    )
    registration.matched_points = int(source.shape[0])
    registration.mirrored = bool(np.linalg.det(registration.matrix[:2, :2]) < 0)
    return registration


def register_cad(
    board: BoardCad,
    detections: Sequence[Any],
    image_size: tuple[int, int],
    board_polygon: Sequence[Sequence[float]] | None = None,
    min_matches: int = 4,
    match_tolerance_ratio: float = 0.10,
    refine_rounds: int = 3,
    max_seed_trials: int = 500,
) -> CadRegistration | None:
    """Align a CAD board to the detections with no manual fiducials.

    Scale, rotation and offset are all unknown up front, so the seed cannot be
    guessed from extents: components never reach the board edges, and the
    located board is often just the whole frame. Instead a similarity transform
    is voted for by RANSAC over pairs of correspondences -- two matched parts
    determine scale, rotation and translation exactly -- and the best-supported
    seed is then refined by nearest-neighbour re-fitting.

    Both y-axis polarities are tried and the better fit wins, because CAD
    formats disagree on whether y grows up or down.

    Returns ``None`` when the fit does not converge on enough agreeing pairs. An
    unusable registration must be reported, not applied: silently mis-registered
    ROIs look exactly like correctly registered ones.
    """

    if not board.components or not detections:
        return None
    if board.extent() is None:
        return None

    targets = np.array(
        [
            [
                (detection.bbox.x1 + detection.bbox.x2) / 2.0,
                (detection.bbox.y1 + detection.bbox.y2) / 2.0,
            ]
            for detection in detections
        ],
        dtype=np.float64,
    )
    target_labels = [str(getattr(detection, "label", "")).lower() for detection in detections]
    sources = np.array(
        [[component.x, component.y] for component in board.components], dtype=np.float64
    )
    source_labels = [
        (component.part_class or "").lower() for component in board.components
    ]

    diagonal = float(np.hypot(image_size[0], image_size[1]))
    tolerance = match_tolerance_ratio * diagonal

    candidates: list[CadRegistration] = []
    for y_flipped in (False, True):
        flipped = sources.copy()
        if y_flipped:
            flipped[:, 1] = -flipped[:, 1]
        for seed in _seed_candidates(
            flipped,
            source_labels,
            targets,
            target_labels,
            image_size,
            board_polygon,
            tolerance,
            max_seed_trials,
        ):
            candidate = _refine_transform(
                flipped,
                source_labels,
                targets,
                target_labels,
                seed,
                tolerance,
                min_matches,
                refine_rounds,
            )
            if candidate is None:
                continue
            candidate.y_flipped = y_flipped
            # Fold the y negation in now so every candidate speaks raw CAD mm
            # and they can be compared against each other directly.
            if y_flipped:
                flip = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
                candidate.matrix = candidate.matrix @ flip
            candidate.mirrored = bool(np.linalg.det(candidate.matrix[:2, :2]) < 0)
            candidates.append(candidate)

    if not candidates:
        return None
    best = candidates[0]
    for candidate in candidates[1:]:
        if _is_better(candidate, best):
            best = candidate
    best.ambiguous = _has_rival(best, candidates, sources)
    return best


def _has_rival(
    best: CadRegistration,
    candidates: Sequence[CadRegistration],
    sources: np.ndarray,
) -> bool:
    """Is there a materially different transform that fits just as well?

    Board layouts are often near-symmetric, and without class information a
    mirrored or rotated alignment can score an identical residual while pairing
    every part with the wrong one. That is not a bug to be silently resolved --
    the data really does not determine the answer -- so it is reported.
    """

    reference = best.to_image(sources)
    spread = reference.max(axis=0) - reference.min(axis=0)
    extent = float(np.hypot(spread[0], spread[1]))
    if extent <= 0.0:
        return False
    for candidate in candidates:
        if candidate is best:
            continue
        if candidate.class_agreements < best.class_agreements:
            continue
        if candidate.matched_points < best.matched_points - 1:
            continue
        if candidate.residual_px > max(2.0 * best.residual_px, best.residual_px + 2.0):
            continue
        displacement = float(
            np.mean(np.linalg.norm(candidate.to_image(sources) - reference, axis=1))
        )
        if displacement > 0.10 * extent:
            return True
    return False


def _seed_candidates(
    sources: np.ndarray,
    source_labels: Sequence[str],
    targets: np.ndarray,
    target_labels: Sequence[str],
    image_size: tuple[int, int],
    board_polygon: Sequence[Sequence[float]] | None,
    tolerance: float,
    max_trials: int,
) -> list[np.ndarray]:
    """Seeds to refine from, best-supported first."""

    seeds: list[tuple[int, np.ndarray]] = []
    for matrix in _two_point_seeds(
        sources, source_labels, targets, target_labels, image_size, tolerance, max_trials
    ):
        seeds.append(
            (
                _support(matrix, sources, source_labels, targets, target_labels, tolerance),
                matrix,
            )
        )
    extent_seed = _extent_seed(sources, image_size, board_polygon)
    if extent_seed is not None:
        seeds.append(
            (
                _support(
                    extent_seed, sources, source_labels, targets, target_labels, tolerance
                ),
                extent_seed,
            )
        )
    seeds.sort(key=lambda item: item[0], reverse=True)
    # Refining every seed would be wasteful; the top few carry the signal.
    return [matrix for _, matrix in seeds[:3]]


def _two_point_seeds(
    sources: np.ndarray,
    source_labels: Sequence[str],
    targets: np.ndarray,
    target_labels: Sequence[str],
    image_size: tuple[int, int],
    tolerance: float,
    max_trials: int,
) -> list[np.ndarray]:
    """RANSAC over pairs of correspondences, keeping the best-supported few.

    Two correspondences pin a similarity transform exactly, so no prior on
    scale or rotation is needed. Sampling is seeded to a constant: a
    registration that changes between identical runs cannot be reviewed.
    """

    if len(sources) < 2 or len(targets) < 2:
        return []
    compatible = [
        (i, k)
        for i in range(len(sources))
        for k in range(len(targets))
        if classes_agree(source_labels[i], target_labels[k])
    ]
    if len(compatible) < 2:
        compatible = [(i, k) for i in range(len(sources)) for k in range(len(targets))]
    if len(compatible) < 2:
        return []

    rng = np.random.default_rng(20260819)
    diagonal = float(np.hypot(image_size[0], image_size[1]))
    scored: list[tuple[int, np.ndarray]] = []
    seen: set[tuple[int, int, int, int]] = set()
    trials = min(max_trials, max(64, len(compatible) * 8))
    for _ in range(trials):
        first, second = rng.choice(len(compatible), size=2, replace=False)
        i, k = compatible[int(first)]
        j, l = compatible[int(second)]
        if i == j or k == l or (i, k, j, l) in seen:
            continue
        seen.add((i, k, j, l))
        matrix = _similarity_from_pair(sources[i], sources[j], targets[k], targets[l])
        if matrix is None or not _plausible(matrix, sources, diagonal):
            continue
        scored.append(
            (
                _support(matrix, sources, source_labels, targets, target_labels, tolerance),
                matrix,
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    return [matrix for score, matrix in scored[:5] if score[1] >= 2]


def _similarity_from_pair(
    source_a: np.ndarray,
    source_b: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
) -> np.ndarray | None:
    """The unique rotation+uniform-scale+translation taking a,b onto their pair."""

    source_vector = source_b - source_a
    target_vector = target_b - target_a
    source_length = float(np.linalg.norm(source_vector))
    target_length = float(np.linalg.norm(target_vector))
    if source_length < 1e-9 or target_length < 1e-9:
        return None
    scale = target_length / source_length
    angle = math.atan2(target_vector[1], target_vector[0]) - math.atan2(
        source_vector[1], source_vector[0]
    )
    cos_a, sin_a = math.cos(angle) * scale, math.sin(angle) * scale
    linear = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    translation = target_a - linear @ source_a
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = linear
    matrix[:2, 2] = translation
    return matrix


def _plausible(matrix: np.ndarray, sources: np.ndarray, diagonal: float) -> bool:
    """Reject transforms that put the board nowhere near the frame."""

    projected = cv2.perspectiveTransform(
        sources.reshape(-1, 1, 2), matrix
    ).reshape(-1, 2)
    if not np.all(np.isfinite(projected)):
        return False
    spread = projected.max(axis=0) - projected.min(axis=0)
    extent = float(np.hypot(spread[0], spread[1]))
    return 0.02 * diagonal <= extent <= 4.0 * diagonal


def _support(
    matrix: np.ndarray,
    sources: np.ndarray,
    source_labels: Sequence[str],
    targets: np.ndarray,
    target_labels: Sequence[str],
    tolerance: float,
) -> tuple[int, int]:
    """Score a transform as ``(class-consistent pairs, all pairs)``.

    Distance alone cannot separate a board from its own mirror image when the
    layout is near-symmetric, and a mirrored fit pairs every part with the wrong
    neighbour while scoring a perfect residual. Class agreement is what breaks
    that tie, so it leads the score rather than acting as a tie-breaker.
    """

    projected = cv2.perspectiveTransform(
        sources.reshape(-1, 1, 2), matrix
    ).reshape(-1, 2)
    pairs = _nearest_pairs(projected, source_labels, targets, target_labels, tolerance)
    consistent = sum(
        1
        for source_index, target_index in pairs
        if classes_agree(source_labels[source_index], target_labels[target_index])
    )
    return (consistent, len(pairs))





def _extent_seed(
    sources: np.ndarray,
    image_size: tuple[int, int],
    board_polygon: Sequence[Sequence[float]] | None,
) -> np.ndarray | None:
    """Fall back to stretching the CAD extent over the located board.

    Only correct when the placed parts really do span the board, so it is a
    last-resort seed rather than the primary one.
    """

    x1, y1 = sources.min(axis=0)
    x2, y2 = sources.max(axis=0)
    if x2 - x1 < 1e-6 or y2 - y1 < 1e-6:
        return None
    if board_polygon is not None and len(board_polygon) >= 3:
        polygon = np.asarray(board_polygon, dtype=np.float64).reshape(-1, 2)
        tx1, ty1 = polygon.min(axis=0)
        tx2, ty2 = polygon.max(axis=0)
    else:
        tx1, ty1, tx2, ty2 = 0.0, 0.0, float(image_size[0]), float(image_size[1])
    # One uniform scale, so the seed cannot distort the board to fit.
    scale = min((tx2 - tx1) / (x2 - x1), (ty2 - ty1) / (y2 - y1))
    offset_x = (tx1 + tx2) / 2.0 - scale * (x1 + x2) / 2.0
    offset_y = (ty1 + ty2) / 2.0 - scale * (y1 + y2) / 2.0
    return np.array(
        [[scale, 0.0, offset_x], [0.0, scale, offset_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _is_better(candidate: CadRegistration, current: CadRegistration) -> bool:
    # Class agreement first: a mirrored fit can match just as many parts with
    # just as small a residual, and only the classes reveal it is wrong.
    if candidate.class_agreements != current.class_agreements:
        return candidate.class_agreements > current.class_agreements
    if candidate.matched_points != current.matched_points:
        return candidate.matched_points > current.matched_points
    return candidate.residual_px < current.residual_px


def _refine_transform(
    sources: np.ndarray,
    source_labels: Sequence[str],
    targets: np.ndarray,
    target_labels: Sequence[str],
    seed: np.ndarray,
    tolerance: float,
    min_matches: int,
    rounds: int,
) -> CadRegistration | None:
    """Iterate nearest-neighbour matching and a RANSAC similarity re-fit."""

    matrix = seed
    best: CadRegistration | None = None
    for _ in range(max(1, rounds)):
        projected = cv2.perspectiveTransform(
            sources.reshape(-1, 1, 2), matrix
        ).reshape(-1, 2)
        pairs = _nearest_pairs(
            projected, source_labels, targets, target_labels, tolerance
        )
        if len(pairs) < min_matches:
            break
        source_subset = np.array([sources[i] for i, _ in pairs], dtype=np.float64)
        target_subset = np.array([targets[j] for _, j in pairs], dtype=np.float64)
        affine, inliers = cv2.estimateAffinePartial2D(
            source_subset,
            target_subset,
            method=cv2.RANSAC,
            ransacReprojThreshold=max(3.0, tolerance * 0.25),
        )
        if affine is None:
            break
        matrix = _to_homography(affine)
        inlier_mask = (
            inliers.ravel().astype(bool)
            if inliers is not None
            else np.ones(len(pairs), dtype=bool)
        )
        inlier_count = int(inlier_mask.sum())
        if inlier_count < min_matches:
            break
        residual = float(
            np.mean(
                np.linalg.norm(
                    cv2.perspectiveTransform(
                        source_subset[inlier_mask].reshape(-1, 1, 2), matrix
                    ).reshape(-1, 2)
                    - target_subset[inlier_mask],
                    axis=1,
                )
            )
        )
        consistent, _ = _support(
            matrix, sources, source_labels, targets, target_labels, tolerance
        )
        best = CadRegistration(
            matrix=matrix,
            method="auto_detection_match",
            inlier_ratio=inlier_count / float(len(sources)),
            residual_px=residual,
            matched_points=inlier_count,
            class_agreements=consistent,
        )
        # Tighten the gate as the fit improves so late rounds stop chasing
        # far-away parts that were never the same component.
        tolerance = max(tolerance * 0.6, 4.0)
    return best


def _nearest_pairs(
    projected: np.ndarray,
    source_labels: Sequence[str],
    targets: np.ndarray,
    target_labels: Sequence[str],
    tolerance: float,
) -> list[tuple[int, int]]:
    """Greedy mutual nearest neighbours, preferring same-class pairs."""

    if projected.size == 0 or targets.size == 0:
        return []
    distances = np.linalg.norm(projected[:, None, :] - targets[None, :, :], axis=2)
    # A class disagreement is a soft penalty, not a veto: the detector's class is
    # itself unreliable, and the designator prefix is only a prior.
    penalty = np.array(
        [
            [
                0.0 if classes_agree(source_label, target_label) else tolerance * 0.5
                for target_label in target_labels
            ]
            for source_label in source_labels
        ],
        dtype=np.float64,
    )
    cost = distances + penalty
    pairs: list[tuple[int, int]] = []
    used_sources: set[int] = set()
    used_targets: set[int] = set()
    order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
    for source_index, target_index in order:
        source_index = int(source_index)
        target_index = int(target_index)
        if cost[source_index, target_index] > tolerance:
            break
        if source_index in used_sources or target_index in used_targets:
            continue
        used_sources.add(source_index)
        used_targets.add(target_index)
        pairs.append((source_index, target_index))
    return pairs


def _to_homography(affine: np.ndarray | None) -> np.ndarray | None:
    if affine is None:
        return None
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :] = np.asarray(affine, dtype=np.float64)
    return matrix
