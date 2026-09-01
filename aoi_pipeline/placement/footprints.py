"""Conservative parsing of package/footprint names from BOM and placement files.

Footprint names are not governed by one universal grammar.  A trailing number
can be a pin count (``SOIC-16``), an industry package identifier (``SOT-23``),
an EIA body size (``0603``), a pitch, or part of a vendor code.  This module
therefore recognises a small set of explicit package grammars and never falls
back to "the last number must be the pin count".

The seven ``package_class`` values mirror the package-label plan.  The separate
``terminal_geometry`` value is deliberately technical: it describes the ROI
policy a solder-joint extractor needs, without making Vietnamese UI labels part
of that algorithmic contract.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

__all__ = [
    "FootprintProfile",
    "PACKAGE_CLASS_SLUGS",
    "PackageClass",
    "TerminalGeometry",
    "normalize_footprint",
    "parse_footprint",
    "profile_for_package_class",
]


PackageClass = Literal[
    "hai_chan",
    "tru_dung",
    "goi_nho",
    "ic_hai_ben",
    "ic_bon_ben",
    "ic_khong_chan",
    "connector",
]

TerminalGeometry = Literal[
    "two_terminal",
    "vertical_two_terminal",
    "sparse_two_sided",
    "dual_sided",
    "four_sided",
    "hidden_terminals",
    "connector_rows",
]

PACKAGE_CLASS_SLUGS: frozenset[str] = frozenset(
    {
        "hai_chan",
        "tru_dung",
        "goi_nho",
        "ic_hai_ben",
        "ic_bon_ben",
        "ic_khong_chan",
        "connector",
    }
)

_GEOMETRY_BY_CLASS: dict[str, TerminalGeometry] = {
    "hai_chan": "two_terminal",
    "tru_dung": "vertical_two_terminal",
    "goi_nho": "sparse_two_sided",
    "ic_hai_ben": "dual_sided",
    "ic_bon_ben": "four_sided",
    "ic_khong_chan": "hidden_terminals",
    "connector": "connector_rows",
}


@dataclass(frozen=True, slots=True)
class FootprintProfile:
    """Structured terminal hints derived from one footprint name.

    ``expected_pin_count`` is populated only when the recognised grammar makes
    an exact claim.  ``expected_pin_count_range`` is used for a recognised but
    inherently variable family.  They are mutually exclusive.  ``lead_sides``
    is the number of distinct body sides/regions where step 5.5 should place
    terminal ROIs.  A vertical can therefore has two opposing ROI regions even
    though its physical leads emerge underneath; ``0`` is reserved for hidden
    terminal packages that must not create top-down ROIs.  ``None`` means the
    name did not specify a safe answer.

    ``confidence`` describes confidence in parsing the name, not confidence in
    an image model and not the probability that the assembled part is correct.
    """

    raw: str
    normalized: str
    package_class: PackageClass
    terminal_geometry: TerminalGeometry
    expected_pin_count: int | None = None
    expected_pin_count_range: tuple[int, int] | None = None
    lead_sides: int | None = None
    confidence: float = 1.0
    reason: str = ""

    def __post_init__(self) -> None:
        if self.package_class not in PACKAGE_CLASS_SLUGS:
            raise ValueError(f"Unknown package class: {self.package_class}")
        expected_geometry = _GEOMETRY_BY_CLASS[self.package_class]
        if self.terminal_geometry != expected_geometry:
            raise ValueError(
                f"{self.package_class} requires terminal geometry "
                f"{expected_geometry}, got {self.terminal_geometry}"
            )
        if self.expected_pin_count is not None:
            if isinstance(self.expected_pin_count, bool) or self.expected_pin_count <= 0:
                raise ValueError("expected_pin_count must be a positive integer")
            if self.expected_pin_count_range is not None:
                raise ValueError("exact pin count and pin-count range are exclusive")
        if self.expected_pin_count_range is not None:
            low, high = self.expected_pin_count_range
            if (
                isinstance(low, bool)
                or isinstance(high, bool)
                or low <= 0
                or high < low
            ):
                raise ValueError("expected_pin_count_range must be positive and ordered")
        if self.lead_sides is not None:
            if isinstance(self.lead_sides, bool) or not 0 <= self.lead_sides <= 4:
                raise ValueError("lead_sides must be between zero and four")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        if not self.normalized:
            raise ValueError("normalized footprint name must not be empty")
        if not self.reason:
            raise ValueError("reason must explain how the footprint was parsed")

    @property
    def pin_count_bounds(self) -> tuple[int, int] | None:
        """Return a uniform ``(minimum, maximum)`` view of the pin hint."""

        if self.expected_pin_count is not None:
            return self.expected_pin_count, self.expected_pin_count
        return self.expected_pin_count_range

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation without workstation state."""

        return {
            "raw": self.raw,
            "normalized": self.normalized,
            "package_class": self.package_class,
            "terminal_geometry": self.terminal_geometry,
            "expected_pin_count": self.expected_pin_count,
            "expected_pin_count_range": (
                list(self.expected_pin_count_range)
                if self.expected_pin_count_range is not None
                else None
            ),
            "lead_sides": self.lead_sides,
            "confidence": self.confidence,
            "reason": self.reason,
        }


def normalize_footprint(value: str) -> str:
    """Normalise separators and case while retaining package-significant text."""

    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", value).strip().upper()
    text = text.replace("×", "X")
    return re.sub(r"[^A-Z0-9]+", "-", text).strip("-")


def _profile(
    raw: str,
    normalized: str,
    package_class: PackageClass,
    *,
    expected_pin_count: int | None = None,
    expected_pin_count_range: tuple[int, int] | None = None,
    lead_sides: int | None,
    confidence: float,
    reason: str,
) -> FootprintProfile:
    return FootprintProfile(
        raw=raw,
        normalized=normalized,
        package_class=package_class,
        terminal_geometry=_GEOMETRY_BY_CLASS[package_class],
        expected_pin_count=expected_pin_count,
        expected_pin_count_range=expected_pin_count_range,
        lead_sides=lead_sides,
        confidence=confidence,
        reason=reason,
    )


_CLASS_DEFAULTS: dict[
    str, tuple[int | None, tuple[int, int] | None, int | None]
] = {
    "hai_chan": (2, None, 2),
    "tru_dung": (2, None, 2),
    "goi_nho": (None, (3, 5), 2),
    "ic_hai_ben": (None, (6, 256), 2),
    "ic_bon_ben": (None, (8, 512), 4),
    "ic_khong_chan": (None, (2, 2048), 0),
    "connector": (None, (1, 512), None),
}


def profile_for_package_class(
    package_class: PackageClass,
    *,
    source: str = "package",
) -> FootprintProfile:
    """Build the topology defaults for an already-known seven-class label.

    This is the bridge for a future package classifier: the classifier already
    supplied the class, so no footprint string is parsed and the broad count
    range must not be presented as an exact count.  Keeping the defaults here
    prevents the training/runtime integration from duplicating the topology
    table used by BOM parsing.
    """

    if package_class not in PACKAGE_CLASS_SLUGS:
        raise ValueError(f"Unknown package class: {package_class}")
    exact, bounds, lead_sides = _CLASS_DEFAULTS[package_class]
    raw = str(source)
    normalized = normalize_footprint(raw) or "PACKAGE"
    return _profile(
        raw,
        normalized,
        package_class,
        expected_pin_count=exact,
        expected_pin_count_range=bounds,
        lead_sides=lead_sides,
        confidence=1.0,
        reason=f"package class {package_class} supplied explicitly by {source}",
    )


def _alias_pattern(aliases: Sequence[str]) -> str:
    # Longest first prevents ``SO`` from consuming the prefix of ``SOIC``.
    return "|".join(re.escape(item) for item in sorted(aliases, key=len, reverse=True))


def _family_count(
    normalized: str,
    aliases: Sequence[str],
    valid_count: Callable[[int], bool],
) -> tuple[str | None, int | None]:
    """Find a named family and an optional, grammar-local pin count.

    The number has to be immediately attached to the family or be its next
    token.  Dimensions such as ``QFN-7X7MM`` consequently do not become seven
    pins.  If a namespace mentions a family before the actual item name (for
    example ``PACKAGE-SO-SOIC-8``), a later match carrying a valid count wins.
    """

    pattern = re.compile(
        rf"(?:^|-)(?P<family>{_alias_pattern(aliases)})(?:-)?"
        rf"(?P<count>\d{{1,4}})?(?=-|$)"
    )
    recognised: str | None = None
    for match in pattern.finditer(normalized):
        family = match.group("family")
        if recognised is None or len(family) > len(recognised):
            recognised = family
        digits = match.group("count")
        if digits is None:
            continue
        count = int(digits)
        if valid_count(count):
            return family, count
    return recognised, None


def _has_alias(normalized: str, aliases: Sequence[str]) -> bool:
    pattern = re.compile(rf"(?:^|-)(?:{_alias_pattern(aliases)})(?=-|$)")
    return bool(pattern.search(normalized))


_PLACEHOLDERS = frozenset(
    {"UNKNOWN", "UNSPECIFIED", "NONE", "NULL", "N-A", "NA", "TBD", "CUSTOM", "GENERIC"}
)

# Imperial and metric EIA chip-size codes seen in common KiCad/Altium exports.
# Regardless of which dimensional convention a code uses, it describes a
# two-terminal chip; it is never a pin count.
_EIA_CODES = frozenset(
    {
        "01005",
        "0201",
        "0402",
        "0603",
        "0805",
        "1005",
        "1206",
        "1210",
        "1218",
        "1608",
        "1812",
        "2010",
        "2012",
        "2512",
        "3216",
        "3225",
        "4532",
        "5025",
        "6332",
    }
)
_EIA_PREFIXES = ("R", "C", "L", "D", "LED", "RES", "CAP", "IND")


def _eia_code(normalized: str) -> str | None:
    for token in normalized.split("-"):
        candidate = token.removesuffix("METRIC")
        if candidate in _EIA_CODES:
            return candidate
        for prefix in _EIA_PREFIXES:
            if candidate.startswith(prefix) and candidate[len(prefix):] in _EIA_CODES:
                return candidate[len(prefix):]
    return None


_SOT_DEFAULT_PINS: dict[int, int] = {
    23: 3,
    89: 3,
    96: 8,  # SOT-96 is the SO-8 style in the package-labelled source set.
    143: 4,
    223: 4,
    323: 3,
    343: 4,
    353: 5,
    363: 6,
    457: 6,
    523: 3,
    553: 5,
    563: 6,
    723: 3,
    753: 5,
    883: 8,
}


def _parse_sot(raw: str, normalized: str) -> FootprintProfile | None:
    match = re.search(r"(?:^|-)SOT-?(?P<base>\d{2,3})(?=-|$)", normalized)
    if match is None:
        return None
    base = int(match.group("base"))

    # SOT-23-N is an established variant notation.  The 23 by itself is a
    # JEDEC family number and means three terminals, never twenty-three.
    variant_count: int | None = None
    if base == 23:
        suffix = re.match(r"-(?P<count>[3-8])(?=-|$)", normalized[match.end():])
        if suffix is not None:
            variant_count = int(suffix.group("count"))

    count = variant_count if variant_count is not None else _SOT_DEFAULT_PINS.get(base)
    if base == 96:
        return _profile(
            raw,
            normalized,
            "ic_hai_ben",
            expected_pin_count=count,
            lead_sides=2,
            confidence=0.97,
            reason="SOT-96 is the dual-row SO-8 package style",
        )
    return _profile(
        raw,
        normalized,
        "goi_nho",
        expected_pin_count=count,
        expected_pin_count_range=None if count is not None else (3, 8),
        lead_sides=2,
        confidence=0.98 if count is not None else 0.84,
        reason=(
            f"recognised SOT-{base} package family"
            + (f" with {count} terminals" if count is not None else "; pin count is not encoded")
        ),
    )


_TO_DEFAULT_PINS: dict[int, int] = {
    39: 3,
    46: 3,
    92: 3,
    126: 3,
    220: 3,
    247: 3,
    251: 3,
    252: 3,
    262: 3,
    263: 3,
}


def _parse_to(raw: str, normalized: str) -> FootprintProfile | None:
    match = re.search(r"(?:^|-)TO-?(?P<base>\d{2,3})(?=-|$)", normalized)
    if match is None:
        return None
    base = int(match.group("base"))
    suffix = re.match(r"-(?P<count>\d{1,2})(?=-|$)", normalized[match.end():])
    explicit_count = int(suffix.group("count")) if suffix is not None else None
    if explicit_count is not None and not 2 <= explicit_count <= 16:
        explicit_count = None
    count = explicit_count if explicit_count is not None else _TO_DEFAULT_PINS.get(base)
    return _profile(
        raw,
        normalized,
        "goi_nho",
        expected_pin_count=count,
        expected_pin_count_range=None if count is not None else (3, 8),
        lead_sides=2,
        confidence=0.95 if count is not None else 0.78,
        reason=(
            f"recognised TO-{base} package family"
            + (f" with {count} terminals" if count is not None else "; pin count is not encoded safely")
        ),
    )


def _connector_profile(raw: str, normalized: str) -> FootprintProfile | None:
    dip_family, dip_count = _family_count(
        normalized,
        ("PDIP", "CDIP", "SDIP", "DIP", "SIP", "SIL"),
        lambda count: 2 <= count <= 128,
    )
    if dip_family is not None:
        sides = 1 if dip_family in {"SIP", "SIL"} else 2
        return _profile(
            raw,
            normalized,
            "connector",
            expected_pin_count=dip_count,
            lead_sides=sides,
            confidence=0.98 if dip_count is not None else 0.90,
            reason=f"recognised {dip_family} through-hole row package",
        )

    connector_aliases = (
        "TERMINAL-BLOCK",
        "PIN-HEADER",
        "PINHEADER",
        "CONNECTOR",
        "HEADER",
        "SOCKET",
        "CONN",
        "MOLEX",
        "JST",
        "IDC",
        "D-SUB",
        "DSUB",
    )
    family, family_count = _family_count(
        normalized, connector_aliases, lambda count: 1 <= count <= 512
    )
    generic_clue = family is not None or _has_alias(
        normalized,
        ("RJ45", "RJ11", "USB", "USB-C", "MICRO-USB", "TYPE-C"),
    )
    if not generic_clue:
        # DB9/DE9 are connector family identifiers with an actual contact count.
        dsub = re.search(r"(?:^|-)(?:DB|DE)-?(?P<count>\d{1,2})(?=-|$)", normalized)
        if dsub is None:
            return None
        count = int(dsub.group("count"))
        if not 1 <= count <= 64:
            return None
        return _profile(
            raw,
            normalized,
            "connector",
            expected_pin_count=count,
            lead_sides=2,
            confidence=0.96,
            reason="recognised D-sub connector contact count",
        )

    # Header notation is rows x positions-per-row.  Requiring a complete token
    # avoids reading dimensions like ``2x5mm`` as ten contacts.
    matrices = list(
        re.finditer(
            r"(?:^|-)(?P<rows>0?[1-4])X(?P<columns>0?\d{1,3})(?=-|$)",
            normalized,
        )
    )
    if matrices:
        matrix = matrices[-1]
        rows = int(matrix.group("rows"))
        columns = int(matrix.group("columns"))
        count = rows * columns
        if columns > 0 and count <= 512:
            return _profile(
                raw,
                normalized,
                "connector",
                expected_pin_count=count,
                lead_sides=1 if rows == 1 else 2,
                confidence=0.99,
                reason=f"connector matrix {rows}x{columns} encodes {count} contacts",
            )

    explicit = re.search(
        r"(?:^|-)(?P<count>\d{1,3})(?:PIN|PINS|POS|POSITION|P)(?=-|$)",
        normalized,
    )
    explicit_count = int(explicit.group("count")) if explicit is not None else None
    if explicit_count is not None and not 1 <= explicit_count <= 512:
        explicit_count = None
    count = family_count if family_count is not None else explicit_count
    return _profile(
        raw,
        normalized,
        "connector",
        expected_pin_count=count,
        lead_sides=None,
        confidence=0.95 if count is not None else 0.86,
        reason=(
            f"recognised {family or 'connector'} package"
            + (f" with {count} contacts" if count is not None else "; contact count is not encoded safely")
        ),
    )


def parse_footprint(value: str | None) -> FootprintProfile | None:
    """Parse a BOM/PnP footprint name into a conservative package profile.

    Unknown or placeholder values return ``None``.  A recognised family can
    still return a profile with no expected pin count; callers may safely use
    its coarse package class, but must not invent a count.  In particular:

    - ``0603`` is an EIA body-size code and always means two terminals;
    - ``SOT-23`` is a three-terminal package family (variants spell out a
      further suffix such as ``SOT-23-5``);
    - only family-specific syntax such as ``SOIC-16`` or ``QFP-64`` is allowed
      to turn a number into a general IC pin count.
    """

    if not isinstance(value, str):
        return None
    raw = value
    normalized = normalize_footprint(value)
    if not normalized or normalized in _PLACEHOLDERS or len(normalized) > 2048:
        return None

    # No-visible-lead packages must be checked before generic connector and EIA
    # clues because real library names often contain dimensions and namespaces.
    hidden_family, hidden_count = _family_count(
        normalized,
        ("DSBGA", "WLCSP", "QFN", "DFN", "BGA", "LGA"),
        lambda count: 2 <= count <= 2048,
    )
    if hidden_family is not None:
        return _profile(
            raw,
            normalized,
            "ic_khong_chan",
            expected_pin_count=hidden_count,
            lead_sides=0,
            confidence=0.99 if hidden_count is not None else 0.94,
            reason=(
                f"{hidden_family} has terminals hidden underneath the body"
                + (f"; name encodes {hidden_count} terminals" if hidden_count is not None else "")
            ),
        )

    four_family, four_count = _family_count(
        normalized,
        ("TQFP", "LQFP", "PQFP", "CQFP", "QFP"),
        lambda count: 12 <= count <= 512 and count % 4 == 0,
    )
    if four_family is not None:
        return _profile(
            raw,
            normalized,
            "ic_bon_ben",
            expected_pin_count=four_count,
            lead_sides=4,
            confidence=0.99 if four_count is not None else 0.94,
            reason=(
                f"{four_family} is a four-sided leaded IC"
                + (f" with {four_count} pins" if four_count is not None else "")
            ),
        )

    dual_family, dual_count = _family_count(
        normalized,
        ("TSSOP", "SSOP", "SOIC", "MSOP", "QSOP", "TSOP", "SOP", "SO"),
        lambda count: 4 <= count <= 256 and count % 2 == 0,
    )
    if dual_family is not None:
        return _profile(
            raw,
            normalized,
            "ic_hai_ben",
            expected_pin_count=dual_count,
            lead_sides=2,
            confidence=0.99 if dual_count is not None else 0.93,
            reason=(
                f"{dual_family} is a dual-sided leaded IC"
                + (f" with {dual_count} pins" if dual_count is not None else "")
            ),
        )

    connector = _connector_profile(raw, normalized)
    if connector is not None:
        return connector

    # A vertical electrolytic/radial capacitor needs its own geometry even
    # though electrically it still has two terminals.
    if _has_alias(normalized, ("RADIAL",)) and _has_alias(
        normalized,
        ("CAPACITOR", "ELECTROLYTIC", "ECAP", "ELCO", "CAP", "CP", "C"),
    ):
        return _profile(
            raw,
            normalized,
            "tru_dung",
            expected_pin_count=2,
            lead_sides=2,
            confidence=0.98,
            reason="radial capacitor is a vertical two-terminal can",
        )

    # SOD numbers, MELF dimensions and DO case numbers are package identifiers,
    # never terminal counts.
    if (
        re.search(r"(?:^|-)SOD(?:-?\d{2,3}[A-Z]*)?(?=-|$)", normalized)
        or _has_alias(normalized, ("MELF", "MINIMELF", "MICROMELF", "LL-34"))
        or re.search(r"(?:^|-)DO-?(?:35|41|213[A-Z]*)(?=-|$)", normalized)
    ):
        return _profile(
            raw,
            normalized,
            "hai_chan",
            expected_pin_count=2,
            lead_sides=2,
            confidence=0.99,
            reason="recognised two-terminal diode/MELF package family",
        )

    sot = _parse_sot(raw, normalized)
    if sot is not None:
        return sot
    to_profile = _parse_to(raw, normalized)
    if to_profile is not None:
        return to_profile

    eia = _eia_code(normalized)
    if eia is not None:
        return _profile(
            raw,
            normalized,
            "hai_chan",
            expected_pin_count=2,
            lead_sides=2,
            confidence=0.99,
            reason=f"{eia} is an EIA chip body-size code, not a pin count",
        )

    # Useful unnumbered names which still make an unambiguous two-terminal
    # geometry claim.  Keep this after explicit families so that a library path
    # mentioning CHIP cannot override QFN/SOIC/etc.
    if _has_alias(normalized, ("AXIAL", "MLCC")) or (
        _has_alias(normalized, ("CHIP", "SMD"))
        and _has_alias(
            normalized,
            ("RESISTOR", "CAPACITOR", "DIODE", "LED", "INDUCTOR", "FUSE"),
        )
    ):
        return _profile(
            raw,
            normalized,
            "hai_chan",
            expected_pin_count=2,
            lead_sides=2,
            confidence=0.90,
            reason="component-family words unambiguously describe a two-terminal package",
        )

    return None
