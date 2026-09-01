"""Fail-visible checks between package evidence and step-5.5 ROI topology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..models import Detection, SolderJoint


_DEFAULT_EXPECTATIONS: dict[str, dict[str, Any]] = {
    "hai_chan": {"pin_range": (2, 2), "lead_sides": (2,)},
    "tru_dung": {"pin_range": (2, 2), "lead_sides": (2,)},
    "goi_nho": {"pin_range": (3, 5), "lead_sides": (2,)},
    "ic_hai_ben": {"pin_range": (6, None), "lead_sides": (2,)},
    "ic_bon_ben": {"pin_range": (8, None), "lead_sides": (4,)},
    "ic_khong_chan": {"pin_range": (0, 0), "lead_sides": (0,)},
    "connector": {"pin_range": (1, None), "lead_sides": (1, 2)},
}


@dataclass(frozen=True, slots=True)
class PackageTopologyCheck:
    """Whether the emitted ROI topology agrees with an independent source."""

    detection_id: str
    package_class: str
    source: str
    status: str
    expected_pin_min: int | None
    expected_pin_max: int | None
    expected_lead_sides: tuple[int, ...]
    actual_pin_count: int | None
    actual_roi_count: int
    actual_lead_sides: int | None
    reason: str = ""

    @property
    def review_required(self) -> bool:
        return self.status == "review"

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "package_class": self.package_class,
            "source": self.source,
            "status": self.status,
            "expected_pin_min": self.expected_pin_min,
            "expected_pin_max": self.expected_pin_max,
            "expected_lead_sides": list(self.expected_lead_sides),
            "actual_pin_count": self.actual_pin_count,
            "actual_roi_count": self.actual_roi_count,
            "actual_lead_sides": self.actual_lead_sides,
            "review_required": self.review_required,
            "reason": self.reason,
        }


def assess_package_topology(
    detections: Sequence[Detection],
    joints: Sequence[SolderJoint],
) -> list[PackageTopologyCheck]:
    """Compare expected package rank/sides with ROIs that really survived.

    An unsplit lead *band* is not counted as one pin.  It is reported as an
    unmeasurable count and sent to review, avoiding the dangerous alternative
    of treating two bands of an SOIC-16 as evidence for two terminals.
    """

    joints_by_detection: dict[str, list[SolderJoint]] = {}
    profiles: dict[str, Mapping[str, Any]] = {}
    for joint in joints:
        if joint.kind == "joint":
            joints_by_detection.setdefault(joint.detection_id, []).append(joint)
        profile = joint.metadata.get("package_profile")
        if isinstance(profile, Mapping) and joint.detection_id:
            profiles[joint.detection_id] = profile
    for detection in detections:
        profile = detection.metadata.get("package_profile")
        if isinstance(profile, Mapping):
            # A CAD/footprint profile attached during fusion outranks the
            # image-model profile that was present on the source detection.
            profiles.setdefault(detection.detection_id, profile)

    checks: list[PackageTopologyCheck] = []
    for detection in detections:
        profile = profiles.get(detection.detection_id)
        if profile is None:
            continue
        package_class = str(profile.get("package_class", "")).strip()
        expectation = _DEFAULT_EXPECTATIONS.get(package_class)
        if expectation is None:
            continue
        source = str(profile.get("source") or "unknown")
        expected_min, expected_max = _expected_pin_range(profile, expectation)
        expected_sides = _expected_sides(profile, expectation)
        own = joints_by_detection.get(detection.detection_id, [])
        actual_sides = _lead_side_count(own, package_class)
        actual_pin_count = _count_pins(own, package_class)

        if package_class == "ic_khong_chan":
            status = "not_inspectable" if not own else "review"
            reason = (
                "Top-down 2D inspection is not applicable to hidden terminals"
                if not own
                else "Hidden-terminal package unexpectedly produced solder ROIs"
            )
        elif not own:
            status = "review"
            reason = "Package evidence expected terminals but step 5.5 emitted no joint ROI"
        elif actual_pin_count is None:
            status = "review"
            reason = "Lead bands could not be split into a trustworthy pin count"
        elif not _within(actual_pin_count, expected_min, expected_max):
            status = "review"
            reason = (
                f"Emitted {actual_pin_count} pin ROIs, outside expected range "
                f"{_range_text(expected_min, expected_max)}"
            )
        elif actual_sides is not None and expected_sides and actual_sides not in expected_sides:
            status = "review"
            reason = (
                f"ROIs occupy {actual_sides} lead sides; expected "
                f"{','.join(str(value) for value in expected_sides)}"
            )
        else:
            status = "pass"
            reason = ""

        checks.append(
            PackageTopologyCheck(
                detection_id=detection.detection_id,
                package_class=package_class,
                source=source,
                status=status,
                expected_pin_min=expected_min,
                expected_pin_max=expected_max,
                expected_lead_sides=expected_sides,
                actual_pin_count=actual_pin_count,
                actual_roi_count=len(own),
                actual_lead_sides=actual_sides,
                reason=reason,
            )
        )
    return checks


def _expected_pin_range(
    profile: Mapping[str, Any], expectation: Mapping[str, Any]
) -> tuple[int | None, int | None]:
    exact = profile.get("expected_pin_count")
    if isinstance(exact, int) and not isinstance(exact, bool) and exact >= 0:
        return exact, exact
    raw = profile.get("expected_pin_count_range") or profile.get("expected_pin_range")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) == 2:
        low = _nonnegative_int(raw[0])
        high = _nonnegative_int(raw[1])
        return low, high
    low, high = expectation["pin_range"]
    return low, high


def _expected_sides(
    profile: Mapping[str, Any], expectation: Mapping[str, Any]
) -> tuple[int, ...]:
    raw = profile.get("lead_sides")
    if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
        return (raw,)
    return tuple(int(value) for value in expectation["lead_sides"])


def _count_pins(joints: Sequence[SolderJoint], package_class: str) -> int | None:
    if package_class in {"hai_chan", "tru_dung"}:
        return len(joints)
    if package_class == "ic_khong_chan":
        return 0 if not joints else len(joints)
    if not joints or any(joint.pin_index is None for joint in joints):
        return None
    return len(joints)


def _lead_side_count(
    joints: Sequence[SolderJoint], package_class: str
) -> int | None:
    if not joints:
        return 0
    if package_class in {"hai_chan", "tru_dung"}:
        return 2
    sides: set[str] = set()
    for joint in joints:
        position = joint.position
        for side in ("left", "right", "top", "bottom"):
            if position.startswith(f"lead_{side}"):
                sides.add(side)
                break
    return len(sides) if sides else None


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _within(value: int, low: int | None, high: int | None) -> bool:
    return (low is None or value >= low) and (high is None or value <= high)


def _range_text(low: int | None, high: int | None) -> str:
    if low == high:
        return str(low)
    return f"{low if low is not None else '-inf'}..{high if high is not None else 'inf'}"


__all__ = ["PackageTopologyCheck", "assess_package_topology"]
