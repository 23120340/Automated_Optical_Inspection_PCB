"""Use detected lead/pad boxes where the detector found them, derived geometry elsewhere.

The detector's ``pads`` and ``pins`` classes describe exactly the regions step
5.5 wants, so when the model actually finds one it beats anything inferred from
the body box. The catch is that it usually does not: on the shipped model those
two classes score 0.00 and 0.14-0.21 recall, because the training set holds 186
and 261 instances against 7775 capacitors.

So this module does not choose between the two approaches. It prefers a real
detection and falls back to the derived ROI per *terminal*, not per component --
a detector that finds one end of a chip resistor and misses the other must not
cost the pipeline the end it missed. That is the same precedence already used
for CAD in :mod:`aoi_pipeline.golden.inspector.fusion`, for the same reason: losing a
joint is worse than inspecting one twice.

The whole module is inert when the detector reports no lead classes, which is
the state of every model trained before this was written.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..config import LeadFusionConfig
from ..models import BoundingBox, Detection, SolderJoint, intersection_over_smaller

__all__ = [
    "LEAD_CLASSES",
    "LeadFusionResult",
    "assign_leads_to_components",
    "fuse_detected_leads",
    "split_lead_detections",
]

#: Detector classes that denote a lead/pad rather than a component body.
#:
#: The first six come from the public component datasets, which is where the
#: names ``pads``/``pins`` originate. ``solder_joint`` and ``joint`` are what a
#: model trained on this project's own hand-drawn boxes calls the same thing --
#: the labelling tool asks a reviewer to box solder joints, not "pads" -- and a
#: pass-2 model whose class name is absent from this set has its every detection
#: dropped here without a word. That failure costs a whole training run to find,
#: so the synonym belongs in the set rather than in the model's class list.
LEAD_CLASSES: frozenset[str] = frozenset(
    {"pads", "pad", "pins", "pin", "lead", "leads", "solder_joint", "joint"}
)


@dataclass(slots=True)
class LeadFusionResult:
    joints: list[SolderJoint] = field(default_factory=list)
    used_detected: int = 0
    used_derived: int = 0
    unassigned_leads: int = 0
    components_with_detected_leads: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "used_detected": self.used_detected,
            "used_derived": self.used_derived,
            "unassigned_leads": self.unassigned_leads,
            "components_with_detected_leads": self.components_with_detected_leads,
            "warnings": list(self.warnings),
        }


def split_lead_detections(
    detections: Sequence[Detection],
) -> tuple[list[Detection], list[Detection]]:
    """Separate lead/pad detections from component-body detections."""

    bodies: list[Detection] = []
    leads: list[Detection] = []
    for detection in detections:
        if str(detection.label).strip().lower() in LEAD_CLASSES:
            leads.append(detection)
        else:
            bodies.append(detection)
    return (bodies, leads)


def assign_leads_to_components(
    bodies: Sequence[Detection],
    leads: Sequence[Detection],
    config: LeadFusionConfig,
) -> dict[str, list[Detection]]:
    """Attach each lead detection to the component it belongs to.

    A lead sits just outside its component's body box, not inside it, so
    containment alone would assign almost nothing. Distance is measured from
    the lead's centre to the body box itself and scaled by the body's own size,
    which keeps one tolerance working across an 0402 chip and a connector.
    """

    assigned: dict[str, list[Detection]] = {}
    if not bodies or not leads:
        return assigned

    for lead in leads:
        if lead.confidence < config.min_lead_confidence:
            continue
        centre = _centre(lead.bbox)
        best_id: str | None = None
        best_distance = math.inf
        for body in bodies:
            reference = max(1.0, min(body.bbox.width, body.bbox.height))
            distance = _distance_to_box(centre, body.bbox) / reference
            if distance < best_distance:
                best_distance, best_id = distance, body.detection_id
        if best_id is not None and best_distance <= config.max_lead_distance_ratio:
            assigned.setdefault(best_id, []).append(lead)
    return assigned


def fuse_detected_leads(
    bodies: Sequence[Detection],
    leads: Sequence[Detection],
    derived_joints: Sequence[SolderJoint],
    config: LeadFusionConfig | None = None,
) -> LeadFusionResult:
    """Replace derived ROIs with detected leads where the detector found them.

    Per terminal, not per component: a body whose left lead was detected and
    whose right lead was missed keeps the detected box on the left and the
    derived box on the right. Swapping the whole component over to "detected"
    would silently drop the end the model could not see, which is precisely the
    end most likely to be defective.
    """

    config = config or LeadFusionConfig()
    result = LeadFusionResult()
    if not config.enabled or not leads:
        result.joints = list(derived_joints)
        result.used_derived = sum(1 for j in derived_joints if j.kind == "joint")
        return result

    assigned = assign_leads_to_components(bodies, leads, config)
    confident = [lead for lead in leads if lead.confidence >= config.min_lead_confidence]
    attached_ids = {
        lead.detection_id for items in assigned.values() for lead in items
    }
    orphans = [lead for lead in confident if lead.detection_id not in attached_ids]
    result.unassigned_leads = len(orphans)
    result.components_with_detected_leads = len(assigned)

    by_detection: dict[str, list[SolderJoint]] = {}
    for joint in derived_joints:
        by_detection.setdefault(joint.detection_id, []).append(joint)

    fused: list[SolderJoint] = []
    for detection_id, joints in by_detection.items():
        detected = assigned.get(detection_id, [])
        if not detected:
            fused.extend(joints)
            result.used_derived += sum(1 for j in joints if j.kind == "joint")
            continue

        own_joints = [j for j in joints if j.kind == "joint"]
        others = [j for j in joints if j.kind != "joint"]
        label = own_joints[0].label if own_joints else "component"
        geometry = own_joints[0].terminal_geometry if own_joints else "multi_pin"

        replaced: set[int] = set()
        for index, lead in enumerate(detected):
            covered = _best_overlap(lead.bbox, own_joints, config.replace_ios)
            if covered is not None:
                replaced.add(covered)
            fused.append(
                _joint_from_lead(lead, detection_id, label, geometry, index)
            )
            result.used_detected += 1

        # Keep every derived ROI the detector did not cover.
        for index, joint in enumerate(own_joints):
            if index in replaced:
                continue
            fused.append(joint)
            result.used_derived += 1
        fused.extend(others)

    # An orphan lead is not attached to any component -- but "belongs to no body"
    # and "is not a joint" are different claims. Refusing to guess an owner is
    # right; discarding a confident detection of a real pad is not.
    if orphans and config.keep_unassigned_leads:
        for index, lead in enumerate(orphans):
            fused.append(
                _joint_from_lead(
                    lead, lead.detection_id, lead.label, "pad_only", index
                )
            )
            result.used_detected += 1

    result.joints = fused
    if result.unassigned_leads:
        if config.keep_unassigned_leads:
            result.warnings.append(
                f"{result.unassigned_leads} lead detection(s) sat too far from any "
                "component to be assigned to one, so each was kept as a standalone "
                "ROI instead of being attached to the nearest body. That covers "
                "test points and unpopulated footprints; set "
                "LeadFusionConfig.keep_unassigned_leads=False to drop them."
            )
        else:
            result.warnings.append(
                f"{result.unassigned_leads} lead detection(s) sat too far from any "
                "component and were DROPPED (keep_unassigned_leads=False). Raise "
                "LeadFusionConfig.max_lead_distance_ratio if your leads really do "
                "stand that far off."
            )
    return result


# --------------------------------------------------------------------------- #


def _joint_from_lead(
    lead: Detection,
    detection_id: str,
    label: str,
    geometry: str,
    index: int,
) -> SolderJoint:
    return SolderJoint(
        detection_id=detection_id,
        joint_id=f"{detection_id}_detected{index:02d}",
        label=label,
        kind="joint",
        bbox=lead.bbox,
        terminal_geometry=geometry,
        position=f"detected_{lead.label.lower()}{index:02d}",
        source="detected",
        metadata={
            "lead_detection_id": lead.detection_id,
            "lead_label": lead.label,
            "lead_confidence": float(lead.confidence),
        },
    )


def _best_overlap(
    box: BoundingBox, joints: Sequence[SolderJoint], threshold: float
) -> int | None:
    """Index of the derived ROI this detection stands in for, if any."""

    best_index, best_score = None, threshold
    for index, joint in enumerate(joints):
        score = _intersection_over_smaller(box, joint.bbox)
        if score >= best_score:
            best_index, best_score = index, score
    return best_index


_intersection_over_smaller = intersection_over_smaller


def _centre(box: BoundingBox) -> tuple[float, float]:
    return ((box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0)


def _distance_to_box(point: tuple[float, float], box: BoundingBox) -> float:
    """Euclidean distance from a point to a box; zero when the point is inside."""

    dx = max(box.x1 - point[0], 0.0, point[0] - box.x2)
    dy = max(box.y1 - point[1], 0.0, point[1] - box.y2)
    return math.hypot(dx, dy)
