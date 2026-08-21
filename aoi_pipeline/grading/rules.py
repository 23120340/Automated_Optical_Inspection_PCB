"""Layer A verdicts: turn solder measurements into a call, with a reason.

Thresholds live in :class:`SolderGradingConfig` rather than here so a line can
be tuned without touching code, and every trip records the number that caused
it. An AOI call an operator cannot trace back to a measurement gets overridden
on the floor within a week, so the reason string is part of the output, not
debug logging.

These rules are deliberately biased toward calling things out: on this layer a
false call costs an operator ten seconds, while an escape ships a bad board.
The fusion layer is where that bias gets balanced against the model.
"""

from __future__ import annotations

from typing import Sequence

from ..config import SolderGradingConfig
from ..models import SolderFeatures, SolderJoint

__all__ = [
    "COMPONENT_CLASSES",
    "JOINT_CLASSES",
    "grade_component_by_rules",
    "grade_joint_by_rules",
    "mark_bridges",
]

#: Joint-scope taxonomy. Keep this list and the trained model's class_names in
#: step, or fusion cannot compare the two.
JOINT_CLASSES: tuple[str, ...] = (
    "good",
    "insufficient",
    "excess",
    "bridge",
    "cold",
    "missing_solder",
)

#: Component-scope taxonomy, graded on the ``body`` ROIs that step 5.5 emits
#: alongside the joints.
COMPONENT_CLASSES: tuple[str, ...] = (
    "ok",
    "missing",
    "tombstone",
    "shifted",
    "wrong_polarity",
)


def grade_joint_by_rules(
    features: SolderFeatures,
    config: SolderGradingConfig,
) -> tuple[str, list[str]]:
    """Classify one joint ROI from its measurements.

    Order matters: the checks run from the most certain physical evidence to
    the least, and the first one to trip wins. Missing solder is unambiguous;
    "cold" is the weakest inference here and only gets to speak once the amount
    of solder has already been found acceptable.
    """

    reasons: list[str] = []

    if features.solder_ratio <= config.missing_solder_ratio:
        reasons.append(
            f"solder_ratio {features.solder_ratio:.3f} <= "
            f"{config.missing_solder_ratio:.3f} (không thấy thiếc)"
        )
        return ("missing_solder", reasons)

    if features.solder_ratio < config.insufficient_solder_ratio:
        reasons.append(
            f"solder_ratio {features.solder_ratio:.3f} < "
            f"{config.insufficient_solder_ratio:.3f} (thiếu thiếc)"
        )
        return ("insufficient", reasons)

    if (
        features.span_ratio < config.insufficient_span_ratio
        and features.solder_ratio < config.excess_solder_ratio
    ):
        reasons.append(
            f"span_ratio {features.span_ratio:.3f} < "
            f"{config.insufficient_span_ratio:.3f} (fillet không với tới hết land)"
        )
        return ("insufficient", reasons)

    if features.solder_ratio > config.excess_solder_ratio:
        reasons.append(
            f"solder_ratio {features.solder_ratio:.3f} > "
            f"{config.excess_solder_ratio:.3f} (thừa thiếc)"
        )
        return ("excess", reasons)

    # Dull and flat with a normal amount of solder is the cold-joint signature.
    # Both conditions are required: a dim ROI on its own is usually exposure.
    if (
        features.specular_ratio < config.cold_specular_ratio
        and features.contrast < config.cold_contrast
    ):
        reasons.append(
            f"specular_ratio {features.specular_ratio:.3f} < "
            f"{config.cold_specular_ratio:.3f} và contrast "
            f"{features.contrast:.3f} < {config.cold_contrast:.3f} (mối hàn xỉn)"
        )
        return ("cold", reasons)

    if features.centroid_offset_ratio > config.max_centroid_offset_ratio:
        reasons.append(
            f"centroid_offset {features.centroid_offset_ratio:.3f} > "
            f"{config.max_centroid_offset_ratio:.3f} (thiếc lệch khỏi land)"
        )
        return ("insufficient", reasons)

    reasons.append(
        f"solder_ratio {features.solder_ratio:.3f}, span {features.span_ratio:.3f}, "
        f"specular {features.specular_ratio:.3f} — trong ngưỡng"
    )
    return ("good", reasons)


def grade_component_by_rules(
    features: SolderFeatures,
    config: SolderGradingConfig,
) -> tuple[str, list[str]]:
    """Classify one component-scope ROI.

    Far weaker than the joint rules, and honestly so: telling a tombstone from
    a correctly seated part needs the two terminals compared against each other,
    which happens in the inspector where both are in hand. Here only the
    unmistakable case is claimed, and everything else defers.
    """

    reasons: list[str] = []
    if features.solder_ratio <= config.missing_component_ratio:
        reasons.append(
            f"solder_ratio {features.solder_ratio:.3f} <= "
            f"{config.missing_component_ratio:.3f} (không thấy kim loại nào ở vị trí này)"
        )
        return ("missing", reasons)
    reasons.append("không có bằng chứng hình học nào đủ chắc ở mức linh kiện")
    return ("ok", reasons)


def mark_bridges(
    joints: Sequence[SolderJoint],
    features: Sequence[SolderFeatures],
    labels: list[str],
    reasons: list[list[str]],
    config: SolderGradingConfig,
) -> None:
    """Upgrade adjacent pin pairs to ``bridge`` in place.

    A bridge is the one joint defect that does not live inside a single ROI:
    it is solder crossing from one pin cell into the next. Both sides of the
    shared boundary have to be covered, which is why this runs over pairs
    rather than in :func:`grade_joint_by_rules`.
    """

    ordered: dict[tuple[str, str], list[int]] = {}
    for index, joint in enumerate(joints):
        if joint.kind != "joint" or joint.pin_index is None:
            continue
        band = joint.position.rsplit("_pin", 1)[0]
        ordered.setdefault((joint.detection_id, band), []).append(index)

    for indices in ordered.values():
        indices.sort(key=lambda index: joints[index].pin_index or 0)
        for first, second in zip(indices, indices[1:]):
            if (joints[second].pin_index or 0) - (joints[first].pin_index or 0) != 1:
                continue
            touching = (
                features[first].edge_contact_end >= config.bridge_edge_contact
                and features[second].edge_contact_start >= config.bridge_edge_contact
            )
            if not touching:
                continue
            if (
                features[first].span_ratio < config.bridge_span_ratio
                or features[second].span_ratio < config.bridge_span_ratio
            ):
                continue
            note = (
                f"thiếc phủ kín biên chung giữa pin {joints[first].pin}"
                f" và pin {joints[second].pin} "
                f"(edge {features[first].edge_contact_end:.2f}/"
                f"{features[second].edge_contact_start:.2f})"
            )
            for index in (first, second):
                labels[index] = "bridge"
                reasons[index] = [note]
