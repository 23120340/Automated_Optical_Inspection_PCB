"""Layer C of step 6.2: fuse the measured rules with the trained model.

Neither layer is trusted alone.

The rules know physics but not appearance: they can tell that four percent of a
land is covered, but not that an adequately-covered joint has the dull rounded
look of one that never wetted. The model knows appearance but has no notion of
what it is looking at; it will confidently call a ROI good when the ROI landed
on bare board because the registration drifted.

So the two vote, and the tie-breaks are asymmetric on purpose. An escape ships
a defective board to a customer; a false call costs an operator ten seconds at
a review station. Wherever the two layers disagree the ROI goes to review
rather than to whichever layer sounds more confident, and a hard physical floor
sits under the model so that no confidence value can pass a land with almost no
solder on it.

With no model configured this degrades to the rules alone and reports
``source="rules"``, so the stage is useful before any training has happened.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import SolderGradingConfig
from ..models import (
    ClassProbability,
    SolderFeatures,
    SolderJointCrop,
    SolderVerdict,
)
from .classifier import ONNXSolderClassifier, create_solder_classifier
from .features import measure_solder
from .rules import (
    COMPONENT_CLASSES,
    JOINT_CLASSES,
    grade_component_by_rules,
    grade_joint_by_rules,
    mark_bridges,
)

__all__ = ["SolderInspector"]

_GOOD_LABELS = {"good", "ok"}


class SolderInspector:
    """Grade every step-5.5 ROI: measure, judge by rule, and fuse with a model."""

    def __init__(
        self,
        config: SolderGradingConfig | None = None,
        classifier: ONNXSolderClassifier | None = None,
        component_classifier: ONNXSolderClassifier | None = None,
    ) -> None:
        self.config = config or SolderGradingConfig()
        self.classifier = classifier
        self.component_classifier = component_classifier
        self.warnings: list[str] = []
        if classifier is None and component_classifier is None:
            built = create_solder_classifier(
                self.config.model_path, self.config.manifest_path, self.config
            )
            if built is not None:
                if built.scope == "component":
                    self.component_classifier = built
                else:
                    self.classifier = built

    @property
    def has_model(self) -> bool:
        return self.classifier is not None or self.component_classifier is not None

    def inspect(
        self,
        crops: Sequence[SolderJointCrop],
        image: np.ndarray | None = None,
    ) -> list[SolderVerdict]:
        """Grade every ROI.

        ``image`` is the analysis frame the ROIs were cut from. Pass it: the
        crops stored on each :class:`SolderJointCrop` are letterboxed to a
        square for the network, and the grey padding counts into every area
        ratio measured on them. On an elongated lead band that padding can be
        half the pixels, which halves ``solder_ratio`` and drags good joints
        below the insufficient threshold. Measuring from the frame uses the
        ROI's own pixels and nothing else.
        """

        if not self.config.enabled or not crops:
            return []

        # Per board, not per session: a warning left over from the previous
        # board would be read as a problem with this one.
        self.warnings = []

        joints = [crop.joint for crop in crops]
        features = [
            measure_solder(
                self._roi_pixels(crop, image),
                along_axis=_axis_for(crop),
                saturation_max=self.config.saturation_max,
                specular_percentile=self.config.specular_percentile,
            )
            for crop in crops
        ]

        rule_labels: list[str] = []
        rule_reasons: list[list[str]] = []
        for crop, measurement in zip(crops, features):
            if crop.joint.kind == "body":
                label, reasons = grade_component_by_rules(measurement, self.config)
            else:
                label, reasons = grade_joint_by_rules(measurement, self.config)
            rule_labels.append(label)
            rule_reasons.append(reasons)

        # Bridges span two ROIs, so they can only be seen with the neighbours
        # in hand.
        mark_bridges(joints, features, rule_labels, rule_reasons, self.config)
        self._mark_tombstones(crops, features, rule_labels, rule_reasons)

        joint_indices = [i for i, crop in enumerate(crops) if crop.joint.kind != "body"]
        body_indices = [i for i, crop in enumerate(crops) if crop.joint.kind == "body"]
        predictions: dict[int, list[ClassProbability]] = {}
        predictions.update(
            self._predict(self.classifier, crops, joint_indices)
        )
        predictions.update(
            self._predict(self.component_classifier, crops, body_indices)
        )

        verdicts: list[SolderVerdict] = []
        for index, crop in enumerate(crops):
            verdicts.append(
                self._fuse(
                    crop,
                    features[index],
                    rule_labels[index],
                    rule_reasons[index],
                    predictions.get(index),
                )
            )
        return verdicts

    # ------------------------------------------------------------------ #

    @staticmethod
    def _roi_pixels(
        crop: SolderJointCrop, image: np.ndarray | None
    ) -> np.ndarray:
        """The ROI's own pixels, without the letterbox padding."""

        if image is None:
            return crop.image
        height, width = image.shape[:2]
        x1, y1, x2, y2 = crop.joint.bbox.clamp(width, height).to_int()
        region = image[y1:y2, x1:x2]
        # Fall back rather than measuring an empty slice if the ROI somehow
        # falls outside the frame it was supposedly cut from.
        return region if region.size else crop.image

    def _predict(
        self,
        classifier: ONNXSolderClassifier | None,
        crops: Sequence[SolderJointCrop],
        indices: Sequence[int],
    ) -> dict[int, list[ClassProbability]]:
        if classifier is None or not indices:
            return {}
        try:
            outputs = classifier.predict([crops[index].image for index in indices])
        except Exception as exc:  # noqa: BLE001 - reported, never silently dropped
            # A model failure must not delete the rule verdicts: rules-only is a
            # supported mode, so degrade to it and say so.
            self.warnings.append(
                f"Model 6.2 lỗi, quay về chấm bằng luật: {type(exc).__name__}: {exc}"
            )
            return {}
        return dict(zip(indices, outputs))

    def _fuse(
        self,
        crop: SolderJointCrop,
        features: SolderFeatures,
        rule_label: str,
        rule_reasons: Sequence[str],
        prediction: Sequence[ClassProbability] | None,
    ) -> SolderVerdict:
        joint = crop.joint
        scope = "component" if joint.kind == "body" else "joint"
        classifier = self.component_classifier if scope == "component" else self.classifier
        reasons = list(rule_reasons)

        base = dict(
            joint_id=joint.joint_id,
            detection_id=joint.detection_id,
            scope=scope,
            designator=joint.designator,
            pin=joint.pin,
            component_label=joint.label,
            rule_label=rule_label,
            features=features,
            metadata={
                "roi_kind": joint.kind,
                "position": joint.position,
                "roi_source": joint.source,
                # Carried so overlays and review stations can draw the verdict
                # without having to re-join against the ROI list.
                "bbox": joint.bbox.as_xyxy(),
                "roi_width_px": int(round(joint.bbox.width)),
                "roi_height_px": int(round(joint.bbox.height)),
            },
        )

        if prediction is None or classifier is None:
            decision = self._rules_only_decision(rule_label)
            return SolderVerdict(
                label=rule_label,
                probability=1.0 if rule_label in _GOOD_LABELS else 0.0,
                decision=decision,
                source="rules",
                reasons=reasons,
                model_version="rules-only",
                **base,
            )

        top = prediction[0]
        model_label = top.label
        model_probability = top.probability
        accept_threshold = classifier.accept_threshold_for(model_label)
        confident = model_probability >= max(
            accept_threshold, self.config.model_accept_probability
        )
        agree = _same_verdict(model_label, rule_label)

        if agree:
            label = model_label
            decision = "accept" if confident else "review"
            source = "model+rules"
            reasons.append(
                f"model đồng ý với luật: {model_label} {model_probability:.2f}"
            )
        elif self.config.disagreement_is_review:
            # Neither layer is reliable enough to overrule the other, and the
            # disagreement itself is the useful signal.
            label = model_label if model_label not in _GOOD_LABELS else rule_label
            decision = "review"
            source = "conflict"
            reasons.append(
                f"model nói {model_label} {model_probability:.2f} nhưng luật nói "
                f"{rule_label} — đưa vào hàng đợi kiểm tra"
            )
        else:
            label = model_label
            decision = "accept" if confident else "review"
            source = "model"
            reasons.append(
                f"model ghi đè luật ({rule_label}): {model_label} {model_probability:.2f}"
            )

        if not confident and decision == "accept":
            decision = "review"

        # The escape guard runs last so nothing above can undo it.
        if (
            self.config.escape_guard_enabled
            and scope == "joint"
            and label in _GOOD_LABELS
            and features.solder_ratio < self.config.escape_guard_solder_ratio
        ):
            label = "insufficient"
            decision = "review"
            source = "escape_guard"
            reasons.append(
                f"chốt chặn: solder_ratio {features.solder_ratio:.3f} < "
                f"{self.config.escape_guard_solder_ratio:.3f}, không cho phép kết luận đạt"
            )

        return SolderVerdict(
            label=label,
            probability=float(model_probability),
            decision=decision,
            source=source,
            model_label=model_label,
            model_probability=float(model_probability),
            top_k=list(prediction[:3]),
            reasons=reasons,
            model_version=classifier.model_version,
            **base,
        )

    def _rules_only_decision(self, rule_label: str) -> str:
        if rule_label in _GOOD_LABELS:
            return "accept"
        return (
            "reject"
            if self.config.rules_only_defect_decision == "reject"
            else "review"
        )

    def _mark_tombstones(
        self,
        crops: Sequence[SolderJointCrop],
        features: Sequence[SolderFeatures],
        labels: list[str],
        reasons: list[list[str]],
    ) -> None:
        """Flag two-terminal parts whose two ends look nothing alike.

        A tombstoned chip has one terminal soldered and one lifted clear of its
        land. Neither ROI is odd on its own -- one looks fine, the other looks
        like a missing joint -- so the evidence only exists in the comparison.
        """

        grouped: dict[str, list[int]] = {}
        for index, crop in enumerate(crops):
            joint = crop.joint
            if joint.kind == "joint" and joint.terminal_geometry == "two_terminal":
                grouped.setdefault(joint.detection_id, []).append(index)

        for indices in grouped.values():
            if len(indices) != 2:
                continue
            first, second = indices
            ratios = (features[first].solder_ratio, features[second].solder_ratio)
            weak, strong = min(ratios), max(ratios)
            if strong < self.config.insufficient_solder_ratio:
                continue  # both ends are poor: that is not a tombstone
            if weak > self.config.missing_solder_ratio:
                continue
            note = (
                f"hai đầu chênh lệch mạnh (solder_ratio {ratios[0]:.3f} vs "
                f"{ratios[1]:.3f}) — nghi dựng bia/hở một đầu"
            )
            index = first if ratios[0] < ratios[1] else second
            labels[index] = "missing_solder"
            reasons[index] = [note]


def _axis_for(crop: SolderJointCrop) -> str:
    """Which image axis the joint runs along, from its own geometry."""

    joint = crop.joint
    if joint.kind == "body":
        return "auto"
    # Terminal ROIs of a two-lead part run across the component axis, so the
    # ROI's own longer side is the right reference.
    return "auto" if joint.bbox.width >= joint.bbox.height else "y"


def _same_verdict(model_label: str, rule_label: str) -> bool:
    if model_label == rule_label:
        return True
    # Both layers agreeing there is no defect counts as agreement even when the
    # two taxonomies spell it differently ("good" vs "ok").
    return model_label in _GOOD_LABELS and rule_label in _GOOD_LABELS
