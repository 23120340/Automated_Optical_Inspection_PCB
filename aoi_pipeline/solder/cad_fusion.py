"""Merge registered CAD geometry with detector-derived solder ROIs.

Neither source is complete on its own:

* CAD knows where every land is and what the part should be, but not where the
  board actually sits under the camera, and it says nothing about parts that
  were placed wrong or not at all.
* The detector knows what is really on this board right now, but its box stops
  at the component body and its class is only a guess.

So the two are combined rather than one chosen. CAD supplies the land geometry
and the true terminal count; the detector supplies a per-component position
correction that turns a globally-approximate registration into locally-accurate
ROIs, and covers every part CAD does not list. Disagreements between them are
not discarded -- a CAD part with no detection is a *missing component* finding,
a detection far from its CAD placement is a *shifted component* finding, and
both are exactly the kind of defect step 6.2 exists to grade.

With no CAD loaded, :func:`fuse_solder_joints` returns the derived ROIs
unchanged, so this module is inert until a board file is dropped in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .cad import BoardCad, CadComponent, CadRegistration, classes_agree, is_informative_label
from ..config import FusionConfig, terminal_geometry
from ..models import BoundingBox, Detection, SolderJoint, intersection_over_smaller
from ..placement.footprints import parse_footprint, profile_for_package_class
from .geometry import ComponentFrame, derive_solder_joints

__all__ = [
    "CadFinding",
    "FusionResult",
    "fuse_solder_joints",
]


@dataclass(slots=True)
class CadFinding:
    """A board-level disagreement between CAD and what the camera saw."""

    kind: str
    severity: str
    message: str
    designator: str | None = None
    detection_id: str | None = None
    expected_class: str | None = None
    observed_class: str | None = None
    bbox: BoundingBox | None = None
    shift_mm: float | None = None

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
            "shift_mm": None if self.shift_mm is None else float(self.shift_mm),
        }


@dataclass(slots=True)
class FusionResult:
    """ROIs after fusion, plus what the comparison revealed."""

    joints: list[SolderJoint] = field(default_factory=list)
    findings: list[CadFinding] = field(default_factory=list)
    registration: CadRegistration | None = None
    used_cad: bool = False
    stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "used_cad": self.used_cad,
            "registration": (
                self.registration.to_dict() if self.registration is not None else None
            ),
            "findings": [finding.to_dict() for finding in self.findings],
            "stats": self.stats,
            "warnings": self.warnings,
        }


def fuse_solder_joints(
    detections: Sequence[Detection],
    derived_joints: Sequence[SolderJoint],
    image_width: int,
    image_height: int,
    board: BoardCad | None = None,
    registration: CadRegistration | None = None,
    config: FusionConfig | None = None,
    image: np.ndarray | None = None,
) -> FusionResult:
    """Combine CAD lands with derived ROIs into one inspection set."""

    config = config or FusionConfig()
    joints = list(derived_joints)

    if board is None or registration is None or not config.enabled:
        return FusionResult(joints=joints, registration=registration, used_cad=False)

    if not registration.is_usable(config.min_inlier_ratio, config.max_residual_px):
        # Applying a bad registration is worse than not applying it: the ROIs
        # would look just as confident while sitting on bare board.
        return FusionResult(
            joints=joints,
            registration=registration,
            used_cad=False,
            warnings=[
                "CAD registration rejected "
                f"(inlier_ratio={registration.inlier_ratio:.2f}, "
                f"residual={registration.residual_px:.1f}px); "
                "using detector-derived ROIs only."
            ],
        )

    scale = registration.scale_px_per_mm
    if scale <= 0.0:
        return FusionResult(
            joints=joints,
            registration=registration,
            used_cad=False,
            warnings=["CAD registration has no usable scale; ROIs stay derived-only."],
        )

    joints_by_detection: dict[str, list[SolderJoint]] = {}
    for joint in derived_joints:
        joints_by_detection.setdefault(joint.detection_id, []).append(joint)

    projected = {
        component.designator: registration.to_image([[component.x, component.y]])[0]
        for component in board.components
    }
    pairs = _match_components(board, detections, projected, scale, config)

    matched_detections = {detection_id for _, detection_id in pairs}
    matched_designators = {designator for designator, _ in pairs}
    detections_by_id = {detection.detection_id: detection for detection in detections}
    components_by_designator = {
        component.designator: component for component in board.components
    }

    fused: list[SolderJoint] = []
    findings: list[CadFinding] = []
    counters = {"cad": 0, "cad+derived": 0, "derived": 0, "cad_only_component": 0}

    for designator, detection_id in pairs:
        component = components_by_designator[designator]
        detection = detections_by_id[detection_id]
        package_profile = _component_package_profile(component)
        if package_profile is not None:
            # BOM/PnP/CAD footprint evidence outranks an image-model package
            # prediction.  Keep it on the detection as well as on joints so a
            # hidden-terminal package remains auditable even though it emits
            # no top-down solder ROI.
            detection.metadata["package_profile"] = package_profile
            detection.metadata["terminal_geometry_override"] = str(
                package_profile["terminal_geometry"]
            )
            detection.metadata["package_axis_known"] = True
            # Cạnh chân do LUẬT ẢNH đo được phải nhường chỗ: CAD mang cả hình
            # học lẫn hướng, nên giữ lại cạnh cũ là trộn hai nguồn. Trong một
            # lượt chạy thì ``derive`` đã xong trước đây, nhưng detection bị
            # sửa TẠI CHỖ và ``last_package_detections`` giữ chính object này —
            # gọi ``make_solder_crops`` lần nữa trên chúng sẽ dính.
            detection.metadata.pop("terminal_lead_edges", None)
            detection.metadata.pop("terminal_lead_edges_space", None)
        own_derived = joints_by_detection.get(detection_id, [])
        offset = _placement_offset(detection, projected[designator])
        shift_mm = float(np.linalg.norm(offset)) / scale

        if shift_mm > config.max_shift_mm:
            findings.append(
                CadFinding(
                    kind="shifted_component",
                    severity="review",
                    message=(
                        f"{designator} sits {shift_mm:.2f} mm from its CAD placement "
                        f"(limit {config.max_shift_mm:.2f} mm)."
                    ),
                    designator=designator,
                    detection_id=detection_id,
                    expected_class=component.part_class,
                    observed_class=detection.label,
                    bbox=detection.bbox,
                    shift_mm=shift_mm,
                )
            )
        if not classes_agree(component.part_class, detection.label):
            findings.append(
                CadFinding(
                    kind="class_mismatch",
                    severity="info",
                    message=(
                        f"{designator}: CAD expects {component.part_class}, detector "
                        f"read {detection.label}."
                    ),
                    designator=designator,
                    detection_id=detection_id,
                    expected_class=component.part_class,
                    observed_class=detection.label,
                    bbox=detection.bbox,
                )
            )

        # Local correction: trust CAD for the shape of the footprint and the
        # detector for where this particular part actually landed.
        correction = offset if config.local_refine else np.zeros(2)
        if float(np.linalg.norm(correction)) > config.max_local_shift_mm * scale:
            correction = np.zeros(2)

        component_joints = _joints_for_matched_component(
            component,
            detection,
            own_derived,
            registration,
            correction,
            scale,
            image_width,
            image_height,
            config,
            image,
        )
        for joint in component_joints:
            counters[joint.source] = counters.get(joint.source, 0) + 1
        fused.extend(component_joints)

    # CAD parts nobody saw: still worth a ROI, because "is anything soldered
    # here" is a question step 6.2 can answer from the land alone.
    for component in board.components:
        if component.designator in matched_designators:
            continue
        findings.append(
            CadFinding(
                kind="missing_component",
                severity="defect",
                message=(
                    f"{component.designator} is in CAD but no component was detected "
                    "at its placement."
                ),
                designator=component.designator,
                expected_class=component.part_class,
                bbox=_component_bbox(component, registration, scale, config, image_width, image_height),
            )
        )
        if config.emit_cad_only_rois and component.has_pads:
            cad_joints = _cad_pad_joints(
                component,
                registration,
                np.zeros(2),
                scale,
                image_width,
                image_height,
                config,
                detection_id="",
                label=component.part_class or "unknown",
            )
            counters["cad_only_component"] += len(cad_joints)
            fused.extend(cad_joints)

    # Detections CAD does not list: keep their derived ROIs untouched. A partial
    # CAD file is common, so this is an observation, not a verdict.
    for detection in detections:
        if detection.detection_id in matched_detections:
            continue
        own = joints_by_detection.get(detection.detection_id, [])
        counters["derived"] += len(own)
        fused.extend(own)
        if config.report_unexpected:
            findings.append(
                CadFinding(
                    kind="unexpected_component",
                    severity="info",
                    message=(
                        f"Detected {detection.label} at "
                        f"({detection.bbox.x1:.0f}, {detection.bbox.y1:.0f}) with no "
                        "matching CAD placement."
                    ),
                    detection_id=detection.detection_id,
                    observed_class=detection.label,
                    bbox=detection.bbox,
                )
            )

    warnings: list[str] = []
    comparable = [
        (designator, detection_id)
        for designator, detection_id in pairs
        if is_informative_label(components_by_designator[designator].part_class)
        and is_informative_label(detections_by_id[detection_id].label)
    ]
    agreeing = sum(
        1
        for designator, detection_id in comparable
        if classes_agree(
            components_by_designator[designator].part_class,
            detections_by_id[detection_id].label,
        )
    )
    if registration.ambiguous:
        warnings.append(
            "CAD registration is ambiguous: a different alignment fits this board "
            "equally well. Confirm the ROI overlay, or pin it down with fiducials "
            "or a saved registration matrix."
        )
    if len(pairs) >= 3 and not comparable:
        warnings.append(
            "No component classes were informative enough to corroborate the CAD "
            "registration; it rests on geometry alone. Load a trained detector or "
            "supply fiducials before trusting these ROIs."
        )
    if len(comparable) >= 3 and agreeing == 0:
        # Distance alone cannot tell a board from its mirror image, and a
        # mirrored fit pairs every part with the wrong one while still scoring a
        # small residual. Zero class agreement across many pairs is that
        # signature.
        warnings.append(
            f"CAD registration matched {len(pairs)} parts but none of the "
            f"{len(comparable)} comparable classes agree; the fit may be mirrored "
            "or misaligned. Check the ROI overlay before using these crops."
        )

    stats = {
        "cad_components": len(board.components),
        "detections": len(detections),
        "matched": len(pairs),
        "missing": sum(1 for f in findings if f.kind == "missing_component"),
        "unexpected": sum(1 for f in findings if f.kind == "unexpected_component"),
        "shifted": sum(1 for f in findings if f.kind == "shifted_component"),
        "class_mismatch": sum(1 for f in findings if f.kind == "class_mismatch"),
        "class_agreements": agreeing,
        "class_comparable": len(comparable),
        "roi_sources": counters,
        "scale_px_per_mm": scale,
    }
    return FusionResult(
        joints=fused,
        findings=findings,
        registration=registration,
        used_cad=True,
        stats=stats,
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #


def _match_components(
    board: BoardCad,
    detections: Sequence[Detection],
    projected: dict[str, np.ndarray],
    scale: float,
    config: FusionConfig,
) -> list[tuple[str, str]]:
    """Greedy nearest pairing of CAD placements to detections."""

    if not detections or not board.components:
        return []
    tolerance_px = config.match_tolerance_mm * scale
    candidates: list[tuple[float, str, str]] = []
    for component in board.components:
        centre = projected[component.designator]
        for detection in detections:
            detection_centre = np.array(
                [
                    (detection.bbox.x1 + detection.bbox.x2) / 2.0,
                    (detection.bbox.y1 + detection.bbox.y2) / 2.0,
                ]
            )
            distance = float(np.linalg.norm(centre - detection_centre))
            if distance > tolerance_px:
                continue
            # Class agreement breaks ties between two equally close parts; it
            # never rules a pair out, because both classes are estimates.
            if not classes_agree(component.part_class, detection.label):
                distance += tolerance_px * config.class_mismatch_penalty
            candidates.append((distance, component.designator, detection.detection_id))

    candidates.sort()
    used_designators: set[str] = set()
    used_detections: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for _, designator, detection_id in candidates:
        if designator in used_designators or detection_id in used_detections:
            continue
        used_designators.add(designator)
        used_detections.add(detection_id)
        pairs.append((designator, detection_id))
    return pairs


def _placement_offset(detection: Detection, projected_centre: np.ndarray) -> np.ndarray:
    centre = np.array(
        [
            (detection.bbox.x1 + detection.bbox.x2) / 2.0,
            (detection.bbox.y1 + detection.bbox.y2) / 2.0,
        ]
    )
    return centre - projected_centre


# --------------------------------------------------------------------------- #
# ROI construction
# --------------------------------------------------------------------------- #


def _joints_for_matched_component(
    component: CadComponent,
    detection: Detection,
    derived: Sequence[SolderJoint],
    registration: CadRegistration,
    correction: np.ndarray,
    scale: float,
    image_width: int,
    image_height: int,
    config: FusionConfig,
    image: np.ndarray | None,
) -> list[SolderJoint]:
    """Build one component's ROIs from whichever evidence is strongest."""

    footprint_profile = parse_footprint(component.footprint)
    if (
        footprint_profile is not None
        and footprint_profile.terminal_geometry == "hidden_terminals"
    ):
        # The CAD pads are real design objects, but on an assembled QFN/BGA/LGA
        # they are underneath the body and invisible to a top-down camera.
        return []

    if component.has_pads:
        cad_joints = _cad_pad_joints(
            component,
            registration,
            correction,
            scale,
            image_width,
            image_height,
            config,
            detection_id=detection.detection_id,
            label=detection.label,
        )
        if cad_joints:
            return _reconcile(cad_joints, derived, config, component.designator)

    # Placement-only CAD (a pick-and-place file). The lands are unknown, but the
    # true centre and rotation are not, so re-derive the same ROI geometry on
    # the CAD frame instead of on the detector's axis-aligned box.
    return _reanchored_derived_joints(
        component,
        detection,
        derived,
        registration,
        correction,
        scale,
        image_width,
        image_height,
        config,
        image,
    )


def _cad_pad_joints(
    component: CadComponent,
    registration: CadRegistration,
    correction: np.ndarray,
    scale: float,
    image_width: int,
    image_height: int,
    config: FusionConfig,
    detection_id: str,
    label: str,
) -> list[SolderJoint]:
    """One ROI per CAD land, grown to include the fillet outside it."""

    if not component.pads:
        return []
    centres = registration.to_image([[pad.x, pad.y] for pad in component.pads])
    centres = centres + correction
    default_size = _default_pad_size_px(component, scale, config)
    board_angle = _registration_angle(registration)
    package_profile = _component_package_profile(component)

    joints: list[SolderJoint] = []
    for index, (pad, centre) in enumerate(zip(component.pads, centres)):
        width_px = pad.width * scale if pad.width > 0 else default_size
        height_px = pad.height * scale if pad.height > 0 else default_size
        margin = 1.0 + 2.0 * config.pad_margin_ratio
        bbox = _rotated_bbox(
            centre,
            (width_px * margin, height_px * margin),
            board_angle + pad.rotation,
            image_width,
            image_height,
        )
        if bbox is None or min(bbox.width, bbox.height) < config.min_roi_pixels:
            continue
        joints.append(
            SolderJoint(
                detection_id=detection_id,
                joint_id=f"{component.designator}_pin{pad.pin or index + 1}",
                label=label,
                kind="joint",
                bbox=bbox,
                terminal_geometry="cad_pad",
                position=f"pin{pad.pin or index + 1}",
                angle=board_angle + pad.rotation,
                pin_index=index,
                source="cad",
                designator=component.designator,
                pin=str(pad.pin or index + 1),
                net=pad.net,
                metadata={
                    "cad_pad": pad.to_dict(),
                    "local_correction_px": [float(correction[0]), float(correction[1])],
                    "pad_size_known": pad.width > 0 and pad.height > 0,
                    **(
                        {"package_profile": dict(package_profile)}
                        if package_profile is not None
                        else {}
                    ),
                },
            )
        )
    if joints and config.include_body_view:
        joints.append(_body_from(joints, component, detection_id, label))
    return joints


def _reanchored_derived_joints(
    component: CadComponent,
    detection: Detection,
    derived: Sequence[SolderJoint],
    registration: CadRegistration,
    correction: np.ndarray,
    scale: float,
    image_width: int,
    image_height: int,
    config: FusionConfig,
    image: np.ndarray | None,
) -> list[SolderJoint]:
    """Re-run the derived geometry on the CAD placement instead of the box."""

    if not config.reanchor_on_placement:
        profile = _component_package_profile(component)
        if profile is not None:
            geometry = str(profile["terminal_geometry"])
            rebuilt = derive_solder_joints(
                detection,
                image_width,
                image_height,
                config.solder,
                image,
                geometry=geometry,
            )
            if rebuilt or geometry in {"hidden_terminals", "ic_khong_chan"}:
                derived = rebuilt
        return [
            _tagged_with_profile(joint, "derived", component.designator, profile)
            for joint in derived
        ]

    centre = registration.to_image([[component.x, component.y]])[0] + correction
    angle = _registration_angle(registration) + component.rotation
    length = max(detection.bbox.width, detection.bbox.height)
    span = min(detection.bbox.width, detection.bbox.height)
    # A pad span from CAD is a real measurement of the part's length; the
    # detector box is only the body, so prefer CAD when it is available.
    pad_span_px = component.pad_span_mm() * scale
    if pad_span_px > length:
        length = pad_span_px

    frame = ComponentFrame(
        center_x=float(centre[0]),
        center_y=float(centre[1]),
        angle=float(angle),
        length=float(length),
        span=float(span),
    )
    geometry = _geometry_from_cad(component, detection)
    joints = derive_solder_joints(
        detection,
        image_width,
        image_height,
        config.solder,
        image,
        frame=frame,
        geometry=geometry,
        # Góc xoay ở đây đến từ file đặt linh kiện, nên trục đầu cực là **đã
        # biết**, không phải phải đoán. Đây là chỗ một file pick-and-place --
        # không cần toạ độ pad -- cứu được ca tụ hoá can tròn: hộp của nó gần
        # vuông (đo được 148x136, tỉ lệ 1,09) nên bộ giải trục từ hộp không
        # chốt được và phát cả hai cặp ROI, trong đó một cặp nằm trên bo trống.
        axis_known=True,
    )
    if not joints and geometry not in {"hidden_terminals", "ic_khong_chan"}:
        return [_tagged(joint, "derived", component.designator) for joint in derived]
    return [
        _tagged(joint, "cad+derived", component.designator) for joint in joints
    ]


def _geometry_from_cad(component: CadComponent, detection: Detection) -> str:
    """Terminal topology from the real pad count when CAD knows it."""

    if component.pads:
        return "two_terminal" if len(component.pads) == 2 else "multi_pin"
    footprint_profile = parse_footprint(component.footprint)
    if footprint_profile is not None:
        return footprint_profile.terminal_geometry
    detection_profile = detection.metadata.get("package_profile")
    if isinstance(detection_profile, dict):
        geometry = str(detection_profile.get("terminal_geometry") or "").strip()
        if geometry:
            return geometry
        package_class = str(detection_profile.get("package_class") or "").strip()
        if package_class:
            return terminal_geometry(detection.label, package=package_class)
    if component.part_class:
        return terminal_geometry(component.part_class)
    return terminal_geometry(detection.label)


def _component_package_profile(component: CadComponent) -> dict[str, Any] | None:
    """Return topology evidence carried by a CAD/BOM/PnP component row."""

    profile = parse_footprint(component.footprint)
    if profile is None and len(component.pads) == 2:
        profile = profile_for_package_class("hai_chan", source="cad_pad_count")
    if profile is None:
        return None
    payload = profile.to_dict()
    payload.update(
        {
            "source": "cad_pads" if component.pads else "footprint",
            "designator": component.designator,
            "footprint": component.footprint,
        }
    )
    if component.pads:
        footprint_count = payload.get("expected_pin_count")
        if footprint_count is not None and footprint_count != len(component.pads):
            payload["footprint_expected_pin_count"] = footprint_count
        payload["expected_pin_count"] = len(component.pads)
        payload["expected_pin_count_range"] = None
    return payload


def _reconcile(
    cad_joints: Sequence[SolderJoint],
    derived: Sequence[SolderJoint],
    config: FusionConfig,
    designator: str,
) -> list[SolderJoint]:
    """Mark CAD ROIs the detector independently agreed with, and keep extras.

    A derived ROI with no CAD land under it is kept rather than dropped: CAD
    exports routinely omit thermal pads, shields and mechanical lands, and
    losing those silently would leave joints uninspected.
    """

    derived_joints = [joint for joint in derived if joint.kind == "joint"]
    used: set[int] = set()
    result: list[SolderJoint] = []
    for joint in cad_joints:
        if joint.kind != "joint":
            result.append(joint)
            continue
        best_index, best_overlap = -1, 0.0
        for index, candidate in enumerate(derived_joints):
            if index in used:
                continue
            overlap = _intersection_over_smaller(joint.bbox, candidate.bbox)
            if overlap > best_overlap:
                best_index, best_overlap = index, overlap
        if best_index >= 0 and best_overlap >= config.agreement_ios:
            used.add(best_index)
            merged = _merge_boxes(joint.bbox, derived_joints[best_index].bbox, config)
            joint = _replace_joint(joint, bbox=merged, source="cad+derived")
        result.append(joint)

    if config.keep_unmatched_derived:
        for index, candidate in enumerate(derived_joints):
            if index in used:
                continue
            result.append(_tagged(candidate, "derived", designator))
    return result


def _merge_boxes(
    cad_box: BoundingBox, derived_box: BoundingBox, config: FusionConfig
) -> BoundingBox:
    if config.merge_mode == "cad":
        return cad_box
    if config.merge_mode == "union":
        return BoundingBox(
            min(cad_box.x1, derived_box.x1),
            min(cad_box.y1, derived_box.y1),
            max(cad_box.x2, derived_box.x2),
            max(cad_box.y2, derived_box.y2),
        )
    return BoundingBox(
        max(cad_box.x1, derived_box.x1),
        max(cad_box.y1, derived_box.y1),
        min(cad_box.x2, derived_box.x2),
        min(cad_box.y2, derived_box.y2),
    )


_intersection_over_smaller = intersection_over_smaller


def _default_pad_size_px(
    component: CadComponent, scale: float, config: FusionConfig
) -> float:
    """Land size for formats that give positions but not pad dimensions."""

    if len(component.pads) >= 2:
        points = np.array([[pad.x, pad.y] for pad in component.pads])
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        pitch_mm = float(distances.min())
        if math.isfinite(pitch_mm) and pitch_mm > 0:
            return max(config.min_roi_pixels, pitch_mm * scale * config.pitch_pad_ratio)
    return max(config.min_roi_pixels, config.fallback_pad_mm * scale)


def _registration_angle(registration: CadRegistration) -> float:
    linear = registration.matrix[:2, :2]
    return float(math.degrees(math.atan2(linear[1, 0], linear[0, 0])))


def _rotated_bbox(
    centre: Sequence[float],
    size: tuple[float, float],
    angle_deg: float,
    image_width: int,
    image_height: int,
) -> BoundingBox | None:
    half_width, half_height = size[0] / 2.0, size[1] / 2.0
    corners = np.array(
        [
            [-half_width, -half_height],
            [half_width, -half_height],
            [half_width, half_height],
            [-half_width, half_height],
        ],
        dtype=np.float64,
    )
    radians = math.radians(angle_deg)
    cos_a, sin_a = math.cos(radians), math.sin(radians)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    corners = corners @ rotation.T
    corners[:, 0] += centre[0]
    corners[:, 1] += centre[1]
    bbox = BoundingBox(
        float(corners[:, 0].min()),
        float(corners[:, 1].min()),
        float(corners[:, 0].max()),
        float(corners[:, 1].max()),
    ).clamp(image_width, image_height)
    return bbox if bbox.width > 0 and bbox.height > 0 else None


def _component_bbox(
    component: CadComponent,
    registration: CadRegistration,
    scale: float,
    config: FusionConfig,
    image_width: int,
    image_height: int,
) -> BoundingBox | None:
    points = [[pad.x, pad.y] for pad in component.pads] or [[component.x, component.y]]
    projected = registration.to_image(points)
    margin = max(config.min_roi_pixels, config.fallback_pad_mm * scale)
    return BoundingBox(
        float(projected[:, 0].min()) - margin,
        float(projected[:, 1].min()) - margin,
        float(projected[:, 0].max()) + margin,
        float(projected[:, 1].max()) + margin,
    ).clamp(image_width, image_height)


def _body_from(
    joints: Sequence[SolderJoint],
    component: CadComponent,
    detection_id: str,
    label: str,
) -> SolderJoint:
    boxes = [joint.bbox for joint in joints]
    bbox = BoundingBox(
        min(box.x1 for box in boxes),
        min(box.y1 for box in boxes),
        max(box.x2 for box in boxes),
        max(box.y2 for box in boxes),
    )
    return SolderJoint(
        detection_id=detection_id,
        joint_id=f"{component.designator}_body",
        label=label,
        kind="body",
        bbox=bbox,
        terminal_geometry="cad_pad",
        position="body",
        source="cad",
        designator=component.designator,
        metadata={"pad_count": len(component.pads)},
    )


def _tagged(
    joint: SolderJoint, source: str, designator: str | None
) -> SolderJoint:
    return _replace_joint(joint, source=source, designator=designator or joint.designator)


def _tagged_with_profile(
    joint: SolderJoint,
    source: str,
    designator: str | None,
    profile: dict[str, Any] | None,
) -> SolderJoint:
    metadata = dict(joint.metadata)
    if profile is not None:
        metadata["package_profile"] = dict(profile)
    return _replace_joint(
        joint,
        source=source,
        designator=designator or joint.designator,
        metadata=metadata,
    )


def _replace_joint(
    joint: SolderJoint,
    bbox: BoundingBox | None = None,
    source: str | None = None,
    designator: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> SolderJoint:
    return SolderJoint(
        detection_id=joint.detection_id,
        joint_id=joint.joint_id,
        label=joint.label,
        kind=joint.kind,
        bbox=bbox if bbox is not None else joint.bbox,
        terminal_geometry=joint.terminal_geometry,
        position=joint.position,
        angle=joint.angle,
        pin_index=joint.pin_index,
        source=source if source is not None else joint.source,
        designator=designator if designator is not None else joint.designator,
        pin=joint.pin,
        net=joint.net,
        metadata=dict(joint.metadata) if metadata is None else dict(metadata),
    )
