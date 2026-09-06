"""Production inspection orchestration built on recipe-defined core services."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..imaging.alignment import PCBAligner, StrictAlignmentResult
from ..detection.detectors import ComponentDetector, CVComponentDetector, detector_identifier
from ..exceptions import InvalidImageError
from .compare import GoldenComparator, GoldenCompareResult
from ..imaging.image_io import ensure_bgr
from ..models import BoundingBox, Detection, utc_now_iso
from .position import PositionMeasurer, PositionResult
from .recipe import GOLDEN_COORDINATE_SPACE, InspectionRecipe, SlotRecipe


@dataclass(frozen=True, slots=True)
class InspectionConfig:
    """Board aggregation and runtime safety policy."""

    require_production_eligible: bool = True
    extras_are_ng: bool = True
    allow_class_mismatch_candidate: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("require_production_eligible", self.require_production_eligible),
            ("extras_are_ng", self.extras_are_ng),
            ("allow_class_mismatch_candidate", self.allow_class_mismatch_candidate),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True, slots=True)
class DetectionSummary:
    detection_id: str
    label: str
    class_id: int | None
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    source: str

    @classmethod
    def from_detection(cls, detection: Detection) -> DetectionSummary:
        return cls(
            detection_id=detection.detection_id,
            label=detection.label,
            class_id=detection.class_id,
            confidence=float(detection.confidence),
            bbox_xyxy=tuple(float(value) for value in detection.bbox.as_xyxy()),
            source=detection.source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "label": self.label,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "bbox_xyxy": list(self.bbox_xyxy),
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class SlotInspectionResult:
    slot_id: str
    candidate: DetectionSummary | None
    class_hint_match: bool | None
    position: PositionResult
    appearance: GoldenCompareResult
    status: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "candidate": None if self.candidate is None else self.candidate.to_dict(),
            "class_hint_match": self.class_hint_match,
            "position": self.position.to_dict(),
            "appearance": self.appearance.to_dict(),
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class InspectionRun:
    status: str
    reason: str
    alignment: StrictAlignmentResult
    slots: tuple[SlotInspectionResult, ...]
    extras: tuple[DetectionSummary, ...]
    recipe_schema_version: str
    recipe_sha256: str
    golden_sha256: str
    model_identifiers: Mapping[str, str]
    runtime_detector: str
    runtime_detector_identifier: str
    started_at: str = field(default_factory=utc_now_iso)
    coordinate_space: str = GOLDEN_COORDINATE_SPACE
    #: Ba cổng an toàn production có được thi hành trong lần chạy này không.
    #: ``InspectionConfig.require_production_eligible`` tắt được để chạy thử,
    #: nhưng trước đây bản ghi kết quả **không nói ra điều đó** — một JSON
    #: ``status: pass`` không cho biết nó chạy ở chế độ nào. Với một kho lịch sử
    #: kiểm tra thì đó là lỗ hổng ở ngay nguồn: không phân biệt được lần chạy
    #: thử với lần chạy thật.
    production_gates_enforced: bool = True
    #: Những cổng LẼ RA đã chặn lần chạy này, đánh giá **bất kể** cờ trên.
    #:
    #: Rỗng khi cổng bật (vì đã chặn thật rồi). Khi cổng tắt, nó phân biệt hai
    #: chuyện rất khác nhau: "chạy thử nhưng mọi thứ vẫn đạt chuẩn production"
    #: và "lần PASS này CHỈ đạt được nhờ tắt cổng". Không có trường này thì hai
    #: bản ghi đó giống hệt nhau.
    production_gate_findings: tuple[str, ...] = ()
    #: Bo va MAT mà lần chạy này thuộc về, chép thẳng từ recipe.
    #:
    #: Recipe đã có ``board_id``/``side`` và validate ``side in {top, bottom}``,
    #: nhưng bản ghi kết quả trước đây không mang chúng — muốn biết một lần chạy
    #: thuộc mặt nào thì phải tra ngược ``recipe_sha256`` về kho recipe. Dây
    #: chuyền kiểm **cả hai mặt**, nên câu "PCB này đã đủ điều kiện chưa" là câu
    #: hỏi trên NHIỀU lần chạy; bắt nó phụ thuộc kho recipe là làm nó vỡ ngay khi
    #: recipe bị dọn hoặc đổi phiên bản.
    board_id: str = ""
    side: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "started_at": self.started_at,
            "coordinate_space": self.coordinate_space,
            "recipe_schema_version": self.recipe_schema_version,
            "recipe_sha256": self.recipe_sha256,
            "golden_sha256": self.golden_sha256,
            "model_identifiers": dict(sorted(self.model_identifiers.items())),
            "runtime_detector": self.runtime_detector,
            "runtime_detector_identifier": self.runtime_detector_identifier,
            "board_id": self.board_id,
            "side": self.side,
            "production_gates_enforced": self.production_gates_enforced,
            "production_gate_findings": list(self.production_gate_findings),
            "alignment": self.alignment.to_dict(),
            "slots": [slot.to_dict() for slot in self.slots],
            "extras": [detection.to_dict() for detection in self.extras],
            "summary": {
                "slot_count": len(self.slots),
                "extra_count": len(self.extras),
                "slot_statuses": _status_counts(self.slots),
            },
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the portable runtime contract without image arrays or paths."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
        )


class AOIInspector:
    """Fail-closed inspection service; this class is not an AI model.

    The injected detector proposes presence candidates only. Its bounding-box
    centre is used for deterministic slot association but never for the final
    position measurement, which remains fixed-ROI/template based.
    """

    def __init__(
        self,
        detector: ComponentDetector,
        *,
        aligner: PCBAligner | None = None,
        position_measurer: PositionMeasurer | None = None,
        comparator: GoldenComparator | None = None,
        config: InspectionConfig | None = None,
        candidate_provider: Callable[[np.ndarray], Sequence[Detection]] | None = None,
        runtime_detector_identifier: str | None = None,
    ) -> None:
        if not isinstance(detector, ComponentDetector):
            raise TypeError("detector must implement ComponentDetector")
        self.detector = detector
        self.aligner = aligner or PCBAligner()
        self.position_measurer = position_measurer or PositionMeasurer()
        self.comparator = comparator or GoldenComparator()
        self.config = config or InspectionConfig()
        self._candidate_provider = candidate_provider or detector.detect
        self.runtime_detector_identifier = (
            runtime_detector_identifier or detector_identifier(detector)
        )

    @classmethod
    def from_pipeline(cls, pipeline: Any, **kwargs: Any) -> AOIInspector:
        """Reuse both ``AOIPipeline.detector`` and its tiled detection facade."""

        detector = getattr(pipeline, "detector", None)
        detect_components = getattr(pipeline, "detect_components", None)
        if not isinstance(detector, ComponentDetector) or not callable(detect_components):
            raise TypeError("pipeline must expose a ComponentDetector and detect_components")

        def provide(image: np.ndarray) -> Sequence[Detection]:
            return detect_components(image, frame_id="inspection")

        return cls(
            detector,
            candidate_provider=provide,
            **kwargs,
        )

    def _production_gate_findings(
        self, recipe: InspectionRecipe
    ) -> tuple[str, ...]:
        """Ba cổng an toàn production nào KHÔNG đạt, đánh giá bất kể cờ bật/tắt.

        Tách ra khỏi chỗ quyết định vì hai câu khác nhau: *"có chặn không"* phụ
        thuộc ``require_production_eligible``, còn *"lẽ ra có bị chặn không"* thì
        không. Câu thứ hai mới là thứ một kho lịch sử kiểm tra cần lưu — thiếu
        nó thì một lần PASS chạy thử và một lần PASS thật trông giống hệt nhau.

        Thứ tự giữ nguyên như bản trước, vì lần chạy bị chặn lấy cổng ĐẦU TIÊN
        hỏng làm ``reason``.
        """

        findings: list[str] = []
        if not recipe.production_eligible:
            findings.append("recipe_not_production_eligible")
        if isinstance(self.detector, CVComponentDetector):
            findings.append("runtime_detector_not_production_capable")
        if recipe.model_identifiers.get("component_detector") != (
            self.runtime_detector_identifier
        ):
            findings.append("runtime_detector_mismatch")
        return tuple(findings)

    def inspect(
        self,
        test_image: np.ndarray,
        recipe: InspectionRecipe,
        recipe_root: str | Path,
        *,
        source_valid_mask: np.ndarray | None = None,
    ) -> InspectionRun:
        """Inspect one measurement image and aggregate independent decisions."""

        runtime_detector = type(self.detector).__name__
        # Đánh giá cả ba cổng BẤT KỂ cờ, để bản ghi nói được lần chạy này lẽ ra
        # có bị chặn hay không. Thứ tự giữ nguyên như cũ vì ``reason`` của lần
        # chạy bị chặn là cổng ĐẦU TIÊN hỏng.
        findings = self._production_gate_findings(recipe)
        enforced = self.config.require_production_eligible
        if enforced and findings:
            return _invalid_run(
                recipe,
                runtime_detector,
                self.runtime_detector_identifier,
                findings[0],
                production_gates_enforced=True,
                production_gate_findings=findings,
            )

        try:
            image = ensure_bgr(test_image)
        except InvalidImageError:
            return _invalid_run(
                recipe,
                runtime_detector,
                self.runtime_detector_identifier,
                "invalid_measurement_image",
                production_gates_enforced=enforced,
                production_gate_findings=findings,
            )
        alignment = self.aligner.align_to_recipe(
            image,
            recipe,
            recipe_root,
            source_valid_mask=source_valid_mask,
        )
        if not alignment.success or alignment.image is None:
            return InspectionRun(
                status="invalid",
                reason=f"alignment_invalid:{alignment.reason}",
                alignment=alignment,
                slots=(),
                extras=(),
                recipe_schema_version=recipe.schema_version,
                recipe_sha256=recipe.content_sha256,
                golden_sha256=recipe.golden_sha256,
                model_identifiers=recipe.model_identifiers,
                runtime_detector=runtime_detector,
                runtime_detector_identifier=self.runtime_detector_identifier,
                production_gates_enforced=enforced,
                production_gate_findings=findings,
                board_id=recipe.board_id,
                side=recipe.side,
            )

        try:
            raw_detections = list(self._candidate_provider(alignment.image))
        except Exception as exc:
            return InspectionRun(
                status="invalid",
                reason=f"detector_failure:{type(exc).__name__}",
                alignment=alignment,
                slots=(),
                extras=(),
                recipe_schema_version=recipe.schema_version,
                recipe_sha256=recipe.content_sha256,
                golden_sha256=recipe.golden_sha256,
                model_identifiers=recipe.model_identifiers,
                runtime_detector=runtime_detector,
                runtime_detector_identifier=self.runtime_detector_identifier,
                production_gates_enforced=enforced,
                production_gate_findings=findings,
                board_id=recipe.board_id,
                side=recipe.side,
            )
        detections = _valid_detections(raw_detections, alignment.image.shape[:2])
        associations, unmatched = _associate_candidates(
            recipe.slots,
            detections,
            allow_class_mismatch=self.config.allow_class_mismatch_candidate,
        )

        slot_results: list[SlotInspectionResult] = []
        for slot in recipe.slots:
            candidate = associations.get(slot.slot_id)
            position = self.position_measurer.measure(
                alignment.image,
                slot,
                recipe_root,
                recipe.metrology,
                candidate=candidate,
                global_valid_mask=alignment.valid_mask,
            )
            appearance = self.comparator.compare(
                alignment.image,
                slot,
                recipe_root,
                position,
                global_valid_mask=alignment.valid_mask,
            )
            status, reason = _slot_decision(position, appearance)
            slot_results.append(
                SlotInspectionResult(
                    slot_id=slot.slot_id,
                    candidate=(
                        None
                        if candidate is None
                        else DetectionSummary.from_detection(candidate)
                    ),
                    class_hint_match=(
                        None if candidate is None else _class_matches(slot, candidate)
                    ),
                    position=position,
                    appearance=appearance,
                    status=status,
                    reason=reason,
                )
            )

        extras = tuple(DetectionSummary.from_detection(item) for item in unmatched)
        board_status, board_reason = _board_decision(
            slot_results,
            extras,
            extras_are_ng=self.config.extras_are_ng,
        )
        return InspectionRun(
            status=board_status,
            reason=board_reason,
            alignment=alignment,
            slots=tuple(slot_results),
            extras=extras,
            recipe_schema_version=recipe.schema_version,
            recipe_sha256=recipe.content_sha256,
            golden_sha256=recipe.golden_sha256,
            model_identifiers=recipe.model_identifiers,
            runtime_detector=runtime_detector,
            runtime_detector_identifier=self.runtime_detector_identifier,
            production_gates_enforced=enforced,
            production_gate_findings=findings,
            board_id=recipe.board_id,
            side=recipe.side,
        )


def render_inspection_overlay(
    run: InspectionRun,
    recipe: InspectionRecipe,
) -> np.ndarray | None:
    """Render core inspection decisions in canonical Golden coordinates.

    The function is presentation-only: it consumes the already-decided slot,
    alignment, and anomaly results and never recalculates PASS/NG policy.
    Invalid alignments have no canonical image and therefore return ``None``.
    """

    if run.alignment.image is None:
        return None
    canvas = ensure_bgr(run.alignment.image).copy()
    height, width = canvas.shape[:2]
    results = {item.slot_id: item for item in run.slots}
    colors = {
        "pass": (45, 190, 70),
        "ng_position": (0, 165, 255),
        "ng_appearance": (40, 40, 230),
        "ng_position_and_appearance": (30, 30, 255),
        "ng_missing": (190, 40, 190),
        "review": (0, 215, 255),
    }
    for slot in recipe.slots:
        result = results.get(slot.slot_id)
        if result is None:
            continue
        x1, y1, x2, y2 = slot.fixed_roi_xyxy.clamp(width, height).to_int()
        if x2 <= x1 or y2 <= y1:
            continue
        anomaly_mask = result.appearance.anomaly_mask
        if (
            anomaly_mask is not None
            and anomaly_mask.shape == (y2 - y1, x2 - x1)
            and np.any(anomaly_mask > 0)
        ):
            roi = canvas[y1:y2, x1:x2]
            pixels = anomaly_mask > 0
            tint = np.zeros_like(roi)
            tint[:, :] = (20, 20, 255)
            roi[pixels] = cv2.addWeighted(roi[pixels], 0.45, tint[pixels], 0.55, 0.0)
        color = colors.get(result.status, (160, 160, 160))
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), color, 2)
        _draw_overlay_label(
            canvas,
            f"{slot.slot_id} {result.status}",
            (x1, max(14, y1 - 5)),
            color,
        )
    for extra in run.extras:
        x1, y1, x2, y2 = BoundingBox(*extra.bbox_xyxy).clamp(width, height).to_int()
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (255, 80, 20), 2)
        _draw_overlay_label(
            canvas,
            f"extra {extra.label}",
            (x1, max(14, y1 - 5)),
            (255, 80, 20),
        )
    return canvas


def _draw_overlay_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )
    x, y = origin
    x = min(max(0, int(x)), max(0, image.shape[1] - text_width - 4))
    y = min(max(text_height + baseline + 4, int(y)), image.shape[0] - 1)
    cv2.rectangle(
        image,
        (x, y - text_height - baseline - 4),
        (x + text_width + 4, y + 2),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 2, y - baseline),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _associate_candidates(
    slots: Sequence[SlotRecipe],
    detections: Sequence[Detection],
    *,
    allow_class_mismatch: bool,
) -> tuple[dict[str, Detection], list[Detection]]:
    candidate_edges: list[tuple[int, int, int, float]] = []
    detection_keys = [
        (
            (detection.bbox.y1 + detection.bbox.y2) / 2.0,
            (detection.bbox.x1 + detection.bbox.x2) / 2.0,
            detection.label,
            detection.class_id if detection.class_id is not None else -1,
            -float(detection.confidence),
            tuple(detection.bbox.as_xyxy()),
        )
        for detection in detections
    ]
    ranked_indices = sorted(range(len(detections)), key=lambda index: detection_keys[index])
    detection_rank = {index: rank for rank, index in enumerate(ranked_indices)}
    for slot_index, slot in enumerate(slots):
        roi = slot.fixed_roi_xyxy
        margin = float(slot.search_margin_px)
        search = BoundingBox(
            roi.x1 - margin,
            roi.y1 - margin,
            roi.x2 + margin,
            roi.y2 + margin,
        )
        expected_x, expected_y = slot.expected_center_px
        for index, detection in enumerate(detections):
            center_x = (detection.bbox.x1 + detection.bbox.x2) / 2.0
            center_y = (detection.bbox.y1 + detection.bbox.y2) / 2.0
            if not (
                search.x1 <= center_x <= search.x2
                and search.y1 <= center_y <= search.y2
            ):
                continue
            class_match = _class_matches(slot, detection)
            if not class_match and not allow_class_mismatch:
                continue
            distance = math.hypot(center_x - expected_x, center_y - expected_y)
            candidate_edges.append(
                (
                    slot_index,
                    index,
                    0 if class_match else 1,
                    distance,
                )
            )

    matched_pairs = _minimum_cost_maximum_matching(
        len(slots),
        len(detections),
        candidate_edges,
        detection_rank,
    )
    associations: dict[str, Detection] = {}
    assigned_detections: set[int] = set()
    for slot_index, detection_index in matched_pairs:
        associations[slots[slot_index].slot_id] = detections[detection_index]
        assigned_detections.add(detection_index)
    unmatched = [
        detection
        for index, detection in enumerate(detections)
        if index not in assigned_detections
    ]
    return associations, unmatched


@dataclass(slots=True)
class _FlowEdge:
    to: int
    reverse: int
    capacity: int
    cost: tuple[int, float, int]


def _minimum_cost_maximum_matching(
    slot_count: int,
    detection_count: int,
    candidate_edges: Sequence[tuple[int, int, int, float]],
    detection_rank: Mapping[int, int],
) -> list[tuple[int, int]]:
    """Maximize presence assignments, then minimize class/distance cost."""

    source = 0
    first_slot = 1
    first_detection = first_slot + slot_count
    sink = first_detection + detection_count
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    for slot_index in range(slot_count):
        _add_flow_edge(graph, source, first_slot + slot_index, (0, 0.0, 0))
    for detection_index in range(detection_count):
        _add_flow_edge(
            graph,
            first_detection + detection_index,
            sink,
            (0, 0.0, 0),
        )

    assignment_edges: dict[tuple[int, int], _FlowEdge] = {}
    ordered_edges = sorted(
        candidate_edges,
        key=lambda item: (
            item[0],
            item[2],
            item[3],
            detection_rank[item[1]],
        ),
    )
    for slot_index, detection_index, class_penalty, distance in ordered_edges:
        tie_break = slot_index * max(1, detection_count) + detection_rank[detection_index]
        edge = _add_flow_edge(
            graph,
            first_slot + slot_index,
            first_detection + detection_index,
            (int(class_penalty), float(distance), int(tie_break)),
        )
        assignment_edges[(slot_index, detection_index)] = edge

    node_count = len(graph)
    while True:
        distances: list[tuple[int, float, int] | None] = [None] * node_count
        previous: list[tuple[int, int] | None] = [None] * node_count
        distances[source] = (0, 0.0, 0)
        for _ in range(node_count - 1):
            changed = False
            for node, edges in enumerate(graph):
                if distances[node] is None:
                    continue
                for edge_index, edge in enumerate(edges):
                    if edge.capacity <= 0:
                        continue
                    candidate_cost = _add_cost(distances[node], edge.cost)
                    if distances[edge.to] is None or candidate_cost < distances[edge.to]:
                        distances[edge.to] = candidate_cost
                        previous[edge.to] = (node, edge_index)
                        changed = True
            if not changed:
                break
        if previous[sink] is None:
            break
        node = sink
        while node != source:
            previous_node, edge_index = previous[node]  # type: ignore[misc]
            edge = graph[previous_node][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = previous_node

    return sorted(
        pair for pair, edge in assignment_edges.items() if edge.capacity == 0
    )


def _add_flow_edge(
    graph: list[list[_FlowEdge]],
    source: int,
    target: int,
    cost: tuple[int, float, int],
) -> _FlowEdge:
    forward = _FlowEdge(target, len(graph[target]), 1, cost)
    reverse = _FlowEdge(
        source,
        len(graph[source]),
        0,
        (-cost[0], -cost[1], -cost[2]),
    )
    graph[source].append(forward)
    graph[target].append(reverse)
    return forward


def _add_cost(
    first: tuple[int, float, int], second: tuple[int, float, int]
) -> tuple[int, float, int]:
    return (
        first[0] + second[0],
        first[1] + second[1],
        first[2] + second[2],
    )


def _valid_detections(
    detections: Sequence[Detection], image_shape: tuple[int, int]
) -> list[Detection]:
    height, width = image_shape
    valid: list[Detection] = []
    for detection in detections:
        if not isinstance(detection, Detection):
            continue
        bbox = detection.bbox.clamp(width, height)
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        valid.append(
            Detection(
                label=detection.label,
                confidence=detection.confidence,
                bbox=bbox,
                class_id=detection.class_id,
                source=detection.source,
                detection_id=detection.detection_id,
            )
        )
    return valid


def _class_matches(slot: SlotRecipe, detection: Detection) -> bool:
    if slot.class_id is not None and detection.class_id is not None:
        return slot.class_id == detection.class_id
    return slot.label_hint == detection.label


def _slot_decision(
    position: PositionResult,
    appearance: GoldenCompareResult,
) -> tuple[str, str]:
    if position.status == "missing_candidate":
        return "ng_missing", "missing_candidate"
    if position.status == "unmeasurable":
        return "review", f"position_unmeasurable:{position.reason}"
    position_ng = position.status == "ng"
    appearance_ng = appearance.status == "anomaly"
    if position_ng and appearance_ng:
        return "ng_position_and_appearance", "position_and_appearance_failed"
    if position_ng:
        return "ng_position", "position_failed"
    if appearance_ng:
        return "ng_appearance", "appearance_failed"
    if appearance.status != "pass":
        return "review", f"appearance_not_evaluated:{appearance.reason}"
    return "pass", "within_all_thresholds"


def _board_decision(
    slots: Sequence[SlotInspectionResult],
    extras: Sequence[DetectionSummary],
    *,
    extras_are_ng: bool,
) -> tuple[str, str]:
    ng_slots = [slot for slot in slots if slot.status.startswith("ng_")]
    if ng_slots:
        return "ng", f"{len(ng_slots)}_slot_failures"
    if extras and extras_are_ng:
        return "ng", f"{len(extras)}_extra_candidates"
    review_slots = [slot for slot in slots if slot.status == "review"]
    if review_slots or (extras and not extras_are_ng):
        return "review", "manual_review_required"
    return "pass", "all_slots_passed"


def _invalid_run(
    recipe: InspectionRecipe,
    runtime_detector: str,
    runtime_detector_identifier: str,
    reason: str,
    *,
    production_gates_enforced: bool = True,
    production_gate_findings: tuple[str, ...] = (),
) -> InspectionRun:
    alignment = StrictAlignmentResult(
        status="invalid",
        image=None,
        transform=None,
        residual_px=None,
        matched_anchors=0,
        inliers=0,
        inlier_ratio=0.0,
        scale=None,
        rotation_deg=None,
        canvas_overlap_ratio=None,
        valid_mask=None,
        reason=reason,
    )
    return InspectionRun(
        status="invalid",
        reason=reason,
        alignment=alignment,
        slots=(),
        extras=(),
        recipe_schema_version=recipe.schema_version,
        recipe_sha256=recipe.content_sha256,
        golden_sha256=recipe.golden_sha256,
        model_identifiers=recipe.model_identifiers,
        runtime_detector=runtime_detector,
        runtime_detector_identifier=runtime_detector_identifier,
        production_gates_enforced=production_gates_enforced,
        production_gate_findings=production_gate_findings,
        board_id=recipe.board_id,
        side=recipe.side,
    )


def missing_required_sides(
    runs: Sequence[InspectionRun],
    *,
    required: Sequence[str] = ("top", "bottom"),
) -> tuple[str, ...]:
    """Những mặt bắt buộc chưa có lần chạy ĐẠT — theo thứ tự ``required``.

    Kế hoạch số hoá §4.4: TOP và BOTTOM thuộc cùng một PCB vật lý nhưng có lần
    kiểm tra và recipe riêng, nên **không được lấy kết quả một mặt thay cho cả
    PCB**. Danh sách lỗi rỗng ở một mặt không nói gì về mặt kia.

    ``required`` là tham số chứ không phải hằng số giấu trong hàm: bo một mặt,
    hoặc công đoạn chỉ soi một mặt, là chuyện có thật. Nhưng mặc định là **cả
    hai**, vì bỏ sót một mặt là cho lọt, còn đòi thừa một mặt chỉ tốn công.

    Chỉ tính lần chạy có cổng production được **thi hành** — một lần PASS ở chế
    độ chạy thử không đủ tư cách làm căn cứ (§9.3).
    """

    passed = {
        run.side
        for run in runs
        if run.status == "pass" and run.production_gates_enforced
    }
    return tuple(side for side in required if side not in passed)


def _status_counts(slots: Sequence[SlotInspectionResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for slot in slots:
        counts[slot.status] = counts.get(slot.status, 0) + 1
    return dict(sorted(counts.items()))


__all__ = [
    "AOIInspector",
    "DetectionSummary",
    "InspectionConfig",
    "InspectionRun",
    "SlotInspectionResult",
]
