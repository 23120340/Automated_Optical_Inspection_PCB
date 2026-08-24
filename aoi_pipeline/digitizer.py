"""Deterministic multi-frame component consensus for draft PCB placement data.

The inputs to this module are already expressed in one Golden canvas.  This
module deliberately does *not* align images, run OCR, infer a footprint or
invent an orientation from an axis-aligned detector box.  Its authoritative
artifact is therefore pixel-native.  A placement-shaped CSV can be exported
only when the caller supplies an explicit pixel-to-millimetre homography, and
every row in that file remains marked ``NEEDS_REVIEW``.

Coordinates use the repository-wide ``xyxy`` convention with exclusive right
and bottom edges.  ``pixel_to_mm_homography`` maps Golden pixel points
``[x_px, y_px, 1]`` to board millimetres by homogeneous division.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .models import BoundingBox, Detection

PIXEL_SCHEMA_VERSION = "aoi-pnp-consensus-pixels/1.0"
PIXEL_COORDINATE_SPACE = "golden_board_pixels"
PIXEL_ARTIFACT_STATUS = "AUTHORITATIVE_PIXEL_CONSENSUS"
PLACEMENT_DRAFT_STATUS = "NEEDS_REVIEW"
DEFAULT_EXCLUDED_LABELS = frozenset({"pad", "pads", "pin", "pins"})

# Prefer familiar reference-designator prefixes when the detector class has an
# unambiguous equivalent.  These are still synthetic IDs, never OCR claims.
_AUTO_PREFIX_BY_LABEL: Mapping[str, str] = {
    "battery": "BT",
    "buzzer": "BZ",
    "capacitor": "C",
    "clock": "Y",
    "connector": "J",
    "diode": "D",
    "fuse": "F",
    "heatsink": "HS",
    "ic": "U",
    "inductor": "L",
    "led": "LED",
    "potentiometer": "RV",
    "relay": "K",
    "resistor": "R",
    "switch": "SW",
    "transducer": "M",
    "transformer": "T",
    "transistor": "Q",
}

_PIXEL_CSV_COLUMNS = (
    "schema_version",
    "artifact_status",
    "coordinate_space",
    "designator",
    "designator_source",
    "label",
    "center_x_px",
    "center_y_px",
    "x1_px",
    "y1_px",
    "x2_px",
    "y2_px",
    "rotation_deg",
    "rotation_source",
    "footprint",
    "observation_count",
    "frame_count",
    "support_ratio",
    "class_purity",
    "center_mad_px",
    "median_confidence",
    "consensus_status",
    "review_reasons",
    "class_counts_json",
    "frame_ids_json",
)

_PLACEMENT_CSV_COLUMNS = (
    "Designator",
    "Mid X",
    "Mid Y",
    "Rotation",
    "Layer",
    "Footprint",
    "Comment",
    "Status",
    "Expected Class",
    "Support Ratio",
    "Class Purity",
    "Center MAD PX",
    "Source Coordinate Space",
)


@dataclass(frozen=True, slots=True)
class ConsensusConfig:
    """Quality and association policy for one same-SKU image set.

    ``cluster_radius_px`` is a gate in the shared Golden canvas, not a physical
    tolerance.  A class mismatch adds a deterministic association penalty but
    does not prohibit a match; otherwise detector label jitter would be hidden
    as two falsely pure components instead of lowering ``class_purity``.
    """

    cluster_radius_px: float = 24.0
    min_support_ratio: float = 0.80
    min_class_purity: float = 0.80
    class_mismatch_penalty_ratio: float = 0.10
    excluded_labels: frozenset[str] = field(
        default_factory=lambda: DEFAULT_EXCLUDED_LABELS
    )

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.cluster_radius_px)) or self.cluster_radius_px <= 0:
            raise ValueError("cluster_radius_px must be a positive finite value")
        for name, value in (
            ("min_support_ratio", self.min_support_ratio),
            ("min_class_purity", self.min_class_purity),
        ):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if (
            not math.isfinite(float(self.class_mismatch_penalty_ratio))
            or self.class_mismatch_penalty_ratio < 0
        ):
            raise ValueError("class_mismatch_penalty_ratio must be non-negative")
        normalized = frozenset(
            str(label).strip().lower()
            for label in self.excluded_labels
            if str(label).strip()
        )
        object.__setattr__(self, "excluded_labels", normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_radius_px": float(self.cluster_radius_px),
            "min_support_ratio": float(self.min_support_ratio),
            "min_class_purity": float(self.min_class_purity),
            "class_mismatch_penalty_ratio": float(
                self.class_mismatch_penalty_ratio
            ),
            "excluded_labels": sorted(self.excluded_labels),
        }


@dataclass(frozen=True, slots=True)
class ConsensusComponent:
    """One spatial site summarized across unique source frames."""

    designator: str
    label: str
    bbox: BoundingBox
    center_px: tuple[float, float]
    observation_count: int
    frame_count: int
    support_ratio: float
    class_purity: float
    center_mad_px: float
    median_confidence: float
    class_counts: tuple[tuple[str, int], ...]
    frame_ids: tuple[str, ...]
    consensus_status: str
    review_reasons: tuple[str, ...] = ()

    @property
    def eligible_for_placement(self) -> bool:
        """Whether the component passed the configured consensus gates.

        Passing these gates does not approve a PnP row.  Its designator is still
        synthetic and rotation/footprint are still unknown, which is why the
        placement export remains ``NEEDS_REVIEW``.
        """

        return self.consensus_status == "CONSENSUS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "designator": self.designator,
            "designator_source": "synthetic_auto",
            "label": self.label,
            "bbox_xyxy": self.bbox.as_xyxy(),
            "center_px": [float(self.center_px[0]), float(self.center_px[1])],
            "rotation_deg": None,
            "rotation_source": "unknown_axis_aligned_detection",
            "footprint": None,
            "observation_count": int(self.observation_count),
            "frame_count": int(self.frame_count),
            "support_ratio": float(self.support_ratio),
            "class_purity": float(self.class_purity),
            "center_mad_px": float(self.center_mad_px),
            "median_confidence": float(self.median_confidence),
            "class_counts": dict(self.class_counts),
            "frame_ids": list(self.frame_ids),
            "consensus_status": self.consensus_status,
            "review_reasons": list(self.review_reasons),
        }


@dataclass(frozen=True, slots=True)
class PnpConsensus:
    """Portable, deterministic result for one same-SKU Golden canvas."""

    frame_ids: tuple[str, ...]
    components: tuple[ConsensusComponent, ...]
    config: ConsensusConfig
    canvas_size: tuple[int, int] | None = None
    excluded_observation_count: int = 0
    invalid_observation_count: int = 0
    duplicate_observation_count: int = 0
    schema_version: str = PIXEL_SCHEMA_VERSION
    coordinate_space: str = PIXEL_COORDINATE_SPACE

    @property
    def frame_count(self) -> int:
        return len(self.frame_ids)

    @property
    def eligible_components(self) -> tuple[ConsensusComponent, ...]:
        return tuple(
            component
            for component in self.components
            if component.eligible_for_placement
        )

    def to_dict(self) -> dict[str, Any]:
        canvas = (
            None
            if self.canvas_size is None
            else {"width": int(self.canvas_size[0]), "height": int(self.canvas_size[1])}
        )
        return {
            "schema_version": self.schema_version,
            "artifact_status": PIXEL_ARTIFACT_STATUS,
            "coordinate_space": self.coordinate_space,
            "canvas_size": canvas,
            "frame_ids": list(self.frame_ids),
            "config": self.config.to_dict(),
            "summary": {
                "frame_count": self.frame_count,
                "component_count": len(self.components),
                "eligible_component_count": len(self.eligible_components),
                "excluded_observation_count": int(
                    self.excluded_observation_count
                ),
                "invalid_observation_count": int(self.invalid_observation_count),
                "duplicate_observation_count": int(
                    self.duplicate_observation_count
                ),
            },
            "components": [component.to_dict() for component in self.components],
        }


@dataclass(frozen=True, slots=True)
class _Observation:
    frame_id: str
    label: str
    confidence: float
    bbox: BoundingBox

    @property
    def center(self) -> tuple[float, float]:
        return self.bbox.center


@dataclass(slots=True)
class _Cluster:
    observations: list[_Observation]

    @property
    def frame_ids(self) -> frozenset[str]:
        return frozenset(item.frame_id for item in self.observations)

    @property
    def center(self) -> tuple[float, float]:
        points = np.asarray([item.center for item in self.observations], dtype=np.float64)
        median = np.median(points, axis=0)
        return (float(median[0]), float(median[1]))

    @property
    def majority_label(self) -> str:
        return _majority_label(self.observations)[0]


@dataclass(frozen=True, slots=True)
class _ComponentStats:
    label: str
    bbox: BoundingBox
    center_px: tuple[float, float]
    observation_count: int
    support_ratio: float
    class_purity: float
    center_mad_px: float
    median_confidence: float
    class_counts: tuple[tuple[str, int], ...]
    frame_ids: tuple[str, ...]
    consensus_status: str
    review_reasons: tuple[str, ...]


DetectionFrames = (
    Mapping[str, Sequence[Detection]]
    | Iterable[tuple[str, Sequence[Detection]]]
)


def build_consensus(
    detections_by_frame: DetectionFrames,
    *,
    config: ConsensusConfig | None = None,
    canvas_size: tuple[int, int] | None = None,
) -> PnpConsensus:
    """Aggregate aligned detections into deterministic spatial consensus.

    Mapping order, frame order and detection order do not affect the result.
    Frames are normalized and processed by a deterministic density/name order,
    and association is one-to-one for every frame.  Consequently no component
    can claim more than one supporting observation from the same source image.

    Empty frames remain in the support denominator.  This is intentional: a
    detector miss must reduce support instead of disappearing from the audit.
    """

    policy = config or ConsensusConfig()
    normalized_canvas = _validate_canvas_size(canvas_size)
    frames, excluded, invalid, duplicate = _normalize_frames(
        detections_by_frame, policy
    )
    frame_ids = tuple(sorted(frames))
    if not frame_ids:
        raise ValueError("at least one uniquely named frame is required")

    clusters = _associate_frames(frames, policy)
    stats = [
        _summarize_cluster(cluster, len(frame_ids), policy) for cluster in clusters
    ]
    stats.sort(key=_component_stats_sort_key)
    components = _assign_designators(stats, len(frame_ids))
    return PnpConsensus(
        frame_ids=frame_ids,
        components=components,
        config=policy,
        canvas_size=normalized_canvas,
        excluded_observation_count=excluded,
        invalid_observation_count=invalid,
        duplicate_observation_count=duplicate,
    )


def build_pnp_consensus(
    detections_by_frame: DetectionFrames,
    *,
    config: ConsensusConfig | None = None,
    canvas_size: tuple[int, int] | None = None,
) -> PnpConsensus:
    """Explicitly named alias for :func:`build_consensus`."""

    return build_consensus(
        detections_by_frame, config=config, canvas_size=canvas_size
    )


def export_pixel_json(consensus: PnpConsensus, path: str | Path) -> Path:
    """Write the authoritative pixel-native consensus as deterministic JSON."""

    destination = _destination(path)
    destination.write_text(
        json.dumps(consensus.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def export_authoritative_pixel_json(
    consensus: PnpConsensus, path: str | Path
) -> Path:
    """Descriptive alias for :func:`export_pixel_json`."""

    return export_pixel_json(consensus, path)


def export_pixel_csv(consensus: PnpConsensus, path: str | Path) -> Path:
    """Write one auditable row per pixel-native consensus component."""

    destination = _destination(path)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=_PIXEL_CSV_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        for component in consensus.components:
            writer.writerow(_pixel_csv_row(consensus, component))
    return destination


def export_authoritative_pixel_csv(
    consensus: PnpConsensus, path: str | Path
) -> Path:
    """Descriptive alias for :func:`export_pixel_csv`."""

    return export_pixel_csv(consensus, path)


def placement_draft_rows(
    consensus: PnpConsensus,
    *,
    pixel_to_mm_homography: Sequence[Sequence[float]] | np.ndarray | None,
    side: str = "top",
    include_review_components: bool = False,
) -> list[dict[str, Any]]:
    """Return placement-shaped rows after an explicit pixel-to-mm transform.

    Rotation and footprint are always empty because neither is observable from
    the input contract.  ``Status`` is always ``NEEDS_REVIEW`` even for a site
    that passed support/purity gates: synthetic AUTO designators are not OCR and
    this draft must not masquerade as manufacturing CAD.
    """

    matrix = _pixel_to_mm_matrix(pixel_to_mm_homography)
    normalized_side = str(side).strip().lower()
    if normalized_side not in {"top", "bottom"}:
        raise ValueError("side must be 'top' or 'bottom'")
    selected = (
        consensus.components
        if include_review_components
        else consensus.eligible_components
    )
    if not selected:
        return []
    centres = np.asarray([item.center_px for item in selected], dtype=np.float64)
    projected = _project_points(centres, matrix)
    rows: list[dict[str, Any]] = []
    for component, (x_mm, y_mm) in zip(selected, projected, strict=True):
        rows.append(
            {
                "Designator": component.designator,
                "Mid X": f"{float(x_mm):.6f}",
                "Mid Y": f"{float(y_mm):.6f}",
                "Rotation": "",
                "Layer": normalized_side.title(),
                "Footprint": "",
                "Comment": (
                    "Synthetic AUTO designator; verify RefDes, rotation and "
                    "footprint before use"
                ),
                "Status": PLACEMENT_DRAFT_STATUS,
                "Expected Class": _safe_csv_text(component.label),
                "Support Ratio": f"{component.support_ratio:.6f}",
                "Class Purity": f"{component.class_purity:.6f}",
                "Center MAD PX": f"{component.center_mad_px:.6f}",
                "Source Coordinate Space": consensus.coordinate_space,
            }
        )
    return rows


def export_placement_draft(
    consensus: PnpConsensus,
    path: str | Path,
    *,
    pixel_to_mm_homography: Sequence[Sequence[float]] | np.ndarray | None,
    side: str = "top",
    include_review_components: bool = False,
) -> Path:
    """Write a review-only placement CSV after explicit pixel-to-mm mapping."""

    rows = placement_draft_rows(
        consensus,
        pixel_to_mm_homography=pixel_to_mm_homography,
        side=side,
        include_review_components=include_review_components,
    )
    destination = _destination(path)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=_PLACEMENT_CSV_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return destination


def _normalize_frames(
    detections_by_frame: DetectionFrames,
    policy: ConsensusConfig,
) -> tuple[dict[str, tuple[_Observation, ...]], int, int, int]:
    if isinstance(detections_by_frame, Mapping):
        items = list(detections_by_frame.items())
    else:
        items = list(detections_by_frame)

    frames: dict[str, tuple[_Observation, ...]] = {}
    excluded = 0
    invalid = 0
    duplicate = 0
    for raw_frame_id, raw_detections in items:
        frame_id = str(raw_frame_id).strip()
        if not frame_id:
            raise ValueError("frame IDs must be non-empty")
        if Path(frame_id).is_absolute():
            raise ValueError("frame IDs must be portable IDs, not absolute paths")
        if frame_id in frames:
            raise ValueError(f"duplicate frame ID: {frame_id}")
        if isinstance(raw_detections, (str, bytes)):
            raise TypeError(f"detections for {frame_id} must be a sequence")

        by_exact_site: dict[tuple[Any, ...], _Observation] = {}
        for detection in raw_detections:
            if not isinstance(detection, Detection):
                raise TypeError(
                    f"detections for {frame_id} must contain Detection instances"
                )
            label = str(detection.label).strip().lower()
            if label in policy.excluded_labels:
                excluded += 1
                continue
            bbox = detection.bbox
            values = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
            if (
                not label
                or not all(math.isfinite(float(value)) for value in values)
                or bbox.width <= 0
                or bbox.height <= 0
            ):
                invalid += 1
                continue
            observation = _Observation(
                frame_id=frame_id,
                label=label,
                confidence=float(detection.confidence),
                bbox=bbox,
            )
            key = (label, *[float(value) for value in values])
            previous = by_exact_site.get(key)
            if previous is not None:
                duplicate += 1
                if observation.confidence > previous.confidence:
                    by_exact_site[key] = observation
            else:
                by_exact_site[key] = observation
        frames[frame_id] = tuple(sorted(by_exact_site.values(), key=_observation_sort_key))
    return frames, excluded, invalid, duplicate


def _associate_frames(
    frames: Mapping[str, tuple[_Observation, ...]], policy: ConsensusConfig
) -> list[_Cluster]:
    clusters: list[_Cluster] = []
    # The densest frame is the least likely seed to omit a real site.  Frame ID
    # breaks ties, so mapping/insertion order cannot affect the result.
    ordered_frames = sorted(frames, key=lambda key: (-len(frames[key]), key))
    mismatch_penalty = (
        float(policy.cluster_radius_px)
        * float(policy.class_mismatch_penalty_ratio)
    )

    for frame_id in ordered_frames:
        observations = frames[frame_id]
        if not clusters:
            clusters.extend(_Cluster([item]) for item in observations)
            continue

        cluster_snapshots = [
            (cluster.center, cluster.majority_label, _cluster_sort_key(cluster))
            for cluster in clusters
        ]
        candidates: list[tuple[Any, ...]] = []
        for observation_index, observation in enumerate(observations):
            ox, oy = observation.center
            for cluster_index, (center, label, cluster_key) in enumerate(
                cluster_snapshots
            ):
                cx, cy = center
                distance = math.hypot(ox - cx, oy - cy)
                if distance > float(policy.cluster_radius_px):
                    continue
                cost = distance + (
                    mismatch_penalty if observation.label != label else 0.0
                )
                candidates.append(
                    (
                        cost,
                        distance,
                        _observation_sort_key(observation),
                        cluster_key,
                        observation_index,
                        cluster_index,
                    )
                )

        used_observations: set[int] = set()
        used_clusters: set[int] = set()
        for *_, observation_index, cluster_index in sorted(candidates):
            if (
                observation_index in used_observations
                or cluster_index in used_clusters
            ):
                continue
            cluster = clusters[cluster_index]
            # Defensive invariant: a frame is processed once and association is
            # one-to-one, so this should never trigger.
            if frame_id in cluster.frame_ids:
                continue
            cluster.observations.append(observations[observation_index])
            used_observations.add(observation_index)
            used_clusters.add(cluster_index)

        for observation_index, observation in enumerate(observations):
            if observation_index not in used_observations:
                clusters.append(_Cluster([observation]))

    for cluster in clusters:
        frame_ids = [item.frame_id for item in cluster.observations]
        if len(frame_ids) != len(set(frame_ids)):
            raise AssertionError("a consensus cluster contains duplicate frame evidence")
    return clusters


def _summarize_cluster(
    cluster: _Cluster, frame_count: int, policy: ConsensusConfig
) -> _ComponentStats:
    observations = tuple(sorted(cluster.observations, key=_observation_sort_key))
    points = np.asarray([item.center for item in observations], dtype=np.float64)
    centre = np.median(points, axis=0)
    distances = np.linalg.norm(points - centre, axis=1)
    bbox_values = np.asarray(
        [item.bbox.as_xyxy() for item in observations], dtype=np.float64
    )
    bbox_median = np.median(bbox_values, axis=0)
    label, class_counts = _majority_label(observations)
    observation_count = len(observations)
    support = observation_count / frame_count
    purity = class_counts[label] / observation_count
    reasons: list[str] = []
    if support < float(policy.min_support_ratio):
        reasons.append("low_support")
    if purity < float(policy.min_class_purity):
        reasons.append("class_ambiguous")
    return _ComponentStats(
        label=label,
        bbox=BoundingBox(*[float(value) for value in bbox_median]),
        center_px=(float(centre[0]), float(centre[1])),
        observation_count=observation_count,
        support_ratio=float(support),
        class_purity=float(purity),
        center_mad_px=float(np.median(distances)),
        median_confidence=float(
            np.median([item.confidence for item in observations])
        ),
        class_counts=tuple(sorted(class_counts.items())),
        frame_ids=tuple(sorted(item.frame_id for item in observations)),
        consensus_status="CONSENSUS" if not reasons else PLACEMENT_DRAFT_STATUS,
        review_reasons=tuple(reasons),
    )


def _assign_designators(
    stats: Sequence[_ComponentStats], frame_count: int
) -> tuple[ConsensusComponent, ...]:
    counters: defaultdict[str, int] = defaultdict(int)
    output: list[ConsensusComponent] = []
    for item in stats:
        prefix = _AUTO_PREFIX_BY_LABEL.get(item.label, "AUTO")
        counters[prefix] += 1
        designator = (
            f"AUTO_{counters[prefix]:04d}"
            if prefix == "AUTO"
            else f"{prefix}_AUTO_{counters[prefix]:04d}"
        )
        output.append(
            ConsensusComponent(
                designator=designator,
                label=item.label,
                bbox=item.bbox,
                center_px=item.center_px,
                observation_count=item.observation_count,
                frame_count=frame_count,
                support_ratio=item.support_ratio,
                class_purity=item.class_purity,
                center_mad_px=item.center_mad_px,
                median_confidence=item.median_confidence,
                class_counts=item.class_counts,
                frame_ids=item.frame_ids,
                consensus_status=item.consensus_status,
                review_reasons=item.review_reasons,
            )
        )
    return tuple(output)


def _majority_label(
    observations: Sequence[_Observation],
) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter(item.label for item in observations)
    label = min(counts, key=lambda item: (-counts[item], item))
    return label, counts


def _observation_sort_key(observation: _Observation) -> tuple[Any, ...]:
    x, y = observation.center
    return (
        float(y),
        float(x),
        observation.label,
        float(observation.bbox.y1),
        float(observation.bbox.x1),
        float(observation.bbox.y2),
        float(observation.bbox.x2),
        -float(observation.confidence),
        observation.frame_id,
    )


def _cluster_sort_key(cluster: _Cluster) -> tuple[Any, ...]:
    x, y = cluster.center
    return (
        float(y),
        float(x),
        cluster.majority_label,
        tuple(sorted(cluster.frame_ids)),
    )


def _component_stats_sort_key(item: _ComponentStats) -> tuple[Any, ...]:
    return (
        float(item.center_px[1]),
        float(item.center_px[0]),
        item.label,
        *item.bbox.as_xyxy(),
        item.frame_ids,
    )


def _pixel_csv_row(
    consensus: PnpConsensus, component: ConsensusComponent
) -> dict[str, Any]:
    return {
        "schema_version": consensus.schema_version,
        "artifact_status": PIXEL_ARTIFACT_STATUS,
        "coordinate_space": consensus.coordinate_space,
        "designator": component.designator,
        "designator_source": "synthetic_auto",
        "label": _safe_csv_text(component.label),
        "center_x_px": f"{component.center_px[0]:.6f}",
        "center_y_px": f"{component.center_px[1]:.6f}",
        "x1_px": f"{component.bbox.x1:.6f}",
        "y1_px": f"{component.bbox.y1:.6f}",
        "x2_px": f"{component.bbox.x2:.6f}",
        "y2_px": f"{component.bbox.y2:.6f}",
        "rotation_deg": "",
        "rotation_source": "unknown_axis_aligned_detection",
        "footprint": "",
        "observation_count": component.observation_count,
        "frame_count": component.frame_count,
        "support_ratio": f"{component.support_ratio:.6f}",
        "class_purity": f"{component.class_purity:.6f}",
        "center_mad_px": f"{component.center_mad_px:.6f}",
        "median_confidence": f"{component.median_confidence:.6f}",
        "consensus_status": component.consensus_status,
        "review_reasons": ";".join(component.review_reasons),
        "class_counts_json": json.dumps(
            dict(component.class_counts), ensure_ascii=False, sort_keys=True
        ),
        "frame_ids_json": json.dumps(
            list(component.frame_ids), ensure_ascii=False
        ),
    }


def _validate_canvas_size(
    canvas_size: tuple[int, int] | None,
) -> tuple[int, int] | None:
    if canvas_size is None:
        return None
    if len(canvas_size) != 2:
        raise ValueError("canvas_size must be (width, height)")
    width, height = canvas_size
    if isinstance(width, bool) or isinstance(height, bool):
        raise ValueError("canvas_size values must be positive integers")
    if int(width) != width or int(height) != height or width <= 0 or height <= 0:
        raise ValueError("canvas_size values must be positive integers")
    return (int(width), int(height))


def _pixel_to_mm_matrix(
    value: Sequence[Sequence[float]] | np.ndarray | None,
) -> np.ndarray:
    if value is None:
        raise ValueError(
            "pixel_to_mm_homography is required; pixel coordinates must not be "
            "written into millimetre placement columns"
        )
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("pixel_to_mm_homography must be a 3x3 matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("pixel_to_mm_homography must contain finite values")
    # Homographies are defined only up to a non-zero scalar, so an absolute
    # determinant threshold would reject a perfectly valid uniformly scaled
    # matrix. Rank is invariant to that representation choice.
    if int(np.linalg.matrix_rank(matrix)) < 3:
        raise ValueError("pixel_to_mm_homography must be invertible")
    return matrix


def _project_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack(
        [points.astype(np.float64, copy=False), np.ones(len(points), dtype=np.float64)]
    )
    projected = homogeneous @ matrix.T
    scale = projected[:, 2]
    if np.any(~np.isfinite(scale)) or np.any(np.abs(scale) <= 1e-12):
        raise ValueError("pixel_to_mm_homography projects a point to infinity")
    result = projected[:, :2] / scale[:, None]
    if not np.all(np.isfinite(result)):
        raise ValueError("pixel_to_mm_homography produced non-finite coordinates")
    return result


def _destination(path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _safe_csv_text(value: Any) -> str:
    text = str(value)
    return f"'{text}" if text.startswith(("=", "+", "-", "@", "\t", "\r")) else text


__all__ = [
    "ConsensusComponent",
    "ConsensusConfig",
    "DEFAULT_EXCLUDED_LABELS",
    "PIXEL_ARTIFACT_STATUS",
    "PIXEL_COORDINATE_SPACE",
    "PIXEL_SCHEMA_VERSION",
    "PLACEMENT_DRAFT_STATUS",
    "PnpConsensus",
    "build_consensus",
    "build_pnp_consensus",
    "export_authoritative_pixel_csv",
    "export_authoritative_pixel_json",
    "export_pixel_csv",
    "export_pixel_json",
    "export_placement_draft",
    "placement_draft_rows",
]
