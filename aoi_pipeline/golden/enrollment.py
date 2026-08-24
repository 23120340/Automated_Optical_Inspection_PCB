"""Deterministic Golden-image enrollment from repeated captures of one PCB SKU.

The selector deliberately returns one of the supplied files.  It never blends,
stacks, or otherwise synthesises a reference image: a Golden image must remain
traceable to a real acquisition.  Candidate diagnostics are safe to persist;
they contain basenames and content hashes, but no filesystem paths.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from ..imaging.alignment import PCBAligner
from ..imaging.image_io import load_image
from ..models import AlignmentResult


AlignmentCallback = Callable[[np.ndarray, np.ndarray], AlignmentResult]


@dataclass(frozen=True, slots=True)
class ReferenceSelectionConfig:
    """Quality and consensus gates used while enrolling a Golden image."""

    min_images: int = 3
    min_valid_peers: int = 2
    min_peer_ratio: float = 0.80
    diagnostic_max_side: int = 1024
    min_blur_variance: float = 5.0
    min_relative_blur: float = 0.15
    max_clipped_ratio: float = 0.98
    min_mean_luma: float = 0.02
    max_mean_luma: float = 0.98
    min_valid_overlap_ratio: float = 0.80

    def __post_init__(self) -> None:
        if self.min_images < 2:
            raise ValueError("min_images must be at least 2")
        if self.min_valid_peers < 1:
            raise ValueError("min_valid_peers must be at least 1")
        if self.diagnostic_max_side < 32:
            raise ValueError("diagnostic_max_side must be at least 32 pixels")
        if self.min_blur_variance < 0.0 or self.min_relative_blur < 0.0:
            raise ValueError("blur gates must be non-negative")
        for name, value in (
            ("min_peer_ratio", self.min_peer_ratio),
            ("max_clipped_ratio", self.max_clipped_ratio),
            ("min_mean_luma", self.min_mean_luma),
            ("max_mean_luma", self.max_mean_luma),
            ("min_valid_overlap_ratio", self.min_valid_overlap_ratio),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.min_mean_luma >= self.max_mean_luma:
            raise ValueError("min_mean_luma must be lower than max_mean_luma")


@dataclass(frozen=True, slots=True)
class ReferenceQualityMetrics:
    """Radiometric and focus measurements from a diagnostic-size image."""

    laplacian_variance: float
    dark_clipped_ratio: float
    bright_clipped_ratio: float
    clipped_ratio: float
    mean_luma: float
    exposure_deviation: float
    p01_luma: float
    p99_luma: float

    def to_dict(self) -> dict[str, float]:
        return {
            "laplacian_variance": float(self.laplacian_variance),
            "dark_clipped_ratio": float(self.dark_clipped_ratio),
            "bright_clipped_ratio": float(self.bright_clipped_ratio),
            "clipped_ratio": float(self.clipped_ratio),
            "mean_luma": float(self.mean_luma),
            "exposure_deviation": float(self.exposure_deviation),
            "p01_luma": float(self.p01_luma),
            "p99_luma": float(self.p99_luma),
        }


@dataclass(frozen=True, slots=True)
class ReferenceCandidateReport:
    """Serializable diagnostics for one source capture."""

    basename: str
    sha256: str
    image_size: tuple[int, int]
    quality: ReferenceQualityMetrics
    selected: bool
    eligible: bool
    rejection_reasons: tuple[str, ...]
    aligned_peer_count: int
    total_peer_count: int
    alignment_success_ratio: float
    mean_distance: float | None
    median_distance: float | None
    p95_distance: float | None
    medoid_score: float | None
    alignment_method_counts: tuple[tuple[str, int], ...]
    median_inlier_ratio: float | None
    median_correlation: float | None
    median_overlap_ratio: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "basename": self.basename,
            "sha256": self.sha256,
            "image_size": {
                "height": int(self.image_size[0]),
                "width": int(self.image_size[1]),
            },
            "quality": self.quality.to_dict(),
            "selected": bool(self.selected),
            "eligible": bool(self.eligible),
            "rejection_reasons": list(self.rejection_reasons),
            "aligned_peer_count": int(self.aligned_peer_count),
            "total_peer_count": int(self.total_peer_count),
            "alignment_success_ratio": float(self.alignment_success_ratio),
            "mean_distance": _optional_float(self.mean_distance),
            "median_distance": _optional_float(self.median_distance),
            "p95_distance": _optional_float(self.p95_distance),
            "medoid_score": _optional_float(self.medoid_score),
            "alignment_method_counts": dict(self.alignment_method_counts),
            "median_inlier_ratio": _optional_float(self.median_inlier_ratio),
            "median_correlation": _optional_float(self.median_correlation),
            "median_overlap_ratio": _optional_float(self.median_overlap_ratio),
        }


@dataclass(frozen=True, slots=True)
class ReferenceSelectionReport:
    """Path-free audit record for a reference selection attempt."""

    algorithm: str
    source_count: int
    image_size: tuple[int, int] | None
    diagnostic_image_size: tuple[int, int] | None
    quality_candidate_count: int
    required_peers: int
    selected_basename: str | None
    selected_sha256: str | None
    candidates: tuple[ReferenceCandidateReport, ...]
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        selection = None
        if self.selected_basename is not None and self.selected_sha256 is not None:
            selection = {
                "basename": self.selected_basename,
                "sha256": self.selected_sha256,
            }
        return {
            "algorithm": self.algorithm,
            "reference_policy": "single_actual_source",
            "source_count": int(self.source_count),
            "image_size": _size_dict(self.image_size),
            "diagnostic_image_size": _size_dict(self.diagnostic_image_size),
            "quality_candidate_count": int(self.quality_candidate_count),
            "required_peers": int(self.required_peers),
            "selection": selection,
            "failure_reason": self.failure_reason,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return deterministic JSON which is safe to attach to a recipe."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )


@dataclass(frozen=True, slots=True)
class ReferenceSelectionResult:
    """Selected source path plus its persistable, path-free audit report."""

    reference_path: Path = field(repr=False)
    report: ReferenceSelectionReport

    @property
    def selected_path(self) -> Path:
        """Compatibility alias emphasizing that this is an existing source."""

        return self.reference_path


class ReferenceSelectionError(RuntimeError):
    """Raised when enrollment cannot meet every quality/consensus gate."""

    def __init__(
        self,
        message: str,
        report: ReferenceSelectionReport | None = None,
    ) -> None:
        super().__init__(message)
        self.report = report


@dataclass(slots=True)
class _Candidate:
    path: Path
    basename: str
    sha256: str
    image: np.ndarray
    diagnostic_image: np.ndarray
    quality: ReferenceQualityMetrics
    rejection_reasons: list[str]
    distances: list[float]
    methods: Counter[str]
    inlier_ratios: list[float]
    correlations: list[float]
    overlap_ratios: list[float]
    medoid_score: float | None = None


@dataclass(frozen=True, slots=True)
class _PairObservation:
    distance: float
    method: str
    inlier_ratio: float | None
    correlation: float | None
    overlap_ratio: float


def select_reference(
    sources: Sequence[str | Path],
    *,
    config: ReferenceSelectionConfig | None = None,
    align: AlignmentCallback | None = None,
) -> ReferenceSelectionResult:
    """Choose an aligned medoid from same-size captures of one PCB SKU.

    Selection is invariant to ``sources`` order.  A candidate must first pass
    focus, clipping, and exposure gates, then align successfully to enough
    quality-passing peers.  Missing pairwise alignments receive the maximum
    normalized image-distance penalty, so a weakly connected candidate cannot
    beat a well-supported medoid.

    Args:
        sources: Paths to real captured images. Arrays are intentionally not
            accepted because enrollment requires content-addressable lineage.
        config: Optional quality and consensus gates.
        align: Optional callback with the same source/reference contract as
            :meth:`PCBAligner.align`; useful for a calibrated production
            aligner and for deterministic tests.

    Raises:
        ReferenceSelectionError: if decoding, batch validation, quality, or
            alignment consensus fails. Error text and reports never expose an
            absolute path.
    """

    settings = config or ReferenceSelectionConfig()
    if isinstance(sources, (str, Path)):
        raise TypeError("sources must be a sequence of image paths")

    source_paths = [Path(source).expanduser() for source in sources]
    if not source_paths:
        raise ReferenceSelectionError("No reference candidates were supplied")

    loaded: list[tuple[Path, str, str, np.ndarray]] = []
    for path in source_paths:
        basename = path.name or "<unnamed>"
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ReferenceSelectionError(
                f"Could not read reference candidate {basename!r}"
            ) from exc
        sha256 = hashlib.sha256(payload).hexdigest()
        try:
            image = load_image(payload)
        except Exception as exc:
            raise ReferenceSelectionError(
                f"Could not decode reference candidate {basename!r} (sha256={sha256})"
            ) from exc
        loaded.append((path, basename, sha256, image))

    # Content identity, not caller order or directory spelling, drives every
    # later iteration and tie-break.
    loaded.sort(key=lambda item: (item[2], item[1].casefold(), item[1]))
    content_hashes = [item[2] for item in loaded]
    if len(set(content_hashes)) != len(content_hashes):
        raise ReferenceSelectionError(
            "The candidate list contains duplicate image content; peers must be independent captures"
        )

    shapes = [tuple(int(value) for value in item[3].shape[:2]) for item in loaded]
    expected_size = sorted(Counter(shapes).items(), key=lambda item: (-item[1], item[0]))[0][0]
    diagnostic_size = _diagnostic_size(expected_size, settings.diagnostic_max_side)

    candidates: list[_Candidate] = []
    for (path, basename, sha256, image), image_size in zip(loaded, shapes, strict=True):
        diagnostic = _resize_diagnostic(image, settings.diagnostic_max_side)
        quality = _quality_metrics(diagnostic)
        reasons: list[str] = []
        if image_size != expected_size:
            reasons.append(
                "image_size_mismatch:"
                f"{image_size[0]}x{image_size[1]}!={expected_size[0]}x{expected_size[1]}"
            )
        candidates.append(
            _Candidate(
                path=path,
                basename=basename,
                sha256=sha256,
                image=image,
                diagnostic_image=diagnostic,
                quality=quality,
                rejection_reasons=reasons,
                distances=[],
                methods=Counter(),
                inlier_ratios=[],
                correlations=[],
                overlap_ratios=[],
            )
        )

    if any(size != expected_size for size in shapes):
        for candidate in candidates:
            if not candidate.rejection_reasons:
                candidate.rejection_reasons.append("batch_image_size_mismatch")
        report = _build_report(
            candidates,
            source_count=len(candidates),
            image_size=expected_size,
            diagnostic_image_size=diagnostic_size,
            quality_candidate_count=0,
            required_peers=0,
            selected=None,
            failure_reason="image_size_mismatch",
        )
        raise ReferenceSelectionError(
            "Reference candidates must all have the same pixel dimensions",
            report,
        )

    if len(candidates) < settings.min_images:
        reason = f"insufficient_source_images:{len(candidates)}/{settings.min_images}"
        for candidate in candidates:
            candidate.rejection_reasons.append(reason)
        report = _build_report(
            candidates,
            source_count=len(candidates),
            image_size=expected_size,
            diagnostic_image_size=diagnostic_size,
            quality_candidate_count=0,
            required_peers=0,
            selected=None,
            failure_reason="insufficient_source_images",
        )
        raise ReferenceSelectionError(
            f"At least {settings.min_images} reference candidates are required",
            report,
        )

    blur_gate = max(
        settings.min_blur_variance,
        float(np.median([candidate.quality.laplacian_variance for candidate in candidates]))
        * settings.min_relative_blur,
    )
    for candidate in candidates:
        if candidate.quality.laplacian_variance < blur_gate:
            candidate.rejection_reasons.append("blur_below_batch_gate")
        if candidate.quality.clipped_ratio > settings.max_clipped_ratio:
            candidate.rejection_reasons.append("clipping_ratio_above_gate")
        if candidate.quality.mean_luma < settings.min_mean_luma:
            candidate.rejection_reasons.append("exposure_below_gate")
        elif candidate.quality.mean_luma > settings.max_mean_luma:
            candidate.rejection_reasons.append("exposure_above_gate")

    quality_candidates = [candidate for candidate in candidates if not candidate.rejection_reasons]
    available_peers = max(0, len(quality_candidates) - 1)
    required_peers = max(
        settings.min_valid_peers,
        int(math.ceil(settings.min_peer_ratio * available_peers)),
    )
    if required_peers > available_peers:
        reason = f"insufficient_quality_peers:{available_peers}/{required_peers}"
        for candidate in quality_candidates:
            candidate.rejection_reasons.append(reason)
        report = _build_report(
            candidates,
            source_count=len(candidates),
            image_size=expected_size,
            diagnostic_image_size=diagnostic_size,
            quality_candidate_count=len(quality_candidates),
            required_peers=required_peers,
            selected=None,
            failure_reason="insufficient_quality_candidates",
        )
        raise ReferenceSelectionError(
            "Too few quality-passing candidates remain for reference consensus",
            report,
        )

    align_callback = align or PCBAligner().align
    for first_index, first in enumerate(quality_candidates):
        for second in quality_candidates[first_index + 1 :]:
            observation = _observe_pair(
                first.diagnostic_image,
                second.diagnostic_image,
                align_callback,
                settings.min_valid_overlap_ratio,
            )
            if observation is None:
                continue
            for candidate in (first, second):
                candidate.distances.append(observation.distance)
                candidate.methods[observation.method] += 1
                candidate.overlap_ratios.append(observation.overlap_ratio)
                if observation.inlier_ratio is not None:
                    candidate.inlier_ratios.append(observation.inlier_ratio)
                if observation.correlation is not None:
                    candidate.correlations.append(observation.correlation)

    eligible: list[_Candidate] = []
    for candidate in quality_candidates:
        peer_count = len(candidate.distances)
        if peer_count < required_peers:
            candidate.rejection_reasons.append(
                f"insufficient_aligned_peers:{peer_count}/{required_peers}"
            )
            continue
        missing_peers = available_peers - peer_count
        candidate.medoid_score = (
            float(sum(candidate.distances)) + float(missing_peers)
        ) / max(1, available_peers)
        eligible.append(candidate)

    if not eligible:
        report = _build_report(
            candidates,
            source_count=len(candidates),
            image_size=expected_size,
            diagnostic_image_size=diagnostic_size,
            quality_candidate_count=len(quality_candidates),
            required_peers=required_peers,
            selected=None,
            failure_reason="insufficient_alignment_consensus",
        )
        raise ReferenceSelectionError(
            "No reference candidate aligned successfully to enough peers",
            report,
        )

    selected = min(
        eligible,
        key=lambda candidate: (
            float(candidate.medoid_score),
            _percentile(candidate.distances, 95.0) or 0.0,
            candidate.sha256,
            candidate.basename.casefold(),
            candidate.basename,
        ),
    )
    report = _build_report(
        candidates,
        source_count=len(candidates),
        image_size=expected_size,
        diagnostic_image_size=diagnostic_size,
        quality_candidate_count=len(quality_candidates),
        required_peers=required_peers,
        selected=selected,
        failure_reason=None,
    )
    return ReferenceSelectionResult(reference_path=selected.path, report=report)


def _quality_metrics(image: np.ndarray) -> ReferenceQualityMetrics:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = gray.astype(np.float64) / 255.0
    dark_ratio = float(np.mean(gray <= 2))
    bright_ratio = float(np.mean(gray >= 253))
    mean_luma = float(np.mean(normalized))
    return ReferenceQualityMetrics(
        laplacian_variance=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        dark_clipped_ratio=dark_ratio,
        bright_clipped_ratio=bright_ratio,
        clipped_ratio=dark_ratio + bright_ratio,
        mean_luma=mean_luma,
        exposure_deviation=abs(mean_luma - 0.5),
        p01_luma=float(np.percentile(normalized, 1.0)),
        p99_luma=float(np.percentile(normalized, 99.0)),
    )


def _observe_pair(
    first: np.ndarray,
    second: np.ndarray,
    align: AlignmentCallback,
    min_overlap_ratio: float,
) -> _PairObservation | None:
    # One deterministic direction is enough for a symmetric medoid distance;
    # retrying the inverse direction prevents an asymmetric feature failure
    # from discarding an otherwise valid pair.
    for source, reference in ((second, first), (first, second)):
        try:
            result = align(source, reference)
        except Exception:
            continue
        try:
            if not bool(getattr(result, "success", False)):
                continue
            aligned = getattr(result, "image", None)
            if not isinstance(aligned, np.ndarray) or aligned.shape[:2] != reference.shape[:2]:
                continue

            valid_mask = np.ones(reference.shape[:2], dtype=bool)
            homography = getattr(result, "homography", None)
            if homography is not None:
                matrix = np.asarray(homography, dtype=np.float64)
                if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                    continue
                condition = float(np.linalg.cond(matrix))
                if not math.isfinite(condition) or condition > 1e12:
                    continue
                source_mask = np.full(source.shape[:2], 255, dtype=np.uint8)
                warped_mask = cv2.warpPerspective(
                    source_mask,
                    matrix,
                    (reference.shape[1], reference.shape[0]),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                valid_mask = warped_mask > 0
            overlap_ratio = float(np.mean(valid_mask))
            if overlap_ratio < min_overlap_ratio or not np.any(valid_mask):
                continue

            aligned_gray = cv2.cvtColor(load_image(aligned), cv2.COLOR_BGR2GRAY)
            reference_gray = cv2.cvtColor(load_image(reference), cv2.COLOR_BGR2GRAY)
            difference = cv2.absdiff(aligned_gray, reference_gray)
            distance = float(np.mean(difference[valid_mask]) / 255.0)
            if not math.isfinite(distance):
                continue
            return _PairObservation(
                distance=distance,
                method=str(getattr(result, "method", "unknown")),
                inlier_ratio=_finite_optional(getattr(result, "inlier_ratio", None)),
                correlation=_finite_optional(getattr(result, "correlation", None)),
                overlap_ratio=overlap_ratio,
            )
        except Exception:
            continue
    return None


def _build_report(
    candidates: Sequence[_Candidate],
    *,
    source_count: int,
    image_size: tuple[int, int] | None,
    diagnostic_image_size: tuple[int, int] | None,
    quality_candidate_count: int,
    required_peers: int,
    selected: _Candidate | None,
    failure_reason: str | None,
) -> ReferenceSelectionReport:
    total_peers = max(0, quality_candidate_count - 1)
    candidate_reports: list[ReferenceCandidateReport] = []
    for candidate in candidates:
        aligned_peers = len(candidate.distances)
        eligible = candidate.medoid_score is not None and not candidate.rejection_reasons
        candidate_reports.append(
            ReferenceCandidateReport(
                basename=candidate.basename,
                sha256=candidate.sha256,
                image_size=tuple(int(value) for value in candidate.image.shape[:2]),
                quality=candidate.quality,
                selected=candidate is selected,
                eligible=eligible,
                rejection_reasons=tuple(candidate.rejection_reasons),
                aligned_peer_count=aligned_peers,
                total_peer_count=total_peers,
                alignment_success_ratio=(
                    float(aligned_peers / total_peers) if total_peers else 0.0
                ),
                mean_distance=_mean(candidate.distances),
                median_distance=_median(candidate.distances),
                p95_distance=_percentile(candidate.distances, 95.0),
                medoid_score=candidate.medoid_score,
                alignment_method_counts=tuple(sorted(candidate.methods.items())),
                median_inlier_ratio=_median(candidate.inlier_ratios),
                median_correlation=_median(candidate.correlations),
                median_overlap_ratio=_median(candidate.overlap_ratios),
            )
        )
    return ReferenceSelectionReport(
        algorithm="aligned_source_medoid_v1",
        source_count=source_count,
        image_size=image_size,
        diagnostic_image_size=diagnostic_image_size,
        quality_candidate_count=quality_candidate_count,
        required_peers=required_peers,
        selected_basename=None if selected is None else selected.basename,
        selected_sha256=None if selected is None else selected.sha256,
        candidates=tuple(candidate_reports),
        failure_reason=failure_reason,
    )


def _resize_diagnostic(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, float(max_side) / max(height, width))
    if scale == 1.0:
        return image.copy()
    output_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)


def _diagnostic_size(image_size: tuple[int, int], max_side: int) -> tuple[int, int]:
    height, width = image_size
    scale = min(1.0, float(max_side) / max(height, width))
    return (max(1, int(round(height * scale))), max(1, int(round(width * scale))))


def _finite_optional(value: object) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _median(values: Sequence[float]) -> float | None:
    return None if not values else float(np.median(values))


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    return None if not values else float(np.percentile(values, percentile))


def _optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def _size_dict(size: tuple[int, int] | None) -> dict[str, int] | None:
    if size is None:
        return None
    return {"height": int(size[0]), "width": int(size[1])}


__all__ = [
    "AlignmentCallback",
    "ReferenceCandidateReport",
    "ReferenceQualityMetrics",
    "ReferenceSelectionConfig",
    "ReferenceSelectionError",
    "ReferenceSelectionReport",
    "ReferenceSelectionResult",
    "select_reference",
]
