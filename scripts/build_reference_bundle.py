"""Build a review-only Golden recipe and draft PnP from repeated OK images.

This command is deliberately conservative:

* one real source image is selected as the Golden medoid; no composite is made;
* every detector observation is aligned into that Golden canvas before voting;
* the complete pixel consensus is retained as a detector audit, not claimed as
  component ground truth; a millimetre PnP is emitted only when the caller
  supplies a measured/nominal board outline, and remains explicitly
  ``NEEDS_REVIEW``;
* demo grid anchors and unverified metrology keep the recipe ineligible for
  production until a human approves the physical setup.

Example for the deterministic VisA PCB2 bootstrap set::

    python scripts/build_reference_bundle.py ^
        datasets/reference_sets/visa_pcb2_30 ^
        --output golden_recipes/visa_pcb2/bottom/draft_v1 ^
        --model models/active/detector/best.onnx ^
        --board-id visa_pcb2 --side bottom

Omitting both board dimensions deliberately builds a pixel-only bootstrap.  It
contains a lossless Golden, the complete pixel consensus and a selected-Golden-
anchored PnP review queue in ``golden_board_pixels``, but no recipe or invented
millimetre registration.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Callable, Mapping, Protocol, Sequence

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from aoi_pipeline import AOIPipeline, PipelineConfig  # noqa: E402
from aoi_pipeline.detection.detectors import detector_identifier  # noqa: E402
from aoi_pipeline.digitizer import (  # noqa: E402
    ConsensusComponent,
    ConsensusConfig,
    PnpConsensus,
    build_consensus,
    export_pixel_csv,
    export_pixel_json,
    export_placement_draft,
)
from aoi_pipeline.golden.enrollment import (  # noqa: E402
    ReferenceSelectionConfig,
    ReferenceSelectionResult,
    select_reference,
)
from aoi_pipeline.golden.recipe import (  # noqa: E402
    AppearanceThresholds,
    MetrologyCalibration,
    PositionTolerance,
    create_grid_alignment_recipe,
    create_recipe,
    validate_recipe_assets,
)
from aoi_pipeline.imaging.image_io import encode_image, load_image  # noqa: E402
from aoi_pipeline.models import (  # noqa: E402
    AlignmentResult,
    BoardRegion,
    BoundingBox,
    Detection,
)
from aoi_pipeline.solder.cad import CadRegistration, load_cad  # noqa: E402


BUNDLE_SCHEMA_VERSION = "aoi-reference-bundle/1.0"
REGISTRATION_SCHEMA_VERSION = "aoi-pnp-registration-draft/1.0"
BUNDLE_STATUS = "NEEDS_REVIEW"
PNP_PIXEL_REVIEW_SCHEMA_VERSION = "aoi-pnp-pixels-review-draft/1.0"
PNP_PIXEL_PROPOSAL_BASIS = (
    "median_consensus_cluster_with_selected_golden_observation"
)

_PNP_PIXEL_REVIEW_COLUMNS = (
    "schema_version",
    "artifact_status",
    "coordinate_space",
    "proposal_basis",
    "selected_golden_frame_id",
    "selected_golden_sha256",
    "selected_golden_observation_present",
    "board_id",
    "side",
    "designator",
    "designator_source",
    "class_label",
    "center_x_px",
    "center_y_px",
    "x1_px",
    "y1_px",
    "x2_px",
    "y2_px",
    "rotation_deg",
    "footprint",
    "observation_count",
    "frame_count",
    "support_ratio",
    "class_purity",
    "center_mad_px",
    "median_confidence",
    "consensus_status",
    "review_status",
    "review_reasons",
    "class_counts_json",
    "frame_ids_json",
)


class BundleError(RuntimeError):
    """Raised when a bundle cannot meet its review-data safety gates."""


@dataclass(frozen=True, slots=True)
class ReferenceInput:
    path: Path
    frame_id: str
    sha256: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class BundleConfig:
    reference_root: Path
    output_dir: Path
    model_path: Path
    board_id: str
    side: str
    board_width_mm: float | None = None
    board_height_mm: float | None = None
    expected_count: int = 30
    detector_confidence: float = 0.25
    min_aligned_ratio: float = 0.80
    diagnostic_max_side: int = 768
    cluster_radius_px: float = 24.0
    min_support_ratio: float = 0.80
    min_class_purity: float = 0.80
    use_upstream_alignment: bool = False

    def __post_init__(self) -> None:
        if int(self.expected_count) < 3:
            raise ValueError("expected_count must be at least 3")
        if self.side not in {"top", "bottom"}:
            raise ValueError("side must be 'top' or 'bottom'")
        if (self.board_width_mm is None) != (self.board_height_mm is None):
            raise ValueError(
                "board_width_mm and board_height_mm must be supplied together"
            )
        dimensions = ()
        if self.board_width_mm is not None and self.board_height_mm is not None:
            dimensions = (
                ("board_width_mm", self.board_width_mm),
                ("board_height_mm", self.board_height_mm),
            )
        for name, value in (*dimensions, ("cluster_radius_px", self.cluster_radius_px)):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name, value in (
            ("detector_confidence", self.detector_confidence),
            ("min_aligned_ratio", self.min_aligned_ratio),
            ("min_support_ratio", self.min_support_ratio),
            ("min_class_purity", self.min_class_purity),
        ):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

    @property
    def has_physical_outline(self) -> bool:
        """Whether both caller-supplied board dimensions are available."""

        return self.board_width_mm is not None and self.board_height_mm is not None


@dataclass(frozen=True, slots=True)
class BundleResult:
    output_dir: Path
    selected_source: Path
    accepted_frame_count: int
    component_count: int
    eligible_component_count: int


class DetectionRuntime(Protocol):
    """Small injectable surface used by the real build and unit tests."""

    @property
    def identifier(self) -> str: ...

    def align(self, image: np.ndarray, reference: np.ndarray) -> AlignmentResult: ...

    def localize(self, image: np.ndarray) -> BoardRegion: ...

    def detect(
        self,
        image: np.ndarray,
        board_region: BoardRegion,
        *,
        frame_id: str,
    ) -> list[Detection]: ...


class _PipelineRuntime:
    def __init__(self, config: BundleConfig) -> None:
        pipeline_config = PipelineConfig()
        pipeline_config.model_detector.confidence = float(
            config.detector_confidence
        )
        pipeline_config.model_detector.iou = 0.70
        pipeline_config.model_detector.end2end = False
        pipeline_config.tiling.mode = "off"
        pipeline_config.solder_defect_detection.enabled = False
        pipeline_config.solder_grading.enabled = False
        self.pipeline = AOIPipeline(pipeline_config, model_path=config.model_path)

    @property
    def identifier(self) -> str:
        return detector_identifier(self.pipeline.detector)

    def align(self, image: np.ndarray, reference: np.ndarray) -> AlignmentResult:
        return self.pipeline.align(image, reference)

    def localize(self, image: np.ndarray) -> BoardRegion:
        return self.pipeline.detect_board(image)

    def detect(
        self,
        image: np.ndarray,
        board_region: BoardRegion,
        *,
        frame_id: str,
    ) -> list[Detection]:
        return self.pipeline.detect_components(
            image,
            board_region,
            frame_id=frame_id,
        )


Selector = Callable[..., ReferenceSelectionResult]


def _validate_same_layout_policy(manifest: Mapping[str, Any]) -> None:
    """Reject an explicit manifest policy that makes consensus invalid.

    Legacy manifests did not declare these flags, so absence remains allowed.
    Newer source-set manifests place them under ``selection``; accepting the
    same declarations at the top level keeps the safety contract unambiguous.
    """

    scopes: list[tuple[str, Mapping[str, Any]]] = [("manifest", manifest)]
    selection = manifest.get("selection")
    if isinstance(selection, Mapping):
        scopes.append(("selection", selection))
    for scope_name, scope in scopes:
        if scope.get("same_layout_consensus_allowed") is False:
            raise BundleError(
                "Reference manifest explicitly forbids same-layout consensus "
                f"in {scope_name}"
            )
        if scope.get("different_layout_mixing_allowed") is True:
            raise BundleError(
                "Reference manifest explicitly allows different-layout mixing "
                f"in {scope_name}; consensus requires one layout"
            )


def _require_upstream_alignment_provenance(manifest: Mapping[str, Any]) -> str:
    """Return declared upstream registration or reject identity enrollment.

    Equal canvas sizes are not evidence that frames share a registered
    coordinate system. Identity transforms are therefore an explicit opt-in
    reserved for a dataset whose manifest states every part of that contract.
    The resulting registration is suitable only for this enrollment bundle;
    it does not validate registration from a production camera or fixture.
    """

    if manifest.get("source_kind") != "dataset_preprocessed_aligned":
        raise BundleError(
            "Upstream alignment requires "
            "source_kind='dataset_preprocessed_aligned'"
        )
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise BundleError("Upstream alignment requires a selection manifest object")
    if selection.get("same_layout_consensus_allowed") is not True:
        raise BundleError(
            "Upstream alignment requires "
            "selection.same_layout_consensus_allowed=true"
        )
    if selection.get("different_layout_mixing_allowed") is not False:
        raise BundleError(
            "Upstream alignment requires "
            "selection.different_layout_mixing_allowed=false"
        )
    board = manifest.get("board")
    if not isinstance(board, Mapping):
        raise BundleError("Upstream alignment requires a board manifest object")
    registration = board.get("upstream_registration")
    if not isinstance(registration, str) or not registration.strip():
        raise BundleError(
            "Upstream alignment requires non-empty board.upstream_registration"
        )
    return registration.strip()


def load_reference_inputs(
    reference_root: str | Path,
    *,
    expected_count: int = 30,
) -> tuple[list[ReferenceInput], Mapping[str, Any], str]:
    """Validate and return manifest-addressed images without trusting paths.

    The manifest is the source of ordering and identity. File contents,
    dimensions, labels and unique hashes are checked again here so a tampered
    reference set cannot silently produce a new recipe.
    """

    root = Path(reference_root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError("Reference set has no readable manifest.json") from exc
    if not isinstance(manifest, Mapping):
        raise BundleError("Reference manifest must be a JSON object")
    _validate_same_layout_policy(manifest)
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != int(expected_count):
        raise BundleError(
            f"Reference manifest must contain exactly {expected_count} files"
        )

    inputs: list[ReferenceInput] = []
    seen_hashes: set[str] = set()
    seen_ids: set[str] = set()
    expected_size: tuple[int, int] | None = None
    for entry in files:
        if not isinstance(entry, Mapping):
            raise BundleError("Reference manifest contains a non-object file entry")
        relative_text = str(entry.get("path", ""))
        relative = PurePosixPath(relative_text)
        if (
            not relative_text
            or relative.is_absolute()
            or ".." in relative.parts
            or relative_text != relative.as_posix()
        ):
            raise BundleError("Reference manifest contains an unsafe file path")
        path = (root / Path(*relative.parts)).resolve()
        try:
            path.relative_to(root)
            payload = path.read_bytes()
        except (ValueError, OSError) as exc:
            raise BundleError(
                f"Reference image {relative.name!r} is missing or outside the set"
            ) from exc

        expected_sha = str(entry.get("sha256", "")).lower()
        actual_sha = sha256(payload).hexdigest()
        if expected_sha != actual_sha:
            raise BundleError(f"Reference image {relative.name!r} failed SHA-256")
        if actual_sha in seen_hashes:
            raise BundleError("Reference images must have unique content hashes")
        seen_hashes.add(actual_sha)
        if entry.get("label") != 0 or isinstance(entry.get("label"), bool):
            raise BundleError("Every reference image must have upstream label=0")

        try:
            image = load_image(payload)
        except Exception as exc:
            raise BundleError(
                f"Reference image {relative.name!r} could not be decoded"
            ) from exc
        height, width = image.shape[:2]
        if (width, height) != (entry.get("width"), entry.get("height")):
            raise BundleError(
                f"Reference image {relative.name!r} dimensions do not match manifest"
            )
        if expected_size is None:
            expected_size = (width, height)
        elif expected_size != (width, height):
            raise BundleError("Reference images must share one pixel canvas size")

        frame_id = relative.name
        if frame_id in seen_ids:
            raise BundleError("Reference image basenames must be unique")
        seen_ids.add(frame_id)
        inputs.append(
            ReferenceInput(
                path=path,
                frame_id=frame_id,
                sha256=actual_sha,
                width=width,
                height=height,
            )
        )

    return inputs, manifest, sha256(manifest_bytes).hexdigest()


def order_board_quad(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Return board corners as TL, TR, BR, BL in image coordinates."""

    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if array.shape != (4, 2) or not np.all(np.isfinite(array)):
        raise BundleError("Board localization must provide four finite corners")
    sums = array[:, 0] + array[:, 1]
    differences = array[:, 1] - array[:, 0]
    indices = (
        int(np.argmin(sums)),
        int(np.argmin(differences)),
        int(np.argmax(sums)),
        int(np.argmax(differences)),
    )
    if len(set(indices)) != 4:
        raise BundleError("Board corners are degenerate or ambiguously ordered")
    ordered = array[list(indices)]
    if abs(float(cv2.contourArea(ordered.astype(np.float32)))) < 1.0:
        raise BundleError("Board localization polygon has negligible area")
    return ordered


def provisional_registration(
    board_quad_px: Sequence[Sequence[float]],
    *,
    board_width_mm: float,
    board_height_mm: float,
) -> tuple[np.ndarray, CadRegistration, tuple[float, float]]:
    """Create an explicitly provisional Golden-pixel to nominal-mm mapping.

    The origin is the visually top-left board-outline corner, X grows toward
    the visually top-right corner and Y grows toward the visually bottom-left
    corner. On a photographed bottom side this convention may be mirrored
    relative to manufacturing CAD; human review is mandatory.
    """

    quad = order_board_quad(board_quad_px).astype(np.float32)
    width_mm = float(board_width_mm)
    height_mm = float(board_height_mm)
    mm_quad = np.float32(
        [[0.0, 0.0], [width_mm, 0.0], [width_mm, height_mm], [0.0, height_mm]]
    )
    pixel_to_mm = cv2.getPerspectiveTransform(quad, mm_quad).astype(np.float64)
    if not np.all(np.isfinite(pixel_to_mm)) or np.linalg.matrix_rank(pixel_to_mm) < 3:
        raise BundleError("Could not derive a finite board-outline homography")
    mm_to_pixel = np.linalg.inv(pixel_to_mm)
    registration = CadRegistration(
        matrix=mm_to_pixel,
        method="nominal_board_outline_homography_NEEDS_REVIEW",
        inlier_ratio=0.0,
        residual_px=0.0,
        matched_points=4,
        ambiguous=True,
    )
    px_per_mm_x = (
        np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])
    ) / (2.0 * width_mm)
    px_per_mm_y = (
        np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])
    ) / (2.0 * height_mm)
    return pixel_to_mm, registration, (
        float(px_per_mm_x),
        float(px_per_mm_y),
    )


def _alignment_overlap(
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    homography: np.ndarray,
) -> float:
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return 0.0
    mask = np.full(source_shape[:2], 255, dtype=np.uint8)
    warped = cv2.warpPerspective(
        mask,
        matrix,
        (target_shape[1], target_shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return float(np.count_nonzero(warped) / warped.size)


def _anchor_fractions(board_region: BoardRegion, image: np.ndarray) -> tuple[tuple[float, float], ...]:
    height, width = image.shape[:2]
    bbox = board_region.bbox.clamp(width, height)
    fractions: list[tuple[float, float]] = []
    for fy in (0.20, 0.50, 0.80):
        for fx in (0.20, 0.50, 0.80):
            x = bbox.x1 + fx * bbox.width
            y = bbox.y1 + fy * bbox.height
            fractions.append((x / max(1, width - 1), y / max(1, height - 1)))
    return tuple(fractions)


def _consensus_detections(consensus: PnpConsensus) -> list[Detection]:
    return [
        Detection(
            label=component.label,
            confidence=component.median_confidence,
            bbox=component.bbox,
            source="multi_frame_consensus",
            metadata={
                "synthetic_designator": component.designator,
                "support_ratio": component.support_ratio,
                "class_purity": component.class_purity,
                "center_mad_px": component.center_mad_px,
            },
        )
        for component in consensus.eligible_components
    ]


def _draw_consensus_overlay(
    image: np.ndarray,
    consensus: PnpConsensus,
    board_quad: np.ndarray,
) -> np.ndarray:
    canvas = image.copy()
    cv2.polylines(
        canvas,
        [np.round(board_quad).astype(np.int32)],
        True,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )
    for component in consensus.components:
        colour = (40, 210, 40) if component.eligible_for_placement else (0, 165, 255)
        x1, y1, x2, y2 = component.bbox.to_int()
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        text = (
            f"{component.designator} {component.label} "
            f"s={component.support_ratio:.2f} p={component.class_purity:.2f}"
        )
        cv2.putText(
            canvas,
            text,
            (max(0, x1), max(14, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            colour,
            1,
            cv2.LINE_AA,
        )
    return canvas


def _golden_anchored_components(
    consensus: PnpConsensus,
    *,
    selected_frame_id: str,
) -> tuple[ConsensusComponent, ...]:
    """Return consensus sites backed by exactly one selected-Golden frame.

    Consensus association is one-to-one within each frame.  Rechecking the
    exported summary here keeps the review PnP subset honest if that upstream
    contract is changed or a hand-constructed consensus reaches this builder.
    The returned geometry remains the multi-frame median geometry; the selected
    Golden observation is only an anchor proving that the proposed site is
    visible in the actual Golden source.
    """

    selected: list[ConsensusComponent] = []
    for component in consensus.components:
        frame_ids = tuple(component.frame_ids)
        if len(frame_ids) != len(set(frame_ids)):
            raise BundleError(
                f"Consensus component {component.designator} contains duplicate "
                "frame evidence"
            )
        if component.observation_count != len(frame_ids):
            raise BundleError(
                f"Consensus component {component.designator} observation/frame "
                "summary is inconsistent"
            )
        selected_observations = frame_ids.count(selected_frame_id)
        if selected_observations > 1:
            raise BundleError(
                f"Consensus component {component.designator} contains more than "
                "one selected-Golden observation"
            )
        if selected_observations == 1:
            selected.append(component)
    return tuple(selected)


def _safe_csv_text(value: Any) -> str:
    text = str(value)
    return f"'{text}" if text.startswith(("=", "+", "-", "@", "\t", "\r")) else text


def _write_pnp_pixel_review_csv(
    destination: Path,
    components: Sequence[ConsensusComponent],
    *,
    coordinate_space: str,
    selected_frame_id: str,
    selected_sha256: str,
    board_id: str,
    side: str,
) -> None:
    """Write a pixel-native proposal queue that never claims CAD authority."""

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=_PNP_PIXEL_REVIEW_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        for component in components:
            reasons = (
                "synthetic_auto_designator",
                "rotation_not_estimated",
                "footprint_unknown",
                "selected_golden_presence_not_identity_verification",
                *component.review_reasons,
            )
            writer.writerow(
                {
                    "schema_version": PNP_PIXEL_REVIEW_SCHEMA_VERSION,
                    "artifact_status": BUNDLE_STATUS,
                    "coordinate_space": coordinate_space,
                    "proposal_basis": PNP_PIXEL_PROPOSAL_BASIS,
                    "selected_golden_frame_id": _safe_csv_text(selected_frame_id),
                    "selected_golden_sha256": selected_sha256,
                    "selected_golden_observation_present": "true",
                    "board_id": _safe_csv_text(board_id),
                    "side": side,
                    "designator": _safe_csv_text(component.designator),
                    "designator_source": "synthetic_auto",
                    "class_label": _safe_csv_text(component.label),
                    "center_x_px": f"{component.center_px[0]:.6f}",
                    "center_y_px": f"{component.center_px[1]:.6f}",
                    "x1_px": f"{component.bbox.x1:.6f}",
                    "y1_px": f"{component.bbox.y1:.6f}",
                    "x2_px": f"{component.bbox.x2:.6f}",
                    "y2_px": f"{component.bbox.y2:.6f}",
                    "rotation_deg": "",
                    "footprint": "",
                    "observation_count": component.observation_count,
                    "frame_count": component.frame_count,
                    "support_ratio": f"{component.support_ratio:.6f}",
                    "class_purity": f"{component.class_purity:.6f}",
                    "center_mad_px": f"{component.center_mad_px:.6f}",
                    "median_confidence": f"{component.median_confidence:.6f}",
                    "consensus_status": component.consensus_status,
                    "review_status": BUNDLE_STATUS,
                    "review_reasons": ";".join(dict.fromkeys(reasons)),
                    "class_counts_json": json.dumps(
                        dict(component.class_counts),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "frame_ids_json": json.dumps(
                        list(component.frame_ids),
                        ensure_ascii=False,
                    ),
                }
            )


def _draw_pnp_review_overlay(
    image: np.ndarray,
    components: Sequence[ConsensusComponent],
    board_quad: np.ndarray,
) -> np.ndarray:
    """Draw only selected-Golden-anchored review proposals."""

    canvas = image.copy()
    cv2.polylines(
        canvas,
        [np.round(board_quad).astype(np.int32)],
        True,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )
    colour = (255, 180, 0)
    for component in components:
        x1, y1, x2, y2 = component.bbox.to_int()
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            canvas,
            f"{component.designator} {component.label} NEEDS_REVIEW",
            (max(0, x1), max(14, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            colour,
            1,
            cv2.LINE_AA,
        )
    return canvas


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "byte_size": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }


def _write_review_notes(
    destination: Path,
    *,
    selected_basename: str,
    board_width_mm: float | None,
    board_height_mm: float | None,
) -> None:
    has_outline = board_width_mm is not None and board_height_mm is not None
    if has_outline:
        physical_note = f"""- `placement_draft_NEEDS_REVIEW.csv` uses a provisional {board_width_mm:g} x
  {board_height_mm:g} mm outline. Its AUTO designators are not OCR RefDes, and
  rotation/footprint are intentionally blank.
- `registration_draft_NEEDS_REVIEW.json` uses visual top-left as origin, X to
  visual right and Y down. Confirm mirroring for the photographed board side.
- Recipe anchors are demo grid patches and metrology is unverified, therefore
  `recipe.json` must remain `production_eligible=false`."""
    else:
        physical_note = """- No physical board dimensions or fiducial registration were supplied. The
  millimetre placement CSV, metrology registration and `recipe.json` are
  intentionally absent rather than populated with guessed values.
- Measure the real board and register fiducials/CAD before exporting PnP in mm
  or creating an inspection recipe. Until then, use only the pixel-native audit
  and review artifacts."""
    (destination / "README_FIRST.md").write_text(
        f"""# Draft Golden/PnP — review before use

Status: **NEEDS_REVIEW**. This bundle is not production-ready.

- Golden is one real source frame: `{selected_basename}`; `golden.png` is its
  lossless decoded copy, not a median/composite image.
- `consensus_components.json/csv` is the complete generated detector audit in
  `golden_board_pixels`; it is not human-verified component ground truth or PnP.
- `pnp_pixels_NEEDS_REVIEW.csv` is a non-authoritative proposal queue. It contains
  exactly the consensus sites observed in the selected Golden frame, while its
  centers and boxes remain multi-frame medians. Synthetic designators, rotation,
  footprint and component identity still require human review.
- `overlay_pnp_NEEDS_REVIEW.png` draws only that selected-Golden-anchored subset.
{physical_note}

Green boxes in `overlay_consensus.png` passed the configured multi-frame
support/purity gate; orange boxes did not. Inclusion in
`pnp_pixels_NEEDS_REVIEW.csv` is instead based on presence in the selected
Golden, so either gate colour can occur. Every PnP review row still requires
human verification.
""",
        encoding="utf-8",
    )
    (destination / "NEEDS_REVIEW.md").write_text(
        """# Approval checklist

- [ ] Confirm all 30 frames are the intended SKU and photographed side.
- [ ] Confirm the selected Golden master and stable alignment features.
- [ ] Replace every synthetic `*_AUTO_*` designator with the real RefDes.
- [ ] Add/correct missing components, connector/header and crystal sites.
- [ ] Verify each component class, center, footprint and rotation.
- [ ] Measure the real board outline, origin, X/Y direction and bottom-side mirror.
- [ ] Replace nominal registration with measured fiducials/CAD registration.
- [ ] Verify pixels/mm using the actual camera, lens, height and fixture.
- [ ] Tune position/appearance thresholds from reviewed repeated OK captures.
- [ ] Complete component, pad/pin and solder-defect ground-truth labels.
- [ ] Rebuild and approve anchors before enabling production inspection.
""",
        encoding="utf-8",
    )


def build_reference_bundle(
    config: BundleConfig,
    *,
    selector: Selector = select_reference,
    runtime: DetectionRuntime | None = None,
) -> BundleResult:
    """Build all portable artifacts, failing before publishing a partial bundle."""

    inputs, source_manifest, source_manifest_sha = load_reference_inputs(
        config.reference_root,
        expected_count=config.expected_count,
    )
    upstream_registration: str | None = None
    if config.use_upstream_alignment:
        upstream_registration = _require_upstream_alignment_provenance(source_manifest)
    alignment_mode = (
        "upstream_dataset_identity_enrollment_only"
        if config.use_upstream_alignment
        else "runtime_frame_to_golden"
    )
    destination = config.output_dir.expanduser().resolve()
    stage = destination.with_name(f".{destination.name}.staging")
    if destination.exists():
        raise BundleError(f"Output already exists: {destination.name}")
    if stage.exists():
        raise BundleError(
            f"Staging output already exists: {stage.name}; inspect it before retrying"
        )
    stage.parent.mkdir(parents=True, exist_ok=True)

    selection_config = ReferenceSelectionConfig(
        min_images=config.expected_count,
        diagnostic_max_side=config.diagnostic_max_side,
    )
    selection = selector(
        [item.path for item in inputs],
        config=selection_config,
    )
    selected = next(
        (item for item in inputs if item.path.resolve() == selection.reference_path.resolve()),
        None,
    )
    if selected is None:
        raise BundleError("Reference selector returned a source outside the manifest")
    golden = load_image(selected.path)

    active_runtime = runtime or _PipelineRuntime(config)
    board_region = active_runtime.localize(golden)
    board_quad = order_board_quad(board_region.polygon)
    pixel_to_mm: np.ndarray | None = None
    registration: CadRegistration | None = None
    px_per_mm: tuple[float, float] | None = None
    if config.has_physical_outline:
        assert config.board_width_mm is not None
        assert config.board_height_mm is not None
        pixel_to_mm, registration, px_per_mm = provisional_registration(
            board_quad,
            board_width_mm=config.board_width_mm,
            board_height_mm=config.board_height_mm,
        )

    detections_by_frame: dict[str, list[Detection]] = {}
    alignment_rows: list[dict[str, Any]] = []
    accepted_frames = 0
    for item in inputs:
        image = load_image(item.path)
        if item.path.resolve() == selected.path.resolve():
            aligned = golden.copy()
            result = AlignmentResult(
                image=aligned,
                method="selected_reference_identity",
                success=True,
                homography=np.eye(3, dtype=np.float64),
                inlier_ratio=1.0,
                correlation=1.0,
                message="Selected Golden source; identity transform",
            )
        elif config.use_upstream_alignment:
            assert upstream_registration is not None
            aligned = image.copy()
            result = AlignmentResult(
                image=aligned,
                method="upstream_dataset_identity",
                success=True,
                homography=np.eye(3, dtype=np.float64),
                message=(
                    "Using manifest-declared upstream dataset registration for "
                    "enrollment only; this is not production camera/fixture "
                    f"registration. Provenance: {upstream_registration}"
                ),
            )
        else:
            result = active_runtime.align(image, golden)
            aligned = result.image
        homography = result.homography
        overlap = (
            0.0
            if homography is None
            else _alignment_overlap(image.shape, golden.shape, homography)
        )
        accepted = bool(
            result.success
            and result.method not in {"resize_fallback", "disabled", "not_requested"}
            and aligned.shape == golden.shape
            and overlap >= 0.80
        )
        if accepted:
            accepted_frames += 1
            detections_by_frame[item.frame_id] = active_runtime.detect(
                aligned,
                board_region,
                frame_id=item.frame_id,
            )
        else:
            # An alignment miss stays in the support denominator. It is not
            # removed merely because the detector never received the frame.
            detections_by_frame[item.frame_id] = []
        row = result.to_dict()
        row.update(
            {
                "frame_id": item.frame_id,
                "source_sha256": item.sha256,
                "accepted_for_detection": accepted,
                "canvas_overlap_ratio": overlap,
                "alignment_mode": alignment_mode,
            }
        )
        if config.use_upstream_alignment:
            row.update(
                {
                    # Identity is inherited from dataset provenance, not fitted
                    # here. Do not present invented feature-fit metrics.
                    "inlier_ratio": None,
                    "correlation": None,
                    "fit_metrics_status": "not_measured_upstream_identity",
                    "upstream_registration_provenance": upstream_registration,
                    "production_registration_eligible": False,
                }
            )
        alignment_rows.append(row)

    required_aligned = int(math.ceil(config.min_aligned_ratio * len(inputs)))
    if accepted_frames < required_aligned:
        raise BundleError(
            f"Only {accepted_frames}/{len(inputs)} frames aligned; "
            f"at least {required_aligned} are required"
        )

    consensus = build_consensus(
        detections_by_frame,
        config=ConsensusConfig(
            cluster_radius_px=config.cluster_radius_px,
            min_support_ratio=config.min_support_ratio,
            min_class_purity=config.min_class_purity,
        ),
        canvas_size=(golden.shape[1], golden.shape[0]),
    )
    if not consensus.eligible_components:
        raise BundleError("No component passed the multi-frame consensus gates")

    stage.mkdir()
    (config.reference_root.resolve() / "reference_selection.json").write_text(
        selection.report.to_json() + "\n",
        encoding="utf-8",
    )
    (stage / "reference_selection.json").write_text(
        selection.report.to_json() + "\n",
        encoding="utf-8",
    )
    _write_json(
        stage / "alignment_report.json",
        {
            "coordinate_space": "golden_board_pixels",
            "alignment_mode": alignment_mode,
            "alignment_provenance": (
                {
                    "kind": "manifest_declared_upstream_dataset_registration",
                    "description": upstream_registration,
                    "scope": "enrollment_only",
                    "production_registration_eligible": False,
                }
                if config.use_upstream_alignment
                else {"kind": "runtime_frame_to_selected_golden"}
            ),
            "selected_source": {
                "basename": selected.frame_id,
                "sha256": selected.sha256,
            },
            "summary": {
                "source_frame_count": len(inputs),
                "accepted_frame_count": accepted_frames,
                "required_aligned_count": required_aligned,
            },
            "frames": alignment_rows,
        },
    )
    export_pixel_json(consensus, stage / "consensus_components.json")
    export_pixel_csv(consensus, stage / "consensus_components.csv")
    pnp_review_components = _golden_anchored_components(
        consensus,
        selected_frame_id=selected.frame_id,
    )
    _write_pnp_pixel_review_csv(
        stage / "pnp_pixels_NEEDS_REVIEW.csv",
        pnp_review_components,
        coordinate_space=consensus.coordinate_space,
        selected_frame_id=selected.frame_id,
        selected_sha256=selected.sha256,
        board_id=config.board_id,
        side=config.side,
    )
    key_paths = [
        stage / "golden.png",
        stage / "reference_selection.json",
        stage / "alignment_report.json",
        stage / "consensus_components.json",
        stage / "consensus_components.csv",
        stage / "pnp_pixels_NEEDS_REVIEW.csv",
    ]
    if config.has_physical_outline:
        assert pixel_to_mm is not None
        assert registration is not None
        assert px_per_mm is not None
        assert config.board_width_mm is not None
        assert config.board_height_mm is not None
        export_placement_draft(
            consensus,
            stage / "placement_draft_NEEDS_REVIEW.csv",
            pixel_to_mm_homography=pixel_to_mm,
            side=config.side,
        )
        placement = load_cad(
            stage / "placement_draft_NEEDS_REVIEW.csv",
            fmt="placement_csv",
            units="mm",
            side=config.side,
        )
        if len(placement.components) != len(consensus.eligible_components):
            raise BundleError("Draft PnP did not round-trip through the CAD loader")

        registration_payload = {
            "schema_version": REGISTRATION_SCHEMA_VERSION,
            "status": BUNDLE_STATUS,
            "verified": False,
            "coordinate_space": "golden_board_pixels",
            "nominal_board": {
                "width_mm": float(config.board_width_mm),
                "height_mm": float(config.board_height_mm),
                "dimension_source": "caller_supplied_nominal_outline",
            },
            "draft_axis_convention": {
                "origin": "visually_top_left_board_outline_corner",
                "x_positive": "toward_visually_top_right_corner",
                "y_positive": "toward_visually_bottom_left_corner",
                "bottom_side_mirroring_verified": False,
            },
            "board_quad_px_tl_tr_br_bl": board_quad.tolist(),
            "pixel_to_mm_homography": pixel_to_mm.tolist(),
            "mm_to_pixel_homography": registration.matrix.tolist(),
            "cad_registration_diagnostic": registration.to_dict(),
        }
        _write_json(
            stage / "registration_draft_NEEDS_REVIEW.json",
            registration_payload,
        )

        alignment_recipe = create_grid_alignment_recipe(
            golden,
            stage,
            template_size_px=65,
            search_margin_px=48,
            grid_fractions=_anchor_fractions(board_region, golden),
        )
        recipe_result = create_recipe(
            golden,
            _consensus_detections(consensus),
            stage,
            board_id=config.board_id,
            side=config.side,
            metrology=MetrologyCalibration(
                px_per_mm[0],
                px_per_mm[1],
                verified=False,
            ),
            roi_padding_px=8,
            search_margin_px=16,
            position_tolerance=PositionTolerance(
                max_abs_dx_mm=0.50,
                max_abs_dy_mm=0.50,
                max_abs_angle_deg=None,
            ),
            appearance_thresholds=AppearanceThresholds(
                min_ssim=0.85,
                max_diff_ratio=0.15,
                max_edge_diff_ratio=0.18,
                max_blob_area_px=100,
                min_valid_overlap_ratio=0.80,
            ),
            alignment=alignment_recipe,
            model_identifiers={"component_detector": active_runtime.identifier},
            measurement_metadata={
                "bundle_status": BUNDLE_STATUS,
                "source_kind": source_manifest.get("source_kind", "unknown"),
                "source_manifest_sha256": source_manifest_sha,
                "golden_policy": "single_actual_source_decoded_to_lossless_png",
                "selected_source_basename": selected.frame_id,
                "selected_source_sha256": selected.sha256,
                "alignment_domain": (
                    alignment_mode
                    if config.use_upstream_alignment
                    else "source_frames_warped_to_selected_golden"
                ),
                "metrology_source": "nominal_board_outline_NEEDS_REVIEW",
                "starter_thresholds_verified": False,
            },
        )
        if recipe_result.recipe.production_eligible:
            raise BundleError("Draft recipe unexpectedly became production eligible")
        validate_recipe_assets(recipe_result.recipe, stage)
        key_paths.extend(
            (
                stage / "recipe.json",
                stage / "placement_draft_NEEDS_REVIEW.csv",
                stage / "registration_draft_NEEDS_REVIEW.json",
            )
        )
    else:
        # A Golden asset is still useful without physical calibration, but a
        # recipe is not: its required px/mm tolerances would be fabricated.
        (stage / "golden.png").write_bytes(encode_image(golden, ".png"))

    overlay = _draw_consensus_overlay(golden, consensus, board_quad)
    (stage / "overlay_consensus.png").write_bytes(encode_image(overlay, ".png"))
    pnp_overlay = _draw_pnp_review_overlay(
        golden,
        pnp_review_components,
        board_quad,
    )
    (stage / "overlay_pnp_NEEDS_REVIEW.png").write_bytes(
        encode_image(pnp_overlay, ".png")
    )
    _write_review_notes(
        stage,
        selected_basename=selected.frame_id,
        board_width_mm=config.board_width_mm,
        board_height_mm=config.board_height_mm,
    )

    key_paths.extend(
        (
            stage / "overlay_consensus.png",
            stage / "overlay_pnp_NEEDS_REVIEW.png",
            stage / "README_FIRST.md",
            stage / "NEEDS_REVIEW.md",
        )
    )
    _write_json(
        stage / "bundle_manifest.json",
        {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "status": BUNDLE_STATUS,
            "production_eligible": False,
            "board_id": config.board_id,
            "side": config.side,
            "bundle_mode": (
                "draft_recipe_with_nominal_outline"
                if config.has_physical_outline
                else "pixel_only_no_physical_calibration"
            ),
            "recipe_emitted": config.has_physical_outline,
            "deferred_without_physical_calibration": (
                []
                if config.has_physical_outline
                else [
                    "recipe.json",
                    "placement_draft_NEEDS_REVIEW.csv",
                    "registration_draft_NEEDS_REVIEW.json",
                ]
            ),
            "source_manifest_sha256": source_manifest_sha,
            "alignment_mode": alignment_mode,
            "source_frame_count": len(inputs),
            "aligned_frame_count": accepted_frames,
            "selected_source": {
                "basename": selected.frame_id,
                "sha256": selected.sha256,
            },
            "consensus": {
                "component_count": len(consensus.components),
                "eligible_component_count": len(consensus.eligible_components),
            },
            "pnp_pixels_review": {
                "schema_version": PNP_PIXEL_REVIEW_SCHEMA_VERSION,
                "status": BUNDLE_STATUS,
                "coordinate_space": consensus.coordinate_space,
                "proposal_basis": PNP_PIXEL_PROPOSAL_BASIS,
                "selected_golden_frame_id": selected.frame_id,
                "component_count": len(pnp_review_components),
                "authoritative": False,
                "verified": False,
            },
            "artifacts": [_artifact(path, stage) for path in key_paths],
        },
    )
    stage.rename(destination)
    return BundleResult(
        output_dir=destination,
        selected_source=selected.path,
        accepted_frame_count=accepted_frames,
        component_count=len(consensus.components),
        eligible_component_count=len(consensus.eligible_components),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a review-only Golden and pixel PnP from repeated OK images; "
            "emit a draft recipe only when both board dimensions are supplied."
        )
    )
    parser.add_argument("reference_set", help="Folder containing manifest.json and images/.")
    parser.add_argument("--output", required=True, help="New recipe bundle directory.")
    parser.add_argument("--model", required=True, help="Component detector .onnx/.pt artifact.")
    parser.add_argument("--board-id", required=True)
    parser.add_argument("--side", choices=("top", "bottom"), required=True)
    parser.add_argument(
        "--board-width-mm",
        type=float,
        help="Measured/nominal board width; requires --board-height-mm.",
    )
    parser.add_argument(
        "--board-height-mm",
        type=float,
        help="Measured/nominal board height; requires --board-width-mm.",
    )
    parser.add_argument("--expected-count", type=int, default=30)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--min-aligned-ratio", type=float, default=0.80)
    parser.add_argument("--diagnostic-max-side", type=int, default=768)
    parser.add_argument("--cluster-radius-px", type=float, default=24.0)
    parser.add_argument("--min-support-ratio", type=float, default=0.80)
    parser.add_argument("--min-class-purity", type=float, default=0.80)
    parser.add_argument(
        "--use-upstream-alignment",
        action="store_true",
        help=(
            "Use identity transforms only when manifest provenance explicitly "
            "declares a pre-aligned same-layout dataset; enrollment-only."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_reference_bundle(
            BundleConfig(
                reference_root=Path(args.reference_set),
                output_dir=Path(args.output),
                model_path=Path(args.model),
                board_id=args.board_id,
                side=args.side,
                board_width_mm=args.board_width_mm,
                board_height_mm=args.board_height_mm,
                expected_count=args.expected_count,
                detector_confidence=args.conf,
                min_aligned_ratio=args.min_aligned_ratio,
                diagnostic_max_side=args.diagnostic_max_side,
                cluster_radius_px=args.cluster_radius_px,
                min_support_ratio=args.min_support_ratio,
                min_class_purity=args.min_class_purity,
                use_upstream_alignment=args.use_upstream_alignment,
            )
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"Built {BUNDLE_STATUS} bundle -> {result.output_dir}\n"
        f"Golden source: {result.selected_source.name}; "
        f"aligned: {result.accepted_frame_count}; "
        f"consensus: {result.eligible_component_count}/{result.component_count} eligible"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
