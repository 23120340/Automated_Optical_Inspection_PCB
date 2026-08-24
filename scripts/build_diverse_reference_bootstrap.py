"""Build one review-only reference/PnP seed per distinct PCB image.

This command is intentionally different from ``build_reference_bundle.py``.
The latter aggregates repeated photographs of one SKU into one Golden canvas;
this command receives *different layouts* and never aligns or votes across
boards.  Every image remains in ``source_image_pixels`` and produces its own
Golden candidate plus a pixel-native PnP draft.

The TU Wien PCB DSLR annotations cover IC packages only.  They are preserved
as upstream oriented rectangles and merged with current detector proposals for
other component families.  Neither source provides manufacturing RefDes,
footprint, layer, physical scale or a verified PnP rotation, so every row stays
``NEEDS_REVIEW`` and no millimetre placement file is emitted.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
from typing import Any, Mapping, Protocol, Sequence

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from aoi_pipeline.config import ModelDetectorConfig  # noqa: E402
from aoi_pipeline.detection.detectors import (  # noqa: E402
    UltralyticsDetector,
    detector_identifier,
)
from aoi_pipeline.imaging.image_io import load_image  # noqa: E402
from aoi_pipeline.models import BoundingBox, Detection  # noqa: E402


BOOTSTRAP_SCHEMA_VERSION = "aoi-diverse-pcb-bootstrap/1.0"
PNP_SCHEMA_VERSION = "aoi-pnp-draft-source-pixels/1.0"
REFERENCE_STATUS = "GOLDEN_CANDIDATE_NEEDS_REVIEW"
REVIEW_STATUS = "NEEDS_REVIEW"
COORDINATE_SPACE = "source_image_pixels"
EXCLUDED_DETECTOR_LABELS = frozenset({"pad", "pads", "pin", "pins"})

_PREFIX_BY_LABEL: Mapping[str, str] = {
    "battery": "BT",
    "button": "SW",
    "buzzer": "BZ",
    "capacitor": "C",
    "clock": "Y",
    "connector": "J",
    "diode": "D",
    "display": "DS",
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

_PNP_COLUMNS = (
    "schema_version",
    "status",
    "board_id",
    "coordinate_space",
    "slot_id",
    "designator",
    "designator_source",
    "expected_class",
    "center_x_px",
    "center_y_px",
    "x1_px",
    "y1_px",
    "x2_px",
    "y2_px",
    "rotation_deg_observed",
    "rotation_source",
    "footprint",
    "layer",
    "proposal_source",
    "confidence",
    "upstream_text",
    "review_reasons",
)


class BootstrapError(RuntimeError):
    """Raised when source identity or an output safety contract is violated."""


@dataclass(frozen=True, slots=True)
class SourceBoard:
    board_id: str
    upstream_board_id: str
    image_path: Path
    mask_path: Path
    annotation_path: Path
    image_sha256: str
    mask_sha256: str
    annotation_sha256: str
    width: int
    height: int
    source_archive: str
    source_entry: str


@dataclass(frozen=True, slots=True)
class UpstreamIC:
    center_x: float
    center_y: float
    width: float
    height: float
    angle_deg: float
    text: str
    bbox: BoundingBox


@dataclass(frozen=True, slots=True)
class Proposal:
    label: str
    bbox: BoundingBox
    center_x: float
    center_y: float
    confidence: float
    proposal_source: str
    rotation_deg: float | None = None
    rotation_source: str = "unknown"
    upstream_text: str = ""


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    dataset_root: Path
    output_dir: Path
    model_path: Path
    expected_count: int = 30
    detector_confidence: float = 0.25
    preview_max_side: int = 1600

    def __post_init__(self) -> None:
        if int(self.expected_count) < 1:
            raise ValueError("expected_count must be positive")
        if not 0.0 <= float(self.detector_confidence) <= 1.0:
            raise ValueError("detector_confidence must be between 0 and 1")
        if int(self.preview_max_side) < 256:
            raise ValueError("preview_max_side must be at least 256")


class ProposalRuntime(Protocol):
    @property
    def identifier(self) -> str: ...

    def detect(
        self, image: np.ndarray, board_bbox: BoundingBox
    ) -> list[Detection]: ...


class _DetectorRuntime:
    def __init__(self, model_path: Path, confidence: float) -> None:
        self._confidence = float(confidence)
        self._empty_fallback_confidence = min(self._confidence, 0.10)
        config = ModelDetectorConfig(
            confidence=self._confidence,
            iou=0.70,
            max_detections=2000,
            end2end=False,
        )
        self.detector = UltralyticsDetector(model_path, config)
        self._identifier = detector_identifier(self.detector)

    @property
    def identifier(self) -> str:
        return self._identifier

    def detect(
        self, image: np.ndarray, board_bbox: BoundingBox
    ) -> list[Detection]:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = board_bbox.clamp(width, height).to_int()
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            raise BootstrapError("Board mask produced an empty detector crop")
        translated: list[Detection] = []
        raw = self.detector.detect(crop)
        used_empty_fallback = False
        if not raw and self._confidence > self._empty_fallback_confidence:
            raw = self.detector.detect(
                crop, confidence=self._empty_fallback_confidence
            )
            used_empty_fallback = True
        for item in raw:
            bbox = item.bbox.translated(x1, y1).clamp(width, height)
            if bbox.width <= 0 or bbox.height <= 0:
                continue
            translated.append(
                Detection(
                    label=item.label,
                    confidence=item.confidence,
                    bbox=bbox,
                    class_id=item.class_id,
                    source=item.source,
                    detection_id=item.detection_id,
                    metadata={
                        **item.metadata,
                        "detector_crop_xyxy": [x1, y1, x2, y2],
                        "coordinate_space": COORDINATE_SPACE,
                        "empty_result_fallback": used_empty_fallback,
                        "applied_confidence": (
                            self._empty_fallback_confidence
                            if used_empty_fallback
                            else self._confidence
                        ),
                    },
                )
            )
        return translated


def _portable_relative(value: Any, *, field: str) -> PurePosixPath:
    text = str(value or "")
    relative = PurePosixPath(text)
    if (
        not text
        or relative.is_absolute()
        or ".." in relative.parts
        or text != relative.as_posix()
    ):
        raise BootstrapError(f"Manifest field {field!r} is not a safe relative path")
    return relative


def _entry_path(entry: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = entry.get(key)
        if isinstance(value, Mapping):
            nested = value.get("path")
            if nested:
                return str(nested)
        elif value:
            return str(value)
    return ""


def load_source_boards(
    dataset_root: str | Path, *, expected_count: int = 30
) -> tuple[list[SourceBoard], Mapping[str, Any], str]:
    """Load and revalidate the downloader manifest without trusting its paths."""

    root = Path(dataset_root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapError("Dataset has no readable manifest.json") from exc
    if not isinstance(manifest, Mapping):
        raise BootstrapError("Dataset manifest must be a JSON object")

    raw_entries = manifest.get("boards", manifest.get("files"))
    if not isinstance(raw_entries, list) or len(raw_entries) != int(expected_count):
        raise BootstrapError(
            f"Dataset manifest must contain exactly {expected_count} board entries"
        )

    boards: list[SourceBoard] = []
    seen_board_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for index, raw in enumerate(raw_entries, start=1):
        if not isinstance(raw, Mapping):
            raise BootstrapError("Dataset manifest contains a non-object board entry")
        upstream_id = str(
            raw.get("upstream_board_id", raw.get("source_board_id", raw.get("pcb_id", "")))
        ).strip()
        board_id = str(raw.get("board_id") or f"pcb_dslr_{index:03d}").strip()
        if not board_id or board_id in seen_board_ids:
            raise BootstrapError("Dataset board_id values must be non-empty and unique")
        seen_board_ids.add(board_id)

        image_text = _entry_path(raw, "image_path", "image", "path")
        mask_text = _entry_path(raw, "mask_path", "mask")
        annotation_text = _entry_path(
            raw, "annotation_path", "annotation", "upstream_annotation"
        )
        relative_paths = {
            "image": _portable_relative(image_text, field="image_path"),
            "mask": _portable_relative(mask_text, field="mask_path"),
            "annotation": _portable_relative(annotation_text, field="annotation_path"),
        }
        resolved: dict[str, Path] = {}
        for name, relative in relative_paths.items():
            path = (root / Path(*relative.parts)).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise BootstrapError(f"{name} path escapes the dataset root") from exc
            if not path.is_file():
                raise BootstrapError(f"Missing {name} file for {board_id}: {relative}")
            resolved[name] = path

        payload = resolved["image"].read_bytes()
        actual_sha = sha256(payload).hexdigest()
        expected_sha = str(
            raw.get("image_sha256", raw.get("sha256", ""))
        ).lower()
        if expected_sha and expected_sha != actual_sha:
            raise BootstrapError(f"Image {relative_paths['image'].name!r} failed SHA-256")
        if actual_sha in seen_hashes:
            raise BootstrapError("The 30 boards must have unique image content")
        seen_hashes.add(actual_sha)
        mask_sha = sha256(resolved["mask"].read_bytes()).hexdigest()
        annotation_sha = sha256(resolved["annotation"].read_bytes()).hexdigest()
        for name, actual_asset_sha, expected_key in (
            ("mask", mask_sha, "mask_sha256"),
            ("annotation", annotation_sha, "annotation_sha256"),
        ):
            expected_asset_sha = str(raw.get(expected_key, "")).lower()
            if expected_asset_sha and expected_asset_sha != actual_asset_sha:
                raise BootstrapError(f"{name.title()} failed SHA-256 for {board_id}")
        try:
            image = load_image(payload)
            mask = cv2.imread(str(resolved["mask"]), cv2.IMREAD_GRAYSCALE)
        except Exception as exc:
            raise BootstrapError(f"Could not decode source assets for {board_id}") from exc
        if mask is None or mask.size == 0:
            raise BootstrapError(f"Could not decode board mask for {board_id}")
        height, width = image.shape[:2]
        if mask.shape[:2] != (height, width):
            raise BootstrapError(f"Mask dimensions do not match image for {board_id}")
        declared_width = raw.get("width", raw.get("width_px", width))
        declared_height = raw.get("height", raw.get("height_px", height))
        if (declared_width, declared_height) != (width, height):
            raise BootstrapError(f"Image dimensions do not match manifest for {board_id}")

        boards.append(
            SourceBoard(
                board_id=board_id,
                upstream_board_id=upstream_id or f"pcb{index}",
                image_path=resolved["image"],
                mask_path=resolved["mask"],
                annotation_path=resolved["annotation"],
                image_sha256=actual_sha,
                mask_sha256=mask_sha,
                annotation_sha256=annotation_sha,
                width=width,
                height=height,
                source_archive=str(raw.get("source_archive", raw.get("archive", ""))),
                source_entry=str(raw.get("source_entry", raw.get("zip_entry", ""))),
            )
        )
    boards.sort(key=lambda item: item.board_id)
    return boards, manifest, sha256(manifest_bytes).hexdigest()


def parse_upstream_ic_annotations(
    text: str, *, image_width: int, image_height: int
) -> list[UpstreamIC]:
    """Parse ``cx cy width height OpenCV-angle [chip text]`` records."""

    annotations: list[UpstreamIC] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise BootstrapError(f"Invalid upstream annotation line {line_number}")
        try:
            cx, cy, width, height, angle = (float(value) for value in parts[:5])
        except ValueError as exc:
            raise BootstrapError(
                f"Non-numeric upstream annotation line {line_number}"
            ) from exc
        values = (cx, cy, width, height, angle)
        if not all(math.isfinite(value) for value in values) or width <= 0 or height <= 0:
            raise BootstrapError(f"Invalid geometry on upstream annotation line {line_number}")
        corners = cv2.boxPoints(((cx, cy), (width, height), angle))
        x1 = float(np.min(corners[:, 0]))
        y1 = float(np.min(corners[:, 1]))
        x2 = float(np.max(corners[:, 0]))
        y2 = float(np.max(corners[:, 1]))
        bbox = BoundingBox(x1, y1, x2, y2).clamp(image_width, image_height)
        if bbox.width <= 0 or bbox.height <= 0:
            raise BootstrapError(
                f"Upstream annotation line {line_number} lies outside the image"
            )
        annotations.append(
            UpstreamIC(
                center_x=cx,
                center_y=cy,
                width=width,
                height=height,
                angle_deg=angle,
                text=" ".join(parts[5:]),
                bbox=bbox,
            )
        )
    annotations.sort(key=lambda item: (item.center_y, item.center_x, item.angle_deg))
    return annotations


def _intersection_over_smaller(first: BoundingBox, second: BoundingBox) -> float:
    x1, y1 = max(first.x1, second.x1), max(first.y1, second.y1)
    x2, y2 = min(first.x2, second.x2), min(first.y2, second.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    denominator = min(first.area, second.area)
    return intersection / denominator if denominator > 0 else 0.0


def merge_proposals(
    detections: Sequence[Detection], upstream_ics: Sequence[UpstreamIC]
) -> list[Proposal]:
    """Prefer upstream IC rectangles and retain detector proposals elsewhere."""

    proposals = [
        Proposal(
            label="ic",
            bbox=item.bbox,
            center_x=item.center_x,
            center_y=item.center_y,
            confidence=1.0,
            proposal_source="pcb_dslr_upstream_ic_annotation",
            rotation_deg=item.angle_deg,
            rotation_source="upstream_opencv_rotated_rect_not_verified_pnp",
            upstream_text=item.text,
        )
        for item in upstream_ics
    ]
    for detection in detections:
        label = str(detection.label).strip().lower()
        if not label or label in EXCLUDED_DETECTOR_LABELS:
            continue
        if label == "ic" and any(
            _intersection_over_smaller(detection.bbox, item.bbox) >= 0.35
            for item in upstream_ics
        ):
            continue
        cx, cy = detection.bbox.center
        proposals.append(
            Proposal(
                label=label,
                bbox=detection.bbox,
                center_x=float(cx),
                center_y=float(cy),
                confidence=float(detection.confidence),
                proposal_source=(
                    "current_component_detector_low_confidence_fallback"
                    if detection.metadata.get("empty_result_fallback")
                    else "current_component_detector"
                ),
            )
        )
    proposals.sort(
        key=lambda item: (
            round(item.center_y, 6),
            round(item.center_x, 6),
            item.label,
            -item.confidence,
        )
    )
    return proposals


def _board_bbox(mask: np.ndarray) -> BoundingBox:
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    if mask.ndim != 2:
        raise BootstrapError("Board segmentation mask must be single-channel")
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0 or ys.size == 0:
        raise BootstrapError("Board segmentation mask is empty")
    return BoundingBox(
        float(xs.min()),
        float(ys.min()),
        float(xs.max() + 1),
        float(ys.max() + 1),
    )


def _focus_score(image: np.ndarray, mask: np.ndarray) -> float:
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    if mask.ndim != 2:
        raise BootstrapError("Board segmentation mask must be single-channel")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    scale = min(1.0, 1600.0 / max(height, width))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    values = cv2.Laplacian(gray, cv2.CV_64F)[mask > 0]
    return float(values.var()) if values.size else 0.0


def _relative_posix(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path, start)).as_posix()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _proposal_rows(board_id: str, proposals: Sequence[Proposal]) -> list[dict[str, Any]]:
    counters: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for index, proposal in enumerate(proposals, start=1):
        prefix = _PREFIX_BY_LABEL.get(proposal.label, "AUTO")
        counters[prefix] = counters.get(prefix, 0) + 1
        designator = f"{prefix}_AUTO_{counters[prefix]:04d}"
        reasons = ["verify_refdes", "verify_footprint", "verify_layer"]
        if proposal.rotation_deg is None:
            reasons.append("rotation_unknown")
        else:
            reasons.append("rotation_not_manufacturing_verified")
        if proposal.proposal_source.endswith("low_confidence_fallback"):
            reasons.append("low_confidence_empty_result_fallback")
        rows.append(
            {
                "schema_version": PNP_SCHEMA_VERSION,
                "status": REVIEW_STATUS,
                "board_id": board_id,
                "coordinate_space": COORDINATE_SPACE,
                "slot_id": f"slot_{index:04d}",
                "designator": designator,
                "designator_source": "synthetic_auto_not_ocr",
                "expected_class": proposal.label,
                "center_x_px": f"{proposal.center_x:.6f}",
                "center_y_px": f"{proposal.center_y:.6f}",
                "x1_px": f"{proposal.bbox.x1:.6f}",
                "y1_px": f"{proposal.bbox.y1:.6f}",
                "x2_px": f"{proposal.bbox.x2:.6f}",
                "y2_px": f"{proposal.bbox.y2:.6f}",
                "rotation_deg_observed": (
                    "" if proposal.rotation_deg is None else f"{proposal.rotation_deg:.6f}"
                ),
                "rotation_source": proposal.rotation_source,
                "footprint": "",
                "layer": "",
                "proposal_source": proposal.proposal_source,
                "confidence": f"{proposal.confidence:.6f}",
                "upstream_text": proposal.upstream_text,
                "review_reasons": ";".join(reasons),
            }
        )
    return rows


def _preview(
    image: np.ndarray,
    board_bbox: BoundingBox,
    rows: Sequence[Mapping[str, Any]],
    *,
    max_side: int,
) -> np.ndarray:
    canvas = image.copy()
    x1, y1, x2, y2 = board_bbox.to_int()
    cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (255, 200, 0), 8)
    for row in rows:
        colour = (
            (80, 220, 80)
            if row["proposal_source"] == "pcb_dslr_upstream_ic_annotation"
            else (0, 170, 255)
        )
        bx1 = int(round(float(row["x1_px"])))
        by1 = int(round(float(row["y1_px"])))
        bx2 = int(round(float(row["x2_px"])))
        by2 = int(round(float(row["y2_px"])))
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), colour, 5)
    scale = min(1.0, float(max_side) / max(canvas.shape[:2]))
    if scale < 1.0:
        canvas = cv2.resize(
            canvas,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    return canvas


def _contact_sheet(previews: Sequence[tuple[str, np.ndarray]]) -> np.ndarray:
    tile_width, tile_height = 360, 260
    columns = 5
    rows = math.ceil(len(previews) / columns)
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 245, np.uint8)
    for index, (board_id, image) in enumerate(previews):
        row, column = divmod(index, columns)
        available_height = tile_height - 32
        scale = min(tile_width / image.shape[1], available_height / image.shape[0])
        resized = cv2.resize(
            image,
            (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        y0 = row * tile_height + 30 + (available_height - resized.shape[0]) // 2
        x0 = column * tile_width + (tile_width - resized.shape[1]) // 2
        sheet[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        cv2.putText(
            sheet,
            board_id,
            (column * tile_width + 8, row * tile_height + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return sheet


def build_bootstrap(
    config: BootstrapConfig, *, runtime: ProposalRuntime | None = None
) -> Path:
    """Build all per-board artifacts atomically and return the output path."""

    dataset_root = config.dataset_root.expanduser().resolve()
    output = config.output_dir.expanduser().resolve()
    try:
        output.relative_to(dataset_root)
    except ValueError as exc:
        raise BootstrapError("Output directory must stay inside the dataset root") from exc
    if output.exists():
        raise BootstrapError(f"Output already exists; refusing to overwrite: {output}")
    boards, source_manifest, source_manifest_sha = load_source_boards(
        dataset_root, expected_count=config.expected_count
    )
    detector_runtime = runtime or _DetectorRuntime(
        config.model_path, config.detector_confidence
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    previews: list[tuple[str, np.ndarray]] = []
    index_rows: list[dict[str, Any]] = []
    board_records: list[dict[str, Any]] = []
    try:
        references_dir = staging / "references"
        references_dir.mkdir()
        for board in boards:
            image = load_image(board.image_path.read_bytes())
            mask = cv2.imread(str(board.mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise BootstrapError(f"Could not read mask for {board.board_id}")
            board_bbox = _board_bbox(mask)
            upstream_ics = parse_upstream_ic_annotations(
                board.annotation_path.read_text(encoding="utf-8"),
                image_width=board.width,
                image_height=board.height,
            )
            detections = detector_runtime.detect(image, board_bbox)
            proposals = merge_proposals(detections, upstream_ics)
            rows = _proposal_rows(board.board_id, proposals)

            board_dir = references_dir / board.board_id
            board_dir.mkdir()
            pnp_path = board_dir / "pnp_pixels_NEEDS_REVIEW.csv"
            with pnp_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=_PNP_COLUMNS, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            pnp_json_path = board_dir / "pnp_pixels_NEEDS_REVIEW.json"
            _write_json(
                pnp_json_path,
                {
                    "schema_version": PNP_SCHEMA_VERSION,
                    "status": REVIEW_STATUS,
                    "board_id": board.board_id,
                    "coordinate_space": COORDINATE_SPACE,
                    "detector_identifier": detector_runtime.identifier,
                    "summary": {
                        "proposal_count": len(rows),
                        "upstream_ic_count": len(upstream_ics),
                        "detector_raw_count": len(detections),
                    },
                    "warnings": [
                        "Synthetic *_AUTO_* names are not OCR or manufacturing RefDes.",
                        "Observed upstream angle is an OpenCV rectangle angle, not a verified PnP rotation.",
                        "No millimetre coordinates are emitted without physical calibration.",
                    ],
                    "proposals": [dict(row) for row in rows],
                },
            )

            reference_path = board_dir / "golden_candidate.json"
            _write_json(
                reference_path,
                {
                    "schema_version": BOOTSTRAP_SCHEMA_VERSION,
                    "status": REFERENCE_STATUS,
                    "production_eligible": False,
                    "board_id": board.board_id,
                    "upstream_board_id": board.upstream_board_id,
                    "side": "unknown",
                    "coordinate_space": COORDINATE_SPACE,
                    "image": {
                        "path": _relative_posix(board.image_path, board_dir),
                        "sha256": board.image_sha256,
                        "width": board.width,
                        "height": board.height,
                        "source_kind": "dataset_preprocessed_source_as_received",
                        "format_warning": "Source is JPEG; approve then create a lossless PNG/TIFF production Golden.",
                    },
                    "board_mask": {
                        "path": _relative_posix(board.mask_path, board_dir),
                        "sha256": board.mask_sha256,
                        "bbox_xyxy": list(board_bbox.as_xyxy()),
                    },
                    "upstream_ic_annotation": {
                        "path": _relative_posix(board.annotation_path, board_dir),
                        "sha256": board.annotation_sha256,
                        "count": len(upstream_ics),
                    },
                    "metrology": {"verified": False, "pixel_to_mm": None},
                    "registration": {"verified": False, "transform": None},
                    "review_required": [
                        "confirm_board_id_revision_and_side",
                        "confirm_board_is_acceptable_reference",
                        "review_every_pnp_proposal",
                        "capture_repeated_ok_frames_for_this_sku",
                        "calibrate_physical_coordinates_before_mm_export",
                    ],
                },
            )
            overlay = _preview(
                image,
                board_bbox,
                rows,
                max_side=config.preview_max_side,
            )
            overlay_path = board_dir / "pnp_preview.jpg"
            if not cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise BootstrapError(f"Could not write preview for {board.board_id}")
            previews.append((board.board_id, overlay))
            focus = _focus_score(image, mask)
            index_rows.append(
                {
                    "board_id": board.board_id,
                    "upstream_board_id": board.upstream_board_id,
                    "image_path": _relative_posix(board.image_path, staging),
                    "image_sha256": board.image_sha256,
                    "width_px": board.width,
                    "height_px": board.height,
                    "focus_laplacian_var_at_max1600": f"{focus:.6f}",
                    "upstream_ic_count": len(upstream_ics),
                    "pnp_proposal_count": len(rows),
                    "reference_status": REFERENCE_STATUS,
                    "review_status": REVIEW_STATUS,
                }
            )
            board_records.append(
                {
                    "board_id": board.board_id,
                    "upstream_board_id": board.upstream_board_id,
                    "golden_candidate": f"references/{board.board_id}/golden_candidate.json",
                    "pnp_csv": f"references/{board.board_id}/pnp_pixels_NEEDS_REVIEW.csv",
                    "pnp_json": f"references/{board.board_id}/pnp_pixels_NEEDS_REVIEW.json",
                    "preview": f"references/{board.board_id}/pnp_preview.jpg",
                    "focus_laplacian_var_at_max1600": focus,
                    "proposal_count": len(rows),
                }
            )

        with (staging / "reference_index.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(index_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(index_rows)
        contact_sheet = _contact_sheet(previews)
        if not cv2.imwrite(
            str(staging / "contact_sheet.jpg"),
            contact_sheet,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        ):
            raise BootstrapError("Could not write contact sheet")
        (staging / "README.md").write_text(
            "# Per-board Golden/PnP bootstrap\n\n"
            "The 30 source images are different layouts. Each folder under "
            "`references/` is independent; do not build a cross-board consensus.\n\n"
            "- `golden_candidate.json`: source reference identity and review gates.\n"
            "- `pnp_pixels_NEEDS_REVIEW.csv/json`: pixel-native pseudo-labels only.\n"
            "- `pnp_preview.jpg`: green boxes are upstream IC annotations; orange "
            "boxes are detector proposals.\n"
            "- `reference_index.csv`: image identity, focus score and proposal counts.\n"
            "- `contact_sheet.jpg`: diversity/quality review for all 30 boards.\n\n"
            "No file here contains verified millimetres, manufacturing RefDes, "
            "footprints, board side or production acceptance. Follow "
            "`../NHUNG_VIEC_BAN_CAN_LAM.md` before promoting any artifact.\n",
            encoding="utf-8",
        )
        _write_json(
            staging / "bootstrap_manifest.json",
            {
                "schema_version": BOOTSTRAP_SCHEMA_VERSION,
                "status": REVIEW_STATUS,
                "production_eligible": False,
                "coordinate_space": COORDINATE_SPACE,
                "source_manifest_sha256": source_manifest_sha,
                "source_dataset": source_manifest.get("source_dataset", "TU Wien PCB DSLR"),
                "detector_identifier": detector_runtime.identifier,
                "board_count": len(board_records),
                "distinct_layout_policy": "one source board -> one reference and one PnP draft",
                "quality_summary": {
                    "unique_source_sha256_count": len(
                        {item["image_sha256"] for item in index_rows}
                    ),
                    "focus_metric": "Laplacian variance inside board mask at max side 1600",
                    "focus_min": min(
                        float(item["focus_laplacian_var_at_max1600"])
                        for item in index_rows
                    ),
                    "focus_median": float(
                        np.median(
                            [
                                float(item["focus_laplacian_var_at_max1600"])
                                for item in index_rows
                            ]
                        )
                    ),
                    "focus_max": max(
                        float(item["focus_laplacian_var_at_max1600"])
                        for item in index_rows
                    ),
                    "human_visual_review_required": True,
                },
                "boards": board_records,
            },
        )
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "active" / "detector" / "best.onnx",
    )
    parser.add_argument("--expected-count", type=int, default=30)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--preview-max-side", type=int, default=1600)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.dataset_root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "bootstrap"
    )
    try:
        result = build_bootstrap(
            BootstrapConfig(
                dataset_root=root,
                output_dir=output,
                model_path=args.model,
                expected_count=args.expected_count,
                detector_confidence=args.conf,
                preview_max_side=args.preview_max_side,
            )
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Built {REVIEW_STATUS} per-board references -> {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
