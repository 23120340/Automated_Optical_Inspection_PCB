"""Create an unreviewed joint-box draft with the project's solder detector.

The labelling queue contains thousands of component crops cut from only a few
hundred source photographs.  Running the fixed-shape 1280 px ONNX graph on
every crop is both slow and a scale mismatch.  This command therefore runs the
detector once per source photograph and maps each detection back into the best
matching component crop.

The output deliberately keeps ``status`` empty for *every* crop.  A model box
is a proposal, and a missing model box is not evidence that a joint is good.
After loading the JSON in ``label_boxes.html``, a person must press Enter to
accept proposed boxes or C to explicitly approve a clean crop.  The labelling
page's final export then contains only reviewed records.

Example::

    python scripts/prelabel_joint_boxes.py \
        datasets/labelling/winnies_components \
        --output datasets/labelling/winnies_components/joint_boxes.ai_draft.json

The command checkpoints periodically.  Re-run it with ``--resume`` after an
interruption to skip source photographs already processed by the same draft.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

# Direct execution sets ``sys.path[0]`` to ``scripts/`` rather than the
# repository root.  Add the known parent so the same documented command works
# without requiring callers to set PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aoi_pipeline.models import BoundingBox, Detection
from aoi_pipeline.solder.defect_detection import create_solder_defect_detector

try:  # Works both as ``python scripts/...`` and as an imported test module.
    from scripts.crop_components_for_labelling import Source, parse_labels
except ModuleNotFoundError:  # pragma: no cover - direct-script import path
    from crop_components_for_labelling import Source, parse_labels


SCHEMA = "aoi-joint-boxes/1.0"
COORDINATE_SPACE = "crop_pixels_top_left_origin"
DEFAULT_MODEL = Path("models/active/solder/segmenter/best.onnx")
DEFAULT_MODEL_MANIFEST = Path("models/active/solder/segmenter/model_manifest.json")
ORDER_RE = re.compile(r"__(\d+)__[^/\\]+\.[^.]+$")


@dataclass(frozen=True, slots=True)
class CropGeometry:
    """One crop expressed in the source photograph's pixel coordinates."""

    crop_path: str
    crop: BoundingBox
    body: BoundingBox
    component_class: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dataset_id(dataset_name: str, rows: Sequence[Mapping[str, str]], classes: Sequence[str]) -> str:
    """Match the content key embedded by :mod:`build_joint_box_app`."""

    if not rows:
        raise ValueError("cannot identify an empty crop manifest")
    material = (
        f"{dataset_name}|{len(rows)}|{rows[0]['crop_path']}|{','.join(classes)}"
    ).encode()
    return hashlib.sha256(material).hexdigest()[:16]


def crop_order(crop_path: str) -> int:
    match = ORDER_RE.search(crop_path)
    if match is None:
        raise ValueError(f"crop filename has no source annotation order: {crop_path}")
    return int(match.group(1))


def geometry_for_row(
    row: Mapping[str, str], source_box: Any, width: int, height: int
) -> CropGeometry:
    """Reconstruct the exact crop rectangle without guessing its margin.

    ``manifest.csv`` stores the body offset inside the crop.  Combining that
    offset with the ordered upstream component annotation recovers the clamped
    source-pixel crop origin exactly, including edge crops where the nominal
    30 percent margin was truncated.
    """

    body_left = int(round((float(source_box.cx) - float(source_box.w) / 2.0) * width))
    body_top = int(round((float(source_box.cy) - float(source_box.h) / 2.0) * height))
    body_w = int(round(float(source_box.w) * width))
    body_h = int(round(float(source_box.h) * height))
    left = body_left - int(row["body_x"])
    top = body_top - int(row["body_y"])
    crop_w = int(row["crop_w"])
    crop_h = int(row["crop_h"])
    crop = BoundingBox(float(left), float(top), float(left + crop_w), float(top + crop_h))
    body = BoundingBox(
        float(body_left),
        float(body_top),
        float(body_left + body_w),
        float(body_top + body_h),
    )
    if crop.x1 < 0 or crop.y1 < 0 or crop.x2 > width or crop.y2 > height:
        raise ValueError(
            f"reconstructed crop is outside {width}x{height}: {row['crop_path']} {crop.as_xyxy()}"
        )
    return CropGeometry(
        crop_path=str(row["crop_path"]),
        crop=crop,
        body=body,
        component_class=str(row["component_class"]),
    )


def _intersection(first: BoundingBox, second: BoundingBox) -> BoundingBox | None:
    x1, y1 = max(first.x1, second.x1), max(first.y1, second.y1)
    x2, y2 = min(first.x2, second.x2), min(first.y2, second.y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return BoundingBox(x1, y1, x2, y2)


def _body_match_score(point: tuple[float, float], geometry: CropGeometry) -> float:
    """Prefer the component whose body boundary is nearest the defect centre."""

    x, y = point
    body = geometry.body
    if body.x1 <= x <= body.x2:
        dx_edge = min(x - body.x1, body.x2 - x)
    else:
        dx_edge = min(abs(x - body.x1), abs(x - body.x2))
    if body.y1 <= y <= body.y2:
        dy_edge = min(y - body.y1, body.y2 - y)
    else:
        dy_edge = min(abs(y - body.y1), abs(y - body.y2))
    edge_distance = min(dx_edge, dy_edge)
    cx, cy = body.center
    centre_distance = float(np.hypot(x - cx, y - cy))
    scale = max(1.0, float(np.hypot(body.width, body.height)))
    return edge_distance / scale + 0.05 * centre_distance / scale


def map_detections_to_crops(
    detections: Iterable[Detection],
    geometries: Sequence[CropGeometry],
    class_names: Sequence[str],
    *,
    min_containment: float = 0.50,
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """Assign each full-scene detection to one crop and convert to crop xywh.

    A proposal must have its centre inside both the padded crop and its target
    component annotation, with at least ``min_containment`` of its area visible
    in the crop.  This rejects responses on neighbouring parts that happen to
    sit in the 30 percent context margin.  Overlapping crops are common, so one
    source detection is assigned only to the component whose annotated body is
    the best geometric match.
    """

    allowed = set(class_names)
    mapped: dict[str, list[dict[str, Any]]] = {g.crop_path: [] for g in geometries}
    unmapped = 0
    for detection in detections:
        if detection.label not in allowed:
            raise ValueError(f"model returned class outside contract: {detection.label!r}")
        if detection.bbox.area <= 0:
            unmapped += 1
            continue
        centre = detection.bbox.center
        candidates: list[tuple[float, CropGeometry, BoundingBox]] = []
        for geometry in geometries:
            if not (
                geometry.crop.x1 <= centre[0] <= geometry.crop.x2
                and geometry.crop.y1 <= centre[1] <= geometry.crop.y2
            ):
                continue
            # The padded crop deliberately includes context.  Do not attach a
            # response on a neighbouring part to the crop's target component.
            # The accepted upstream sources include leads/pads in their
            # component annotations, so a real target joint centre remains in
            # this box even when its fillet reaches into the margin.
            if not (
                geometry.body.x1 <= centre[0] <= geometry.body.x2
                and geometry.body.y1 <= centre[1] <= geometry.body.y2
            ):
                continue
            overlap = _intersection(detection.bbox, geometry.crop)
            if overlap is None or overlap.area / detection.bbox.area < min_containment:
                continue
            candidates.append((_body_match_score(centre, geometry), geometry, overlap))
        if not candidates:
            unmapped += 1
            continue
        _, chosen, overlap = min(candidates, key=lambda item: (item[0], item[1].crop.area))
        x0 = int(np.floor(overlap.x1 - chosen.crop.x1))
        y0 = int(np.floor(overlap.y1 - chosen.crop.y1))
        x1 = int(np.ceil(overlap.x2 - chosen.crop.x1))
        y1 = int(np.ceil(overlap.y2 - chosen.crop.y1))
        if x1 - x0 < 2 or y1 - y0 < 2:
            unmapped += 1
            continue
        mapped[chosen.crop_path].append(
            {
                "cls": detection.label,
                "x": x0,
                "y": y0,
                "w": x1 - x0,
                "h": y1 - y0,
                # The current page ignores unknown box fields, while retaining
                # this value in the draft makes audits and future tooling easier.
                "proposal_confidence": round(float(detection.confidence), 6),
            }
        )
    return mapped, unmapped


def _resolve_repo_path(raw: str | Path, repo_root: Path) -> Path:
    value = Path(raw).expanduser()
    if value.is_absolute():
        return value.resolve()
    candidates = [(Path.cwd() / value).resolve(), (repo_root / value).resolve()]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} has no rows")
    return rows


def _decode_image(blob: bytes, name: str) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"could not decode source image {name}")
    return image


def _draft_record(boxes: Sequence[Mapping[str, Any]], confidence: float) -> dict[str, Any]:
    if boxes:
        summary = ", ".join(
            f"{box['cls']}={float(box['proposal_confidence']):.3f}" for box in boxes
        )
        notes = f"AI đề xuất chưa duyệt (conf>={confidence:.3f}): {summary}."
    else:
        notes = (
            f"AI không đề xuất box ở conf>={confidence:.3f}; vẫn CHƯA được coi là sạch."
        )
    return {"status": "", "notes": notes, "boxes": [dict(box) for box in boxes]}


def _write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")


def _initial_payload(
    crop_dir: Path,
    rows: Sequence[Mapping[str, str]],
    class_names: Sequence[str],
    model: Path,
    model_manifest: Path,
    confidence: float,
    iou: float,
    source_images: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "dataset_id": dataset_id(crop_dir.name, rows, class_names),
        "dataset": crop_dir.name,
        # reviewer_id is intentionally human-owned.  Do not put the model name
        # here, otherwise a final export falsely claims human review provenance.
        "reviewer_id": "",
        "exported_at": utc_now(),
        "coordinate_space": COORDINATE_SPACE,
        "classes": list(class_names),
        "proposal": {
            "kind": "model_first_pass_unreviewed",
            "created_by": "Codex/model-assisted-first-pass",
            "approval_required": True,
            "empty_status_means_unreviewed": True,
            "no_box_is_not_clean": True,
            "model_path": str(model),
            "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "model_manifest": str(model_manifest),
            "confidence": confidence,
            "iou": iou,
            "total_source_images": len(source_images),
            "processed_source_images": [],
            "inference_errors": {},
            "mapped_detections": 0,
            "unmapped_detections": 0,
            "complete": False,
        },
        "crops": {
            str(row["crop_path"]): {
                "status": "",
                "notes": "AI draft chưa chạy tới ảnh nguồn này; chưa được coi là sạch.",
                "boxes": [],
            }
            for row in rows
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("crop_dir", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-manifest", type=Path, default=DEFAULT_MODEL_MANIFEST)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--min-containment", type=float, default=0.50)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--limit-source-images", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be in [0, 1]")
    if not 0.0 <= args.iou <= 1.0:
        parser.error("--iou must be in [0, 1]")
    if not 0.0 <= args.min_containment <= 1.0:
        parser.error("--min-containment must be in [0, 1]")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")

    repo_root = REPO_ROOT
    crop_dir = args.crop_dir.expanduser().resolve()
    manifest_path = crop_dir / "manifest.csv"
    provenance_path = crop_dir / "provenance.json"
    for required in (manifest_path, provenance_path, crop_dir / "crops"):
        if not required.exists():
            raise SystemExit(f"missing {required}")
    rows = _read_rows(manifest_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    source_path = _resolve_repo_path(provenance["source"], repo_root)
    model_path = _resolve_repo_path(args.model, repo_root)
    model_manifest_path = _resolve_repo_path(args.model_manifest, repo_root)
    for required in (source_path, model_path, model_manifest_path):
        if not required.exists():
            raise SystemExit(f"missing {required}")

    detector = create_solder_defect_detector(model_path, model_manifest_path)
    if detector is None:  # pragma: no cover - complete pair above makes this impossible
        raise SystemExit("could not create solder detector")
    class_names = list(detector.class_names)
    source = Source(source_path)
    upstream_classes = source.classes()
    if not upstream_classes:
        raise SystemExit(f"{source_path} has no class list")

    by_source: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_source.setdefault(row["source_image"], []).append(row)
    source_images = sorted(by_source)
    if args.limit_source_images:
        source_images = source_images[: args.limit_source_images]
    output = (
        args.output.expanduser().resolve()
        if args.output
        else crop_dir / "joint_boxes.ai_draft.json"
    )

    expected_id = dataset_id(crop_dir.name, rows, class_names)
    if args.resume and output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("dataset_id") != expected_id:
            raise SystemExit("resume draft belongs to a different crop set")
        if payload.get("classes") != class_names:
            raise SystemExit("resume draft class list differs from the model contract")
    else:
        payload = _initial_payload(
            crop_dir,
            rows,
            class_names,
            model_path,
            model_manifest_path,
            args.confidence,
            args.iou,
            source_images,
        )

    proposal = payload["proposal"]
    processed = set(proposal.get("processed_source_images", []))
    pending = [name for name in source_images if name not in processed]
    print(
        f"{crop_dir.name}: {len(rows)} crop, {len(source_images)} source image, "
        f"{len(pending)} pending; model classes={class_names}",
        flush=True,
    )

    for position, image_name in enumerate(pending, start=1):
        scene_rows = by_source[image_name]
        try:
            label_name = source.label_for(image_name)
            if label_name is None:
                raise ValueError(f"source label is missing for {image_name}")
            source_boxes = parse_labels(
                source.read(label_name).decode("utf-8", errors="replace"), upstream_classes
            )
            image = _decode_image(source.read(image_name), image_name)
            height, width = image.shape[:2]
            geometries: list[CropGeometry] = []
            for row in scene_rows:
                order = crop_order(row["crop_path"])
                if order >= len(source_boxes):
                    raise ValueError(
                        f"annotation order {order} is outside {len(source_boxes)} boxes for {image_name}"
                    )
                source_box = source_boxes[order]
                if source_box.cls != row["component_class"]:
                    raise ValueError(
                        f"class mismatch for {row['crop_path']}: {source_box.cls!r} != "
                        f"{row['component_class']!r}"
                    )
                geometries.append(geometry_for_row(row, source_box, width, height))

            detections = detector.detect(image, confidence=args.confidence)
            mapped, unmapped = map_detections_to_crops(
                detections,
                geometries,
                class_names,
                min_containment=args.min_containment,
            )
            mapped_count = sum(len(value) for value in mapped.values())
            for row in scene_rows:
                boxes = mapped.get(row["crop_path"], [])
                payload["crops"][row["crop_path"]] = _draft_record(boxes, args.confidence)
            proposal["mapped_detections"] = int(proposal.get("mapped_detections", 0)) + mapped_count
            proposal["unmapped_detections"] = int(proposal.get("unmapped_detections", 0)) + unmapped
            detail = f"det={len(detections)}, mapped={mapped_count}, unmapped={unmapped}"
        except Exception as exc:  # Continue safely: records remain explicitly unreviewed.
            proposal.setdefault("inference_errors", {})[image_name] = str(exc)
            for row in scene_rows:
                payload["crops"][row["crop_path"]] = {
                    "status": "",
                    "notes": f"AI inference lỗi: {exc}; chưa được coi là sạch.",
                    "boxes": [],
                }
            detail = f"ERROR: {exc}"

        processed.add(image_name)
        proposal["processed_source_images"] = sorted(processed)
        proposal["complete"] = len(processed) == len(source_images)
        payload["exported_at"] = utc_now()
        print(f"[{position}/{len(pending)}] {image_name}: {detail}", flush=True)
        if position % args.checkpoint_every == 0:
            _write_payload(output, payload)
            print(f"  checkpoint -> {output}", flush=True)

    proposal["complete"] = len(processed) == len(source_images)
    _write_payload(output, payload)
    proposed_crops = sum(1 for record in payload["crops"].values() if record["boxes"])
    proposed_boxes = sum(len(record["boxes"]) for record in payload["crops"].values())
    print(
        f"wrote {output}\n"
        f"  {len(payload['crops'])} unreviewed crop records; "
        f"{proposed_crops} crop / {proposed_boxes} proposed box\n"
        f"  errors={len(proposal.get('inference_errors', {}))}; "
        "no-box records are NOT marked clean",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
