"""Verify a reference source set against its bootstrap bundle, then emit review scaffolds.

This command never decides whether a frame, a Golden or a component proposal is
good.  It only re-derives the facts a human reviewer needs before deciding:
file hashes, Golden traceability, alignment gate state and the consensus/PnP
snapshot.  Every quality verdict column it writes is left empty for the
reviewer, and no millimetre value is ever invented.

Checks performed by ``verify``:

* every ``manifest.json`` entry exists on disk and its SHA-256 and byte size match
* the bundle Golden decodes bit-exactly to one manifest source JPEG
* the alignment report keeps its declared mode, accepts every frame and never
  falls back to a resized canvas
* the consensus audit and the PnP queue agree with each other and with the
  manifest frame list

``scaffold`` writes ``frame_review.csv``, ``pnp_pixels_REVIEWED.csv``, the
``labels/`` contract skeleton and the empty metrology stubs.  Existing files are
never overwritten unless ``--force`` is given, so reviewer work cannot be lost.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

PNP_REVIEW_FIELDS = [
    "record_id",
    "source_auto_id",
    "action",
    "verified_refdes",
    "class",
    "footprint",
    "center_x_px",
    "center_y_px",
    "rotation_deg",
    "polarity",
    "review_status",
    "label_source",
    "notes",
]

FRAME_REVIEW_FIELDS = [
    "image_id",
    "sha256",
    "review_status",
    "layout_ok",
    "side_ok",
    "focus_ok",
    "board_complete",
    "visible_anomaly",
    "notes",
    "reviewer",
    "reviewed_at",
    # Measured evidence below; these are inputs to the human decision, not the
    # decision itself.  They come from manifest.json and are full-resolution.
    "measured_focus_laplacian_variance",
    "measured_mean_luminance",
    "measured_focus_rank",
    "is_selected_golden",
]

LABEL_MANIFEST_FIELDS = [
    "image_id",
    "sha256",
    "board_id",
    "revision",
    "side",
    "annotation_schema",
    "coordinate_space",
    "transform_id",
    "label_source",
    "annotator",
    "reviewer",
    "review_status",
    "notes",
]


class Result:
    """Collects pass/fail checks so one run reports every problem at once."""

    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append({"check": name, "passed": passed, "detail": detail})
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {name}: {detail}")

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [c for c in self.checks if not c["passed"]]


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(set_dir: Path, manifest: dict[str, Any], result: Result) -> dict[str, str]:
    """Re-hash every listed image and return image_id -> sha256 for good rows."""
    files = manifest["files"]
    hashes: dict[str, str] = {}
    bad: list[str] = []
    for entry in files:
        path = set_dir / entry["path"]
        if not path.exists():
            bad.append(f"missing:{entry['path']}")
            continue
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            bad.append(f"sha256:{entry['path']}")
            continue
        if path.stat().st_size != entry["byte_size"]:
            bad.append(f"bytes:{entry['path']}")
            continue
        hashes[path.stem] = digest

    result.add(
        "manifest_hashes",
        not bad,
        f"{len(hashes)}/{len(files)} images match SHA-256 and byte size"
        + (f"; problems: {bad}" if bad else ""),
    )

    on_disk = {p.name for p in (set_dir / "images").glob("*.jpg")}
    listed = {Path(e["path"]).name for e in files}
    extra = sorted(on_disk - listed)
    result.add(
        "no_unlisted_images",
        not extra,
        "images/ contains only manifest-listed files" if not extra else f"unlisted: {extra}",
    )
    return hashes


def verify_golden(
    bundle_dir: Path, set_dir: Path, manifest: dict[str, Any], result: Result
) -> dict[str, Any]:
    """Prove the bundle Golden is a lossless decode of exactly one source JPEG."""
    golden_png = bundle_dir / "golden.png"
    selection = json.loads((bundle_dir / "reference_selection.json").read_text(encoding="utf-8"))
    chosen = selection["selection"]
    source_jpg = set_dir / "images" / chosen["basename"]

    info: dict[str, Any] = {
        "golden_png_sha256": sha256_file(golden_png),
        "source_basename": chosen["basename"],
        "source_sha256": chosen["sha256"],
        "reference_policy": selection.get("reference_policy"),
    }

    result.add(
        "golden_source_is_manifest_frame",
        any(e["sha256"] == chosen["sha256"] for e in manifest["files"]),
        f"selected Golden {chosen['basename']} is one of the {len(manifest['files'])} manifest frames",
    )

    try:
        import numpy as np
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        golden = np.array(Image.open(golden_png).convert("RGB"))
        source = np.array(Image.open(source_jpg).convert("RGB"))
        identical = golden.shape == source.shape and bool((golden == source).all())
        info["pixel_identical"] = identical
        info["pixel_array_sha256"] = sha256(golden.tobytes()).hexdigest()
        info["canvas"] = {"width": int(golden.shape[1]), "height": int(golden.shape[0])}
        result.add(
            "golden_is_lossless_decode",
            identical,
            f"golden.png is bit-exact with {chosen['basename']} at {golden.shape[1]}x{golden.shape[0]}"
            if identical
            else "golden.png differs from its declared source JPEG",
        )
    except ImportError:
        info["pixel_identical"] = None
        result.add(
            "golden_is_lossless_decode", False, "Pillow/numpy unavailable; could not compare pixels"
        )

    return info


def verify_alignment(bundle_dir: Path, expected_frames: int, result: Result) -> dict[str, Any]:
    report = json.loads((bundle_dir / "alignment_report.json").read_text(encoding="utf-8"))
    frames = report["frames"]
    summary = report["summary"]
    mode = report["alignment_mode"]

    result.add(
        "alignment_mode",
        mode == "upstream_dataset_identity_enrollment_only",
        f"alignment_mode={mode}",
    )
    result.add(
        "alignment_frame_count",
        summary["accepted_frame_count"] == expected_frames == summary["source_frame_count"],
        f"{summary['accepted_frame_count']}/{summary['source_frame_count']} frames accepted "
        f"(manifest lists {expected_frames})",
    )

    methods = sorted({f["method"] for f in frames})
    resize_fallback = [f["frame_id"] for f in frames if "resize" in str(f["method"]).lower()]
    result.add(
        "no_resize_fallback",
        not resize_fallback,
        f"methods used: {methods}" if not resize_fallback else f"resize fallback on: {resize_fallback}",
    )

    shapes = {(f["output_shape"]["width"], f["output_shape"]["height"]) for f in frames}
    result.add(
        "native_canvas_preserved",
        len(shapes) == 1,
        f"all frames output at {sorted(shapes)}",
    )

    # The plan forbids reading identity artefacts as measured registration
    # accuracy, so record that the fit metrics are explicitly unmeasured.
    unmeasured = sum(1 for f in frames if f["fit_metrics_status"] == "not_measured_upstream_identity")
    result.add(
        "fit_metrics_declared_unmeasured",
        unmeasured == len(frames),
        f"{unmeasured}/{len(frames)} frames report fit_metrics_status=not_measured_upstream_identity "
        "(overlap 1.0 is an identity artefact, not measured registration accuracy)",
    )

    return {
        "mode": mode,
        "accepted_frame_count": summary["accepted_frame_count"],
        "methods": methods,
        "production_registration_eligible": report["alignment_provenance"][
            "production_registration_eligible"
        ],
        "fit_metrics_status": "not_measured_upstream_identity",
        "per_frame_alignment_overlay_present": False,
    }


def verify_consensus(bundle_dir: Path, expected_frames: int, result: Result) -> dict[str, Any]:
    consensus = json.loads((bundle_dir / "consensus_components.json").read_text(encoding="utf-8"))
    components = consensus["components"]
    summary = consensus["summary"]
    config = consensus["config"]

    gate_pass = [c for c in components if c["consensus_status"] == "CONSENSUS"]
    result.add(
        "consensus_gate_counts_agree",
        summary["eligible_component_count"] == len(gate_pass)
        and summary["component_count"] == len(components),
        f"{len(components)} audit clusters, {len(gate_pass)} pass "
        f"support>={config['min_support_ratio']} and purity>={config['min_class_purity']}",
    )
    result.add(
        "consensus_frame_count",
        summary["frame_count"] == expected_frames,
        f"consensus built from {summary['frame_count']} frames",
    )

    with (bundle_dir / "pnp_pixels_NEEDS_REVIEW.csv").open(encoding="utf-8", newline="") as handle:
        queue = list(csv.DictReader(handle))

    audit_ids = {c["designator"] for c in components}
    orphans = [r["designator"] for r in queue if r["designator"] not in audit_ids]
    result.add(
        "pnp_queue_traceable_to_audit",
        not orphans,
        f"all {len(queue)} PnP proposals map back to an audit cluster"
        if not orphans
        else f"untraceable proposals: {orphans}",
    )

    not_on_golden = [r["designator"] for r in queue if r["selected_golden_observation_present"] != "true"]
    result.add(
        "pnp_queue_anchored_on_golden",
        not not_on_golden,
        "every proposal has an observation in the selected Golden"
        if not not_on_golden
        else f"proposals without a Golden observation: {not_on_golden}",
    )

    audit_classes: dict[str, int] = {}
    for comp in components:
        audit_classes[comp["label"]] = audit_classes.get(comp["label"], 0) + 1
    queue_classes: dict[str, int] = {}
    for row in queue:
        queue_classes[row["class_label"]] = queue_classes.get(row["class_label"], 0) + 1

    missing_by_class = {
        label: audit_classes.get(label, 0) - queue_classes.get(label, 0)
        for label in sorted(audit_classes)
    }

    return {
        "audit_cluster_count": len(components),
        "gate_pass_count": len(gate_pass),
        "gate_config": config,
        "audit_class_counts": dict(sorted(audit_classes.items(), key=lambda kv: -kv[1])),
        "pnp_proposal_count": len(queue),
        "pnp_class_counts": dict(sorted(queue_classes.items(), key=lambda kv: -kv[1])),
        "pnp_low_support_rows": sum(1 for r in queue if "low_support" in r["review_reasons"]),
        "pnp_rotation_populated": sum(1 for r in queue if r["rotation_deg"].strip()),
        "pnp_footprint_populated": sum(1 for r in queue if r["footprint"].strip()),
        "audit_minus_queue_by_class": missing_by_class,
    }


def cmd_verify(args: argparse.Namespace) -> int:
    set_dir = args.set_dir.resolve()
    bundle_dir = args.bundle_dir.resolve()
    manifest = json.loads((set_dir / "manifest.json").read_text(encoding="utf-8"))
    result = Result()

    verify_manifest(set_dir, manifest, result)
    golden = verify_golden(bundle_dir, set_dir, manifest, result)
    alignment = verify_alignment(bundle_dir, len(manifest["files"]), result)
    consensus = verify_consensus(bundle_dir, len(manifest["files"]), result)

    report = {
        "schema_version": "aoi-reference-set-verification/1.0",
        "set_dir": str(set_dir),
        "bundle_dir": str(bundle_dir),
        "frame_count": len(manifest["files"]),
        "golden": golden,
        "alignment": alignment,
        "consensus": consensus,
        "checks": result.checks,
        "all_checks_passed": not result.failed,
        "production_eligible": False,
        "production_eligible_reason": (
            "no physical board measurement, fiducial registration or real camera/fixture "
            "validation exists for this set"
        ),
    }
    out = set_dir / "verification_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")
    print(f"{len(result.checks) - len(result.failed)}/{len(result.checks)} checks passed")
    return 1 if result.failed else 0


def cmd_scaffold(args: argparse.Namespace) -> int:
    set_dir = args.set_dir.resolve()
    bundle_dir = args.bundle_dir.resolve()
    manifest = json.loads((set_dir / "manifest.json").read_text(encoding="utf-8"))
    selection = json.loads((bundle_dir / "reference_selection.json").read_text(encoding="utf-8"))
    golden_sha = selection["selection"]["sha256"]
    board = manifest["board"]

    written: list[Path] = []
    skipped: list[Path] = []

    def guard(path: Path) -> bool:
        if path.exists() and not args.force:
            skipped.append(path)
            return False
        return True

    # --- frame_review.csv -------------------------------------------------
    entries = manifest["files"]
    focus_sorted = sorted(entries, key=lambda e: -e["focus_laplacian_variance"])
    focus_rank = {e["path"]: i + 1 for i, e in enumerate(focus_sorted)}

    frame_path = set_dir / "frame_review.csv"
    if guard(frame_path):
        with frame_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FRAME_REVIEW_FIELDS)
            writer.writeheader()
            for entry in entries:
                is_golden = entry["sha256"] == golden_sha
                writer.writerow(
                    {
                        "image_id": Path(entry["path"]).stem,
                        "sha256": entry["sha256"],
                        # Human-owned columns stay empty; "unknown" is the plan's
                        # explicit state for "not enough evidence yet".
                        "review_status": "unknown",
                        "layout_ok": "",
                        "side_ok": "",
                        "focus_ok": "",
                        "board_complete": "",
                        "visible_anomaly": "",
                        "notes": "",
                        "reviewer": "",
                        "reviewed_at": "",
                        "measured_focus_laplacian_variance": entry["focus_laplacian_variance"],
                        "measured_mean_luminance": entry["mean_luminance"],
                        "measured_focus_rank": focus_rank[entry["path"]],
                        "is_selected_golden": "true" if is_golden else "false",
                    }
                )
        written.append(frame_path)

    # --- pnp_pixels_REVIEWED.csv -----------------------------------------
    with (bundle_dir / "pnp_pixels_NEEDS_REVIEW.csv").open(encoding="utf-8", newline="") as handle:
        queue = list(csv.DictReader(handle))

    pnp_path = set_dir / "pnp_pixels_REVIEWED.csv"
    if guard(pnp_path):
        with pnp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PNP_REVIEW_FIELDS)
            writer.writeheader()
            for index, row in enumerate(queue, start=1):
                support = float(row["support_ratio"])
                observations = int(row["observation_count"])
                note = (
                    f"support_ratio={support:.3f}; observations={observations}/{row['frame_count']}; "
                    f"class_purity={float(row['class_purity']):.3f}; "
                    f"center_mad_px={float(row['center_mad_px']):.2f}; "
                    f"detector_gate={row['consensus_status']}"
                )
                if observations == 1:
                    note += "; single-frame cluster, verify directly on Golden"
                writer.writerow(
                    {
                        "record_id": f"PNP_{index:04d}",
                        "source_auto_id": row["designator"],
                        # action is intentionally empty: the plan allows only
                        # accept/correct/add/reject and each is a human verdict.
                        "action": "",
                        "verified_refdes": "",
                        "class": row["class_label"],
                        "footprint": "",
                        "center_x_px": row["center_x_px"],
                        "center_y_px": row["center_y_px"],
                        "rotation_deg": "",
                        "polarity": "",
                        "review_status": "draft",
                        "label_source": "pseudo_label",
                        "notes": note,
                    }
                )
        written.append(pnp_path)

    # --- labels/ contract skeleton ---------------------------------------
    labels_dir = set_dir / "labels"
    for sub in ("board_geometry", "semantic", "components", "pads_and_joints", "traces"):
        (labels_dir / sub).mkdir(parents=True, exist_ok=True)
        keep = labels_dir / sub / ".gitkeep"
        if not keep.exists():
            keep.write_text("", encoding="utf-8")

    label_manifest = labels_dir / "label_manifest.csv"
    if guard(label_manifest):
        with label_manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LABEL_MANIFEST_FIELDS)
            writer.writeheader()
            for entry in entries:
                writer.writerow(
                    {
                        "image_id": Path(entry["path"]).stem,
                        "sha256": entry["sha256"],
                        "board_id": "mpi_pcb_gas_pump",
                        "revision": "unknown",
                        "side": board["side"],
                        "annotation_schema": "",
                        "coordinate_space": "",
                        "transform_id": "",
                        "label_source": "unknown",
                        "annotator": "",
                        "reviewer": "",
                        "review_status": "draft",
                        "notes": "",
                    }
                )
        written.append(label_manifest)

    # --- metrology stubs, deliberately empty of numbers -------------------
    measurements = set_dir / "board_measurements.json"
    if guard(measurements):
        measurements.write_text(
            json.dumps(
                {
                    "schema_version": "aoi-board-measurements/1.0",
                    "status": "NOT_MEASURED",
                    "status_note": (
                        "Every numeric field is null on purpose. Fill them only from a "
                        "measured physical board or official CAD. Guessed dimensions or "
                        "scale taken from an image are forbidden by the review plan."
                    ),
                    "board_id": "mpi_pcb_gas_pump",
                    "revision": None,
                    "side": board["side"],
                    "units": None,
                    "board_width": None,
                    "board_height": None,
                    "cross_check_distance": None,
                    "origin": {"x": None, "y": None, "definition": None},
                    "axis_convention": {"x_direction": None, "y_direction": None},
                    "rotation_convention": {
                        "zero_angle_definition": None,
                        "positive_direction": None,
                    },
                    "mirror_definition_for_bottom_side": None,
                    "calibration": {
                        "camera_id": None,
                        "lens_id": None,
                        "working_distance": None,
                        "intrinsics_file": None,
                        "distortion_model": None,
                        "measured_scale": None,
                        "residual": None,
                        "calibration_version": None,
                        "measured_at": None,
                    },
                    "production_eligible": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        written.append(measurements)

    fiducials = set_dir / "fiducials.csv"
    if guard(fiducials):
        fiducials.write_text(
            "fiducial_id,type,x_mm,y_mm,diameter_mm,center_x_px,center_y_px,"
            "coordinate_space,measurement_method,measured_by,measured_at,notes\n",
            encoding="utf-8",
        )
        written.append(fiducials)

    for path in written:
        print(f"wrote    {path.relative_to(set_dir)}")
    for path in skipped:
        print(f"skipped  {path.relative_to(set_dir)} (exists; use --force to overwrite)")
    if skipped:
        print("\nNothing was overwritten. Reviewer edits are safe.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--set-dir",
        type=Path,
        required=True,
        help="reference source set directory containing manifest.json and images/",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        required=True,
        help="bootstrap bundle directory containing golden.png and the consensus artifacts",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="re-derive and check the bundle facts")
    scaffold = sub.add_parser("scaffold", help="write the empty reviewer deliverables")
    scaffold.add_argument("--force", action="store_true", help="overwrite existing scaffolds")

    args = parser.parse_args(argv)
    if args.command == "verify":
        return cmd_verify(args)
    return cmd_scaffold(args)


if __name__ == "__main__":
    sys.exit(main())
