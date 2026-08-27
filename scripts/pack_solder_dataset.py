"""Check a labelled step-6.2 dataset against the training contract, then zip it.

The notebook drops rows it cannot use and says why in its own log, which is the
worst place to find out: by then a Kaggle run has already been spent. This
command applies the same rules locally and refuses to package a dataset that
would arrive mostly empty.

Rules mirrored from ``pcb_solder_defect_v2_kaggle.py`` at joint scope:

* only ``label_status`` in verified/approved/adjudicated is used;
* ``bridge`` is rejected -- a solder bridge spans two joints and one crop cannot
  express it (``bridge_is_pair_rule_not_single_joint_classifier``);
* component-placement labels are rejected -- they belong to a different profile;
* a crop labelled ``unknown`` must also carry a non-joint ``label_scope``;
* an unlabelled row is fine and simply unused.

    python scripts/pack_solder_dataset.py <export_dir> --output solder_v1.zip
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path
import sys
import zipfile

GOOD_ALIASES = {"good", "ok", "normal", "pass", "no_defect", "defect_free"}
UNKNOWN_ALIASES = {
    "unknown", "background", "not_a_joint", "invalid_roi", "wrong_crop",
    "false_crop", "out_of_distribution", "ood",
}
JOINT_LABELS = {"good", "insufficient", "excess", "cold", "missing_solder"}
COMPONENT_LABELS = {
    "shift_component", "shifted", "missing_component", "missing",
    "tombstone", "wrong_polarity", "misalignment",
}
VERIFIED = {"verified", "approved", "adjudicated", "verified_legacy"}
REQUIRED = ("crop_path", "defect_class", "board_id", "capture_id", "dataset_source", "roi_kind")


def normalize(value: object) -> str:
    text = str(value or "").strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def audit(rows: list[dict[str, str]], crops: Path) -> tuple[list[dict], list[tuple[str, str]]]:
    usable: list[dict] = []
    rejected: list[tuple[str, str]] = []
    for row in rows:
        name = row.get("crop_path", "")
        label = normalize(row.get("defect_class"))
        status = normalize(row.get("label_status"))
        scope = normalize(row.get("label_scope") or row.get("roi_kind") or "joint")

        if not (crops / name).exists():
            rejected.append((name, "crop file missing on disk"))
            continue
        if not label:
            rejected.append((name, "unlabelled (ignored by the notebook, not an error)"))
            continue
        if status not in VERIFIED:
            rejected.append((name, f"label_status={status or 'empty'} is not verified"))
            continue
        if label == "bridge":
            rejected.append((name, "bridge is a pair rule, not a single-joint class"))
            continue
        if label in COMPONENT_LABELS:
            rejected.append((name, "component-placement label outside joint scope"))
            continue
        if label in UNKNOWN_ALIASES and scope == "joint":
            rejected.append((name, "unknown must carry a non-joint label_scope"))
            continue
        if label not in JOINT_LABELS and label not in UNKNOWN_ALIASES and label not in GOOD_ALIASES:
            rejected.append((name, f"unrecognised label {label!r}"))
            continue
        usable.append(row)
    return usable, rejected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None, help="zip to write")
    parser.add_argument(
        "--min-usable", type=int, default=50,
        help="refuse to package below this many usable rows (default 50)",
    )
    parser.add_argument("--force", action="store_true", help="package anyway")
    args = parser.parse_args(argv)

    export_dir = args.export_dir.resolve()
    manifest = export_dir / "solder_dataset.csv"
    crops = export_dir / "crops"
    if not manifest.exists():
        raise SystemExit(f"missing {manifest}")

    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        missing_cols = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
    if missing_cols:
        raise SystemExit(
            f"{manifest} is missing required columns {missing_cols}. Export from "
            "label_app.html rather than editing the step-5.5 manifest by hand."
        )

    usable, rejected = audit(rows, crops)
    by_class = collections.Counter(normalize(r["defect_class"]) for r in usable)
    by_board = collections.Counter(r.get("board_id", "") for r in usable)
    reasons = collections.Counter(reason for _, reason in rejected)

    print(f"{len(rows)} rows · {len(usable)} usable · {len(rejected)} not used")
    print("\nby class:")
    for name, count in by_class.most_common():
        print(f"   {name:16s} {count:5d}")
    print("\nby board:")
    for name, count in by_board.most_common():
        print(f"   {name or '(empty)':16s} {count:5d}")
    if reasons:
        print("\nnot used, by reason:")
        for reason, count in reasons.most_common():
            print(f"   {count:5d}  {reason}")

    problems: list[str] = []
    if len(by_board) < 2:
        problems.append(
            "only one board_id: the notebook splits train/val/test by board, so a "
            "single board cannot produce an honest split"
        )
    if by_class.get("good", 0) == 0:
        problems.append("no 'good' rows: a classifier with no negative class cannot be trained")
    if len(by_class) < 2:
        problems.append("fewer than two classes present")
    if len(usable) < args.min_usable:
        problems.append(f"only {len(usable)} usable rows, below --min-usable={args.min_usable}")

    for problem in problems:
        print(f"\nBLOCKED: {problem}")
    if problems and not args.force:
        print("\nNothing was packaged. Fix the above, or pass --force to package anyway.")
        return 1

    output = args.output or export_dir.with_suffix(".zip")
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(manifest, "solder_dataset.csv")
        for row in usable:
            name = row["crop_path"]
            archive.write(crops / name, f"crops/{name}")
    size = output.stat().st_size / 1024 / 1024
    print(f"\nwrote {output} ({size:.1f} MB, {len(usable)} crops)")
    print("Add Input on Kaggle, then set run_mode='camera_finetune' in the notebook.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
