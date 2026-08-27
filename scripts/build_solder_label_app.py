"""Turn a step-5.5 ROI export into an offline labelling app for step 6.2.

``export_solder_dataset.py`` cuts one crop per derived joint and writes a CSV
whose ``defect_class`` column is left empty. This command wraps that output in a
browser app so filling the column is a keystroke per crop instead of a spreadsheet
row, and so the reviewer sees the pixels next to the decision.

The app writes back the exact schema ``pcb_solder_defect_v2_kaggle.py`` reads in
``camera_finetune`` mode -- ``crop_path, defect_class, board_id, capture_id,
dataset_source, roi_kind`` plus ``label_status``, ``reviewer_id``,
``label_scope``, ``split`` -- so the labelled folder zips straight into a Kaggle
input with nothing to reshape.

It offers only the six labels that notebook accepts at joint scope. ``bridge``
and the component-placement labels are deliberately absent: the notebook routes
them out with ``bridge_is_pair_rule_not_single_joint_classifier`` and
``component_label_routed_out_of_joint_scope``, so offering them would collect
work that is silently discarded at training time.

    python scripts/build_solder_label_app.py <export_dir> --board-id BOARD1

Everything runs from disk: the crops are referenced by relative path and the row
index is inlined, so the page needs no server. Progress autosaves to
``localStorage`` keyed by the dataset id.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

TEMPLATE_PATH = Path(__file__).with_name("_solder_label_app_template.html")

#: Carried through to the app so the reviewer can see what they are judging.
CONTEXT_FIELDS = (
    "component_label",
    "position",
    "roi_kind",
    "roi_width_px",
    "roi_height_px",
    "source_image",
    "terminal_geometry",
    "detector_confidence",
)


def build_rows(
    manifest: Path,
    board_id: str,
    capture_id: str,
    dataset_source: str,
    joints_only: bool,
) -> list[dict[str, object]]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle))
    if not raw:
        raise SystemExit(f"{manifest} has no rows")

    rows: list[dict[str, object]] = []
    for row in raw:
        # ``kind`` on a step-5.5 export, ``roi_kind`` once this tool has written
        # it back. Reading only the first would let body views through on a
        # second pass over an already-labelled folder.
        kind = (row.get("kind") or row.get("roi_kind") or "joint").strip().lower()
        if joints_only and kind != "joint":
            continue
        entry: dict[str, object] = {
            "crop_path": Path(row["crop_path"]).name,
            # Empty on purpose. The notebook ignores an unlabelled row; it does
            # not ignore a guessed one.
            "defect_class": (row.get("defect_class") or "").strip(),
            "board_id": board_id or (row.get("board_id") or "").strip(),
            "capture_id": capture_id or Path(row.get("source_image", "")).stem,
            "dataset_source": dataset_source,
            "roi_kind": kind,
            "label_status": "",
            "reviewer_id": "",
            "label_scope": "",
            "split": "",
            "notes": "",
        }
        for field in CONTEXT_FIELDS:
            if field in row:
                entry[field] = row[field]
        rows.append(entry)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "export_dir",
        type=Path,
        help="directory written by export_solder_dataset.py (holds crops/ and solder_dataset.csv)",
    )
    parser.add_argument(
        "--board-id",
        default="",
        help="board this export came from; the notebook splits train/val/test by it, "
        "so leaving every crop on one id makes an honest split impossible",
    )
    parser.add_argument("--capture-id", default="", help="defaults to the source image stem")
    parser.add_argument(
        "--dataset-source",
        default="local_camera",
        help="provenance tag carried into training (default: local_camera)",
    )
    parser.add_argument(
        "--include-body-views",
        action="store_true",
        help="also label the whole-component views; they are not joints and the "
        "notebook routes them out, so this is for inspection only",
    )
    args = parser.parse_args(argv)

    export_dir = args.export_dir.resolve()
    manifest = export_dir / "solder_dataset.csv"
    crops = export_dir / "crops"
    for path in (manifest, crops):
        if not path.exists():
            raise SystemExit(f"missing {path}; run export_solder_dataset.py first")

    rows = build_rows(
        manifest,
        args.board_id,
        args.capture_id,
        args.dataset_source,
        joints_only=not args.include_body_views,
    )
    missing = [r["crop_path"] for r in rows if not (crops / str(r["crop_path"])).exists()][:5]
    if missing:
        raise SystemExit(f"manifest references crops that are not on disk: {missing}")

    payload = {
        # Keyed on content so relabelling a different export cannot silently
        # inherit the previous one's saved progress.
        "dataset_id": hashlib.sha256(
            f"{export_dir.name}|{len(rows)}|{rows[0]['crop_path']}".encode()
        ).hexdigest()[:16],
        "crops_dir": "crops",
        "rows": rows,
    }
    html = (
        TEMPLATE_PATH.read_text(encoding="utf-8")
        .replace("__DATA__", json.dumps(payload, ensure_ascii=False))
        .replace("__DATASET__", export_dir.name)
    )
    out = export_dir / "label_app.html"
    out.write_text(html, encoding="utf-8")

    boards = {r["board_id"] for r in rows}
    print(f"wrote {out}")
    print(f"  {len(rows)} crops to label, board_id={sorted(boards)}")
    if boards == {""}:
        print("  WARNING: no board_id. Pass --board-id, or the notebook cannot split by board.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
