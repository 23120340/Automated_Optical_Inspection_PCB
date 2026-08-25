"""Generate a self-contained offline review app for the PnP proposal queue.

The app has two modes over one shared board image:

* **Overview** draws every proposal on the whole Golden at once, coloured by its
  review state, so coverage is visible: what is done, what is untouched, and
  which parts of the board carry no box at all.  The 221 consensus clusters the
  detector found but never promoted into the PnP queue are drawn as faint ghost
  boxes, so adding a missed component is one click on an already-localised site
  rather than a hunt across 4096x2816 pixels.
* **Detail** windows the same image around one proposal for reading silkscreen
  and dragging the box onto the real component.

Everything runs from disk, so nothing may be fetched at runtime: the geometry is
inlined into the HTML and only ``board.jpg`` is loaded by relative path.  Work
autosaves to ``localStorage`` keyed by the Golden SHA-256 and can be re-imported
from a previously exported CSV.

Nothing here decides identity.  Rows start with an empty ``action`` and stay
``draft`` / ``pseudo_label`` until the reviewer changes them.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

BOARD_IMAGE = "board.jpg"

TEMPLATE_PATH = Path(__file__).with_name("_pnp_review_app_template.html")


def build_payload(bundle_dir: Path) -> dict:
    selection = json.loads((bundle_dir / "reference_selection.json").read_text(encoding="utf-8"))
    consensus = json.loads((bundle_dir / "consensus_components.json").read_text(encoding="utf-8"))

    with (bundle_dir / "pnp_pixels_NEEDS_REVIEW.csv").open(encoding="utf-8", newline="") as handle:
        queue = list(csv.DictReader(handle))

    records = []
    for index, row in enumerate(queue, start=1):
        x1, y1 = float(row["x1_px"]), float(row["y1_px"])
        x2, y2 = float(row["x2_px"]), float(row["y2_px"])
        records.append(
            {
                "record_id": f"PNP_{index:04d}",
                "source_auto_id": row["designator"],
                "action": "",
                "verified_refdes": "",
                "class": row["class_label"],
                "det_class": row["class_label"],
                "footprint": "",
                # Seeded from the detector box so a row exported without ever
                # being opened still carries its centre instead of a blank.
                "center_x_px": f"{(x1 + x2) / 2:.3f}",
                "center_y_px": f"{(y1 + y2) / 2:.3f}",
                "rotation_deg": "",
                "polarity": "",
                "review_status": "draft",
                "label_source": "pseudo_label",
                "notes": "",
                "x1_px": round(x1, 2),
                "y1_px": round(y1, 2),
                "x2_px": round(x2, 2),
                "y2_px": round(y2, 2),
                "support_ratio": round(float(row["support_ratio"]), 3),
                "observation_count": int(row["observation_count"]),
                "class_purity": round(float(row["class_purity"]), 3),
                "center_mad_px": round(float(row["center_mad_px"]), 2),
                "detector_gate": row["consensus_status"],
            }
        )

    # Ghosts: audit clusters the detector localised but did not promote into the
    # PnP queue.  These are the "add" candidates the plan asks for.
    #
    # They are noisy: most appear in only one or two frames, many sit on top of
    # each other, and many repeat a site the queue already covers.  Each ghost
    # therefore carries the flag needed to filter it, and the app hides the
    # queue-duplicates outright rather than inviting a double entry.
    in_queue = {row["designator"] for row in queue}
    queue_boxes = [(r["x1_px"], r["y1_px"], r["x2_px"], r["y2_px"]) for r in records]

    def iou(a: tuple, b: tuple) -> float:
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        if inter <= 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter)

    ghosts = []
    for comp in consensus["components"]:
        if comp["designator"] in in_queue:
            continue
        x1, y1, x2, y2 = comp["bbox_xyxy"]
        box = (x1, y1, x2, y2)
        ghosts.append(
            {
                "id": comp["designator"],
                "cls": comp["label"],
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
                "obs": comp["observation_count"],
                "sup": round(comp["support_ratio"], 3),
                "dupq": 1 if any(iou(box, q) > 0.5 for q in queue_boxes) else 0,
            }
        )

    return {
        "golden_sha256": selection["selection"]["sha256"],
        "golden_basename": selection["selection"]["basename"],
        "coordinate_space": "golden_board_pixels",
        "board_image": BOARD_IMAGE,
        "board_w": consensus["canvas_size"]["width"],
        "board_h": consensus["canvas_size"]["height"],
        "records": records,
        "ghosts": ghosts,
    }


def write_board_image(bundle_dir: Path, out_dir: Path, quality: int) -> int:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    golden = Image.open(bundle_dir / "golden.png").convert("RGB")
    target = out_dir / BOARD_IMAGE
    golden.save(target, "JPEG", quality=quality, optimize=True, progressive=True)
    return target.stat().st_size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--board", default="mpi_pcb_gas_pump / top")
    parser.add_argument(
        "--quality",
        type=int,
        default=92,
        help="JPEG quality for the embedded board image (default 92)",
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(args.bundle_dir)
    size = write_board_image(args.bundle_dir, args.out_dir, args.quality)

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = template.replace("__DATA__", json.dumps(payload)).replace("__BOARD__", args.board)
    out = args.out_dir / "review_app.html"
    out.write_text(html, encoding="utf-8")

    print(f"wrote {out}")
    print(f"  {len(payload['records'])} PnP proposals, {len(payload['ghosts'])} audit ghosts")
    print(f"  {BOARD_IMAGE}: {size / 1024 / 1024:.1f} MB at {payload['board_w']}x{payload['board_h']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
