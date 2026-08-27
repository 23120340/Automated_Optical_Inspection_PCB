"""Search Roboflow Universe for solder-joint data and measure it before trusting it.

Universe search returns names and class lists, and neither says the thing that
decides whether a dataset is usable here: how big a solder joint is inside the
frame. This project shoots whole boards at 46 um/px, where a pad is about 23 px
in a 1024 px tile -- roughly 2% of the frame width. Two public sets have already
been rejected for getting that wrong, so the probe measures it rather than
reading the description.

For each candidate it reports:

* ``box_frac`` -- median annotation width as a share of image width. Near 0.02
  matches this project; an order of magnitude above means the camera was looking
  at one joint, not a board, and a detector trained there will find nothing on a
  full-board frame.
* ``colour`` -- whether the images carry chroma at all. ``segment_solder`` and
  the axis probe both read saturation, so a monochrome source is a second
  domain gap on top of any scale gap.
* ``scenes`` -- distinct source images behind the augmented count, so a set of
  3,908 images that is really six photographs cannot present itself as large.

    python scripts/probe_roboflow_solder.py --api-key KEY --queries "solder joint"
    python scripts/probe_roboflow_solder.py --api-key KEY --probe work-6qkmv/pcb-solder-joint

The key is never written to disk; pass it per invocation or via ROBOFLOW_API_KEY.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import os
import re
import statistics
import sys
import urllib.parse
import urllib.request
import zipfile

API = "https://api.roboflow.com"
#: Universe matches "solder" against "soldier" and "smd" against unrelated sets,
#: so anything whose classes look like these is dropped before download.
NOISE = re.compile(
    r"soldier|army|water|sea_|7up|bottle|can\b|truck|civil|screw|thread", re.I
)
SOLDER_HINT = re.compile(
    r"solder|joint|cold|bridge|excess|insufficient|spike|pad|pin|tomb|short", re.I
)


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.loads(response.read())


def search(key: str, queries: list[str], page_size: int) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for query in queries:
        url = f"{API}/universe/search?api_key={key}&q={urllib.parse.quote_plus(query)}&page_size={page_size}"
        try:
            payload = get(url)
        except Exception as exc:  # pragma: no cover - network shape varies
            print(f"  search {query!r} failed: {exc}")
            continue
        for row in payload.get("results", []):
            found[row["url"]] = row
    return found


def is_candidate(row: dict) -> bool:
    if row.get("type") not in {"object-detection", "instance-segmentation"}:
        return False
    blob = " ".join([row.get("name", ""), *(row.get("classes") or [])])
    return bool(SOLDER_HINT.search(blob)) and not NOISE.search(blob)


def probe(key: str, slug: str, sample: int) -> dict:
    """Download one version and measure what the listing cannot tell you."""
    import numpy as np
    from PIL import Image

    meta = get(f"{API}/{slug}?api_key={key}")
    versions = meta.get("versions") or []
    if not versions:
        return {"slug": slug, "error": "no versions"}
    version = versions[0]["id"].rsplit("/", 1)[-1]
    export = get(f"{API}/{slug}/{version}/yolov8?api_key={key}").get("export", {})
    link = export.get("link")
    if not link:
        return {"slug": slug, "error": "no export link"}

    with urllib.request.urlopen(link, timeout=900) as response:
        payload = response.read()
    archive = zipfile.ZipFile(io.BytesIO(payload))
    images = [n for n in archive.namelist() if n.lower().endswith((".jpg", ".png"))]
    labels = [n for n in archive.namelist() if n.endswith(".txt") and "/labels/" in n]
    if not images:
        return {"slug": slug, "error": "no images in export"}

    widths, chroma = [], []
    for name in images[: min(sample, len(images))]:
        with Image.open(io.BytesIO(archive.read(name))) as handle:
            rgb = np.asarray(handle.convert("RGB")).astype(int)
            widths.append(handle.width)
            chroma.append(float(np.abs(rgb.max(axis=2) - rgb.min(axis=2)).mean()))
    width = statistics.median(widths)

    fracs: list[float] = []
    for name in labels[: min(sample * 4, len(labels))]:
        for line in archive.read(name).decode(errors="ignore").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                fracs.append(float(parts[3]))

    # Roboflow renames every augmented copy but keeps the original stem in front
    # of the ``.rf.`` marker, so the real scene count survives the augmentation.
    scenes = {n.rsplit("/", 1)[-1].split(".rf.")[0] for n in images}
    return {
        "slug": slug,
        "version": version,
        "images": len(images),
        "scenes": len(scenes),
        "size_mb": round(len(payload) / 1024 / 1024, 1),
        "image_width": int(width),
        "box_frac": round(statistics.median(fracs), 4) if fracs else None,
        "box_px": round(statistics.median(fracs) * width) if fracs else None,
        "chroma": round(statistics.mean(chroma), 1),
    }


def verdict(row: dict) -> str:
    if row.get("error"):
        return row["error"]
    notes = []
    frac = row.get("box_frac") or 0
    if frac > 0.12:
        notes.append(f"macro: a joint fills {frac:.0%} of the frame against ~2% here")
    elif frac > 0.06:
        notes.append(f"closer than this project ({frac:.0%} vs ~2%)")
    if (row.get("chroma") or 0) < 12:
        notes.append("monochrome; saturation-based segmentation cannot transfer")
    if row.get("scenes", 0) < 40:
        notes.append(f"only {row['scenes']} distinct scenes behind {row['images']} images")
    return " · ".join(notes) if notes else "worth a closer look"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY", ""))
    parser.add_argument("--queries", nargs="*", default=[
        "solder joint", "solder defect", "pcb solder", "cold solder joint",
        "smd solder defect", "solder joint inspection",
    ])
    parser.add_argument("--page-size", type=int, default=30)
    parser.add_argument("--probe", nargs="*", default=None,
                        help="workspace/project slugs to download and measure")
    parser.add_argument("--top", type=int, default=6, help="how many candidates to probe")
    parser.add_argument("--sample", type=int, default=40)
    args = parser.parse_args(argv)

    if not args.api_key:
        raise SystemExit("need --api-key or ROBOFLOW_API_KEY")

    if args.probe:
        slugs = args.probe
    else:
        found = search(args.api_key, args.queries, args.page_size)
        candidates = [r for r in found.values() if is_candidate(r)]
        candidates.sort(key=lambda r: -(r.get("images") or 0))
        print(f"{len(found)} projects seen, {len(candidates)} look like solder data\n")
        for row in candidates[: args.top * 2]:
            classes = ", ".join((row.get("classes") or [])[:5])
            print(f"{row.get('images') or 0:6d}  {str(row.get('license'))[:12]:12s} {classes[:56]}")
            print(f"        {row['url']}")
        slugs = ["/".join(r["url"].split("/")[-2:]) for r in candidates[: args.top]]
        print()

    print(f"{'images':>7} {'scenes':>7} {'box_px':>7} {'box%':>6} {'chroma':>7}  slug")
    for slug in slugs:
        try:
            row = probe(args.api_key, slug, args.sample)
        except Exception as exc:  # pragma: no cover
            row = {"slug": slug, "error": str(exc)[:80]}
        if row.get("error"):
            print(f"{'-':>7} {'-':>7} {'-':>7} {'-':>6} {'-':>7}  {slug}  ({row['error']})")
            continue
        print(f"{row['images']:7d} {row['scenes']:7d} {row['box_px']:7d} "
              f"{row['box_frac']:6.1%} {row['chroma']:7.1f}  {slug}")
        print(f"        -> {verdict(row)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
