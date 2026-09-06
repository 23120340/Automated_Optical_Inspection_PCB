"""Fetch the public solder datasets that are actually usable, and say why the rest are not.

The survey behind this lives in ``datasets/public/README.md`` and
``docs/surveys/dataset_lead_detection.md``. The short version was once "no public source
gives both the right pixel scale and box labels"; that held for *solder-joint*
labels and still does, but it stopped being the whole story once whole-board
sets with component boxes were measured -- their crops carry joints that a
person can label. ``crop_components_for_labelling.py`` is that path, and
``datasets/train/solder_joint_v1`` is what came out of it.

So this command still downloads little: it exists to make the usable sources
reproducible and to stop the unusable ones being re-downloaded every few weeks
by someone who has forgotten why they were rejected.

    python scripts/fetch_public_solder_datasets.py --list
    python scripts/fetch_public_solder_datasets.py ulger
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEST = PROJECT_ROOT / "datasets" / "public"

USABLE = {
    "ulger": {
        "url": "https://github.com/furkanulger/solder-joint-dataset/archive/refs/heads/main.zip",
        "folder": "ulger_solder_joints",
        "inner": "solder-joint-dataset-main",
        "what": "3,389 joint crops, 5 classes, ~20-25 um/px -- closest public source to this project's 46 um/px",
        "use": "6.2 classifier and backbone pretraining. NOT detection: crops only, no boxes, no board images",
        "licence": "NONE. GitHub API returns license=null (checked 2026-08-25). Author retains all rights: "
                   "internal research is normal practice, redistribution and commercial use are not.",
    },
}

REJECTED = {
    "phme2022": (
        "CSV only -- no images at all. data/ holds SPI_training_*.csv.zip with columns "
        "PosX(mm), Volume(%), Height(um), Area(%). Also SPI is solder paste BEFORE "
        "placement, and the experiments are password-gated behind registration (CC BY-NC-SA)."
    ),
    "soldef_ai": (
        "1-3 um/px against this project's 46 um/px, a 20x mismatch: measured 0 boxes on the "
        "project board at every 1x-12x zoom. Already fetched automatically by "
        "training/kaggle/pcb_solder_defect_v2_kaggle.py in public_bootstrap mode, so there is "
        "nothing to label by hand."
    ),
    "consolidated": (
        "Component detection, not solder joints. Its pads/pins classes hold 186/261 instances "
        "across ~30 of 670 images; measured recall 0.072."
    ),
}

CREDENTIALED = {
    "roboflow": (
        "Has box-labelled Dry_joint/Cold Solder data -- the one gap worth filling -- but "
        "universe.roboflow.com returns 403 and api.roboflow.com needs a key. Get a free key "
        "from your Roboflow workspace settings, then pass --api-key."
    ),
    "kaggle": (
        "SolDef_AI and Consolidated live on Kaggle and need ~/.kaggle/kaggle.json "
        "(Kaggle > Account > Create New API Token). Both are rejected above, so this is only "
        "worth doing to inspect them yourself."
    ),
}


def fetch(name: str, force: bool) -> int:
    spec = USABLE[name]
    target = DEST / spec["folder"]
    if target.exists() and not force:
        print(f"{target} already exists; pass --force to re-download")
        return 0
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"downloading {spec['url']}")
    with urllib.request.urlopen(spec["url"], timeout=300) as response:
        payload = response.read()
    print(f"  {len(payload) / 1024 / 1024:.1f} MB")
    if target.exists():
        shutil.rmtree(target)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(DEST)
    extracted = DEST / spec["inner"]
    if not extracted.exists():
        raise SystemExit(f"archive did not contain {spec['inner']}")
    extracted.rename(target)

    counts = {
        child.name: sum(1 for _ in child.iterdir())
        for child in sorted(target.iterdir())
        if child.is_dir()
    }
    print(f"\n{target}")
    for label, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"   {label:24s} {count:6d}")
    print(f"\n  use : {spec['use']}")
    print(f"  LICENCE: {spec['licence']}")
    return 0


def show_list() -> int:
    print("USABLE — downloadable now\n")
    for name, spec in USABLE.items():
        print(f"  {name}\n      {spec['what']}\n      {spec['use']}\n")
    print("REJECTED — measured, do not re-download\n")
    for name, why in REJECTED.items():
        print(f"  {name}\n      {why}\n")
    print("NEEDS CREDENTIALS\n")
    for name, why in CREDENTIALED.items():
        print(f"  {name}\n      {why}\n")
    print("Full reasoning: datasets/public/README.md")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "dataset", nargs="?", choices=sorted(USABLE) + sorted(CREDENTIALED),
        help="which source to fetch; omit with --list to see every verdict",
    )
    parser.add_argument("--list", action="store_true", help="print every source and its verdict")
    parser.add_argument("--force", action="store_true", help="re-download over an existing copy")
    parser.add_argument("--api-key", default="", help="Roboflow API key")
    args = parser.parse_args(argv)

    if args.list or not args.dataset:
        return show_list()
    if args.dataset in CREDENTIALED:
        print(f"{args.dataset}: {CREDENTIALED[args.dataset]}")
        if args.dataset == "roboflow" and not args.api_key:
            print("\nNo --api-key given, so nothing was fetched.")
            return 1
        print("\nNot automated yet: pick the specific dataset and version on Universe first, "
              "then add it here so the choice is recorded rather than ad hoc.")
        return 1
    return fetch(args.dataset, args.force)


if __name__ == "__main__":
    sys.exit(main())
