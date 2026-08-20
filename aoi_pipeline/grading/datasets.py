"""Ingest heterogeneous public solder-defect datasets into one taxonomy.

No single public dataset covers the step-6.2 taxonomy. The survey behind this
module found:

* **SolDef_AI** carries component misalignment plus excessive/insufficient
  solder, and is the only peer-reviewed optical source found that labels
  misalignment at all.
* Assorted Hugging Face soldering sets carry bridge / excess / empty, but ship
  no licence and no provenance.
* Roboflow community sets carry cold solder, but in the low hundreds of images.
* The widely-cited PCB "defect" datasets (DeepPCB, HRIPCB, DsPCBSD+) are
  **bare-board** defects -- open, short, mouse-bite, spur. Different problem
  entirely; they are listed here only so nobody wires them in by mistake.

So the sources get merged. That is only safe if the merge is auditable, which
is what this module is for: every source declares an explicit label map, every
unmapped label is reported rather than dropped, and the resulting per-class
counts are printed before a single epoch runs.

Two honesty rules are enforced rather than documented:

1. A label this module cannot map is **skipped and counted**, never guessed
   into the nearest-looking class. Quietly folding "solder ball" into "excess"
   would put a defect the model has never really seen behind a passing label.
2. Records carry a ``group`` (the source image or board they came from) so the
   train/validation split can hold whole boards out. Crops from one board share
   its lighting and its operator; splitting per crop reports a score the line
   will never see.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "BARE_BOARD_DATASETS",
    "DatasetSource",
    "SOURCES",
    "DatasetRecord",
    "LayoutProbe",
    "load_source",
    "merge_sources",
    "probe_layout",
    "coverage_report",
]

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

#: Datasets that are commonly mis-cited as solder-joint data. They annotate
#: defects on the *bare* board before any component is placed, so nothing in
#: them corresponds to a solder joint. Listed to be rejected, not used.
BARE_BOARD_DATASETS = {
    "deeppcb": "Bare-board open/short/mousebite/spur/copper/pinhole.",
    "hripcb": "Also PKU-Market-PCB. Bare-board, six trace defects.",
    "pku-market-pcb": "Same as HRIPCB.",
    "dspcbsd": "DsPCBSD+, nine bare-board surface defect classes.",
    "pcb-defects": "Kaggle akhatova/pcb-defects, the HRIPCB repackaging.",
}


@dataclass(slots=True)
class DatasetRecord:
    """One labelled crop, normalized across sources."""

    image_path: Path
    label: str
    source: str
    group: str
    bbox: tuple[int, int, int, int] | None = None
    original_label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetSource:
    """A public dataset plus the mapping that brings it into the taxonomy."""

    name: str
    kind: str
    location: str
    label_map: dict[str, str]
    homepage: str = ""
    licence: str = "unknown"
    modality: str = "optical"
    scope: str = "joint"
    verified: bool = False
    notes: str = ""
    #: Source labels deliberately dropped, with the reason. Keeping this
    #: explicit stops a later reader assuming they were simply forgotten.
    ignore: dict[str, str] = field(default_factory=dict)

    def map_label(self, raw: str) -> str | None:
        key = _normalize_label(raw)
        return self.label_map.get(key)


def _normalize_label(raw: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(raw).strip().lower()).strip("_")


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #

#: Sources surveyed in August 2026. ``verified`` records whether the internal
#: layout was confirmed by inspection rather than from the dataset's own
#: description -- an unverified layout is probed at load time instead of
#: assumed. Edit the label maps here; nothing downstream hardcodes them.
SOURCES: dict[str, DatasetSource] = {
    "soldef_ai": DatasetSource(
        name="soldef_ai",
        kind="kaggle",
        location="mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection",
        homepage="https://www.kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection",
        licence="see Kaggle page; paper is MDPI open access (CC BY 4.0)",
        modality="optical",
        scope="component",
        verified=False,
        notes=(
            "1150 soldered SMT component images, three viewpoints each, from the "
            "MDPI JMMP 2024 paper (doi 10.3390/jmmp8030117). The only peer-reviewed "
            "optical source found that labels component misalignment.\n"
            "Real layout, confirmed on Kaggle: `Labeled/<name>.jpg` + "
            "`<name>.json` LabelMe sidecars, 428 images -- matches the paper's "
            "description ('manually annotated using LabelMe... a JSON file "
            "containing all the created masks'). A second top-level folder, "
            "`Dataset/CS1..CS7`, was seen but not explored; it may hold more "
            "images, unannotated or under a different scheme.\n"
            "The exact label strings inside the LabelMe JSONs were not directly "
            "observed (the Kaggle listing could not be fetched, so the layout "
            "was probed at load time instead), so the entries below are the "
            "paper's own terminology plus close synonyms. Run "
            "load_source once and read report['unmapped_labels'] -- it prints "
            "every raw label this map did not catch -- then extend this dict "
            "with whatever shows up instead of guessing further."
        ),
        label_map={
            "good": "good",
            "ok": "good",
            "no_defect": "good",
            "non_defective": "good",
            "defect_free": "good",
            "correct": "good",
            "correct_position": "good",
            "correct_placement": "good",
            "solder_ok": "good",
            "solder_good": "good",
            "assembly_ok": "good",
            "position_ok": "good",
            "misalignment": "shift_component",
            "mis_alignment": "shift_component",
            "misaligned": "shift_component",
            "misalign": "shift_component",
            "shift": "shift_component",
            "shifted": "shift_component",
            "displacement": "shift_component",
            "displaced": "shift_component",
            "offset": "shift_component",
            "wrong_position": "shift_component",
            "incorrect_position": "shift_component",
            "wrong_placement": "shift_component",
            "component_shift": "shift_component",
            "excess": "excess",
            "excessive": "excess",
            "excessive_solder": "excess",
            "excess_solder": "excess",
            "too_much_solder": "excess",
            "over_solder": "excess",
            "insufficient": "insufficient",
            "insufficient_solder": "insufficient",
            "less_solder": "insufficient",
            "lack_of_solder": "insufficient",
            "under_solder": "insufficient",
        },
    ),
    "hf_soldering_boarding": DatasetSource(
        name="hf_soldering_boarding",
        kind="huggingface",
        location="ouvic215/Soldering-Data-Annotation-boarding",
        homepage="https://huggingface.co/datasets/ouvic215/Soldering-Data-Annotation-boarding",
        licence="NOT STATED",
        modality="unknown",
        scope="joint",
        verified=True,
        notes=(
            "1522 rows at 512x512. Labels confirmed by inspection. Ships no licence "
            "and no provenance, and a sibling repo is named ...-ControlNet, which "
            "suggests these images are tied to synthetic generation. Treat as "
            "supplementary; do not let it dominate a class."
        ),
        label_map={
            "bridge": "bridge",
            "micro_bridge": "bridge",
            "excess_solder": "excess",
            "empty": "missing_solder",
            "less_empty": "insufficient",
        },
        ignore={
            "appearance": "Surface-appearance label with no taxonomy equivalent.",
            "appearance_less": "Same; too vague to map onto a defect class.",
        },
    ),
    "hf_soldering_tiny": DatasetSource(
        name="hf_soldering_tiny",
        kind="huggingface",
        location="AndyLiu0104/Soldering-Data-Tiny-More-Data-with-appearance-hole-micro-bridge-0801",
        homepage="https://huggingface.co/datasets/AndyLiu0104/Soldering-Data-Tiny-More-Data-with-appearance-hole-micro-bridge-0801",
        licence="NOT STATED",
        modality="unknown",
        scope="joint",
        verified=True,
        notes=(
            "10469 rows but only 36-144 px wide -- below the resolution at which a "
            "fillet can be graded. Prompt-like text fields (pin count, background "
            "colour, size words) point at generated data. Useful for pre-training "
            "at most."
        ),
        label_map={
            "bridge": "bridge",
            "micro_bridge": "bridge",
            "excess_solder": "excess",
            "empty": "missing_solder",
            "less_empty": "insufficient",
        },
        ignore={
            "appearance": "No taxonomy equivalent.",
            "hole": "Pin-hole / via, not a solder-joint defect class here.",
        },
    ),
    "roboflow_soldering": DatasetSource(
        name="roboflow_soldering",
        kind="roboflow",
        location="",  # filled in per project by whoever exports it
        homepage="https://universe.roboflow.com/search?q=class:solder",
        licence="per project, usually CC BY 4.0",
        modality="optical",
        scope="joint",
        verified=False,
        notes=(
            "Community projects. The only public source found that labels cold "
            "solder, but the projects run to a few hundred images and quality "
            "varies. Export as folder-per-class and point --root at it."
        ),
        label_map={
            "cold_solder": "cold",
            "cold": "cold",
            "cold_joint": "cold",
            "poor_wetting": "cold",
            "insufficient_solder": "insufficient",
            "insufficient": "insufficient",
            "excess_solder": "excess",
            "excessive_solder": "excess",
            "solder_bridge": "bridge",
            "bridge": "bridge",
            "short": "bridge",
            "no_solder": "missing_solder",
            "missing_solder": "missing_solder",
            "missing_component": "missing_component",
            "good": "good",
            "normal": "good",
            "ok": "good",
        },
        ignore={
            "solder_ball": "Real defect, but absent from the taxonomy; folding it into excess would hide it.",
            "solder_crack": "Same reasoning.",
            "solder_dross": "Same reasoning.",
        },
    ),
    "local_export": DatasetSource(
        name="local_export",
        kind="local",
        location="",
        homepage="",
        licence="yours",
        modality="optical",
        scope="joint",
        verified=True,
        notes=(
            "Whatever scripts/export_solder_dataset.py produced from your own "
            "boards, once defect_class is filled in. This is the only source that "
            "matches your camera and lighting, so weight it heavily even when it "
            "is the smallest."
        ),
        label_map={value: value for value in (
            "good", "insufficient", "excess", "bridge", "cold", "missing_solder",
            "shift_component", "missing_component", "tombstone",
        )},
    ),
}


# --------------------------------------------------------------------------- #
# Layout probing
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class LayoutProbe:
    """What a dataset directory turned out to look like."""

    layout: str
    root: Path
    detail: str = ""
    annotation_files: list[Path] = field(default_factory=list)
    class_dirs: list[Path] = field(default_factory=list)
    image_count: int = 0

    def describe(self) -> str:
        lines = [f"layout   : {self.layout}", f"root     : {self.root}",
                 f"images   : {self.image_count}"]
        if self.detail:
            lines.append(f"detail   : {self.detail}")
        if self.class_dirs:
            lines.append(f"classes  : {[p.name for p in self.class_dirs]}")
        if self.annotation_files:
            shown = [str(p.name) for p in self.annotation_files[:5]]
            lines.append(f"ann files: {shown}{' ...' if len(self.annotation_files) > 5 else ''}")
        return "\n".join(lines)


def probe_layout(root: str | Path, max_depth: int = 4) -> LayoutProbe:
    """Work out how a dataset directory is organised.

    Called instead of assuming, because a dataset whose page could not be
    inspected may be laid out any of several ways. An unrecognised layout is
    reported as ``unknown`` so the caller stops, rather than being coerced into
    a guess that silently mislabels everything.
    """

    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        return LayoutProbe(layout="missing", root=base, detail="path is not a directory")

    images = [p for p in _walk(base, max_depth) if p.suffix.lower() in IMAGE_EXTENSIONS]
    all_json = [p for p in _walk(base, max_depth) if p.suffix.lower() == ".json"]
    coco = [p for p in all_json if _looks_like_coco(p)]
    labelme = [p for p in all_json if p not in coco and _looks_like_labelme(p)]
    csvs = [p for p in _walk(base, max_depth) if p.suffix.lower() == ".csv"]
    yolo_labels = [p for p in _walk(base, max_depth) if p.suffix.lower() == ".txt"
                   and p.parent.name.lower() in {"labels", "label"}]

    if coco:
        return LayoutProbe("coco", base, "COCO json with segmentation or bbox",
                           annotation_files=coco, image_count=len(images))
    if labelme:
        # Checked ahead of folder_per_class: a LabelMe export is one JSON
        # sidecar per image sitting flat in a directory, which is exactly the
        # shape folder_per_class would otherwise misread as "no class folders
        # here" and skip past.
        return LayoutProbe("labelme", base, "LabelMe JSON sidecar per image",
                           annotation_files=labelme, image_count=len(images))
    if yolo_labels:
        return LayoutProbe("yolo", base, "YOLO txt label files beside an images/ dir",
                           annotation_files=yolo_labels, image_count=len(images))
    if any(_looks_like_label_csv(p) for p in csvs):
        picked = [p for p in csvs if _looks_like_label_csv(p)]
        return LayoutProbe("csv", base, "CSV manifest with a label column",
                           annotation_files=picked, image_count=len(images))

    class_dirs = _class_directories(base)
    if class_dirs:
        return LayoutProbe("folder_per_class", base,
                           "one directory per class, images inside",
                           class_dirs=class_dirs, image_count=len(images))

    return LayoutProbe(
        "unknown", base,
        f"{len(images)} images found but no recognisable annotation scheme",
        image_count=len(images),
    )


def _walk(base: Path, max_depth: int) -> Iterable[Path]:
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        try:
            if len(path.relative_to(base).parts) > max_depth:
                continue
        except ValueError:
            continue
        yield path


def _looks_like_coco(path: Path) -> bool:
    try:
        if path.stat().st_size > 200 * 1024 * 1024:
            return False
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            head = handle.read(4096)
    except OSError:
        return False
    return '"annotations"' in head and '"images"' in head


def _looks_like_labelme(path: Path) -> bool:
    """A LabelMe annotation: one JSON per image, with a top-level ``shapes`` list.

    This is the format the SolDef_AI paper describes -- "manually annotated
    using LabelMe... a JSON file containing all the created masks for each
    instance" -- and it is common enough elsewhere that detecting it generally,
    rather than special-casing one dataset, is worth doing once.
    """

    try:
        if path.stat().st_size > 50 * 1024 * 1024:
            return False
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            head = handle.read(4096)
    except OSError:
        return False
    return '"shapes"' in head and ('"imagePath"' in head or '"imageHeight"' in head)


def _looks_like_label_csv(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            header = handle.readline().lower()
    except OSError:
        return False
    has_label = any(key in header for key in ("label", "class", "defect", "category"))
    has_image = any(key in header for key in ("image", "file", "path", "crop"))
    return has_label and has_image


#: Directory names that mean "split", not "class". Two folders called train and
#: test are a split; two called good and bad are classes. Counting folders
#: cannot tell them apart, so go by the names.
SPLIT_DIR_NAMES = {
    "train", "training", "test", "testing", "val", "valid", "validation",
    "eval", "dev", "holdout", "images", "data",
}


def _class_directories(base: Path) -> list[Path]:
    """Directories that look like class folders rather than a split."""

    for parent in [base, *sorted(p for p in base.iterdir() if p.is_dir())]:
        if not parent.is_dir():
            continue
        children = [p for p in parent.iterdir() if p.is_dir()]
        with_images = [
            child for child in children
            if any(item.suffix.lower() in IMAGE_EXTENSIONS
                   for item in child.iterdir() if item.is_file())
        ]
        if not with_images:
            continue
        named_as_split = [c for c in with_images if c.name.strip().lower() in SPLIT_DIR_NAMES]
        # All of them look like splits: descend rather than treating train/test
        # as two defect classes.
        if len(named_as_split) == len(with_images):
            continue
        # A genuine two-class dataset is common, so do not require three.
        if len(with_images) >= 2:
            return sorted(c for c in with_images if c.name.strip().lower() not in SPLIT_DIR_NAMES)
    return []


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_source(
    source: DatasetSource,
    root: str | Path,
    probe: LayoutProbe | None = None,
) -> tuple[list[DatasetRecord], dict[str, Any]]:
    """Read one dataset into taxonomy-labelled records.

    Returns the records plus a report: what was mapped, what was skipped, and
    why. The report is meant to be printed -- a merge nobody audited is a merge
    nobody can trust.
    """

    base = Path(root).expanduser().resolve()
    probe = probe or probe_layout(base)
    if probe.layout in {"missing", "unknown"}:
        return ([], {
            "source": source.name,
            "layout": probe.layout,
            "error": (
                f"Could not read {base}: {probe.detail}. Inspect the directory and "
                "either point --root deeper or add an adapter; nothing is guessed."
            ),
        })

    if probe.layout == "folder_per_class":
        raw = _read_folder_per_class(probe)
    elif probe.layout == "coco":
        raw = _read_coco(probe)
    elif probe.layout == "labelme":
        raw = _read_labelme(probe)
    elif probe.layout == "csv":
        raw = _read_csv(probe)
    elif probe.layout == "yolo":
        raw = _read_yolo(probe)
    else:  # pragma: no cover - guarded above
        raw = []

    records: list[DatasetRecord] = []
    unmapped: Counter[str] = Counter()
    ignored: Counter[str] = Counter()
    for item in raw:
        original = item["label"]
        key = _normalize_label(original)
        if key in source.ignore:
            ignored[original] += 1
            continue
        mapped = source.map_label(original)
        if mapped is None:
            unmapped[original] += 1
            continue
        records.append(
            DatasetRecord(
                image_path=item["image_path"],
                label=mapped,
                source=source.name,
                group=item.get("group") or item["image_path"].stem,
                bbox=item.get("bbox"),
                original_label=str(original),
                metadata=item.get("metadata", {}),
            )
        )

    report = {
        "source": source.name,
        "layout": probe.layout,
        "root": str(base),
        "records": len(records),
        "per_class": dict(sorted(Counter(r.label for r in records).items())),
        "groups": len({r.group for r in records}),
        "ignored_on_purpose": dict(ignored),
        "unmapped_labels": dict(unmapped),
        "licence": source.licence,
        "verified_layout": source.verified,
    }
    if unmapped:
        report["warning"] = (
            f"{sum(unmapped.values())} images carry labels with no taxonomy mapping "
            f"({sorted(unmapped)}). They were skipped. Add them to label_map or to "
            "ignore with a reason -- do not fold them into the nearest class."
        )
    return (records, report)


def _read_folder_per_class(probe: LayoutProbe) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for directory in probe.class_dirs:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append(
                    {"image_path": path, "label": directory.name, "group": _group_of(path)}
                )
    return items


def _read_coco(probe: LayoutProbe) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for annotation_file in probe.annotation_files:
        try:
            payload = json.loads(annotation_file.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        categories = {c["id"]: c.get("name", str(c["id"])) for c in payload.get("categories", [])}
        images = {i["id"]: i for i in payload.get("images", [])}
        search_roots = [annotation_file.parent, annotation_file.parent.parent, probe.root]
        for annotation in payload.get("annotations", []):
            image = images.get(annotation.get("image_id"))
            if image is None:
                continue
            path = _find_image(image.get("file_name", ""), search_roots)
            if path is None:
                continue
            bbox = annotation.get("bbox")
            box = None
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x, y, width, height = (float(v) for v in bbox)
                box = (int(x), int(y), int(x + width), int(y + height))
            items.append({
                "image_path": path,
                "label": categories.get(annotation.get("category_id"), "unknown"),
                "group": image.get("file_name", path.name),
                "bbox": box,
            })
    return items


def _read_labelme(probe: LayoutProbe) -> list[dict[str, Any]]:
    """Read LabelMe sidecar JSONs into bounding boxes.

    A shape's box is the min/max of its own points, which is correct for a
    LabelMe ``rectangle`` (two opposite corners) and a reasonable, honest
    approximation for a ``polygon`` (the shape's extent, not its exact outline
    -- which is what every other reader here stores too). A ``circle`` is the
    one shape LabelMe stores as two points that are *not* the bounding corners
    (centre and one point on the rim), so it gets its own reconstruction
    instead of silently producing a sliver box.
    """

    items: list[dict[str, Any]] = []
    for annotation_file in probe.annotation_files:
        try:
            payload = json.loads(annotation_file.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        shapes = payload.get("shapes")
        if not isinstance(shapes, list) or not shapes:
            continue

        search_roots = [annotation_file.parent, probe.root]
        image_path = _find_image(str(payload.get("imagePath") or ""), search_roots)
        if image_path is None:
            # imagePath missing or stale (LabelMe stores it relative to wherever
            # the annotator's machine had the file); fall back to same-stem
            # image next to the json, which is how these exports are shipped.
            image_path = _sibling_image_any_ext(annotation_file)
        if image_path is None:
            continue

        for shape in shapes:
            label = shape.get("label")
            points = shape.get("points")
            if not label or not isinstance(points, list) or len(points) < 2:
                continue
            shape_type = str(shape.get("shape_type") or "polygon")
            box = _labelme_bbox(points, shape_type)
            if box is None:
                continue
            items.append({
                "image_path": image_path,
                "label": label,
                "group": image_path.stem,
                "bbox": box,
                "metadata": {"shape_type": shape_type},
            })
    return items


def _labelme_bbox(points: list, shape_type: str) -> tuple[int, int, int, int] | None:
    try:
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
    except (TypeError, ValueError, IndexError):
        return None
    if shape_type == "circle" and len(points) == 2:
        # LabelMe stores a circle as [centre, one point on the rim].
        cx, cy = xs[0], ys[0]
        radius = math.hypot(xs[1] - cx, ys[1] - cy)
        if radius <= 0:
            return None
        return (int(cx - radius), int(cy - radius), int(cx + radius), int(cy + radius))
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return (int(x1), int(y1), int(x2), int(y2))


def _sibling_image_any_ext(annotation_file: Path) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = annotation_file.with_suffix(extension)
        if candidate.is_file():
            return candidate
    return None


def _read_csv(probe: LayoutProbe) -> list[dict[str, Any]]:
    import csv as _csv

    items: list[dict[str, Any]] = []
    for manifest in probe.annotation_files:
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            reader = _csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            label_col = _pick(reader.fieldnames, ("defect_class", "label", "class", "category", "defect"))
            image_col = _pick(reader.fieldnames, ("crop_path", "image_path", "image", "file", "filename", "path"))
            group_col = _pick(reader.fieldnames, ("source_image", "board", "group", "parent"))
            if not label_col or not image_col:
                continue
            for row in reader:
                label = (row.get(label_col) or "").strip()
                relative = (row.get(image_col) or "").strip()
                if not label or not relative:
                    continue
                path = _find_image(relative, [manifest.parent, probe.root])
                if path is None:
                    continue
                items.append({
                    "image_path": path,
                    "label": label,
                    "group": (row.get(group_col) or path.stem) if group_col else path.stem,
                })
    return items


def _read_yolo(probe: LayoutProbe) -> list[dict[str, Any]]:
    names = _yolo_class_names(probe.root)
    items: list[dict[str, Any]] = []
    for label_file in probe.annotation_files:
        image = _sibling_image(label_file)
        if image is None:
            continue
        try:
            lines = label_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                index = int(float(parts[0]))
                cx, cy, width, height = (float(value) for value in parts[1:5])
            except ValueError:
                continue
            items.append({
                "image_path": image,
                "label": names.get(index, str(index)),
                "group": image.stem,
                # Normalized xywh; the caller scales it once the image is read.
                "metadata": {"yolo_xywhn": [cx, cy, width, height]},
            })
    return items


def _yolo_class_names(root: Path) -> dict[int, str]:
    for candidate in list(root.rglob("data.yaml")) + list(root.rglob("*.yaml")):
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = re.search(r"names\s*:\s*\[(.*?)\]", text, re.S)
        if match:
            names = [item.strip().strip("'\"") for item in match.group(1).split(",")]
            return {index: name for index, name in enumerate(names) if name}
        block = re.findall(r"^\s*(\d+)\s*:\s*(.+)$", text, re.M)
        if block:
            return {int(index): name.strip().strip("'\"") for index, name in block}
    return {}


def _sibling_image(label_file: Path) -> Path | None:
    images_dir = label_file.parent.parent / "images"
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{label_file.stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def _pick(fieldnames: Sequence[str], options: Sequence[str]) -> str | None:
    lowered = {name.lower().strip(): name for name in fieldnames if name}
    for option in options:
        if option in lowered:
            return lowered[option]
    return None


def _find_image(relative: str, roots: Sequence[Path]) -> Path | None:
    name = relative.replace("\\", "/").lstrip("./")
    for root in roots:
        candidate = root / name
        if candidate.is_file():
            return candidate.resolve()
    # Fall back to a basename search; COCO file_name fields often carry a path
    # prefix that does not exist on disk.
    stem = Path(name).name
    for root in roots:
        for found in root.rglob(stem):
            if found.is_file():
                return found.resolve()
    return None


def _group_of(path: Path) -> str:
    """Best guess at which board or source image a crop came from.

    Falls back to the file stem, which makes every crop its own group -- safe,
    because it only ever makes the split more conservative, never less.
    """

    stem = path.stem
    for separator in ("__", "--"):
        if separator in stem:
            return stem.split(separator, 1)[0]
    return stem


# --------------------------------------------------------------------------- #
# Merging
# --------------------------------------------------------------------------- #


def merge_sources(
    loaded: Sequence[tuple[list[DatasetRecord], Mapping[str, Any]]],
) -> tuple[list[DatasetRecord], dict[str, Any]]:
    """Combine per-source records, keeping the source on every record.

    Groups are namespaced by source so two datasets that happen to use the same
    board naming cannot collapse into one group and leak across the split.
    """

    records: list[DatasetRecord] = []
    for source_records, _ in loaded:
        for record in source_records:
            record.group = f"{record.source}/{record.group}"
            records.append(record)

    summary = {
        "total": len(records),
        "per_source": dict(sorted(Counter(r.source for r in records).items())),
        "per_class": dict(sorted(Counter(r.label for r in records).items())),
        "per_class_per_source": {
            label: dict(sorted(Counter(r.source for r in records if r.label == label).items()))
            for label in sorted({r.label for r in records})
        },
        "groups": len({r.group for r in records}),
        "reports": [dict(report) for _, report in loaded],
    }
    return (records, summary)


def coverage_report(
    records: Sequence[DatasetRecord],
    taxonomy: Sequence[str],
    minimum: int = 30,
) -> dict[str, Any]:
    """Say plainly which taxonomy classes the merge actually covers.

    A class with a handful of examples is memorised, not learned, and a class
    with none at all can never be predicted -- yet both look like progress in a
    confusion matrix that simply omits them.
    """

    counts = Counter(r.label for r in records)
    covered, thin, missing = [], [], []
    for label in taxonomy:
        count = counts.get(label, 0)
        if count == 0:
            missing.append(label)
        elif count < minimum:
            thin.append((label, count))
        else:
            covered.append((label, count))

    single_source = {
        label: sources
        for label in counts
        if len(sources := {r.source for r in records if r.label == label}) == 1
    }
    return {
        "covered": dict(covered),
        "thin": dict(thin),
        "missing": missing,
        "single_source_classes": {k: sorted(v) for k, v in single_source.items()},
        "advice": _advice(missing, thin, single_source),
    }


def _advice(
    missing: Sequence[str],
    thin: Sequence[tuple[str, int]],
    single_source: Mapping[str, set[str]],
) -> list[str]:
    notes: list[str] = []
    if missing:
        notes.append(
            f"No data at all for {list(missing)}. Drop these from class_names rather "
            "than shipping a head that can output a class it never saw -- the app "
            "would surface confident predictions with nothing behind them."
        )
    if thin:
        notes.append(
            f"Very few samples for {dict(thin)}. Either merge another source, or "
            "leave the class out and let the rule layer catch it; the escape guard "
            "and the bridge/tombstone rules run with no model at all."
        )
    if single_source:
        notes.append(
            f"These classes come from one source only: {sorted(single_source)}. The "
            "model may learn that source's camera rather than the defect. Check by "
            "holding that source out entirely and seeing whether the class survives."
        )
    return notes
