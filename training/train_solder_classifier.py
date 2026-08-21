"""Train the step-6.2 solder-defect classifier and export what the app expects.

Input is the dataset that ``scripts/export_solder_dataset.py`` writes, once the
``defect_class`` column has been filled in. Output is exactly two files:

    best.onnx            raw-logit model
    model_manifest.json  class order, preprocessing, calibration, thresholds

Drop both into the app and step 6.2 picks them up; nothing else changes.

    python training/train_solder_classifier.py datasets/solder_v1 ^
        --output models/solder --epochs 30

Three things here are not defaults you should change without reason:

* **The split is by board, not by ROI.** Two joints from the same board share
  its lighting, its focus and often its operator error. Splitting by ROI puts
  near-duplicates on both sides and reports a score the line will never see.
* **The reported metric is escape rate, not accuracy.** A line that is 99.5%
  good scores 99.5% by calling everything good. What matters is how many
  defects were called good (escapes) and how many good joints were called
  defective (false calls).
* **Class weights are on.** Without them the loss is dominated by the good
  class and the model learns to abstain from ever failing anything.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# torch's ONNX exporter prints status with emoji. On a Windows console that
# defaults to cp1252 the print itself raises UnicodeEncodeError and kills the
# run *after* every epoch has already been spent, so widen the streams first.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

MANIFEST_SCHEMA = "pcb-solder-defect-classifier/1.0"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and export the step-6.2 solder-defect classifier."
    )
    parser.add_argument("dataset", help="Folder written by export_solder_dataset.py.")
    parser.add_argument("--output", required=True, help="Where best.onnx and the manifest go.")
    parser.add_argument(
        "--csv",
        default="solder_dataset.csv",
        help="Label sheet inside the dataset folder (default: solder_dataset.csv).",
    )
    parser.add_argument(
        "--scope",
        default="joint",
        choices=["joint", "component"],
        help="Train on joint ROIs or on component body views (default: joint).",
    )
    parser.add_argument("--good-label", default="good", help="Label meaning 'no defect'.")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=10,
        help="Refuse to train a class with fewer samples than this (default: 10).",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def load_rows(dataset: Path, csv_name: str, scope: str) -> list[dict[str, str]]:
    manifest = dataset / csv_name
    if not manifest.is_file():
        raise SystemExit(f"Label sheet not found: {manifest}")
    wanted_kind = "body" if scope == "component" else "joint"
    rows: list[dict[str, str]] = []
    unlabelled = 0
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("kind") or "joint") != wanted_kind:
                continue
            label = (row.get("defect_class") or "").strip()
            if not label:
                unlabelled += 1
                continue
            rows.append({**row, "defect_class": label})
    if unlabelled:
        print(f"Skipped {unlabelled} unlabelled rows (empty defect_class).")
    if not rows:
        raise SystemExit(
            "No labelled rows found. Fill in the 'defect_class' column first."
        )
    return rows


def split_by_board(
    rows: list[dict[str, str]], val_fraction: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Hold out whole boards, never individual ROIs.

    Joints from one board share lighting, focus and paste deposition. Splitting
    per ROI leaks all of that across the boundary and inflates the score.
    """

    boards = sorted({row.get("source_image", "") for row in rows})
    if len(boards) < 2:
        raise SystemExit(
            "All ROIs come from a single board image. A model validated on the "
            "same board it trained on says nothing; export more boards first."
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(boards))
    holdout = max(1, int(round(len(boards) * val_fraction)))
    validation_boards = {boards[int(index)] for index in order[:holdout]}
    train = [row for row in rows if row.get("source_image") not in validation_boards]
    validation = [row for row in rows if row.get("source_image") in validation_boards]
    if not train or not validation:
        raise SystemExit("Board split left one side empty; adjust --val-fraction.")
    print(
        f"Split by board: {len(boards) - holdout} train / {holdout} validation boards "
        f"({len(train)} / {len(validation)} ROIs)"
    )
    return train, validation


def check_class_balance(rows: list[dict[str, str]], minimum: int) -> list[str]:
    counts = Counter(row["defect_class"] for row in rows)
    print("\nClass counts:")
    for label, count in counts.most_common():
        print(f"  {label:18s} {count:6d}")
    too_few = [label for label, count in counts.items() if count < minimum]
    if too_few:
        raise SystemExit(
            f"These classes have fewer than {minimum} samples: {sorted(too_few)}. "
            "Collect more before training, or merge them into another class -- a "
            "class the model sees a handful of times is memorised, not learned."
        )
    if len(counts) < 2:
        raise SystemExit("At least two classes are required.")
    return sorted(counts)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        import torchvision
    except ImportError as exc:
        raise SystemExit(
            f"Training needs torch and torchvision: {exc}\n"
            "  pip install torch torchvision"
        ) from exc
    import cv2

    from aoi_pipeline import letterbox_normalize

    dataset_dir = Path(args.dataset).expanduser().resolve()
    rows = load_rows(dataset_dir, args.csv, args.scope)
    class_names = check_class_balance(rows, args.min_per_class)
    if args.good_label not in class_names:
        raise SystemExit(
            f"--good-label '{args.good_label}' is not among the labels: {class_names}"
        )
    train_rows, val_rows = split_by_board(rows, args.val_fraction, args.seed)

    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    size = (args.image_size, args.image_size)
    index_of = {name: index for index, name in enumerate(class_names)}
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)

    class SolderDataset(Dataset):
        def __init__(self, items: list[dict[str, str]], augment: bool) -> None:
            self.items = items
            self.augment = augment

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int):
            row = self.items[index]
            path = dataset_dir / row["crop_path"]
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"Could not read crop: {path}")
            if self.augment:
                image = _augment(image)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized = letterbox_normalize(rgb, size, (114, 114, 114))
            tensor = resized.astype(np.float32) / 255.0
            tensor = (tensor - mean) / std
            return (
                torch.from_numpy(np.ascontiguousarray(tensor.transpose(2, 0, 1))),
                index_of[row["defect_class"]],
            )

    def _augment(image: np.ndarray) -> np.ndarray:
        # Flips and small rotations only. A solder joint has no canonical
        # orientation, but heavy colour jitter would erase the specular cue the
        # model is meant to learn.
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 0)
        if np.random.rand() < 0.3:
            angle = float(np.random.uniform(-12, 12))
            height, width = image.shape[:2]
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
            image = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
        if np.random.rand() < 0.4:
            scale = float(np.random.uniform(0.88, 1.12))
            image = np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        return image

    train_loader = DataLoader(
        SolderDataset(train_rows, augment=True),
        batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False,
    )
    val_loader = DataLoader(
        SolderDataset(val_rows, augment=False),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    model = torchvision.models.mobilenet_v3_small(weights="DEFAULT")
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
    model = model.to(device)

    counts = Counter(row["defect_class"] for row in train_rows)
    weights = torch.tensor(
        [len(train_rows) / (len(class_names) * counts[name]) for name in class_names],
        dtype=torch.float32, device=device,
    )
    print(f"Class weights: {dict(zip(class_names, [round(float(w), 2) for w in weights]))}")
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_state = None
    best_score = -1.0
    good_index = index_of[args.good_label]

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * images.size(0)
        scheduler.step()

        model.eval()
        predictions: list[int] = []
        truths: list[int] = []
        with torch.no_grad():
            for images, targets in val_loader:
                logits = model(images.to(device))
                predictions.extend(logits.argmax(1).cpu().tolist())
                truths.extend(targets.tolist())
        escape, false_call = _line_metrics(truths, predictions, good_index)
        # Escapes dominate: a model that never lets a defect through is worth
        # more than one with a better average.
        score = (1.0 - escape) * 2.0 + (1.0 - false_call)
        if score > best_score:
            best_score, best_state = score, {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        print(
            f"epoch {epoch:3d}  loss {total_loss / max(1, len(train_rows)):.4f}  "
            f"escape {escape:.3%}  false_call {false_call:.3%}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval().to("cpu")

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    onnx_path = output / "best.onnx"
    dummy = torch.zeros(1, 3, args.image_size, args.image_size)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        # 18, not 17: newer torch exporters emit 18 and then fail the automatic
        # down-conversion, leaving an 18 graph plus a confusing error either way.
        opset_version=18,
    )
    _collapse_external_data(onnx_path)
    _verify_onnx(onnx_path, model, dummy, torch)

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "task": "solder_defect_classification",
        "scope": args.scope,
        "model_format": "onnx",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "class_names": class_names,
        "good_label": args.good_label,
        "input": {
            "name": "input",
            "size": [args.image_size, args.image_size],
            "color_space": "RGB",
            "resize_mode": "letterbox",
            "letterbox_value": 114,
            "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        },
        "output": {"name": "logits", "type": "raw_logits"},
        "calibration": {"temperature": 1.0},
        "decision_thresholds": {"accept": 0.85, "review": 0.50, "accept_by_class": {}},
        "model": {
            "version": f"solder-{args.scope}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}",
            "architecture": "mobilenet_v3_small",
            "sha256": _sha256(onnx_path),
        },
        "training": {
            "boards_total": len({row.get("source_image") for row in rows}),
            "roi_train": len(train_rows),
            "roi_val": len(val_rows),
            "class_counts": dict(Counter(row["defect_class"] for row in rows)),
            "epochs": args.epochs,
            "seed": args.seed,
        },
    }
    manifest_path = output / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote {onnx_path}")
    print(f"Wrote {manifest_path}")
    print(
        "\nThe accept/review thresholds in the manifest are starting points. "
        "Sweep them against your own validation set and set the operating point "
        "your line can staff, then edit decision_thresholds before deploying."
    )
    return 0


def _line_metrics(truths, predictions, good_index: int) -> tuple[float, float]:
    """Escape rate and false-call rate -- the two numbers a line is judged on."""

    truths = np.asarray(truths)
    predictions = np.asarray(predictions)
    defects = truths != good_index
    goods = truths == good_index
    escape = float(np.mean(predictions[defects] == good_index)) if defects.any() else 0.0
    false_call = float(np.mean(predictions[goods] != good_index)) if goods.any() else 0.0
    return escape, false_call


def _collapse_external_data(path: Path) -> None:
    """Fold weights back into the .onnx file itself.

    torch's exporter may write tensors to a sibling ``best.onnx.data``. The app
    is documented as needing exactly two files -- the .onnx and its manifest --
    and the manifest's SHA-256 covers only the .onnx, so a split export loads on
    the machine that trained it and fails with "External data path validation
    failed" everywhere else.
    """

    try:
        import onnx
    except ImportError:
        print("onnx not installed; cannot make the export self-contained.", file=sys.stderr)
        return
    model = onnx.load(str(path))
    onnx.save_model(model, str(path), save_as_external_data=False)
    for orphan in path.parent.glob(f"{path.name}*.data"):
        orphan.unlink()
    for orphan in path.parent.glob(f"{path.stem}*.data"):
        orphan.unlink()


def _verify_onnx(path: Path, model, dummy, torch) -> None:
    """Refuse to ship an export that does not reproduce the trained model."""

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skipped export verification.", file=sys.stderr)
        return
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"input": dummy.numpy()})[0]
    with torch.no_grad():
        torch_out = model(dummy).numpy()
    difference = float(np.max(np.abs(onnx_out - torch_out)))
    if difference > 1e-3:
        raise SystemExit(f"ONNX export differs from the torch model by {difference:.5f}")
    print(f"ONNX verified against torch (max diff {difference:.2e})")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
