# %% [markdown]
# # AOI PCB — bước 6.1: phân loại family linh kiện
#
# Notebook này tạo classifier cho các crop ở bước 5. Dataset mặc định được ghi cố định,
# không để người chạy phải đoán:
#
# - **Kaggle dataset:** `aryanstein/pcb-component-detection-consolidated-dataset`
# - **Dataset version đã kiểm tra:** `1`
# - **YOLO YAML ưu tiên:** `components_data_uncropped/data.yaml`
# - **Trang dữ liệu:** https://www.kaggle.com/datasets/aryanstein/pcb-component-detection-consolidated-dataset/data
#
# Trước khi Run All: chọn GPU T4 x2 hoặc mới hơn (không dùng P100 với PyTorch wheel
# không còn `sm_60`), Add Input dataset trên, và bật Internet lần đầu để
# TorchVision tải ImageNet weights. Notebook biến YOLO bounding box thành crop phân loại,
# không huấn luyện trực tiếp trên ảnh nguyên board. Test split được khóa; validation gốc
# được chia theo **ảnh cha** thành model-validation và calibration để tránh crop leakage.
#
# Đầu ra cần đưa vào app local là `best.onnx` và `model_manifest.json` trong
# `/kaggle/working/pcb_component_classifier_artifacts.zip`.

# %%
CONFIG = {
    "seed": 42,
    "dataset_slug": "aryanstein/pcb-component-detection-consolidated-dataset",
    "dataset_version": 1,
    "dataset_source": "/kaggle/input/pcb-component-detection-consolidated-dataset",
    "data_yaml": "components_data_uncropped/data.yaml",
    "work_dir": "/kaggle/working/pcb_component_classification",
    "crop_padding_ratio": 0.15,
    "model_name": "efficientnet_b0",
    "input_size": 224,
    "letterbox_value": 114,
    "batch_size": 64,
    "epochs": 40,
    "freeze_epochs": 3,
    "patience": 8,
    "head_lr": 3e-4,
    "backbone_lr": 5e-5,
    "weight_decay": 1e-4,
    "label_smoothing": 0.05,
    "max_class_weight": 10.0,
    "num_workers": 2,
    "target_accept_precision": 0.98,
    "default_accept_threshold": 0.90,
    "review_threshold": 0.50,
    "minimum_test_accuracy": 0.50,
    "minimum_test_weighted_f1": 0.50,
    "strict_audit": True,
    "max_preview": 16,
    # Đặt True chỉ khi muốn thử pipeline trên CPU. Huấn luyện đầy đủ sẽ rất chậm.
    "allow_cpu_training": False,
}

# Mapping có chủ ý từ 22 nhãn detector sang family có thể phân biệt bằng ngoại quan.
# `pads`/`pins` là negative/reject-support class, không phải family linh kiện dương.
SOURCE_TO_FAMILY = {
    "battery": "battery_power_input",
    "button": "switch_control",
    "buzzer": "acoustic",
    "capacitor": "capacitor",
    "clock": "timing",
    "connector": "connector",
    "diode": "diode",
    "display": "display",
    "fuse": "protection",
    "ic": "ic",
    "inductor": "magnetic",
    "led": "led",
    "pads": "false_crop_background",
    "pins": "false_crop_background",
    "potentiometer": "switch_control",
    "relay": "relay",
    "resistor": "resistor",
    "switch": "switch_control",
    "transformer": "magnetic",
    "transistor": "discrete_semiconductor",
}
IGNORED_SOURCE_CLASSES = {
    "heatsink": "Dataset v1 chỉ có 4 box train; không đủ để học family thermal_mechanical an toàn.",
    "transducer": "Không có mẫu train đáng tin cậy trong preset đã audit; nhãn quá rộng.",
}
UNSUPPORTED_BASELINE_FAMILIES = [
    "chip_passive",
    "module",
    "other_component",
    "physical_port",
    "power_semiconductor",
]
CLASS_NAMES = sorted(set(SOURCE_TO_FAMILY.values()))
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
print(f"Classifier taxonomy ({len(CLASS_NAMES)} classes):", CLASS_NAMES)

# %% [markdown]
# ## 1. Cài dependency export và cố định seed
#
# Kaggle đã có PyTorch/TorchVision. Cell chỉ bổ sung ONNX nếu image hiện tại thiếu,
# rồi chạy CUDA probe để phát hiện GPU/PyTorch không tương thích trước khi tạo crop.

# %%
import importlib.util
import subprocess
import sys

required = {"onnx": "onnx", "onnxruntime": "onnxruntime", "onnxscript": "onnxscript"}
missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

# %%
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import yaml


def select_training_device():
    """Fail early when the Kaggle GPU cannot execute this PyTorch build."""
    if not torch.cuda.is_available():
        if CONFIG["allow_cpu_training"]:
            print("WARNING: đang chạy CPU; huấn luyện đầy đủ sẽ rất chậm.")
            return torch.device("cpu")
        raise RuntimeError(
            "Kaggle chưa cấp GPU. Vào Settings > Accelerator, chọn GPU rồi "
            "Save/Restart session. Nếu chủ động muốn chạy CPU, đặt "
            "CONFIG['allow_cpu_training'] = True."
        )

    device_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device_index)
    capability = torch.cuda.get_device_capability(device_index)
    gpu_arch = f"sm_{capability[0]}{capability[1]}"
    compiled_arches = torch.cuda.get_arch_list()
    print(f"GPU: {gpu_name} | capability: {gpu_arch}")
    print("PyTorch CUDA architectures:", compiled_arches or "không được công bố")

    # Allocation alone is insufficient: launch a tiny kernel so an unsupported
    # architecture is reported before crops/model training consume time.
    try:
        probe = torch.ones(1, device=f"cuda:{device_index}")
        probe.add_(1)
        torch.cuda.synchronize(device_index)
    except Exception as exc:
        raise RuntimeError(
            f"GPU {gpu_name} ({gpu_arch}) không chạy được kernel của PyTorch "
            f"{torch.__version__}. Wheel hiện tại khai báo {compiled_arches}. "
            "Nếu đây là Tesla P100/sm_60 trên Kaggle, hãy vào Settings > "
            "Accelerator, chọn GPU T4 x2 hoặc GPU mới hơn, Save/Restart session "
            "rồi Run All. Chỉ chạy lại cell train trong phiên P100 sẽ không sửa "
            "được lỗi 'no kernel image is available'."
        ) from exc
    return torch.device(f"cuda:{device_index}")


DEVICE = select_training_device()
random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(CONFIG["seed"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

WORK_DIR = Path(CONFIG["work_dir"])
CROP_DIR = WORK_DIR / "crops"
REPORT_DIR = WORK_DIR / "reports"
ARTIFACT_DIR = WORK_DIR / "artifacts"
for directory in (CROP_DIR, REPORT_DIR, ARTIFACT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
print("Device:", DEVICE)

# %% [markdown]
# ## 2. Tìm đúng YOLO YAML trong Kaggle Input
#
# Cell này không ghép nhầm `/kaggle/input/components_data_uncropped/data.yaml`.
# Nó thử preset đầy đủ trước, sau đó mới quét các YAML YOLO và chấm điểm theo class map.

# %%
INPUT_ROOT = Path("/kaggle/input")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def normalize_names(raw_names):
    if isinstance(raw_names, dict):
        return [str(raw_names[key]).strip().lower() for key in sorted(raw_names, key=lambda x: int(x))]
    if isinstance(raw_names, list):
        return [str(value).strip().lower() for value in raw_names]
    return []


def load_yolo_yaml(path):
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or "train" not in data:
        return None
    if "val" not in data and "valid" not in data:
        return None
    names = normalize_names(data.get("names"))
    return data if len(names) >= 2 else None


def select_dataset_yaml():
    configured_root = Path(CONFIG["dataset_source"])
    configured_yaml = configured_root / CONFIG["data_yaml"]
    if load_yolo_yaml(configured_yaml) is not None:
        return configured_yaml.resolve()

    candidates = []
    if configured_root.exists():
        candidates.extend(configured_root.rglob("*.yaml"))
        candidates.extend(configured_root.rglob("*.yml"))
    if INPUT_ROOT.exists():
        candidates.extend(INPUT_ROOT.rglob("*.yaml"))
        candidates.extend(INPUT_ROOT.rglob("*.yml"))

    expected = set(SOURCE_TO_FAMILY) | set(IGNORED_SOURCE_CLASSES)
    scored = []
    for candidate in sorted(set(candidates)):
        data = load_yolo_yaml(candidate)
        if data is None:
            continue
        names = set(normalize_names(data.get("names")))
        score = len(names & expected)
        if "components_data_uncropped" in candidate.as_posix():
            score += 100
        scored.append((score, candidate.resolve(), data))
    if not scored:
        raise FileNotFoundError(
            "Không tìm thấy YOLO detect YAML. Trong Kaggle chọn Add Input và gắn dataset "
            f"'{CONFIG['dataset_slug']}', rồi chạy lại từ cell CONFIG."
        )
    scored.sort(key=lambda item: (-item[0], str(item[1])))
    best_score = scored[0][0]
    tied = [item for item in scored if item[0] == best_score]
    if len(tied) > 1:
        exact = [item for item in tied if set(normalize_names(item[2].get("names"))) >= expected]
        if len(exact) == 1:
            return exact[0][1]
        listing = "\n".join(f"- {item[1]}" for item in tied[:10])
        raise RuntimeError(
            "Có nhiều YOLO YAML cùng mức phù hợp; đặt CONFIG['dataset_source'] và "
            f"CONFIG['data_yaml'] chính xác:\n{listing}"
        )
    return scored[0][1]


SOURCE_YAML = select_dataset_yaml()
RAW_YAML = load_yolo_yaml(SOURCE_YAML)
SOURCE_CLASS_NAMES = normalize_names(RAW_YAML["names"])
print("Source YAML:", SOURCE_YAML)
print("Source classes:", SOURCE_CLASS_NAMES)

unknown_source_classes = sorted(
    set(SOURCE_CLASS_NAMES) - set(SOURCE_TO_FAMILY) - set(IGNORED_SOURCE_CLASSES)
)
if unknown_source_classes:
    raise RuntimeError(
        "Dataset có class chưa được mapping. Cập nhật SOURCE_TO_FAMILY có chủ ý: "
        + ", ".join(unknown_source_classes)
    )

# %% [markdown]
# ## 3. Resolve split, khử ảnh trùng chéo split và tạo crop manifest
#
# Quy tắc ưu tiên khi cùng một file ảnh xuất hiện ở nhiều split là `train > val > test`.
# Kaggle Input chỉ đọc nên notebook không sửa dữ liệu nguồn; nó chỉ bỏ đường dẫn trùng khỏi
# manifest làm việc. Bounding box chạm/vượt nhẹ mép ảnh được clip, còn dòng label sai thật
# được ghi vào `invalid_labels.csv` và chặn train khi `strict_audit=True`.

# %%
def find_dataset_mount_root(yaml_path):
    """Find the attached dataset root in both legacy and namespaced Kaggle mounts."""
    expected_name = CONFIG["dataset_slug"].split("/")[-1].lower()
    for parent in (Path(yaml_path).parent, *Path(yaml_path).parents):
        if parent.name.lower() == expected_name:
            return parent.resolve()
    return Path(yaml_path).parent.resolve()


DATASET_MOUNT_ROOT = find_dataset_mount_root(SOURCE_YAML)


def yaml_base_dir(data, yaml_path):
    yaml_parent = Path(yaml_path).parent.resolve()
    base = yaml_parent
    declared = data.get("path")
    if declared:
        declared_path = Path(str(declared).strip().replace("\\", "/"))
        candidates = (
            [declared_path]
            if declared_path.is_absolute()
            else [yaml_parent / declared_path, DATASET_MOUNT_ROOT / declared_path]
        )
        for candidate in candidates:
            if candidate.exists():
                base = candidate
                break
    return base.resolve()


def conventional_split_candidates(base, split):
    aliases = {
        "train": ["train"],
        "val": ["val", "valid", "validation"],
        "test": ["test"],
    }[split]
    candidates = []
    for alias in aliases:
        candidates.extend([base / alias / "images", base / "images" / alias])
    return candidates


def resolve_path(value, base, split, list_parent=None):
    raw_value = str(value).strip().replace("\\", "/")
    if not raw_value:
        raise ValueError(f"Split '{split}' có entry rỗng")
    path = Path(raw_value)
    if path.is_absolute():
        candidates = [path]
    else:
        parts = list(path.parts)
        leading_parent = bool(parts and parts[0] == "..")
        while parts and parts[0] == "..":
            parts.pop(0)
        stripped = Path(*parts) if parts else Path(".")
        candidates = []
        if list_parent is not None:
            # File-list paths are normally relative to the list itself.
            candidates.extend([list_parent / path, list_parent / stripped])
        if leading_parent:
            # Some public YOLO exports contain ../ even though Kaggle nests the
            # split inside the same dataset folder. Try both intended layouts.
            candidates.extend(
                [
                    SOURCE_YAML.parent / stripped,
                    Path(base) / stripped,
                    DATASET_MOUNT_ROOT / stripped,
                    SOURCE_YAML.parent / path,
                    Path(base) / path,
                ]
            )
        else:
            candidates.extend(
                [
                    Path(base) / path,
                    SOURCE_YAML.parent / path,
                    DATASET_MOUNT_ROOT / path,
                ]
            )
        if leading_parent:
            for candidate_base in (
                SOURCE_YAML.parent,
                Path(base),
                DATASET_MOUNT_ROOT,
            ):
                candidates.extend(conventional_split_candidates(candidate_base, split))

    tried = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in tried:
            continue
        tried.append(resolved)
        if resolved.exists():
            return resolved
    tried_text = "\n".join(f"- {candidate}" for candidate in tried)
    raise FileNotFoundError(
        f"Không resolve được split '{split}' từ '{value}'. Đã thử:\n{tried_text}"
    )


def resolve_split_images(entry, base, split):
    entries = entry if isinstance(entry, list) else [entry]
    images = []
    for raw_entry in entries:
        path = resolve_path(raw_entry, base, split)
        if path.is_dir():
            images.extend(
                item.resolve() for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    image = resolve_path(line, base, split, path.parent)
                    if image.suffix.lower() in IMAGE_SUFFIXES:
                        images.append(image.resolve())
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path.resolve())
        else:
            raise FileNotFoundError(f"Không resolve được split entry: {raw_entry} -> {path}")
    return sorted(set(images))


def label_path_for(image_path):
    parts = list(image_path.parts)
    image_indices = [index for index, part in enumerate(parts) if part.lower() == "images"]
    candidates = []
    for index in reversed(image_indices):
        replaced = parts.copy()
        replaced[index] = "labels"
        candidates.append(Path(*replaced).with_suffix(".txt"))
    candidates.append(image_path.with_suffix(".txt"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


DATASET_BASE = yaml_base_dir(RAW_YAML, SOURCE_YAML)
split_entries = {
    "train": RAW_YAML["train"],
    "val": RAW_YAML.get("val", RAW_YAML.get("valid")),
}
if RAW_YAML.get("test"):
    split_entries["test"] = RAW_YAML["test"]
split_images = {
    name: resolve_split_images(entry, DATASET_BASE, name)
    for name, entry in split_entries.items()
}
print("Resolved images:", {name: len(paths) for name, paths in split_images.items()})
print("Dataset mount root:", DATASET_MOUNT_ROOT)
print("Dataset base:", DATASET_BASE)

priority = {"train": 0, "val": 1, "test": 2}
seen_hashes = {}
duplicates = []
deduplicated = {name: [] for name in split_images}
for split_name in sorted(split_images, key=lambda name: priority[name]):
    for image_path in split_images[split_name]:
        image_hash = sha256_file(image_path)
        if image_hash in seen_hashes:
            duplicates.append(
                {
                    "sha256": image_hash,
                    "kept_split": seen_hashes[image_hash][0],
                    "kept_path": str(seen_hashes[image_hash][1]),
                    "dropped_split": split_name,
                    "dropped_path": str(image_path),
                }
            )
            continue
        seen_hashes[image_hash] = (split_name, image_path)
        deduplicated[split_name].append(image_path)
pd.DataFrame(duplicates).to_csv(REPORT_DIR / "cross_split_duplicates.csv", index=False)
print("Dropped exact duplicate images:", len(duplicates))

# %%
def calibration_role(image_path):
    token = f"{CONFIG['seed']}::{image_path.as_posix()}".encode()
    value = int(hashlib.sha256(token).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "calibration" if value < 0.5 else "val"


issues = []
records = []
audit_counts = Counter()
class_counts = Counter()

for source_split, images in deduplicated.items():
    for image_path in images:
        label_path = label_path_for(image_path)
        if not label_path.exists():
            issues.append({"severity": "error", "type": "missing_label", "image": str(image_path)})
            continue
        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                image.load()
        except Exception as exc:
            issues.append(
                {"severity": "error", "type": "decode_error", "image": str(image_path), "detail": str(exc)}
            )
            continue
        width, height = image.size
        output_split = calibration_role(image_path) if source_split == "val" else source_split
        seen_rows = set()
        for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw_line.strip()
            if not line:
                continue
            if line in seen_rows:
                audit_counts["duplicate_label_rows"] += 1
                continue
            seen_rows.add(line)
            fields = line.split()
            if len(fields) != 5:
                issues.append(
                    {"severity": "error", "type": "column_count", "label": str(label_path), "line": line_number}
                )
                continue
            try:
                class_value, xc, yc, bw, bh = map(float, fields)
            except ValueError:
                issues.append(
                    {"severity": "error", "type": "non_numeric", "label": str(label_path), "line": line_number}
                )
                continue
            class_id = int(class_value)
            if class_value != class_id or not 0 <= class_id < len(SOURCE_CLASS_NAMES):
                issues.append(
                    {"severity": "error", "type": "class_id", "label": str(label_path), "line": line_number}
                )
                continue
            values = np.asarray([xc, yc, bw, bh], dtype=float)
            if not np.isfinite(values).all() or not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                issues.append(
                    {"severity": "error", "type": "coordinates", "label": str(label_path), "line": line_number}
                )
                continue
            source_class = SOURCE_CLASS_NAMES[class_id]
            if source_class in IGNORED_SOURCE_CLASSES:
                audit_counts[f"ignored_{source_class}"] += 1
                continue
            family = SOURCE_TO_FAMILY[source_class]
            x1 = (xc - bw / 2) * width
            y1 = (yc - bh / 2) * height
            x2 = (xc + bw / 2) * width
            y2 = (yc + bh / 2) * height
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                audit_counts["edge_crossing_boxes_clipped"] += 1
            pad = CONFIG["crop_padding_ratio"] * max(x2 - x1, y2 - y1)
            left = max(0, math.floor(x1 - pad))
            top = max(0, math.floor(y1 - pad))
            right = min(width, math.ceil(x2 + pad))
            bottom = min(height, math.ceil(y2 + pad))
            if right <= left or bottom <= top:
                issues.append(
                    {"severity": "error", "type": "empty_crop", "label": str(label_path), "line": line_number}
                )
                continue
            token = hashlib.sha1(
                f"{image_path.as_posix()}::{line_number}::{line}".encode()
            ).hexdigest()[:16]
            crop_path = CROP_DIR / output_split / family / f"{token}.jpg"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            image.crop((left, top, right, bottom)).save(crop_path, quality=95, subsampling=0)
            records.append(
                {
                    "split": output_split,
                    "source_split": source_split,
                    "crop_path": str(crop_path),
                    "source_image": str(image_path),
                    "source_label": str(label_path),
                    "line_number": line_number,
                    "source_class": source_class,
                    "family": family,
                    "class_id": CLASS_TO_ID[family],
                    "bbox_xyxy": json.dumps([left, top, right, bottom]),
                    "source_width": width,
                    "source_height": height,
                }
            )
            class_counts[(output_split, family)] += 1
        audit_counts[f"images_{source_split}"] += 1

manifest_frame = pd.DataFrame(records)
issues_frame = pd.DataFrame(issues)
manifest_frame.to_csv(WORK_DIR / "crop_manifest.csv", index=False)
issues_frame.to_csv(REPORT_DIR / "invalid_labels.csv", index=False)
distribution = (
    manifest_frame.groupby(["split", "family"]).size().rename("count").reset_index()
    if not manifest_frame.empty
    else pd.DataFrame(columns=["split", "family", "count"])
)
distribution.to_csv(REPORT_DIR / "class_distribution.csv", index=False)
audit_report = {
    "dataset_slug": CONFIG["dataset_slug"],
    "dataset_version": CONFIG["dataset_version"],
    "source_yaml": str(SOURCE_YAML),
    "source_classes": SOURCE_CLASS_NAMES,
    "family_classes": CLASS_NAMES,
    "mapping": SOURCE_TO_FAMILY,
    "ignored_source_classes": IGNORED_SOURCE_CLASSES,
    "image_counts_after_dedup": {name: len(paths) for name, paths in deduplicated.items()},
    "crop_count": len(manifest_frame),
    "error_count": sum(item.get("severity") == "error" for item in issues),
    "counters": dict(audit_counts),
    "exact_cross_split_duplicates_dropped": len(duplicates),
}
(REPORT_DIR / "dataset_audit.json").write_text(
    json.dumps(audit_report, ensure_ascii=False, indent=2), encoding="utf-8"
)
display(distribution.pivot(index="family", columns="split", values="count").fillna(0).astype(int))
if CONFIG["strict_audit"] and audit_report["error_count"]:
    display(issues_frame.head(100))
    raise RuntimeError(
        f"Dataset audit thất bại với {audit_report['error_count']} lỗi thật. "
        f"Xem {REPORT_DIR / 'dataset_audit.json'} và invalid_labels.csv."
    )
required_splits = {"train", "val", "calibration"}
if RAW_YAML.get("test"):
    required_splits.add("test")
missing_splits = required_splits - set(manifest_frame["split"].unique())
if missing_splits:
    raise RuntimeError(f"Không tạo được crop cho split: {sorted(missing_splits)}")

# %% [markdown]
# ## 4. Xem crop và định nghĩa đúng preprocessing contract
#
# Augmentation chỉ dùng quay 0/90/180/270 và thay đổi màu nhẹ; không mirror footprint.
# Letterbox, RGB và ImageNet normalization dưới đây sẽ được ghi nguyên vẹn vào manifest
# để app local thực hiện giống hệt.

# %%
sample_frame = manifest_frame.sample(min(CONFIG["max_preview"], len(manifest_frame)), random_state=CONFIG["seed"])
columns = 4
rows = math.ceil(len(sample_frame) / columns)
figure, axes = plt.subplots(rows, columns, figsize=(14, 3.5 * rows))
axes = np.atleast_1d(axes).ravel()
for axis, (_, row) in zip(axes, sample_frame.iterrows()):
    with Image.open(row.crop_path) as image:
        axis.imshow(image.convert("RGB"))
    axis.set_title(f"{row.family}\n{row.split}")
    axis.axis("off")
for axis in axes[len(sample_frame):]:
    axis.axis("off")
plt.tight_layout()

# %%
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Letterbox:
    def __init__(self, size, value=114):
        self.size = int(size)
        self.value = int(value)

    def __call__(self, image):
        image = image.convert("RGB")
        width, height = image.size
        scale = min(self.size / max(1, width), self.size / max(1, height))
        resized = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS if scale < 1 else Image.Resampling.BICUBIC,
        )
        canvas = Image.new("RGB", (self.size, self.size), (self.value,) * 3)
        canvas.paste(resized, ((self.size - resized.width) // 2, (self.size - resized.height) // 2))
        return canvas


train_transform = transforms.Compose(
    [
        transforms.RandomChoice(
            [
                transforms.Lambda(lambda image: image),
                transforms.Lambda(lambda image: image.rotate(90, expand=True)),
                transforms.Lambda(lambda image: image.rotate(180, expand=True)),
                transforms.Lambda(lambda image: image.rotate(270, expand=True)),
            ]
        ),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        Letterbox(CONFIG["input_size"], CONFIG["letterbox_value"]),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.10, scale=(0.02, 0.08), ratio=(0.5, 2.0), value="random"),
    ]
)
eval_transform = transforms.Compose(
    [
        Letterbox(CONFIG["input_size"], CONFIG["letterbox_value"]),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


class CropDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.crop_path) as opened:
            image = opened.convert("RGB")
        return self.transform(image), int(row.class_id), str(row.crop_path)


frames = {split: manifest_frame[manifest_frame.split == split].copy() for split in required_splits}
datasets = {
    name: CropDataset(frame, train_transform if name == "train" else eval_transform)
    for name, frame in frames.items()
}
loaders = {
    name: DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=name == "train",
        num_workers=CONFIG["num_workers"],
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=CONFIG["num_workers"] > 0,
    )
    for name, dataset in datasets.items()
}
print({name: len(dataset) for name, dataset in datasets.items()})

# %% [markdown]
# ## 5. EfficientNet-B0 pretrained và train theo macro-F1
#
# Baseline dùng ImageNet pretrained `EfficientNet_B0_Weights.DEFAULT`, input 224.
# Đây là lựa chọn ưu tiên độ chính xác nhưng vẫn đủ gọn cho Raspberry Pi: khoảng
# 5,3 triệu tham số và 0,39 GFLOPs theo TorchVision, nhẹ hơn nhiều so với
# EfficientNetV2-S. YOLO không tham gia cell phân loại này.
# Chỉ classifier head được train trong 3 epoch đầu, sau đó fine-tune toàn mạng với learning
# rate backbone thấp hơn. Loss dùng class weights (không dùng thêm sampler để tránh bù hai lần).

# %%
if CONFIG["model_name"] != "efficientnet_b0":
    raise ValueError("Notebook 6.1 hiện chỉ khóa contract cho efficientnet_b0")
weights = models.EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
model = model.to(DEVICE)

train_counts = frames["train"]["class_id"].value_counts().reindex(range(len(CLASS_NAMES)), fill_value=0)
if (train_counts == 0).any():
    absent = [CLASS_NAMES[index] for index, count in train_counts.items() if count == 0]
    raise RuntimeError(f"Không thể train: family không có sample train: {absent}")
raw_class_weights = np.sqrt(train_counts.max() / train_counts.to_numpy(dtype=float))
class_weights = np.minimum(raw_class_weights, float(CONFIG["max_class_weight"]))
class_weights /= class_weights.mean()
print(
    "Class weights:",
    {name: round(float(weight), 3) for name, weight in zip(CLASS_NAMES, class_weights)},
)
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE),
    label_smoothing=CONFIG["label_smoothing"],
)


def set_backbone_trainable(trainable):
    for parameter in model.features.parameters():
        parameter.requires_grad = trainable


def build_optimizer(backbone_trainable):
    groups = [{"params": model.classifier.parameters(), "lr": CONFIG["head_lr"]}]
    if backbone_trainable:
        groups.append({"params": model.features.parameters(), "lr": CONFIG["backbone_lr"]})
    return torch.optim.AdamW(groups, weight_decay=CONFIG["weight_decay"])


@torch.no_grad()
def predict(loader, active_model=model):
    active_model.eval()
    all_logits, all_targets, all_paths = [], [], []
    for images, targets, paths in loader:
        logits = active_model(images.to(DEVICE, non_blocking=True))
        all_logits.append(logits.cpu())
        all_targets.append(targets.cpu())
        all_paths.extend(paths)
    return torch.cat(all_logits), torch.cat(all_targets), all_paths


set_backbone_trainable(False)
optimizer = build_optimizer(False)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, CONFIG["epochs"]))
USE_AMP = DEVICE.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
best_selection_score = -1.0
epochs_without_improvement = 0
history = []
checkpoint_path = ARTIFACT_DIR / "best.pt"

for epoch in range(CONFIG["epochs"]):
    if epoch == CONFIG["freeze_epochs"]:
        set_backbone_trainable(True)
        optimizer = build_optimizer(True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, CONFIG["epochs"] - epoch)
        )
    model.train()
    running_loss = 0.0
    started = time.perf_counter()
    for images, targets, _ in loaders["train"]:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * len(targets)
    scheduler.step()

    val_logits, val_targets, _ = predict(loaders["val"])
    val_predictions = val_logits.argmax(1).numpy()
    val_true = val_targets.numpy()
    val_macro_f1 = f1_score(val_true, val_predictions, average="macro", zero_division=0)
    val_weighted_f1 = f1_score(val_true, val_predictions, average="weighted", zero_division=0)
    # Prevent selection of a checkpoint that learns rare classes while
    # collapsing high-support capacitor/resistor families.
    val_selection_score = math.sqrt(max(0.0, val_macro_f1 * val_weighted_f1))
    row = {
        "epoch": epoch + 1,
        "train_loss": running_loss / len(datasets["train"]),
        "val_accuracy": float((val_true == val_predictions).mean()),
        "val_macro_f1": val_macro_f1,
        "val_weighted_f1": val_weighted_f1,
        "val_selection_score": val_selection_score,
        "seconds": time.perf_counter() - started,
        "backbone_trainable": epoch >= CONFIG["freeze_epochs"],
    }
    history.append(row)
    print(row)
    if val_selection_score > best_selection_score + 1e-4:
        best_selection_score = val_selection_score
        epochs_without_improvement = 0
        torch.save(
            {
                "state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "architecture": CONFIG["model_name"],
                "input_size": CONFIG["input_size"],
                "epoch": epoch + 1,
                "val_macro_f1": val_macro_f1,
                "val_weighted_f1": val_weighted_f1,
                "val_selection_score": val_selection_score,
            },
            checkpoint_path,
        )
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= CONFIG["patience"]:
            print("Early stopping")
            break

pd.DataFrame(history).to_csv(REPORT_DIR / "training_history.csv", index=False)
checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint["state_dict"])
print(
    "Best epoch:", checkpoint["epoch"],
    "selection score:", checkpoint["val_selection_score"],
    "macro-F1:", checkpoint["val_macro_f1"],
    "weighted-F1:", checkpoint["val_weighted_f1"],
)

# %% [markdown]
# ## 6. Temperature calibration, confidence gate và locked-test evaluation
#
# Temperature chỉ fit trên calibration split. Ngưỡng `accept` chọn theo precision mục tiêu;
# `review` nằm giữa accept và unknown. Đây là **confidence reject**, chưa phải bằng chứng OOD
# hoàn chỉnh: cần thêm ảnh crop ngoài taxonomy/camera thật trước production.

# %%
def fit_temperature(logits, targets):
    # `clone()` converts tensors created by inference_mode in an older/executed
    # predict cell into ordinary tensors that autograd may save for backward.
    calibration_logits = logits.detach().clone().to(dtype=torch.float32)
    calibration_targets = targets.detach().clone().to(dtype=torch.long)
    if calibration_logits.ndim != 2 or calibration_targets.ndim != 1:
        raise ValueError(
            "Temperature calibration expects logits [N, C] and targets [N]"
        )
    if len(calibration_logits) == 0 or len(calibration_logits) != len(calibration_targets):
        raise ValueError("Calibration split is empty or logits/targets lengths differ")
    if not torch.isfinite(calibration_logits).all():
        raise ValueError("Calibration logits contain NaN/Inf")

    log_temperature = torch.zeros(
        1,
        dtype=calibration_logits.dtype,
        device=calibration_logits.device,
        requires_grad=True,
    )
    calibration_optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.05, max_iter=100
    )

    def calibration_closure():
        calibration_optimizer.zero_grad(set_to_none=True)
        candidate_temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.cross_entropy(
            calibration_logits / candidate_temperature,
            calibration_targets,
        )
        loss.backward()
        return loss

    calibration_optimizer.step(calibration_closure)
    fitted_temperature = float(
        log_temperature.detach().exp().clamp(0.05, 20.0).item()
    )
    return fitted_temperature, calibration_logits, calibration_targets


calibration_logits, calibration_targets, _ = predict(loaders["calibration"])
temperature, calibration_logits, calibration_targets = fit_temperature(
    calibration_logits,
    calibration_targets,
)
calibration_probabilities = torch.softmax(calibration_logits / temperature, dim=1)
calibration_confidence, calibration_prediction = calibration_probabilities.max(1)
calibration_correct = calibration_prediction.eq(calibration_targets)


def choose_accept_threshold(confidence, correct, target_precision, fallback):
    best = None
    for threshold in np.linspace(0.50, 0.99, 100):
        accepted = confidence.numpy() >= threshold
        if accepted.sum() == 0:
            continue
        precision = float(correct.numpy()[accepted].mean())
        coverage = float(accepted.mean())
        if precision >= target_precision and (best is None or coverage > best[2]):
            best = (float(threshold), precision, coverage)
    if best is None:
        accepted = confidence.numpy() >= fallback
        precision = float(correct.numpy()[accepted].mean()) if accepted.any() else 0.0
        return float(fallback), precision, float(accepted.mean())
    return best


accept_threshold, calibration_accept_precision, calibration_coverage = choose_accept_threshold(
    calibration_confidence,
    calibration_correct,
    CONFIG["target_accept_precision"],
    CONFIG["default_accept_threshold"],
)
review_threshold = min(float(CONFIG["review_threshold"]), accept_threshold)
print(
    {
        "temperature": temperature,
        "accept_threshold": accept_threshold,
        "review_threshold": review_threshold,
        "calibration_accept_precision": calibration_accept_precision,
        "calibration_coverage": calibration_coverage,
    }
)

# %%
evaluation_split = "test" if "test" in loaders else "val"
test_logits, test_targets, test_paths = predict(loaders[evaluation_split])
test_probabilities = torch.softmax(test_logits / temperature, dim=1)
test_confidence, test_prediction = test_probabilities.max(1)
test_true = test_targets.numpy()
test_pred = test_prediction.numpy()
report = classification_report(
    test_true,
    test_pred,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)
per_class = pd.DataFrame(report).T
per_class.to_csv(REPORT_DIR / "per_class_metrics.csv")
matrix = confusion_matrix(test_true, test_pred, labels=list(range(len(CLASS_NAMES))) )
np.savetxt(REPORT_DIR / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")

plt.figure(figsize=(16, 14))
sns.heatmap(matrix, cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=180)
plt.show()

decisions = np.where(
    test_confidence.numpy() >= accept_threshold,
    "accept",
    np.where(test_confidence.numpy() >= review_threshold, "review", "unknown"),
)
test_predictions = pd.DataFrame(
    {
        "crop_path": test_paths,
        "true_family": [CLASS_NAMES[index] for index in test_true],
        "predicted_family": [CLASS_NAMES[index] for index in test_pred],
        "probability": test_confidence.numpy(),
        "unknown_score": 1.0 - test_confidence.numpy(),
        "decision": decisions,
        "correct": test_true == test_pred,
    }
)
test_predictions.to_csv(REPORT_DIR / "test_predictions.csv", index=False)
metrics_summary = {
    "evaluation_split": evaluation_split,
    "accuracy": float((test_true == test_pred).mean()),
    "macro_f1": float(f1_score(test_true, test_pred, average="macro", zero_division=0)),
    "weighted_f1": float(f1_score(test_true, test_pred, average="weighted", zero_division=0)),
    "temperature": temperature,
    "accept_threshold": accept_threshold,
    "review_threshold": review_threshold,
    "accepted_precision": float(
        test_predictions.loc[test_predictions.decision == "accept", "correct"].mean()
    ) if (test_predictions.decision == "accept").any() else None,
    "accepted_coverage": float((test_predictions.decision == "accept").mean()),
    "unknown_policy_limit": "Confidence reject only; OOD behavior is not validated by this dataset.",
}
(REPORT_DIR / "metrics_summary.json").write_text(
    json.dumps(metrics_summary, indent=2), encoding="utf-8"
)
display(per_class)
print(metrics_summary)

quality_failures = []
if metrics_summary["accuracy"] < float(CONFIG["minimum_test_accuracy"]):
    quality_failures.append(
        f"accuracy={metrics_summary['accuracy']:.3f} < {CONFIG['minimum_test_accuracy']:.3f}"
    )
if metrics_summary["weighted_f1"] < float(CONFIG["minimum_test_weighted_f1"]):
    quality_failures.append(
        f"weighted_f1={metrics_summary['weighted_f1']:.3f} < "
        f"{CONFIG['minimum_test_weighted_f1']:.3f}"
    )
if quality_failures:
    raise RuntimeError(
        "Classifier quality gate failed; refusing to export a deployment artifact: "
        + "; ".join(quality_failures)
    )

# %% [markdown]
# ## 7. Export ONNX raw logits, kiểm tra parity và tạo model manifest
#
# App chỉ nhận ONNX, không nhận checkpoint `.pt` của classifier. Manifest khóa class order,
# RGB/letterbox/ImageNet normalization, temperature, thresholds, dataset và SHA-256 model.

# %%
import onnx
import onnxruntime as ort

model = model.cpu().eval()
dummy = torch.zeros(1, 3, CONFIG["input_size"], CONFIG["input_size"], dtype=torch.float32)
onnx_path = ARTIFACT_DIR / "best.onnx"
exporter = "dynamo"
try:
    torch.onnx.export(
        model,
        (dummy,),
        onnx_path,
        input_names=["images"],
        output_names=["logits"],
        dynamic_shapes={"images": {0: "batch"}},
        opset_version=18,
        dynamo=True,
    )
except Exception as exc:
    print("Dynamo exporter không tương thích runtime hiện tại; dùng legacy fallback:", exc)
    exporter = "legacy"
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
with torch.inference_mode():
    torch_logits = model(dummy).numpy()
onnx_logits = session.run(["logits"], {"images": dummy.numpy()})[0]
max_abs_error = float(np.max(np.abs(torch_logits - onnx_logits)))
if max_abs_error > 1e-3:
    raise RuntimeError(f"ONNX parity vượt tolerance: {max_abs_error}")
dynamic_batch_logits = session.run(
    ["logits"], {"images": np.repeat(dummy.numpy(), 2, axis=0)}
)[0]
if dynamic_batch_logits.shape != (2, len(CLASS_NAMES)):
    raise RuntimeError(
        f"ONNX dynamic batch sai contract: {dynamic_batch_logits.shape}"
    )

onnx_sha256 = sha256_file(onnx_path)
created_at = datetime.now(timezone.utc).isoformat()
manifest = {
    "schema_version": "pcb-component-classifier/1.0",
    "task": "component_family_classification",
    "model_format": "onnx",
    "created_at": created_at,
    "class_names": CLASS_NAMES,
    "input": {
        "name": "images",
        "size": [CONFIG["input_size"], CONFIG["input_size"]],
        "layout": "NCHW",
        "dtype": "float32",
        "color_space": "RGB",
        "resize_mode": "letterbox",
        "letterbox_value": CONFIG["letterbox_value"],
        "scale": 1.0 / 255.0,
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    },
    "output": {"name": "logits", "semantics": "raw_logits", "dynamic_batch": True},
    "calibration": {
        "method": "temperature_scaling",
        "temperature": temperature,
        "split": "calibration_from_original_validation_grouped_by_parent_image",
    },
    "decision_thresholds": {
        "accept": accept_threshold,
        "review": review_threshold,
        "accept_by_class": {},
        "unknown_score": "1 - max(softmax(logits / temperature))",
    },
    "model": {
        "filename": "best.onnx",
        "sha256": onnx_sha256,
        "version": f"efficientnet_b0-{created_at[:10]}",
        "architecture": "efficientnet_b0",
        "pretrained_weights": "EfficientNet_B0_Weights.DEFAULT",
        "exporter": exporter,
        "max_abs_parity_error": max_abs_error,
    },
    "dataset": {
        "provider": "Kaggle",
        "slug": CONFIG["dataset_slug"],
        "version": CONFIG["dataset_version"],
        "source_yaml": str(SOURCE_YAML),
        "source_classes": SOURCE_CLASS_NAMES,
        "source_to_family": SOURCE_TO_FAMILY,
        "ignored_source_classes": IGNORED_SOURCE_CLASSES,
        "license_note": (
            "Kaggle uploader lists Apache 2.0. This is a consolidated dataset; "
            "verify licenses of every upstream source before commercial use."
        ),
    },
    "taxonomy": {
        "supported_families": CLASS_NAMES,
        "unsupported_baseline_families": UNSUPPORTED_BASELINE_FAMILIES,
        "pads_and_pins_policy": "false_crop_background/reject-support",
        "transducer_policy": "ignored because the preset has no usable train support",
        "scope_limit": (
            "Visual family only. Do not infer IC function/part number; use OCR and BOM matching later."
        ),
    },
    "deployment": {
        "target": "raspberry_pi_arm64_cpu",
        "runtime": "onnxruntime",
        "precision": "fp32",
        "quantization_status": (
            "not_applied; benchmark FP32 first, then calibrate/evaluate INT8 on held-out "
            "camera crops before replacing the production artifact"
        ),
    },
    "metrics": metrics_summary,
    "training": {
        key: CONFIG[key]
        for key in (
            "seed",
            "model_name",
            "input_size",
            "batch_size",
            "epochs",
            "freeze_epochs",
            "patience",
            "head_lr",
            "backbone_lr",
            "weight_decay",
            "label_smoothing",
        )
    },
    "runtime_versions": {
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
    },
}
manifest_path = ARTIFACT_DIR / "model_manifest.json"
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
(ARTIFACT_DIR / "taxonomy.json").write_text(
    json.dumps(manifest["taxonomy"], ensure_ascii=False, indent=2), encoding="utf-8"
)
(ARTIFACT_DIR / "label_mapping.json").write_text(
    json.dumps(
        {"source_to_family": SOURCE_TO_FAMILY, "ignored": IGNORED_SOURCE_CLASSES},
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
(REPORT_DIR / "onnx_verification.json").write_text(
    json.dumps(
        {
            "checker": "passed",
            "providers": session.get_providers(),
            "input": session.get_inputs()[0].name,
            "output": session.get_outputs()[0].name,
            "max_abs_error": max_abs_error,
            "dynamic_batch_shape": list(dynamic_batch_logits.shape),
            "sha256": onnx_sha256,
        },
        indent=2,
    ),
    encoding="utf-8",
)
print("ONNX:", onnx_path)
print("Manifest:", manifest_path)

# %% [markdown]
# ## 8. Đóng gói để đưa model vào app
#
# Tải ZIP, giải nén và nạp **đồng thời** `best.onnx` + `model_manifest.json` ở sidebar
# “Model phân loại 6.1”. `best.pt` chỉ để resume/debug trong môi trường tin cậy.

# %%
artifact_zip = Path("/kaggle/working/pcb_component_classifier_artifacts.zip")
package_files = [
    ARTIFACT_DIR / "best.onnx",
    ARTIFACT_DIR / "best.pt",
    ARTIFACT_DIR / "model_manifest.json",
    ARTIFACT_DIR / "taxonomy.json",
    ARTIFACT_DIR / "label_mapping.json",
    WORK_DIR / "crop_manifest.csv",
    REPORT_DIR / "dataset_audit.json",
    REPORT_DIR / "invalid_labels.csv",
    REPORT_DIR / "cross_split_duplicates.csv",
    REPORT_DIR / "class_distribution.csv",
    REPORT_DIR / "training_history.csv",
    REPORT_DIR / "metrics_summary.json",
    REPORT_DIR / "per_class_metrics.csv",
    REPORT_DIR / "confusion_matrix.csv",
    REPORT_DIR / "confusion_matrix.png",
    REPORT_DIR / "test_predictions.csv",
    REPORT_DIR / "onnx_verification.json",
]
with zipfile.ZipFile(artifact_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in package_files:
        if path.exists():
            archive.write(path, arcname=path.name)
print(f"DONE: {artifact_zip} ({artifact_zip.stat().st_size / 1024**2:.1f} MB)")
print("Đưa vào app: best.onnx + model_manifest.json")
