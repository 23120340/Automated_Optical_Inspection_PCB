# %% [markdown]
# # PCB package classifier — step 5.2 (7 lớp)
#
# Notebook này **chỉ chạy sau khi người duyệt đã giải quyết toàn bộ `unknown`**
# trong `label_packages.html` và pack bằng
# `scripts/pack_package_classification_dataset.py`. Dataset được chia theo
# board/`scene_id`; notebook từ chối rò rỉ một board sang nhiều split.
#
# Artifact sinh ra vẫn **mặc định tắt**. Sau khi tải về local phải chạy
# `scripts/evaluate_package_roi_gate.py`; notebook không tự promote model.

# %%
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import zipfile

import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, recall_score
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


SEED = 52026
INPUT_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 35
PATIENCE = 7
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
PACKAGE_CLASSES = [
    "hai_chan",
    "tru_dung",
    "goi_nho",
    "ic_hai_ben",
    "ic_bon_ben",
    "ic_khong_chan",
    "connector",
]
DATASET_SCHEMA = "aoi-package-imagefolder/1.0"
MANIFEST_SCHEMA = "pcb-package-classifier/1.0"
TASK = "component_package_classification"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)

# %% [markdown]
# ## 1. Tìm và kiểm contract dataset
#
# Không chọn file “gần đúng”: phải có đúng một ZIP chứa `dataset_manifest.json`
# với taxonomy cố định. Nếu còn `unknown`, packer đã dừng trước bước này.

# %%
INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/package_classifier")
candidate_zips = []
for candidate in sorted(INPUT_ROOT.rglob("*.zip")):
    try:
        with zipfile.ZipFile(candidate) as archive:
            if "dataset_manifest.json" in archive.namelist():
                candidate_zips.append(candidate)
    except zipfile.BadZipFile:
        continue
if len(candidate_zips) != 1:
    raise RuntimeError(
        "Cần đúng một package dataset ZIP có dataset_manifest.json; "
        f"tìm thấy {candidate_zips}"
    )
DATASET_ZIP = candidate_zips[0]
if WORK_ROOT.exists():
    shutil.rmtree(WORK_ROOT)
WORK_ROOT.mkdir(parents=True)
with zipfile.ZipFile(DATASET_ZIP) as archive:
    archive.extractall(WORK_ROOT)
dataset_manifest = json.loads(
    (WORK_ROOT / "dataset_manifest.json").read_text(encoding="utf-8")
)
if dataset_manifest.get("schema_version") != DATASET_SCHEMA:
    raise RuntimeError("Sai dataset schema")
if dataset_manifest.get("task") != TASK:
    raise RuntimeError("Sai task: không dùng dataset family 6.1 cho package 5.2")
if dataset_manifest.get("class_names") != PACKAGE_CLASSES:
    raise RuntimeError("Sai thứ tự 7 lớp package")
if dataset_manifest.get("split_unit") != "board_scene_id":
    raise RuntimeError("Dataset không khai báo split theo board/scene_id")

split_groups = {
    split: set(dataset_manifest["split_groups"][split])
    for split in ("train", "val", "test")
}
if any(
    split_groups[left] & split_groups[right]
    for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
):
    raise RuntimeError("Rò rỉ board giữa train/val/test")
samples = dataset_manifest.get("samples")
if not isinstance(samples, list) or not samples:
    raise RuntimeError("Dataset không có samples")
for split in ("train", "val", "test"):
    present = {row["class"] for row in samples if row["split"] == split}
    missing = set(PACKAGE_CLASSES) - present
    if missing:
        raise RuntimeError(f"Split {split} thiếu lớp {sorted(missing)}")
print(DATASET_ZIP, len(samples), "samples")
print({split: len(groups) for split, groups in split_groups.items()}, "boards")

# %% [markdown]
# ## 2. Dataset RGB + letterbox 128
#
# Hàm này khớp runtime `ONNXPackageClassifier`: RGB, giữ tỉ lệ, pad 114,
# ImageNet mean/std. Class index lấy từ contract, không lấy thứ tự alphabet của
# thư mục (`ImageFolder` sẽ đặt `connector` lên đầu và làm sai toàn bộ mapping).

# %%
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def letterbox_rgb(image: Image.Image, size: int = INPUT_SIZE) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    scale = min(size / max(1, width), size / max(1, height))
    resized = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


train_transform = transforms.Compose([
    transforms.Lambda(letterbox_rgb),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(12, fill=(114, 114, 114)),
    transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_transform = transforms.Compose([
    transforms.Lambda(letterbox_rgb),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class PackageDataset(Dataset):
    def __init__(self, split: str, transform) -> None:
        self.rows = [row for row in samples if row["split"] == split]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(WORK_ROOT / row["path"])
        return self.transform(image), PACKAGE_CLASSES.index(row["class"])


datasets = {
    "train": PackageDataset("train", train_transform),
    "val": PackageDataset("val", eval_transform),
    "test": PackageDataset("test", eval_transform),
}
loaders = {
    split: DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=split == "train",
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )
    for split, dataset in datasets.items()
}
train_counts = Counter(row["class"] for row in datasets["train"].rows)
class_weights = torch.tensor(
    [len(datasets["train"]) / (len(PACKAGE_CLASSES) * train_counts[name])
     for name in PACKAGE_CLASSES],
    dtype=torch.float32,
    device=DEVICE,
)
print({split: len(dataset) for split, dataset in datasets.items()})
print("train class counts:", train_counts)

# %% [markdown]
# ## 3. MobileNetV3-small và train theo macro recall
#
# Đây là classifier crop nhỏ; không train lại detector. Checkpoint tốt nhất chọn
# bằng macro recall validation để lớp `hai_chan` đông không che bốn lớp hiếm.

# %%
weights = models.MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=weights)
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, len(PACKAGE_CLASSES))
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")


@torch.no_grad()
def predict(loader: DataLoader):
    model.eval()
    logits_parts, target_parts = [], []
    for images, targets in loader:
        logits_parts.append(model(images.to(DEVICE)).cpu())
        target_parts.append(targets.cpu())
    return torch.cat(logits_parts), torch.cat(target_parts)


best_recall = -1.0
best_state = None
stale_epochs = 0
history = []
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for images, targets in loaders["train"]:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += float(loss.detach()) * len(targets)
    scheduler.step()
    val_logits, val_targets = predict(loaders["val"])
    val_predictions = val_logits.argmax(1).numpy()
    val_recall = recall_score(
        val_targets.numpy(), val_predictions, average="macro", zero_division=0
    )
    row = {
        "epoch": epoch,
        "train_loss": running_loss / len(datasets["train"]),
        "val_macro_recall": float(val_recall),
    }
    history.append(row)
    print(row)
    if val_recall > best_recall + 1e-5:
        best_recall = float(val_recall)
        best_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        stale_epochs = 0
    else:
        stale_epochs += 1
        if stale_epochs >= PATIENCE:
            print("early stop")
            break
if best_state is None:
    raise RuntimeError("Không tạo được checkpoint")
model.load_state_dict(best_state)
model.to(DEVICE).eval()

# %% [markdown]
# ## 4. Temperature calibration và test board-held-out

# %%
val_logits, val_targets = predict(loaders["val"])
log_temperature = torch.zeros(1, requires_grad=True)
calibration_optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=80)
calibration_loss = nn.CrossEntropyLoss()


def calibration_closure():
    calibration_optimizer.zero_grad()
    temperature = log_temperature.exp().clamp(0.05, 20.0)
    loss = calibration_loss(val_logits / temperature, val_targets)
    loss.backward()
    return loss


calibration_optimizer.step(calibration_closure)
temperature = float(log_temperature.detach().exp().clamp(0.05, 20.0))
test_logits, test_targets = predict(loaders["test"])
test_probabilities = torch.softmax(test_logits / temperature, dim=1)
test_predictions = test_probabilities.argmax(1).numpy()
truth = test_targets.numpy()
test_macro_recall = float(
    recall_score(truth, test_predictions, average="macro", zero_division=0)
)
matrix = confusion_matrix(
    truth, test_predictions, labels=list(range(len(PACKAGE_CLASSES)))
).astype(int)
print(classification_report(
    truth, test_predictions, target_names=PACKAGE_CLASSES, digits=4, zero_division=0
))
print("temperature:", temperature)
print("test macro recall:", test_macro_recall)
print(matrix)
dual_index = PACKAGE_CLASSES.index("ic_hai_ben")
hidden_index = PACKAGE_CLASSES.index("ic_khong_chan")
dangerous_confusions = int(
    matrix[dual_index, hidden_index] + matrix[hidden_index, dual_index]
)
print("ic_hai_ben <-> ic_khong_chan confusions:", dangerous_confusions)

# %% [markdown]
# ## 5. Export ONNX raw logits + manifest ghim toàn bộ contract

# %%
import onnxruntime as ort

ARTIFACT_ROOT = Path("/kaggle/working/package_classifier_artifacts")
if ARTIFACT_ROOT.exists():
    shutil.rmtree(ARTIFACT_ROOT)
ARTIFACT_ROOT.mkdir(parents=True)
onnx_path = ARTIFACT_ROOT / "best.onnx"
cpu_model = deepcopy(model).cpu().eval()
example = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
# Nêu `dynamo` TƯỜNG MINH và có đường thứ hai. Mặc định của torch đổi theo
# phiên bản: từ 2.9 nó là `dynamo=True`, đường đó uỷ quyền cho `onnxscript`, và
# image Kaggle không có gói ấy. Cell này chạy sau khi mọi giờ GPU đã tiêu xong,
# nên một lời gọi trần là một điểm hỏng đơn ở đúng chỗ đắt nhất.
_export_kwargs = dict(
    input_names=["images"],
    output_names=["logits"],
    dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
    do_constant_folding=True,
)
try:
    torch.onnx.export(cpu_model, example, onnx_path, dynamo=False, **_export_kwargs)
    onnx_exporter = "torchscript"
except Exception as exc:  # noqa: BLE001 - bộ xuất nào hỏng cũng phải sang đường kia
    print(
        f"Bộ xuất TorchScript không dùng được ({type(exc).__name__}: {exc});"
        " chuyển sang dynamo. Cần `pip install onnxscript` nếu chưa có."
    )
    torch.onnx.export(cpu_model, example, onnx_path, dynamo=True, **_export_kwargs)
    onnx_exporter = "dynamo"
print(f"Bộ xuất ONNX: {onnx_exporter}")
session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
probe = np.random.default_rng(SEED).normal(size=(2, 3, INPUT_SIZE, INPUT_SIZE)).astype("float32")
with torch.no_grad():
    torch_output = cpu_model(torch.from_numpy(probe)).numpy()
onnx_output = session.run(None, {session.get_inputs()[0].name: probe})[0]
max_abs_diff = float(np.max(np.abs(torch_output - onnx_output)))
if onnx_output.shape != (2, len(PACKAGE_CLASSES)) or max_abs_diff > 1e-4:
    raise RuntimeError(
        f"ONNX verification failed: shape={onnx_output.shape}, diff={max_abs_diff}"
    )
onnx_sha256 = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
manifest = {
    "schema_version": MANIFEST_SCHEMA,
    "task": TASK,
    "model_format": "onnx",
    "class_names": PACKAGE_CLASSES,
    "input": {
        "name": "images",
        "size": [INPUT_SIZE, INPUT_SIZE],
        "color_space": "RGB",
        "resize_mode": "letterbox",
        "letterbox_value": 114,
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
    },
    "output": {"name": "logits"},
    "calibration": {"temperature": temperature},
    "decision_thresholds": {"accept": 0.85, "review": 0.55, "accept_by_class": {}},
    "model": {
        "filename": "best.onnx",
        "version": datetime.now(timezone.utc).strftime("package-mobilenetv3-%Y%m%dT%H%M%SZ"),
        "architecture": "mobilenet_v3_small",
        "sha256": onnx_sha256,
    },
    "training": {
        "seed": SEED,
        "best_epoch": int(max(history, key=lambda row: row["val_macro_recall"])["epoch"]),
        "best_val_macro_recall": best_recall,
        "dataset_schema": dataset_manifest["schema_version"],
        "source_dataset_id": dataset_manifest["source"].get("source_dataset_id"),
    },
    "evaluation": {
        "split_unit": "board",
        "test_group_ids": sorted(split_groups["test"]),
        "test_macro_recall": test_macro_recall,
        "dangerous_confusions": dangerous_confusions,
        "test_confusion_matrix": {
            "class_names": PACKAGE_CLASSES,
            "matrix": matrix.tolist(),
        },
        "onnx_max_abs_diff": max_abs_diff,
    },
    "deployment": {
        "default_enabled": False,
        "promotion": "manual_after_package_roi_gate",
        "classification_gate_passed": (
            test_macro_recall >= 0.85 and dangerous_confusions == 0
        ),
        "real_board_roi_gate_passed": False,
    },
    "created_at": datetime.now(timezone.utc).isoformat(),
}
manifest_path = ARTIFACT_ROOT / "model_manifest.json"
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(ARTIFACT_ROOT / "training_history.json").write_text(
    json.dumps(history, indent=2) + "\n", encoding="utf-8"
)
artifact_zip = shutil.make_archive(
    "/kaggle/working/package_classifier_artifacts", "zip", ARTIFACT_ROOT
)
print("artifact:", artifact_zip)
print("sha256:", onnx_sha256)

# %% [markdown]
# ## 6. Việc bắt buộc sau khi tải artifact về
#
# Notebook chỉ giải quyết gate phân loại. Trên máy local, chạy:
#
# ```bash
# python scripts/evaluate_package_roi_gate.py \
#   package_classifier_artifacts/best.onnx \
#   package_classifier_artifacts/model_manifest.json \
#   --output package_roi_gate.json
# ```
#
# Chỉ khi report `passed: true` mới **tự tay** chép cặp artifact vào
# `models/active/package/`. Script gate không chép và UI không tự bật model.
