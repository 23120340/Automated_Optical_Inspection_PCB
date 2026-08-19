# %% [markdown]
# # AOI PCB — bước 6.1 v2: phân loại family linh kiện, train kỹ
#
# Train lại classifier bước 6.1 với backbone mới và công thức mạnh hơn hẳn
# notebook v1. Xuất đúng `best.onnx` + `model_manifest.json` mà app đang chờ.
#
# ## Đổi backbone: EfficientNetV2-S, và vì sao
#
# | Backbone | Lý do chọn / bỏ |
# |---|---|
# | EfficientNet-B0 (v1 đang dùng) | Ổn nhưng train chậm hội tụ, và progressive-resize của V2 hợp dữ liệu nhỏ hơn |
# | **EfficientNetV2-S** ✅ | Fine-tune tốt trên tập nhỏ/vừa, cân bằng accuracy–latency, ONNX export sạch |
# | ConvNeXt-V2 Tiny | Accuracy tốt nhưng nặng hơn đáng kể trên CPU ARM |
# | MobileNetV4-Conv-S | **Chậm hơn MobileNetV3-Small ~59%** một luồng trên CPU — không đáng đổi |
# | MobileNetV3-Small | Nhanh nhất CPU; để làm phương án nếu Raspberry Pi không kham nổi V2-S |
#
# Notebook cho chọn backbone ở `CONFIG["model_name"]`. Mặc định EfficientNetV2-S;
# nếu latency trên Pi không đạt thì đổi sang `mobilenet_v3_small` và train lại —
# mọi thứ khác giữ nguyên.
#
# ## Công thức train: những thứ thật sự dịch kim
#
# 1. **Chia theo ảnh cha, không theo crop.** Nhiều crop cắt ra từ cùng một ảnh
#    board; chia theo crop là rò rỉ và cho điểm số ảo.
# 2. **Warmup + freeze backbone vài epoch**, rồi mở khoá với **layer-wise LR
#    decay**. Fine-tune toàn bộ ngay từ epoch 1 sẽ phá weight pretrain.
# 3. **RandAugment + Mixup/CutMix + label smoothing.** Với fine-grained ít dữ
#    liệu đây là nhóm augmentation có tác động lớn nhất.
# 4. **EMA của weight.** Gần như luôn cho +0.5–1.5% mà không tốn gì.
# 5. **Class-balanced sampler.** Dataset linh kiện lệch cực mạnh (capacitor và
#    resistor chiếm phần lớn), không cân thì các class hiếm không bao giờ học được.
# 6. **Cosine schedule + patience dài.**
# 7. **Calibration tách riêng.** Ngưỡng accept/review phải đo trên tập chưa dùng
#    để chọn model, nếu không ngưỡng sẽ lạc quan.
#
# ## Trước khi Run All
#
# - GPU **T4 x2 trở lên**.
# - Add Input dataset (xem `CONFIG`).
# - Bật Internet lần đầu để tải weight pretrain.
#
# Đầu ra: `/kaggle/working/pcb_classifier_v2_artifacts.zip`.

# %%
CONFIG = {
    "seed": 42,
    "dataset_root": "/kaggle/input/datasets/aryanstein/pcb-component-detection-consolidated-dataset/components_data_uncropped",
    "work_dir": "/kaggle/working/pcb_classifier_v2",
    "artifact_dir": "/kaggle/working/pcb_classifier_v2_artifacts",

    # efficientnet_v2_s | mobilenet_v3_small | convnext_tiny | efficientnet_b0
    "model_name": "efficientnet_v2_s",
    "input_size": 224,
    "letterbox_value": 114,
    # Phải khớp công thức crop của app: pad = 0.15 * max(w,h), không ép vuông.
    "crop_padding_ratio": 0.15,

    "batch_size": 64,
    "epochs": 60,
    "freeze_epochs": 3,
    "patience": 12,
    "head_lr": 1e-3,
    "backbone_lr": 1e-4,
    "layer_decay": 0.80,
    "weight_decay": 0.02,
    "label_smoothing": 0.10,
    "mixup_alpha": 0.2,
    "cutmix_alpha": 1.0,
    "mix_probability": 0.5,
    "ema_decay": 0.999,
    "balanced_sampler": True,

    "val_fraction": 0.20,      # tách từ train, theo ảnh cha
    "calibration_fraction": 0.30,  # tách từ val gốc của dataset
    "min_per_class": 40,
    "num_workers": 2,
    "opset": 18,
}

import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

print("backbone:", CONFIG["model_name"], "| input:", CONFIG["input_size"])

# %% [markdown]
# ## 1. Cắt crop từ box YOLO
#
# Dataset là ảnh board + box YOLO. Crop được cắt theo **đúng công thức app dùng**
# (`pad = 0.15 * max(w,h)`, cắt theo biên, không ép vuông) — nếu train trên công
# thức khác thì model gặp phân bố đầu vào khác lúc chạy thật.

# %%
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

SEED = CONFIG["seed"]
random.seed(SEED)
np.random.seed(SEED)

root = Path(CONFIG["dataset_root"])
if not root.is_dir():
    found = sorted(Path("/kaggle/input").rglob("data.yaml"))
    print("Không thấy dataset_root. data.yaml tìm được:")
    for item in found[:20]:
        print("  ", item)
    raise SystemExit("Sửa CONFIG['dataset_root'].")

data_yaml = next(iter(sorted(root.rglob("data.yaml"))), None)
spec = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
raw_names = spec.get("names")
class_names_all = (
    [raw_names[k] for k in sorted(raw_names)] if isinstance(raw_names, dict) else list(raw_names)
)
print(f"{len(class_names_all)} class nguồn: {class_names_all}")

# `pads`/`pins` là vùng hàn, không phải family linh kiện. Giữ chúng làm class
# reject/negative thay vì trộn vào family — nhầm một pad thành resistor tệ hơn
# là biết rằng đó không phải linh kiện.
NON_COMPONENT = {"pads", "pins"}

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
work = Path(CONFIG["work_dir"])
crops_dir = work / "crops"
if crops_dir.exists():
    shutil.rmtree(crops_dir)
crops_dir.mkdir(parents=True, exist_ok=True)


def label_path(image: Path) -> Path:
    parts = list(image.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def split_dir(split: str):
    for candidate in (root / split / "images", root / "images" / split):
        if candidate.is_dir():
            return candidate
    return None


def cut_crops(split: str) -> list[dict]:
    directory = split_dir(split)
    if directory is None:
        return []
    records = []
    images = sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXT)
    for image_path in images:
        labels = label_path(image_path)
        if not labels.is_file():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        for line_number, line in enumerate(
            labels.read_text(encoding="utf-8", errors="replace").splitlines()
        ):
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                index = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            name = class_names_all[index] if index < len(class_names_all) else str(index)
            x1 = (cx - bw / 2) * width
            y1 = (cy - bh / 2) * height
            x2 = (cx + bw / 2) * width
            y2 = (cy + bh / 2) * height
            # Bỏ box tràn biên: crop của chúng bị cắt cụt và dạy model học viền.
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                continue
            pad = CONFIG["crop_padding_ratio"] * max(x2 - x1, y2 - y1)
            left = max(0, math.floor(x1 - pad))
            top = max(0, math.floor(y1 - pad))
            right = min(width, math.ceil(x2 + pad))
            bottom = min(height, math.ceil(y2 + pad))
            if right - left < 8 or bottom - top < 8:
                continue
            crop = image[top:bottom, left:right]
            out = crops_dir / split / name
            out.mkdir(parents=True, exist_ok=True)
            filename = f"{image_path.stem}__{line_number:04d}.jpg"
            cv2.imwrite(str(out / filename), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            records.append({
                "path": out / filename,
                "label": name,
                "parent": image_path.stem,
                "split": split,
            })
    return records


train_records = cut_crops("train")
val_records = cut_crops("valid") or cut_crops("val")
test_records = cut_crops("test")
print(f"crop: train {len(train_records)} | val {len(val_records)} | test {len(test_records)}")
print("train theo class:", dict(sorted(Counter(r['label'] for r in train_records).items())))

# %% [markdown]
# ## 2. Chọn class và chia tập theo ảnh cha
#
# Chia theo `parent` (ảnh board gốc). Nhiều crop từ cùng một board dùng chung ánh
# sáng và tiêu cự; chia theo crop đặt các mẫu gần trùng ở cả hai phía.

# %%
counts = Counter(r["label"] for r in train_records)
CLASS_NAMES = sorted(
    name for name, count in counts.items()
    if count >= CONFIG["min_per_class"] and name not in NON_COMPONENT
)
dropped = sorted(set(counts) - set(CLASS_NAMES))
print(f"Train {len(CLASS_NAMES)} class: {CLASS_NAMES}")
if dropped:
    print(f"Loại: {dropped}")
    print("  (dưới min_per_class, hoặc là pads/pins — vùng hàn chứ không phải family)")
if len(CLASS_NAMES) < 2:
    raise SystemExit("Cần ít nhất 2 class đủ dữ liệu.")

train_records = [r for r in train_records if r["label"] in CLASS_NAMES]
val_records = [r for r in val_records if r["label"] in CLASS_NAMES]
test_records = [r for r in test_records if r["label"] in CLASS_NAMES]

# Val gốc của dataset được tách đôi theo ảnh cha: một nửa chọn model, một nửa
# hiệu chỉnh ngưỡng. Dùng chung một tập cho cả hai làm ngưỡng lạc quan.
parents = sorted({r["parent"] for r in val_records})
rng = np.random.default_rng(SEED)
order = rng.permutation(len(parents))
cut = max(1, int(round(len(parents) * CONFIG["calibration_fraction"])))
calibration_parents = {parents[int(i)] for i in order[:cut]}
calibration_records = [r for r in val_records if r["parent"] in calibration_parents]
model_val_records = [r for r in val_records if r["parent"] not in calibration_parents]

print(f"\nval gốc {len(val_records)} crop / {len(parents)} ảnh cha")
print(f"  -> model-val {len(model_val_records)} | calibration {len(calibration_records)}")
print(f"test (khoá) {len(test_records)}")
if not model_val_records or not calibration_records:
    raise SystemExit("Chia val làm rỗng một phía; giảm calibration_fraction.")

# %% [markdown]
# ## 3. Dataset và augmentation
#
# Letterbox đúng như app, RandAugment cho biến thể, Mixup/CutMix áp ở mức batch.

# %%
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
print("device:", device)

SIZE = (CONFIG["input_size"], CONFIG["input_size"])
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
INDEX_OF = {name: i for i, name in enumerate(CLASS_NAMES)}


def letterbox(image, size, value):
    target_w, target_h = size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (nw, nh), interpolation=interpolation)
    canvas = np.full((target_h, target_w, 3), value, np.uint8)
    x, y = (target_w - nw) // 2, (target_h - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


train_augment = v2.Compose([
    v2.RandomHorizontalFlip(0.5),
    v2.RandomVerticalFlip(0.5),
    v2.RandAugment(num_ops=2, magnitude=7),
    v2.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])


class CropDataset(Dataset):
    def __init__(self, records, training):
        self.records = records
        self.training = training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = cv2.imread(str(record["path"]))
        if image is None:
            image = np.full((32, 32, 3), CONFIG["letterbox_value"], np.uint8)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxed = letterbox(rgb, SIZE, CONFIG["letterbox_value"])
        tensor = torch.from_numpy(np.ascontiguousarray(boxed.transpose(2, 0, 1)))
        if self.training:
            tensor = train_augment(tensor)
        tensor = tensor.float() / 255.0
        tensor = (tensor - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)
        return tensor, INDEX_OF[record["label"]]


def make_loader(records, training):
    dataset = CropDataset(records, training)
    if training and CONFIG["balanced_sampler"]:
        counts = Counter(r["label"] for r in records)
        # Nghịch đảo tần suất: class hiếm được lấy mẫu thường xuyên hơn, thay vì
        # bị capacitor và resistor nhấn chìm.
        weights = [1.0 / counts[r["label"]] for r in records]
        sampler = WeightedRandomSampler(weights, num_samples=len(records), replacement=True)
        return DataLoader(dataset, batch_size=CONFIG["batch_size"], sampler=sampler,
                          num_workers=CONFIG["num_workers"], drop_last=True)
    return DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=training,
                      num_workers=CONFIG["num_workers"])


train_loader = make_loader(train_records, True)
val_loader = make_loader(model_val_records, False)
calibration_loader = make_loader(calibration_records, False)
test_loader = make_loader(test_records, False) if test_records else None

# %% [markdown]
# ## 4. Model, layer-wise LR decay, EMA

# %%
def build_model(name, num_classes):
    if name == "efficientnet_v2_s":
        model = torchvision.models.efficientnet_v2_s(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        head_names = ["classifier"]
    elif name == "efficientnet_b0":
        model = torchvision.models.efficientnet_b0(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        head_names = ["classifier"]
    elif name == "mobilenet_v3_small":
        model = torchvision.models.mobilenet_v3_small(weights="DEFAULT")
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        head_names = ["classifier"]
    elif name == "convnext_tiny":
        model = torchvision.models.convnext_tiny(weights="DEFAULT")
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        head_names = ["classifier"]
    else:
        raise SystemExit(f"Backbone chưa hỗ trợ: {name}")
    return model, head_names


model, head_names = build_model(CONFIG["model_name"], len(CLASS_NAMES))
model = model.to(device)

def stage_blocks(network, head_names):
    """Các stage của backbone, từ gần input ra gần head.

    `model.children()` của EfficientNet/ConvNeXt chỉ trả về `features`,
    `avgpool`, `classifier` — dùng thẳng nó thì "layer-wise decay" thoái hoá
    thành hai nhóm và không phân tầng gì cả. Phải đi xuống một cấp vào
    `features` mới có các stage thật.
    """

    stages = []
    for name, module in network.named_children():
        if any(head in name for head in head_names):
            continue
        children = list(module.children())
        # `features` là Sequential nhiều stage; avgpool thì không có gì bên trong.
        if len(children) > 1:
            stages.extend(children)
        elif any(p.requires_grad for p in module.parameters()):
            stages.append(module)
    return stages


head_parameters = [
    p for name, p in model.named_parameters()
    if any(head in name for head in head_names)
]
stages = stage_blocks(model, head_names)
depth = len(stages)

groups = [{"params": head_parameters, "lr": CONFIG["head_lr"], "name": "head"}]
for position, block in enumerate(stages):
    parameters = [p for p in block.parameters() if p.requires_grad]
    if not parameters:
        continue
    # Tầng gần input học chậm nhất: chúng mang đặc trưng tổng quát, phá chúng
    # bằng LR lớn là vứt bỏ pretrain.
    scale = CONFIG["layer_decay"] ** (depth - 1 - position)
    groups.append({
        "params": parameters,
        "lr": CONFIG["backbone_lr"] * scale,
        "name": f"stage{position}",
    })

if len(groups) < 3:
    print(
        f"!! Chỉ dựng được {len(groups)} param group cho '{CONFIG['model_name']}'. "
        "Layer-wise decay đang không phân tầng — kiểm tra stage_blocks() nếu bạn "
        "vừa đổi backbone."
    )

optimizer = torch.optim.AdamW(groups, weight_decay=CONFIG["weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])

ema = torch.optim.swa_utils.AveragedModel(
    model, avg_fn=lambda avg, new, _: CONFIG["ema_decay"] * avg + (1 - CONFIG["ema_decay"]) * new
)

mixup = v2.MixUp(alpha=CONFIG["mixup_alpha"], num_classes=len(CLASS_NAMES))
cutmix = v2.CutMix(alpha=CONFIG["cutmix_alpha"], num_classes=len(CLASS_NAMES))

lrs = [f"{g['name']}={g['lr']:.2e}" for g in groups]
print(f"{len(groups)} param group | EMA decay {CONFIG['ema_decay']}")
print("  LR:", ", ".join(lrs))

# %%
def evaluate(network, loader):
    network.eval()
    correct = total = 0
    per_class_correct, per_class_total = Counter(), Counter()
    with torch.no_grad():
        for images, targets in loader:
            outputs = network(images.to(device))
            predicted = outputs.argmax(1).cpu()
            correct += int((predicted == targets).sum())
            total += targets.numel()
            for truth, prediction in zip(targets.tolist(), predicted.tolist()):
                per_class_total[truth] += 1
                per_class_correct[truth] += int(truth == prediction)
    accuracy = correct / max(1, total)
    # Macro recall: trung bình theo class, nên một class hiếm bị bỏ rơi không bị
    # accuracy tổng che mất.
    recalls = [per_class_correct[i] / per_class_total[i]
               for i in per_class_total if per_class_total[i]]
    return accuracy, (sum(recalls) / len(recalls) if recalls else 0.0)


best_score, best_state, history = -1.0, None, []
for epoch in range(1, CONFIG["epochs"] + 1):
    if epoch <= CONFIG["freeze_epochs"]:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = any(head in name for head in head_names)
    elif epoch == CONFIG["freeze_epochs"] + 1:
        for parameter in model.parameters():
            parameter.requires_grad = True
        print(f"epoch {epoch}: mở khoá toàn bộ backbone")

    model.train()
    running = 0.0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        if np.random.rand() < CONFIG["mix_probability"]:
            images, targets = (mixup if np.random.rand() < 0.5 else cutmix)(images, targets)
        optimizer.zero_grad()
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
        ema.update_parameters(model)
        running += float(loss.detach()) * images.size(0)
    scheduler.step()

    accuracy, macro_recall = evaluate(model, val_loader)
    ema_accuracy, ema_macro = evaluate(ema.module, val_loader)
    use_ema = ema_macro > macro_recall
    score = max(macro_recall, ema_macro)
    history.append({"epoch": epoch, "loss": running / max(1, len(train_records)),
                    "acc": accuracy, "macro_recall": macro_recall,
                    "ema_acc": ema_accuracy, "ema_macro": ema_macro})
    if score > best_score:
        best_score = score
        source = ema.module if use_ema else model
        best_state = {k: v.detach().cpu().clone() for k, v in source.state_dict().items()}
    print(f"epoch {epoch:3d}  loss {history[-1]['loss']:.4f}  "
          f"acc {accuracy:.4f} macroR {macro_recall:.4f}  |  "
          f"ema acc {ema_accuracy:.4f} macroR {ema_macro:.4f}{'  <- EMA' if use_ema else ''}")

    if len(history) > CONFIG["patience"]:
        window = [h for h in history[-CONFIG["patience"]:]]
        if max(max(h["macro_recall"], h["ema_macro"]) for h in window) < best_score:
            print(f"Dừng sớm ở epoch {epoch}: {CONFIG['patience']} epoch không cải thiện")
            break

if best_state is not None:
    model.load_state_dict(best_state)
model.eval()
print(f"\nmacro recall tốt nhất: {best_score:.4f}")

# %% [markdown]
# ## 5. Hiệu chỉnh nhiệt độ và chọn ngưỡng
#
# Đo trên tập calibration — tập chưa dùng để chọn model. Ngưỡng đo trên tập đã
# dùng chọn model luôn lạc quan.

# %%
model_cpu = model.to("cpu").eval()


def collect(loader):
    logits_all, truths = [], []
    with torch.no_grad():
        for images, targets in loader:
            logits_all.append(model_cpu(images).numpy())
            truths.extend(targets.tolist())
    return (np.concatenate(logits_all) if logits_all else np.zeros((0, len(CLASS_NAMES))),
            np.asarray(truths))


cal_logits, cal_truths = collect(calibration_loader)

TEMPERATURE_GRID = np.arange(0.5, 3.05, 0.05)
best_temperature, best_nll = 1.0, math.inf
for temperature in TEMPERATURE_GRID:
    scaled = cal_logits / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    probabilities = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
    nll = -np.mean(np.log(np.clip(probabilities[np.arange(len(cal_truths)), cal_truths], 1e-12, 1)))
    if nll < best_nll:
        best_nll, best_temperature = nll, float(temperature)
print(f"Temperature hiệu chỉnh: {best_temperature:.2f} (NLL {best_nll:.4f})")

# Chạm biên nghĩa là tối ưu nằm NGOÀI dải quét, và giá trị lấy được chỉ là mép
# dải chứ không phải tối ưu. Im lặng ở đây sẽ ghi một temperature sai vào
# manifest và mọi xác suất sau đó đều lệch.
if abs(best_temperature - TEMPERATURE_GRID[0]) < 1e-9:
    print(
        f"!! Temperature chạm BIÊN DƯỚI ({TEMPERATURE_GRID[0]:.2f}). Model đang "
        "under-confident hơn cả mức dải quét cho phép — thường là dấu hiệu train "
        "chưa đủ hoặc tập calibration quá nhỏ. Đừng tin con số này; train đủ "
        "epoch rồi hiệu chỉnh lại."
    )
elif abs(best_temperature - TEMPERATURE_GRID[-1]) < 1e-9:
    print(
        f"!! Temperature chạm BIÊN TRÊN ({TEMPERATURE_GRID[-1]:.2f}). Model "
        "over-confident nặng — nới dải quét hoặc xem lại overfitting."
    )

scaled = cal_logits / best_temperature
scaled = scaled - scaled.max(axis=1, keepdims=True)
cal_probabilities = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
predicted = cal_probabilities.argmax(1)
confidence = cal_probabilities.max(1)

print(f"\n{'accept':>8s} {'phủ':>8s} {'đúng khi accept':>16s} {'review %':>10s}")
sweep = []
for threshold in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95):
    accepted = confidence >= threshold
    coverage = float(accepted.mean())
    precision = float((predicted[accepted] == cal_truths[accepted]).mean()) if accepted.any() else float("nan")
    sweep.append({"accept": threshold, "coverage": coverage, "precision": precision})
    print(f"{threshold:8.2f} {coverage:8.1%} {precision:16.3%} {1 - coverage:10.1%}")

print("\nChọn ngưỡng: accept cao thì ít sai nhưng nhiều crop rơi vào review.")

# %% [markdown]
# ## 6. Kết quả trên test (tập khoá, chưa từng chạm)

# %%
if test_loader is not None:
    accuracy, macro_recall = evaluate(model_cpu, test_loader)
    print(f"TEST accuracy {accuracy:.4f} | macro recall {macro_recall:.4f}")
    test_logits, test_truths = collect(test_loader)
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), int)
    for truth, prediction in zip(test_truths, test_logits.argmax(1)):
        matrix[truth, prediction] += 1
    print(f"\n{'class':16s} {'n':>6s} {'recall':>8s}")
    for index, name in enumerate(CLASS_NAMES):
        total = matrix[index].sum()
        recall = matrix[index, index] / total if total else float("nan")
        print(f"{name:16s} {total:6d} {recall:8.3f}")
else:
    print("Không có split test trong dataset.")

# %% [markdown]
# ## 7. Xuất artifact

# %%
import hashlib
from datetime import datetime, timezone

artifacts = Path(CONFIG["artifact_dir"])
artifacts.mkdir(parents=True, exist_ok=True)
onnx_path = artifacts / "best.onnx"

dummy = torch.zeros(1, 3, CONFIG["input_size"], CONFIG["input_size"])
torch.onnx.export(
    model_cpu, dummy, str(onnx_path),
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=CONFIG["opset"],
)

try:
    import onnx
    graph = onnx.load(str(onnx_path))
    onnx.save_model(graph, str(onnx_path), save_as_external_data=False)
    for orphan in artifacts.glob("best.onnx*.data"):
        orphan.unlink()
    print(f"ONNX tự chứa ({onnx_path.stat().st_size / 1e6:.1f} MB)")
except ImportError:
    print("Không có onnx; không kiểm tra được export tự chứa.")

try:
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"input": dummy.numpy()})[0]
    with torch.no_grad():
        torch_out = model_cpu(dummy).numpy()
    difference = float(np.max(np.abs(onnx_out - torch_out)))
    assert difference < 1e-3, f"ONNX lệch torch {difference}"
    print(f"ONNX khớp torch (lệch {difference:.2e})")
except ImportError:
    print("Không có onnxruntime; bỏ qua đối chiếu.")


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


chosen = next((s for s in sweep if s["accept"] == 0.85), sweep[-1])
manifest = {
    "schema_version": "pcb-component-classifier/1.0",
    "task": "component_family_classification",
    "model_format": "onnx",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "class_names": CLASS_NAMES,
    "input": {
        "name": "input",
        "size": [CONFIG["input_size"], CONFIG["input_size"]],
        "color_space": "RGB",
        "resize_mode": "letterbox",
        "letterbox_value": CONFIG["letterbox_value"],
        "normalization": {"mean": MEAN.tolist(), "std": STD.tolist()},
    },
    "output": {"name": "logits", "type": "raw_logits"},
    "calibration": {
        "temperature": best_temperature,
        "hit_sweep_boundary": bool(
            abs(best_temperature - TEMPERATURE_GRID[0]) < 1e-9
            or abs(best_temperature - TEMPERATURE_GRID[-1]) < 1e-9
        ),
    },
    "decision_thresholds": {"accept": 0.85, "review": 0.50, "accept_by_class": {}},
    "model": {
        "version": f"classifier-v2-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}",
        "architecture": CONFIG["model_name"],
        "sha256": sha256_of(onnx_path),
    },
    "training": {
        "crop_padding_ratio": CONFIG["crop_padding_ratio"],
        "crops_train": len(train_records),
        "crops_model_val": len(model_val_records),
        "crops_calibration": len(calibration_records),
        "crops_test": len(test_records),
        "classes_dropped": dropped,
        "best_macro_recall": best_score,
        "threshold_sweep": sweep,
        "epochs_run": len(history),
        "seed": SEED,
    },
}
(artifacts / "model_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
)

shutil.make_archive("/kaggle/working/pcb_classifier_v2_artifacts", "zip", artifacts)
print("\nTải: /kaggle/working/pcb_classifier_v2_artifacts.zip")

# %% [markdown]
# ## 8. Đưa vào app
#
# Nạp `best.onnx` + `model_manifest.json` ở sidebar **Model phân loại 6.1**.
# Schema giữ nguyên `pcb-component-classifier/1.0` nên không phải sửa gì trong app.
#
# **Một điều phải kiểm tra:** app cắt crop theo `pad = 0.15 * max(w,h)`, không ép
# vuông. Notebook này cắt đúng công thức đó. Nếu bạn sửa `CropConfig` trong app
# thì phải train lại — lệch công thức crop là lệch phân bố đầu vào, và nó biểu
# hiện thành accuracy tụt mà không rõ lý do.
