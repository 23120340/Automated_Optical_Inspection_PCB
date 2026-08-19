# %% [markdown]
# # AOI PCB — bước 4 v2: detector thân linh kiện **và** chân/pad
#
# Notebook này train lại detector với công thức nhắm thẳng vào chỗ model hiện tại
# đang hỏng: **class `pads` recall 0.000, `pins` recall 0.14–0.21**.
#
# ## Đọc kỹ phần này trước khi chạy — nó quyết định kỳ vọng của bạn
#
# Model hiện tại của bạn (`pcb_component_detector_artifacts`):
#
# | Class | Instance train | Recall (val) | mAP50-95 |
# |---|---:|---:|---:|
# | capacitor | 7775 | 0.654 | 0.375 |
# | resistor | 7133 | 0.446 | 0.307 |
# | ic | 2220 | 0.836 | 0.514 |
# | **pins** | **261** | **0.145** | 0.106 |
# | **pads** | **186** | **0.000** | 0.0019 |
#
# Đây **không phải lỗi kiến trúc**. `pads` có 186 instance so với 7775 của
# capacitor — ít hơn 42 lần — và mỗi pad chỉ vài chục pixel. Đổi sang RT-DETR,
# D-FINE hay YOLOv12 **không sửa được 186 instance**. Ai hứa ngược lại là đang
# bán kiến trúc chứ không nhìn dữ liệu.
#
# Thêm nữa, YOLO26 vốn đã có **STAL** (small-target-aware label assignment):
# nó ép tối thiểu 4 anchor cho vật thể nhỏ hơn 8 px và tách hình học chọn ứng
# viên khỏi hình học hồi quy. Đó đúng là cơ chế cho bài toán vật thể nhỏ. Nên
# khuyến nghị là **giữ YOLO26**, sửa dữ liệu và công thức train.
#
# ## Notebook này làm gì khác notebook cũ
#
# 1. **Tăng độ phân giải** 1280 → 1536. Pad vài chục pixel ở 1280 thì sau khi
#    xuống stride 8 chỉ còn 2–3 pixel đặc trưng.
# 2. **Oversample ảnh chứa class hiếm.** Ultralytics không có sampler theo class,
#    nên notebook nhân bản đường dẫn ảnh chứa `pads`/`pins` trong file train list.
#    Đây là cách duy nhất tác động được vào tần suất mà không sửa loader.
# 3. **`copy_paste` bật.** Với instance segmentation nó dán vật thể hiếm sang ảnh
#    khác; với box thì hiệu quả kém hơn nhưng vẫn dương.
# 4. **Lịch train dài hơn + `close_mosaic` muộn.** Class hiếm cần nhiều epoch mới
#    hội tụ; tắt mosaic sớm quá thì chúng chưa kịp học.
# 5. **Cổng kiểm tra thẳng thắn ở cuối.** Nếu recall `pads` vẫn ~0 sau tất cả,
#    notebook nói rõ: vấn đề là dữ liệu, và đưa ra phương án thực tế.
#
# ## Trước khi Run All
#
# - GPU **T4 x2 trở lên**. Không dùng P100 (PyTorch wheel đã bỏ `sm_60`).
# - Add Input dataset YOLO (xem `CONFIG` bên dưới).
# - Bật Internet lần đầu để tải weight pretrain.
#
# Đầu ra: `/kaggle/working/pcb_detector_v2_artifacts.zip` gồm `best.pt`,
# `best.onnx`, `model_manifest.json`, metrics theo class và confusion matrix.

# %%
CONFIG = {
    "seed": 42,
    # Dataset gốc bạn đang dùng. Add Input rồi sửa đường dẫn nếu khác.
    "dataset_root": "/kaggle/input/datasets/aryanstein/pcb-component-detection-consolidated-dataset/components_data_uncropped",
    "work_dir": "/kaggle/working/pcb_detector_v2",
    "artifact_dir": "/kaggle/working/pcb_detector_v2_artifacts",

    "model": "yolo26s.pt",
    # 1536 thay vì 1280. Đây là đòn bẩy lớn nhất cho vật thể nhỏ, và cũng là
    # thứ tốn VRAM nhất — hạ xuống 1280 nếu OOM.
    "imgsz": 1536,
    "epochs": 150,
    "patience": 40,
    "batch": -1,               # auto-batch theo VRAM
    "close_mosaic": 25,        # muộn hơn mặc định 10

    # Class hiếm cần đẩy tần suất lên. Ảnh chứa các class này được nhân bản
    # trong train list; hệ số là số lần lặp thêm.
    "rare_classes": ["pads", "pins"],
    "rare_oversample": 6,
    # Trần an toàn: nhân bản quá tay làm model overfit đúng vài ảnh đó.
    "max_oversample_fraction": 0.35,

    "copy_paste": 0.30,
    "mosaic": 0.60,
    "scale": 0.45,
    "degrees": 180.0,
    "flipud": 0.5,
    "fliplr": 0.5,
    "hsv_h": 0.010,
    "hsv_s": 0.35,
    "hsv_v": 0.25,
    "erasing": 0.30,

    "conf": 0.001,             # để đánh giá; ngưỡng triển khai đặt ở app
    "iou": 0.7,
    "max_det": 2000,
    # Giữ đúng hợp đồng head của app: one-to-many + NMS ngoài.
    "end2end": False,
    "opset": 18,
}

import os
import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

print("imgsz:", CONFIG["imgsz"], "| epochs:", CONFIG["epochs"])
print("oversample class hiếm:", CONFIG["rare_classes"], "x", CONFIG["rare_oversample"])

# %% [markdown]
# ## 1. Kiểm tra dataset và đo mức mất cân bằng
#
# Cell này đo trước khi train. Nếu class hiếm quá ít, không công thức nào cứu
# được và tốt hơn là biết ngay bây giờ.

# %%
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

root = Path(CONFIG["dataset_root"])
if not root.is_dir():
    # Kaggle gắn dataset ở nhiều nơi tuỳ cách Add Input; dò thay vì bắt đoán.
    candidates = sorted(Path("/kaggle/input").rglob("data.yaml"))
    print("Không thấy dataset_root. Các data.yaml tìm được:")
    for candidate in candidates[:20]:
        print("  ", candidate)
    raise SystemExit(
        "Sửa CONFIG['dataset_root'] cho khớp một trong các đường dẫn trên."
    )

data_yaml = next(iter(sorted(root.rglob("data.yaml"))), None)
if data_yaml is None:
    raise SystemExit(f"Không thấy data.yaml dưới {root}")
spec = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
names = spec.get("names")
if isinstance(names, dict):
    class_names = [names[k] for k in sorted(names)]
else:
    class_names = list(names)
print(f"data.yaml: {data_yaml}")
print(f"{len(class_names)} class: {class_names}")


def split_dir(split: str) -> Path | None:
    for candidate in (root / split / "images", root / "images" / split):
        if candidate.is_dir():
            return candidate
    return None


def label_path(image: Path) -> Path:
    parts = list(image.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def scan(split: str) -> tuple[list[Path], Counter, dict[Path, set[int]]]:
    directory = split_dir(split)
    if directory is None:
        return ([], Counter(), {})
    images = sorted(
        p for p in directory.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )
    counts: Counter = Counter()
    per_image: dict[Path, set[int]] = {}
    for image in images:
        labels = label_path(image)
        present: set[int] = set()
        if labels.is_file():
            for line in labels.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    index = int(float(parts[0]))
                except ValueError:
                    continue
                counts[index] += 1
                present.add(index)
        per_image[image] = present
    return (images, counts, per_image)


train_images, train_counts, train_per_image = scan("train")
val_images, val_counts, _ = scan("valid") or scan("val")
if not val_images:
    val_images, val_counts, _ = scan("val")

print(f"\ntrain: {len(train_images)} ảnh | val: {len(val_images)} ảnh")
print(f"\n{'class':18s} {'train':>8s} {'val':>8s}  {'% train':>8s}")
print("-" * 50)
total = max(1, sum(train_counts.values()))
rare_indices = []
for index, name in enumerate(class_names):
    share = 100.0 * train_counts.get(index, 0) / total
    flag = ""
    if name in CONFIG["rare_classes"]:
        rare_indices.append(index)
        flag = "  <-- hiếm"
    print(f"{name:18s} {train_counts.get(index,0):8d} {val_counts.get(index,0):8d} {share:7.2f}%{flag}")

print(f"\nClass hiếm: {[class_names[i] for i in rare_indices]} -> index {rare_indices}")
for index in rare_indices:
    if val_counts.get(index, 0) == 0:
        print(
            f"!! '{class_names[index]}' KHÔNG có instance nào trong val. "
            "Recall của nó sẽ không đo được — kết quả train sẽ không nói lên điều gì."
        )

# %% [markdown]
# ## 2. Oversample ảnh chứa class hiếm
#
# Ultralytics không có sampler theo class. Cách tác động được vào tần suất mà
# không phải sửa loader là **nhân bản đường dẫn ảnh trong file train list** —
# Ultralytics chấp nhận một file `.txt` liệt kê ảnh thay cho thư mục.
#
# Có trần: nhân bản quá tay thì model học thuộc đúng vài ảnh đó thay vì học
# class. Cell in ra tỉ lệ thực tế để bạn thấy mình đang đẩy đến đâu.

# %%
work = Path(CONFIG["work_dir"])
work.mkdir(parents=True, exist_ok=True)

rare_set = set(rare_indices)
with_rare = [img for img, present in train_per_image.items() if present & rare_set]
without_rare = [img for img, present in train_per_image.items() if not (present & rare_set)]

print(f"{len(with_rare)} ảnh chứa class hiếm / {len(train_images)} ảnh train")

repeat = max(1, int(CONFIG["rare_oversample"]))
listed = list(train_images) + with_rare * (repeat - 1)
fraction = (len(with_rare) * repeat) / max(1, len(listed))

if fraction > CONFIG["max_oversample_fraction"] and with_rare:
    # Hạ hệ số xuống mức trần thay vì cứ thế nhân.
    allowed = CONFIG["max_oversample_fraction"]
    repeat = max(1, int((allowed * len(train_images)) / max(1, len(with_rare) * (1 - allowed))))
    listed = list(train_images) + with_rare * (repeat - 1)
    fraction = (len(with_rare) * repeat) / max(1, len(listed))
    print(f"Hệ số bị hạ xuống x{repeat} để không vượt trần {allowed:.0%}")

train_list = work / "train_oversampled.txt"
train_list.write_text("\n".join(str(p) for p in listed) + "\n", encoding="utf-8")
print(f"train list: {len(listed)} dòng ({len(train_images)} ảnh gốc), "
      f"ảnh chứa class hiếm chiếm {fraction:.1%}")

val_dir = split_dir("valid") or split_dir("val")
training_yaml = work / "data_v2.yaml"
training_yaml.write_text(
    yaml.safe_dump(
        {
            "path": str(root),
            "train": str(train_list),
            "val": str(val_dir),
            "names": {i: n for i, n in enumerate(class_names)},
        },
        sort_keys=False, allow_unicode=True,
    ),
    encoding="utf-8",
)
print(f"data yaml: {training_yaml}")

# %% [markdown]
# ## 3. Train
#
# `close_mosaic` để muộn (25 epoch cuối) vì class hiếm cần nhiều epoch có mosaic
# mới gặp đủ biến thể. `copy_paste` dán vật thể hiếm sang ảnh khác — đòn bẩy
# trực tiếp nhất cho mất cân bằng ở mức instance.

# %%
import subprocess

try:
    from ultralytics import YOLO
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)
    from ultralytics import YOLO

model = YOLO(CONFIG["model"])
results = model.train(
    data=str(training_yaml),
    imgsz=CONFIG["imgsz"],
    epochs=CONFIG["epochs"],
    patience=CONFIG["patience"],
    batch=CONFIG["batch"],
    seed=CONFIG["seed"],
    deterministic=True,
    close_mosaic=CONFIG["close_mosaic"],
    copy_paste=CONFIG["copy_paste"],
    mosaic=CONFIG["mosaic"],
    scale=CONFIG["scale"],
    degrees=CONFIG["degrees"],
    flipud=CONFIG["flipud"],
    fliplr=CONFIG["fliplr"],
    hsv_h=CONFIG["hsv_h"],
    hsv_s=CONFIG["hsv_s"],
    hsv_v=CONFIG["hsv_v"],
    erasing=CONFIG["erasing"],
    max_det=CONFIG["max_det"],
    iou=CONFIG["iou"],
    project=str(work / "runs"),
    name="detector_v2",
    exist_ok=True,
    plots=True,
    val=True,
)
print("Train xong.")

# %% [markdown]
# ## 4. Metric theo class — đây mới là phần cần đọc
#
# mAP tổng thể bị chi phối bởi capacitor và resistor. Con số quyết định notebook
# này có thành công hay không là **recall của `pads` và `pins`**.

# %%
metrics = model.val(
    data=str(training_yaml), imgsz=CONFIG["imgsz"],
    conf=CONFIG["conf"], iou=CONFIG["iou"], max_det=CONFIG["max_det"],
    # Không để mặc định: Ultralytics sẽ ghi vào ./runs của thư mục đang đứng,
    # tức là rải file vào repo khi chạy lại notebook ở máy.
    project=str(work / "runs"), name="val_v2", exist_ok=True,
)

per_class = {}
try:
    precisions, recalls, ap50s, ap5095s = metrics.box.p, metrics.box.r, metrics.box.ap50, metrics.box.ap
    indices = list(metrics.box.ap_class_index)
    for position, class_index in enumerate(indices):
        per_class[class_names[int(class_index)]] = {
            "precision": float(precisions[position]),
            "recall": float(recalls[position]),
            "map50": float(ap50s[position]),
            "map50_95": float(ap5095s[position]),
        }
except Exception as exc:
    print(f"Không đọc được metric theo class: {exc}")

print(f"\n{'class':18s} {'P':>8s} {'R':>8s} {'mAP50':>8s} {'mAP50-95':>9s}")
print("-" * 56)
for name in class_names:
    row = per_class.get(name)
    if row is None:
        print(f"{name:18s} {'—':>8s} {'—':>8s} {'—':>8s} {'—':>9s}   (không có trong val)")
        continue
    flag = "  <-- hiếm" if name in CONFIG["rare_classes"] else ""
    print(f"{name:18s} {row['precision']:8.3f} {row['recall']:8.3f} "
          f"{row['map50']:8.3f} {row['map50_95']:9.3f}{flag}")

print(f"\nmAP50 tổng: {float(metrics.box.map50):.4f}")
print(f"mAP50-95 tổng: {float(metrics.box.map):.4f}")

# %% [markdown]
# ## 5. Cổng kiểm tra thẳng thắn
#
# Baseline cũ: `pads` recall 0.000, `pins` recall 0.145. Nếu sau tất cả những
# thay đổi trên mà con số vẫn không nhúc nhích, kết luận là **dữ liệu**, không
# phải công thức — và notebook nói thẳng ra thay vì để bạn tự đoán.

# %%
BASELINE = {"pads": 0.000, "pins": 0.145}
verdict_lines = []
improved, stuck = [], []

for name, baseline in BASELINE.items():
    row = per_class.get(name)
    if row is None:
        stuck.append(f"{name}: không có trong val, không đo được")
        continue
    recall = row["recall"]
    if recall >= max(0.35, baseline * 2.5):
        improved.append(f"{name}: recall {baseline:.3f} -> {recall:.3f}")
    else:
        stuck.append(f"{name}: recall {baseline:.3f} -> {recall:.3f} (chưa đủ)")

print("=" * 70)
if improved:
    print("CẢI THIỆN:")
    for line in improved:
        print("  ", line)
if stuck:
    print("\nCHƯA ĐẠT:")
    for line in stuck:
        print("  ", line)
    print("""
Nếu tới đây mà pads/pins vẫn kẹt, đừng đổ thêm epoch hay đổi kiến trúc. Số
instance mới là ràng buộc. Ba phương án thực tế, theo thứ tự chi phí:

1. GIỮ ROI SUY RA (rẻ nhất, đã chạy được).
   Pipeline hiện tại đã suy ROI mối hàn từ box thân + topology chân, và
   `aoi_pipeline/inspection/leads.py` đã hợp nhất: có detection chân thật thì
   dùng, không có thì dùng ROI suy ra. Detector yếu ở pads/pins KHÔNG chặn
   bước 5.5 — nó chỉ khiến nhánh "detected" hiếm khi được kích hoạt.

2. GÁN NHÃN BOOTSTRAP (vừa phải, hiệu quả nhất).
   Chạy `scripts/export_solder_dataset.py --overlays` trên board của bạn: nó
   sinh sẵn ROI chân ứng viên. Người chỉ cần SỬA các box đó thay vì vẽ từ đầu —
   nhanh hơn nhiều lần. Vài trăm ảnh board đã đủ vượt xa 186 instance hiện có.

3. THÊM DATASET (không chắc ăn).
   Roboflow 100 'printed-circuit-board' có class Pads/Pins, nhưng nhiều khả
   năng chính là nguồn gốc của dataset bạn đang dùng, nên ghép vào có thể
   không thêm được instance mới nào. Kiểm tra trùng lặp trước khi tốn công.
""")
else:
    print("\nCả pads và pins đều vượt ngưỡng. Đây là detector đáng đưa vào bước 5.5.")
print("=" * 70)

# %% [markdown]
# ## 6. Xuất artifact
#
# Giữ đúng hợp đồng app đang dùng: head one-to-many, NMS ngoài (`end2end=False`).

# %%
import hashlib
from datetime import datetime, timezone

artifacts = Path(CONFIG["artifact_dir"])
artifacts.mkdir(parents=True, exist_ok=True)

run_dir = Path(model.trainer.save_dir) if hasattr(model, "trainer") else work / "runs" / "detector_v2"
for name in ("best.pt", "last.pt"):
    source = run_dir / "weights" / name
    if source.is_file():
        shutil.copy2(source, artifacts / name)
for pattern in ("*.png", "*.jpg", "results.csv", "args.yaml"):
    for item in run_dir.glob(pattern):
        shutil.copy2(item, artifacts / item.name)

best = YOLO(str(artifacts / "best.pt"))
onnx_path = best.export(
    format="onnx", imgsz=CONFIG["imgsz"], opset=CONFIG["opset"],
    dynamic=False, simplify=True, nms=False,
)
onnx_file = Path(onnx_path)
target = artifacts / "best.onnx"
if onnx_file.resolve() != target.resolve():
    shutil.copy2(onnx_file, target)

# Trọng số phải nằm trong chính file .onnx: app chỉ copy .onnx + manifest, nên
# một export bị tách ra .data sẽ chạy ở đây và hỏng ở mọi nơi khác.
try:
    import onnx
    graph = onnx.load(str(target))
    onnx.save_model(graph, str(target), save_as_external_data=False)
    for orphan in target.parent.glob(f"{target.name}*.data"):
        orphan.unlink()
    for orphan in target.parent.glob(f"{target.stem}*.data"):
        orphan.unlink()
    print(f"ONNX tự chứa ({target.stat().st_size / 1e6:.1f} MB)")
except ImportError:
    print("Không có onnx; không kiểm tra được export có tự chứa hay không.")


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest = {
    "schema_version": "pcb-component-detector/2.0",
    "task": "component_and_lead_detection",
    "model_format": "onnx",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "class_names": class_names,
    "lead_classes": [n for n in class_names if n.lower() in {"pads", "pins"}],
    "input": {"size": [CONFIG["imgsz"], CONFIG["imgsz"]], "color_space": "RGB"},
    "head": {"end2end": CONFIG["end2end"], "nms": "external", "max_det": CONFIG["max_det"]},
    "model": {
        "version": f"detector-v2-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}",
        "architecture": CONFIG["model"],
        "sha256": sha256_of(target),
    },
    "metrics": {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": per_class,
    },
    "training": {
        "imgsz": CONFIG["imgsz"],
        "epochs": CONFIG["epochs"],
        "rare_classes": CONFIG["rare_classes"],
        "rare_oversample_applied": repeat,
        "rare_image_fraction": fraction,
        "train_instances": {class_names[i]: c for i, c in sorted(train_counts.items())},
        "seed": CONFIG["seed"],
    },
}
(artifacts / "model_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
)

shutil.make_archive("/kaggle/working/pcb_detector_v2_artifacts", "zip", artifacts)
print("\nTải: /kaggle/working/pcb_detector_v2_artifacts.zip")
for item in sorted(artifacts.iterdir()):
    print(f"  {item.name}")

# %% [markdown]
# ## 7. Đưa vào app
#
# 1. Giải nén, đặt `best.onnx` vào `models/detector/`.
# 2. Nạp ở sidebar bước 4.
# 3. Bước 5.5 **tự động** dùng detection `pads`/`pins` nếu model tìm được, và
#    quay về ROI suy ra ở chỗ không tìm được — không cần bật gì thêm. Cấu hình ở
#    `LeadFusionConfig` nếu muốn siết `min_lead_confidence`.
#
# Kiểm tra nhanh sau khi nạp: chạy một board rồi xem cảnh báo
# `"dùng N ROI từ detection chân/pad thật và M ROI suy ra"`. N > 0 nghĩa là
# detector mới thực sự đang đóng góp.
