# %% [markdown]
# # AOI PCB — bước 6.2: **detector** lỗi mối hàn (không phải classifier)
#
# Hai notebook 6.2 còn lại (`pcb_solder_defect_kaggle`, `..._v2_kaggle`) train
# **classifier**: đưa vào một ROI đã cắt, trả về tốt/lỗi. Notebook này train
# **detector**: đưa vào ảnh board, trả về *lỗi nằm ở đâu*. Hai bài toán khác
# nhau, và cho tới giờ dự án chưa có notebook nào cho bài thứ hai.
#
# ## Dữ liệu, và vì sao lần này khác
#
# Nguồn: `universe.roboflow.com/pcb-vndkd/solder-dbcbh` v3, **CC BY 4.0**.
# Đã tải sẵn về `datasets/public/roboflow_solder_leadjoints/`.
#
# Khảo sát trước đây (`docs/surveys/dataset_lead_detection.md`) kết luận *"không nguồn
# công khai nào có đủ cả đúng tỉ lệ lẫn box"*. Bộ này lật kết luận đó. Đo được:
#
# | | Bộ này | Board dự án | SolDef_AI |
# |---|---|---|---|
# | Box chiếm bao nhiêu khung | **2,3 %** | ~2,2 % | ~40 % (macro) |
# | Ảnh màu | có (chroma 55,9) | có | có |
# | Có box | **11.033** | — | có |
#
# Ảnh màu, IC gull-wing trên board xanh, box đặt trên **từng mối hàn chân riêng
# lẻ** — đúng độ hạt bước 6.2 cần.
#
# ## Vì sao KHÔNG trộn SolDef_AI để bù lớp `good`
#
# Ý tưởng tự nhiên là lấy `good` từ SolDef_AI vì bộ này chỉ khoanh lỗi. **Đừng.**
#
# SolDef_AI chụp 1–3 µm/px, bộ này ~2,3 % khung. Nếu mọi mẫu `good` đến từ một
# nguồn và mọi mẫu `defect` đến từ nguồn kia thì **nhãn tương quan 100 % với
# chữ ký camera** — độ nét, nhiễu sensor, tông màu. Model tách hai lớp bằng
# camera chứ không bằng hình thái mối hàn, val đẹp và production vô dụng, vì lúc
# chạy thật mọi crop đều từ cùng một camera nên shortcut biến mất.
#
# Không cách chia tập nào cứu được: lỗi nằm ở chỗ nhãn dính vào nguồn, không
# phải ở chỗ chia.
#
# **Với detector thì vấn đề này không tồn tại.** Box = lỗi, mọi thứ còn lại là
# background — lấy từ chính ảnh đó. Mỗi ảnh có ~28 chân IC mà trung vị chỉ 3 box,
# nên ~25 mối hàn lành mỗi ảnh đã là negative sample, cùng camera cùng tỉ lệ.
#
# ## Trước khi Run All
#
# - Add Input bộ Roboflow, rồi trỏ `CONFIG["data_root"]` **thẳng vào thư mục có
#   `data.yaml`**, không trỏ vào thư mục cha.
#
#   > Lần train đầu đã hỏng đúng ở đây: Kaggle input chứa cả
#   > `roboflow_solder_extra` lẫn `roboflow_solder_leadjoints`, notebook trỏ vào
#   > thư mục cha, và `find_dataset()` cũ lặng lẽ lấy cái đứng đầu alphabet —
#   > tức bộ 109 cảnh thay vì bộ 1.257 cảnh. Metrics ra `mAP50 0.95` trên **11
#   > cảnh** test và manifest vẫn khai nhầm nguồn vì nguồn bị hardcode. Cả hai
#   > đã sửa: nhiều `data.yaml` giờ là lỗi, và nguồn đọc từ `data.yaml`.
# - GPU T4 là đủ. `imgsz=1280` nên một epoch chậm hơn mặc định khoảng 4 lần.
#
# Đầu ra: `/kaggle/working/pcb_solder_detector_artifacts.zip` gồm `best.pt`,
# `best.onnx`, `model_manifest.json` và metrics theo lớp.

# %%
CONFIG = {
    "seed": 42,

    # --- dữ liệu ---------------------------------------------------------
    "data_root": "/kaggle/input/roboflow-solder-leadjoints",
    "work_dir": "/kaggle/working/pcb_solder_detector",
    "artifact_dir": "/kaggle/working/pcb_solder_detector_artifacts",

    # --- model -----------------------------------------------------------
    # yolo11s chứ không phải n: box trung vị 17 px trên khung 640 là vật thể
    # nhỏ, mà nhánh P3 của model n mỏng nên hụt. Cũng không phải m/l: 1.257
    # cảnh gốc thì model to sẽ học thuộc cảnh.
    "model": "yolo11s.pt",

    # imgsz=1280, chọn theo số đo chứ không theo mặc định. Đếm box có cạnh ngắn
    # dưới 8 px — tức dưới bước nhảy nhỏ nhất của đầu dò:
    #
    #     imgsz    box < 8 px
    #      640     2.259  (20,5 %)
    #      960       455  ( 4,1 %)
    #     1280        31  ( 0,3 %)
    #
    # Ảnh gốc là 640×640 nên phóng lên 1280 không thêm chi tiết mới; nó chỉ đưa
    # vật thể nhỏ lên trên ngưỡng mà đầu dò nhìn thấy được. Một phần năm số nhãn
    # nằm dưới ngưỡng đó ở 640 là quá nhiều để bỏ qua.
    "imgsz": 1280,
    "epochs": 100,
    "batch": 8,          # 1280 tốn VRAM gấp 4 lần 640
    "patience": 25,

    # --- cổng chặn -------------------------------------------------------
    # Split của Roboflow đã kiểm 2026-08-25: 897/243/117 cảnh, **0 cảnh trùng**
    # giữa ba tập. Cổng này chạy lại phép kiểm đó, vì bản v4 nào đó về sau có thể
    # không còn sạch, và một cảnh nằm cả train lẫn val thì mAP val là số ảo.
    "require_clean_scene_split": True,
    # Dưới ngưỡng này thì model tệ hơn thứ nó thay thế, đừng xuất.
    "min_map50": 0.35,
    # mAP đo trên ít cảnh gốc thì không phải phép đo. Một lần train trước cho
    # mAP50 = 0.95 trên **11 cảnh** test -- con số đẹp và vô nghĩa. Ngưỡng này
    # không chặn xuất file, nó chỉ bắt manifest phải nói thật.
    "min_test_scenes_for_confidence": 40,
}

# %% [markdown]
# ## Cell 1 — Môi trường

# %%
import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

try:
    import ultralytics
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)
    import ultralytics

from ultralytics import YOLO

print("ultralytics", ultralytics.__version__)

# %% [markdown]
# ## Cell 2 — Tìm dataset và đọc `data.yaml`

# %%
def find_dataset(root: str) -> Path:
    """Tìm đúng một dataset, hoặc dừng lại và bắt chọn.

    Bản đầu lấy ``sorted(glob("*/data.yaml"))[0]`` và **đã gây ra một lần train
    hỏng**: một Kaggle input chứa cả ``roboflow_solder_extra`` lẫn
    ``roboflow_solder_leadjoints``, ``extra`` đứng trước theo alphabet, nên nó
    âm thầm train trên bộ 109 cảnh thay vì bộ 1.257 cảnh. Không có gì trong log
    nói rằng một lựa chọn vừa được đưa ra.

    Nên giờ nhiều ứng viên là **lỗi**, không phải chuyện để đoán.
    """
    base = Path(root)
    if (base / "data.yaml").exists():
        return base
    candidates = sorted(base.glob("*/data.yaml")) or sorted(base.glob("*/*/data.yaml"))
    if not candidates:
        raise FileNotFoundError(f"Không thấy data.yaml dưới {root}")
    if len(candidates) > 1:
        listing = "\n".join(f"    {c.parent}" for c in candidates)
        raise SystemExit(
            f"Thấy {len(candidates)} dataset dưới {root}:\n{listing}\n\n"
            "Trỏ CONFIG['data_root'] thẳng vào một cái. Chọn hộ bạn là cách một "
            "lần train đã chạy nhầm bộ nhỏ hơn mà không ai biết."
        )
    return candidates[0].parent


DATA_ROOT = find_dataset(CONFIG["data_root"])
spec = yaml.safe_load((DATA_ROOT / "data.yaml").read_text(encoding="utf-8"))
CLASS_NAMES = list(spec["names"]) if isinstance(spec["names"], list) else list(spec["names"].values())

# Nguồn ĐỌC TỪ DỮ LIỆU, không hardcode. Bản đầu ghi cứng tên bộ leadjoints vào
# manifest, nên khi train chạy nhầm bộ extra thì manifest vẫn khai leadjoints --
# hồ sơ nói dối về chính model nó mô tả.
rf = spec.get("roboflow") or {}
DATA_SOURCE = (
    f"{rf.get('workspace')}/{rf.get('project')} v{rf.get('version')}"
    if rf.get("project") else f"(không có khối roboflow trong data.yaml) {DATA_ROOT.name}"
)
DATA_LICENSE = rf.get("license") or "unknown"
print("dataset:", DATA_ROOT)
print("source :", DATA_SOURCE, "|", DATA_LICENSE)
print("classes:", CLASS_NAMES)

# %% [markdown]
# ## Cell 3 — Cổng chặn: split phải sạch theo **cảnh**, không phải theo ảnh
#
# Roboflow sinh nhiều bản augment từ một ảnh gốc và đặt tên
# `<stem>.rf.<hash>.jpg`. Đếm theo file thì train có 2.401 ảnh, nhưng theo cảnh
# thì chỉ 897. Nếu hai bản augment của cùng một cảnh rơi vào train và val thì
# mAP val đo lại chính ảnh đã học.

# %%
def scenes_by_split(root: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for split in ("train", "valid", "test"):
        folder = root / split / "labels"
        if not folder.exists():
            continue
        for path in folder.glob("*.txt"):
            out[split].add(path.name.split(".rf.")[0])
    return out


scenes = scenes_by_split(DATA_ROOT)
for split, names in scenes.items():
    images = len(list((DATA_ROOT / split / "images").glob("*")))
    print(f"{split:6s} {images:5d} ảnh · {len(names):5d} cảnh gốc")

overlaps = {
    f"{a}-{b}": len(scenes.get(a, set()) & scenes.get(b, set()))
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))
}
print("trùng cảnh giữa các tập:", overlaps)

if CONFIG["require_clean_scene_split"] and any(overlaps.values()):
    raise SystemExit(
        f"Split rò rỉ theo cảnh: {overlaps}. mAP val sẽ là số ảo. Chia lại theo "
        "cảnh gốc trước khi train, hoặc đặt require_clean_scene_split=False nếu "
        "bạn cố ý chấp nhận."
    )

# %% [markdown]
# ## Cell 4 — Thống kê nhãn
#
# Bộ này **chỉ khoanh mối hàn lỗi**. Không có lớp `good`, và với detector thì
# đúng như vậy: phần chân không được khoanh chính là negative, lấy từ cùng ảnh
# cùng camera.

# %%
counts = Counter()
box_sides = []
for split in ("train", "valid", "test"):
    folder = DATA_ROOT / split / "labels"
    if not folder.exists():
        continue
    for path in folder.glob("*.txt"):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                counts[(split, CLASS_NAMES[int(parts[0])])] += 1
                box_sides.append(min(float(parts[3]), float(parts[4])) * CONFIG["imgsz"])

for key in sorted(counts):
    print(f"{key[0]:6s} {key[1]:16s} {counts[key]:6d}")
tiny = sum(1 for s in box_sides if s < 8)
print(f"\nbox có cạnh ngắn < 8 px ở imgsz={CONFIG['imgsz']}: {tiny} / {len(box_sides)}"
      f" ({tiny / max(len(box_sides), 1):.1%})")

# %% [markdown]
# ## Cell 5 — Train

# %%
work = Path(CONFIG["work_dir"])
work.mkdir(parents=True, exist_ok=True)

# Ultralytics đọc đường dẫn tương đối so với chính data.yaml, nên ghi lại một
# bản trỏ tuyệt đối thay vì dựa vào thư mục hiện hành.
local_yaml = work / "data.yaml"
local_yaml.write_text(
    yaml.safe_dump(
        {
            "path": str(DATA_ROOT),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(CLASS_NAMES),
            "names": CLASS_NAMES,
        },
        sort_keys=False,
        allow_unicode=True,
    ),
    encoding="utf-8",
)

model = YOLO(CONFIG["model"])
results = model.train(
    data=str(local_yaml),
    epochs=CONFIG["epochs"],
    imgsz=CONFIG["imgsz"],
    batch=CONFIG["batch"],
    patience=CONFIG["patience"],
    seed=CONFIG["seed"],
    project=str(work),
    name="run",
    exist_ok=True,
    plots=True,
)
weights = work / "run" / "weights" / "best.pt"
print("best:", weights)

# %% [markdown]
# ## Cell 6 — Chấm trên tập test đã khoá
#
# `valid` đã bị dùng để chọn epoch tốt nhất nên nó không còn vô tư. Con số đem
# báo cáo lấy từ `test`, tập chưa từng chạm vào trong lúc train.

# %%
best = YOLO(str(weights))
metrics = best.val(data=str(local_yaml), split="test", imgsz=CONFIG["imgsz"], plots=True)
map50 = float(metrics.box.map50)
map5095 = float(metrics.box.map)
per_class = {
    CLASS_NAMES[i]: {
        "map50": float(metrics.box.ap50[idx]),
        "precision": float(metrics.box.p[idx]),
        "recall": float(metrics.box.r[idx]),
    }
    for idx, i in enumerate(metrics.box.ap_class_index)
}
test_scenes = len(scenes.get("test", set()))
enough = test_scenes >= CONFIG["min_test_scenes_for_confidence"]
print(f"test mAP50={map50:.4f}  mAP50-95={map5095:.4f}  trên {test_scenes} cảnh gốc")
if not enough:
    print(f"\n!! CẢNH BÁO: chỉ {test_scenes} cảnh gốc trong tập test, dưới ngưỡng "
          f"{CONFIG['min_test_scenes_for_confidence']}. Con số trên là chỉ báo, "
          "không phải phép đo. Manifest sẽ ghi rõ điều này.")
gap = map50 - map5095
if gap > 0.35:
    print(f"!! mAP50 ({map50:.2f}) cách xa mAP50-95 ({map5095:.2f}): model tìm đúng "
          "chỗ nhưng box lỏng. Với mối hàn thì biên box chính là thứ cần đo.")
for name, row in per_class.items():
    print(f"   {name:16s} mAP50={row['map50']:.3f} P={row['precision']:.3f} R={row['recall']:.3f}")

if map50 < CONFIG["min_map50"]:
    raise SystemExit(
        f"test mAP50={map50:.3f} dưới ngưỡng {CONFIG['min_map50']}. Không xuất "
        "artifact: một detector yếu hơn thứ nó thay thế thì đưa vào app chỉ tạo "
        "cảm giác an toàn giả."
    )

# %% [markdown]
# ## Cell 7 — Xuất ONNX + manifest

# %%
art = Path(CONFIG["artifact_dir"])
art.mkdir(parents=True, exist_ok=True)
onnx_path = best.export(format="onnx", imgsz=CONFIG["imgsz"], opset=12, simplify=True)
shutil.copy(weights, art / "best.pt")
shutil.copy(onnx_path, art / "best.onnx")

# Manifest viết đúng schema mà `aoi_pipeline.solder.defect_detection` đọc, không
# phải một schema tự chế. Bản đầu ghi `input.color` và thiếu `head`/`model`, nên
# app từ chối nạp và cách sửa duy nhất khi đó là chỉnh tay file -- đúng thói quen
# đã sinh ra một manifest khai sai nguồn.
import hashlib
from datetime import datetime, timezone

onnx_bytes = Path(onnx_path).read_bytes()
manifest = {
    "schema_version": "aoi-solder-defect-detection/1.0",
    # Registry doc ngay tao de xep hang model; thieu no thi model khong tu mo ta
    # duoc va bi bo phieu "khong khai ngay tao".
    "created_at": datetime.now(timezone.utc).isoformat(),
    "task": "solder_defect_detection",
    "pipeline_step": "6_2_solder_defect_localization",
    "model_format": "onnx",
    "model_family": "yolo11",
    "base_model": CONFIG["model"],
    "class_names": CLASS_NAMES,
    "class_map": {str(i): name for i, name in enumerate(CLASS_NAMES)},
    "input": {
        "layout": "NCHW",
        "color_space": "RGB",
        "resize_mode": "letterbox",
        "shape": [1, 3, CONFIG["imgsz"], CONFIG["imgsz"]],
    },
    "head": {"type": "detect", "end2end": False, "max_det": 300},
    "postprocessing": {
        "recommended_confidence": 0.25,
        "recommended_iou_nms": 0.45,
        # Detector không sinh mask; giữ khoá để chung một schema, giá trị vô hại.
        "mask_threshold": 0.5,
    },
    "model": {
        "version": f"pcb-solder-detector-{CONFIG['model'].replace('.pt','')}-{DATA_ROOT.name}",
        "architecture": CONFIG["model"].replace(".pt", ""),
        "sha256": hashlib.sha256(onnx_bytes).hexdigest(),
        "bytes": len(onnx_bytes),
    },
    "reported_metrics": {
        "split": "test (locked, không dùng để chọn epoch)",
        "test_scenes": len(scenes.get("test", set())),
        "confidence_note": (
            "Đủ cảnh để coi là phép đo."
            if enough else
            f"CHỈ {len(scenes.get('test', set()))} cảnh gốc trong tập test — dưới "
            f"{CONFIG['min_test_scenes_for_confidence']}. Đây là chỉ báo, không "
            "phải phép đo; đừng trích con số này ra ngoài mà bỏ câu này lại."
        ),
        "map50_minus_map50_95": round(map50 - map5095, 4),
        "map50": round(map50, 4),
        "map50_95": round(map5095, 4),
        "per_class": per_class,
    },
    "training_data": {
        "source": DATA_SOURCE,
        "license": DATA_LICENSE,
        "dataset_folder": DATA_ROOT.name,
        "images": sum(len(list((DATA_ROOT / s / "images").glob("*"))) for s in ("train", "valid", "test")),
        "scenes": sum(len(v) for v in scenes.values()),
        "split_check": overlaps,
        "test_scenes": len(scenes.get("test", set())),
        "metrics_are_indicative_only": not enough,
    },
    "aoi_compatibility": {
        "solder_defect_detector": True,
        "solder_classifier_6_2": False,
        "required_ultralytics_task": "detect",
        "integration": "aoi_pipeline.solder.defect_detection.SolderDefectDetector",
        "diagnostic_only": True,
        "bootstrap_only": True,
        "reason": (
            "Train hoàn toàn trên dữ liệu công khai của camera khác. Đúng dải tỉ lệ "
            "(box ~2,3 % khung, board dự án ~2,2 %) nên khác hẳn SolDef_AI, nhưng "
            "ánh sáng, ống kính, lớp mạ và phổ lỗi của dây chuyền bạn vẫn chưa được "
            "đại diện. Dùng để khoanh vùng nghi ngờ, không tự quyết PASS/NG, và "
            "phải fine-tune trên ảnh board thật trước khi tin số liệu."
        ),
        "no_good_class": (
            "Dataset chỉ khoanh mối hàn lỗi. Model không biết phát biểu 'mối hàn này "
            "tốt' — nó chỉ nói 'chỗ này giống lỗi'. Vắng box không phải bằng chứng đạt."
        ),
    },
}
(art / "model_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
)
shutil.make_archive(str(art), "zip", art)
print("artifacts:", art.with_suffix(".zip"))

# %% [markdown]
# ## Cell 8 — Nạp vào app
#
# 1. Tải `pcb_solder_detector_artifacts.zip`.
# 2. Đặt `best.onnx` + `model_manifest.json` vào `models/active/solder/segmenter/`.
# 3. Mở app → **Model mối hàn 6.2 · Segment / Classify** → chọn cặp file.
#
# > Manifest ghi `bootstrap_only: true` và `diagnostic_only: true` một cách có
# > chủ ý. Model này chưa từng thấy camera của bạn, nên nó khoanh vùng để người
# > soát nhìn, không tự quyết PASS/NG. Khi có ảnh board thật đã gắn nhãn bằng
# > `scripts/build_solder_label_app.py`, fine-tune từ `best.pt` rồi cập nhật
# > manifest — lúc đó mới bỏ được hai cờ này.
