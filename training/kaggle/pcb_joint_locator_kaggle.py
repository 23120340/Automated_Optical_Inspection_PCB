# %% [markdown]
# # Lượt 2 — **định vị mối hàn** trong crop linh kiện (một lớp)
#
# Notebook này train model trả lời câu **"mối hàn nằm ở đâu"**, không phải câu
# "mối hàn nào lỗi". Hai câu đó cần hai bộ dữ liệu khác nhau, và trộn chúng lại
# là hỏng — phần dưới có bằng chứng.
#
# ## Vì sao không dùng lại `pcb_solder_detector_kaggle`
#
# Notebook đó train trên `roboflow_solder_leadjoints`, bộ **chỉ khoanh mối hàn
# LỖI**. Kiểm bằng mắt trên ảnh của chính bộ đó: một SOIC có ~28 chân nhìn thấy
# được nhưng chỉ có **2–6 box** trên vài chân cụ thể; hơn 20 chân lành để trống.
#
# Nhãn của bộ này thì khoanh **mọi** chân — IC DIP 18 chân có đúng 18 box.
#
# Ghép hai bộ vào nhau là dạy model hai điều trái ngược trên **cùng một mẫu
# vật**: một mối hàn lành là *nền* ở bộ Roboflow và là *positive* ở bộ này.
# Không cách chia tập nào cứu được, vì mâu thuẫn nằm trong chính nhãn.
#
# Nên notebook này **không** merge Roboflow. Đó là quyết định, không phải thiếu sót.
#
# ## Model này lấp chỗ trống nào
#
# Bước 5.5 hiện *suy ra* ROI mối hàn bằng hình học, vì detector lượt 1 cho lớp
# `pads` recall **0.072** — 30 trên 670 ảnh train có lớp đó. Hình học không biết
# trên board thực sự có gì: nó không phân biệt được trục nào của một linh kiện
# gần vuông mang chân, và phải trừu ra `_cross` khi hai trục đo gần bằng nhau.
#
# `aoi_pipeline/solder/lead_detection.py` đã sẵn đường ống cho lượt 2; thứ thiếu
# duy nhất là model này.
#
# > **Tên lớp không tuỳ tiện.** `aoi_pipeline/solder/leads.py` chỉ hợp nhất
# > detection có nhãn nằm trong `LEAD_CLASSES`. `solder_joint` đã được thêm vào
# > đó. Đổi tên lớp ở đây mà không sửa set kia thì mọi detection bị **bỏ qua
# > lặng lẽ** — không lỗi, không cảnh báo, chỉ là lượt 2 không làm gì cả.
#
# ## Dữ liệu vào
#
# `datasets/train/solder_joint_v1`, do
# `scripts/pack_joint_detection_dataset.py` đóng gói từ hai phiên gắn nhãn:
#
# | | crop | box | ảnh nền | cảnh gốc |
# |---|---:|---:|---:|---:|
# | `fpic_components` | 1.046 | 4.614 | 425 | 94 |
# | `winnies_components` | 1.008 | 4.618 | 0 | 71 |
# | **tổng** | **2.054** | **9.232** | **425** | **165** |
#
# Đã chia sẵn **theo cảnh**, kiểm 0 rò rỉ. Notebook này **không chia lại** — chia
# lại theo crop sẽ đặt anh em ruột vào cả train lẫn val.
#
# ## Trước khi Run All
#
# - Add Input thư mục `solder_joint_v1`, trỏ `CONFIG["data_root"]` **thẳng vào
#   thư mục có `data.yaml`**.
# - GPU T4 đủ. `imgsz=640` nên nhanh hơn notebook detector lỗi (1280) khoảng 4 lần.
#
# Đầu ra: `/kaggle/working/pcb_joint_locator_artifacts.zip` gồm `best.pt`,
# `best.onnx`, `model_manifest.json`.

# %%
CONFIG = {
    "seed": 42,

    # --- dữ liệu ---------------------------------------------------------
    "data_root": "/kaggle/input/solder-joint-v1",
    "work_dir": "/kaggle/working/pcb_joint_locator",
    "artifact_dir": "/kaggle/working/pcb_joint_locator_artifacts",

    # --- model -----------------------------------------------------------
    # yolo11s, không phải n. Vật thể nhỏ (cạnh ngắn trung vị 12 px trên ảnh gốc)
    # nên nhánh P3 mỏng của model n là bất lợi thật. Cũng không phải m/l: chỉ có
    # 165 cảnh gốc, model to sẽ học thuộc cảnh thay vì học hình dạng mối hàn.
    #
    # Chỉ báo để biết lựa chọn này sai: khoảng cách train↔test lớn ở Cell 5.
    # Nếu lệch nhiều thì đổi về yolo11n, đừng thêm epoch.
    "model": "yolo11s.pt",

    # imgsz=640, chọn theo số đo trên chính bộ này. Đếm box có cạnh ngắn tụt
    # dưới 8 px sau khi crop được co về imgsz — tức dưới bước nhảy nhỏ nhất của
    # đầu dò:
    #
    #     imgsz    box < 8 px
    #      256     1.474  (16,0 %)
    #      384       301  ( 3,3 %)
    #      512        90  ( 1,0 %)
    #      640        28  ( 0,3 %)
    #      960         0  ( 0,0 %)
    #
    # Cạnh dài crop trải 115..1994 px, trung vị 181. Con số này KHÔNG chọn theo
    # trung vị: imgsz nhỏ thì phóng to crop bé (vô hại) nhưng THU NHỎ crop lớn,
    # mà crop lớn chính là IC nhiều chân. 640 là mức đầu tiên đưa tỉ lệ box quá
    # nhỏ xuống dưới 1 %; 960 không còn box nào nhưng tốn gấp 2,2 lần thời gian
    # để mua 0,3 %.
    "imgsz": 640,
    "epochs": 120,
    "batch": 16,
    "patience": 30,

    # --- cổng chặn -------------------------------------------------------
    # Chia tập do packer làm và đã kiểm; cổng này chạy lại phép kiểm đó, vì một
    # bản đóng gói lại về sau có thể không còn sạch, và một cảnh nằm cả train
    # lẫn val thì mAP val là số ảo.
    "require_clean_scene_split": True,
    # Dưới ngưỡng này thì hình học của bước 5.5 vẫn tốt hơn — đừng xuất.
    "min_map50": 0.50,
    # mAP đo trên ít cảnh gốc thì không phải phép đo. Ngưỡng này không chặn xuất
    # file, nó chỉ bắt manifest phải nói thật.
    "min_test_scenes_for_confidence": 20,
    # Chênh lệch train↔test lớn hơn mức này là dấu hiệu học thuộc cảnh.
    "max_train_test_gap": 0.20,
}

# %% [markdown]
# ## Cell 1 — Cài đặt

# %%
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "ultralytics", "onnx", "onnxruntime"],
    check=True,
)

# %%
import collections
from datetime import datetime, timezone
import hashlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())

# %% [markdown]
# ## Cell 2 — Tìm dataset, đọc lớp từ `data.yaml`
#
# Trỏ nhầm vào thư mục cha là lỗi đã xảy ra thật một lần: Kaggle input chứa hai
# bộ, notebook lấy cái đứng đầu alphabet, và train trên bộ nhỏ hơn 10 lần trong
# khi manifest vẫn khai nguồn đúng. Nhiều `data.yaml` giờ là **lỗi**, không phải
# một phép đoán.


# %%
def find_dataset(root: str) -> Path:
    start = Path(root)
    if not start.exists():
        raise SystemExit(f"Không thấy {start}. Đã Add Input chưa?")
    if (start / "data.yaml").is_file():
        return start
    found = sorted(start.rglob("data.yaml"))
    if not found:
        raise SystemExit(f"Không có data.yaml nào dưới {start}.")
    if len(found) > 1:
        listing = "\n".join(f"  - {p.parent}" for p in found)
        raise SystemExit(
            f"Có {len(found)} data.yaml dưới {start}:\n{listing}\n"
            "Trỏ CONFIG['data_root'] thẳng vào một thư mục cụ thể."
        )
    return found[0].parent


DATA_ROOT = find_dataset(CONFIG["data_root"])
spec = yaml.safe_load((DATA_ROOT / "data.yaml").read_text(encoding="utf-8"))
CLASS_NAMES = (
    list(spec["names"]) if isinstance(spec["names"], list) else list(spec["names"].values())
)
print(f"dataset : {DATA_ROOT}")
print(f"lớp     : {CLASS_NAMES}")
if len(CLASS_NAMES) != 1:
    print(f"!! Bộ này có {len(CLASS_NAMES)} lớp. Notebook viết cho bài một lớp; "
          "số liệu theo lớp vẫn chạy nhưng phần diễn giải bên dưới nói về một lớp.")

PACK = DATA_ROOT / "pack_manifest.json"
pack_manifest = json.loads(PACK.read_text(encoding="utf-8")) if PACK.is_file() else {}
if pack_manifest:
    print(f"\nđóng gói từ {len(pack_manifest.get('sources', []))} phiên gắn nhãn:")
    for src in pack_manifest.get("sources", []):
        print(f"  {src['tag']:<24}{src['crops_written']:>5} crop  "
              f"{src['boxes_written']:>5} box  (người gắn: {src.get('reviewer_id') or '—'})")
    if pack_manifest.get("collapsed_from"):
        print(f"  lớp gốc {pack_manifest['collapsed_from']} đã gộp -> {CLASS_NAMES}")

# %% [markdown]
# ## Cell 3 — Cổng chặn: chia tập phải sạch
#
# Tên file mang tiền tố nguồn và tên cảnh, nên kiểm được rò rỉ mà không cần
# manifest. Đây là phép kiểm chạy lại, không phải phép tin.

# %%
scenes: dict[str, set[str]] = {}
counts: dict[str, int] = {}
for split in ("train", "valid", "test"):
    directory = DATA_ROOT / split / "images"
    names = sorted(p.name for p in directory.iterdir()) if directory.is_dir() else []
    counts[split] = len(names)
    bucket: set[str] = set()
    for name in names:
        stem = Path(name).stem
        # <source>__<scene>__<index>__<class>
        parts = stem.split("__")
        bucket.add("__".join(parts[:2]) if len(parts) >= 3 else stem)
    scenes[split] = bucket
    print(f"{split:6s} {counts[split]:5d} ảnh · {len(bucket):4d} cảnh gốc")

overlaps = {
    f"{a}-{b}": sorted(scenes[a] & scenes[b])
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))
}
for pair, shared in overlaps.items():
    print(f"cảnh trùng {pair}: {len(shared)}")
if CONFIG["require_clean_scene_split"] and any(overlaps.values()):
    raise SystemExit(
        "Có cảnh nằm ở hai tập. mAP val sẽ là số ảo. Đóng gói lại bằng "
        "scripts/pack_joint_detection_dataset.py, vốn chia theo cảnh."
    )

boxes_per_split = {}
empty_per_split = {}
for split in ("train", "valid", "test"):
    directory = DATA_ROOT / split / "labels"
    total = empty = 0
    for label in directory.iterdir() if directory.is_dir() else []:
        lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        total += len(lines)
        empty += not lines
    boxes_per_split[split] = total
    empty_per_split[split] = empty
    print(f"{split:6s} {total:5d} box · {empty:4d} ảnh nền")
print(f"\ntổng {sum(boxes_per_split.values())} box trên {sum(counts.values())} ảnh")

# %% [markdown]
# ## Cell 4 — Kiểm tra ngược: vẽ lại nhãn lên ảnh
#
# Đọc ngược file YOLO và vẽ lên ảnh. Box lệch thì lệch ở đây, không phải sau 120
# epoch. Một lỗi dấu trừ trong phép quy đổi toạ độ trông y hệt một model kém.

# %%
import cv2
import matplotlib.pyplot as plt

samples = sorted((DATA_ROOT / "train" / "images").iterdir())[:8]
if samples:
    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    for axis, image_path in zip(axes.ravel(), samples):
        patch = cv2.imread(str(image_path))
        height, width = patch.shape[:2]
        label_path = DATA_ROOT / "train" / "labels" / f"{image_path.stem}.txt"
        drawn = 0
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            x1, y1 = int((cx - bw / 2) * width), int((cy - bh / 2) * height)
            x2, y2 = int((cx + bw / 2) * width), int((cy + bh / 2) * height)
            cv2.rectangle(patch, (x1, y1), (x2, y2), (0, 255, 0), 1)
            drawn += 1
        axis.imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        axis.set_title(f"{drawn} box · {width}x{height}", fontsize=8)
        axis.axis("off")
    plt.tight_layout()
    plt.show()
    print("Nhìn kỹ: box phải nằm trên kim loại. Lệch ở đây thì đừng train tiếp.")

# %% [markdown]
# ## Cell 5 — Train
#
# Augmentation chọn cho bài toán **crop**, khác hẳn detect trên board rộng:
#
# - `mosaic` thấp. Mosaic ghép 4 ảnh; trên crop nó tạo ra thứ không bao giờ tồn
#   tại lúc chạy — lượt 2 luôn nhìn đúng một linh kiện.
# - `scale` hẹp. Crop được cắt theo chính hộp linh kiện nên tỉ lệ tương đối gần
#   như cố định. Dạy bất biến tỉ lệ là phí sức.
# - `fliplr` **và** `flipud` đều bật. Linh kiện nằm mọi hướng trên board.
# - `degrees` nhỏ. Linh kiện đặt theo lưới; lệch nhiều đã là lỗi lắp.
# - `hsv_v` rộng hơn mặc định: hai nguồn ảnh có độ sáng khác nhau rõ, và lúc
#   chạy thật sẽ là camera thứ ba.

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
    pretrained=True,
    mosaic=0.15,
    scale=0.25,
    fliplr=0.5,
    flipud=0.5,
    degrees=7.0,
    hsv_v=0.5,
    hsv_s=0.5,
    translate=0.08,
)
weights = Path(results.save_dir) / "weights" / "best.pt"
print("best.pt:", weights)

# %% [markdown]
# ## Cell 6 — Đo trên `test`
#
# `valid` đã bị dùng để chọn epoch tốt nhất nên nó không còn vô tư. Con số đem
# báo cáo lấy từ `test`, tập chưa từng chạm vào trong lúc train.
#
# Đo thêm trên `train` để lộ ra việc học thuộc cảnh: với 165 cảnh gốc thì đây là
# rủi ro chính, và mAP test một mình không nói được.

# %%
best = YOLO(str(weights))
metrics = best.val(data=str(local_yaml), split="test", imgsz=CONFIG["imgsz"], plots=True)
map50 = float(metrics.box.map50)
map5095 = float(metrics.box.map)
precision = float(metrics.box.mp)
recall = float(metrics.box.mr)

train_metrics = best.val(data=str(local_yaml), split="train", imgsz=CONFIG["imgsz"], plots=False)
train_map50 = float(train_metrics.box.map50)
gap = train_map50 - map50

test_scenes = len(scenes.get("test", set()))
enough = test_scenes >= CONFIG["min_test_scenes_for_confidence"]
print(f"test  mAP50={map50:.4f}  mAP50-95={map5095:.4f}  P={precision:.4f}  R={recall:.4f}")
print(f"      trên {test_scenes} cảnh gốc")
print(f"train mAP50={train_map50:.4f}  →  chênh lệch train-test = {gap:+.4f}")

if not enough:
    print(f"\n!! CẢNH BÁO: chỉ {test_scenes} cảnh gốc trong tập test, dưới ngưỡng "
          f"{CONFIG['min_test_scenes_for_confidence']}. Con số trên là chỉ báo, "
          "không phải phép đo. Manifest sẽ ghi rõ điều này.")
if gap > CONFIG["max_train_test_gap"]:
    print(f"\n!! Chênh lệch train-test {gap:.3f} vượt {CONFIG['max_train_test_gap']}: "
          "model đang học thuộc cảnh. Đổi sang yolo11n hoặc gắn nhãn thêm board — "
          "thêm epoch chỉ làm nặng thêm.")
if map50 - map5095 > 0.35:
    print(f"\n!! mAP50 ({map50:.2f}) cách xa mAP50-95 ({map5095:.2f}): model tìm đúng "
          "chỗ nhưng box lỏng. Với mối hàn thì biên box chính là thứ bước 6.2 đo.")

if map50 < CONFIG["min_map50"]:
    raise SystemExit(
        f"test mAP50={map50:.3f} dưới ngưỡng {CONFIG['min_map50']}. Không xuất "
        "artifact: bước 5.5 đang suy ra ROI bằng hình học và đạt 0 pad bỏ sót "
        "trên board thật. Một model yếu hơn thế thì thay vào chỉ làm tệ đi."
    )

# %% [markdown]
# ## Cell 7 — Xuất ONNX + manifest
#
# Manifest khai đúng vai trò: đây là model **định vị**, không phải model chấm
# lỗi. Nó không có lớp nào cho "mối hàn lỗi" và không được dùng để ra phán quyết
# PASS/NG.

# %%
artifacts = Path(CONFIG["artifact_dir"])
artifacts.mkdir(parents=True, exist_ok=True)

onnx_path = Path(
    best.export(format="onnx", imgsz=CONFIG["imgsz"], opset=12, simplify=True, dynamic=False)
)
shutil.copy2(weights, artifacts / "best.pt")
shutil.copy2(onnx_path, artifacts / "best.onnx")

digest = hashlib.sha256((artifacts / "best.onnx").read_bytes()).hexdigest()
sources = pack_manifest.get("sources", [])
manifest = {
    "schema_version": "aoi-lead-detection/1.0",
    # Không có nó thì bảng chọn model in "—" ở cột ngày và hai lần train của
    # cùng một notebook không phân biệt được bằng mắt.
    "created_at": datetime.now(timezone.utc).isoformat(),
    "task": "solder_joint_localization",
    "pipeline_step": "5_5_pass2_lead_detection",
    "model_format": "onnx",
    "model_family": "yolo11",
    "base_model": CONFIG["model"],
    "class_names": CLASS_NAMES,
    "input": {
        "layout": "NCHW",
        "color_space": "RGB",
        "resize_mode": "letterbox",
        "shape": [1, 3, CONFIG["imgsz"], CONFIG["imgsz"]],
    },
    "reported_metrics": {
        "split": "test (khoá, không dùng để chọn epoch)",
        "test_scenes": test_scenes,
        "confidence_note": (
            "Đủ cảnh để coi là phép đo."
            if enough
            else f"Chỉ {test_scenes} cảnh gốc — chỉ báo, không phải phép đo."
        ),
        "map50": round(map50, 4),
        "map50_95": round(map5095, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "train_map50": round(train_map50, 4),
        "train_test_gap": round(gap, 4),
        "scene_memorisation_warning": bool(gap > CONFIG["max_train_test_gap"]),
    },
    "training_data": {
        "sources": [
            {
                "tag": s["tag"],
                "crops": s["crops_written"],
                "boxes": s["boxes_written"],
                "reviewer_id": s.get("reviewer_id", ""),
                "boxes_sha256": s.get("boxes_sha256", ""),
            }
            for s in sources
        ],
        "images": sum(counts.values()),
        "boxes": sum(boxes_per_split.values()),
        "background_images": sum(empty_per_split.values()),
        "scenes": {split: len(names) for split, names in scenes.items()},
        "split_by": pack_manifest.get("split_by", "scene_id"),
        "collapsed_from": pack_manifest.get("collapsed_from"),
        "split_check": {pair: len(shared) for pair, shared in overlaps.items()},
    },
    "aoi_compatibility": {
        "lead_detector_pass2": True,
        "solder_defect_detector": False,
        "solder_classifier_6_2": False,
        "required_ultralytics_task": "detect",
        "integration": "aoi_pipeline.solder.lead_detection.detect_leads_in_components",
        "config_key": "lead_detection.model_path",
        "fusion_note": (
            "Tên lớp phải nằm trong aoi_pipeline.solder.leads.LEAD_CLASSES, nếu "
            "không mọi detection bị bỏ qua lặng lẽ ở bước hợp nhất. "
            f"{CLASS_NAMES} hiện nằm trong set đó."
        ),
        "not_a_defect_detector": (
            "Model này khoanh MỌI mối hàn, kể cả mối hàn tốt. Nó không có lớp "
            "nào cho lỗi và không được dùng để ra phán quyết PASS/NG. Việc chấm "
            "lỗi thuộc về bước 6.2 (classifier hoặc detector lỗi)."
        ),
        "bootstrap_only": True,
        "reason": (
            "Ảnh công khai (fpic_boards_rf100 và pcb_packages_winnies, cả hai "
            "CC BY 4.0), box do người của dự án vẽ tay. Ánh sáng, ống kính và "
            "lớp mạ của dây chuyền bạn chưa được đại diện, nên phải fine-tune "
            "trên ảnh board thật trước khi tin số liệu."
        ),
    },
    "postprocessing": {
        "recommended_confidence": 0.25,
        "recommended_iou_nms": 0.45,
    },
    "model": {
        "version": f"pcb-joint-locator-{CONFIG['model'].removesuffix('.pt')}-solderjoint",
        "architecture": CONFIG["model"].removesuffix(".pt"),
        "sha256": digest,
        "bytes": (artifacts / "best.onnx").stat().st_size,
    },
}
(artifacts / "model_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(manifest["reported_metrics"], indent=2, ensure_ascii=False))

# %% [markdown]
# ## Cell 8 — Đóng gói
#
# Tải file zip về, giải nén vào `models/active/lead_detector/`, rồi trỏ
# `lead_detection.model_path` vào `best.onnx`.

# %%
archive = shutil.make_archive(
    str(Path("/kaggle/working") / "pcb_joint_locator_artifacts"),
    "zip",
    root_dir=artifacts,
)
print("xong:", archive, f"({Path(archive).stat().st_size / 1e6:.1f} MB)")
for item in sorted(artifacts.iterdir()):
    print(f"  {item.name:<24}{item.stat().st_size / 1e6:>8.2f} MB")
