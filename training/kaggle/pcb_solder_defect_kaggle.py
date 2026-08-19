# %% [markdown]
# # AOI PCB — bước 6.2: phân loại lỗi mối hàn (ghép nhiều dataset)
#
# Notebook này train classifier lỗi mối hàn cho bước 6.2 và xuất đúng hai file
# mà app cần: `best.onnx` và `model_manifest.json`.
#
# ## Vì sao phải ghép dataset
#
# Khảo sát tháng 8/2026 cho kết quả thẳng thắn: **không có dataset công khai nào
# phủ đủ taxonomy của bước 6.2.** Nguồn tốt nhất mỗi nguồn phủ được một phần:
#
# | Nguồn | Phủ được | Ghi chú |
# |---|---|---|
# | **SolDef_AI** (Kaggle, MDPI JMMP 2024) | good, insufficient, excess, **shift_component** | Nguồn peer-reviewed duy nhất tìm được có gán nhãn lệch vị trí linh kiện |
# | HF `ouvic215` / `AndyLiu0104` | bridge, excess, missing_solder | **Không license, không nguồn gốc**; repo anh em tên `...-ControlNet` ⇒ nghi là dữ liệu sinh |
# | Roboflow soldering-defects | **cold**, bridge, insufficient | Nguồn công khai duy nhất có cold solder, nhưng chỉ vài trăm ảnh |
# | Export từ board của bạn | tất cả, nếu bạn gán nhãn | Nguồn **duy nhất** khớp camera và ánh sáng của bạn |
#
# Bị loại thẳng, đừng nối nhầm: **DeepPCB, HRIPCB/PKU-Market-PCB, DsPCBSD+,
# akhatova/pcb-defects** là lỗi **board trần** (open/short/mousebite/spur) — bài
# toán hoàn toàn khác, không có mối hàn nào trong đó. **AXI_PCB** là ảnh X-quang.
# **PCBSPDefect** chưa phát hành.
#
# ## Ba nguyên tắc notebook này cưỡng chế
#
# 1. **Nhãn không map được thì bỏ và đếm, không đoán.** Gộp bừa `solder_ball`
#    vào `excess` là giấu một loại lỗi model chưa từng thấy sau một nhãn đạt.
# 2. **Chia tập theo board/ảnh gốc, không theo crop.** Các crop cùng board dùng
#    chung ánh sáng và tiêu cự; chia theo crop cho ra điểm số dây chuyền không
#    bao giờ thấy.
# 3. **Lớp không có dữ liệu bị loại khỏi `class_names`.** Một head xuất ra lớp
#    nó chưa từng thấy sẽ cho dự đoán tự tin mà không có gì đằng sau.
#
# ## Trước khi Run All
#
# - Chọn GPU T4 x2 hoặc mới hơn.
# - **Add Input** ít nhất một dataset (xem `SOURCES` ở cell dưới).
# - Bật **Internet** nếu dùng nguồn Hugging Face hoặc muốn tải ImageNet weights.
#
# Đầu ra: `/kaggle/working/pcb_solder_defect_artifacts.zip`.

# %%
CONFIG = {
    "seed": 42,
    "work_dir": "/kaggle/working/pcb_solder_defect",
    "artifact_dir": "/kaggle/working/pcb_solder_defect_artifacts",
    # Scope quyết định head nào được train và taxonomy nào được dùng.
    #   "joint"     -> good/insufficient/excess/bridge/cold/missing_solder
    #   "component" -> ok/missing/tombstone/shifted/wrong_polarity
    "scope": "joint",
    "good_label": "good",
    "model_name": "mobilenet_v3_small",
    "input_size": 128,
    "letterbox_value": 114,
    "batch_size": 64,
    "epochs": 30,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "val_fraction": 0.2,
    # Lớp ít hơn ngưỡng này bị loại khỏi taxonomy kèm cảnh báo, thay vì train
    # một lớp mà model chỉ có thể học thuộc.
    "min_per_class": 40,
    "num_workers": 2,
}

# Bật/tắt từng nguồn ở đây. `root` là đường dẫn sau khi Add Input trên Kaggle;
# notebook tự dò cấu trúc thư mục nên không cần biết trước layout.
SOURCES = [
    {
        "name": "soldef_ai",
        "enabled": True,
        "root": "/kaggle/input/soldef-ai-pcb-dataset-for-defect-detection",
        "note": "Add Input: mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection",
    },
    {
        "name": "roboflow_soldering",
        "enabled": False,
        "root": "/kaggle/input/solder-defects-roboflow",
        "note": "Export folder-per-class từ Roboflow Universe rồi upload thành Kaggle dataset.",
    },
    {
        "name": "local_export",
        "enabled": False,
        "root": "/kaggle/input/my-solder-crops",
        "note": "Kết quả scripts/export_solder_dataset.py sau khi điền defect_class.",
    },
    {
        "name": "hf_soldering_boarding",
        "enabled": False,
        "root": "",  # tải qua Internet, xem cell Hugging Face
        "note": "Không license, nghi dữ liệu sinh. Chỉ dùng bổ sung, đừng để chiếm đa số một lớp.",
    },
]

print("Scope:", CONFIG["scope"])
for source in SOURCES:
    flag = "ON " if source["enabled"] else "off"
    print(f"  [{flag}] {source['name']:24s} {source['root'] or '(internet)'}")

# %% [markdown]
# ## 1. Lấy code ingest
#
# Module `aoi_pipeline.grading.datasets` giữ registry nguồn và bảng ánh xạ nhãn.
# Notebook nhúng lại bản rút gọn để chạy độc lập trên Kaggle, nhưng bảng ánh xạ
# giữ **nguyên** như trong repo — sửa ở một nơi rồi copy sang, đừng sửa hai nơi
# rồi để chúng trôi khỏi nhau.

# %%
import json
import os
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Notebook này in tiếng Việt và torch.onnx in emoji. Kaggle chạy UTF-8 nên không
# sao, nhưng chạy lại trên console Windows cp1252 thì chính lệnh print sẽ ném
# UnicodeEncodeError và giết run sau khi đã train xong. Nới stream ngay từ đầu.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

SEED = CONFIG["seed"]
random.seed(SEED)
np.random.seed(SEED)

JOINT_CLASSES = ("good", "insufficient", "excess", "bridge", "cold", "missing_solder")
COMPONENT_CLASSES = ("ok", "missing", "tombstone", "shifted", "wrong_polarity")
TAXONOMY = JOINT_CLASSES if CONFIG["scope"] == "joint" else COMPONENT_CLASSES

# Taxonomy mở rộng: shift_component là lỗi mức linh kiện nhưng người dùng muốn
# nó nằm cùng head với lỗi mối hàn, nên nó được phép xuất hiện ở scope "joint".
EXTRA_CLASSES = ("shift_component", "missing_component", "tombstone")
ALLOWED = tuple(dict.fromkeys(list(TAXONOMY) + list(EXTRA_CLASSES)))
print("Taxonomy đích:", ALLOWED)

# %%
# Bảng ánh xạ nhãn — giống hệt aoi_pipeline/grading/datasets.py
LABEL_MAPS = {
    "soldef_ai": {
        "good": "good", "ok": "good", "no_defect": "good", "non_defective": "good",
        "defect_free": "good", "correct": "good",
        "misalignment": "shift_component", "misaligned": "shift_component",
        "shift": "shift_component", "shifted": "shift_component",
        "displacement": "shift_component", "wrong_position": "shift_component",
        "incorrect_position": "shift_component",
        "excess": "excess", "excessive_solder": "excess", "excess_solder": "excess",
        "too_much_solder": "excess",
        "insufficient": "insufficient", "insufficient_solder": "insufficient",
        "less_solder": "insufficient", "lack_of_solder": "insufficient",
    },
    "roboflow_soldering": {
        "cold_solder": "cold", "cold": "cold", "cold_joint": "cold", "poor_wetting": "cold",
        "insufficient_solder": "insufficient", "insufficient": "insufficient",
        "excess_solder": "excess", "excessive_solder": "excess",
        "solder_bridge": "bridge", "bridge": "bridge", "short": "bridge",
        "no_solder": "missing_solder", "missing_solder": "missing_solder",
        "missing_component": "missing_component",
        "good": "good", "normal": "good", "ok": "good",
    },
    "hf_soldering_boarding": {
        "bridge": "bridge", "micro_bridge": "bridge", "excess_solder": "excess",
        "empty": "missing_solder", "less_empty": "insufficient",
    },
    "hf_soldering_tiny": {
        "bridge": "bridge", "micro_bridge": "bridge", "excess_solder": "excess",
        "empty": "missing_solder", "less_empty": "insufficient",
    },
    "local_export": {name: name for name in ALLOWED},
}

# Nhãn cố tình bỏ, kèm lý do. Ghi rõ để người đọc sau không tưởng là quên.
IGNORE = {
    "roboflow_soldering": {
        "solder_ball": "Lỗi thật nhưng không có trong taxonomy; gộp vào excess là giấu nó.",
        "solder_crack": "Như trên.",
        "solder_dross": "Như trên.",
    },
    "hf_soldering_boarding": {
        "appearance": "Nhãn ngoại quan chung chung, không có lớp tương đương.",
        "appearance_less": "Như trên.",
    },
    "hf_soldering_tiny": {
        "appearance": "Không có lớp tương đương.",
        "hole": "Via/lỗ, không phải lỗi mối hàn.",
    },
}

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def normalize_label(raw):
    return re.sub(r"[^a-z0-9]+", "_", str(raw).strip().lower()).strip("_")


print(f"{len(LABEL_MAPS)} bảng ánh xạ, {sum(len(v) for v in IGNORE.values())} nhãn bị bỏ có lý do")

# %% [markdown]
# ## 2. Dò cấu trúc từng dataset
#
# Không đoán layout. Trang Kaggle của SolDef_AI không đọc được từ ngoài lúc viết
# notebook này, nên cấu trúc thư mục được **dò tại chỗ** rồi in ra. Layout không
# nhận dạng được thì dừng và báo, chứ không ép vào một phỏng đoán rồi gán nhãn
# sai âm thầm.

# %%
def walk_files(base, max_depth=5):
    base = Path(base)
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        try:
            if len(path.relative_to(base).parts) > max_depth:
                continue
        except ValueError:
            continue
        yield path


def looks_like_coco(path):
    try:
        if path.stat().st_size > 200 * 1024 * 1024:
            return False
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            head = handle.read(4096)
    except OSError:
        return False
    return '"annotations"' in head and '"images"' in head


def looks_like_label_csv(path):
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            header = handle.readline().lower()
    except OSError:
        return False
    return (any(k in header for k in ("label", "class", "defect", "category"))
            and any(k in header for k in ("image", "file", "path", "crop")))


def class_directories(base):
    base = Path(base)
    for parent in [base, *[p for p in base.iterdir() if p.is_dir()]]:
        if not parent.is_dir():
            continue
        children = [p for p in parent.iterdir() if p.is_dir()]
        with_images = [
            child for child in children
            if any(i.suffix.lower() in IMAGE_EXTENSIONS for i in child.iterdir() if i.is_file())
        ]
        if len(with_images) >= 3:
            return sorted(with_images)
    return []


def probe_layout(root):
    base = Path(root)
    if not base.is_dir():
        return {"layout": "missing", "root": str(base), "detail": "không phải thư mục"}
    files = list(walk_files(base))
    images = [p for p in files if p.suffix.lower() in IMAGE_EXTENSIONS]
    coco = [p for p in files if p.suffix.lower() == ".json" and looks_like_coco(p)]
    csvs = [p for p in files if p.suffix.lower() == ".csv" and looks_like_label_csv(p)]
    yolo = [p for p in files if p.suffix.lower() == ".txt" and p.parent.name.lower() in {"labels", "label"}]

    if coco:
        return {"layout": "coco", "root": str(base), "ann": coco, "images": len(images)}
    if yolo:
        return {"layout": "yolo", "root": str(base), "ann": yolo, "images": len(images)}
    if csvs:
        return {"layout": "csv", "root": str(base), "ann": csvs, "images": len(images)}
    dirs = class_directories(base)
    if dirs:
        return {"layout": "folder_per_class", "root": str(base), "class_dirs": dirs, "images": len(images)}
    return {"layout": "unknown", "root": str(base), "images": len(images),
            "detail": f"{len(images)} ảnh nhưng không nhận ra cách gán nhãn"}


def show_tree(root, limit=40):
    """In cây thư mục để mắt người kiểm tra được cái máy vừa dò."""
    base = Path(root)
    if not base.is_dir():
        print(f"  (không có: {base})")
        return
    shown = 0
    for path in sorted(base.rglob("*")):
        if shown >= limit:
            print("  ...")
            break
        depth = len(path.relative_to(base).parts)
        if depth > 3:
            continue
        marker = "/" if path.is_dir() else ""
        print("  " + "  " * (depth - 1) + path.name + marker)
        shown += 1


for source in SOURCES:
    if not source["enabled"] or not source["root"]:
        continue
    print(f"\n=== {source['name']} ===")
    probe = probe_layout(source["root"])
    source["probe"] = probe
    print(f"  layout: {probe['layout']} | ảnh: {probe.get('images', 0)}")
    if probe["layout"] in {"missing", "unknown"}:
        print(f"  CHÚ Ý: {probe.get('detail', '')}")
        print("  Cây thư mục:")
        show_tree(source["root"])
    elif probe["layout"] == "folder_per_class":
        print(f"  thư mục lớp: {[p.name for p in probe['class_dirs']]}")
    else:
        print(f"  file nhãn: {[Path(p).name for p in probe['ann'][:5]]}")

# %% [markdown]
# ## 3. Đọc và ánh xạ nhãn
#
# Mỗi record mang theo `group` — ảnh gốc hoặc board mà crop được cắt ra. Bước
# chia tập dùng chính field này để giữ nguyên board ở một phía.

# %%
import csv as csv_module


def find_image(relative, roots):
    name = str(relative).replace("\\", "/").lstrip("./")
    for root in roots:
        candidate = Path(root) / name
        if candidate.is_file():
            return candidate
    stem = Path(name).name
    for root in roots:
        for found in Path(root).rglob(stem):
            if found.is_file():
                return found
    return None


def read_folder_per_class(probe):
    items = []
    for directory in probe["class_dirs"]:
        for path in sorted(Path(directory).rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                stem = path.stem
                group = stem.split("__", 1)[0] if "__" in stem else stem
                items.append({"image": path, "label": directory.name, "group": group, "bbox": None})
    return items


def read_coco(probe):
    items = []
    for annotation_file in probe["ann"]:
        annotation_file = Path(annotation_file)
        try:
            payload = json.loads(annotation_file.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:
            print(f"  bỏ qua {annotation_file.name}: {exc}")
            continue
        categories = {c["id"]: c.get("name", str(c["id"])) for c in payload.get("categories", [])}
        images = {i["id"]: i for i in payload.get("images", [])}
        roots = [annotation_file.parent, annotation_file.parent.parent, Path(probe["root"])]
        for annotation in payload.get("annotations", []):
            image = images.get(annotation.get("image_id"))
            if image is None:
                continue
            path = find_image(image.get("file_name", ""), roots)
            if path is None:
                continue
            bbox = annotation.get("bbox")
            box = None
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x, y, w, h = (float(v) for v in bbox)
                box = (int(x), int(y), int(x + w), int(y + h))
            items.append({
                "image": path,
                "label": categories.get(annotation.get("category_id"), "unknown"),
                "group": image.get("file_name", path.name),
                "bbox": box,
            })
    return items


def read_csv_manifest(probe):
    items = []
    for manifest in probe["ann"]:
        manifest = Path(manifest)
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv_module.DictReader(handle)
            if not reader.fieldnames:
                continue
            lowered = {n.lower().strip(): n for n in reader.fieldnames if n}
            label_col = next((lowered[k] for k in ("defect_class", "label", "class", "category", "defect") if k in lowered), None)
            image_col = next((lowered[k] for k in ("crop_path", "image_path", "image", "file", "filename", "path") if k in lowered), None)
            group_col = next((lowered[k] for k in ("source_image", "board", "group", "parent") if k in lowered), None)
            if not label_col or not image_col:
                continue
            for row in reader:
                label = (row.get(label_col) or "").strip()
                relative = (row.get(image_col) or "").strip()
                if not label or not relative:
                    continue
                path = find_image(relative, [manifest.parent, Path(probe["root"])])
                if path is None:
                    continue
                items.append({
                    "image": path, "label": label, "bbox": None,
                    "group": (row.get(group_col) or path.stem) if group_col else path.stem,
                })
    return items


def read_yolo(probe):
    names = {}
    for candidate in Path(probe["root"]).rglob("*.yaml"):
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = re.search(r"names\s*:\s*\[(.*?)\]", text, re.S)
        if match:
            parts = [i.strip().strip("'\"") for i in match.group(1).split(",")]
            names = {i: n for i, n in enumerate(parts) if n}
            break
        block = re.findall(r"^\s*(\d+)\s*:\s*(.+)$", text, re.M)
        if block:
            names = {int(i): n.strip().strip("'\"") for i, n in block}
            break
    items = []
    for label_file in probe["ann"]:
        label_file = Path(label_file)
        image = None
        for extension in IMAGE_EXTENSIONS:
            candidate = label_file.parent.parent / "images" / f"{label_file.stem}{extension}"
            if candidate.is_file():
                image = candidate
                break
        if image is None:
            continue
        for line in label_file.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                index = int(float(parts[0]))
                cx, cy, w, h = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            items.append({
                "image": image, "label": names.get(index, str(index)),
                "group": image.stem, "bbox": None, "yolo": (cx, cy, w, h),
            })
    return items


READERS = {
    "folder_per_class": read_folder_per_class,
    "coco": read_coco,
    "csv": read_csv_manifest,
    "yolo": read_yolo,
}

records = []
for source in SOURCES:
    if not source["enabled"] or not source.get("probe"):
        continue
    probe = source["probe"]
    if probe["layout"] not in READERS:
        print(f"\n{source['name']}: layout '{probe['layout']}' — BỎ QUA, không đoán.")
        continue

    raw = READERS[probe["layout"]](probe)
    label_map = LABEL_MAPS.get(source["name"], {})
    ignore_map = IGNORE.get(source["name"], {})
    unmapped, ignored, kept = Counter(), Counter(), []
    for item in raw:
        key = normalize_label(item["label"])
        if key in ignore_map:
            ignored[item["label"]] += 1
            continue
        mapped = label_map.get(key)
        if mapped is None or mapped not in ALLOWED:
            unmapped[item["label"]] += 1
            continue
        kept.append({**item, "mapped": mapped, "source": source["name"],
                     "group": f"{source['name']}/{item['group']}"})
    records.extend(kept)

    print(f"\n=== {source['name']} ===")
    print(f"  đọc {len(raw)} annotation -> giữ {len(kept)}")
    print(f"  theo lớp: {dict(sorted(Counter(k['mapped'] for k in kept).items()))}")
    print(f"  số group (board/ảnh gốc): {len({k['group'] for k in kept})}")
    if ignored:
        print(f"  bỏ có chủ ý: {dict(ignored)}")
    if unmapped:
        print(f"  !! KHÔNG ÁNH XẠ ĐƯỢC (đã bỏ, KHÔNG đoán): {dict(unmapped)}")
        print("     Thêm vào LABEL_MAPS hoặc IGNORE kèm lý do rồi chạy lại.")

print(f"\nTổng: {len(records)} record từ {len({r['source'] for r in records})} nguồn")

# %% [markdown]
# ## 4. Ma trận phủ taxonomy
#
# Đây là cell quan trọng nhất trước khi train. Nó nói thẳng lớp nào có đủ dữ
# liệu, lớp nào mỏng, lớp nào **không có gì cả** — và loại lớp rỗng khỏi
# `class_names` thay vì train một head xuất ra lớp nó chưa từng thấy.

# %%
counts = Counter(r["mapped"] for r in records)
per_class_sources = {
    label: dict(sorted(Counter(r["source"] for r in records if r["mapped"] == label).items()))
    for label in sorted(counts)
}

print(f"{'lớp':18s} {'số mẫu':>8s}  nguồn")
print("-" * 70)
for label in ALLOWED:
    count = counts.get(label, 0)
    flag = "  " if count >= CONFIG["min_per_class"] else ("!!" if count else "XX")
    print(f"{flag} {label:16s} {count:8d}  {per_class_sources.get(label, {})}")

missing = [c for c in ALLOWED if counts.get(c, 0) == 0]
thin = {c: counts[c] for c in ALLOWED if 0 < counts.get(c, 0) < CONFIG["min_per_class"]}
single_source = {c: s for c, s in per_class_sources.items() if len(s) == 1}

CLASS_NAMES = [c for c in ALLOWED if counts.get(c, 0) >= CONFIG["min_per_class"]]

print("\n" + "=" * 70)
if missing:
    print(f"KHÔNG CÓ DỮ LIỆU: {missing}")
    print("  -> loại khỏi class_names. Tầng luật của bước 6.2 vẫn bắt được các lỗi")
    print("     này mà không cần model (escape_guard, bridge, tombstone).")
if thin:
    print(f"QUÁ MỎNG (<{CONFIG['min_per_class']}): {thin}")
    print("  -> cũng loại. Một lớp vài chục mẫu là học thuộc, không phải học.")
if single_source:
    print(f"CHỈ TỪ MỘT NGUỒN: {sorted(single_source)}")
    print("  -> model có thể học camera của nguồn đó thay vì học lỗi. Kiểm tra bằng")
    print("     cách giữ nguyên nguồn đó ra ngoài validation xem lớp còn sống không.")

if len(CLASS_NAMES) < 2:
    raise SystemExit(
        f"Chỉ còn {len(CLASS_NAMES)} lớp đủ dữ liệu ({CLASS_NAMES}). Cần ít nhất 2. "
        "Add Input thêm dataset, hoặc hạ CONFIG['min_per_class'] nếu bạn chấp nhận "
        "rủi ro của lớp mỏng."
    )
if CONFIG["good_label"] not in CLASS_NAMES:
    raise SystemExit(
        f"Lớp '{CONFIG['good_label']}' không đủ dữ liệu. Không có lớp 'đạt' thì "
        "fusion ở bước 6.2 không biết đâu là không lỗi, và chốt chặn escape mất tác dụng."
    )

records = [r for r in records if r["mapped"] in CLASS_NAMES]
print(f"\nTrain trên {len(CLASS_NAMES)} lớp: {CLASS_NAMES}")
print(f"Còn {len(records)} record")

# %% [markdown]
# ## 5. Cắt crop
#
# Nguồn dạng COCO/YOLO là ảnh nguyên linh kiện kèm box, nên phải cắt ra trước.
# Nguồn dạng crop sẵn thì dùng thẳng. Padding quanh box giữ lại phần fillet nằm
# ngoài annotation — cùng lý do bước 5.5 nới ROI.

# %%
import cv2

CROP_PADDING_RATIO = 0.15
work = Path(CONFIG["work_dir"])
crop_dir = work / "crops"
if crop_dir.exists():
    shutil.rmtree(crop_dir)
crop_dir.mkdir(parents=True, exist_ok=True)

prepared = []
failed = 0
for index, record in enumerate(records):
    image_path = Path(record["image"])
    if record.get("bbox") is None and record.get("yolo") is None:
        prepared.append({**record, "path": image_path})
        continue

    image = cv2.imread(str(image_path))
    if image is None:
        failed += 1
        continue
    height, width = image.shape[:2]
    if record.get("yolo") is not None:
        cx, cy, bw, bh = record["yolo"]
        x1 = (cx - bw / 2) * width
        y1 = (cy - bh / 2) * height
        x2 = (cx + bw / 2) * width
        y2 = (cy + bh / 2) * height
    else:
        x1, y1, x2, y2 = record["bbox"]

    pad = CROP_PADDING_RATIO * max(x2 - x1, y2 - y1)
    x1 = max(0, int(x1 - pad)); y1 = max(0, int(y1 - pad))
    x2 = min(width, int(x2 + pad)); y2 = min(height, int(y2 + pad))
    if x2 - x1 < 8 or y2 - y1 < 8:
        failed += 1
        continue
    crop = image[y1:y2, x1:x2]
    out = crop_dir / f"{index:06d}_{record['mapped']}.png"
    cv2.imwrite(str(out), crop)
    prepared.append({**record, "path": out})

print(f"Chuẩn bị {len(prepared)} crop ({failed} bỏ vì không đọc/cắt được)")
print("Theo lớp:", dict(sorted(Counter(r["mapped"] for r in prepared).items())))

# %% [markdown]
# ## 6. Chia tập theo group
#
# Giữ nguyên board ở một phía. Nếu một lớp chỉ tồn tại ở vài group thì việc chia
# có thể xoá sạch lớp đó khỏi validation — notebook báo ra thay vì để bạn đọc
# một confusion matrix thiếu hàng mà không biết.

# %%
groups = sorted({r["group"] for r in prepared})
rng = np.random.default_rng(SEED)
order = rng.permutation(len(groups))
holdout = max(1, int(round(len(groups) * CONFIG["val_fraction"])))
val_groups = {groups[int(i)] for i in order[:holdout]}

train_records = [r for r in prepared if r["group"] not in val_groups]
val_records = [r for r in prepared if r["group"] in val_groups]

print(f"{len(groups)} group -> {len(groups) - holdout} train / {holdout} validation")
print(f"{len(train_records)} / {len(val_records)} crop")
print("train:", dict(sorted(Counter(r['mapped'] for r in train_records).items())))
print("val  :", dict(sorted(Counter(r['mapped'] for r in val_records).items())))

val_counts = Counter(r["mapped"] for r in val_records)
absent = [c for c in CLASS_NAMES if val_counts.get(c, 0) == 0]
if absent:
    print(f"\n!! Các lớp KHÔNG xuất hiện trong validation: {absent}")
    print("   Escape rate của chúng sẽ không đo được. Tăng val_fraction, hoặc")
    print("   thu thập thêm board cho các lớp này.")
if not train_records or not val_records:
    raise SystemExit("Chia tập làm rỗng một phía; chỉnh val_fraction hoặc thêm dữ liệu.")

# %% [markdown]
# ## 7. Train
#
# Class weight bật sẵn: không có nó, loss bị lớp `good` chi phối và model học
# cách không bao giờ đánh trượt thứ gì.

# %%
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
print("Device:", device)

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


def augment(image):
    # Lật và xoay nhẹ. Mối hàn không có hướng chuẩn, nhưng jitter màu mạnh sẽ
    # xoá mất chính vệt phản chiếu mà model cần học.
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 0)
    if np.random.rand() < 0.3:
        angle = float(np.random.uniform(-12, 12))
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if np.random.rand() < 0.4:
        gain = float(np.random.uniform(0.88, 1.12))
        image = np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return image


class SolderDataset(Dataset):
    def __init__(self, items, training):
        self.items = items
        self.training = training

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        record = self.items[index]
        image = cv2.imread(str(record["path"]))
        if image is None:
            image = np.full((32, 32, 3), CONFIG["letterbox_value"], np.uint8)
        if self.training:
            image = augment(image)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = letterbox(rgb, SIZE, CONFIG["letterbox_value"]).astype(np.float32) / 255.0
        tensor = (tensor - MEAN) / STD
        return (torch.from_numpy(np.ascontiguousarray(tensor.transpose(2, 0, 1))),
                INDEX_OF[record["mapped"]])


train_loader = DataLoader(SolderDataset(train_records, True), batch_size=CONFIG["batch_size"],
                          shuffle=True, num_workers=CONFIG["num_workers"])
val_loader = DataLoader(SolderDataset(val_records, False), batch_size=CONFIG["batch_size"],
                        shuffle=False, num_workers=CONFIG["num_workers"])

model = torchvision.models.mobilenet_v3_small(weights="DEFAULT")
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASS_NAMES))
model = model.to(device)

train_counts = Counter(r["mapped"] for r in train_records)
weights = torch.tensor(
    [len(train_records) / (len(CLASS_NAMES) * max(1, train_counts[n])) for n in CLASS_NAMES],
    dtype=torch.float32, device=device,
)
print("Class weights:", {n: round(float(w), 2) for n, w in zip(CLASS_NAMES, weights)})

criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

# %%
GOOD_INDEX = INDEX_OF[CONFIG["good_label"]]


def line_metrics(truths, predictions):
    """Escape rate và false-call rate — hai con số dây chuyền bị đánh giá bằng.

    Không dùng accuracy: dây chuyền 99.5% đạt thì cứ gọi tất cả là đạt đã được
    99.5%, trong khi bỏ lọt toàn bộ lỗi.
    """
    truths, predictions = np.asarray(truths), np.asarray(predictions)
    defects, goods = truths != GOOD_INDEX, truths == GOOD_INDEX
    escape = float(np.mean(predictions[defects] == GOOD_INDEX)) if defects.any() else float("nan")
    false_call = float(np.mean(predictions[goods] != GOOD_INDEX)) if goods.any() else float("nan")
    return escape, false_call


best_state, best_score = None, -1.0
history = []
for epoch in range(1, CONFIG["epochs"] + 1):
    model.train()
    total = 0.0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * images.size(0)
    scheduler.step()

    model.eval()
    predictions, truths = [], []
    with torch.no_grad():
        for images, targets in val_loader:
            logits = model(images.to(device))
            predictions.extend(logits.argmax(1).cpu().tolist())
            truths.extend(targets.tolist())
    escape, false_call = line_metrics(truths, predictions)
    # Escape áp đảo: model không bao giờ để lọt lỗi đáng giá hơn model có trung
    # bình đẹp hơn.
    score = (1.0 - (0.0 if np.isnan(escape) else escape)) * 2.0 + \
            (1.0 - (0.0 if np.isnan(false_call) else false_call))
    history.append({"epoch": epoch, "loss": total / max(1, len(train_records)),
                    "escape": escape, "false_call": false_call})
    if score > best_score:
        best_score = score
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print(f"epoch {epoch:3d}  loss {history[-1]['loss']:.4f}  "
          f"escape {escape:.3%}  false_call {false_call:.3%}")

if best_state is not None:
    model.load_state_dict(best_state)
model.eval()
print(f"\nGiữ checkpoint tốt nhất (score {best_score:.4f})")

# %% [markdown]
# ## 8. Đánh giá và chọn điểm vận hành
#
# Ngưỡng accept không phải hằng số. Quét nó và chọn điểm mà số lượng review
# đúng bằng cái dây chuyền của bạn kham được.

# %%
model_cpu = model.to("cpu")
probabilities, truths = [], []
with torch.no_grad():
    for images, targets in val_loader:
        logits = model_cpu(images)
        probabilities.append(torch.softmax(logits, dim=1).numpy())
        truths.extend(targets.tolist())
probabilities = np.concatenate(probabilities) if probabilities else np.zeros((0, len(CLASS_NAMES)))
truths = np.asarray(truths)

print("Confusion matrix (hàng = thật, cột = dự đoán):")
predicted = probabilities.argmax(1) if len(probabilities) else np.zeros(0, int)
matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), int)
for t, p in zip(truths, predicted):
    matrix[t, p] += 1
header = "".join(f"{n[:9]:>10s}" for n in CLASS_NAMES)
print(f"{'':18s}{header}")
for i, name in enumerate(CLASS_NAMES):
    print(f"{name:18s}" + "".join(f"{v:10d}" for v in matrix[i]))

print("\nQuét ngưỡng accept (dự đoán dưới ngưỡng -> hàng đợi review):")
print(f"{'accept':>8s} {'escape':>10s} {'false_call':>12s} {'review %':>10s}")
sweep = []
for threshold in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99):
    confident = probabilities.max(1) >= threshold if len(probabilities) else np.zeros(0, bool)
    decided = predicted.copy()
    # Không đủ tự tin thì không kết luận đạt; đẩy sang review.
    decided_good = confident & (decided == GOOD_INDEX)
    defects, goods = truths != GOOD_INDEX, truths == GOOD_INDEX
    escape = float(np.mean(decided_good[defects])) if defects.any() else float("nan")
    false_call = float(np.mean(~decided_good[goods])) if goods.any() else float("nan")
    review = float(np.mean(~confident)) if len(confident) else float("nan")
    sweep.append({"accept": threshold, "escape": escape,
                  "false_call": false_call, "review_rate": review})
    print(f"{threshold:8.2f} {escape:10.3%} {false_call:12.3%} {review:10.1%}")

print("\nChọn ngưỡng có escape chấp nhận được TRƯỚC, rồi mới xét false_call.")
print("Sửa decision_thresholds trong manifest theo lựa chọn đó.")

# %% [markdown]
# ## 9. Xuất artifact
#
# Đúng hai file app cần. Manifest theo schema `pcb-solder-defect-classifier/1.0`;
# runtime **từ chối** manifest sai schema thay vì đoán — đoán sai thứ tự class là
# biến mọi lỗi thành "đạt".

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
    # 18, không phải 17: exporter mới phát 18 rồi thất bại khi tự hạ xuống 17.
    opset_version=18,
)


def collapse_external_data(path):
    """Gộp trọng số trở lại chính file .onnx.

    Exporter của torch có thể ghi tensor ra file `best.onnx.data` bên cạnh. App
    được ghi tài liệu là cần đúng hai file — .onnx và manifest — và SHA-256
    trong manifest chỉ phủ file .onnx, nên một export bị tách sẽ nạp tốt trên
    máy vừa train và hỏng ở mọi nơi khác với lỗi "External data path validation
    failed". Gộp lại khi file .data còn nằm cạnh.
    """
    import onnx
    model = onnx.load(str(path))
    onnx.save_model(model, str(path), save_as_external_data=False)
    for orphan in Path(path).parent.glob(f"{Path(path).name}*.data"):
        orphan.unlink()
    for orphan in Path(path).parent.glob(f"{Path(path).stem}*.data"):
        orphan.unlink()
    print(f"ONNX tự chứa ({Path(path).stat().st_size / 1e6:.1f} MB)")


collapse_external_data(onnx_path)

# Không xuất một model mà bản ONNX không tái tạo được.
try:
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"input": dummy.numpy()})[0]
    with torch.no_grad():
        torch_out = model_cpu(dummy).numpy()
    difference = float(np.max(np.abs(onnx_out - torch_out)))
    assert difference < 1e-3, f"ONNX lệch torch {difference}"
    print(f"ONNX khớp torch (lệch tối đa {difference:.2e})")
except ImportError:
    print("Không có onnxruntime; bỏ qua bước đối chiếu export.")


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest = {
    "schema_version": "pcb-solder-defect-classifier/1.0",
    "task": "solder_defect_classification",
    "scope": CONFIG["scope"],
    "model_format": "onnx",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "class_names": CLASS_NAMES,
    "good_label": CONFIG["good_label"],
    "input": {
        "name": "input",
        "size": [CONFIG["input_size"], CONFIG["input_size"]],
        "color_space": "RGB",
        "resize_mode": "letterbox",
        "letterbox_value": CONFIG["letterbox_value"],
        "normalization": {"mean": MEAN.tolist(), "std": STD.tolist()},
    },
    "output": {"name": "logits", "type": "raw_logits"},
    "calibration": {"temperature": 1.0},
    "decision_thresholds": {"accept": 0.85, "review": 0.50, "accept_by_class": {}},
    "model": {
        "version": f"solder-{CONFIG['scope']}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}",
        "architecture": CONFIG["model_name"],
        "sha256": sha256_of(onnx_path),
    },
    "training": {
        "sources": sorted({r["source"] for r in prepared}),
        "groups_total": len(groups),
        "roi_train": len(train_records),
        "roi_val": len(val_records),
        "class_counts": dict(sorted(Counter(r["mapped"] for r in prepared).items())),
        "classes_dropped_for_lack_of_data": missing + list(thin),
        "single_source_classes": sorted(single_source),
        "epochs": CONFIG["epochs"],
        "seed": SEED,
        "threshold_sweep": sweep,
        "history": history,
    },
}
(artifacts / "model_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
)

shutil.make_archive("/kaggle/working/pcb_solder_defect_artifacts", "zip", artifacts)
print(f"\nĐã ghi {onnx_path.name} và model_manifest.json")
print("Tải: /kaggle/working/pcb_solder_defect_artifacts.zip")

# %% [markdown]
# ## 10. Trước khi đưa lên dây chuyền
#
# **Kiểm tra artifact trước:**
#
# ```powershell
# .\.venv\Scripts\python.exe scripts\verify_solder_model.py `
#   models\solder\best.onnx models\solder\model_manifest.json
# ```
#
# Lệnh này nạp cặp file qua đúng runtime app dùng, nên pass ở đây nghĩa là app
# nạp được.
#
# **Ba giới hạn phải nhớ:**
#
# 1. **Khoảng cách miền.** Model này học camera, ống kính và ánh sáng của các
#    dataset công khai, không phải của bạn. Nó là điểm khởi đầu để fine-tune
#    trên board của bạn, không phải model production. Bước 6.2 hợp nhất nó với
#    tầng luật đo chính vì lý do này: khi hai tầng bất đồng, ROI đi vào hàng đợi
#    kiểm tra thay vì tin model.
# 2. **Lớp bị loại vẫn được bắt.** Lớp nào không đủ dữ liệu đã bị loại khỏi
#    `class_names`, nhưng tầng luật của bước 6.2 vẫn bắt được chúng mà không cần
#    model — `escape_guard` cho thiếu thiếc, luật cặp chân cho bridge, luật so
#    hai đầu cho dựng bia.
# 3. **Ngưỡng trong manifest là điểm khởi đầu.** Dùng bảng quét ở cell 8 để chọn
#    điểm vận hành, rồi sửa `decision_thresholds` trước khi triển khai.
#
# Và nhớ chạy `scripts/calibrate_solder_thresholds.py` trên board đạt chuẩn của
# bạn — tầng luật đo cần ngưỡng của dây chuyền bạn, không phải ngưỡng mặc định.
