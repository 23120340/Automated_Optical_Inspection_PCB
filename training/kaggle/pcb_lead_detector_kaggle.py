# %% [markdown]
# # Lượt 2 — detect chân/pad BÊN TRONG crop linh kiện
#
# Notebook này train model cho **giai đoạn C** của hướng detect 2 lượt
# (`Docs/tien_do_detect_2_luot.md`). Đường ống phía app đã sẵn sàng
# (`aoi_pipeline/lead_detection.py`); thứ còn thiếu duy nhất là model này.
#
# ## Vì sao cần notebook riêng, không dùng lại detector cũ
#
# Đã đo trên board thật của dự án, không phải suy đoán: chạy
# `models/detector/kaggle/ver2/best.onnx` lên crop của 8 linh kiện × 3 mức biên
# cho ra **`pads`/`pins` = 0 trong cả 24 cấu hình**, và nhãn trả về là rác — một
# con điện trở chip phóng to thành `clock`, `transistor`, `potentiometer`.
#
# Lý do đơn giản: nó học trên **ảnh board rộng**, nơi một linh kiện chỉ chiếm vài
# chục pixel giữa cả nghìn pixel khác. Đưa cho nó một crop mà linh kiện chiếm
# trọn khung là đưa một phân bố nó chưa từng thấy.
#
# **Bài học rút ra và là nguyên tắc số một của notebook này:**
#
# > Ảnh lúc train phải giống ảnh lúc chạy. Lượt 2 nhìn crop, nên train phải
# > nhìn crop. Train trên ảnh board rồi đem chạy trên crop là lặp lại đúng
# > sai lầm vừa đo được.
#
# Cùng bài học đó lặp lại lần thứ hai với SolDef_AI: model YOLO11m-seg train
# trên ảnh macro (1–3 µm/px) đạt Box mAP50 0.771 trên chính nó, nhưng cho **0
# box** trên board của dự án (46 µm/px) ở mọi mức phóng đại 1×–12×. Phóng to
# bằng phần mềm không tạo ra chi tiết chưa từng được chụp.
#
# ## Dữ liệu vào
#
# Đầu vào là thư mục do `scripts/bootstrap_lead_labels.py` xuất ra **và đã được
# người sửa nhãn**:
#
# ```
# datasets/leads_v1/
#   images/<stem>.png        khung ảnh phân tích của cả board
#   labels/<stem>.txt        YOLO: pads/pins theo toạ độ BOARD
#   components/<stem>.json   box linh kiện của lượt 1  <-- notebook này cần
#   data.yaml
# ```
#
# Notebook tự chuyển nó thành dataset **theo crop** trước khi train.
#
# > **Nhãn bootstrap chưa sửa là nhãn giả.** Train thẳng trên chúng chỉ dạy model
# > tái tạo lại đúng hình học đã sinh ra chúng — điểm sẽ đẹp mà không học được
# > gì thật. Notebook có cổng chặn ở Cell 3 nếu thấy dấu hiệu chưa sửa.
#
# ## Trước khi Run All
#
# - GPU T4 là đủ. Bài này **dễ hơn** detect trên board rộng rất nhiều: mỗi crop
#   chỉ có một linh kiện và 2–4 pad.
# - Add Input dataset đã sửa nhãn.
# - Bật Internet lần đầu để tải weight pretrain.
#
# Đầu ra: `/kaggle/working/pcb_lead_detector_artifacts.zip` gồm `best.pt`,
# `best.onnx`, `model_manifest.json` và metrics theo lớp.

# %%
CONFIG = {
    "seed": 42,

    # --- dữ liệu ---------------------------------------------------------
    # Thư mục do bootstrap_lead_labels.py xuất ra SAU KHI đã sửa nhãn.
    "bootstrap_root": "/kaggle/input/pcb-leads-corrected",
    "work_dir": "/kaggle/working/pcb_lead_detector",
    "artifact_dir": "/kaggle/working/pcb_lead_detector_artifacts",

    # Bỏ qua cổng kiểm tra nhãn giả. CHỈ đặt True khi bạn biết chắc mình đang
    # làm gì — ví dụ chạy thử đường ống trên dữ liệu chưa sửa.
    "allow_unreviewed_labels": False,

    # --- hình học crop: phải KHỚP với lúc chạy ---------------------------
    # Đây là những con số của LeadDetectionConfig trong aoi_pipeline/config.py.
    # Lệch giữa train và inference là lệch phân bố, và không ai báo cho bạn biết.
    "crop_margin_ratio": 0.35,
    "crop_margin_min_px": 6,
    "min_crop_px": 24,
    # Crop không chứa pad nào sau khi cắt: giữ một phần làm ảnh nền (negative)
    # để model biết "chỗ này không có chân", nhưng đừng để chúng lấn át.
    "max_empty_crop_ratio": 0.15,

    # --- model -----------------------------------------------------------
    # yolo11s là đủ: crop chỉ có một linh kiện, và lượt 2 chạy MỘT LẦN CHO MỖI
    # LINH KIỆN — cỡ 1000 forward pass mỗi board — nên tốc độ quan trọng hơn
    # sức mạnh. Đổi sang yolo11n nếu thời gian chu kỳ căng, yolo11m nếu đã có
    # nhiều dữ liệu và cần thêm điểm.
    "model": "yolo11s.pt",
    "imgsz": 640,
    "epochs": 120,
    "batch": 16,
    "patience": 30,

    # --- chia tập --------------------------------------------------------
    # THEO BOARD, không theo crop. Hai crop của cùng một board có cùng ánh sáng,
    # cùng lô hàn, thường cùng loại linh kiện — chia theo crop là rò rỉ, và điểm
    # val sẽ đẹp một cách vô nghĩa.
    "val_fraction": 0.15,
    "test_fraction": 0.15,
}

CLASS_NAMES = ["pads", "pins"]

# %% [markdown]
# ## Cell 1 — Môi trường

# %%
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

print("Python :", sys.version.split()[0])
print("OS     :", platform.platform())
try:
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                          "--format=csv,noheader"], capture_output=True,
                         text=True, check=False).stdout.strip() or "không thấy GPU")
except FileNotFoundError:
    print("nvidia-smi không có — sẽ chạy CPU, rất chậm")

# %%
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "ultralytics", "opencv-python-headless", "pyyaml"], check=False)

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
print("ultralytics sẵn sàng")

# %% [markdown]
# ## Cell 2 — Đọc dữ liệu bootstrap
#
# Đọc ba thứ đi cùng nhau cho mỗi board: ảnh phân tích, nhãn pad/pin theo toạ độ
# board, và box linh kiện của lượt 1. Thiếu sidecar `components/` thì không cắt
# được crop — đó là lý do `bootstrap_lead_labels.py` được sửa để xuất nó.

# %%
ROOT = Path(CONFIG["bootstrap_root"])
IMAGES_DIR = ROOT / "images"
LABELS_DIR = ROOT / "labels"
COMPONENTS_DIR = ROOT / "components"

for directory in (IMAGES_DIR, LABELS_DIR, COMPONENTS_DIR):
    if not directory.is_dir():
        raise SystemExit(
            f"Thiếu {directory}.\n"
            "Cần thư mục do scripts/bootstrap_lead_labels.py xuất ra. Nếu thiếu\n"
            "riêng components/ thì bạn đang dùng bản export cũ — chạy lại script."
        )


def read_yolo_labels(path: Path, width: int, height: int):
    """YOLO chuẩn hoá -> box pixel trên khung board."""

    boxes = []
    if not path.is_file():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        boxes.append((
            class_id,
            (cx - bw / 2) * width,
            (cy - bh / 2) * height,
            (cx + bw / 2) * width,
            (cy + bh / 2) * height,
        ))
    return boxes


boards = []
for image_path in sorted(IMAGES_DIR.glob("*.png")):
    stem = image_path.stem
    sidecar = COMPONENTS_DIR / f"{stem}.json"
    if not sidecar.is_file():
        print(f"  bỏ qua {stem}: không có components/{stem}.json")
        continue
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    frame = meta.get("frame", {})
    width, height = int(frame.get("width", 0)), int(frame.get("height", 0))
    if width <= 0 or height <= 0:
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
    boards.append({
        "stem": stem,
        "image": image_path,
        "width": width,
        "height": height,
        "leads": read_yolo_labels(LABELS_DIR / f"{stem}.txt", width, height),
        "components": meta.get("components", []),
    })

print(f"{len(boards)} board")
print(f"  tổng box chân : {sum(len(b['leads']) for b in boards)}")
print(f"  tổng linh kiện: {sum(len(b['components']) for b in boards)}")

# %% [markdown]
# ## Cell 3 — Cổng chặn nhãn giả
#
# Bootstrap sinh ra box **suy ra từ hình học**. Train trên chúng mà chưa sửa thì
# model chỉ học lại chính hình học đó: nó sẽ đạt điểm cao trên giả định của
# chính mình và không biết thêm gì về board thật. Tệ hơn, nó thừa hưởng nguyên
# những lỗi hình học đã đo được — đặt ROI lên cạnh không có chân, chọn sai trục
# của linh kiện gần vuông.
#
# Cổng này không hoàn hảo, nhưng nó bắt được trường hợp rõ ràng nhất: thư mục
# vẫn mang tên `needs_review`, hoặc manifest vẫn ghi trạng thái chưa sửa.

# %%
manifest_path = ROOT / "bootstrap_manifest.json"
looks_unreviewed = "needs_review" in str(ROOT).lower()
if manifest_path.is_file():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") == "PSEUDO_LABELS_NEED_REVIEW":
        looks_unreviewed = True
    print("manifest bootstrap:", {k: manifest.get(k) for k in
                                  ("status", "images", "images_with_boxes", "class_counts")})

if looks_unreviewed and not CONFIG["allow_unreviewed_labels"]:
    raise SystemExit(
        "Dữ liệu trông như chưa được sửa nhãn.\n\n"
        "Box do bootstrap sinh ra là PHỎNG ĐOÁN HÌNH HỌC, không phải sự thật.\n"
        "Train trên chúng chỉ dạy model chép lại hình học đã có sẵn trong\n"
        "aoi_pipeline/solder.py — không thêm được thông tin nào mới, và thừa\n"
        "hưởng luôn các lỗi hình học đã biết.\n\n"
        "Việc cần làm: mở dataset trong LabelImg/CVAT/Roboflow, KÉO box về\n"
        "đúng vùng kim loại, XOÁ box trên chỗ không phải chân, và quan trọng\n"
        "nhất là THÊM box ở những chân hình học bỏ sót — đó chính là thứ model\n"
        "cần học mà hình học không biết.\n\n"
        "Sửa xong: đổi tên thư mục cho khác 'needs_review', hoặc đặt\n"
        "CONFIG['allow_unreviewed_labels'] = True nếu bạn cố ý chạy thử đường ống."
    )
print("Cổng nhãn: OK")

# %% [markdown]
# ## Cell 4 — Chuyển board thành crop
#
# **Đây là cell quan trọng nhất của notebook.**
#
# Hình học cắt crop phải khớp từng con số với `LeadDetectionConfig` mà lượt 2
# dùng lúc chạy. Biên crop là phần bắt buộc: fillet nằm **ngoài** thân linh
# kiện, nên crop dừng ở đúng hộp là giấu mất chính thứ đi tìm.
#
# Quy đổi toạ độ chỉ là một phép trừ — ngược đúng phép cộng mà
# `to_board_coordinates` làm lúc chạy.

# %%
def component_crop_window(box, image_width, image_height):
    """Bản sao của aoi_pipeline.lead_detection.component_crop_window.

    Chép lại thay vì import, vì notebook chạy trên Kaggle không có repo. Hai
    bản này phải khớp — lệch là lệch phân bố train/inference.
    """

    # Clamp EVERY corner into the frame, both directions — the same thing
    # BoundingBox.clamp does. Clamping only one side leaves a box that starts
    # outside the frame and ends inside it, and the margin is then computed from
    # a negative side. A test in the repo compares this function against the
    # library one; keep them identical.
    x1, y1, x2, y2 = (
        min(max(int(round(box[0])), 0), image_width),
        min(max(int(round(box[1])), 0), image_height),
        min(max(int(round(box[2])), 0), image_width),
        min(max(int(round(box[3])), 0), image_height),
    )
    margin = int(round(CONFIG["crop_margin_ratio"] * max(x2 - x1, y2 - y1)))
    margin = max(margin, CONFIG["crop_margin_min_px"])
    return (max(0, x1 - margin), max(0, y1 - margin),
            min(image_width, x2 + margin), min(image_height, y2 + margin))


def clip_to_window(lead_box, window):
    """Box chân theo toạ độ board -> toạ độ crop, cắt theo cửa sổ.

    Giữ lại chỉ khi phần nằm trong crop còn đủ lớn: một mẩu 2 pixel của pad
    hàng xóm lọt vào rìa crop là nhiễu nhãn, không phải mục tiêu.
    """

    class_id, bx1, by1, bx2, by2 = lead_box
    wx1, wy1, wx2, wy2 = window
    ix1, iy1 = max(bx1, wx1), max(by1, wy1)
    ix2, iy2 = min(bx2, wx2), min(by2, wy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    original = (bx2 - bx1) * (by2 - by1)
    kept = (ix2 - ix1) * (iy2 - iy1)
    if original <= 0 or kept / original < 0.55:
        return None
    return (class_id, ix1 - wx1, iy1 - wy1, ix2 - wx1, iy2 - wy1)


CROP_ROOT = Path(CONFIG["work_dir"]) / "crops"
if CROP_ROOT.exists():
    shutil.rmtree(CROP_ROOT)

records = []
empty_crops = []
skipped_small = 0

for board in boards:
    image = cv2.imread(str(board["image"]))
    if image is None:
        continue
    height, width = image.shape[:2]
    for component in board["components"]:
        window = component_crop_window(component["bbox"], width, height)
        wx1, wy1, wx2, wy2 = window
        if (wx2 - wx1) < CONFIG["min_crop_px"] or (wy2 - wy1) < CONFIG["min_crop_px"]:
            skipped_small += 1
            continue
        patch = image[wy1:wy2, wx1:wx2]
        if patch.size == 0:
            continue
        boxes = [clip_to_window(lead, window) for lead in board["leads"]]
        boxes = [b for b in boxes if b is not None]
        record = {
            "board": board["stem"],
            "component_id": component.get("detection_id", ""),
            "component_label": component.get("label", ""),
            "patch": patch,
            "boxes": boxes,
        }
        (records if boxes else empty_crops).append(record)

print(f"crop có chân : {len(records)}")
print(f"crop trống   : {len(empty_crops)}")
print(f"crop quá nhỏ bị bỏ: {skipped_small}")

# Giữ một phần crop trống làm negative: model cần biết "ở đây không có chân",
# nếu không nó sẽ vẽ pad lên mọi thứ. Nhưng đừng để chúng lấn át.
random.shuffle(empty_crops)
keep_empty = int(CONFIG["max_empty_crop_ratio"] * max(1, len(records)))
records.extend(empty_crops[:keep_empty])
print(f"giữ thêm {min(keep_empty, len(empty_crops))} crop trống làm negative")
print(f"tổng cộng: {len(records)} crop")

if not records:
    raise SystemExit(
        "Không sinh được crop nào. Kiểm tra: components/*.json có box không, và\n"
        "labels/*.txt có nằm trong vùng các box đó không."
    )

# %% [markdown]
# ## Cell 5 — Thống kê: crop và pad thật sự to bao nhiêu
#
# Con số quyết định notebook này có ý nghĩa hay không. Pad chỉ vài pixel thì
# không model nào cứu được — đó là giới hạn của khâu chụp ảnh, không phải của
# thuật toán. Xem `Docs/yeu_cau_phan_cung_camera.md`.

# %%
crop_sides = np.array([[r["patch"].shape[1], r["patch"].shape[0]] for r in records])
pad_sides = np.array([
    [b[3] - b[1], b[4] - b[2]] for r in records for b in r["boxes"]
]) if any(r["boxes"] for r in records) else np.zeros((0, 2))

print("kích thước crop (px):")
print(f"  trung vị {np.median(crop_sides[:, 0]):.0f} x {np.median(crop_sides[:, 1]):.0f}"
      f" · nhỏ nhất {crop_sides.min():.0f} · lớn nhất {crop_sides.max():.0f}")
if len(pad_sides):
    shortest = pad_sides.min(axis=1)
    print("kích thước pad (px, cạnh ngắn):")
    print(f"  trung vị {np.median(shortest):.1f} · phân vị 10 {np.percentile(shortest, 10):.1f}"
          f" · nhỏ nhất {shortest.min():.1f}")
    tiny = int((shortest < 8).sum())
    print(f"  pad dưới 8 px cạnh ngắn: {tiny}/{len(shortest)} ({tiny/len(shortest):.1%})")
    if tiny / len(shortest) > 0.5:
        print("\n  CẢNH BÁO: quá nửa số pad dưới 8 px. Ở kích thước đó không còn")
        print("  hình dạng để học. Vấn đề nằm ở độ phân giải khi chụp, không phải")
        print("  ở model — xem Docs/yeu_cau_phan_cung_camera.md trước khi train.")

print("\nsố pad mỗi crop:", dict(Counter(len(r["boxes"]) for r in records)))
print("theo lớp:", dict(Counter(CLASS_NAMES[b[0]] for r in records for b in r["boxes"])))

# %% [markdown]
# ## Cell 6 — Chia tập THEO BOARD
#
# Hai crop cắt từ cùng một board có cùng ánh sáng, cùng lô hàn, thường cùng loại
# linh kiện. Chia ngẫu nhiên theo crop sẽ đặt anh em ruột vào cả train lẫn val,
# và điểm val sẽ đẹp một cách vô nghĩa. Đây đúng là lỗi đã làm con số 6.2 lần
# trước từ 97.65% rơi về 89.9% khi sửa lại.

# %%
by_board = defaultdict(list)
for record in records:
    by_board[record["board"]].append(record)

board_names = sorted(by_board)
random.Random(CONFIG["seed"]).shuffle(board_names)

n_val = max(1, int(round(CONFIG["val_fraction"] * len(board_names))))
n_test = max(1, int(round(CONFIG["test_fraction"] * len(board_names))))
if len(board_names) < 3:
    raise SystemExit(
        f"Chỉ có {len(board_names)} board. Chia theo board cần ít nhất 3.\n"
        "Chụp thêm board — đây là ràng buộc dữ liệu, không phải ràng buộc code."
    )

splits = {
    "val": board_names[:n_val],
    "test": board_names[n_val:n_val + n_test],
    "train": board_names[n_val + n_test:],
}
for name, group in splits.items():
    crops = sum(len(by_board[b]) for b in group)
    print(f"{name:5s}: {len(group):3d} board · {crops:5d} crop")
assert not (set(splits["train"]) & set(splits["val"]) & set(splits["test"]))

# %% [markdown]
# ## Cell 7 — Ghi dataset YOLO

# %%
for split in ("train", "val", "test"):
    (CROP_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
    (CROP_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

written = Counter()
for split, group in splits.items():
    for board_name in group:
        for index, record in enumerate(by_board[board_name]):
            stem = f"{board_name}__{index:04d}"
            patch = record["patch"]
            height, width = patch.shape[:2]
            cv2.imwrite(str(CROP_ROOT / "images" / split / f"{stem}.png"), patch)
            lines = []
            for class_id, x1, y1, x2, y2 in record["boxes"]:
                cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
                bw, bh = (x2 - x1) / width, (y2 - y1) / height
                if bw <= 0 or bh <= 0:
                    continue
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (CROP_ROOT / "labels" / split / f"{stem}.txt").write_text(
                ("\n".join(lines) + "\n") if lines else "", encoding="utf-8"
            )
            written[split] += 1

DATA_YAML = CROP_ROOT / "data.yaml"
DATA_YAML.write_text(yaml.safe_dump({
    "path": str(CROP_ROOT),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": {index: name for index, name in enumerate(CLASS_NAMES)},
}, sort_keys=False, allow_unicode=True), encoding="utf-8")

print(dict(written))
print(DATA_YAML.read_text(encoding="utf-8"))

# %% [markdown]
# ## Cell 8 — Kiểm tra ngược: vẽ lại nhãn lên crop
#
# Đọc ngược file YOLO vừa ghi và vẽ lên ảnh. Nếu box lệch thì lệch ở đây, không
# phải sau 120 epoch. Một lỗi dấu trừ trong phép quy đổi toạ độ trông y hệt một
# model kém.

# %%
import matplotlib.pyplot as plt

samples = sorted((CROP_ROOT / "images" / "train").glob("*.png"))[:8]
if samples:
    figure, axes = plt.subplots(2, 4, figsize=(16, 7))
    for axis, image_path in zip(axes.ravel(), samples):
        patch = cv2.imread(str(image_path))
        height, width = patch.shape[:2]
        label_path = CROP_ROOT / "labels" / "train" / f"{image_path.stem}.txt"
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            x1, y1 = int((cx - bw / 2) * width), int((cy - bh / 2) * height)
            x2, y2 = int((cx + bw / 2) * width), int((cy + bh / 2) * height)
            color = (0, 255, 0) if class_id == 0 else (0, 165, 255)
            cv2.rectangle(patch, (x1, y1), (x2, y2), color, 1)
        axis.imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        axis.set_title(f"{image_path.stem[:22]} · {patch.shape[1]}x{patch.shape[0]}", fontsize=8)
        axis.axis("off")
    plt.tight_layout()
    plt.show()
    print("Nhìn kỹ: box phải nằm trên kim loại. Lệch ở đây thì đừng train tiếp.")

# %% [markdown]
# ## Cell 9 — Train
#
# Augmentation chọn cho bài toán crop, khác hẳn detect trên board rộng:
#
# - `mosaic` **thấp**. Mosaic ghép 4 ảnh lại; trên crop nó tạo ra thứ không bao
#   giờ tồn tại lúc chạy — lượt 2 luôn nhìn đúng một linh kiện.
# - `scale` hẹp. Lượt 2 luôn thấy linh kiện ở cùng một tỉ lệ tương đối, vì crop
#   được cắt theo chính hộp của linh kiện. Dạy nó bất biến tỉ lệ là phí sức.
# - `fliplr` **và** `flipud` đều bật. Linh kiện nằm mọi hướng trên board.
# - `degrees` nhỏ. Linh kiện đặt theo lưới, lệch nhiều là đã thành lỗi lắp.

# %%
model = YOLO(CONFIG["model"])
RUN_DIR = Path(CONFIG["work_dir"]) / "runs"

results = model.train(
    data=str(DATA_YAML),
    imgsz=CONFIG["imgsz"],
    epochs=CONFIG["epochs"],
    batch=CONFIG["batch"],
    patience=CONFIG["patience"],
    seed=CONFIG["seed"],
    deterministic=True,
    optimizer="AdamW",
    lr0=1e-3,
    lrf=1e-2,
    cos_lr=True,
    warmup_epochs=3.0,
    mosaic=0.10,
    close_mosaic=20,
    mixup=0.0,
    copy_paste=0.0,
    scale=0.20,
    degrees=8.0,
    translate=0.08,
    shear=2.0,
    perspective=0.0,
    fliplr=0.5,
    flipud=0.5,
    hsv_h=0.015,
    hsv_s=0.30,
    hsv_v=0.30,
    project=str(RUN_DIR),
    name="lead_detector",
    exist_ok=True,
    plots=True,
    val=True,
)

WEIGHTS_DIR = RUN_DIR / "lead_detector" / "weights"
BEST_PT = WEIGHTS_DIR / "best.pt"
if not BEST_PT.is_file():
    # Ultralytics chỉ ghi best.pt khi vượt best_fitness; nếu không có thì last.pt
    # vẫn là model đã train xong.
    BEST_PT = WEIGHTS_DIR / "last.pt"
print("weights:", BEST_PT)

# %% [markdown]
# ## Cell 10 — Đánh giá trên tập test độc lập
#
# `split="test"`, không phải val. Val đã tham gia chọn checkpoint nên điểm của
# nó lạc quan theo đúng nghĩa.

# %%
best = YOLO(str(BEST_PT))
test_metrics = best.val(
    data=str(DATA_YAML), split="test", imgsz=CONFIG["imgsz"],
    batch=CONFIG["batch"], plots=True,
    project=str(RUN_DIR), name="test_evaluation", exist_ok=True,
)

per_class = {}
for index, name in enumerate(CLASS_NAMES):
    try:
        precision, recall, ap50, ap = test_metrics.box.class_result(index)
        per_class[name] = {"precision": float(precision), "recall": float(recall),
                           "map50": float(ap50), "map50_95": float(ap)}
    except (IndexError, AttributeError):
        per_class[name] = None

print(f"Box mAP50    : {float(test_metrics.box.map50):.4f}")
print(f"Box mAP50-95 : {float(test_metrics.box.map):.4f}")
for name, values in per_class.items():
    print(f"  {name:6s}: {values}")

# %% [markdown]
# ## Cell 11 — Cổng phán quyết
#
# Notebook nói thẳng model này có đáng đưa vào lượt 2 hay không.
#
# Ngưỡng lấy từ mục đích sử dụng, không phải từ thói quen: lượt 2 tồn tại để
# **thay hình học suy ra**. Hình học không bao giờ bỏ sót chân — nó luôn đặt ROI
# ở đâu đó — nên một model có recall thấp hơn thì tệ hơn thứ nó thay thế. Recall
# quan trọng hơn precision: ROI thừa thì người soi mất vài giây, ROI thiếu thì
# mối hàn không bao giờ được kiểm.

# %%
RECALL_GATE = 0.70
MAP_GATE = 0.50

recalls = {n: v["recall"] for n, v in per_class.items() if v}
worst = min(recalls.values()) if recalls else 0.0
passed = worst >= RECALL_GATE and float(test_metrics.box.map50) >= MAP_GATE

print("=" * 66)
if passed:
    print(f"ĐẠT. Recall thấp nhất {worst:.3f} >= {RECALL_GATE}, "
          f"mAP50 {float(test_metrics.box.map50):.3f} >= {MAP_GATE}.")
    print("Model này đáng đưa vào lượt 2. Export rồi nạp qua sidebar.")
else:
    print(f"CHƯA ĐẠT. Recall thấp nhất {worst:.3f} (cần >= {RECALL_GATE}), "
          f"mAP50 {float(test_metrics.box.map50):.3f} (cần >= {MAP_GATE}).")
    print()
    print("Đừng đổ thêm epoch hay đổi sang model to hơn trước khi kiểm ba thứ:")
    print()
    print("1. PAD CÓ ĐỦ TO KHÔNG? Xem lại Cell 5. Dưới 8 px cạnh ngắn thì không")
    print("   còn hình dạng để học — đó là giới hạn khâu chụp ảnh.")
    print("   Xem Docs/yeu_cau_phan_cung_camera.md.")
    print("2. NHÃN ĐÃ ĐƯỢC SỬA CHƯA? Nhãn bootstrap chưa sửa dạy model chép lại")
    print("   hình học cũ. Xem lại Cell 8: box có nằm trên kim loại không?")
    print("3. CÓ ĐỦ BOARD KHÁC NHAU CHƯA? Nhiều crop từ ít board không phải là")
    print("   nhiều dữ liệu. Số BOARD mới là thứ quyết định khả năng tổng quát.")
    print()
    print("Lượt 2 không đạt KHÔNG chặn dây chuyền: bước 5.5 vẫn dùng hình học")
    print("suy ra như hiện nay. Đừng nạp một model tệ hơn thứ nó thay thế.")
print("=" * 66)

# %% [markdown]
# ## Cell 12 — Export và manifest

# %%
ARTIFACTS = Path(CONFIG["artifact_dir"])
ARTIFACTS.mkdir(parents=True, exist_ok=True)

onnx_path = None
try:
    exported = best.export(format="onnx", imgsz=CONFIG["imgsz"], opset=12,
                           dynamic=False, simplify=True)
    onnx_path = Path(exported)
    shutil.copy2(onnx_path, ARTIFACTS / "best.onnx")
    print("ONNX:", ARTIFACTS / "best.onnx")
except Exception as exc:
    print("Export ONNX thất bại:", type(exc).__name__, exc)
    print("Không có ONNX thì app phải nạp .pt, mà .pt chứa pickle nên bị chặn")
    print("cho tới khi người dùng tự xác nhận nguồn tin cậy.")

shutil.copy2(BEST_PT, ARTIFACTS / "best.pt")

manifest = {
    "schema_version": "pcb-lead-detector/1.0",
    "task": "lead_pad_detection_in_component_crop",
    "pipeline_step": "5.5-pass2",
    "class_names": CLASS_NAMES,
    "model": {"base": CONFIG["model"], "weights": BEST_PT.name},
    "input": {
        "imgsz": CONFIG["imgsz"],
        # Lượt 2 CẮT crop bằng đúng những con số này. Lệch là lệch phân bố.
        "crop_margin_ratio": CONFIG["crop_margin_ratio"],
        "crop_margin_min_px": CONFIG["crop_margin_min_px"],
        "min_crop_px": CONFIG["min_crop_px"],
    },
    "metrics": {
        "test_map50": float(test_metrics.box.map50),
        "test_map50_95": float(test_metrics.box.map),
        "per_class": per_class,
        "verdict_gate_passed": bool(passed),
    },
    "data": {
        "boards": len(board_names),
        "crops": sum(written.values()),
        "split_by": "board",
        "split_counts": dict(written),
    },
    "warning": None if passed else
    "Model KHÔNG qua cổng phán quyết. Đừng dùng trong sản xuất.",
}
(ARTIFACTS / "model_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(manifest, indent=2, ensure_ascii=False)[:1200])

shutil.make_archive(str(ARTIFACTS), "zip", ARTIFACTS)
print("\nĐã đóng gói:", str(ARTIFACTS) + ".zip")

# %% [markdown]
# ## Cell 13 — Nạp vào dự án
#
# 1. Tải `pcb_lead_detector_artifacts.zip`.
# 2. Đặt `best.onnx` + `model_manifest.json` vào `models/lead_detector/`.
# 3. Bật lượt 2 trong config:
#
# ```python
# config["lead_detection"] = {
#     "enabled": True,
#     "crop_margin_ratio": 0.35,   # phải khớp manifest
#     "confidence": 0.25,
# }
# ```
#
# 4. Truyền detector vào pipeline:
#
# ```python
# from aoi_pipeline import AOIPipeline
# from aoi_pipeline.detectors import UltralyticsDetector
# from aoi_pipeline.config import ModelDetectorConfig
#
# pipeline = AOIPipeline(
#     config=config,
#     model_path="models/detector/kaggle/ver2/best.onnx",   # lượt 1
#     lead_detector=UltralyticsDetector(                     # lượt 2
#         "models/lead_detector/best.onnx",
#         ModelDetectorConfig(confidence=0.25, imgsz=640),
#     ),
# )
# ```
#
# Từ đó `aoi_pipeline/leads.py` tự lo phần còn lại: chân đo được **thắng** hình
# học suy ra, theo từng chân chứ không theo cả linh kiện, và chân nào lượt 2
# không thấy thì vẫn giữ ROI suy ra. Không có gì khác trong pipeline phải đổi.
#
# ## Ghi chú về tốc độ
#
# Lượt 2 chạy **một lần cho mỗi linh kiện**. Board dày cỡ 1000 linh kiện thì đó
# là 1000 forward pass. Nên:
#
# - Giữ model nhỏ (`yolo11n`/`yolo11s`).
# - Chạy theo lô, đừng chạy từng cái một.
# - Đo thời gian chu kỳ thật trước khi hứa với dây chuyền.
#
# `aoi_pipeline/lead_detection.py` hiện gọi từng crop một cho dễ đọc và dễ test.
# Khi có model thật và đo được nút thắt ở đây thì hãy đổi sang chạy lô — đừng
# tối ưu trước khi đo.
