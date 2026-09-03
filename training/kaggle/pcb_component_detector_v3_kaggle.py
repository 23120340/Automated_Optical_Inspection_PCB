# %% [markdown]
# # Detector thân linh kiện — một lớp `component` (lượt 1, v3)
#
# Khác hẳn detector v1/v2 ở ba điểm, và cả ba đều là lý do bộ này tồn tại:
#
# 1. **Một lớp duy nhất `component`.** v1/v2 học 22 lớp từ Consolidated và chỉ
#    3 lớp đủ dữ liệu (capacitor/resistor/ic = 81% toàn bộ nhãn). Bước 5.5 chỉ
#    cần biết *có linh kiện ở đâu*; việc nó là gì là câu hỏi của 6.1. Bỏ 22 lớp
#    xuống 1 là bỏ đúng phần dữ liệu không đủ để học.
# 2. **Nhãn do người của dự án vẽ**, trên chính tile của dự án — 95 tile,
#    9.486 box, đã soi tay trước khi đóng gói. Detector đang chạy bỏ sót **46%**
#    số box đó, và đó là con số phải vượt.
# 3. **Split đã KHOÁ theo bo vật lý** ngay trong gói dataset. Notebook này
#    **không được chia lại**: tile chồng nhau 256 px nên hai tile cạnh nhau chứa
#    cùng một linh kiện, và chia ngẫu nhiên theo ảnh là rò rỉ train/val.
#
# Dữ liệu: gói bằng `scripts/pack_component_detection_dataset.py`, đã dọn bằng
# `scripts/apply_box_exclusions.py`. Chỉ cần Add Input gói đó —
# `resolve_dataset_root` tự tìm, không phải sửa `dataset_root` bằng tay.

# %%
import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

CONFIG = {
    "seed": 42,
    # Thư mục chứa data.yaml của gói đã pack. Add Input rồi sửa cho khớp.
    "dataset_root": "/kaggle/input/pcb-component-detect-v1/component_detect_v1_tiled",
    "work_dir": "/kaggle/working/pcb_component_detector_v3",
    "artifact_dir": "/kaggle/working/pcb_component_detector_v3_artifacts",

    "model": "yolo26s.pt",
    "resume_from": None,     # "/kaggle/input/.../last.pt" để chạy tiếp một lần train đứt
    "export_from": None,     # "/kaggle/input/.../best.pt" để bỏ qua train, chỉ export

    # ĐO TRÊN LẦN TRAIN THẬT: AutoBatch đã profile chính model này trên T4.
    #
    #   batch @1280   VRAM       kết quả
    #      1          3,87 G
    #      2          8,78 G     <-- AutoBatch chọn, 9,04/14,56 G (62%)
    #      4         16,97 G     OOM (card chỉ có 14,56 G)
    #
    # Tức imgsz=1280 CHẶN CỨNG batch ở 2 trên T4. Đó là một chế độ train thật
    # sự xấu: YOLO dùng BatchNorm khắp nơi, mà thống kê chuẩn hoá lấy từ 2 ảnh
    # thì rất nhiễu. Gradient đã được `nbs=64` cộng dồn bù lại, còn BN thì
    # không có gì bù.
    #
    # Đổi lại, 1024 là kích thước GỐC của tile — mọi ảnh train/valid/test đều
    # 1024x1024 hoặc nhỏ hơn — nên 1280 chỉ là phóng to 1,25 lần: không thêm
    # thông tin, chỉ đẩy box nhỏ vượt ngưỡng ô lưới P3 (stride 8):
    #
    #   imgsz   trung vị   <8px   batch tối đa trên T4
    #    1024     16,2     8,0%     4
    #    1280     20,2     2,8%     2   <-- lần train đầu
    #    1536     24,2     1,0%     1
    #
    # Chọn 1024 + batch 4: đổi 5 điểm phần trăm box dưới ngưỡng để lấy BN gấp
    # đôi số mẫu và epoch nhanh gần gấp đôi (2:50 -> ~1:50). 8,0% vẫn dưới
    # ngưỡng cảnh báo 10% mà chính notebook này đặt ở cell kiểm tra dataset.
    # Muốn quay lại: đặt imgsz 1280 và batch 2.
    #
    # ĐỪNG để `batch: -1`. AutoBatch nhắm mục 60% VRAM nên ở 1024 nó vẫn sẽ
    # chọn 2-3. batch 4 @1024 ăn ~10,9 G (75%) — vừa đủ; nếu OOM thì hạ batch
    # xuống 3, đừng hạ imgsz.
    "imgsz": 1024,
    "epochs": 150,
    "patience": 40,
    "batch": 4,
    "close_mosaic": 25,
    # Mặc định Ultralytics là 300, và lần train thật đầu tiên đã cảnh báo
    # đúng chỗ này: "Dataset images contain up to 358 objects (train=358,
    # val=210), but max_det=300. This mismatch can cap recall and produce
    # invalid validation results."
    # Tile PCB dày đặc linh kiện — trần 300 là trần CỦA PHÉP ĐO, không phải
    # của model: mọi detection thứ 301 trở đi bị vứt trước khi tính recall,
    # nên recall thấp giả mà không có gì báo. 600 phủ mức dày nhất (358) với
    # biên rộng; NMS đắt thêm không đáng kể ở batch này.
    "max_det": 600,

    # Một lớp thì không có class hiếm để cân bằng. Augmentation giữ ở mức của
    # dây chuyền thật: board có thể xoay 180°, không bao giờ lộn gương.
    "fliplr": 0.5,
    "flipud": 0.5,
    "degrees": 10.0,
    # scale 0.25 chứ KHÔNG phải 0.5 mặc định, và mosaic 0.5 chứ không phải 1.0.
    # Đo bằng CHÍNH pipeline augmentation của Ultralytics, trên bộ đã cắt tile,
    # ở imgsz 1280 — tức đúng cấu hình notebook này chạy:
    #
    #   cấu hình                     trung vị   <8px    <4px
    #   không augment hình học         21.5     2.3%   0.2%
    #   scale=0.5, không mosaic        20.5     6.8%   0.0%
    #   mosaic=1.0 + scale=0.5         17.2    12.6%   0.5%   <- mặc định
    #   mosaic=1.0 + scale=0.25        21.6     3.1%   0.4%
    #   mosaic=0.5 + scale=0.25        21.7     2.1%   0.3%   <- đang dùng
    #
    # `scale` là thủ phạm chính chứ không phải mosaic. Mosaic dán 4 ảnh vào
    # canvas 2s×2s ở kích thước GỐC rồi crop về s, nên nó đổi VÙNG NHÌN chứ
    # không co vật thể — đọc code `Mosaic.get_params` để khỏi đoán sai.
    # Đánh đổi: bớt đa dạng augmentation. Chấp nhận được vì dữ liệu đã đa dạng
    # sẵn (3 nguồn, 28 bo, và tiling vừa nhân số ảnh train lên 6 lần).
    "scale": 0.25,
    "mosaic": 0.5,
    "copy_paste": 0.0,       # một lớp, dán chéo không thêm thông tin gì

    # Cổng phán quyết. Detector đang chạy bỏ sót 46% box tay trên chính các tile
    # này, tức recall ~0,54. Model mới không vượt được con số đó thì không có lý
    # do gì để thay.
    # Thành phần train ĐO ĐƯỢC: 1.412 tile RF100 + 436 Winnies + **74 local**
    # = 1.922 ảnh. Tức miền đích chỉ chiếm **3,9%** dữ liệu học, trong khi
    # valid/test là **100%** local. Model đang tối ưu cho miền mà nó không
    # bao giờ bị chấm điểm.
    # Nhân bản tile local trong DANH SÁCH train (không nhân file trên đĩa):
    # 6 lần đưa local lên ~19% train. Đánh đổi thật: chỉ có 74 ảnh local
    # DUY NHẤT, nhân lên không tạo thông tin mới và có thể học thuộc chúng.
    # Đặt 1 để tắt. Nếu bật, so recall trên valid với chính 74 ảnh đó —
    # chênh lớn nghĩa là đang học thuộc chứ không phải học hình dạng.
    "local_oversample": 6,
    "local_prefix": "local_component_bodies__",

    "gate_recall": 0.70,
    "gate_map50": 0.60,
    "incumbent_recall_on_hand_boxes": 0.54,
}

random.seed(CONFIG["seed"])


def resolve_dataset_root(configured: str) -> Path:
    """Tự tìm gói dataset thay vì bắt sửa `dataset_root` bằng tay mỗi lần.

    Kaggle đặt Input ở chỗ khác nhau tuỳ cách bạn add. Lần train đầu CONFIG ghi
    `/kaggle/input/pcb-component-detect-v1/...` nhưng chỗ thật là
    `/kaggle/input/datasets/<user>/<slug>/...`, nên phải sửa tay giữa phiên.
    """
    wanted = Path(configured)
    if (wanted / "data.yaml").is_file():
        return wanted

    # Tìm theo ĐÚNG tên thư mục trong CONFIG, không quét bừa: nếu có nhiều gói
    # cùng attach thì bắt chọn tay chứ không đoán.
    name = wanted.name
    found = []
    for base in (Path("/kaggle/input"), Path.cwd()):
        if not base.is_dir():
            continue
        for depth in ("", "*/", "*/*/", "*/*/*/"):
            found += [p.parent for p in base.glob(f"{depth}{name}/data.yaml")]
    found = sorted({p.resolve() for p in found})

    if len(found) == 1:
        print(f"dataset_root trong CONFIG không tồn tại; tự tìm thấy: {found[0]}")
        return found[0]
    if not found:
        raise SystemExit(
            f"không thấy thư mục '{name}' có data.yaml ở /kaggle/input.\n"
            "Add Input gói dataset, rồi sửa CONFIG['dataset_root'] cho khớp."
        )
    raise SystemExit(
        f"thấy {len(found)} gói tên '{name}' — không đoán:\n  "
        + "\n  ".join(str(p) for p in found)
        + "\nSửa CONFIG['dataset_root'] trỏ đúng một cái."
    )


DATASET = resolve_dataset_root(CONFIG["dataset_root"])
WORK = Path(CONFIG["work_dir"])
ARTIFACTS = Path(CONFIG["artifact_dir"])
print("dataset:", DATASET)

# %% [markdown]
# ## 0. Cổng phần cứng
#
# Kaggle đôi khi gán P100 thay vì T4/L4. Ở `imgsz=1536` khác biệt đó là vài giờ,
# và phát hiện ra sau khi train xong thì đã muộn. Chặn ngay.

# %%
try:
    import torch
except ImportError:  # pragma: no cover - chỉ chạy trên Kaggle
    raise SystemExit("không có torch — chọn accelerator GPU trong Settings")

if not torch.cuda.is_available():
    raise SystemExit(
        "KHÔNG có GPU. Settings -> Accelerator -> GPU T4 x2 (hoặc L4). "
        "Train imgsz=1536 trên CPU là không khả thi."
    )
gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu} ({vram:.1f} GB)")
if "P100" in gpu:
    print(
        "  CẢNH BÁO: P100 không có Tensor Core cho fp16 hiệu quả; ở imgsz 1536 "
        "nó chậm hơn T4 đáng kể. Cân nhắc đổi accelerator rồi chạy lại."
    )

# %% [markdown]
# ## 1. Hợp đồng dữ liệu
#
# Ba thứ phải đúng trước khi tiêu một giờ GPU nào. Mỗi cái từng là một lỗi thật
# trong dự án này, nên đều được kiểm chứ không tin.

# %%
data_yaml = DATASET / "data.yaml"
if not data_yaml.is_file():
    raise SystemExit(f"không thấy {data_yaml} — gói dataset thiếu data.yaml")

yaml_text = data_yaml.read_text(encoding="utf-8")
print(yaml_text)

# (a) đúng MỘT lớp, tên đúng
if "nc: 1" not in yaml_text or "component" not in yaml_text:
    raise SystemExit("data.yaml không phải bộ một lớp 'component'")

# (b) cả ba split đều có ảnh — split rỗng nghĩa là packer đã bị ép ghi
counts = {}
for split in ("train", "valid", "test"):
    images = sorted((DATASET / split / "images").glob("*"))
    labels = sorted((DATASET / split / "labels").glob("*.txt"))
    if not images:
        raise SystemExit(f"split {split} rỗng")
    if len(images) != len(labels):
        raise SystemExit(f"{split}: {len(images)} ảnh nhưng {len(labels)} file nhãn")
    counts[split] = (len(images), sum(
        len([l for l in p.read_text().splitlines() if l.strip()]) for p in labels))
    print(f"{split:6s} {counts[split][0]:4d} ảnh  {counts[split][1]:6d} box")

# (c) KHÔNG có ảnh nào xuất hiện ở hai split — đây là kiểm tra rò rỉ
stems = {s: {p.stem for p in (DATASET / s / "images").glob("*")} for s in counts}
for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
    overlap = stems[a] & stems[b]
    if overlap:
        raise SystemExit(f"RÒ RỈ: {len(overlap)} ảnh có ở cả {a} và {b}: {sorted(overlap)[:3]}")
print("\nkhông có ảnh trùng giữa các split")

manifest_path = DATASET / "pack_manifest.json"
pack_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
if pack_manifest:
    readiness = pack_manifest.get("readiness", {})
    print(f"gói: {readiness.get('verified_target_groups')} bo đích, "
          f"ready={readiness.get('ready_to_pack')}")

# %% [markdown]
# ## 2. Cỡ box — vì sao bộ này đã được CẮT TILE
#
# Cell này **đo** thay vì giả định. Nhãn YOLO là toạ độ chuẩn hoá, nhân với
# `imgsz` là ra kích thước box model thực sự nhìn thấy. Ngưỡng đọc: **stride nhỏ
# nhất của YOLO là 8** — dưới đó box chiếm chưa tới một ô lưới P3.
#
# **Gói chưa cắt tile có vấn đề thật, và nó nằm ở một nguồn duy nhất.** Đo theo
# nguồn ở `imgsz=1536`:
#
# | nguồn | box | `<8px` | trung vị |
# |---|---:|---:|---:|
# | rf100 | 36.673 | **29,7%** | 10,8 px |
# | nhãn của dự án | 9.486 | **2,7%** | 21,0 px |
# | winnies | 7.095 | 0,0% | 26,2 px |
#
# Nhãn người trong dự án vẽ hoàn toàn khoẻ. RF100 mới là chỗ hỏng: ảnh gốc rộng
# 504–5985 px bị letterbox về 1536, có ảnh co tới 3,9 lần, nên linh kiện 30 px
# thành 7,7 px.
#
# **Cách sửa là cắt tile, không phải nâng imgsz.** `scripts/tile_packed_dataset.py`
# cắt ảnh train quá lớn thành tile 1024 (chồng 256 px), giữ nguyên độ phân giải
# gốc. Kết quả đo được:
#
# | | ảnh train | box | trung vị @1536 | `<8px` |
# |---|---:|---:|---:|---:|
# | chưa cắt | 318 | 51.314 | 13,6 px | **21,2%** |
# | **đã cắt** | 1.922 | 94.905 | **25,4 px** | **1,0%** |
#
# Số box tăng ~1,85 lần vì tile chồng nhau: một linh kiện nằm trong vùng chồng
# lấn xuất hiện ở cả hai tile. Đó là hành vi đúng của tiling chồng lấn — linh
# kiện bị đường cắt xén ở tile này thì còn nguyên ở tile kia — và tất cả đều
# nằm trong `train` nên không đụng tới thước đo.
#
# `valid`/`test` **không bị cắt**: chúng là tile 1024 của dự án, đã khoá theo bo,
# và là thứ dùng để chấm điểm.

# %%
import numpy as np

shorts = []
for split in ("train", "valid", "test"):
    for label in (DATASET / split / "labels").glob("*.txt"):
        for line in label.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                shorts.append(min(float(parts[3]), float(parts[4])))
shorts = np.array(shorts)
print(f"{len(shorts)} box\n")
print(f"{'imgsz':>6s} {'p05':>7s} {'p25':>7s} {'trung vị':>9s} {'<8px':>8s}")
for size in (1024, 1280, 1536, 1792):
    px = shorts * size
    print(f"{size:6d} {np.percentile(px,5):7.1f} {np.percentile(px,25):7.1f} "
          f"{np.median(px):9.1f} {100*(px<8).mean():7.1f}%")
print(
    "\nĐọc bảng: cột '<8px' là tỉ lệ box dưới một ô lưới P3."
    "\nTrên bộ ĐÃ CẮT TILE, imgsz 1280 để lại ~2,7% — so với 21% của bộ chưa"
    "\ncắt ở imgsz 1536. Nếu con số bạn thấy ở đây gần 20%, bạn đang trỏ vào"
    "\ngói CHƯA cắt: chạy scripts/tile_packed_dataset.py trước."
)
if (np.array(shorts) * CONFIG["imgsz"] < 8).mean() > 0.10:
    raise SystemExit(
        "Hơn 10% box dưới 8px ở imgsz đang đặt. Gói này nhiều khả năng CHƯA được "
        "cắt tile — chạy scripts/tile_packed_dataset.py rồi trỏ dataset_root vào "
        "bản đã cắt. Train tiếp ở trạng thái này là tiêu giờ GPU cho box mà model "
        "không có chỗ để hồi quy."
    )

# %% [markdown]
# ## 2b. Cân lại thành phần train
#
# `valid`/`test` là **100% tile của dự án**, còn `train` chỉ có **3,9%**. Đó là
# hệ quả trực tiếp của một quy tắc đúng — dữ liệu công khai chỉ được vào train,
# vì nó có bản augment trùng lặp và chồng nguồn với PCB-DSLR nên chấm điểm trên
# nó cho số đẹp giả. Nhưng hệ quả là model tối ưu cho miền nó không bị chấm.
#
# Cách chữa rẻ nhất: **nhân bản dòng trong danh sách train**, không nhân file.
# Ultralytics chấp nhận `train:` trỏ tới một file `.txt` liệt kê đường dẫn ảnh,
# nên việc này không cần ghi vào thư mục input (vốn chỉ đọc trên Kaggle).

# %%
if CONFIG["local_oversample"] > 1:
    train_images = sorted((DATASET / "train" / "images").glob("*"))
    lines, local_count = [], 0
    for path in train_images:
        repeats = 1
        if path.name.startswith(CONFIG["local_prefix"]):
            repeats = CONFIG["local_oversample"]
            local_count += 1
        lines.extend([str(path.resolve())] * repeats)

    listing = WORK / "train_oversampled.txt"
    listing.parent.mkdir(parents=True, exist_ok=True)
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # data.yaml riêng trong /kaggle/working, dùng đường dẫn TUYỆT ĐỐI để khỏi
    # phụ thuộc vào chỗ Ultralytics giải đường dẫn tương đối.
    data_yaml = WORK / "data_oversampled.yaml"
    data_yaml.write_text(
        f"train: {listing}\n"
        f"val: {(DATASET / 'valid' / 'images').resolve()}\n"
        f"test: {(DATASET / 'test' / 'images').resolve()}\n"
        "nc: 1\n"
        "names: ['component']\n",
        encoding="utf-8",
    )
    share = CONFIG["local_oversample"] * local_count / len(lines)
    print(f"{local_count} tile local x{CONFIG['local_oversample']} "
          f"+ {len(train_images) - local_count} ảnh công khai")
    print(f"danh sách train: {len(lines)} dòng, local chiếm {100*share:.1f}% "
          f"(trước khi nhân: {100*local_count/len(train_images):.1f}%)")
    print("data.yaml dùng để train:", data_yaml)
else:
    print("không nhân bản; train giữ nguyên thành phần gốc")

# %% [markdown]
# ## 3. Train
#
# Khi `resume_from` được đặt, Ultralytics **đọc lại cấu hình từ chính
# checkpoint** — mọi tham số epoch/imgsz/augmentation ở CONFIG bị bỏ qua. Đó là
# hành vi đúng (chạy tiếp phải giống hệt lần đầu) nhưng dễ gây bất ngờ.

# %%
from ultralytics import YOLO

WORK.mkdir(parents=True, exist_ok=True)
run_dir = None

if CONFIG["export_from"]:
    print(f"BỎ QUA TRAIN — export thẳng từ {CONFIG['export_from']}")
    model = YOLO(CONFIG["export_from"])
elif CONFIG["resume_from"]:
    print(f"CHẠY TIẾP từ {CONFIG['resume_from']}")
    model = YOLO(CONFIG["resume_from"])
    results = model.train(resume=True)
    run_dir = Path(model.trainer.save_dir) if getattr(model, "trainer", None) is not None else None
else:
    model = YOLO(CONFIG["model"])
    results = model.train(
        data=str(data_yaml),
        epochs=CONFIG["epochs"],
        imgsz=CONFIG["imgsz"],
        batch=CONFIG["batch"],
        patience=CONFIG["patience"],
        close_mosaic=CONFIG["close_mosaic"],
        max_det=CONFIG["max_det"],
        seed=CONFIG["seed"],
        project=str(WORK),
        name="train",
        exist_ok=True,
        fliplr=CONFIG["fliplr"],
        flipud=CONFIG["flipud"],
        degrees=CONFIG["degrees"],
        scale=CONFIG["scale"],
        mosaic=CONFIG["mosaic"],
        copy_paste=CONFIG["copy_paste"],
    )
    run_dir = Path(model.trainer.save_dir) if getattr(model, "trainer", None) is not None else None

# `hasattr(model, "trainer")` LUÔN True (Ultralytics đặt self.trainer = None
# trong __init__), nên phải dùng getattr(...) is not None. Đã dính lỗi này.
if run_dir is not None:
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    # Sau resume, best.pt có thể KHÔNG được ghi lại: Ultralytics chỉ ghi khi
    # fitness vượt best_fitness đọc từ checkpoint, mà đỉnh có thể đã đạt trước
    # lúc đứt. Lùi về last.pt kèm thông báo thay vì chết ở cell export.
    if best.is_file():
        weights = best
    elif last.is_file():
        print("KHÔNG có best.pt (thường gặp sau resume) — dùng last.pt")
        weights = last
    else:
        raise SystemExit(f"không thấy trọng số nào trong {run_dir/'weights'}")
    print("trọng số:", weights)
    model = YOLO(str(weights))
else:
    weights = Path(CONFIG["export_from"])

# %% [markdown]
# ## 4. Đo trên valid và test
#
# Đo **cả hai**, và báo cả hai. Chỉ báo val là cách dễ nhất để tự lừa mình: val
# tham gia vào việc chọn epoch tốt nhất, test thì không.

# %%
metrics = {}
for split in ("val", "test"):
    res = model.val(data=str(data_yaml), split=split, imgsz=CONFIG["imgsz"],
                    max_det=CONFIG["max_det"], verbose=False)
    metrics[split] = {
        "map50": float(res.box.map50),
        "map50_95": float(res.box.map),
        "precision": float(res.box.mp),
        "recall": float(res.box.mr),
    }
    m = metrics[split]
    print(f"{split:5s} mAP50={m['map50']:.4f}  mAP50-95={m['map50_95']:.4f}  "
          f"P={m['precision']:.4f}  R={m['recall']:.4f}")

gap = metrics["val"]["map50"] - metrics["test"]["map50"]
print(f"\nchênh val-test mAP50: {gap:+.4f}")
if gap > 0.15:
    print(
        "  CẢNH BÁO: chênh lớn. Với 28 bo mà tile chồng nhau 256 px, đây là dấu "
        "\n  hiệu model học thuộc danh tính bo chứ không học hình dạng linh kiện."
    )

# %% [markdown]
# ## 4b. Con số vừa in chính xác đến đâu?
#
# `valid` có **3 bo vật lý**, `test` có **3 bo**. Đó mới là cỡ mẫu thật: 1.300
# và 640 box nghe thì nhiều, nhưng box trong cùng một bo không độc lập với nhau
# — cùng ánh sáng, cùng loại linh kiện, cùng nhà sản xuất. Và một bo duy nhất
# chi phối gần một nửa thước đo (board017 = 49,5% box của valid, board009 =
# 42,2% của test).
#
# Đo trên chính log lần train đầu: mAP50 của valid dao động 0,127–0,292 qua 18
# epoch; sau khi trừ xu hướng học, nhiễu còn **sd = 0,038**, tức khoảng ±0,077
# ở mức 95%. Toàn bộ tiến bộ sau 18 epoch chỉ là +0,071 — **dải nhiễu rộng hơn
# cả tín hiệu**. Hệ quả trực tiếp: chọn `best.pt` = lấy max qua 150 lần rút
# thăm nhiễu, nên con số val của `best.pt` bị thổi lên khoảng **+0,10 mAP50**
# so với chất lượng thật. Đừng bao giờ báo cáo số val của `best.pt`.
#
# Cell này vì thế báo recall **theo từng bo**, kèm khoảng tin cậy bootstrap lấy
# mẫu lại ở mức **bo** chứ không phải mức box. Recall cộng gộp được chính xác
# (= tổng TP / tổng GT) nên bootstrap này đúng về mặt số học; mAP thì không
# cộng gộp tuyến tính nên chỉ báo biên độ giữa các bo.

# %%
import re as _re
from collections import defaultdict as _dd


def _canon_board(stem: str):
    """pcb15 và pcb_dslr_015 là CÙNG một bo vật lý dưới hai tên."""
    m = _re.match(r"local_component_bodies__pcb(?:_dslr)?_?(\d+)__", stem)
    return f"board{int(m.group(1)):03d}" if m else None


per_board = {}
for split, folder in (("val", "valid"), ("test", "test")):
    groups = _dd(list)
    for p in sorted((DATASET / folder / "images").glob("*")):
        board = _canon_board(p.stem)
        if board:
            groups[board].append(p)

    rows = []
    for board, paths in sorted(groups.items()):
        listing = WORK / f"_board_{split}_{board}.txt"
        listing.write_text("\n".join(str(q.resolve()) for q in paths) + "\n",
                           encoding="utf-8")
        board_yaml = WORK / f"_board_{split}_{board}.yaml"
        board_yaml.write_text(
            f"train: {listing}\nval: {listing}\nnc: 1\nnames: ['component']\n",
            encoding="utf-8")
        r = model.val(data=str(board_yaml), split="val", imgsz=CONFIG["imgsz"],
                      max_det=CONFIG["max_det"], verbose=False, plots=False)
        gt = sum(
            len([l for l in (DATASET / folder / "labels" / f"{q.stem}.txt")
                 .read_text(encoding="utf-8").splitlines() if l.strip()])
            for q in paths
        )
        rows.append({
            "board": board, "tiles": len(paths), "gt": gt,
            "recall": float(np.nan_to_num(r.box.mr)),
            "map50": float(np.nan_to_num(r.box.map50)),
        })
    per_board[split] = rows

    print(f"--- {split}: {len(rows)} bo vật lý ---")
    for row in rows:
        print(f"  {row['board']}  {row['tiles']:2d} tile  {row['gt']:5d} box "
              f"({100*row['gt']/sum(x['gt'] for x in rows):4.1f}% thước đo)  "
              f"recall={row['recall']:.3f}  mAP50={row['map50']:.3f}")

# %%
_rng = np.random.default_rng(CONFIG["seed"])
board_ci = {}
for split, rows in per_board.items():
    tp = np.array([row["recall"] * row["gt"] for row in rows])
    gt = np.array([float(row["gt"]) for row in rows])
    draw = _rng.integers(0, len(rows), size=(20000, len(rows)))
    pooled = tp[draw].sum(axis=1) / np.maximum(gt[draw].sum(axis=1), 1.0)
    lo, hi = (float(v) for v in np.percentile(pooled, [2.5, 97.5]))
    board_ci[split] = (lo, hi)
    maps = [row["map50"] for row in rows]
    print(f"{split:5s} recall = {metrics[split]['recall']:.3f}   "
          f"khoảng tin cậy 95% khi lấy mẫu lại theo BO: {lo:.3f} – {hi:.3f} "
          f"(rộng {hi - lo:.3f})")
    print(f"      mAP50 từng bo trải từ {min(maps):.3f} đến {max(maps):.3f}")

if board_ci["test"][1] - board_ci["test"][0] > 0.10:
    print(
        "\nKHOẢNG TIN CẬY RỘNG HƠN 0,10. Đây KHÔNG phải lỗi của model — đây là\n"
        "giới hạn của thước đo: 3 bo thì không đo chính xác hơn được. Mọi so\n"
        "sánh A/B có chênh lệch nhỏ hơn khoảng này đều KHÔNG kết luận được.\n"
        "Muốn kết luận thì phải k-fold theo bo trên cả 28 bo, không phải chỉ 3."
    )

# %% [markdown]
# ## 5. Cổng phán quyết
#
# Lệ của repo: model mới phải **hơn thứ nó thay thế**, đo được, trước khi
# promote. Thứ nó thay thế ở đây là detector 22 lớp đang chạy, và con số của nó
# trên chính các tile này là **bỏ sót 46% box tay** — tức recall ≈ 0,54.

# %%
test = metrics["test"]
test_lo, test_hi = board_ci["test"]
checks = {
    f"recall test >= {CONFIG['gate_recall']}": test["recall"] >= CONFIG["gate_recall"],
    f"mAP50 test >= {CONFIG['gate_map50']}": test["map50"] >= CONFIG["gate_map50"],
    # Phán trên CẬN DƯỚI, không phán trên con số điểm. Với 3 bo, recall đo được
    # 0,70 có thể là recall thật 0,56 hoặc 0,84 — xác suất một model recall thật
    # 0,65 lọt qua cổng 0,70 là 24%. Đòi cận dưới vượt incumbent nghĩa là: kể cả
    # khi 3 bo này là 3 bo may nhất, model vẫn hơn thứ nó thay thế.
    f"CẬN DƯỚI recall ({test_lo:.3f}) > detector đang chạy "
    f"({CONFIG['incumbent_recall_on_hand_boxes']})":
        test_lo > CONFIG["incumbent_recall_on_hand_boxes"],
}
for name, ok in checks.items():
    print(f"  {'ĐẠT ' if ok else 'KHÔNG'}  {name}")

print(f"\nrecall test = {test['recall']:.3f}, khoảng tin cậy theo bo "
      f"{test_lo:.3f} – {test_hi:.3f}")
if test["recall"] > CONFIG["incumbent_recall_on_hand_boxes"] >= test_lo:
    print("  Con số điểm hơn incumbent nhưng CẬN DƯỚI thì không — chưa đủ bằng\n"
          "  chứng để kết luận model mới tốt hơn. Đây là giới hạn của 3 bo, và\n"
          "  cách duy nhất để gỡ là k-fold theo bo trên cả 28 bo.")

verdict = all(checks.values())
print(f"\nPHÁN QUYẾT: {'ĐẠT — đáng promote' if verdict else 'CHƯA ĐẠT'}")
if not verdict:
    print(
        "Không đạt thì ĐỪNG promote. Ba chỗ nên xem trước khi đổ lỗi cho model:\n"
        "  1. nhóm box lồng nhau (40 box bao box khác, cái tệ nhất bao 55) — xem\n"
        "     box_exclusions.json mục kept_after_review;\n"
        "  2. imgsz — thử 1280 (batch phải hạ về 2) nếu recall thấp ở box nhỏ;\n"
        "  3. số bo: 28 bo là ít, và test chỉ 3 bo nên khoảng tin cậy rất rộng —\n"
        "     xem mục 4b để biết chênh lệch bao nhiêu mới là chênh lệch thật."
    )

# %% [markdown]
# ## 6. Export ONNX + manifest
#
# Ultralytics tự truyền `dynamo=False` cho torch >= 2.4 nên đường export này
# không dính lỗi `onnxscript` mà các notebook classifier từng dính.
#
# Manifest ghi **số đo thật của chính trọng số được export**, không phải con số
# chép từ CONFIG — bản trước từng mô tả sai artifact vì chép CONFIG.

# %%
ARTIFACTS.mkdir(parents=True, exist_ok=True)
onnx_path = model.export(format="onnx", imgsz=CONFIG["imgsz"], dynamic=False, simplify=True)
onnx_path = Path(onnx_path)
shutil.copy2(onnx_path, ARTIFACTS / "best.onnx")
shutil.copy2(weights, ARTIFACTS / "best.pt")

# Đọc shape THẬT từ file ONNX. Export dynamic=False khoá cứng kích thước, và app
# tự đọc lại con số này — mô tả sai ở đây là nạp model vào app thì nổ.
import onnx

graph = onnx.load(str(ARTIFACTS / "best.onnx"))
shape = [d.dim_value or d.dim_param for d in
         graph.graph.input[0].type.tensor_type.shape.dim]
output_shape = [d.dim_value or d.dim_param for d in
                graph.graph.output[0].type.tensor_type.shape.dim]
print("ONNX input :", shape)
print("ONNX output:", output_shape)

import hashlib

digest = hashlib.sha256((ARTIFACTS / "best.onnx").read_bytes()).hexdigest()
manifest = {
    "schema_version": "pcb-component-detector/1.0",
    "task": "component_body_detection",
    "pipeline_step": "4 (lượt 1)",
    "model_family": "ultralytics-yolo",
    "base_model": CONFIG["model"],
    "class_names": ["component"],
    "class_map": {"0": "component"},
    "input": {"shape": shape, "imgsz": CONFIG["imgsz"], "dynamic": False},
    "output": {"shape": output_shape},
    "onnx": {"sha256": digest, "bytes": (ARTIFACTS / "best.onnx").stat().st_size},
    "metrics": metrics,
    "verdict": {
        "passed": verdict,
        "checks": {k: bool(v) for k, v in checks.items()},
        "incumbent_recall_on_hand_boxes": CONFIG["incumbent_recall_on_hand_boxes"],
    },
    "dataset": {
        "name": "component_detect_v1",
        "images": {k: v[0] for k, v in counts.items()},
        "boxes": {k: v[1] for k, v in counts.items()},
        "split_unit": "bo vật lý (khoá trong gói, notebook KHÔNG chia lại)",
        "pack_manifest": pack_manifest.get("readiness", {}),
        "curation": "7 box loại bằng scripts/apply_box_exclusions.py sau khi soi tay",
    },
    "train_config": {k: CONFIG[k] for k in
                     ("seed", "imgsz", "epochs", "patience", "batch", "close_mosaic",
                      "max_det", "local_oversample",
                      "fliplr", "flipud", "degrees", "scale", "mosaic", "copy_paste")},
    "known_limits": [
        "Chỉ 28 bo vật lý, test 11 ảnh — khoảng tin cậy của mọi con số ở trên rất rộng.",
        "Ảnh train là CVL PCB-DSLR + RF100 + Winnies, KHÔNG phải camera dây chuyền. "
        "Phải fine-tune trên ảnh thật trước khi tin số đo ở production.",
        "Bộ đã cắt tile: ~2,7% box dưới 8px ở imgsz 1280 (trước khi cắt là 21% ở "
        "1536). Recall ở nhóm box nhỏ vẫn nên xem riêng, đừng đọc mAP tổng.",
        "Tile chồng lấn nhân số box train lên ~1,85 lần; đó là cùng linh kiện nhìn "
        "từ hai khung, không phải dữ liệu mới.",
    ],
}
(ARTIFACTS / "model_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nghi", ARTIFACTS / "model_manifest.json")
print(json.dumps(manifest["metrics"], indent=2))
print(f"\nTải về {ARTIFACTS} rồi copy best.onnx + model_manifest.json vào "
      "models/active/detector/ (giữ bản cũ ở models/archive/).")
