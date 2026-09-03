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
# `scripts/apply_box_exclusions.py`. Add Input gói đó rồi sửa `dataset_root`.

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
    "dataset_root": "/kaggle/input/pcb-component-detect-v1/component_detect_v1",
    "work_dir": "/kaggle/working/pcb_component_detector_v3",
    "artifact_dir": "/kaggle/working/pcb_component_detector_v3_artifacts",

    "model": "yolo26s.pt",
    "resume_from": None,     # "/kaggle/input/.../last.pt" để chạy tiếp một lần train đứt
    "export_from": None,     # "/kaggle/input/.../best.pt" để bỏ qua train, chỉ export

    # 1536 chứ không phải 1024. Đo trên chính gói này (53.254 box, xem cell 2):
    # ở imgsz 1024 có 42,8% box nhỏ hơn 8 px; 1536 hạ xuống 20,9%; 1792 còn
    # 14,3%. Stride nhỏ nhất của YOLO là 8, nên dưới ngưỡng đó box chiếm chưa
    # tới một ô lưới P3.
    # 1536 là điểm dừng, KHÔNG phải điểm tối ưu — nó chỉ là chỗ đánh đổi giữa
    # VRAM và tỉ lệ box quá nhỏ. Nếu recall ở nhóm box nhỏ kém, đòn bẩy đúng
    # KHÔNG phải nâng tiếp imgsz (1792 chỉ bớt được 6 điểm phần trăm và tốn hơn
    # 36% compute) mà là CẮT TILE ảnh công khai, xem ghi chú ở cell 2.
    "imgsz": 1536,
    "epochs": 150,
    "patience": 40,
    "batch": -1,             # auto theo VRAM; hạ imgsz xuống 1280 nếu OOM
    "close_mosaic": 25,

    # Một lớp thì không có class hiếm để cân bằng. Augmentation giữ ở mức của
    # dây chuyền thật: board có thể xoay 180°, không bao giờ lộn gương.
    "fliplr": 0.5,
    "flipud": 0.5,
    "degrees": 10.0,
    "scale": 0.5,
    "mosaic": 1.0,
    "copy_paste": 0.0,       # một lớp, dán chéo không thêm thông tin gì

    # Cổng phán quyết. Detector đang chạy bỏ sót 46% box tay trên chính các tile
    # này, tức recall ~0,54. Model mới không vượt được con số đó thì không có lý
    # do gì để thay.
    "gate_recall": 0.70,
    "gate_map50": 0.60,
    "incumbent_recall_on_hand_boxes": 0.54,
}

random.seed(CONFIG["seed"])
DATASET = Path(CONFIG["dataset_root"])
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
    raise SystemExit(f"không thấy {data_yaml} — Add Input gói dataset rồi sửa dataset_root")

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
# ## 2. Cỡ box — vì sao `imgsz=1536`, và giới hạn của lựa chọn đó
#
# Cell này **đo** thay vì giả định. Nhãn YOLO là toạ độ đã chuẩn hoá, nên nhân
# với `imgsz` là ra kích thước box model thực sự nhìn thấy.
#
# Ngưỡng đọc kết quả: **stride nhỏ nhất của YOLO là 8**. Box dưới 8 px ở đầu vào
# chiếm chưa tới một ô lưới P3 — model gần như không có chỗ để hồi quy nó.
#
# **Đọc con số cho đúng, vì nó không đẹp.** Ở 1536 vẫn còn khoảng **21% box dưới
# 8 px**. Lý do không phải nhãn xấu: 94% ảnh trong gói là ảnh công khai (RF100
# rộng 504–5985 px, Winnies 1536×2048) bị letterbox về 1536, nên linh kiện nhỏ
# của chúng co lại rất nhiều. Tile 1024 của dự án thì ngược lại — chúng được
# phóng LÊN 1,5 lần.
#
# Nên đừng đọc "21%" là "một phần năm nhãn vô dụng". Nó là "một phần năm nhãn
# đến từ ảnh board rộng, và ở độ phân giải đó chúng vốn đã nhỏ". YOLO26 có
# **STAL**, cơ chế ép ít nhất 4 anchor cho vật thể dưới 8 px, và đó chính là lý
# do dự án giữ YOLO26 thay vì đổi kiến trúc.
#
# **Nếu recall ở nhóm box nhỏ kém, đòn bẩy đúng là cắt tile ảnh công khai** (như
# `scripts/tile_test_images.py` đã làm cho ảnh của dự án) chứ không phải nâng
# tiếp imgsz. Packer hiện xuất nguyên cảnh công khai; cắt chúng thành tile 1024
# sẽ đưa phân bố cỡ box của phần công khai về gần phần local.

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
    "\nĐọc bảng: cột '<8px' là tỉ lệ box rơi xuống dưới một ô lưới P3."
    "\nĐo trên gói hiện tại, 1536 vẫn để lại ~21% box dưới ngưỡng, và phần lớn"
    "\nsố đó đến từ ẢNH CÔNG KHAI cỡ lớn bị letterbox — không phải từ tile của"
    "\ndự án. Nâng imgsz lên 1792 chỉ bớt được ~6 điểm phần trăm mà tốn thêm"
    "\n36% compute; cắt tile ảnh công khai mới là cách sửa gốc."
)

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
    res = model.val(data=str(data_yaml), split=split, imgsz=CONFIG["imgsz"], verbose=False)
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
# ## 5. Cổng phán quyết
#
# Lệ của repo: model mới phải **hơn thứ nó thay thế**, đo được, trước khi
# promote. Thứ nó thay thế ở đây là detector 22 lớp đang chạy, và con số của nó
# trên chính các tile này là **bỏ sót 46% box tay** — tức recall ≈ 0,54.

# %%
test = metrics["test"]
checks = {
    f"recall test >= {CONFIG['gate_recall']}": test["recall"] >= CONFIG["gate_recall"],
    f"mAP50 test >= {CONFIG['gate_map50']}": test["map50"] >= CONFIG["gate_map50"],
    f"recall > detector đang chạy ({CONFIG['incumbent_recall_on_hand_boxes']})":
        test["recall"] > CONFIG["incumbent_recall_on_hand_boxes"],
}
for name, ok in checks.items():
    print(f"  {'ĐẠT ' if ok else 'KHÔNG'}  {name}")

verdict = all(checks.values())
print(f"\nPHÁN QUYẾT: {'ĐẠT — đáng promote' if verdict else 'CHƯA ĐẠT'}")
if not verdict:
    print(
        "Không đạt thì ĐỪNG promote. Ba chỗ nên xem trước khi đổ lỗi cho model:\n"
        "  1. nhóm box lồng nhau (40 box bao box khác, cái tệ nhất bao 55) — xem\n"
        "     box_exclusions.json mục kept_after_review;\n"
        "  2. imgsz — thử 1792 nếu recall thấp ở box nhỏ;\n"
        "  3. số bo: 28 bo là ít, và test chỉ 11 ảnh nên khoảng tin cậy rất rộng."
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
                      "fliplr", "flipud", "degrees", "scale", "mosaic", "copy_paste")},
    "known_limits": [
        "Chỉ 28 bo vật lý, test 11 ảnh — khoảng tin cậy của mọi con số ở trên rất rộng.",
        "Ảnh train là CVL PCB-DSLR + RF100 + Winnies, KHÔNG phải camera dây chuyền. "
        "Phải fine-tune trên ảnh thật trước khi tin số đo ở production.",
        "Đo trên gói: ở imgsz 1536 vẫn còn ~21% box dưới 8px (dưới một ô lưới P3), "
        "phần lớn đến từ ảnh công khai cỡ lớn bị letterbox. Recall ở nhóm box nhỏ "
        "phải xem riêng, đừng đọc mAP tổng.",
    ],
}
(ARTIFACTS / "model_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nghi", ARTIFACTS / "model_manifest.json")
print(json.dumps(manifest["metrics"], indent=2))
print(f"\nTải về {ARTIFACTS} rồi copy best.onnx + model_manifest.json vào "
      "models/active/detector/ (giữ bản cũ ở models/archive/).")
