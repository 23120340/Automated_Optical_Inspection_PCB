# %% [markdown]
# # AOI PCB — Solder defect classifier v2 (public bootstrap → camera fine-tune)
#
# Notebook có hai chế độ, chọn bằng đúng một khóa `run_mode` trong CONFIG:
#
# - **`public_bootstrap` (mặc định, dùng ngay):** tự lấy SolDef_AI, chỉ giữ polygon của
#   từng mối hàn ở ảnh 45°, chia `train / val / calibration / public proxy holdout` theo
#   ảnh gốc, rồi xuất cặp `best.onnx` + `model_manifest.json` để nạp thử vào app. Manifest
#   luôn ghi `bootstrap_only`, không giả vờ đây là bằng chứng production.
# - **`camera_finetune` (sau khi gắn camera):** public data chỉ replay ở train; val,
#   calibration và locked-test bắt buộc là crop AOI thật, được khóa theo board/lot. Chỉ
#   chế độ này có quyền PASS production quality gate.
#
# SolDef_AI chứa hai bài toán khác nhau trong cùng thư mục LabelMe. Top-view
# `good/no_good` khoanh cả linh kiện và pad là **placement**, không phải joint. Notebook
# loại phần đó; chỉ dùng dataset_2 45° (`good`, `exc_solder`, `poor_solder`, `spike`),
# trong đó mỗi polygon là một joint. Trộn hai phần là một nguyên nhân trực tiếp làm đánh
# giá classifier có vẻ cao nhưng sai với ROI mà bước 6.2 nhận.
#
# Public bootstrap dùng profile an toàn `good / defect`. Dataset công khai không có
# ground truth `unknown` đúng nghĩa (background/wrong crop cùng camera), nên notebook
# không bịa lớp này từ noise hoặc ảnh toàn board. Khi có camera, mode fine-tune tự chuyển
# sang `good / defect / unknown`.
#
# Cả hai chế độ đều khử exact/perceptual duplicate, tách val/calibration/evaluation,
# kiểm ONNX parity trên crop thật của miền đang đánh giá, và lưu checkpoint PyTorch để
# tiếp tục fine-tune sau này.
#
# ## Khi đã có dữ liệu camera AOI
#
# Export ROI bằng đúng hình học app đang chạy:
#
# ```powershell
# .\.venv\Scripts\python.exe scripts\export_solder_dataset.py <boards> `
#   --output <dataset> --model models\active\detector\best.onnx `
#   --split-pins --joints-only --overlays
# ```
#
# Sau đó điền nhãn và thêm tối thiểu các cột:
# `crop_path, defect_class, board_id, capture_id, dataset_source, roi_kind`.
# Nên có thêm `lot_id, camera_id, session_id, source_image, label_status, reviewer_id,
# split, preprocess_id`. `split=test` phải do người phụ trách dữ liệu khóa trước khi chạy.
#
# > Lưu ý runtime: profile `joint_gate_v2` là bộ lọc một chiều. `good` đủ tự tin có thể
# > đồng thuận với rule để PASS; `defect` và `unknown` sẽ đi conflict/review vì runtime
# > hiện vẫn giữ subtype vật lý (`insufficient`, `bridge`, ...). Đây là hành vi an toàn.

# %%
CONFIG = {
    "seed": 42,
    "work_dir": "/kaggle/working/pcb_solder_defect_v2",
    # Hiện tại giữ public_bootstrap. Sau khi có camera chỉ cần đổi thành camera_finetune.
    "run_mode": "public_bootstrap",
    # "auto" chọn joint_bootstrap_v2 ở public và joint_gate_v2 ở camera.
    "taxonomy_profile": "auto",
    # Để trống sẽ tự dò solder_dataset.csv trong /kaggle/input. Nên ghi path rõ ràng
    # khi có nhiều dataset camera được attach. Public mode không cần file này.
    "local_manifests": [],
    # Manifest public bổ sung (nếu có) phải cùng schema canonical.
    "public_manifests": [],
    # SolDef_AI được ưu tiên từ Add Input; nếu chưa attach, KaggleHub tự tải/attach.
    "auto_download_public_sources": True,
    "public_crop_padding_ratio": 0.15,
    # Optional Roboflow YOLO export. Để rỗng thì chỉ dùng SolDef_AI.
    "roboflow_solder_root": "",
    # Camera mode: test phải được chỉ định sẵn. Public mode luôn tự group-split proxy.
    "require_explicit_locked_test": True,
    "allow_research_auto_split": False,
    "split_fractions": {
        "train": 0.65,
        "val": 0.15,
        "calibration": 0.10,
        "test": 0.10,
    },
    "split_search_attempts": 500,
    "min_short_edge": 24,
    "near_duplicate_hamming": 4,
    "near_duplicate_color_distance": 0.035,
    "minimum_local_label_retained_ratio": 0.80,
    "maximum_invalid_image_ratio": 0.02,
    "input_size": 128,
    "letterbox_value": 114,
    "model_name": "mobilenet_v3_small",  # hoặc convnext_tiny để thử nghiệm GPU
    "pretrained": True,
    # Camera mode: attach ZIP/checkpoint từ public bootstrap rồi điền path .pt tại đây.
    "bootstrap_checkpoint": "",
    "batch_size": 64,
    "num_workers": 2,
    "epochs": 30,
    "freeze_epochs": 2,
    "patience": 7,
    "head_lr": 3e-4,
    "backbone_lr": 8e-5,
    "weight_decay": 1e-4,
    "label_smoothing": 0.01,
    "max_class_weight": 5.0,
    "max_val_argmax_escape": 0.10,
    # Runtime luôn áp floor model_accept_probability=0.80.
    "runtime_accept_floor": 0.80,
    "review_threshold": 0.50,
    # Fallback có thể dùng để thử app khi public CI quá rộng; vẫn được ghi provisional.
    "public_proxy_escape_target": 0.10,
    "escape_target": 0.01,
    "invalid_good_accept_target": 0.01,
    "ci_alpha": 0.05,
    "minimum_auto_decision_precision_lcb": 0.90,
    # Locked-test được làm giàu defect/unknown để đo safety nên tổng review_rate của nó
    # không đại diện lưu lượng line. Capacity gate dùng good_review (normal production).
    "max_good_review_rate": 0.20,
    # 0 escape trên 299 defect độc lập mới có one-sided 95% upper bound xấp xỉ 1%.
    "minimum_test_defects": 299,
    "minimum_test_good": 100,
    "minimum_test_unknown": 100,
    "minimum_test_boards": 20,
    "minimum_test_defect_boards": 299,
    "minimum_test_unknown_boards": 299,
    "minimum_test_good_boards": 149,
    "minimum_groups_per_class_eval": 2,
    "minimum_macro_f1": 0.80,
    "minimum_per_class_f1": 0.60,
    "minimum_defect_recall": 0.95,
    "maximum_false_reject": 0.02,
    "minimum_subtype_defect_recall": 0.90,
    "subtype_escape_target": 0.05,
    "minimum_subtype_boards": 59,
    "bootstrap_iterations": 1000,
    "onnx_opset": 18,
    "onnx_parity_atol": 1e-3,
    "onnx_parity_samples": 64,
    # Giá trị này phải giống recipe đã dùng lúc export và lúc app chạy.
    "production_preprocess_id": "aoi-ui-production-v1",
}

# %% [markdown]
# ## 1. Môi trường và contract runtime
#
# Cell cài dependency tải nguồn/export/parity nếu image Kaggle chưa có. Không được bỏ qua
# ONNX Runtime: export thành công nhưng runtime cho kết quả khác vẫn là artifact hỏng.

# %%
import hashlib
import importlib.util
import json
import math
import random
import re
import shutil
import subprocess
import sys
import warnings
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

required_packages = {
    "onnx": "onnx",
    "onnxruntime": "onnxruntime",
    "onnxscript": "onnxscript",
}
if bool(CONFIG["auto_download_public_sources"]) or str(CONFIG["roboflow_solder_root"]).strip():
    required_packages.update({"kagglehub": "kagglehub", "yaml": "pyyaml"})
missing_packages = [
    package for module, package in required_packages.items()
    if importlib.util.find_spec(module) is None
]
if missing_packages:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *missing_packages]
    )

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from scipy.stats import beta
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


def json_ready(value):
    """Convert numpy/non-finite values to strict RFC-compatible JSON values."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def json_text(value):
    return json.dumps(json_ready(value), ensure_ascii=False, indent=2, allow_nan=False)

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

SEED = int(CONFIG["seed"])
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

WORK_DIR = Path(CONFIG["work_dir"])
REPORT_DIR = WORK_DIR / "reports"
CANDIDATE_DIR = WORK_DIR / "candidate"
BOOTSTRAP_DIR = WORK_DIR / "public_bootstrap"
PRODUCTION_DIR = WORK_DIR / "production"
# Mỗi Run All bắt đầu sạch. Chỉ xóa đúng work_dir đã khai báo dưới /kaggle/working và
# các ZIP có tên cố định; như vậy run FAIL không thể để lại best.onnx từ run PASS cũ.
kaggle_working = Path("/kaggle/working").resolve()
resolved_work_dir = WORK_DIR.resolve()
if not resolved_work_dir.is_relative_to(kaggle_working) or resolved_work_dir == kaggle_working:
    raise RuntimeError(f"work_dir phải là thư mục con cụ thể của {kaggle_working}")
if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)
for stale_zip in (
    Path("/kaggle/working/pcb_solder_defect_v2_candidate_artifacts.zip"),
    Path("/kaggle/working/pcb_solder_defect_v2_public_bootstrap_artifacts.zip"),
    Path("/kaggle/working/pcb_solder_defect_v2_production_artifacts.zip"),
):
    if stale_zip.is_file():
        stale_zip.unlink()
for directory in (WORK_DIR, REPORT_DIR, CANDIDATE_DIR, BOOTSTRAP_DIR, PRODUCTION_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def select_training_device():
    if not torch.cuda.is_available():
        warnings.warn("Không có CUDA; notebook vẫn chạy nhưng train sẽ chậm.")
        return torch.device("cpu")
    device_index = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_index)
    gpu_arch = f"sm_{capability[0]}{capability[1]}"
    compiled_arches = torch.cuda.get_arch_list()
    # Không so tuple capability với chuỗi sm_XX: wheel có thể chạy bằng PTX forward
    # compatibility. Launch kernel thật mới là phép kiểm đáng tin cậy.
    try:
        probe = torch.ones(1, device=f"cuda:{device_index}")
        probe.add_(1)
        torch.cuda.synchronize(device_index)
    except Exception as exc:
        raise RuntimeError(
            f"GPU {torch.cuda.get_device_name(device_index)} ({gpu_arch}) không chạy "
            f"được kernel PyTorch {torch.__version__}; compiled={compiled_arches}. "
            "Hãy chọn GPU T4 x2 hoặc image Kaggle mới hơn."
        ) from exc
    print("GPU:", torch.cuda.get_device_name(device_index), "capability", gpu_arch)
    print("PyTorch CUDA architectures:", compiled_arches or "không công bố")
    return torch.device(f"cuda:{device_index}")


DEVICE = select_training_device()
print("Work dir:", WORK_DIR)

# %%
RUN_MODE_POLICIES = {
    "public_bootstrap": {
        "taxonomy_profile": "joint_bootstrap_v2",
        "primary_source_kind": "public",
        "train_only_source_kinds": {"local"},
        "require_local": False,
        "allow_auto_split": True,
        "production_allowed": False,
        "evaluation_role": "public_proxy_holdout",
        "evaluation_domain": "public_solder_joint_crops",
    },
    "camera_finetune": {
        "taxonomy_profile": "joint_gate_v2",
        "primary_source_kind": "local",
        "train_only_source_kinds": {"public"},
        "require_local": True,
        "allow_auto_split": bool(CONFIG["allow_research_auto_split"]),
        "production_allowed": True,
        "evaluation_role": "camera_locked_test",
        "evaluation_domain": "aoi_camera_production_roi",
    },
}
RUN_MODE = str(CONFIG["run_mode"]).strip().lower()
if RUN_MODE not in RUN_MODE_POLICIES:
    raise ValueError(f"run_mode không hỗ trợ: {RUN_MODE}")
MODE_POLICY = RUN_MODE_POLICIES[RUN_MODE]
PRIMARY_SOURCE_KIND = MODE_POLICY["primary_source_kind"]
TRAIN_ONLY_SOURCE_KINDS = set(MODE_POLICY["train_only_source_kinds"])
EVALUATION_ROLE = MODE_POLICY["evaluation_role"]
EVALUATION_DOMAIN = MODE_POLICY["evaluation_domain"]

PROFILE_CONFIGS = {
    "joint_bootstrap_v2": {
        "scope": "joint",
        "class_names": ["good", "defect"],
        "good_label": "good",
        "advisory_non_good": True,
    },
    "joint_gate_v2": {
        "scope": "joint",
        "class_names": ["good", "defect", "unknown"],
        "good_label": "good",
        "advisory_non_good": True,
    },
    "joint_subtype_v2": {
        "scope": "joint",
        "class_names": [
            "good", "insufficient", "excess", "cold", "missing_solder"
        ],
        "good_label": "good",
        "advisory_non_good": False,
    },
    "component_placement_v2": {
        "scope": "component",
        "class_names": ["ok", "missing", "tombstone", "shifted", "wrong_polarity"],
        "good_label": "ok",
        "advisory_non_good": False,
    },
}
TAXONOMY_PROFILE = str(CONFIG["taxonomy_profile"]).strip().lower()
if TAXONOMY_PROFILE == "auto":
    TAXONOMY_PROFILE = str(MODE_POLICY["taxonomy_profile"])
if TAXONOMY_PROFILE not in PROFILE_CONFIGS:
    raise ValueError(f"Taxonomy profile không hỗ trợ: {TAXONOMY_PROFILE}")

PROFILE = PROFILE_CONFIGS[TAXONOMY_PROFILE]
CLASS_NAMES = list(PROFILE["class_names"])


def validate_mode_taxonomy(run_mode, selected_profile, expected_profile, class_names):
    """Keep deployment eligibility tied to the mode's audited class contract."""
    if selected_profile != expected_profile:
        raise ValueError(
            f"run_mode={run_mode} bắt buộc taxonomy_profile={expected_profile}; "
            f"không được override thành {selected_profile}."
        )
    if run_mode == "camera_finetune" and "unknown" not in class_names:
        raise ValueError(
            "camera_finetune bắt buộc class unknown từ wrong-crop/OOD camera thật."
        )


validate_mode_taxonomy(
    RUN_MODE, TAXONOMY_PROFILE, str(MODE_POLICY["taxonomy_profile"]), CLASS_NAMES
)
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
GOOD_LABEL = PROFILE["good_label"]
GOOD_INDEX = CLASS_TO_INDEX[GOOD_LABEL]
UNKNOWN_INDEX = CLASS_TO_INDEX.get("unknown")

JOINT_SUBTYPES = {"insufficient", "excess", "bridge", "cold", "missing_solder"}
SINGLE_JOINT_SUBTYPES = {"insufficient", "excess", "cold", "missing_solder"}
EXPECTED_RAW_SUBTYPES = (
    sorted(SINGLE_JOINT_SUBTYPES)
    if PROFILE["scope"] == "joint"
    else ["missing", "tombstone", "shifted", "wrong_polarity"]
)
COMPONENT_LABELS = {
    "shift_component": "shifted",
    "shifted": "shifted",
    "misalignment": "shifted",
    "component_misalignment": "shifted",
    "missing_component": "missing",
    "missing": "missing",
    "tombstone": "tombstone",
    "wrong_polarity": "wrong_polarity",
    "ok": "ok",
    "good": "ok",
}
GOOD_ALIASES = {"good", "ok", "normal", "pass", "no_defect", "defect_free"}
UNKNOWN_ALIASES = {
    "unknown", "background", "not_a_joint", "invalid_roi", "wrong_crop",
    "false_crop", "out_of_distribution", "ood",
}
AMBIGUOUS_LABELS = {
    "", "nan", "none", "unlabeled", "unreviewed", "ambiguous", "uncertain",
    "poor_solder", "no_good",
}
SPLIT_ALIASES = {
    "validation": "val", "valid": "val", "val": "val",
    "calib": "calibration", "cal": "calibration", "calibration": "calibration",
    "locked_test": "test", "locked-test": "test", "test": "test",
    "train": "train", "training": "train", "ignore": "ignore", "": "",
}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def normalize_label(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def map_target_label(row):
    """Map only explicit semantics; ambiguous/unreviewed is ignored, never `unknown`."""
    raw = normalize_label(row.get("defect_class", ""))
    status = normalize_label(row.get("label_status", "verified_legacy"))
    roi_kind = normalize_label(row.get("roi_kind", "joint"))
    source_kind = normalize_label(row.get("source_kind", "local"))
    label_scope = normalize_label(row.get("label_scope", roi_kind))
    public_annotation = source_kind == "public" and status == "source_annotation"
    if status not in {"verified", "approved", "adjudicated", "verified_legacy"} and not public_annotation:
        return None, "label_not_verified"
    if raw in AMBIGUOUS_LABELS:
        return None, "ambiguous_or_missing_label"

    if PROFILE["scope"] == "joint":
        if roi_kind != "joint":
            return None, "non_production_crop_routed_out_of_joint_scope"
        if (raw in COMPONENT_LABELS and raw not in GOOD_ALIASES) or raw in {
            "shift_component", "missing_component", "tombstone", "wrong_polarity"
        }:
            return None, "component_label_routed_out_of_joint_scope"
        if raw == "bridge":
            return None, "bridge_is_pair_rule_not_single_joint_classifier"
        if label_scope in {"invalid", "background", "not_a_joint", "wrong_crop"}:
            if raw not in UNKNOWN_ALIASES:
                return None, "invalid_scope_must_be_labeled_unknown"
        if TAXONOMY_PROFILE in {"joint_bootstrap_v2", "joint_gate_v2"}:
            if raw in GOOD_ALIASES:
                return "good", "mapped"
            if raw in UNKNOWN_ALIASES and "unknown" in CLASS_NAMES:
                return "unknown", "mapped"
            if raw in SINGLE_JOINT_SUBTYPES or raw in {"defect", "ng", "fail", "spike"}:
                return "defect", "mapped"
        else:
            if raw in GOOD_ALIASES:
                return "good", "mapped"
            if raw in SINGLE_JOINT_SUBTYPES:
                return raw, "mapped"
            if raw in UNKNOWN_ALIASES:
                return None, "unknown_not_in_subtype_profile"
    else:
        if roi_kind != "body":
            return None, "joint_label_routed_out_of_component_scope"
        if label_scope in {"invalid", "background", "not_a_component", "wrong_crop"}:
            return None, "invalid_component_scope_requires_separate_unknown_profile"
        mapped = COMPONENT_LABELS.get(raw)
        if mapped in CLASS_NAMES:
            return mapped, "mapped"

    return None, "unmapped_label"


print("Run mode:", RUN_MODE)
print("Evaluation:", EVALUATION_ROLE, "| production evidence:", MODE_POLICY["production_allowed"])
print("Profile:", TAXONOMY_PROFILE, PROFILE)
print("Class order (phải trùng logits):", CLASS_NAMES)

# %% [markdown]
# ## 2. Lấy và chuẩn hóa dữ liệu công khai
#
# Nguồn mặc định là
# [SolDef_AI](https://www.kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection)
# (GPL-3.0; paper: https://doi.org/10.3390/jmmp8030117). Notebook ưu tiên dataset đã
# **Add Input**; nếu chưa có thì `kagglehub.dataset_download()` tự attach/tải nguồn public.
#
# Adapter SolDef không map nhãn theo tên một cách mù quáng. JSON có `no_good`, hoặc một
# polygon `good` duy nhất bao cả component/pads, thuộc dataset_1 placement và bị route out.
# JSON dataset_2 có polygon từng joint (nhãn chuyên biệt hoặc từ hai joint trở lên) mới
# được cắt ROI. `poor_solder` được chuẩn hóa thành `insufficient`; `spike` được giữ riêng
# trong báo cáo subtype rồi gộp vào `defect` ở profile bootstrap.
#
# Nguồn Roboflow `Solder_detection_type2` là tùy chọn vì provenance yếu hơn. Export v1
# dạng YOLO, Add Input/upload lên Kaggle, rồi điền `CONFIG['roboflow_solder_root']`.
# Adapter chỉ nhận box joint đã định nghĩa rõ; bridge/short vẫn thuộc luật cặp joint, còn
# residue/charred/misalignment bị route out thay vì ép nhãn.

# %%
PUBLIC_SOURCE_SPECS = [
    {
        "name": "soldef_ai",
        "enabled": True,
        "required": RUN_MODE == "public_bootstrap",
        "provider": "kaggle",
        "handle": "mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection",
        "preferred_roots": [
            "/kaggle/input/soldef-ai-pcb-dataset-for-defect-detection",
            "/kaggle/input/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection",
        ],
        "adapter": "soldef_labelme_joint_v2",
        "license": "GPL-3.0",
        "homepage": "https://www.kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection",
        "citation": "https://doi.org/10.3390/jmmp8030117",
        "roi_semantics": "45_degree_single_solder_joint_polygon",
    },
    {
        "name": "roboflow_solder_type2",
        "enabled": bool(str(CONFIG["roboflow_solder_root"]).strip()),
        "required": False,
        "provider": "roboflow_manual_export",
        "handle": "pcb-qbyda/solder_detection_type2/dataset/1",
        "preferred_roots": [str(CONFIG["roboflow_solder_root"]).strip()],
        "adapter": "roboflow_yolo_joint_v2",
        "license": "CC-BY-4.0",
        "homepage": "https://universe.roboflow.com/pcb-qbyda/solder_detection_type2/dataset/1",
        "citation": "Roboflow Universe project page",
        "roi_semantics": "audited_joint_defect_bbox",
    },
]

SOLDEF_JOINT_LABELS = {
    "good": "good",
    "exc_solder": "excess",
    "poor_solder": "insufficient",
    "spike": "spike",
}
SOLDEF_PUBLISHED_SCOPE_COUNTS = {
    "component_placement": 228,
    "solder_joint": 200,
}
ROBOFLOW_JOINT_LABELS = {
    "good": "good",
    "normal": "good",
    "excessive_solder": "excess",
    "excess_solder": "excess",
    "insufficient_solder": "insufficient",
    "no_solder": "missing_solder",
    "missing_solder": "missing_solder",
    "cold_solder": "cold",
    "cold_joint": "cold",
}
_ROBOFLOW_FILE = re.compile(
    r"^(?P<stem>.+?)_(?:jpg|jpeg|png|bmp|tif|tiff)\.rf\.[0-9a-f]{6,}$",
    re.IGNORECASE,
)


def public_source_group(image_path):
    """Group every annotation/augmentation from one public origin image together."""
    stem = Path(image_path).stem
    match = _ROBOFLOW_FILE.match(stem)
    if match:
        stem = match.group("stem")
    stem = re.sub(r"(?:__|_)(?:aug|flip|rot|copy)[-_]?\d*$", "", stem, flags=re.I)
    return normalize_label(stem) or hashlib.sha256(str(image_path).encode()).hexdigest()[:16]


def _existing_public_root(spec):
    candidates = [Path(value) for value in spec.get("preferred_roots", []) if value]
    slug = str(spec.get("handle", "")).split("/")[-1]
    input_root = Path("/kaggle/input")
    if slug:
        candidates.extend([input_root / slug, input_root / "datasets" / slug])
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve(), "attached_input"
    if input_root.is_dir() and slug:
        normalized_slug = normalize_label(slug)
        for candidate in input_root.iterdir():
            if candidate.is_dir() and normalize_label(candidate.name) == normalized_slug:
                return candidate.resolve(), "attached_input"
    return None, None


def resolve_public_source(spec):
    root, acquisition = _existing_public_root(spec)
    if root is not None:
        return root, acquisition
    if (
        spec.get("provider") == "kaggle"
        and CONFIG["auto_download_public_sources"]
        and spec.get("handle")
    ):
        try:
            import kagglehub

            downloaded = Path(kagglehub.dataset_download(spec["handle"])).resolve()
            if downloaded.is_dir():
                return downloaded, "kagglehub_download"
        except Exception as exc:
            if spec.get("required"):
                raise RuntimeError(
                    f"Không tải được {spec['handle']} qua KaggleHub. Bật Internet hoặc "
                    "Add Input dataset này trước khi Run All."
                ) from exc
            warnings.warn(f"Bỏ qua nguồn public tùy chọn {spec['name']}: {exc}")
    if spec.get("required"):
        raise FileNotFoundError(
            f"Thiếu nguồn bắt buộc {spec['name']}. Add Input {spec.get('handle')} hoặc "
            "bật CONFIG['auto_download_public_sources']."
        )
    return None, None


def _find_labelme_image(annotation_path, payload, source_root):
    raw_name = str(payload.get("imagePath") or "").replace("\\", "/")
    basename = raw_name.rsplit("/", 1)[-1]
    candidates = []
    if basename:
        candidates.append(annotation_path.parent / basename)
    candidates.extend(annotation_path.with_suffix(ext) for ext in IMAGE_EXTENSIONS)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    if basename:
        matches = list(Path(source_root).rglob(basename))
        if matches:
            return matches[0].resolve()
    return None


def _labelme_bbox(shape):
    points = shape.get("points")
    if not isinstance(points, list) or len(points) < 2:
        return None
    try:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
    except (TypeError, ValueError, IndexError):
        return None
    shape_type = normalize_label(shape.get("shape_type") or "polygon")
    if shape_type == "circle" and len(points) == 2:
        cx, cy = xs[0], ys[0]
        radius = math.hypot(xs[1] - cx, ys[1] - cy)
        if radius <= 0:
            return None
        return (cx - radius, cy - radius, cx + radius, cy + radius)
    box = (min(xs), min(ys), max(xs), max(ys))
    return box if box[2] > box[0] and box[3] > box[1] else None


def soldef_annotation_task(shapes):
    """Separate placement dataset_1 from joint dataset_2 before mapping `good`."""
    usable = [shape for shape in shapes if _labelme_bbox(shape) is not None]
    labels = [normalize_label(shape.get("label")) for shape in usable]
    if any(label in {"exc_solder", "poor_solder", "spike"} for label in labels):
        return "solder_joint"
    if "no_good" in labels:
        return "component_placement"
    if len(usable) >= 2 and labels and set(labels) <= {"good"}:
        return "solder_joint"
    if labels and set(labels) <= {"good", "no_good"}:
        return "component_placement"
    return "unknown_scope"


def validate_soldef_scope_counts(task_counts):
    """Validate the published 428-file release against the paper's two task sizes."""
    observed = {str(key): int(value) for key, value in task_counts.items()}
    expected = dict(SOLDEF_PUBLISHED_SCOPE_COUNTS)
    annotated_files = int(sum(observed.values()))
    applicable = annotated_files == int(sum(expected.values()))
    passed = None if not applicable else all(
        observed.get(name, 0) == count for name, count in expected.items()
    ) and not observed.get("unknown_scope", 0)
    report = {
        "applicable_to_published_428_file_release": applicable,
        "annotated_files": annotated_files,
        "expected_task_files": expected,
        "observed_task_files": dict(sorted(observed.items())),
        "passed": passed,
    }
    if applicable and not passed:
        raise RuntimeError(
            "Không tách được đúng 228 placement + 200 solder-joint files của bản "
            f"SolDef_AI đã công bố: {report}"
        )
    return report


def read_soldef_joint_records(source_root):
    records, task_counts, raw_labels = [], Counter(), Counter()
    invalid_json = missing_annotated_images = 0
    for annotation_path in sorted(Path(source_root).rglob("*.json")):
        try:
            payload = json.loads(annotation_path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            invalid_json += 1
            continue
        shapes = payload.get("shapes")
        if not isinstance(shapes, list) or not shapes:
            continue
        task = soldef_annotation_task(shapes)
        task_counts[task] += 1
        for shape in shapes:
            raw_labels[normalize_label(shape.get("label"))] += 1
        if task != "solder_joint":
            continue
        image_path = _find_labelme_image(annotation_path, payload, source_root)
        if image_path is None:
            missing_annotated_images += 1
            continue
        for shape_index, shape in enumerate(shapes):
            original_label = normalize_label(shape.get("label"))
            canonical_label = SOLDEF_JOINT_LABELS.get(original_label)
            bbox = _labelme_bbox(shape)
            if canonical_label is None or bbox is None:
                continue
            records.append({
                "image": image_path,
                "bbox": bbox,
                "defect_class": canonical_label,
                "original_label": original_label,
                "group": public_source_group(image_path),
                "shape_index": shape_index,
                "annotation_path": annotation_path,
                "annotation_layout": "labelme",
                "dataset_source": "soldef_ai",
            })
    scope_validation = validate_soldef_scope_counts(task_counts)
    if scope_validation["applicable_to_published_428_file_release"] and missing_annotated_images:
        raise RuntimeError(
            "Bản SolDef_AI 428-file thiếu ảnh sidecar cho "
            f"{missing_annotated_images} annotation solder-joint."
        )
    return records, {
        "task_files": dict(sorted(task_counts.items())),
        "scope_count_validation": scope_validation,
        "raw_label_counts": dict(sorted(raw_labels.items())),
        "invalid_json": int(invalid_json),
        "missing_annotated_images": int(missing_annotated_images),
    }


def _yolo_class_names(source_root):
    import yaml

    for yaml_path in sorted(Path(source_root).rglob("*.yaml")):
        try:
            payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, yaml.YAMLError):
            continue
        raw_names = payload.get("names") if isinstance(payload, dict) else None
        if isinstance(raw_names, list):
            return {index: str(value) for index, value in enumerate(raw_names)}
        if isinstance(raw_names, dict):
            try:
                return {int(index): str(value) for index, value in raw_names.items()}
            except (TypeError, ValueError):
                continue
    return {}


def read_roboflow_joint_records(source_root):
    names = _yolo_class_names(source_root)
    if not names:
        raise RuntimeError("Roboflow YOLO export thiếu data.yaml/names; không đoán class index.")
    records, raw_labels, routed_out = [], Counter(), Counter()
    for label_path in sorted(Path(source_root).rglob("*.txt")):
        if normalize_label(label_path.parent.name) not in {"label", "labels"}:
            continue
        image_path = None
        for extension in IMAGE_EXTENSIONS:
            candidate = label_path.parent.parent / "images" / f"{label_path.stem}{extension}"
            if candidate.is_file():
                image_path = candidate.resolve()
                break
        if image_path is None:
            continue
        for line_index, line in enumerate(label_path.read_text(encoding="utf-8", errors="replace").splitlines()):
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                class_index = int(float(parts[0]))
                yolo_box = tuple(float(value) for value in parts[1:5])
            except ValueError:
                continue
            original_label = normalize_label(names.get(class_index, str(class_index)))
            raw_labels[original_label] += 1
            canonical_label = ROBOFLOW_JOINT_LABELS.get(original_label)
            if canonical_label is None:
                routed_out[original_label] += 1
                continue
            records.append({
                "image": image_path,
                "yolo": yolo_box,
                "defect_class": canonical_label,
                "original_label": original_label,
                "group": public_source_group(image_path),
                "shape_index": line_index,
                "annotation_path": label_path,
                "annotation_layout": "yolo",
                "dataset_source": "roboflow_solder_type2",
            })
    return records, {
        "raw_label_counts": dict(sorted(raw_labels.items())),
        "routed_out_counts": dict(sorted(routed_out.items())),
    }


PUBLIC_READERS = {
    "soldef_labelme_joint_v2": read_soldef_joint_records,
    "roboflow_yolo_joint_v2": read_roboflow_joint_records,
}


def materialize_public_crops(records, output_root, padding_ratio=0.15):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows, failures = [], Counter()
    for record_index, record in enumerate(records):
        image = cv2.imread(str(record["image"]), cv2.IMREAD_COLOR)
        if image is None:
            failures["decode_failed"] += 1
            continue
        height, width = image.shape[:2]
        if record.get("yolo") is not None:
            cx, cy, box_width, box_height = record["yolo"]
            x1 = (cx - box_width / 2.0) * width
            y1 = (cy - box_height / 2.0) * height
            x2 = (cx + box_width / 2.0) * width
            y2 = (cy + box_height / 2.0) * height
        else:
            x1, y1, x2, y2 = record["bbox"]
        pad = float(padding_ratio) * max(float(x2 - x1), float(y2 - y1))
        left, top = max(0, int(math.floor(x1 - pad))), max(0, int(math.floor(y1 - pad)))
        right, bottom = min(width, int(math.ceil(x2 + pad))), min(height, int(math.ceil(y2 + pad)))
        if right - left < 8 or bottom - top < 8:
            failures["crop_too_small"] += 1
            continue
        crop = image[top:bottom, left:right]
        token = hashlib.sha256(
            f"{record['dataset_source']}|{record['image']}|{record['shape_index']}".encode()
        ).hexdigest()[:16]
        source_dir = output_root / record["dataset_source"]
        source_dir.mkdir(parents=True, exist_ok=True)
        crop_path = source_dir / f"{record_index:06d}_{token}_{record['defect_class']}.png"
        if not cv2.imwrite(str(crop_path), crop):
            failures["write_failed"] += 1
            continue
        group = f"{record['dataset_source']}::{record['group']}"
        rows.append({
            "crop_path": str(crop_path),
            "defect_class": record["defect_class"],
            "board_id": group,
            "capture_id": Path(record["image"]).stem,
            "dataset_source": record["dataset_source"],
            "roi_kind": "joint",
            "label_scope": "joint",
            "label_status": "source_annotation",
            "split": "",
            "camera_id": f"public_{record['dataset_source']}",
            "lot_id": "public_dataset",
            "session_id": record["dataset_source"],
            "source_image": str(record["image"]),
            "preprocess_id": "public-native-joint-crop-v2",
            "original_label": record["original_label"],
            "annotation_layout": record["annotation_layout"],
            "annotation_instance_id": (
                f"{record['dataset_source']}::{record['group']}::"
                f"{record['original_label']}::{record['shape_index']}"
            ),
            "group_semantics": "source_image_proxy_not_physical_board",
        })
    return pd.DataFrame(rows), dict(sorted(failures.items()))


def prepare_public_manifest():
    records, inventory = [], []
    for spec in PUBLIC_SOURCE_SPECS:
        if not spec["enabled"]:
            continue
        root, acquisition = resolve_public_source(spec)
        if root is None:
            continue
        source_records, adapter_report = PUBLIC_READERS[spec["adapter"]](root)
        inventory.append({
            **{key: spec[key] for key in (
                "name", "provider", "handle", "adapter", "license", "homepage",
                "citation", "roi_semantics",
            )},
            "acquisition": acquisition,
            "root": str(root),
            "record_count_before_crop": int(len(source_records)),
            "adapter_report": adapter_report,
        })
        records.extend(source_records)
    if not records:
        if RUN_MODE == "public_bootstrap":
            raise RuntimeError("Không đọc được ROI joint nào từ nguồn public đã bật.")
        return None, inventory
    public_root = WORK_DIR / "public_prepared"
    frame, crop_failures = materialize_public_crops(
        records, public_root / "crops", CONFIG["public_crop_padding_ratio"]
    )
    if frame.empty:
        raise RuntimeError("Đã đọc annotation public nhưng không cắt được ROI hợp lệ nào.")
    manifest_path = public_root / "solder_dataset.csv"
    frame.to_csv(manifest_path, index=False)
    inventory_summary = {
        "schema": "aoi-public-dataset-provenance/1.0",
        "training_stage": RUN_MODE,
        "evaluation_role": EVALUATION_ROLE,
        "sources": inventory,
        "materialized_rows": int(len(frame)),
        "materialized_class_counts": frame["defect_class"].value_counts().sort_index().to_dict(),
        "crop_failures": crop_failures,
        "limitations": [
            "public source groups prevent joint-level leakage only; the flat LabelMe "
            "release has no physical PCB ID, so same-PCB/acquisition leakage cannot be ruled out",
            "SolDef_AI comes from six PCBs and multiple controlled viewpoints/lighting setups",
            "public holdout metrics are not production-domain evidence",
            "no public unknown/wrong-crop class is synthesized",
        ],
    }
    (REPORT_DIR / "dataset_provenance.json").write_text(
        json_text(inventory_summary), encoding="utf-8"
    )
    print("Public crops:", len(frame), frame["defect_class"].value_counts().to_dict())
    return manifest_path, inventory_summary


PUBLIC_MANIFEST_PATH, PUBLIC_SOURCE_INVENTORY = prepare_public_manifest()

# %% [markdown]
# ## 3. Nạp canonical CSV và tách đúng scope
#
# `source_image` chỉ là fallback legacy. Artifact production bắt buộc có `board_id` vật lý;
# nếu thiếu notebook vẫn cho nghiên cứu nhưng quality gate sẽ không cho xuất `best.onnx`.

# %%
CANONICAL_REQUIRED = {
    "crop_path", "defect_class", "board_id", "capture_id", "dataset_source", "roi_kind"
}


def canonicalize_manifest(frame, manifest_path, source_kind="local"):
    manifest_path = Path(manifest_path).resolve()
    data = frame.copy()
    original_columns = set(data.columns)
    missing_canonical = sorted(CANONICAL_REQUIRED - original_columns)
    if "crop_path" not in data or "defect_class" not in data:
        missing = sorted({"crop_path", "defect_class"} - set(data.columns))
        raise ValueError(f"{manifest_path}: thiếu cột bắt buộc {missing}")

    data["source_kind"] = source_kind
    if "dataset_source" not in data:
        data["dataset_source"] = manifest_path.parent.name
    data["dataset_source"] = data["dataset_source"].fillna(manifest_path.parent.name).map(normalize_label)

    physical_board_column = "board_id" in original_columns
    if "board_id" not in data:
        if "group_id" in data:
            data["board_id"] = data["group_id"]
        elif "source_image" in data:
            data["board_id"] = data["source_image"].map(lambda value: Path(str(value)).stem)
        else:
            data["board_id"] = ""
    if "capture_id" not in data:
        if "source_image" in data:
            data["capture_id"] = data["source_image"].map(lambda value: Path(str(value)).stem)
        else:
            data["capture_id"] = data["board_id"]
    if "roi_kind" not in data:
        data["roi_kind"] = data["kind"] if "kind" in data else PROFILE["scope"]
    if "label_scope" not in data:
        data["label_scope"] = data["roi_kind"]
    if "label_status" not in data:
        data["label_status"] = (
            "source_annotation" if source_kind == "public" else "verified_legacy"
        )
    if "split" not in data:
        data["split"] = ""
    if "camera_id" not in data:
        data["camera_id"] = "unknown_camera"
    if "lot_id" not in data:
        data["lot_id"] = "unknown_lot"
    if "session_id" not in data:
        data["session_id"] = "unknown_session"
    if "preprocess_id" not in data:
        data["preprocess_id"] = "unknown"
    if "annotation_instance_id" not in data:
        data["annotation_instance_id"] = ""

    def resolve_crop(value):
        candidate = Path(str(value))
        if not candidate.is_absolute():
            candidate = manifest_path.parent / candidate
        return str(candidate.resolve())

    data["crop_path"] = data["crop_path"].map(resolve_crop)
    data["defect_class"] = data["defect_class"].map(normalize_label)
    data["board_id"] = data["board_id"].fillna("").astype(str).str.strip()
    data["capture_id"] = data["capture_id"].fillna("").astype(str).str.strip()
    data["roi_kind"] = data["roi_kind"].map(normalize_label)
    data["split_requested"] = data["split"].fillna("").map(normalize_label).map(
        lambda value: SPLIT_ALIASES.get(value, "invalid")
    )
    data["physical_board_id_present"] = bool(physical_board_column) & data["board_id"].ne("")
    data["label_status_explicit"] = bool("label_status" in original_columns)
    data["canonical_missing_columns"] = ",".join(missing_canonical)
    # Local board_id là ID vật lý toàn cục giữa mọi manifest/camera source. Không namespace
    # theo dataset_source vì cùng board được export hai lần vẫn phải ở cùng split.
    data["physical_board"] = np.where(
        data["source_kind"].eq("local"),
        "local::" + data["board_id"],
        data["dataset_source"] + "::" + data["board_id"],
    )
    data["manifest_path"] = str(manifest_path)
    mapped = data.apply(map_target_label, axis=1, result_type="expand")
    mapped.columns = ["target_label", "routing_reason"]
    data[["target_label", "routing_reason"]] = mapped
    data["eligible"] = data["target_label"].isin(CLASS_NAMES)
    ignored = data["split_requested"].eq("ignore")
    data.loc[ignored, "eligible"] = False
    data.loc[ignored, "routing_reason"] = "split_ignore"
    return data


def discover_local_manifests(required=True):
    configured = [Path(value) for value in CONFIG["local_manifests"]]
    existing = [path for path in configured if path.is_file()]
    if configured and len(existing) != len(configured):
        missing = [str(path) for path in configured if not path.is_file()]
        raise FileNotFoundError(f"Không thấy local manifest: {missing}")
    if existing:
        return existing
    if not required:
        return []
    input_root = Path("/kaggle/input")
    discovered = sorted(input_root.rglob("solder_dataset.csv")) if input_root.is_dir() else []
    if not discovered:
        raise FileNotFoundError(
            "Không thấy solder_dataset.csv. Hãy Add Input dataset export từ AOI hoặc điền "
            "CONFIG['local_manifests']."
        )
    print("Tự dò local manifests:", [str(path) for path in discovered])
    return discovered


local_manifest_paths = discover_local_manifests(required=MODE_POLICY["require_local"])
frames = [
    canonicalize_manifest(pd.read_csv(path), path, source_kind="local")
    for path in local_manifest_paths
]
public_manifest_paths = []
if PUBLIC_MANIFEST_PATH is not None:
    public_manifest_paths.append(Path(PUBLIC_MANIFEST_PATH))
public_manifest_paths.extend(Path(value) for value in CONFIG["public_manifests"])
seen_public_paths = set()
for path in public_manifest_paths:
    resolved_path = path.resolve()
    if resolved_path in seen_public_paths:
        continue
    seen_public_paths.add(resolved_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không thấy public manifest: {path}")
    frames.append(canonicalize_manifest(pd.read_csv(path), path, source_kind="public"))

if not frames:
    raise RuntimeError("Không có manifest local/public nào để train.")

raw_manifest = pd.concat(frames, ignore_index=True)
raw_manifest.insert(0, "row_id", np.arange(len(raw_manifest), dtype=np.int64))
routing_export = raw_manifest.copy()
routing_export["crop_name"] = routing_export["crop_path"].map(lambda value: Path(value).name)
routing_columns = [
    "row_id", "crop_name", "defect_class", "target_label", "routing_reason",
    "eligible", "source_kind", "dataset_source", "board_id", "capture_id",
    "roi_kind", "label_scope", "label_status", "label_status_explicit",
    "canonical_missing_columns", "split_requested", "camera_id", "lot_id",
    "session_id", "preprocess_id",
]
routing_export = routing_export[[column for column in routing_columns if column in routing_export]]
routing_export.to_csv(REPORT_DIR / "manifest_routing_all.csv", index=False)
routing_summary = (
    raw_manifest.groupby(["source_kind", "routing_reason"], dropna=False)
    .size().rename("count").reset_index().sort_values("count", ascending=False)
)
display(routing_summary)

eligible_manifest = raw_manifest[raw_manifest["eligible"]].copy()
if eligible_manifest.empty:
    raise RuntimeError("Không có crop nào đúng taxonomy/scope sau bước routing.")
if eligible_manifest["crop_path"].duplicated().any():
    duplicates = eligible_manifest.loc[
        eligible_manifest["crop_path"].duplicated(False), "crop_path"
    ].head(20).tolist()
    raise RuntimeError(f"crop_path bị lặp trong manifest: {duplicates}")

print("Eligible:", len(eligible_manifest))
display(pd.crosstab(eligible_manifest["source_kind"], eligible_manifest["target_label"]))

# %% [markdown]
# ## 3. Audit ảnh, exact hash, perceptual hash và label conflict
#
# Ảnh lỗi decode bị loại, không được thay bằng canvas xám. Duplicate mang hai nhãn khác
# nhau bị quarantine và làm production gate fail. `leakage_group` nối cả board vật lý lẫn
# duplicate cluster để chúng không thể rơi sang hai split khác nhau.

# %%
_ROBOFLOW_EXPORT = re.compile(
    r"^(?P<stem>.+?)_(?:jpg|jpeg|png|bmp|tif|tiff)\.rf\.[0-9a-f]{6,}$",
    re.IGNORECASE,
)


def normalized_origin(path_value):
    stem = Path(str(path_value)).stem
    match = _ROBOFLOW_EXPORT.match(stem)
    if match:
        stem = match.group("stem")
    stem = re.sub(r"(?:__|_)(?:aug|flip|rot|copy)[-_]?\d*$", "", stem, flags=re.I)
    return normalize_label(stem)


def sha256_pixels(rgb):
    digest = hashlib.sha256()
    digest.update(np.asarray(rgb.shape, dtype=np.int32).tobytes())
    digest.update(np.ascontiguousarray(rgb).tobytes())
    return digest.hexdigest()


def perceptual_hash(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    frequency = cv2.dct(small)[:8, :8]
    values = frequency.flatten()[1:]
    bits = values >= np.median(values)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming_hash(first, second):
    return (int(first, 16) ^ int(second, 16)).bit_count()


class DisjointSet:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first, second):
        first, second = self.find(first), self.find(second)
        if first == second:
            return
        if self.rank[first] < self.rank[second]:
            first, second = second, first
        self.parent[second] = first
        if self.rank[first] == self.rank[second]:
            self.rank[first] += 1


class BKTree:
    """Small Hamming-distance index used to avoid an O(N²) pHash scan."""
    def __init__(self):
        self.root = None

    def add(self, value, index):
        if self.root is None:
            self.root = [value, [index], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [index], {}]
                return
            node = node[2][distance]

    def query(self, value, radius):
        if self.root is None:
            return []
        found, stack = [], [self.root]
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                found.extend(node[1])
            low, high = distance - radius, distance + radius
            stack.extend(child for edge, child in node[2].items() if low <= edge <= high)
        return found


image_audit_rows = []
for row in eligible_manifest.itertuples(index=False):
    path = Path(row.crop_path)
    issue = ""
    width = height = 0
    pixel_sha256 = phash = ""
    color_signature = None
    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
        issue = "missing_or_unsupported_image"
    else:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            issue = "decode_failed"
        else:
            height, width = bgr.shape[:2]
            if min(width, height) < int(CONFIG["min_short_edge"]):
                issue = "roi_too_small"
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pixel_sha256 = sha256_pixels(rgb)
            phash = perceptual_hash(rgb)
            normalized = rgb.astype(np.float32) / 255.0
            color_signature = np.concatenate(
                [normalized.mean(axis=(0, 1)), normalized.std(axis=(0, 1))]
            ).round(6).tolist()
    image_audit_rows.append({
        "row_id": int(row.row_id), "width": width, "height": height,
        "pixel_sha256": pixel_sha256, "phash": phash,
        "color_signature": color_signature, "image_issue": issue,
    })

image_audit = pd.DataFrame(image_audit_rows)
audited = eligible_manifest.merge(image_audit, on="row_id", how="left", validate="one_to_one")
audited["normalized_origin"] = np.where(
    audited["annotation_instance_id"].fillna("").astype(str).str.strip().ne(""),
    audited["annotation_instance_id"].map(normalize_label),
    audited["crop_path"].map(normalized_origin),
)
invalid_images = audited[audited["image_issue"].ne("")].copy()
valid = audited[audited["image_issue"].eq("")].copy().reset_index(drop=True)
if valid.empty:
    raise RuntimeError("Không có ảnh hợp lệ sau decode/size audit.")

# Chỉ exact pixels và bản export/augmentation có cùng origin mới được deduplicate hoặc
# quarantine label conflict. pHash gần giống không đủ chứng minh là cùng mẫu vật lý —
# joint thiếc vốn lặp hình — nên chỉ được báo để audit, không xóa hay ép chung split.
dedup_dsu = DisjointSet(len(valid))
for _, indices in valid.groupby("pixel_sha256").groups.items():
    indices = list(indices)
    for other in indices[1:]:
        dedup_dsu.union(indices[0], other)
for _, indices in valid.groupby(["dataset_source", "normalized_origin"]).groups.items():
    indices = list(indices)
    for other in indices[1:]:
        dedup_dsu.union(indices[0], other)

valid["dedup_cluster"] = [f"dedup_{dedup_dsu.find(i):08d}" for i in range(len(valid))]
near_dsu = DisjointSet(len(valid))
for _, indices in valid.groupby("dedup_cluster").groups.items():
    indices = list(indices)
    for other in indices[1:]:
        near_dsu.union(indices[0], other)

# Near-duplicate chỉ nối khi pHash gần, aspect gần và màu tổng thể gần. Điều kiện phụ
# tránh gom nhầm hàng nghìn joint sáng/bạc nhưng là mẫu vật lý khác nhau.
for _, source_indices in valid.groupby("dataset_source").groups.items():
    tree = BKTree()
    for index in source_indices:
        row = valid.loc[index]
        value = int(row.phash, 16)
        first_signature = np.asarray(row.color_signature, dtype=np.float32)
        first_aspect = row.width / max(1, row.height)
        for candidate in tree.query(value, int(CONFIG["near_duplicate_hamming"])):
            other = valid.loc[candidate]
            second_signature = np.asarray(other.color_signature, dtype=np.float32)
            second_aspect = other.width / max(1, other.height)
            if abs(math.log(max(first_aspect, 1e-6) / max(second_aspect, 1e-6))) > 0.03:
                continue
            if float(np.linalg.norm(first_signature - second_signature)) > float(
                CONFIG["near_duplicate_color_distance"]
            ):
                continue
            near_dsu.union(index, candidate)
        tree.add(value, index)

valid["near_duplicate_cluster"] = [
    f"near_{near_dsu.find(i):08d}" for i in range(len(valid))
]
# Split leakage is enforced only for proven duplicates/known augmentations. A pHash match
# remains an audit candidate because using it as identity can transitively merge many valid
# solder joints that merely look alike.
valid["duplicate_cluster"] = valid["dedup_cluster"]
conflicting_clusters = []
for cluster, part in valid.groupby("dedup_cluster"):
    labels = sorted(part["target_label"].unique())
    if len(labels) > 1:
        conflicting_clusters.append({
            "dedup_cluster": cluster,
            "labels": labels,
            "rows": part["row_id"].astype(int).tolist(),
        })
conflicting_ids = {item["dedup_cluster"] for item in conflicting_clusters}
quarantined = valid[valid["dedup_cluster"].isin(conflicting_ids)].copy()
valid = valid[~valid["dedup_cluster"].isin(conflicting_ids)].copy().reset_index(drop=True)

# Gộp duplicate cluster với board vật lý thành leakage group.
leakage_dsu = DisjointSet(len(valid))
for column in ("physical_board", "duplicate_cluster"):
    for _, indices in valid.groupby(column).groups.items():
        indices = list(indices)
        for other in indices[1:]:
            leakage_dsu.union(indices[0], other)
valid["leakage_group"] = [f"lg_{leakage_dsu.find(i):08d}" for i in range(len(valid))]

# Khóa split explicit TRƯỚC khi drop duplicate. Nếu một copy ghi train còn copy kia ghi
# test, không được giữ ngẫu nhiên copy đầu rồi làm xung đột biến mất. Một split explicit
# duy nhất được lan ra toàn leakage group; nhiều split là hard failure có audit trail.
split_assignment_conflicts = []
for leakage_group, part in valid.groupby("leakage_group"):
    requested = sorted(
        value for value in part["split_requested"].unique()
        if value in {"train", "val", "calibration", "test"}
    )
    if len(requested) > 1:
        split_assignment_conflicts.append({
            "leakage_group": leakage_group,
            "requested_splits": requested,
            "rows": part["row_id"].astype(int).tolist(),
        })
    elif requested:
        valid.loc[part.index, "split_requested"] = requested[0]

# Một pixel duplicate cùng nhãn chỉ được giữ một lần để augmentation/public copy không
# chi phối loss. Board/capture vẫn được nối trước khi drop.
before_dedup = len(valid)
valid["_source_priority"] = valid["source_kind"].ne(PRIMARY_SOURCE_KIND).astype(int)
valid = valid.sort_values(["_source_priority", "row_id"], kind="stable")
valid = valid.drop_duplicates(["dedup_cluster", "target_label"], keep="first").reset_index(drop=True)
valid = valid.drop(columns=["_source_priority"])
duplicates_removed = before_dedup - len(valid)

dataset_fingerprint_payload = "\n".join(
    f"{row.pixel_sha256}|{row.target_label}|{row.physical_board}"
    for row in valid.sort_values(["pixel_sha256", "target_label"]).itertuples()
)
DATASET_FINGERPRINT = hashlib.sha256(dataset_fingerprint_payload.encode("utf-8")).hexdigest()

dataset_audit = {
    "schema": "aoi-solder-dataset-audit/2.0",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "run_mode": RUN_MODE,
    "evaluation_role": EVALUATION_ROLE,
    "primary_source_kind": PRIMARY_SOURCE_KIND,
    "primary_group_semantics": (
        "source_image_proxy_not_physical_board"
        if RUN_MODE == "public_bootstrap" else "physical_aoi_board"
    ),
    "rows_input": int(len(raw_manifest)),
    "rows_eligible": int(len(eligible_manifest)),
    "rows_valid_after_audit": int(len(valid)),
    "invalid_images": int(len(invalid_images)),
    "duplicates_removed": int(duplicates_removed),
    "dedup_cluster_count": int(valid["dedup_cluster"].nunique()),
    "near_duplicate_cluster_count": int(valid["near_duplicate_cluster"].nunique()),
    "maximum_near_duplicate_cluster_size": int(
        valid.groupby("near_duplicate_cluster").size().max()
    ),
    "conflicting_labels": conflicting_clusters,
    "split_assignment_conflicts": split_assignment_conflicts,
    "dataset_fingerprint": DATASET_FINGERPRINT,
    "physical_board_id_missing": int((~valid["physical_board_id_present"]).sum()),
    "verified_label_status_missing": int(
        (
            valid["source_kind"].eq("local")
            & (
                ~valid["label_status_explicit"]
                | ~valid["label_status"].map(normalize_label).isin(
                    {"verified", "approved", "adjudicated"}
                )
            )
        ).sum()
    ),
    "canonical_contract_incomplete": int(
        (valid["source_kind"].eq("local") & valid["canonical_missing_columns"].ne("")).sum()
    ),
    "preprocess_id_mismatch": int(
        valid.loc[valid["source_kind"].eq("local"), "preprocess_id"]
        .ne(CONFIG["production_preprocess_id"]).sum()
    ),
    "class_counts": valid["target_label"].value_counts().sort_index().to_dict(),
    "source_counts": valid["dataset_source"].value_counts().sort_index().to_dict(),
}
local_nonignored = raw_manifest[
    raw_manifest["source_kind"].eq("local") & ~raw_manifest["split_requested"].eq("ignore")
]
local_eligible = eligible_manifest[eligible_manifest["source_kind"].eq("local")]
dataset_audit["local_label_retained_ratio"] = float(
    len(local_eligible) / max(1, len(local_nonignored))
)
dataset_audit["invalid_image_ratio"] = float(
    len(invalid_images[invalid_images["source_kind"].eq("local")])
    / max(1, len(local_eligible))
)
(REPORT_DIR / "dataset_audit.json").write_text(
    json_text(dataset_audit), encoding="utf-8"
)
def export_safe_frame(frame, path):
    exported = frame.copy()
    if "crop_path" in exported:
        exported["crop_name"] = exported["crop_path"].map(lambda value: Path(value).name)
    safe_columns = [
        "row_id", "crop_name", "defect_class", "target_label", "routing_reason",
        "source_kind", "dataset_source", "board_id", "capture_id", "physical_board",
        "roi_kind", "label_scope", "label_status", "split_requested", "image_issue",
        "width", "height", "pixel_sha256", "phash", "normalized_origin",
        "annotation_instance_id", "dedup_cluster", "duplicate_cluster",
        "near_duplicate_cluster", "leakage_group",
    ]
    exported = exported[[column for column in safe_columns if column in exported]]
    exported.to_csv(path, index=False)


export_safe_frame(invalid_images, REPORT_DIR / "invalid_images.csv")
export_safe_frame(quarantined, REPORT_DIR / "quarantined_label_conflicts.csv")
print(json_text(dataset_audit))
if split_assignment_conflicts:
    raise RuntimeError(
        "Explicit split conflict trong duplicate/board group; xem dataset_audit.json và "
        "sửa CSV trước khi train."
    )

# %% [markdown]
# ## 4. Preview bắt buộc trước khi train
#
# Nếu ảnh/nhãn ở đây nhìn sai, dừng và sửa CSV. Không có metric nào cứu được ground truth sai.

# %%
print("Dataset đã audit. Preview chỉ được render từ split=train sau khi split đã cố định.")

# %% [markdown]
# ## 5. Chia `train / val / calibration / evaluation` theo group chống leakage
#
# Public bootstrap tự chia nguồn public và coi `test` nội bộ là **public proxy holdout**.
# Camera fine-tune giữ nguyên hành vi nghiêm: public chỉ train, locked-test local explicit
# không bị tự thêm board. Hàm audit kiểm tra lại board/group, SHA và pHash sau khi chia.

# %%
SPLIT_ORDER = ("train", "val", "calibration", "test")


def assign_board_splits(frame, fractions, seed=42, attempts=500):
    result = frame.copy()
    result["split"] = result["split_requested"].fillna("")
    train_only = result["source_kind"].isin(TRAIN_ONLY_SOURCE_KINDS)
    result.loc[train_only, "split"] = "train"
    primary = result[result["source_kind"].eq(PRIMARY_SOURCE_KIND)]
    if primary.empty:
        raise RuntimeError(
            f"Không có row source_kind={PRIMARY_SOURCE_KIND} cho run_mode={RUN_MODE}."
        )
    invalid_requested = sorted(set(primary["split"]) - set(SPLIT_ORDER) - {""})
    if invalid_requested:
        raise ValueError(f"Split không hợp lệ: {invalid_requested}")

    fixed = {}
    for group, part in primary.groupby("leakage_group"):
        values = sorted(value for value in part["split"].unique() if value)
        if len(values) > 1:
            raise RuntimeError(f"Một leakage_group bị gán nhiều split: {group} -> {values}")
        if values:
            fixed[group] = values[0]

    groups = sorted(primary["leakage_group"].unique())
    unassigned = [group for group in groups if group not in fixed]
    has_explicit_test = any(value == "test" for value in fixed.values())
    if unassigned and not has_explicit_test and not MODE_POLICY["allow_auto_split"]:
        raise RuntimeError(
            f"Còn {len(unassigned)} board/group chưa có split và chưa có locked-test. "
            "Điền ít nhất các board split=test, hoặc bật allow_research_auto_split=True "
            "cho một run nghiên cứu không production-ready."
        )

    fraction_values = np.asarray([float(fractions[name]) for name in SPLIT_ORDER])
    fraction_values = fraction_values / fraction_values.sum()
    best_assignment, best_score = None, float("inf")
    labels = sorted(primary["target_label"].unique())
    for attempt in range(max(1, int(attempts))):
        rng = np.random.default_rng(seed + attempt)
        assignment = dict(fixed)
        if unassigned:
            shuffled = list(rng.permutation(unassigned))
            # Locked-test đã explicit thì không tự thêm board vào test. Chỉ train/val/cal
            # được group-stratify lại; vì vậy test không thay đổi sau khi người dùng khóa.
            active_splits = SPLIT_ORDER[:-1] if has_explicit_test else SPLIT_ORDER
            active_fractions = np.asarray([float(fractions[name]) for name in active_splits])
            active_fractions = active_fractions / active_fractions.sum()
            desired = np.floor(active_fractions * len(unassigned)).astype(int)
            while desired.sum() < len(unassigned):
                desired[int(np.argmax(active_fractions - desired / max(1, len(unassigned))))] += 1
            remaining = {
                name: int(desired[index]) for index, name in enumerate(active_splits)
            }
            slots = [name for name in active_splits for _ in range(remaining[name])]
            while len(slots) < len(shuffled):
                slots.append(active_splits[int(rng.choice(len(active_splits), p=active_fractions))])
            rng.shuffle(slots)
            assignment.update(dict(zip(shuffled, slots[:len(shuffled)])))

        candidate = primary["leakage_group"].map(assignment)
        score = 0.0
        for label in labels:
            mask = primary["target_label"].eq(label)
            total = max(1, int(mask.sum()))
            for index, split in enumerate(SPLIT_ORDER):
                observed = float((mask & candidate.eq(split)).sum()) / total
                score += abs(observed - fraction_values[index])
                if (mask & candidate.eq(split)).sum() == 0:
                    score += 25.0
        if score < best_score:
            best_score, best_assignment = score, assignment
    if best_assignment is None:
        raise RuntimeError("Không tìm được group split.")
    primary_indices = result["source_kind"].eq(PRIMARY_SOURCE_KIND)
    result.loc[primary_indices, "split"] = result.loc[
        primary_indices, "leakage_group"
    ].map(best_assignment)
    result["split_origin"] = np.where(
        result["split_requested"].isin(SPLIT_ORDER), "explicit", "auto_research"
    )
    result.loc[train_only, "split_origin"] = (
        result.loc[train_only, "source_kind"].astype(str) + "_train_only"
    )
    return result


def audit_split_integrity(frame):
    def crossing(column):
        if column not in frame:
            return []
        counts = frame.groupby(column)["split"].nunique()
        return sorted(str(value) for value in counts[counts > 1].index)

    report = {
        "cross_split_board_leakage": crossing("physical_board"),
        "cross_split_sha256_leakage": crossing("pixel_sha256"),
        "cross_split_phash_leakage": crossing("duplicate_cluster"),
        "cross_split_phash_candidates": crossing("near_duplicate_cluster"),
        "conflicting_labels": dataset_audit["conflicting_labels"],
        "split_assignment_conflicts": dataset_audit["split_assignment_conflicts"],
        "train_only_rows_outside_train": int(
            (frame["source_kind"].isin(TRAIN_ONLY_SOURCE_KINDS) & ~frame["split"].eq("train")).sum()
        ),
    }
    report["public_rows_outside_train"] = int(
        report["train_only_rows_outside_train"]
        if "public" in TRAIN_ONLY_SOURCE_KINDS else 0
    )
    failures = [
        key for key in (
            "cross_split_board_leakage", "cross_split_sha256_leakage",
            "cross_split_phash_leakage", "conflicting_labels", "split_assignment_conflicts"
        ) if report[key]
    ]
    if report["train_only_rows_outside_train"]:
        failures.append("train_only_rows_outside_train")
    if failures:
        raise RuntimeError(f"Split integrity FAIL: {failures}")
    return report


locked_test_explicit = bool(
    RUN_MODE == "camera_finetune"
    and (valid["source_kind"].eq("local") & valid["split_requested"].eq("test")).any()
)
if (
    MODE_POLICY["production_allowed"]
    and CONFIG["require_explicit_locked_test"]
    and not locked_test_explicit
):
    raise RuntimeError(
        "Không có locked-test explicit. Gán split=test theo board trước khi chạy; "
        "không chọn test ngẫu nhiên sau khi đã nhìn metric."
    )

split_manifest = assign_board_splits(
    valid,
    CONFIG["split_fractions"],
    seed=SEED,
    attempts=CONFIG["split_search_attempts"],
)
split_report = audit_split_integrity(split_manifest)
if set(split_manifest["split"].unique()) != set(SPLIT_ORDER):
    missing = sorted(set(SPLIT_ORDER) - set(split_manifest["split"].unique()))
    raise RuntimeError(f"Thiếu split: {missing}")

split_class_counts = pd.crosstab(split_manifest["split"], split_manifest["target_label"])
missing_class_split = [
    f"{split}:{class_name}"
    for split in SPLIT_ORDER
    for class_name in CLASS_NAMES
    if int(split_class_counts.get(class_name, pd.Series(dtype=int)).get(split, 0)) == 0
]
if missing_class_split:
    raise RuntimeError(
        "Mỗi split phải có đủ class để metric/calibration có nghĩa; đang thiếu "
        f"{missing_class_split}"
    )
split_report["class_group_support"] = {
    split: {
        class_name: int(
            split_manifest.loc[
                split_manifest["split"].eq(split)
                & split_manifest["target_label"].eq(class_name),
                "physical_board",
            ].nunique()
        )
        for class_name in CLASS_NAMES
    }
    for split in SPLIT_ORDER
}

split_manifest = split_manifest.sort_values(["split", "physical_board", "row_id"])
# Fingerprint không phụ thuộc absolute Kaggle mount/path. File chia sẻ cũng không đưa
# `/kaggle/input/...` ra ngoài artifact.
split_fingerprint_columns = [
    "pixel_sha256", "physical_board", "target_label", "split", "dataset_source", "roi_kind"
]
split_fingerprint = split_manifest[split_fingerprint_columns].sort_values(
    split_fingerprint_columns, kind="stable"
).to_csv(index=False, lineterminator="\n")
SPLIT_HASH = hashlib.sha256(split_fingerprint.encode("utf-8")).hexdigest()
split_export_columns = [
    "row_id", "defect_class", "target_label", "source_kind", "dataset_source",
    "board_id", "capture_id", "physical_board", "roi_kind", "label_scope",
    "label_status", "camera_id", "lot_id", "session_id", "preprocess_id",
    "pixel_sha256", "phash", "dedup_cluster", "duplicate_cluster",
    "near_duplicate_cluster", "leakage_group", "split", "split_origin",
]
split_export = split_manifest[
    [column for column in split_export_columns if column in split_manifest]
].copy()
split_export["crop_name"] = split_manifest["crop_path"].map(lambda value: Path(value).name)
split_export.to_csv(WORK_DIR / "split_manifest.csv", index=False)
split_report["split_hash"] = SPLIT_HASH
split_report["locked_test_explicit"] = locked_test_explicit
split_report["run_mode"] = RUN_MODE
split_report["evaluation_role"] = EVALUATION_ROLE
split_report["evaluation_domain"] = EVALUATION_DOMAIN
split_report["is_production_evidence"] = bool(MODE_POLICY["production_allowed"])
(REPORT_DIR / "split_audit.json").write_text(
    json_text(split_report), encoding="utf-8"
)
display(pd.crosstab(split_manifest["split"], split_manifest["target_label"]))
display(split_manifest.groupby("split")["physical_board"].nunique().rename("boards"))

# Visual audit chỉ lấy train. Không hiển thị ảnh/nhãn calibration/evaluation trước khi
# model và threshold được khóa.
train_preview_source = split_manifest[split_manifest["split"].eq("train")]
preview_parts = []
for class_name in CLASS_NAMES:
    part = train_preview_source[train_preview_source["target_label"].eq(class_name)]
    if not part.empty:
        preview_parts.append(part.sample(min(6, len(part)), random_state=SEED))
preview = pd.concat(preview_parts, ignore_index=True).head(min(24, len(train_preview_source)))
columns = 6
rows = max(1, math.ceil(len(preview) / columns))
figure, axes = plt.subplots(rows, columns, figsize=(18, 3.2 * rows))
axes = np.asarray(axes).reshape(-1)
for axis in axes:
    axis.axis("off")
for axis, row in zip(axes, preview.itertuples()):
    axis.imshow(Image.open(row.crop_path).convert("RGB"))
    axis.set_title(f"{row.target_label}\n{row.dataset_source} · train", fontsize=8)
    axis.axis("off")
plt.tight_layout()
plt.savefig(REPORT_DIR / "dataset_preview_train_only.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Preprocess parity, Dataset và augmentation nhẹ
#
# Model nhận RGB, letterbox 114, ImageNet mean/std và raw logits — đúng contract runtime
# `pcb-solder-defect-classifier/1.0`. Augmentation chỉ áp dụng cho train.

# %%
INPUT_SIZE = int(CONFIG["input_size"])
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox_rgb(rgb, size=INPUT_SIZE, value=None):
    value = int(CONFIG["letterbox_value"] if value is None else value)
    target_width = target_height = int(size)
    height, width = rgb.shape[:2]
    scale = min(target_width / max(1, width), target_height / max(1, height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.full((target_height, target_width, 3), value, dtype=np.uint8)
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = resized
    return canvas


def preprocess_rgb(rgb):
    tensor = letterbox_rgb(rgb).astype(np.float32) / 255.0
    tensor = (tensor - MEAN) / STD
    return np.ascontiguousarray(tensor.transpose(2, 0, 1), dtype=np.float32)


def mild_augment(rgb):
    image = rgb.copy()
    if np.random.random() < 0.50:
        image = cv2.flip(image, 1)
    if np.random.random() < 0.50:
        image = cv2.flip(image, 0)
    if np.random.random() < 0.35:
        angle = float(np.random.uniform(-10.0, 10.0))
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        image = cv2.warpAffine(
            image, matrix, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    if np.random.random() < 0.35:
        gain = float(np.random.uniform(0.90, 1.10))
        bias = float(np.random.uniform(-8.0, 8.0))
        image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    if np.random.random() < 0.15:
        image = cv2.GaussianBlur(image, (3, 3), sigmaX=float(np.random.uniform(0.2, 0.8)))
    if np.random.random() < 0.15:
        noise = np.random.normal(0.0, np.random.uniform(1.0, 4.0), image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image


class SolderV2Dataset(Dataset):
    def __init__(self, frame, training=False):
        self.frame = frame.reset_index(drop=True)
        self.training = bool(training)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        bgr = cv2.imread(str(row.crop_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Ảnh đã pass audit nhưng decode lại lỗi: {row.crop_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.training:
            rgb = mild_augment(rgb)
        tensor = torch.from_numpy(preprocess_rgb(rgb))
        target = CLASS_TO_INDEX[row.target_label]
        return tensor, target, int(row.row_id)


# Synthetic parity khóa thứ tự BGR→RGB và vị trí padding trước khi train.
parity_fixture_bgr = np.zeros((19, 31, 3), dtype=np.uint8)
parity_fixture_bgr[..., 0] = 11
parity_fixture_bgr[..., 1] = 97
parity_fixture_bgr[..., 2] = 223
parity_fixture_rgb = cv2.cvtColor(parity_fixture_bgr, cv2.COLOR_BGR2RGB)
parity_tensor = preprocess_rgb(parity_fixture_rgb)
assert parity_tensor.shape == (3, INPUT_SIZE, INPUT_SIZE)
assert np.isfinite(parity_tensor).all()
assert float(parity_tensor[0].max()) > float(parity_tensor[2].max()), "BGR/RGB bị đảo"

frames_by_split = {
    split: split_manifest[split_manifest["split"].eq(split)].copy().reset_index(drop=True)
    for split in SPLIT_ORDER
}
loaders = {
    split: DataLoader(
        SolderV2Dataset(frame, training=(split == "train")),
        batch_size=int(CONFIG["batch_size"]),
        shuffle=(split == "train"),
        num_workers=int(CONFIG["num_workers"]),
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=int(CONFIG["num_workers"]) > 0,
    )
    for split, frame in frames_by_split.items()
}

for split in ("val", "calibration", "test"):
    if not frames_by_split[split]["source_kind"].eq(PRIMARY_SOURCE_KIND).all():
        raise RuntimeError(
            f"{split} phải chỉ chứa source_kind={PRIMARY_SOURCE_KIND} trong mode {RUN_MODE}."
        )

# %% [markdown]
# ## 7. Model và checkpoint selection trên `val`
#
# MobileNetV3-small là mặc định vì target có thể là ARM CPU. `convnext_tiny` có sẵn để
# so sánh capacity, nhưng không sửa được domain shift hoặc ground truth sai. Checkpoint
# được xếp hạng theo safety trước, sau đó macro-F1 và NLL; không dùng binary score che
# lỗi defect→sai subtype như notebook v1.

# %%
def build_model(model_name, class_count, pretrained=True):
    if model_name == "mobilenet_v3_small":
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        try:
            network = torchvision.models.mobilenet_v3_small(weights=weights)
        except Exception as exc:
            warnings.warn(f"Không tải được pretrained weights ({exc}); dùng random init.")
            network = torchvision.models.mobilenet_v3_small(weights=None)
        network.classifier[3] = nn.Linear(network.classifier[3].in_features, class_count)
    elif model_name == "convnext_tiny":
        weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        try:
            network = torchvision.models.convnext_tiny(weights=weights)
        except Exception as exc:
            warnings.warn(f"Không tải được pretrained weights ({exc}); dùng random init.")
            network = torchvision.models.convnext_tiny(weights=None)
        network.classifier[2] = nn.Linear(network.classifier[2].in_features, class_count)
    else:
        raise ValueError("model_name phải là mobilenet_v3_small hoặc convnext_tiny")
    return network


def discover_bootstrap_checkpoint():
    configured = str(CONFIG["bootstrap_checkpoint"]).strip()
    if configured:
        path = Path(configured)
        if not path.is_file():
            raise FileNotFoundError(f"Không thấy bootstrap checkpoint: {path}")
        return path.resolve()
    if RUN_MODE != "camera_finetune":
        return None
    input_root = Path("/kaggle/input")
    discovered = sorted(input_root.rglob("bootstrap_checkpoint.pt")) if input_root.is_dir() else []
    if len(discovered) > 1:
        raise RuntimeError(
            "Tìm thấy nhiều bootstrap_checkpoint.pt; điền CONFIG['bootstrap_checkpoint'] rõ ràng."
        )
    return discovered[0].resolve() if discovered else None


def load_bootstrap_backbone(network, checkpoint_path, expected_architecture):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise RuntimeError("bootstrap_checkpoint.pt không đúng schema/state_dict.")
    architecture = str(payload.get("architecture", ""))
    if architecture != expected_architecture:
        raise RuntimeError(
            f"Checkpoint architecture={architecture}, CONFIG model_name={expected_architecture}."
        )
    state_dict = payload["state_dict"]
    backbone_state = {
        key: value for key, value in state_dict.items() if key.startswith("features.")
    }
    if not backbone_state:
        raise RuntimeError("Checkpoint không có features.* để transfer backbone.")
    incompatible = network.load_state_dict(backbone_state, strict=False)
    unexpected = [key for key in incompatible.unexpected_keys if key.startswith("features.")]
    if unexpected:
        raise RuntimeError(f"Checkpoint có backbone key không khớp: {unexpected[:10]}")
    return {
        "path": str(checkpoint_path),
        "source_class_names": list(payload.get("class_names", [])),
        "loaded_tensor_count": len(backbone_state),
        "source_data_fingerprint": payload.get("data_fingerprint"),
    }


def set_backbone_trainable(network, trainable):
    for parameter in network.features.parameters():
        parameter.requires_grad = bool(trainable)
    for parameter in network.classifier.parameters():
        parameter.requires_grad = True


def build_optimizer(network, backbone_trainable):
    groups = [{"params": network.classifier.parameters(), "lr": float(CONFIG["head_lr"])}]
    if backbone_trainable:
        groups.append({"params": network.features.parameters(), "lr": float(CONFIG["backbone_lr"])})
    return torch.optim.AdamW(groups, weight_decay=float(CONFIG["weight_decay"]))


@torch.no_grad()
def predict(loader, active_model):
    active_model.eval()
    logits_parts, target_parts, row_ids = [], [], []
    for images, targets, rows in loader:
        logits_parts.append(active_model(images.to(DEVICE, non_blocking=True)).detach().cpu())
        target_parts.append(targets.detach().cpu())
        row_ids.extend(int(value) for value in rows)
    if not logits_parts:
        return torch.empty((0, len(CLASS_NAMES))), torch.empty(0, dtype=torch.long), []
    return torch.cat(logits_parts), torch.cat(target_parts), row_ids


def threshold_free_metrics(logits, targets):
    probabilities = torch.softmax(logits, dim=1).numpy()
    truth = targets.numpy()
    predicted = probabilities.argmax(axis=1)
    defect_mask = (truth != GOOD_INDEX) if UNKNOWN_INDEX is None else (
        (truth != GOOD_INDEX) & (truth != UNKNOWN_INDEX)
    )
    unknown_mask = np.zeros_like(truth, dtype=bool) if UNKNOWN_INDEX is None else truth == UNKNOWN_INDEX
    escape = float(np.mean(predicted[defect_mask] == GOOD_INDEX)) if defect_mask.any() else 1.0
    unknown_good = float(np.mean(predicted[unknown_mask] == GOOD_INDEX)) if unknown_mask.any() else 0.0
    macro_f1 = float(f1_score(truth, predicted, labels=range(len(CLASS_NAMES)), average="macro", zero_division=0))
    nll = float(F.cross_entropy(logits, targets).item())
    return {
        "argmax_escape": escape, "argmax_unknown_good": unknown_good,
        "macro_f1": macro_f1, "nll": nll,
    }


def checkpoint_rank(metrics):
    safe = (
        metrics["argmax_escape"] <= float(CONFIG["max_val_argmax_escape"])
        and metrics["argmax_unknown_good"] <= float(CONFIG["max_val_argmax_escape"])
    )
    return (
        int(safe), -metrics["argmax_escape"], -metrics["argmax_unknown_good"],
        metrics["macro_f1"], -metrics["nll"],
    )


train_counts = frames_by_split["train"]["target_label"].value_counts()
missing_train_classes = [name for name in CLASS_NAMES if int(train_counts.get(name, 0)) == 0]
if missing_train_classes:
    raise RuntimeError(f"Train thiếu class: {missing_train_classes}")

# Effective-number weights: một cơ chế cân bằng duy nhất, không chồng thêm sampler.
beta_effective = 0.999
effective_weights = []
for name in CLASS_NAMES:
    count = int(train_counts[name])
    effective = (1.0 - beta_effective ** count) / (1.0 - beta_effective)
    effective_weights.append(1.0 / max(effective, 1e-12))
effective_weights = np.asarray(effective_weights, dtype=np.float32)
effective_weights /= effective_weights.mean()
effective_weights = np.minimum(effective_weights, float(CONFIG["max_class_weight"]))

model = build_model(CONFIG["model_name"], len(CLASS_NAMES), bool(CONFIG["pretrained"])).to(DEVICE)
bootstrap_checkpoint_input = discover_bootstrap_checkpoint()
BOOTSTRAP_INITIALIZATION = None
if bootstrap_checkpoint_input is not None:
    BOOTSTRAP_INITIALIZATION = load_bootstrap_backbone(
        model, bootstrap_checkpoint_input, CONFIG["model_name"]
    )
    print("Đã transfer public bootstrap backbone:", BOOTSTRAP_INITIALIZATION)
elif RUN_MODE == "camera_finetune":
    warnings.warn(
        "camera_finetune chưa có bootstrap_checkpoint.pt; bắt đầu từ ImageNet weights."
    )
set_backbone_trainable(model, False)
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(effective_weights, dtype=torch.float32, device=DEVICE),
    label_smoothing=float(CONFIG["label_smoothing"]),
)
optimizer = build_optimizer(model, backbone_trainable=False)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max(1, int(CONFIG["epochs"]))
)
scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

best_rank = None
best_state = None
best_epoch = 0
epochs_without_improvement = 0
history = []
for epoch in range(1, int(CONFIG["epochs"]) + 1):
    if epoch == int(CONFIG["freeze_epochs"]) + 1:
        set_backbone_trainable(model, True)
        optimizer = build_optimizer(model, backbone_trainable=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(CONFIG["epochs"]) - epoch + 1)
        )
    model.train()
    train_loss = 0.0
    for images, targets, _ in loaders["train"]:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            loss = criterion(model(images), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += float(loss.detach().cpu()) * images.size(0)
    scheduler.step()

    val_logits, val_targets, _ = predict(loaders["val"], model)
    metrics = threshold_free_metrics(val_logits, val_targets)
    rank = checkpoint_rank(metrics)
    row = {
        "epoch": epoch,
        "train_loss": train_loss / max(1, len(frames_by_split["train"])),
        **metrics,
        "backbone_trainable": epoch > int(CONFIG["freeze_epochs"]),
    }
    history.append(row)
    print(
        f"epoch {epoch:02d} loss={row['train_loss']:.4f} "
        f"escape={metrics['argmax_escape']:.2%} unknown→good={metrics['argmax_unknown_good']:.2%} "
        f"macro-F1={metrics['macro_f1']:.4f} NLL={metrics['nll']:.4f}"
    )
    if best_rank is None or rank > best_rank:
        best_rank = rank
        best_epoch = epoch
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= int(CONFIG["patience"]):
            print("Early stop.")
            break

if best_state is None:
    raise RuntimeError("Không chọn được checkpoint.")
model.load_state_dict(best_state)
model = model.to(DEVICE).eval()
pd.DataFrame(history).to_csv(REPORT_DIR / "training_history.csv", index=False)
print("Best epoch:", best_epoch, "rank:", best_rank)

# %% [markdown]
# ## 8. Calibration riêng và chọn operating point
#
# Temperature chỉ fit trên `calibration`. Threshold không bao giờ được chỉnh theo test.
# Với mục tiêu escape 1%, 0 lỗi trên một test nhỏ không đủ chứng minh; notebook dùng
# one-sided Clopper–Pearson upper bound và in số mẫu độc lập tối thiểu cần có. Public
# bootstrap thường không đủ số group cho CI 1%; khi đó notebook chọn một threshold proxy
# theo point estimate, đánh dấu `provisional=true`, chứ không gọi nó là threshold line.

# %%
def fit_temperature(logits, targets):
    calibration_logits = logits.detach().clone().float()
    calibration_targets = targets.detach().clone().long()
    if calibration_logits.numel() == 0:
        raise ValueError("Calibration split rỗng.")
    log_temperature = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=100)

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.cross_entropy(calibration_logits / temperature, calibration_targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(0.05, 20.0).item())
    return temperature, calibration_logits, calibration_targets


def clopper_pearson_upper(events, total, alpha=0.05):
    events, total = int(events), int(total)
    if total <= 0:
        return 1.0
    if events >= total:
        return 1.0
    return float(beta.ppf(1.0 - float(alpha), events + 1, total - events))


def clopper_pearson_lower(events, total, alpha=0.05):
    events, total = int(events), int(total)
    if total <= 0 or events <= 0:
        return 0.0
    return float(beta.ppf(float(alpha), events, total - events + 1))


def safe_rate(numerator_mask, denominator_mask):
    denominator = int(np.asarray(denominator_mask, dtype=bool).sum())
    if denominator == 0:
        return float("nan"), 0, 0
    numerator = int((np.asarray(numerator_mask, dtype=bool) & denominator_mask).sum())
    return float(numerator / denominator), numerator, denominator


def group_event_counts(event_mask, denominator_mask, groups):
    event_mask = np.asarray(event_mask, dtype=bool)
    denominator_mask = np.asarray(denominator_mask, dtype=bool)
    groups = np.asarray(groups).astype(str)
    eligible_groups = np.unique(groups[denominator_mask])
    events = sum(
        bool((event_mask & denominator_mask & (groups == group)).any())
        for group in eligible_groups
    )
    return int(events), int(len(eligible_groups))


def compute_decision_metrics(
    y_true, probabilities, class_names, accept_by_class, groups=None
):
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    class_names = list(class_names)
    if probabilities.shape != (len(y_true), len(class_names)):
        raise ValueError("probabilities phải có shape [samples, classes]")
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities[np.arange(len(probabilities)), predicted]
    good_index = class_names.index("good") if "good" in class_names else class_names.index("ok")
    unknown_index = class_names.index("unknown") if "unknown" in class_names else None
    threshold_vector = np.asarray(
        [float(accept_by_class.get(name, 1.0)) for name in class_names], dtype=np.float64
    )
    auto = confidence >= threshold_vector[predicted]
    auto_good = auto & (predicted == good_index)
    auto_unknown = np.zeros(len(y_true), dtype=bool)
    if unknown_index is not None:
        auto_unknown = auto & (predicted == unknown_index)
    auto_defect = auto & (predicted != good_index)
    if unknown_index is not None:
        auto_defect &= predicted != unknown_index
    review = ~(auto_good | auto_defect | auto_unknown)

    truth_good = y_true == good_index
    truth_unknown = np.zeros(len(y_true), dtype=bool)
    if unknown_index is not None:
        truth_unknown = y_true == unknown_index
    truth_defect = ~(truth_good | truth_unknown)
    if groups is None:
        groups = np.arange(len(y_true)).astype(str)
    if len(groups) != len(y_true):
        raise ValueError("groups phải có cùng số phần tử với y_true")
    escape, escape_events, defect_total = safe_rate(auto_good, truth_defect)
    false_reject, false_reject_events, good_total = safe_rate(auto_defect, truth_good)
    good_review, _, _ = safe_rate(review, truth_good)
    defect_review, _, _ = safe_rate(review, truth_defect)
    invalid_good_accept, invalid_events, unknown_total = safe_rate(auto_good, truth_unknown)
    escape_group_events, defect_group_total = group_event_counts(
        auto_good, truth_defect, groups
    )
    invalid_good_group_events, unknown_group_total = group_event_counts(
        auto_good, truth_unknown, groups
    )
    false_reject_group_events, good_group_total = group_event_counts(
        auto_defect, truth_good, groups
    )
    return {
        "escape": escape,
        "false_reject": false_reject,
        "good_review": good_review,
        "defect_review": defect_review,
        "review_rate": float(review.mean()) if len(review) else float("nan"),
        "invalid_good_accept": invalid_good_accept,
        "auto_good_rate": float(auto_good.mean()) if len(auto_good) else float("nan"),
        "auto_defect_rate": float(auto_defect.mean()) if len(auto_defect) else float("nan"),
        "auto_unknown_rate": float(auto_unknown.mean()) if len(auto_unknown) else float("nan"),
        "escape_events": escape_events,
        "defect_total": defect_total,
        "false_reject_events": false_reject_events,
        "good_total": good_total,
        "invalid_good_accept_events": invalid_events,
        "unknown_total": unknown_total,
        "escape_group_events": escape_group_events,
        "defect_group_total": defect_group_total,
        "invalid_good_accept_group_events": invalid_good_group_events,
        "unknown_group_total": unknown_group_total,
        "false_reject_group_events": false_reject_group_events,
        "good_group_total": good_group_total,
        "decision": np.where(
            auto_good, "auto_good",
            np.where(auto_defect, "auto_defect", np.where(auto_unknown, "auto_unknown", "review")),
        ),
        "predicted_index": predicted,
        "confidence": confidence,
    }


def calibration_scores(probabilities, targets):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[targets]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        selected = (confidence >= lower) & (confidence < upper if upper < 1.0 else confidence <= 1.0)
        if selected.any():
            ece += selected.mean() * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
    return {
        "nll": float(log_loss(targets, probabilities, labels=range(probabilities.shape[1]))),
        "brier": brier,
        "ece": float(ece),
    }


def choose_operating_point(
    y_true, probabilities, class_names, escape_target, invalid_target,
    alpha=0.05, runtime_floor=0.80, minimum_precision_lcb=0.90,
    groups=None,
):
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    class_names = list(class_names)
    good_label = "good" if "good" in class_names else "ok"
    good_index = class_names.index(good_label)
    unknown_index = class_names.index("unknown") if "unknown" in class_names else None
    thresholds = {name: float(runtime_floor) for name in class_names}

    # Generic defect/unknown are advisory in current fusion, so never present them as an
    # auto-accepted subtype. Subtype profile chooses threshold from precision LCB.
    for index, name in enumerate(class_names):
        if name == good_label:
            continue
        if PROFILE["advisory_non_good"]:
            thresholds[name] = 1.0
            continue
        selected_threshold = 1.0
        for threshold in np.linspace(runtime_floor, 0.999, 200):
            predicted = probabilities.argmax(axis=1)
            accepted = (predicted == index) & (probabilities[:, index] >= threshold)
            total = int(accepted.sum())
            correct = int((accepted & (y_true == index)).sum())
            if total and clopper_pearson_lower(correct, total, alpha) >= minimum_precision_lcb:
                selected_threshold = float(threshold)
                break
        thresholds[name] = selected_threshold

    sweep = []
    for threshold in np.linspace(runtime_floor, 0.999, 200):
        candidate = dict(thresholds)
        candidate[good_label] = float(threshold)
        metrics = compute_decision_metrics(
            y_true, probabilities, class_names, candidate, groups=groups
        )
        escape_upper = clopper_pearson_upper(
            metrics["escape_group_events"], metrics["defect_group_total"], alpha
        )
        invalid_upper = clopper_pearson_upper(
            metrics["invalid_good_accept_group_events"], metrics["unknown_group_total"], alpha
        ) if unknown_index is not None else 0.0
        feasible = escape_upper <= escape_target and invalid_upper <= invalid_target
        sweep.append({
            "accept": float(threshold),
            "escape": metrics["escape"],
            "escape_ci_upper": escape_upper,
            "false_reject": metrics["false_reject"],
            "good_review": metrics["good_review"],
            "defect_review": metrics["defect_review"],
            "review_rate": metrics["review_rate"],
            "invalid_good_accept": metrics["invalid_good_accept"],
            "invalid_good_accept_ci_upper": invalid_upper,
            "auto_good_rate": metrics["auto_good_rate"],
            "feasible": bool(feasible),
        })
    feasible_rows = [row for row in sweep if row["feasible"]]
    if feasible_rows:
        selected = max(feasible_rows, key=lambda row: (row["auto_good_rate"], -row["review_rate"]))
        feasible = True
    else:
        selected = sweep[-1]
        feasible = False
    thresholds[good_label] = float(selected["accept"])
    return thresholds, pd.DataFrame(sweep), feasible


minimum_zero_event_samples = math.ceil(
    math.log(float(CONFIG["ci_alpha"])) / math.log(1.0 - float(CONFIG["escape_target"]))
)
print("Số board defect độc lập tối thiểu nếu quan sát 0 escape:", minimum_zero_event_samples)

calibration_logits_raw, calibration_targets_raw, calibration_row_ids = predict(
    loaders["calibration"], model
)
calibration_lookup = (
    frames_by_split["calibration"].set_index("row_id").loc[calibration_row_ids].reset_index()
)
temperature, calibration_logits, calibration_targets = fit_temperature(
    calibration_logits_raw, calibration_targets_raw
)
probabilities_before = torch.softmax(calibration_logits, dim=1).numpy()
calibration_probabilities = torch.softmax(calibration_logits / temperature, dim=1).numpy()
calibration_report = {
    "temperature": temperature,
    "before": calibration_scores(probabilities_before, calibration_targets.numpy()),
    "after": calibration_scores(calibration_probabilities, calibration_targets.numpy()),
}
calibration_report["temperature_at_bound"] = bool(
    temperature <= 0.0501 or temperature >= 19.999
)
calibration_report["nll_improved_or_equal"] = bool(
    calibration_report["after"]["nll"] <= calibration_report["before"]["nll"] + 1e-8
)
accept_by_class, threshold_sweep, calibration_feasible = choose_operating_point(
    calibration_targets.numpy(), calibration_probabilities, CLASS_NAMES,
    float(CONFIG["escape_target"]), float(CONFIG["invalid_good_accept_target"]),
    alpha=float(CONFIG["ci_alpha"]),
    runtime_floor=float(CONFIG["runtime_accept_floor"]),
    minimum_precision_lcb=float(CONFIG["minimum_auto_decision_precision_lcb"]),
    groups=calibration_lookup["physical_board"].to_numpy(),
)
threshold_selection_policy = "production_ci"
if RUN_MODE == "public_bootstrap" and not calibration_feasible:
    proxy_target = float(CONFIG["public_proxy_escape_target"])
    proxy_candidates = threshold_sweep[
        threshold_sweep["escape"].notna() & threshold_sweep["escape"].le(proxy_target)
    ]
    if proxy_candidates.empty:
        proxy_candidates = threshold_sweep.sort_values(
            ["escape", "auto_good_rate"], ascending=[True, False], na_position="last"
        ).head(1)
    selected_proxy = proxy_candidates.sort_values(
        ["auto_good_rate", "review_rate"], ascending=[False, True]
    ).iloc[0]
    accept_by_class[GOOD_LABEL] = float(selected_proxy["accept"])
    threshold_selection_policy = "public_proxy_point_estimate"
threshold_sweep.to_csv(REPORT_DIR / "calibration_threshold_sweep.csv", index=False)
calibration_report["accept_by_class"] = accept_by_class
calibration_report["operating_point_feasible"] = calibration_feasible
calibration_report["threshold_selection_policy"] = threshold_selection_policy
calibration_report["provisional"] = RUN_MODE == "public_bootstrap"
calibration_quality_ok = bool(
    calibration_feasible
    and not calibration_report["temperature_at_bound"]
    and calibration_report["nll_improved_or_equal"]
)
calibration_contract_ok = bool(
    math.isfinite(temperature)
    and temperature > 0
    and all(math.isfinite(float(value)) for value in accept_by_class.values())
)
calibration_report["quality_ok"] = calibration_quality_ok
calibration_report["contract_ok"] = calibration_contract_ok
(REPORT_DIR / "calibration_report.json").write_text(
    json_text(calibration_report), encoding="utf-8"
)
print(json_text(calibration_report))

# %% [markdown]
# ## 9. Evaluation holdout — chạy một lần sau khi khóa model, T và threshold
#
# Confusion matrix/per-class F1 đo khả năng phân loại. Decision metrics đo đúng tác động
# dự kiến. Trong public mode đây chỉ là public proxy, không phải metric trên line.

# %%
test_logits, test_targets_tensor, test_row_ids = predict(loaders["test"], model)
test_probabilities = torch.softmax(test_logits / temperature, dim=1).numpy()
test_targets = test_targets_tensor.numpy()
test_predicted = test_probabilities.argmax(axis=1)
test_lookup = frames_by_split["test"].set_index("row_id").loc[test_row_ids].reset_index()
decision_metrics = compute_decision_metrics(
    test_targets, test_probabilities, CLASS_NAMES, accept_by_class,
    groups=test_lookup["physical_board"].to_numpy(),
)
test_classification = classification_report(
    test_targets, test_predicted, labels=range(len(CLASS_NAMES)), target_names=CLASS_NAMES,
    output_dict=True, zero_division=0,
)
test_calibration = calibration_scores(test_probabilities, test_targets)
test_matrix = confusion_matrix(test_targets, test_predicted, labels=range(len(CLASS_NAMES)))

test_metrics = {
    key: value for key, value in decision_metrics.items()
    if key not in {"decision", "predicted_index", "confidence"}
}
test_metrics.update({
    "escape_ci_upper": clopper_pearson_upper(
        decision_metrics["escape_group_events"], decision_metrics["defect_group_total"],
        float(CONFIG["ci_alpha"]),
    ),
    "invalid_good_accept_ci_upper": clopper_pearson_upper(
        decision_metrics["invalid_good_accept_group_events"],
        decision_metrics["unknown_group_total"],
        float(CONFIG["ci_alpha"]),
    ) if UNKNOWN_INDEX is not None else 0.0,
    "false_reject_ci_upper": clopper_pearson_upper(
        decision_metrics["false_reject_group_events"],
        decision_metrics["good_group_total"],
        float(CONFIG["ci_alpha"]),
    ),
    "macro_f1": float(f1_score(
        test_targets, test_predicted, labels=range(len(CLASS_NAMES)),
        average="macro", zero_division=0,
    )),
    "defect_recall": float(
        np.mean(test_predicted[
            (test_targets != GOOD_INDEX)
            & (True if UNKNOWN_INDEX is None else test_targets != UNKNOWN_INDEX)
        ] != GOOD_INDEX)
    ),
    **test_calibration,
})

predictions_test = test_lookup[[
    "row_id", "pixel_sha256", "defect_class", "target_label", "physical_board", "capture_id",
    "dataset_source", "camera_id", "lot_id", "preprocess_id",
]].copy()
predictions_test["crop_name"] = test_lookup["crop_path"].map(lambda value: Path(value).name)
predictions_test["evaluation_subtype"] = predictions_test["defect_class"].map(
    lambda value: COMPONENT_LABELS.get(str(value), str(value))
    if PROFILE["scope"] == "component" else str(value)
)
predictions_test["predicted_label"] = [CLASS_NAMES[index] for index in test_predicted]
predictions_test["confidence"] = decision_metrics["confidence"]
predictions_test["decision"] = decision_metrics["decision"]
for index, name in enumerate(CLASS_NAMES):
    predictions_test[f"p_{name}"] = test_probabilities[:, index]
PREDICTIONS_FILENAME = (
    "predictions_public_proxy_holdout.csv"
    if RUN_MODE == "public_bootstrap" else "predictions_test.csv"
)
predictions_test.to_csv(REPORT_DIR / PREDICTIONS_FILENAME, index=False)

# Dù profile mặc định gộp defect, vẫn đo safety riêng theo nhãn thô để một subtype như
# cold/insufficient không bị điểm tổng che khuất. Đây là recall/escape gate, không giả vờ
# model 3-class có thể xuất đúng tên subtype.
subtype_rows = []
defect_truth_mask = ~predictions_test["target_label"].isin([GOOD_LABEL, "unknown"])
for subtype, subtype_indices in predictions_test[defect_truth_mask].groupby("evaluation_subtype").groups.items():
    positions = predictions_test.index.get_indexer(subtype_indices)
    subtype_metrics = compute_decision_metrics(
        test_targets[positions], test_probabilities[positions], CLASS_NAMES,
        accept_by_class,
        groups=predictions_test.loc[subtype_indices, "physical_board"].to_numpy(),
    )
    subtype_predicted = test_predicted[positions]
    subtype_rows.append({
        "defect_class": str(subtype),
        "roi_count": int(len(positions)),
        "board_count": int(predictions_test.loc[subtype_indices, "physical_board"].nunique()),
        "defect_detection_recall": float(np.mean(subtype_predicted != GOOD_INDEX)),
        "escape": float(subtype_metrics["escape"]),
        "escape_group_events": int(subtype_metrics["escape_group_events"]),
        "defect_group_total": int(subtype_metrics["defect_group_total"]),
        "escape_ci_upper": clopper_pearson_upper(
            subtype_metrics["escape_group_events"], subtype_metrics["defect_group_total"],
            float(CONFIG["ci_alpha"]),
        ),
        "review_rate": float(subtype_metrics["review_rate"]),
    })
subtype_safety = pd.DataFrame(subtype_rows)
subtype_safety.to_csv(REPORT_DIR / "subtype_safety.csv", index=False)
display(subtype_safety)

figure, axis = plt.subplots(figsize=(7, 6))
image = axis.imshow(test_matrix, cmap="Blues")
axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=35, ha="right")
axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
axis.set_xlabel("Predicted")
axis.set_ylabel("Ground truth")
for row in range(len(CLASS_NAMES)):
    for column in range(len(CLASS_NAMES)):
        axis.text(column, row, int(test_matrix[row, column]), ha="center", va="center")
figure.colorbar(image, ax=axis)
figure.tight_layout()
figure.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
plt.show()

confidence = test_probabilities.max(axis=1)
correct = test_predicted == test_targets
bins = np.linspace(0.0, 1.0, 11)
reliability_rows = []
for lower, upper in zip(bins[:-1], bins[1:]):
    selected = (confidence >= lower) & (confidence < upper if upper < 1.0 else confidence <= 1.0)
    if selected.any():
        reliability_rows.append({
            "confidence": float(confidence[selected].mean()),
            "accuracy": float(correct[selected].mean()),
            "count": int(selected.sum()),
        })
reliability = pd.DataFrame(reliability_rows)
figure, axis = plt.subplots(figsize=(6, 6))
axis.plot([0, 1], [0, 1], "--", color="gray")
if not reliability.empty:
    axis.plot(reliability["confidence"], reliability["accuracy"], marker="o")
axis.set(xlabel="Mean confidence", ylabel="Accuracy", xlim=(0, 1), ylim=(0, 1))
figure.tight_layout()
figure.savefig(REPORT_DIR / "reliability_curve.png", dpi=160)
plt.show()

def slice_decision_report(predictions, probabilities, targets, column):
    rows = []
    for value, indices in predictions.groupby(column).groups.items():
        positions = predictions.index.get_indexer(indices)
        metrics = compute_decision_metrics(
            targets[positions], probabilities[positions], CLASS_NAMES, accept_by_class
        )
        rows.append({
            column: value, "samples": len(positions),
            **{key: metrics[key] for key in (
                "escape", "false_reject", "good_review", "defect_review",
                "review_rate", "invalid_good_accept",
            )},
        })
    return pd.DataFrame(rows)


slice_reports = {}
for slice_column in ("dataset_source", "camera_id", "lot_id", "physical_board"):
    report = slice_decision_report(predictions_test, test_probabilities, test_targets, slice_column)
    report.to_csv(REPORT_DIR / f"slice_{slice_column}.csv", index=False)
    slice_reports[slice_column] = report.to_dict(orient="records")


def cluster_bootstrap_by_board(predictions, probabilities, targets, iterations=1000, seed=42):
    boards = predictions["physical_board"].astype(str).unique()
    if len(boards) < 2:
        return {}
    rng = np.random.default_rng(seed)
    samples = defaultdict(list)
    for _ in range(int(iterations)):
        selected_boards = rng.choice(boards, size=len(boards), replace=True)
        positions = np.concatenate([
            np.flatnonzero(predictions["physical_board"].astype(str).to_numpy() == board)
            for board in selected_boards
        ])
        metrics = compute_decision_metrics(
            targets[positions], probabilities[positions], CLASS_NAMES, accept_by_class
        )
        for key in ("escape", "false_reject", "review_rate", "invalid_good_accept"):
            if np.isfinite(metrics[key]):
                samples[key].append(metrics[key])
    return {
        key: {
            "p2_5": float(np.quantile(values, 0.025)),
            "median": float(np.quantile(values, 0.50)),
            "p97_5": float(np.quantile(values, 0.975)),
        }
        for key, values in samples.items() if values
    }


bootstrap_intervals = cluster_bootstrap_by_board(
    predictions_test, test_probabilities, test_targets,
    iterations=int(CONFIG["bootstrap_iterations"]), seed=SEED,
)
evaluation_report = {
    "schema": "aoi-solder-evaluation/2.0",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "run_mode": RUN_MODE,
    "taxonomy_profile": TAXONOMY_PROFILE,
    "class_names": CLASS_NAMES,
    "evaluation_role": EVALUATION_ROLE,
    "evaluation_domain": EVALUATION_DOMAIN,
    "is_production_evidence": bool(MODE_POLICY["production_allowed"]),
    "group_semantics": (
        "source_image_proxy_not_physical_board"
        if RUN_MODE == "public_bootstrap" else "physical_aoi_board"
    ),
    "locked_test_explicit": locked_test_explicit,
    "evaluation_metrics": test_metrics,
    "classification_report": test_classification,
    "confusion_matrix": test_matrix.tolist(),
    "bootstrap_by_board": bootstrap_intervals,
    "subtype_safety": subtype_rows,
    "slices": slice_reports,
    "calibration": calibration_report,
    "data_fingerprint": DATASET_FINGERPRINT,
    "split_hash": SPLIT_HASH,
}
(REPORT_DIR / "evaluation_report.json").write_text(
    json_text(evaluation_report), encoding="utf-8"
)
print(json_text(test_metrics))

# %% [markdown]
# ## 10. Hard-negative / OOD controls
#
# `unknown` production phải là crop sai/background thật từ cùng camera. Synthetic
# flat/noise dưới đây chỉ là smoke test, không được biến thành nhãn train hay dùng chọn
# threshold. Public bootstrap cố ý không có class `unknown`.

# %%
def predict_rgb_images(images):
    batch = torch.from_numpy(np.stack([preprocess_rgb(image) for image in images])).to(DEVICE)
    with torch.no_grad():
        logits = model(batch).cpu()
    return torch.softmax(logits / temperature, dim=1).numpy()


rng = np.random.default_rng(SEED)
synthetic_controls = []
for value in (0, 24, 64, 128, 192, 255):
    synthetic_controls.append(np.full((96, 128, 3), value, dtype=np.uint8))
for _ in range(24):
    synthetic_controls.append(rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8))
synthetic_probabilities = predict_rgb_images(synthetic_controls)
synthetic_predicted = synthetic_probabilities.argmax(axis=1)
synthetic_confidence = synthetic_probabilities.max(axis=1)
synthetic_auto_good = (
    (synthetic_predicted == GOOD_INDEX)
    & (synthetic_confidence >= float(accept_by_class[GOOD_LABEL]))
)
ood_control_report = {
    "note": (
        "Synthetic smoke test only; public bootstrap has no unknown class. Camera "
        "fine-tune must use real locked-test unknown crops."
    ),
    "run_mode": RUN_MODE,
    "samples": len(synthetic_controls),
    "auto_good": int(synthetic_auto_good.sum()),
    "auto_good_rate": float(synthetic_auto_good.mean()),
    "top_labels": Counter(CLASS_NAMES[index] for index in synthetic_predicted),
}
ood_control_report["top_labels"] = dict(ood_control_report["top_labels"])
(REPORT_DIR / "ood_control_report.json").write_text(
    json_text(ood_control_report), encoding="utf-8"
)
print(json_text(ood_control_report))

# %% [markdown]
# ## 11. Export ONNX candidate và parity trên crop thật
#
# Candidate được export trước để parity trở thành một quality gate. Batch 1/3/32 (hoặc tối
# đa số mẫu test đang có) đều phải khớp. Model ONNX phải tự chứa, không để lại `.data`.

# %%
def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collapse_external_data(path):
    path = Path(path)
    loaded = onnx.load(str(path), load_external_data=True)
    onnx.save_model(loaded, str(path), save_as_external_data=False)
    for orphan in path.parent.glob(f"{path.name}*.data"):
        orphan.unlink()
    for orphan in path.parent.glob(f"{path.stem}*.data"):
        orphan.unlink()


model_cpu = model.to("cpu").eval()
parity_limit = min(int(CONFIG["onnx_parity_samples"]), len(test_lookup))
per_class_limit = max(1, math.ceil(parity_limit / len(CLASS_NAMES)))
parity_frame_parts = []
for class_name in CLASS_NAMES:
    class_part = test_lookup[test_lookup["target_label"].eq(class_name)]
    # Ưu tiên board khác nhau, sau đó mới bổ sung ROI cùng board nếu cần.
    distinct = class_part.drop_duplicates("physical_board", keep="first")
    selected = distinct.sample(
        min(per_class_limit, len(distinct)), random_state=SEED
    ) if len(distinct) else distinct
    if len(selected) < min(per_class_limit, len(class_part)):
        remainder = class_part[~class_part["row_id"].isin(selected["row_id"])]
        selected = pd.concat([
            selected,
            remainder.sample(
                min(per_class_limit - len(selected), len(remainder)), random_state=SEED
            ),
        ])
    parity_frame_parts.append(selected)
parity_frame = pd.concat(parity_frame_parts, ignore_index=True).drop_duplicates("row_id")
if len(parity_frame) < parity_limit:
    remainder = test_lookup[~test_lookup["row_id"].isin(parity_frame["row_id"])]
    parity_frame = pd.concat([
        parity_frame,
        remainder.sample(min(parity_limit - len(parity_frame), len(remainder)), random_state=SEED),
    ], ignore_index=True)
parity_frame = parity_frame.head(parity_limit).reset_index(drop=True)
test_dataset_for_parity = SolderV2Dataset(frames_by_split["test"], training=False)
dataset_position = {
    int(row_id): position
    for position, row_id in enumerate(frames_by_split["test"]["row_id"].tolist())
}
parity_items = [
    test_dataset_for_parity[dataset_position[int(row_id)]]
    for row_id in parity_frame["row_id"]
]
representative_batch = torch.stack([item[0] for item in parity_items])
representative_targets = np.asarray([item[1] for item in parity_items], dtype=np.int64)
representative_groups = parity_frame["physical_board"].astype(str).to_numpy()
if representative_batch.shape[0] < 3:
    raise RuntimeError("Cần ít nhất 3 crop test thật để kiểm tra ONNX dynamic batch.")

candidate_onnx = CANDIDATE_DIR / "candidate.onnx"
export_kwargs = {
    "input_names": ["input"],
    "output_names": ["logits"],
    "dynamic_axes": {"input": {0: "batch"}, "logits": {0: "batch"}},
    "opset_version": int(CONFIG["onnx_opset"]),
}
try:
    torch.onnx.export(
        model_cpu, representative_batch[:1], str(candidate_onnx),
        dynamo=False, **export_kwargs,
    )
    onnx_exporter = "torchscript"
except Exception as first_error:
    print("TorchScript exporter lỗi, thử dynamo:", first_error)
    torch.onnx.export(
        model_cpu, representative_batch[:1], str(candidate_onnx),
        dynamo=True, **export_kwargs,
    )
    onnx_exporter = "dynamo"

collapse_external_data(candidate_onnx)
onnx_model = onnx.load(str(candidate_onnx))
onnx.checker.check_model(onnx_model)
external_data_files = sorted(candidate_onnx.parent.glob("*.data"))
if external_data_files:
    raise RuntimeError(f"ONNX còn external data: {external_data_files}")

session = ort.InferenceSession(str(candidate_onnx), providers=["CPUExecutionProvider"])
parity_rows = []
for batch_size in sorted({1, 3, int(representative_batch.shape[0])}):
    sample = representative_batch[:batch_size]
    with torch.no_grad():
        torch_logits = model_cpu(sample).numpy()
    onnx_logits = session.run(["logits"], {"input": sample.numpy()})[0]
    difference = np.abs(torch_logits - onnx_logits)
    torch_probabilities = torch.softmax(torch.from_numpy(torch_logits) / temperature, dim=1).numpy()
    onnx_probabilities = torch.softmax(torch.from_numpy(onnx_logits) / temperature, dim=1).numpy()
    torch_decisions = compute_decision_metrics(
        representative_targets[:batch_size], torch_probabilities, CLASS_NAMES,
        accept_by_class, groups=representative_groups[:batch_size],
    )["decision"]
    onnx_decisions = compute_decision_metrics(
        representative_targets[:batch_size], onnx_probabilities, CLASS_NAMES,
        accept_by_class, groups=representative_groups[:batch_size],
    )["decision"]
    parity_rows.append({
        "batch_size": batch_size,
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "max_probability_error": float(np.abs(torch_probabilities - onnx_probabilities).max()),
        "argmax_equal": bool(np.array_equal(
            torch_probabilities.argmax(axis=1), onnx_probabilities.argmax(axis=1)
        )),
        "decision_equal": bool(np.array_equal(torch_decisions, onnx_decisions)),
    })
parity_max_abs_error = max(row["max_abs_error"] for row in parity_rows)
parity_pass = bool(
    parity_max_abs_error <= float(CONFIG["onnx_parity_atol"])
    and all(row["argmax_equal"] and row["decision_equal"] for row in parity_rows)
)
onnx_verification = {
    "exporter": onnx_exporter,
    "opset": int(onnx_model.opset_import[0].version),
    "representative_source": (
        "public_proxy_holdout_crops"
        if RUN_MODE == "public_bootstrap" else "camera_locked_test_real_crops"
    ),
    "representative_sampling": (
        "stratified_by_class_and_distinct_source_image_proxy"
        if RUN_MODE == "public_bootstrap"
        else "stratified_by_class_and_distinct_physical_board"
    ),
    "representative_class_counts": parity_frame["target_label"].value_counts().to_dict(),
    "representative_boards": int(parity_frame["physical_board"].nunique()),
    "parity": parity_rows,
    "parity_max_abs_error": parity_max_abs_error,
    "parity_pass": parity_pass,
    "self_contained": not external_data_files,
}
(REPORT_DIR / "onnx_verification.json").write_text(
    json_text(onnx_verification), encoding="utf-8"
)
print(json_text(onnx_verification))

# %% [markdown]
# ## 12. Quality gate và artifact
#
# Public mode dùng **bootstrap integrity gate**: leakage, calibration contract và ONNX
# parity phải hợp lệ để tạo cặp file nạp app, nhưng `production_ready` luôn false vì chưa
# có camera. Camera mode mới chạy full quality gate board-level hiện có.

# %%
def bootstrap_integrity_gate(
    audit_report, split_audit, frame, calibration_contract_is_ok,
    parity_error, parity_ok, config,
):
    failures, warnings_list = [], []
    if audit_report.get("conflicting_labels"):
        failures.append("conflicting_labels")
    if audit_report.get("split_assignment_conflicts"):
        failures.append("split_assignment_conflicts")
    for key in (
        "cross_split_board_leakage", "cross_split_sha256_leakage",
        "cross_split_phash_leakage",
    ):
        if split_audit.get(key):
            failures.append(key)
    if not calibration_contract_is_ok:
        failures.append("calibration_contract_invalid")
    if not parity_ok or parity_error > float(config["onnx_parity_atol"]):
        failures.append("onnx_parity_failed")
    for split in ("val", "calibration", "test"):
        counts = frame.loc[frame["split"].eq(split), "target_label"].value_counts()
        for name in CLASS_NAMES:
            if int(counts.get(name, 0)) < 1:
                failures.append(f"{split}_missing_class[{name}]")
    blockers = [
        "no_camera_domain_data",
        "no_camera_locked_test",
        "public_proxy_metrics_not_deployment_evidence",
        "no_real_camera_unknown_wrong_crop_class",
    ]
    warnings_list.extend([
        "PUBLIC_PROXY metrics can compare experiments but cannot validate AOI deployment",
        "threshold and temperature are provisional until camera calibration",
        "source-image groups are not physical-board/lot groups",
    ])
    if split_audit.get("cross_split_phash_candidates"):
        warnings_list.append(
            "near-pHash lookalike candidates cross proxy splits; they are reported, not "
            "deleted/grouped automatically because solder joints repeat visually"
        )
    if ood_control_report["auto_good"]:
        warnings_list.append("synthetic_ood_has_auto_good; inspect after camera fine-tune")
    return {
        "gate_type": "public_bootstrap_integrity",
        "artifact_loadable": not failures,
        "production_ready": False,
        "failures": failures,
        "production_blockers": blockers,
        "warnings": warnings_list,
        "parity_max_abs_error": float(parity_error),
        "parity_pass": bool(parity_ok),
    }


def quality_gate(
    audit_report, split_audit, metrics, frame, calibration_ok,
    parity_error, parity_ok, locked_test_is_explicit, config,
    subtype_safety_rows, classification_details, expected_raw_subtypes,
):
    failures, warnings_list = [], []

    if audit_report.get("conflicting_labels"):
        failures.append("conflicting_labels")
    if audit_report.get("split_assignment_conflicts"):
        failures.append("split_assignment_conflicts")
    for key in (
        "cross_split_board_leakage", "cross_split_sha256_leakage",
        "cross_split_phash_leakage",
    ):
        if split_audit.get(key):
            failures.append(key)
    if not locked_test_is_explicit:
        failures.append("locked_test_not_explicit")
    if not calibration_ok:
        failures.append("no_feasible_calibration_operating_point")
    if not parity_ok or parity_error > float(config["onnx_parity_atol"]):
        failures.append("onnx_parity_failed")

    test_frame = frame[frame["split"].eq("test")]
    defect_labels = [name for name in CLASS_NAMES if name not in {GOOD_LABEL, "unknown"}]
    test_defects = int(test_frame["target_label"].isin(defect_labels).sum())
    test_good = int(test_frame["target_label"].eq(GOOD_LABEL).sum())
    test_unknown = int(test_frame["target_label"].eq("unknown").sum()) if "unknown" in CLASS_NAMES else 0
    test_boards = int(test_frame["physical_board"].nunique())
    defect_boards = int(
        test_frame.loc[test_frame["target_label"].isin(defect_labels), "physical_board"].nunique()
    )
    good_boards = int(
        test_frame.loc[test_frame["target_label"].eq(GOOD_LABEL), "physical_board"].nunique()
    )
    unknown_boards = int(
        test_frame.loc[test_frame["target_label"].eq("unknown"), "physical_board"].nunique()
    ) if "unknown" in CLASS_NAMES else 0
    if test_defects < int(config["minimum_test_defects"]):
        failures.append(f"test_defects={test_defects}<{config['minimum_test_defects']}")
    if test_good < int(config["minimum_test_good"]):
        failures.append(f"test_good={test_good}<{config['minimum_test_good']}")
    if "unknown" in CLASS_NAMES and test_unknown < int(config["minimum_test_unknown"]):
        failures.append(f"test_unknown={test_unknown}<{config['minimum_test_unknown']}")
    if test_boards < int(config["minimum_test_boards"]):
        failures.append(f"test_boards={test_boards}<{config['minimum_test_boards']}")
    if defect_boards < int(config["minimum_test_defect_boards"]):
        failures.append(
            f"test_defect_boards={defect_boards}<{config['minimum_test_defect_boards']}"
        )
    if good_boards < int(config["minimum_test_good_boards"]):
        failures.append(f"test_good_boards={good_boards}<{config['minimum_test_good_boards']}")
    if "unknown" in CLASS_NAMES and unknown_boards < int(config["minimum_test_unknown_boards"]):
        failures.append(
            f"test_unknown_boards={unknown_boards}<{config['minimum_test_unknown_boards']}"
        )

    if metrics["escape_ci_upper"] > float(config["escape_target"]):
        failures.append(
            f"escape_ci_upper={metrics['escape_ci_upper']:.6f}>{config['escape_target']}"
        )
    if metrics["invalid_good_accept_ci_upper"] > float(config["invalid_good_accept_target"]):
        failures.append(
            "invalid_good_accept_ci_upper="
            f"{metrics['invalid_good_accept_ci_upper']:.6f}>{config['invalid_good_accept_target']}"
        )
    if metrics["false_reject_ci_upper"] > float(config["maximum_false_reject"]):
        failures.append(
            f"false_reject_ci_upper={metrics['false_reject_ci_upper']:.6f}>"
            f"{config['maximum_false_reject']}"
        )
    if metrics["macro_f1"] < float(config["minimum_macro_f1"]):
        failures.append(f"macro_f1={metrics['macro_f1']:.6f}<{config['minimum_macro_f1']}")
    if metrics["defect_recall"] < float(config["minimum_defect_recall"]):
        failures.append(
            f"defect_recall={metrics['defect_recall']:.6f}<{config['minimum_defect_recall']}"
        )
    if metrics["good_review"] > float(config["max_good_review_rate"]):
        failures.append(
            f"good_review={metrics['good_review']:.6f}>{config['max_good_review_rate']}"
        )
    if audit_report.get("physical_board_id_missing", 0):
        failures.append("physical_board_id_missing")
    if audit_report.get("verified_label_status_missing", 0):
        failures.append("verified_label_status_missing")
    if audit_report.get("canonical_contract_incomplete", 0):
        failures.append("canonical_contract_incomplete")
    if audit_report.get("preprocess_id_mismatch", 0):
        failures.append("production_preprocess_id_mismatch")
    if audit_report.get("local_label_retained_ratio", 0.0) < float(
        config["minimum_local_label_retained_ratio"]
    ):
        failures.append(
            "local_label_retained_ratio="
            f"{audit_report.get('local_label_retained_ratio', 0.0):.6f}<"
            f"{config['minimum_local_label_retained_ratio']}"
        )
    if audit_report.get("invalid_image_ratio", 1.0) > float(config["maximum_invalid_image_ratio"]):
        failures.append(
            f"invalid_image_ratio={audit_report.get('invalid_image_ratio', 1.0):.6f}>"
            f"{config['maximum_invalid_image_ratio']}"
        )
    if int(split_audit.get("public_rows_outside_train", 0)):
        failures.append("public_rows_outside_train")

    for split in ("val", "calibration", "test"):
        group_support = (
            frame.loc[frame["split"].eq(split)]
            .groupby("target_label")["physical_board"].nunique().to_dict()
        )
        for name in CLASS_NAMES:
            if int(group_support.get(name, 0)) < int(config["minimum_groups_per_class_eval"]):
                failures.append(
                    f"{split}_groups[{name}]={group_support.get(name, 0)}<"
                    f"{config['minimum_groups_per_class_eval']}"
                )
    observed_subtypes = {str(row["defect_class"]) for row in subtype_safety_rows}
    for missing_subtype in sorted(set(expected_raw_subtypes) - observed_subtypes):
        failures.append(f"missing_locked_test_subtype[{missing_subtype}]")
    for row in subtype_safety_rows:
        name = row["defect_class"]
        if int(row["board_count"]) < int(config["minimum_subtype_boards"]):
            failures.append(
                f"subtype_boards[{name}]={row['board_count']}<{config['minimum_subtype_boards']}"
            )
        if float(row["defect_detection_recall"]) < float(config["minimum_subtype_defect_recall"]):
            failures.append(
                f"subtype_recall[{name}]={row['defect_detection_recall']:.6f}<"
                f"{config['minimum_subtype_defect_recall']}"
            )
        if float(row["escape_ci_upper"]) > float(config["subtype_escape_target"]):
            failures.append(
                f"subtype_escape_ci_upper[{name}]={row['escape_ci_upper']:.6f}>"
                f"{config['subtype_escape_target']}"
            )
    for name in CLASS_NAMES:
        class_f1 = float(classification_details.get(name, {}).get("f1-score", 0.0))
        if class_f1 < float(config["minimum_per_class_f1"]):
            failures.append(
                f"class_f1[{name}]={class_f1:.6f}<{config['minimum_per_class_f1']}"
            )
    if ood_control_report["auto_good"]:
        warnings_list.append("synthetic_ood_has_auto_good; inspect real unknown controls")
    warnings_list.append(
        "total review_rate is reported but not capacity-gated because locked-test is "
        "defect/unknown-enriched; good_review is the normal-production capacity proxy"
    )
    return {
        "production_ready": not failures,
        "failures": failures,
        "warnings": warnings_list,
        "counts": {
            "test_defects": test_defects, "test_good": test_good,
            "test_unknown": test_unknown, "test_boards": test_boards,
            "test_defect_boards": defect_boards, "test_good_boards": good_boards,
            "test_unknown_boards": unknown_boards,
        },
        "parity_max_abs_error": float(parity_error),
        "parity_pass": bool(parity_ok),
        "escape_ci_upper": float(metrics["escape_ci_upper"]),
        "invalid_good_accept_ci_upper": float(metrics["invalid_good_accept_ci_upper"]),
    }


if RUN_MODE == "public_bootstrap":
    gate = bootstrap_integrity_gate(
        dataset_audit, split_report, split_manifest,
        calibration_contract_ok, parity_max_abs_error, parity_pass, CONFIG,
    )
else:
    gate = quality_gate(
        dataset_audit, split_report, test_metrics, split_manifest,
        calibration_quality_ok, parity_max_abs_error, parity_pass,
        locked_test_explicit, CONFIG, subtype_rows, test_classification,
        EXPECTED_RAW_SUBTYPES,
    )
evaluation_report["quality_gate"] = gate
(REPORT_DIR / "evaluation_report.json").write_text(
    json_text(evaluation_report), encoding="utf-8"
)
(REPORT_DIR / "quality_gate.json").write_text(
    json_text(gate), encoding="utf-8"
)

source_class_counts = (
    split_manifest.groupby(["target_label", "dataset_source"]).size().unstack(fill_value=0)
)
single_source_classes = [
    label for label, row in source_class_counts.iterrows() if int((row > 0).sum()) == 1
]
selected_sweep_rows = threshold_sweep.iloc[::20].to_dict(orient="records")
selected_accept = float(accept_by_class[GOOD_LABEL])
if not any(abs(float(row["accept"]) - selected_accept) < 1e-12 for row in selected_sweep_rows):
    selected_sweep_rows.append(
        threshold_sweep.iloc[(threshold_sweep["accept"] - selected_accept).abs().argmin()].to_dict()
    )

def build_manifest(model_filename, model_sha256, production_ready, artifact_status):
    public_sources = []
    if isinstance(PUBLIC_SOURCE_INVENTORY, dict):
        for source in PUBLIC_SOURCE_INVENTORY.get("sources", []):
            public_sources.append({
                key: source.get(key) for key in (
                    "name", "provider", "handle", "adapter", "license", "homepage",
                    "citation", "roi_semantics", "record_count_before_crop",
                )
            })
    return {
        "schema_version": "pcb-solder-defect-classifier/1.0",
        "task": "solder_defect_classification",
        "scope": PROFILE["scope"],
        "model_format": "onnx",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_stage": RUN_MODE,
        "artifact_status": artifact_status,
        "deployment_status": (
            "production_validated" if production_ready else "bootstrap_or_candidate_only"
        ),
        "requires_camera_finetune": RUN_MODE == "public_bootstrap",
        "usage_policy": (
            "evaluation_and_review_assist_only"
            if RUN_MODE == "public_bootstrap" else "production_when_quality_gate_passes"
        ),
        "class_names": CLASS_NAMES,
        "good_label": GOOD_LABEL,
        "input": {
            "name": "input", "size": [INPUT_SIZE, INPUT_SIZE],
            "color_space": "RGB", "resize_mode": "letterbox",
            "letterbox_value": int(CONFIG["letterbox_value"]),
            "normalization": {"mean": MEAN.tolist(), "std": STD.tolist()},
        },
        "output": {"name": "logits", "type": "raw_logits"},
        "calibration": {
            "method": "temperature_scaling",
            "temperature": temperature,
            "provisional": RUN_MODE == "public_bootstrap",
            "domain": EVALUATION_DOMAIN,
            "threshold_selection_policy": threshold_selection_policy,
        },
        "decision_thresholds": {
            "accept": selected_accept,
            "review": min(float(CONFIG["review_threshold"]), selected_accept),
            "accept_by_class": {name: float(value) for name, value in accept_by_class.items()},
        },
        "model": {
            "filename": model_filename,
            "version": f"solder-{TAXONOMY_PROFILE}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}",
            "architecture": CONFIG["model_name"],
            "sha256": model_sha256,
        },
        "taxonomy_profile": TAXONOMY_PROFILE,
        "deployment_behavior": (
            "one_way_good_gate; defect/unknown remain conflict-review with current rule fusion"
            if PROFILE["advisory_non_good"] else "canonical subtype fusion"
        ),
        "data_fingerprint": DATASET_FINGERPRINT,
        "split_hash": SPLIT_HASH,
        "quality_gate": {**gate, "production_ready": bool(production_ready)},
        "data_provenance": {
            "public_sources": public_sources,
            "primary_source_kind": PRIMARY_SOURCE_KIND,
            "group_semantics": (
                "source_image_proxy_not_physical_board"
                if RUN_MODE == "public_bootstrap" else "physical_aoi_board"
            ),
        },
        "training": {
            "sources": sorted(split_manifest["dataset_source"].unique()),
            "class_counts": split_manifest["target_label"].value_counts().sort_index().to_dict(),
            "single_source_classes": sorted(single_source_classes),
            "groups_total": int(split_manifest["leakage_group"].nunique()),
            "roi_train": int(len(frames_by_split["train"])),
            "roi_val": int(len(frames_by_split["val"])),
            "roi_calibration": int(len(frames_by_split["calibration"])),
            "roi_test": int(len(frames_by_split["test"])),
            "best_epoch": int(best_epoch), "seed": SEED,
            "initialization": BOOTSTRAP_INITIALIZATION,
            "bootstrap_checkpoint": (
                "bootstrap_checkpoint.pt" if RUN_MODE == "public_bootstrap" else None
            ),
            "threshold_sweep": selected_sweep_rows,
        },
        "evaluation": {
            "role": EVALUATION_ROLE,
            "domain": EVALUATION_DOMAIN,
            "is_production_evidence": bool(production_ready),
            "metrics": test_metrics,
            "macro_f1": test_metrics["macro_f1"],
            "report_file": "evaluation_report.json",
            "predictions_file": PREDICTIONS_FILENAME,
        },
    }


candidate_manifest = build_manifest(
    candidate_onnx.name, sha256_file(candidate_onnx), production_ready=False,
    artifact_status="diagnostic_candidate",
)
(CANDIDATE_DIR / "model_manifest.candidate.json").write_text(
    json_text(candidate_manifest), encoding="utf-8"
)

bootstrap_checkpoint_path = None
if RUN_MODE == "public_bootstrap":
    bootstrap_checkpoint_path = BOOTSTRAP_DIR / "bootstrap_checkpoint.pt"
    torch.save({
        "schema_version": "pcb-solder-bootstrap-checkpoint/1.0",
        "training_stage": RUN_MODE,
        "architecture": CONFIG["model_name"],
        "class_names": CLASS_NAMES,
        "taxonomy_profile": TAXONOMY_PROFILE,
        "input_size": INPUT_SIZE,
        "state_dict": model_cpu.state_dict(),
        "data_fingerprint": DATASET_FINGERPRINT,
        "split_hash": SPLIT_HASH,
        "best_epoch": int(best_epoch),
    }, bootstrap_checkpoint_path)

report_files = sorted(path for path in REPORT_DIR.rglob("*") if path.is_file())
report_files.append(WORK_DIR / "split_manifest.csv")

candidate_zip = Path("/kaggle/working/pcb_solder_defect_v2_candidate_artifacts.zip")
with zipfile.ZipFile(candidate_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.write(candidate_onnx, "candidate.onnx")
    archive.write(CANDIDATE_DIR / "model_manifest.candidate.json", "model_manifest.candidate.json")
    if bootstrap_checkpoint_path is not None:
        archive.write(bootstrap_checkpoint_path, "bootstrap_checkpoint.pt")
    for path in report_files:
        archive.write(path, path.name)

bootstrap_zip = Path("/kaggle/working/pcb_solder_defect_v2_public_bootstrap_artifacts.zip")
production_zip = Path("/kaggle/working/pcb_solder_defect_v2_production_artifacts.zip")
if RUN_MODE == "public_bootstrap" and gate["artifact_loadable"]:
    bootstrap_onnx = BOOTSTRAP_DIR / "best.onnx"
    shutil.copy2(candidate_onnx, bootstrap_onnx)
    bootstrap_manifest = build_manifest(
        bootstrap_onnx.name, sha256_file(bootstrap_onnx), production_ready=False,
        artifact_status="bootstrap_only",
    )
    bootstrap_manifest_path = BOOTSTRAP_DIR / "model_manifest.json"
    bootstrap_manifest_path.write_text(
        json_text(bootstrap_manifest), encoding="utf-8"
    )
    if bootstrap_manifest_path.stat().st_size >= 1024 * 1024:
        raise RuntimeError("model_manifest.json vượt giới hạn runtime 1 MB")
    with zipfile.ZipFile(bootstrap_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(bootstrap_onnx, "best.onnx")
        archive.write(bootstrap_manifest_path, "model_manifest.json")
        archive.write(bootstrap_checkpoint_path, "bootstrap_checkpoint.pt")
        for path in report_files:
            archive.write(path, path.name)
    print("PUBLIC BOOTSTRAP COMPLETE (KHÔNG PHẢI PRODUCTION):", bootstrap_zip)
elif RUN_MODE == "public_bootstrap":
    print("BOOTSTRAP INTEGRITY FAIL — chỉ tạo candidate để debug.")
    for failure in gate["failures"]:
        print(" -", failure)
elif gate["production_ready"]:
    best_onnx = PRODUCTION_DIR / "best.onnx"
    shutil.copy2(candidate_onnx, best_onnx)
    production_manifest = build_manifest(
        best_onnx.name, sha256_file(best_onnx), production_ready=True,
        artifact_status="production_validated",
    )
    manifest_path = PRODUCTION_DIR / "model_manifest.json"
    manifest_path.write_text(
        json_text(production_manifest), encoding="utf-8"
    )
    if manifest_path.stat().st_size >= 1024 * 1024:
        raise RuntimeError("model_manifest.json vượt giới hạn runtime 1 MB")
    with zipfile.ZipFile(production_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(best_onnx, "best.onnx")
        archive.write(manifest_path, "model_manifest.json")
        for path in report_files:
            archive.write(path, path.name)
    print("PRODUCTION READY:", production_zip)
else:
    print("QUALITY GATE FAIL — không tạo best.onnx/model_manifest.json production.")
    for failure in gate["failures"]:
        print(" -", failure)
print("Candidate để phân tích:", candidate_zip)

# %% [markdown]
# ## 13. Cách dùng kết quả
#
# Với mode mặc định, tải `pcb_solder_defect_v2_public_bootstrap_artifacts.zip`. ZIP có:
#
# - `best.onnx`
# - `model_manifest.json`
# - `bootstrap_checkpoint.pt` (giữ lại để fine-tune khi có camera)
#
# Có thể nạp hai file đầu vào sidebar **Classifier ROI mối hàn · raw logits** để test.
# Model/threshold này chỉ hỗ trợ review và thử pipeline; manifest ghi rõ
# `artifact_status=bootstrap_only`, `production_ready=false`.
#
# Sau khi có camera, đổi `run_mode="camera_finetune"`, attach checkpoint + canonical CSV,
# khóa `split=test` theo board/lot và Run All. Chỉ khi đó notebook mới có thể tạo
# `pcb_solder_defect_v2_production_artifacts.zip`.
#
# Kiểm tra lại bằng runtime của dự án:
#
# ```powershell
# .\.venv\Scripts\python.exe scripts\verify_solder_model.py `
#   best.onnx model_manifest.json
# ```
#
# Nếu chỉ có candidate ZIP, đọc `quality_gate.json`: integrity/parity đã fail nên không
# dùng candidate trong app.
