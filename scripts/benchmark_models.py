"""Đo lại TOÀN BỘ model từ đầu, một lần chạy, cùng ảnh, cùng máy.

Vì sao viết lại thay vì dùng số cũ: bảng xếp hạng trước ghép số từ nhiều lần
chạy khác nhau (thời gian detector lấy cả "1.79 s" của hôm trước lẫn "2.26 s"
của hôm sau). Thời gian đo trên máy có tải khác nhau thì không so được, và một
bảng trộn như thế mời người đọc so nhầm.

Mọi con số trong `docs/evaluation/xep_hang_model.md` phải truy được về file JSON mà script
này sinh ra, và chỉ về nó.

    python scripts/benchmark_models.py <thu-muc-ket-qua> <anh1> <anh2> ...

Bảng trong `docs/evaluation/xep_hang_model.md` dựng từ đầu ra của script này. Muốn kiểm
lại thì chạy nó trên board của bạn — mọi con số ở đó phải khớp, hoặc bảng sai.

Ghi JSON sau mỗi phần, nên nếu máy hết bộ nhớ giữa chừng thì phần đã đo vẫn còn.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"d:\repos\Internship\Automated_Optical_Inspection_PCB")
sys.path.insert(0, str(ROOT))

from aoi_pipeline.config import ModelDetectorConfig, PipelineConfig
from aoi_pipeline.detection.detectors import UltralyticsDetector
from aoi_pipeline.modelops.model_registry import discover_models
from aoi_pipeline.models import intersection_over_union
from aoi_pipeline.solder.geometry import derive_solder_joints

if len(sys.argv) < 3:
    raise SystemExit(
        "Cách dùng: python scripts/benchmark_models.py <thư-mục-kết-quả> <ảnh1> <ảnh2> ... "
        "— cần ít nhất một ảnh board thật; script đo mọi model trên CÙNG các ảnh đó."
    )

OUT = Path(sys.argv[1])
IMAGES = [Path(item) for item in sys.argv[2:]]
missing = [item for item in IMAGES if not item.is_file()]
if missing:
    raise SystemExit("Không thấy ảnh: " + ", ".join(str(item) for item in missing))
OUT.mkdir(parents=True, exist_ok=True)
RESULT: dict = {"protocol": {}, "detector": {}, "classifier": {}, "solder": {}}

TILE = 1024
CONF = 0.25


def save() -> None:
    (OUT / "bench.json").write_text(
        json.dumps(RESULT, indent=2, ensure_ascii=False), encoding="utf-8")


def tile_of(path: Path) -> np.ndarray:
    """Cùng một ô 1024² cắt giữa mỗi ảnh, cho mọi model."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"không đọc được {path}")
    h, w = image.shape[:2]
    half = TILE // 2
    cy, cx = h // 2, w // 2
    tile = image[max(0, cy - half):cy + half, max(0, cx - half):cx + half].copy()
    del image
    return tile


TILES = {path.name: tile_of(path) for path in IMAGES}
RESULT["protocol"] = {
    "tile_px": TILE,
    "confidence": CONF,
    "images": {name: list(tile.shape[:2]) for name, tile in TILES.items()},
    "note": "Mọi model chạy trên CÙNG các ô này, trong CÙNG một tiến trình.",
}
save()
print(f"{len(TILES)} ảnh, ô {TILE}²\n")

# ==========================================================================
# 1. DETECTOR
# ==========================================================================
print("=" * 66)
print("1. DETECTOR")
print("=" * 66)

detector_models = {
    entry.name.rsplit("/", 1)[0]: entry
    for entry in discover_models("detector", require_manifest=False)
}
per_model: dict[str, dict] = {}
boxes_cache: dict[str, dict[str, list]] = {}

for label, entry in detector_models.items():
    summary = entry.summary()
    detector = UltralyticsDetector(str(entry.model_path), ModelDetectorConfig(confidence=CONF))
    detector.detect(next(iter(TILES.values()))[:64, :64])          # nạp model trước khi bấm giờ

    counts, times, confs, classes = {}, [], [], Counter()
    boxes_cache[label] = {}
    for name, tile in TILES.items():
        start = time.perf_counter()
        found = detector.detect(tile)
        times.append(time.perf_counter() - start)
        counts[name] = len(found)
        confs.extend(float(d.confidence) for d in found if d.confidence is not None)
        classes.update(d.label for d in found)
        boxes_cache[label][name] = [
            (d.label, tuple(float(v) for v in d.bbox.as_xyxy()), float(d.confidence or 0.0))
            for d in found
        ]
    per_model[label] = {
        "origin": entry.origin,
        "architecture": summary.architecture,
        "created": summary.created,
        "sha256": summary.sha256,
        "map50_manifest": summary.metric_value if summary.metric_name == "mAP50" else None,
        "size_mb": round(entry.size_mb, 1),
        "counts": counts,
        "total": sum(counts.values()),
        "seconds_median": round(float(np.median(times)), 2),
        "seconds_total": round(float(np.sum(times)), 2),
        "confidence_median": round(float(np.median(confs)), 3) if confs else None,
        "classes": dict(classes.most_common()),
        "pads_pins": int(classes.get("pads", 0) + classes.get("pins", 0)),
    }
    print(f"  {label:<40} {per_model[label]['total']:>4} box "
          f"· {per_model[label]['seconds_median']:.2f}s/ảnh "
          f"· conf {per_model[label]['confidence_median']}")
    del detector
    gc.collect()

# Hai bản nào đồng thuận tới đâu.
names = list(per_model)
agreement = {}
for i, a in enumerate(names):
    for b in names[i + 1:]:
        shared = same_label = only_a = only_b = 0
        for image in TILES:
            boxes_a, boxes_b = boxes_cache[a][image], boxes_cache[b][image]
            used: set[int] = set()
            matched_here = 0
            for label_a, box_a, _ in boxes_a:
                best, best_iou = -1, 0.0
                for index, (label_b, box_b, _) in enumerate(boxes_b):
                    if index in used:
                        continue
                    from aoi_pipeline.models import BoundingBox
                    iou = intersection_over_union(BoundingBox(*box_a), BoundingBox(*box_b))
                    if iou > best_iou:
                        best, best_iou = index, iou
                if best >= 0 and best_iou >= 0.5:
                    used.add(best)
                    shared += 1
                    matched_here += 1
                    same_label += label_a == boxes_b[best][0]
            # Đếm theo TỪNG ảnh: trừ biến cộng dồn sẽ ra số âm.
            only_a += len(boxes_a) - matched_here
            only_b += len(boxes_b) - len(used)
        agreement[f"{a} ↔ {b}"] = {
            "shared": shared, "same_label": same_label,
            "only_first": only_a, "only_second": only_b,
        }

RESULT["detector"] = {"models": per_model, "agreement": agreement}
save()
print()

# ==========================================================================
# 2. CLASSIFIER — trên CÙNG các crop, cắt bằng detector tốt nhất
# ==========================================================================
print("=" * 66)
print("2. CLASSIFIER")
print("=" * 66)

best_detector = max(per_model, key=lambda key: per_model[key]["total"])
print(f"  crop cắt bằng detector nhiều box nhất: {best_detector}")
detector = UltralyticsDetector(
    str(detector_models[best_detector].model_path), ModelDetectorConfig(confidence=CONF))

crops: list[tuple[np.ndarray, str]] = []
for name, tile in TILES.items():
    for found in detector.detect(tile):
        x1, y1, x2, y2 = found.bbox.clamp(tile.shape[1], tile.shape[0]).to_int()
        patch = tile[y1:y2, x1:x2]
        if patch.size and min(patch.shape[:2]) >= 8:
            crops.append((patch.copy(), found.label))
del detector
gc.collect()
print(f"  {len(crops)} crop thật")

import onnxruntime as ort

classifier_models = {
    entry.name.rsplit("/", 1)[0]: entry
    for entry in discover_models("classifier")
}
classifier_out: dict[str, dict] = {}

for label, entry in classifier_models.items():
    manifest = entry.manifest() or {}
    summary = entry.summary()
    size = manifest.get("input", {}).get("size", [224, 224])
    mean = np.array(manifest["input"]["normalization"]["mean"], np.float32)
    std = np.array(manifest["input"]["normalization"]["std"], np.float32)
    pad = manifest["input"].get("letterbox_value", 114)
    classes = manifest.get("class_names", [])
    session = ort.InferenceSession(str(entry.model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def prepare(patch: np.ndarray) -> np.ndarray:
        scale = min(size[0] / patch.shape[1], size[1] / patch.shape[0])
        nw, nh = max(1, int(patch.shape[1] * scale)), max(1, int(patch.shape[0] * scale))
        resized = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size[1], size[0], 3), pad, np.uint8)
        top, left = (size[1] - nh) // 2, (size[0] - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.transpose((rgb - mean) / std, (2, 0, 1))[None]

    session.run(None, {input_name: prepare(crops[0][0])})          # làm nóng
    probs_top, labels, times = [], [], []
    for patch, _ in crops:
        start = time.perf_counter()
        logits = session.run(None, {input_name: prepare(patch)})[0][0]
        times.append(time.perf_counter() - start)
        exp = np.exp(logits - logits.max())
        prob = exp / exp.sum()
        probs_top.append(float(prob.max()))
        labels.append(classes[int(prob.argmax())] if classes else "?")

    probs_top = np.array(probs_top)
    agree = sum(1 for (_, hint), got in zip(crops, labels) if hint == got)
    classifier_out[label] = {
        "origin": entry.origin,
        "architecture": summary.architecture,
        "created": summary.created,
        "sha256": summary.sha256,
        "size_mb": round(entry.size_mb, 1),
        "input_px": size[0],
        "class_count": len(classes),
        "ms_per_crop": round(float(np.mean(times)) * 1000, 1),
        "confidence_median": round(float(np.median(probs_top)), 3),
        "confidence_p10": round(float(np.percentile(probs_top, 10)), 3),
        "auto_accept_085": f"{int((probs_top >= 0.85).sum())}/{len(probs_top)}",
        "auto_accept_rate": round(float((probs_top >= 0.85).mean()), 3),
        "agrees_with_detector": f"{agree}/{len(labels)}",
        "agree_rate": round(agree / len(labels), 3),
        "metric_manifest": (f"{summary.metric_name} {summary.metric_value}"
                            if summary.metric_name else None),
    }
    print(f"  {label:<40} {classifier_out[label]['ms_per_crop']:>6} ms/crop "
          f"· conf {classifier_out[label]['confidence_median']} "
          f"· nhận {classifier_out[label]['auto_accept_085']}")
    del session
    gc.collect()

RESULT["classifier"] = {"crops": len(crops), "cut_with": best_detector,
                        "models": classifier_out}
save()
print()

# ==========================================================================
# 3. SOLDER 6.2
# ==========================================================================
print("=" * 66)
print("3. SOLDER 6.2")
print("=" * 66)

solder_entry = next(iter(discover_models("solder")), None)
if solder_entry is not None:
    manifest = solder_entry.manifest() or {}
    classes = manifest["class_names"]
    size = manifest["input"]["size"]
    mean = np.array(manifest["input"]["normalization"]["mean"], np.float32)
    std = np.array(manifest["input"]["normalization"]["std"], np.float32)
    pad = manifest["input"].get("letterbox_value", 114)
    session = ort.InferenceSession(str(solder_entry.model_path),
                                   providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def prep_solder(patch: np.ndarray) -> np.ndarray:
        scale = min(size[0] / patch.shape[1], size[1] / patch.shape[0])
        nw, nh = max(1, int(patch.shape[1] * scale)), max(1, int(patch.shape[0] * scale))
        resized = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size[1], size[0], 3), pad, np.uint8)
        top, left = (size[1] - nh) // 2, (size[0] - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.transpose((rgb - mean) / std, (2, 0, 1))[None]

    detector = UltralyticsDetector(
        str(detector_models[best_detector].model_path), ModelDetectorConfig(confidence=CONF))
    config = PipelineConfig()
    roi_patches, roi_sizes = [], []
    for name, tile in TILES.items():
        for found in detector.detect(tile):
            for joint in derive_solder_joints(found, tile.shape[1], tile.shape[0],
                                              config.solder, image=tile):
                x1, y1, x2, y2 = joint.bbox.clamp(tile.shape[1], tile.shape[0]).to_int()
                patch = tile[y1:y2, x1:x2]
                if patch.size:
                    roi_patches.append(patch.copy())
                    roi_sizes.append((x2 - x1, y2 - y1))
    del detector
    gc.collect()

    def classify_all(patches):
        out_labels, out_probs = [], []
        for patch in patches:
            logits = session.run(None, {input_name: prep_solder(patch)})[0][0]
            exp = np.exp(logits - logits.max())
            prob = exp / exp.sum()
            out_labels.append(classes[int(prob.argmax())])
            out_probs.append(float(prob.max()))
        return out_labels, np.array(out_probs)

    roi_labels, roi_probs = classify_all(roi_patches)

    # Đối chứng: mảnh board ngẫu nhiên, cùng phân bố kích thước, không phải ROI.
    rng = np.random.default_rng(11)
    control = []
    tiles = list(TILES.values())
    for index in range(len(roi_patches)):
        tile = tiles[index % len(tiles)]
        pw, ph = roi_sizes[index % len(roi_sizes)]
        x = int(rng.integers(0, max(1, tile.shape[1] - pw)))
        y = int(rng.integers(0, max(1, tile.shape[0] - ph)))
        control.append(tile[y:y + ph, x:x + pw])
    ctrl_labels, ctrl_probs = classify_all(control)

    roi_dist = Counter(roi_labels)
    ctrl_dist = Counter(ctrl_labels)
    overlap = sum(
        min(roi_dist.get(name, 0) / len(roi_labels), ctrl_dist.get(name, 0) / len(ctrl_labels))
        for name in classes
    )
    sweep = manifest.get("training", {}).get("threshold_sweep", [])
    accept = manifest.get("decision_thresholds", {}).get("accept")
    here = next((row for row in sweep if abs(row["accept"] - accept) < 1e-9), None)

    RESULT["solder"] = {
        "name": solder_entry.name, "architecture": solder_entry.summary().architecture,
        "sha256": solder_entry.summary().sha256,
        "rois": len(roi_labels),
        "roi_labels": dict(roi_dist.most_common()),
        "control_labels": dict(ctrl_dist.most_common()),
        "distribution_overlap": round(overlap, 3),
        "confidence_median": round(float(np.median(roi_probs)), 3),
        "above_accept": f"{int((roi_probs >= (accept or 0.85)).sum())}/{len(roi_probs)}",
        "accept_threshold": accept,
        "val_review_rate": here.get("review_rate") if here else None,
        "val_escape": here.get("escape") if here else None,
        "val_false_call": here.get("false_call") if here else None,
        "class_imbalance": None,
        "single_source_classes": manifest.get("training", {}).get("single_source_classes"),
        "sources": manifest.get("training", {}).get("sources"),
    }
    counts = manifest.get("training", {}).get("class_counts") or {}
    if counts:
        RESULT["solder"]["class_imbalance"] = round(max(counts.values()) / min(counts.values()), 1)
    print(f"  {len(roi_labels)} ROI · trội nhất {roi_dist.most_common(1)}")
    print(f"  đối chứng mảnh ngẫu nhiên: {ctrl_dist.most_common(1)}")
    print(f"  chồng lấn hai phân bố: {overlap:.3f}")
    del session
    gc.collect()

save()
print(f"\nĐã ghi {OUT / 'bench.json'}")
