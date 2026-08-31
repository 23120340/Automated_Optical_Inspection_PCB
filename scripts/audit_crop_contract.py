"""Khoanh box CHẶT hơn có làm đổi câu trả lời của classifier 6.1 không?

Câu hỏi này không trả lời được bằng suy luận, vì nó phụ thuộc vào một hợp đồng
ngầm: classifier 6.1 được train trên crop cắt từ box của **Consolidated** —
đúng bộ đã train detector đang chạy. Nghĩa là detector hiện tại chính là bản sao
sống của quy ước box mà classifier mong đợi.

Nên phép thử là: chạy detector lên chính các tile đã duyệt, ghép box detector
với box tay theo IoU, rồi hỏi classifier hai lần trên cùng một linh kiện:

    A = box detector + pad   <- đúng thứ classifier được fit
    B = box tay      + pad   <- điều sẽ xảy ra khi detector lượt 1 được train lại

A khác B bao nhiêu chính là mức rủi ro phải trả khi đổi nguồn nhãn. Script còn
tách kết quả theo **mức chênh lệch độ ôm**, để phân biệt hai nguyên nhân dễ lẫn:
box hẹp hơn, hay detector khoanh lệch chỗ.

    python scripts/audit_crop_contract.py \\
        datasets/labelling/component_bodies \\
        --boxes "~/Downloads/joint_boxes (3).json"
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from aoi_pipeline.classification.family import ONNXComponentClassifier  # noqa: E402
from aoi_pipeline.config import CropConfig, ModelDetectorConfig  # noqa: E402
from aoi_pipeline.detection.detectors import create_detector  # noqa: E402
from aoi_pipeline.models import BoundingBox, ComponentCrop  # noqa: E402

#: Dưới mức này thì hai box đang nói về hai linh kiện khác nhau, không so được.
MIN_MATCH_IOU = 0.4


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _cut(image, box: tuple[float, ...], pad_ratio: float):
    x1, y1, x2, y2 = box
    pad = pad_ratio * max(x2 - x1, y2 - y1)
    height, width = image.shape[:2]
    a, b = int(round(max(0, x1 - pad))), int(round(max(0, y1 - pad)))
    c, d = int(round(min(width, x2 + pad))), int(round(min(height, y2 + pad)))
    if c - a < 4 or d - b < 4:
        return None
    return image[b:d, a:c].copy()


def _grow(box: tuple[float, ...], fx: float, fy: float) -> tuple[float, ...]:
    """Giãn quanh TÂM, mỗi trục theo hệ số riêng."""

    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * fx, (y2 - y1) * fy
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _classify(clf, patches: list[Any]) -> list[tuple[str, float, str]]:
    out: list[tuple[str, float, str]] = []
    for start in range(0, len(patches), 32):
        chunk = patches[start:start + 32]
        crops = [
            ComponentCrop(
                image=p, detection_id=f"c{i}", label="component", confidence=1.0,
                source_bbox=BoundingBox(0, 0, p.shape[1], p.shape[0]),
                crop_bbox=BoundingBox(0, 0, p.shape[1], p.shape[0]),
                filename=f"c{i}.png", path=None,
            )
            for i, p in enumerate(chunk)
        ]
        out.extend(
            (r.family, float(r.probability), r.decision) for r in clf.classify(crops)
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("crop_dir", type=Path)
    parser.add_argument("--boxes", type=Path, required=True)
    parser.add_argument("--detector", type=Path,
                        default=PROJECT_ROOT / "models/active/detector/best.onnx")
    parser.add_argument("--classifier", type=Path,
                        default=PROJECT_ROOT / "models/active/classifier/best.onnx")
    parser.add_argument("--pad", type=float, default=CropConfig().padding_ratio)
    parser.add_argument("--wide-pad", type=float, default=None,
                        help="pad thứ hai để thử bù lại vùng nhìn (mặc định: tự tính)")
    args = parser.parse_args(argv)

    crop_root = args.crop_dir.expanduser().resolve()
    detector = create_detector(
        args.detector, model_config=ModelDetectorConfig(confidence=0.25)
    )
    classifier = ONNXComponentClassifier(
        args.classifier, args.classifier.with_name("model_manifest.json")
    )
    accept = classifier.accept_threshold

    payload = json.loads(args.boxes.expanduser().read_text(encoding="utf-8"))
    pairs, unmatched = [], 0
    for name, record in sorted(payload.get("crops", {}).items()):
        if record.get("status") != "verified":
            continue
        image = cv2.imread(str(crop_root / "crops" / name))
        if image is None:
            continue
        found = [(d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2)
                 for d in detector.detect(image)]
        for box in record.get("boxes", []):
            x, y = float(box["x"]), float(box["y"])
            w, h = abs(float(box["w"])), abs(float(box["h"]))
            hand = (x, y, x + w, y + h)
            best, best_iou = None, 0.0
            for candidate in found:
                value = _iou(hand, candidate)
                if value > best_iou:
                    best, best_iou = candidate, value
            if best is None or best_iou < MIN_MATCH_IOU:
                unmatched += 1
                continue
            pairs.append((str(crop_root / "crops" / name), hand, best, w, h))

    if not pairs:
        raise SystemExit("không ghép được cặp box nào")

    # Một phương án cắt = một lượt cắt + chấm + giải phóng. Giữ cả năm lượt patch
    # cùng lúc làm tràn RAM trên máy 8 GB, mà năm lượt tuần tự thì không.
    def band_of(short: float) -> int:
        return 0 if short < 20 else (1 if short < 60 else 2)

    fx = [(d[2] - d[0]) / max(w, 1e-6) for _p, _h, d, w, _hh in pairs]
    fy = [(d[3] - d[1]) / max(h, 1e-6) for _p, _h, d, _w, h in pairs]
    median_ratio = st.median([(a + b) / 2 for a, b in zip(fx, fy)])
    bands: dict[int, tuple[float, float]] = {}
    for index in (0, 1, 2):
        subset = [i for i, item in enumerate(pairs) if band_of(min(item[3], item[4])) == index]
        if subset:
            bands[index] = (st.median([fx[i] for i in subset]),
                            st.median([fy[i] for i in subset]))

    wide = args.wide_pad
    if wide is None:
        wide = round((median_ratio * (1 + 2 * args.pad) - 1) / 2, 2)

    def box_for(variant: str, hand, dbox, w, h):
        if variant == "A":
            return dbox, args.pad
        if variant == "B":
            return hand, args.pad
        if variant == "C":
            return hand, wide
        if variant == "D":
            return _grow(hand, st.median(fx), st.median(fy)), args.pad
        if variant == "F":
            return _grow(hand, *bands[band_of(min(w, h))]), args.pad
        ratio_x = (dbox[2] - dbox[0]) / max(w, 1e-6)
        ratio_y = (dbox[3] - dbox[1]) / max(h, 1e-6)
        return _grow(hand, ratio_x, ratio_y), args.pad

    VARIANTS = ("A", "B", "C", "D", "F", "E")
    results: dict[str, list[tuple[str, float, str]]] = {}
    for variant in VARIANTS:
        out: list[tuple[str, float, str]] = []
        for path_text, hand, dbox, w, h in pairs:
            image = cv2.imread(path_text)
            box, pad_ratio = box_for(variant, hand, dbox, w, h)
            patch = _cut(image, box, pad_ratio)
            if patch is None:
                patch = _cut(image, hand, args.pad)
            out.extend(_classify(classifier, [patch]))
            del image, patch
        results[variant] = out
    total = len(results["A"])
    result_a, result_b = results["A"], results["B"]
    changed = [i for i in range(total) if result_a[i][0] != result_b[i][0]]

    print(f"ghép được {total} cặp; {unmatched} box tay không có detection nào khớp")
    print(f"detector khoanh rộng hơn tay {100 * (median_ratio - 1):+.0f}% mỗi cạnh "
          f"(trung vị)")
    print("hệ số nới theo dải cỡ: " + "  ".join(
        f"{['nhỏ', 'vừa', 'lớn'][k]} x{v[0]:.2f}/{v[1]:.2f}" for k, v in bands.items()))
    print()

    print("ĐỔI NHÃN so với A = box detector + pad (thứ classifier được fit)\n")
    print(f"  {'cách cắt':52s} {'đổi nhãn':>9s} {'%':>7s}")
    rows = (
        ("B  box tay + pad (không bù gì)", "B"),
        (f"C  box tay + pad {wide:.2f} (nới LỀ)", "C"),
        ("D  box tay x hệ số CHUNG + pad (nới BOX)", "D"),
        ("F  box tay x hệ số THEO CỠ + pad (nới BOX có điều kiện)", "F"),
        ("E  box tay x hệ số ORACLE + pad (TRẦN, không cài được)", "E"),
    )
    for label, key in rows:
        flipped = sum(1 for i in range(total) if result_a[i][0] != results[key][i][0])
        print(f"  {label:52s} {flipped:9d} {100 * flipped / total:6.1f}%")

    still_accepted = sum(1 for i in changed if result_b[i][1] >= accept)
    print(f"\n  B: {still_accepted}/{len(changed)} ca đổi nhãn VẪN vượt ngưỡng accept "
          f"{accept:.3f}\n  ⇒ đổi nhãn mà không rơi vào hàng chờ xem tay")

    print("\n  cặp đổi nhiều nhất ở B:")
    flips = collections.Counter((result_a[i][0], result_b[i][0]) for i in changed)
    for (src, dst), count in flips.most_common(5):
        print(f"    {count:4d}  {src} -> {dst}")

    print("\nNGUYÊN NHÂN: do box HẸP HƠN, hay do detector khoanh LỆCH CHỖ?")
    print("  (nếu do hẹp hơn, tỉ lệ đổi nhãn phải tăng theo mức chênh độ ôm)")
    ratios = [(a + b) / 2 for a, b in zip(fx, fy)]
    for low, high, label in ((0.0, 1.10, "gần bằng nhau (<10%)"),
                             (1.10, 1.35, "rộng hơn 10-35%"),
                             (1.35, 99.0, "rộng hơn >35%")):
        index = [i for i, r in enumerate(ratios) if low <= r < high]
        if not index:
            continue
        flipped = sum(1 for i in index if result_a[i][0] != result_b[i][0])
        print(f"    {label:22s} n={len(index):4d}  đổi nhãn {100 * flipped / len(index):5.1f}%")

    print("\nĐỌC KẾT QUẢ: E là trần lý thuyết — nó dùng đúng tỉ lệ thật của TỪNG")
    print("linh kiện, con số mà lúc chạy không ai biết. Khoảng cách F -> E chính là")
    print("phần nhiễu per-box của detector, không quy tắc nào lấy lại được.")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
