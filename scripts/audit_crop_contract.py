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
    patches_a, patches_b, hands, ratios, unmatched = [], [], [], [], 0
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
            cut_a, cut_b = _cut(image, best, args.pad), _cut(image, hand, args.pad)
            if cut_a is None or cut_b is None:
                continue
            patches_a.append(cut_a)
            patches_b.append(cut_b)
            hands.append((str(crop_root / "crops" / name), hand))
            ratios.append(
                ((best[2] - best[0]) / max(w, 1e-6) + (best[3] - best[1]) / max(h, 1e-6)) / 2
            )

    if not patches_a:
        raise SystemExit("không ghép được cặp box nào")

    result_a = _classify(classifier, patches_a)
    result_b = _classify(classifier, patches_b)
    total = len(result_a)
    changed = [i for i in range(total) if result_a[i][0] != result_b[i][0]]
    median_ratio = st.median(ratios)

    print(f"ghép được {total} cặp; {unmatched} box tay không có detection nào khớp")
    print(f"detector khoanh rộng hơn tay {100 * (median_ratio - 1):+.0f}% mỗi cạnh "
          f"(trung vị)\n")

    print(f"ĐỔI NHÃN khi cắt theo box tay thay vì box detector (pad {args.pad:.2f})")
    print(f"  {len(changed)}/{total} = {100 * len(changed) / total:.1f}%")
    still_accepted = sum(1 for i in changed if result_b[i][1] >= accept)
    print(f"  trong đó {still_accepted} ca VẪN vượt ngưỡng accept {accept:.3f} "
          "⇒ đổi nhãn mà không rơi vào hàng chờ xem tay\n")

    print("  cặp đổi nhiều nhất:")
    flips = collections.Counter((result_a[i][0], result_b[i][0]) for i in changed)
    for (src, dst), count in flips.most_common(5):
        print(f"    {count:4d}  {src} -> {dst}")

    print("\nNGUYÊN NHÂN: do box HẸP HƠN, hay do detector khoanh LỆCH CHỖ?")
    print("  (nếu do hẹp hơn, tỉ lệ đổi nhãn phải tăng theo mức chênh độ ôm)")
    for low, high, label in ((0.0, 1.10, "gần bằng nhau (<10%)"),
                             (1.10, 1.35, "rộng hơn 10-35%"),
                             (1.35, 99.0, "rộng hơn >35%")):
        index = [i for i, r in enumerate(ratios) if low <= r < high]
        if not index:
            continue
        flipped = sum(1 for i in index if result_a[i][0] != result_b[i][0])
        print(f"    {label:22s} n={len(index):4d}  đổi nhãn {100 * flipped / len(index):5.1f}%")

    # Nâng pad để bù lại vùng nhìn: có cứu được không? Đo, đừng đoán.
    wide = args.wide_pad
    if wide is None:
        wide = round((median_ratio * (1 + 2 * args.pad) - 1) / 2, 2)
    patches_wide = []
    cache: dict[str, Any] = {}
    for path_text, hand in hands:
        image = cache.get(path_text)
        if image is None:
            image = cv2.imread(path_text)
            cache[path_text] = image
        patch = _cut(image, hand, wide)
        patches_wide.append(patch if patch is not None else patches_b[len(patches_wide)])
    result_c = _classify(classifier, patches_wide)
    changed_c = sum(1 for i in range(total) if result_a[i][0] != result_c[i][0])
    print(f"\nNÂNG PAD CÓ CỨU ĐƯỢC KHÔNG? (pad {args.pad:.2f} -> {wide:.2f})")
    print(f"  công thức giữ nguyên vùng nhìn: "
          f"pad_mới = (tỉ_lệ_rộng x (1 + 2 x pad_cũ) - 1) / 2 = {wide:.2f}")
    print(f"  đổi nhãn: {len(changed)} -> {changed_c} "
          f"({100 * len(changed) / total:.1f}% -> {100 * changed_c / total:.1f}%)")
    print("  Pad phục hồi DIỆN TÍCH nhìn thấy, không phục hồi TỈ LỆ thân/nền trong")
    print("  khung — mà đó mới là thứ classifier đã học. Đừng coi đây là cách sửa.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
