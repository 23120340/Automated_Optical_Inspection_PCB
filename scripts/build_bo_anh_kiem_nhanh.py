"""Dựng bộ ảnh kiểm nhanh: những tile detector ĐANG chạy tốt, chọn bằng SỐ ĐO.

Không chọn bằng mắt. Mỗi tile được chấm bằng chính nhãn tay của dự án
(``joint_boxes_cleaned.json``, 95 tile đã duyệt): ghép nhãn với box detect theo
IoU, rồi lấy **recall** và **tỉ lệ box thừa**. Chỉ tile qua cả hai ngưỡng mới
vào bộ.

**Bộ này để làm gì, và KHÔNG để làm gì.**

Nó là bộ **kiểm hồi quy**: chạy lại sau mỗi thay đổi, số tụt là biết vừa làm hỏng
cái gì. Nó **không** chứng minh hệ thống chạy được trên bo dây chuyền — mọi tile
ở đây đều **trong miền** của tập huấn luyện, mà lỗi nặng nhất đang gặp lại là lỗi
**ngoài miền** (xem ``Docs/danh_gia/loi_pad_tron_bo_du_an.md``: 32% box trên bo
dự án là pad tròn). Một bộ ảnh chọn theo tiêu chí "đang chạy tốt" thì theo định
nghĩa không chứa ca đang hỏng.

    python scripts/build_bo_anh_kiem_nhanh.py --out datasets/test_images/bo_kiem_nhanh
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aoi_pipeline.config import PipelineConfig, PreprocessConfig  # noqa: E402
from aoi_pipeline.imaging.preprocessing import ImagePreprocessor  # noqa: E402
from aoi_pipeline.pipeline import AOIPipeline  # noqa: E402

LABELS = (PROJECT_ROOT / "datasets/labelling/component_bodies_round2_20260830"
          / "joint_boxes_cleaned.json")
TILES = PROJECT_ROOT / "datasets/test_images/tiles_1024"
DETECTOR = PROJECT_ROOT / "models/active/detector/best.onnx"

#: Ghép một box detect với một box nhãn từ ngưỡng này. 0,4 là mức "cùng nói về
#: một linh kiện" chứ không đòi trùng khít -- quy ước khoanh tay không chặt tới
#: mức đó, và đòi cao hơn là đo độ chặt của người khoanh chứ không đo detector.
IOU_MATCH = 0.40


def _iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def score_tile(pipeline: AOIPipeline, name: str, boxes) -> dict | None:
    image = cv2.imread(str(TILES / name))
    if image is None:
        return None
    truth = [
        (float(b["x"]), float(b["y"]),
         float(b["x"]) + float(b["w"]), float(b["y"]) + float(b["h"]))
        for b in boxes if float(b["w"]) > 0 and float(b["h"]) > 0
    ]
    if not truth:
        return None
    analysis = ImagePreprocessor(PreprocessConfig()).process(image).image
    found = [tuple(d.bbox.as_xyxy()) for d in pipeline.detect_components(analysis)]

    used: set[int] = set()
    hits = 0
    for box in truth:
        best, index = 0.0, None
        for position, candidate in enumerate(found):
            if position in used:
                continue
            value = _iou(box, candidate)
            if value > best:
                best, index = value, position
        if best >= IOU_MATCH:
            hits += 1
            used.add(index)
    extra = len(found) - hits
    return {
        "tile": name,
        "board": name.split("__")[0],
        "labelled": len(truth),
        "detected": len(found),
        "matched": hits,
        "recall": round(hits / len(truth), 4),
        "extra": extra,
        "extra_rate": round(extra / max(1, len(found)), 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path,
                        default=Path("datasets/test_images/bo_kiem_nhanh"))
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--max-extra-rate", type=float, default=0.15)
    parser.add_argument("--min-labelled", type=int, default=15)
    parser.add_argument("--per-board", type=int, default=1,
                        help="tối đa bao nhiêu tile mỗi BO, để bộ không lệch về một bo")
    args = parser.parse_args(argv)

    if not DETECTOR.is_file():
        raise SystemExit(f"không thấy detector {DETECTOR}")
    payload = json.loads(LABELS.read_text(encoding="utf-8"))
    pipeline = AOIPipeline(PipelineConfig(), model_path=str(DETECTOR))

    scored = []
    for name, record in payload["crops"].items():
        if record.get("status") != "verified":
            continue
        boxes = record.get("boxes", [])
        if len(boxes) < args.min_labelled:
            continue
        result = score_tile(pipeline, name, boxes)
        if result is not None:
            scored.append(result)
            print(f"  {result['tile'][:44]:44s} recall={result['recall']:5.0%} "
                  f"thua={result['extra_rate']:4.0%}", flush=True)

    passing = [
        item for item in scored
        if item["recall"] >= args.min_recall and item["extra_rate"] <= args.max_extra_rate
    ]
    # Trải theo BO trước rồi mới xếp hạng: một bộ toàn tile của cùng một bo thì
    # chạy tốt cũng không nói được gì về bo khác.
    passing.sort(key=lambda item: (-item["recall"], item["extra_rate"]))
    per_board: dict[str, int] = {}
    chosen = []
    for item in passing:
        if per_board.get(item["board"], 0) >= args.per_board:
            continue
        per_board[item["board"]] = per_board.get(item["board"], 0) + 1
        chosen.append(item)

    out = (PROJECT_ROOT / args.out).resolve()
    (out / "anh").mkdir(parents=True, exist_ok=True)
    for item in chosen:
        shutil.copy2(TILES / item["tile"], out / "anh" / item["tile"])

    (out / "manifest.json").write_text(json.dumps({
        "note": ("Bộ kiểm HỒI QUY, chọn bằng số đo trên nhãn tay. KHÔNG phải "
                 "bằng chứng hệ thống chạy được trên bo dây chuyền: mọi tile ở "
                 "đây đều TRONG MIỀN huấn luyện."),
        "detector": str(DETECTOR.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "iou_match": IOU_MATCH,
        "min_recall": args.min_recall,
        "max_extra_rate": args.max_extra_rate,
        "tiles": chosen,
        "scored_total": len(scored),
        "passing_total": len(passing),
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [
        "# Bộ ảnh kiểm nhanh — detector thân linh kiện", "",
        f"{len(chosen)} tile trên {len(per_board)} bo, chọn **bằng số đo** chứ không bằng mắt:",
        f"mỗi tile được chấm bằng chính nhãn tay của dự án (IoU ≥ {IOU_MATCH}),",
        f"và chỉ tile đạt **recall ≥ {args.min_recall:.0%}** và **thừa ≤ {args.max_extra_rate:.0%}** mới vào bộ.", "",
        "| tile | box nhãn | detect | recall | thừa |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in chosen:
        lines.append(f"| `{item['tile']}` | {item['labelled']} | {item['detected']} | "
                     f"{item['recall']:.0%} | {item['extra']} ({item['extra_rate']:.0%}) |")
    lines += [
        "", "## Bộ này KHÔNG chứng minh điều gì", "",
        "Mọi tile ở đây đều **trong miền** của tập huấn luyện. Lỗi nặng nhất đang",
        "gặp lại là lỗi **ngoài miền** — trên bo của chính dự án, 32% box lượt 1 là",
        "pad tròn chứ không phải linh kiện (`Docs/danh_gia/loi_pad_tron_bo_du_an.md`).",
        "",
        "Một bộ ảnh chọn theo tiêu chí *đang chạy tốt* thì **theo định nghĩa** không",
        "chứa ca đang hỏng. Dùng nó để trả lời *\"thay đổi vừa rồi có làm hỏng gì",
        "không\"*, đừng dùng để trả lời *\"hệ thống đã dùng được chưa\"*.",
        "", "## Chạy lại", "",
        "```bash", "python scripts/build_bo_anh_kiem_nhanh.py", "```", "",
        "> Ảnh trong bộ này phái sinh từ **CVL PCB-DSLR** (giấy phép nghiên cứu",
        "> phi thương mại), nên thư mục nằm dưới `datasets/` và bị `.gitignore`",
        "> chặn — không đưa lên repo. Script và tài liệu thì có trong git, nên",
        "> dựng lại được bất cứ lúc nào.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nchấm {len(scored)} tile, {len(passing)} đạt ngưỡng, "
          f"chọn {len(chosen)} trên {len(per_board)} bo")
    print(f"ghi -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
