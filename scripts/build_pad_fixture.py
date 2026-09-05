"""Dựng một fixture pad đếm tay mới, theo quy ước box MỚI ("chỉ thân").

Vì sao cần: `scripts/evaluate_package_rule_gate.py` chạy được nhưng phát hiện
nhánh ``ic`` của bộ luật **chưa từng được kiểm** — fixture duy nhất đang có
(`tests/data/solder_geometry/board_smd_00001`) dùng box của detector 22 lớp cũ,
vốn khoanh *bao cả chân*. Chân rơi vào TRONG box thân, ``_edge_of`` trả ``None``,
và luật không bao giờ chạy tới nhánh đó.

Hai bước, chạy cách nhau (ở giữa là bạn ngồi khoanh pad):

    # 1. Dựng thư mục khoanh + trang khoanh
    python scripts/build_pad_fixture.py prepare <ảnh tile> --out <thư mục>
    python scripts/build_joint_box_app.py <thư mục>/crops --classes pad

    # 2. Sau khi khoanh xong và tải JSON về
    python scripts/build_pad_fixture.py finish <thư mục> <joint_boxes.json> \\
        --image-name <tên ảnh> --out tests/data/solder_geometry/<tên>.json

Pad được gán về linh kiện bằng chính ``assign_leads_to_components()`` mà bước
5.5 dùng — không viết lại phép gán, để fixture không lệch khỏi hành vi thật.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aoi_pipeline.config import (  # noqa: E402
    LeadFusionConfig, PipelineConfig, PreprocessConfig,
)
from aoi_pipeline.imaging.preprocessing import ImagePreprocessor  # noqa: E402
from aoi_pipeline.models import BoundingBox, Detection  # noqa: E402
from aoi_pipeline.pipeline import AOIPipeline  # noqa: E402
from aoi_pipeline.solder.leads import assign_leads_to_components  # noqa: E402

DETECTOR = PROJECT_ROOT / "models/active/detector/best.onnx"


def prepare(image_path: Path, out: Path) -> int:
    """Chạy detector hiện tại, ghi box thân, và dựng thư mục để khoanh pad."""

    if out.exists():
        raise SystemExit(f"đã có {out}; không ghi đè")
    raw = cv2.imread(str(image_path))
    if raw is None:
        raise SystemExit(f"không đọc được {image_path}")

    # Khoanh trên khung SAU bước 1, vì đó là khung mà 5.5 sẽ nhìn thấy. Khoanh
    # trên ảnh thô rồi so với ROI dựng trên ảnh đã tiền xử lý là so lệch hệ.
    image = ImagePreprocessor(PreprocessConfig()).process(raw).image
    pipeline = AOIPipeline(PipelineConfig(), model_path=DETECTOR)
    detections = pipeline.detect_components(image)

    (out / "crops").mkdir(parents=True)
    cv2.imwrite(str(out / "crops" / image_path.name), image)
    (out / "bodies.json").write_text(json.dumps({
        "image": image_path.name,
        "source_image": str(image_path),
        "detector": str(DETECTOR.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "note": "box theo quy ước MỚI: chỉ thân, loại chân/pad",
        "detections": [
            {"label": d.label, "confidence": round(float(d.confidence), 4),
             "box": [round(float(v), 2) for v in d.bbox.as_xyxy()]}
            for d in detections
        ],
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{len(detections)} thân linh kiện, ghi -> {out}")
    print()
    print("Bước tiếp theo — dựng trang khoanh pad:")
    print(f"    python scripts/build_joint_box_app.py {out}/crops --classes pad")
    print()
    print("Khoanh MỌI pad nhìn thấy được, kể cả pad của linh kiện detector bỏ")
    print("sót — pad không gán được về thân nào sẽ được báo riêng ở bước finish,")
    print("và đó chính là thông tin về recall của lượt 1.")
    return 0


def finish(prepared: Path, labels_path: Path, image_name: str | None,
           out: Path) -> int:
    bodies = json.loads((prepared / "bodies.json").read_text(encoding="utf-8"))
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    crops = payload.get("crops", payload)

    name = image_name or bodies["image"]
    record = crops.get(name)
    if record is None:
        raise SystemExit(
            f"không thấy '{name}' trong {labels_path.name}; có: {sorted(crops)[:5]}"
        )
    pads = [b for b in record.get("boxes", []) if b.get("w", 0) > 0]
    if not pads:
        raise SystemExit(f"'{name}' chưa có pad nào được khoanh")

    detections = [
        Detection(row["label"], row["confidence"], BoundingBox(*row["box"]))
        for row in bodies["detections"]
    ]
    pad_detections = [
        Detection("pad", 1.0,
                  BoundingBox(b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]),
                  detection_id=f"pad_{index:04d}")
        for index, b in enumerate(pads)
    ]
    assigned = assign_leads_to_components(
        detections, pad_detections, LeadFusionConfig()
    )
    index_of = {d.detection_id: i for i, d in enumerate(detections)}
    by_index: dict[int, list[Detection]] = defaultdict(list)
    for detection_id, items in assigned.items():
        by_index[index_of[detection_id]].extend(items)

    attached = {p.detection_id for items in assigned.values() for p in items}
    orphans = [p for p in pad_detections if p.detection_id not in attached]

    components = {}
    for position, (index, items) in enumerate(sorted(by_index.items())):
        components[f"C{position:03d}"] = {
            "package": "unknown",
            "detection_index": index,
            "pads": [[int(round(v)) for v in p.bbox.as_xyxy()] for p in items],
            "note": "pad khoanh tay; trường package cần người điền",
        }

    fixture = {
        "image": bodies["image"],
        "source": bodies.get("source_image", ""),
        "detector": (f"{bodies['detector']}, quy ước box MỚI (chỉ thân); "
                     "khung SAU bước 1"),
        "note": (f"{len(pads)} pad khoanh tay, gán về {len(components)} linh "
                 "kiện bằng chính assign_leads_to_components() của bước 5.5."),
        "detections": bodies["detections"],
        "components": components,
    }
    if orphans:
        fixture["known_limitations"] = {
            "unassigned_pads": {
                "count": len(orphans),
                "boxes": [[int(round(v)) for v in p.bbox.as_xyxy()]
                          for p in orphans],
                "reason": (
                    "Pad không gán được về thân nào. Hoặc lượt 1 bỏ sót linh "
                    "kiện đó, hoặc pad nằm quá xa mọi thân theo "
                    "LeadFusionConfig.max_lead_distance_ratio. Đây là số đo "
                    "recall của lượt 1, không phải lỗi của fixture."
                ),
            }
        }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    source_image = prepared / "crops" / bodies["image"]
    if source_image.is_file():
        shutil.copy2(source_image, out.parent / bodies["image"])

    print(f"{len(pads)} pad -> {len(components)} linh kiện")
    if orphans:
        print(f"  {len(orphans)} pad KHÔNG gán được về thân nào "
              "(ghi vào known_limitations)")
    print(f"ghi -> {out}")
    print()
    print("Còn phải làm bằng tay: điền 'package' cho từng linh kiện trong")
    print("fixture. Cổng luật dùng nó để tách kết quả theo từng topology.")
    print()
    print("Rồi chạy cổng:")
    print(f"    python scripts/evaluate_package_rule_gate.py {out.parent}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("prepare", help="chạy detector, dựng thư mục khoanh pad")
    p.add_argument("image", type=Path)
    p.add_argument("--out", type=Path, required=True)

    f = sub.add_parser("finish", help="ghép pad đã khoanh thành fixture")
    f.add_argument("prepared", type=Path)
    f.add_argument("labels", type=Path)
    f.add_argument("--image-name", default=None)
    f.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.mode == "prepare":
        return prepare(Path(args.image), Path(args.out))
    return finish(Path(args.prepared), Path(args.labels),
                  args.image_name, Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
