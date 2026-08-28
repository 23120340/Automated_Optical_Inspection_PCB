"""Pre-fill the component-body labelling app with the current detector's boxes.

Drawing every box by hand is the wrong shape of work here. The shipped pass-1
detector is not uniformly bad -- it is bad on exactly one thing. Measured on a
tile of pcb7, its largest box is 231x219 at confidence 0.25 and 251x250 even at
0.10, while the QFPs there run 300-400 px, so it misses large fine-pitch
packages and boxes their lead combs instead. On the small and medium parts that
dominate its training data it is serviceable.

So it drafts, and the reviewer corrects. 120 tiles hold roughly 3,600 component
bodies; drawing those from nothing is days of work, while fixing a draft is a
different job -- delete a few wrong boxes, add the large packages it could not
see, approve the rest.

**Nothing here is a label.** Every drafted crop is written with an empty
``status``, which is what the app and ``pack_joint_detection_dataset.py`` both
read as "nobody has looked at this". A drafted box that is never reviewed is
discarded, not trained on. That is deliberate: a model's guess promoted to
ground truth by default would teach the next model its predecessor's blind
spots, and the blind spot is the whole reason for this exercise.

    python scripts/prelabel_component_bodies.py
    # then open datasets/labelling/component_bodies/label_boxes.html

The draft is written as ``draft_boxes.json`` next to the crops. Load it with the
app's "Nạp file" button.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

#: Hộp có tỉ lệ cạnh vượt ngưỡng này gần như chắc chắn là hộp đặt trên một DÃY
#: CHÂN chứ không phải trên gói. Đo được trên tile pcb7: hai hộp như thế có tỉ
#: lệ 4.5 và 5.6, trong khi mọi gói thật đều dưới 2.5. Bỏ chúng khỏi bản nháp:
#: người duyệt xoá một hộp sai mất công bằng vẽ một hộp đúng, nên đưa vào chỉ
#: tổ thêm việc.
MAX_ASPECT = 3.0


def prelabel(
    folder: Path,
    model: Path,
    *,
    confidence: float,
    max_aspect: float,
    dry_run: bool,
) -> int:
    import cv2

    from aoi_pipeline.config import ModelDetectorConfig
    from aoi_pipeline.detection.detectors import create_detector

    manifest = folder / "manifest.csv"
    crops = folder / "crops"
    for path in (manifest, crops):
        if not path.exists():
            raise SystemExit(f"missing {path}; chạy prepare_component_labelling.py trước")

    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    detector = create_detector(str(model),
                               model_config=ModelDetectorConfig(confidence=confidence))
    payload: dict[str, dict[str, object]] = {}
    drawn = dropped = 0
    for index, row in enumerate(rows, start=1):
        image_path = crops / row["crop_path"]
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        boxes = []
        for detection in detector.detect(image):
            box = detection.bbox
            width, height = box.width, box.height
            if width <= 1 or height <= 1:
                continue
            if max(width, height) / max(1.0, min(width, height)) > max_aspect:
                dropped += 1
                continue
            boxes.append({
                "cls": "component",
                "x": int(round(box.x1)), "y": int(round(box.y1)),
                "w": int(round(width)), "h": int(round(height)),
            })
        drawn += len(boxes)
        payload[row["crop_path"]] = {
            # Rỗng, và phải rỗng: app và packer đều đọc chuỗi này là "chưa ai
            # xem". Đặt 'verified' ở đây là biến phỏng đoán của model thành sự
            # thật, và dạy model sau đúng điểm mù của model trước.
            "status": "",
            "notes": "bản nháp do detector lượt 1 vẽ — cần duyệt",
            "boxes": boxes,
        }
        if index % 25 == 0:
            print(f"  {index}/{len(rows)} tile …")

    if dry_run:
        print(f"[dry-run] {drawn} box nháp trên {len(payload)} tile")
        return 0

    out = folder / "draft_boxes.json"
    out.write_text(json.dumps({
        "schema": "aoi-joint-boxes/1.0",
        "dataset": folder.name,
        "reviewer_id": "",
        "coordinate_space": "crop_pixels_top_left_origin",
        "classes": ["component"],
        "note": (
            "Bản nháp của model, KHÔNG phải nhãn. Mọi status để rỗng nên app hiện "
            "chúng ở bộ lọc 'chưa duyệt' và packer bỏ qua cho tới khi có người "
            "bấm duyệt."
        ),
        "crops": payload,
    }, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"\nđã ghi {out}")
    print(f"  {drawn} box nháp trên {len(payload)} tile (trung bình {drawn/max(1,len(payload)):.1f}/tile)")
    print(f"  bỏ {dropped} hộp tỉ lệ cạnh >{max_aspect:g} (hộp đặt trên dãy chân, không phải gói)")
    print("\nMở label_boxes.html → nút 'Nạp file' → chọn draft_boxes.json")
    print("Việc của người duyệt: xoá hộp sai, THÊM gói lớn model không thấy, rồi Enter để duyệt.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path, nargs="?",
                        default=PROJECT_ROOT / "datasets" / "labelling" / "component_bodies")
    parser.add_argument("--model", type=Path,
                        default=PROJECT_ROOT / "models" / "active" / "detector" / "best.onnx")
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--max-aspect", type=float, default=MAX_ASPECT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return prelabel(args.folder.resolve(), args.model.resolve(),
                    confidence=args.confidence, max_aspect=args.max_aspect,
                    dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
