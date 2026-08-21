"""Export geometry-derived lead/pad boxes as a correctable YOLO dataset.

The detector cannot learn ``pads``/``pins`` from the public dataset: only 30 of
its 670 training images contain either class, 186 and 261 instances against
7775 capacitors. Oversampling those 30 images 6x -- which the v2 notebook does
-- raises ``pads`` precision to 0.712 but leaves recall at 0.072, the signature
of a model memorising a handful of images rather than learning a class. More
epochs and a different backbone cannot fix a 30-image class.

New *images* can. Step 5.5 already places a ROI on every terminal of every
detected component, so the boxes a human would have to draw mostly exist
already; they just need correcting. This script writes them out in YOLO format
next to the board images, so they load straight into LabelImg / CVAT / Roboflow
as pre-drawn boxes. Correcting a box is several times faster than drawing one,
and every board photographed on your own line adds unique images rather than
another copy of the same 30.

    python scripts/bootstrap_lead_labels.py boards/ --output datasets/leads_v1 ^
        --model models/detector/kaggle/best.onnx --overlays

**These are pseudo-labels, not ground truth.** Training on them uncorrected
teaches the model to reproduce the geometry it was derived from -- it would
score well against its own assumptions and gain nothing real. The output
directory is deliberately named ``needs_review`` until you rename it, and
``README_FIRST.md`` inside repeats this.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from aoi_pipeline import AOIPipeline, AOIPipelineError, PipelineConfig, load_image  # noqa: E402
from aoi_pipeline.image_io import encode_image  # noqa: E402

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}

#: Class order written into ``data.yaml``. Kept identical to the names the
#: detector already uses so a corrected export can be merged straight into the
#: existing training set without remapping indices.
LEAD_CLASS_NAMES = ("pads", "pins")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export correctable YOLO lead/pad pseudo-labels from board images."
    )
    parser.add_argument("images", nargs="+", help="Board images, folders or globs.")
    parser.add_argument("--output", required=True, help="Dataset directory to create.")
    parser.add_argument("--model", default=None, help="Detector weights (.onnx preferred).")
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Detector input size.")
    parser.add_argument("--cad", default=None, help="Optional board CAD file.")
    parser.add_argument(
        "--overlays",
        action="store_true",
        help="Also write overlays/ so the boxes can be eyeballed before labelling.",
    )
    parser.add_argument(
        "--split-pins",
        action="store_true",
        help="Cut multi-pin bands into one box per pin. Off by default: a band "
        "is one box to correct, per-pin is many, and a bad split costs more "
        "correction time than it saves.",
    )
    parser.add_argument(
        "--min-box",
        type=int,
        default=4,
        help="Drop boxes narrower/shorter than this many pixels (default: 4).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Stop after N images.")
    return parser


def resolve_images(patterns: list[str]) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            found.extend(
                item for item in sorted(path.rglob("*"))
                if item.suffix.lower() in IMAGE_EXTENSIONS
            )
        elif path.is_file():
            found.append(path)
        else:
            found.extend(
                Path(item) for item in sorted(glob.glob(pattern))
                if Path(item).suffix.lower() in IMAGE_EXTENSIONS
            )
    seen: set[Path] = set()
    unique: list[Path] = []
    for item in found:
        resolved = item.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def build_config(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig()
    config.model_detector.confidence = args.conf
    config.model_detector.image_size = args.imgsz
    config.model_detector.end2end = False
    config.solder.enabled = True
    config.solder.split_pins = args.split_pins
    # The body view covers the whole component; it is not a lead box and must
    # never reach the label file.
    config.solder.include_body_view = False
    # Boxes are written for a human to correct, so they should already sit on
    # the metal rather than on the geometric guess around it.
    config.solder.refine_to_metal = True
    config.cad.path = args.cad
    # Nothing here needs the 6.2 grading pass; skip the work.
    config.solder_grading.enabled = False
    return config


def lead_class_for(joint) -> str | None:
    """Which YOLO class a derived ROI should be proposed as.

    A two-terminal part's ends are lands (``pads``); a multi-pin part's leads
    are ``pins``. Anything else -- a body view, an unknown topology -- is not
    proposed at all rather than guessed into one of the two.
    """

    if joint.kind != "joint":
        return None
    geometry = str(joint.terminal_geometry or "").lower()
    if geometry == "two_terminal":
        return "pads"
    if geometry == "pad_only":
        return "pads"
    if geometry == "multi_pin":
        return "pins"
    return None


def to_yolo_line(class_index: int, bbox, width: int, height: int) -> str | None:
    cx = ((bbox.x1 + bbox.x2) / 2.0) / width
    cy = ((bbox.y1 + bbox.y2) / 2.0) / height
    bw = (bbox.x2 - bbox.x1) / width
    bh = (bbox.y2 - bbox.y1) / height
    if not (0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
        return None
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    return f"{class_index} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def draw_overlay(image: np.ndarray, boxes: list[tuple[str, object, str]]) -> np.ndarray:
    canvas = image.copy()
    colours = {"pads": (0, 215, 255), "pins": (255, 190, 0)}
    for class_name, bbox, source in boxes:
        x1, y1, x2, y2 = bbox.to_int()
        colour = colours.get(class_name, (200, 200, 200))
        # A dashed-looking thin box for derived guesses, solid thick for real
        # detections: the eye should go to the guesses, they need the review.
        thickness = 2 if source == "detected" else 1
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)
    return canvas


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    images = resolve_images(args.images)
    if not images:
        print("No board images matched the given paths.", file=sys.stderr)
        return 2
    if args.limit > 0:
        images = images[: args.limit]

    output = Path(args.output).expanduser().resolve()
    images_dir = output / "images"
    labels_dir = output / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = output / "overlays"
    if args.overlays:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    try:
        pipeline = AOIPipeline(build_config(args), model_path=args.model)
    except AOIPipelineError as exc:
        print(f"Could not build the pipeline: {exc}", file=sys.stderr)
        return 1
    if args.model is None:
        print(
            "WARNING: no detector model given; component boxes come from the "
            "OpenCV candidate demo, so the proposed lead boxes are guesses "
            "around guesses. Expect to correct almost all of them.",
            file=sys.stderr,
        )

    # Pass 2 trains on component *crops*, not on whole boards, so the parent
    # boxes have to travel with the labels. Re-detecting them at training time
    # would silently use different boxes than the ones these labels were
    # derived from.
    components_dir = output / "components"
    components_dir.mkdir(parents=True, exist_ok=True)

    class_index = {name: index for index, name in enumerate(LEAD_CLASS_NAMES)}
    counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    per_image: list[dict[str, object]] = []
    failures = 0
    images_with_boxes = 0

    for index, image_path in enumerate(images, start=1):
        try:
            run = pipeline.run(load_image(image_path), source_name=image_path.name)
        except AOIPipelineError as exc:
            failures += 1
            print(f"[{index}/{len(images)}] {image_path.name}: FAILED — {exc}", file=sys.stderr)
            continue

        # The ROI coordinates live in the aligned/preprocessed analysis frame,
        # so that is the image that has to be written beside them. Writing the
        # original file here would silently offset every box.
        analysis = run.alignment_result.image
        height, width = analysis.shape[:2]

        lines: list[str] = []
        overlay_boxes: list[tuple[str, object, str]] = []
        for crop in run.solder_crops:
            joint = crop.joint
            class_name = lead_class_for(joint)
            if class_name is None:
                continue
            bbox = joint.bbox
            if bbox.width < args.min_box or bbox.height < args.min_box:
                continue
            line = to_yolo_line(class_index[class_name], bbox, width, height)
            if line is None:
                continue
            lines.append(line)
            counts[class_name] += 1
            source_counts[joint.source or "derived"] += 1
            overlay_boxes.append((class_name, bbox, joint.source or "derived"))

        stem = image_path.stem
        (components_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "image": f"images/{stem}.png",
                    "frame": {"width": int(width), "height": int(height)},
                    "components": [
                        {
                            "detection_id": detection.detection_id,
                            "label": detection.label,
                            "confidence": round(float(detection.confidence), 4),
                            "bbox": [round(float(v), 2) for v in detection.bbox.to_tuple()]
                            if hasattr(detection.bbox, "to_tuple")
                            else [
                                round(float(detection.bbox.x1), 2),
                                round(float(detection.bbox.y1), 2),
                                round(float(detection.bbox.x2), 2),
                                round(float(detection.bbox.y2), 2),
                            ],
                        }
                        for detection in run.detections
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (images_dir / f"{stem}.png").write_bytes(encode_image(analysis, ".png"))
        (labels_dir / f"{stem}.txt").write_text(
            ("\n".join(lines) + "\n") if lines else "", encoding="utf-8"
        )
        if lines:
            images_with_boxes += 1
        if args.overlays:
            (overlays_dir / f"{stem}.png").write_bytes(
                encode_image(draw_overlay(analysis, overlay_boxes), ".png")
            )
        per_image.append({"image": f"images/{stem}.png", "boxes": len(lines)})
        print(f"[{index}/{len(images)}] {image_path.name}: {len(lines)} box đề xuất")

    if not per_image:
        print("No board produced any proposal; nothing written.", file=sys.stderr)
        return 1

    (output / "data.yaml").write_text(
        "# Pseudo-labels needing human correction -- see README_FIRST.md\n"
        f"path: {output}\n"
        "train: images\n"
        "val: images\n"
        f"nc: {len(LEAD_CLASS_NAMES)}\n"
        "names:\n" + "".join(f"  - {name}\n" for name in LEAD_CLASS_NAMES),
        encoding="utf-8",
    )
    (output / "bootstrap_manifest.json").write_text(
        json.dumps(
            {
                "generated_by": "scripts/bootstrap_lead_labels.py",
                "status": "PSEUDO_LABELS_NEED_REVIEW",
                "detector_model": args.model,
                "detector_confidence": args.conf,
                "split_pins": args.split_pins,
                "images": len(per_image),
                "images_with_boxes": images_with_boxes,
                "failures": failures,
                "component_sidecars": "components/<stem>.json — box linh kiện của lượt 1, "
                "để notebook lượt 2 cắt crop đúng bằng box đã sinh ra nhãn này",
                "class_counts": dict(counts),
                "roi_source_counts": dict(source_counts),
                "per_image": per_image,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output / "README_FIRST.md").write_text(_readme(counts, source_counts, images_with_boxes, len(per_image)), encoding="utf-8")

    print("\n" + "=" * 66)
    print(f"Ảnh xử lý được : {len(per_image)}  (lỗi: {failures})")
    print(f"Ảnh có box     : {images_with_boxes}")
    print(f"Box theo lớp   : {dict(counts)}")
    print(f"Nguồn ROI      : {dict(source_counts)}")
    print(f"\nDataset: {output}")
    print(
        "\nĐây là NHÃN GIẢ do hình học sinh ra, chưa phải ground truth.\n"
        "Train thẳng lên nó chỉ dạy model lặp lại đúng công thức hình học đã\n"
        "sinh ra nó -- điểm số sẽ đẹp mà không học được gì thật. Mở thư mục\n"
        "này trong LabelImg/CVAT/Roboflow, SỬA box, rồi mới train.\n"
        "Đọc README_FIRST.md trong thư mục output."
    )
    print("=" * 66)
    return 0


def _readme(
    counts: Counter[str], source_counts: Counter[str], with_boxes: int, total: int
) -> str:
    return f"""# Nhãn chân/pad bootstrap — CẦN SỬA TRƯỚC KHI TRAIN

Sinh bởi `scripts/bootstrap_lead_labels.py`. **Đây không phải ground truth.**

| | |
|---|---|
| Ảnh | {total} |
| Ảnh có box | {with_boxes} |
| Box theo lớp | {dict(counts)} |
| Nguồn ROI | {dict(source_counts)} |

## Vì sao cần bootstrap

Detector hiện tại không học được `pads`/`pins` vì dataset công khai chỉ có
**30/670 ảnh train** chứa hai lớp này (186 và 261 instance, so với 7775
capacitor). Nhân bản 30 ảnh đó lên 6 lần cho precision 0.712 nhưng recall
**0.072** — dấu hiệu model học thuộc vài tấm ảnh chứ không học được lớp.
Thêm epoch hay đổi kiến trúc không sửa được điều đó; chỉ **ảnh mới** mới sửa
được, và board của chính bạn là nguồn ảnh mới duy nhất khớp camera/ánh sáng
thật.

## Quy trình

1. Xem `overlays/` trước. Box viền **dày** = detection thật của model; box
   viền **mảnh** = suy ra từ hình học (cần soi kỹ hơn).
2. Mở thư mục này bằng LabelImg / CVAT / Roboflow (định dạng YOLO, đọc
   `data.yaml`). Box đã vẽ sẵn — **sửa**, đừng vẽ lại từ đầu.
3. Việc cần làm khi sửa: kéo box về đúng vùng kim loại, xoá box trên chỗ
   không phải chân/pad, **thêm box ở chân mà hình học bỏ sót** (đây là phần
   giá trị nhất — chính là thứ model đang không thấy).
4. Sửa xong: đổi tên thư mục cho khác `needs_review`, upload thành Kaggle
   Dataset, rồi ghép vào tập train của `pcb_detector_v2_kaggle.ipynb`.

## Cảnh báo

Ảnh trong `images/` là **khung ảnh phân tích** (sau tiền xử lý + căn chỉnh),
không phải file gốc — toạ độ box khớp với khung này. Đừng thay bằng ảnh gốc,
mọi box sẽ lệch.

Nếu train thẳng lên nhãn chưa sửa, model chỉ học lại đúng công thức hình học
trong `aoi_pipeline/inspection/solder.py`. Nó sẽ trông như đang hoạt động
(recall cao trên chính tập này) mà không thêm được thông tin nào so với việc
gọi thẳng hàm suy ra ROI — và bạn mất luôn khả năng biết cái nào đúng.
"""


if __name__ == "__main__":
    raise SystemExit(main())
