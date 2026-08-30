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

**Nothing inferred here is a label.** Every detector-drafted crop is written
with an empty ``status``, which the app and packer read as "nobody has looked at
this". With ``--checkpoint``, already verified records are copied exactly and
the detector is not run on them; skipped/unusable pixels are rejected.

    python scripts/prelabel_component_bodies.py
    # then open datasets/labelling/component_bodies/label_boxes.html

The draft is written as ``draft_boxes.json`` next to the crops. Load it with the
app's "Nạp file" button.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

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
    checkpoint: Path | None = None,
    previous_folder: Path | None = None,
    base_draft: Path | None = None,
    output: Path | None = None,
) -> int:
    import cv2

    from aoi_pipeline.config import ModelDetectorConfig
    from aoi_pipeline.detection.detectors import create_detector
    from scripts.build_joint_box_app import dataset_id_for, load_rows
    from scripts.pack_component_detection_dataset import (
        CHECKPOINT_STATUSES,
        _expected_checkpoint_dataset_id,
        _pixel_identity,
        read_boxes,
    )

    if checkpoint is not None and previous_folder is None:
        raise SystemExit("--checkpoint cần --previous-folder để xác thực pixel đã duyệt")
    if checkpoint is None and previous_folder is not None:
        raise SystemExit("--previous-folder chỉ có nghĩa khi dùng --checkpoint")
    if base_draft is not None and previous_folder is None:
        raise SystemExit("--base-draft cần --checkpoint và --previous-folder")

    manifest = folder / "manifest.csv"
    crops = folder / "crops"
    for path in (manifest, crops):
        if not path.exists():
            raise SystemExit(f"missing {path}; chạy prepare_component_labelling.py trước")

    with manifest.open(encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if not manifest_rows:
        raise SystemExit(f"{manifest} has no rows")
    names = [str(row.get("crop_path", "")) for row in manifest_rows]
    if "" in names or len(names) != len(set(names)):
        raise SystemExit(f"{manifest}: crop_path is missing or duplicated")
    app_rows = load_rows(manifest, crops)

    out = output.resolve() if output is not None else folder / "draft_boxes.json"
    if not dry_run and out.exists():
        raise SystemExit(f"output already exists; refusing to overwrite: {out}")

    reviewed: dict[str, dict[str, Any]] = {}
    verified_names: set[str] = set()
    verified_hashes: set[str] = set()
    rejected_hashes: set[str] = set()
    checkpoint_sha256 = ""
    if checkpoint is not None and previous_folder is not None:
        old_manifest = previous_folder / "manifest.csv"
        old_crops = previous_folder / "crops"
        for required in (checkpoint, old_manifest, old_crops):
            if not required.exists():
                raise SystemExit(f"missing {required}")
        with old_manifest.open(encoding="utf-8", newline="") as handle:
            old_rows = list(csv.DictReader(handle))
        old_names = {str(row.get("crop_path", "")) for row in old_rows}
        checkpoint_payload = read_boxes(checkpoint)
        if checkpoint_payload.get("dataset") != previous_folder.name:
            raise SystemExit(
                f"checkpoint dataset mismatch: expected {previous_folder.name!r}, "
                f"got {checkpoint_payload.get('dataset')!r}"
            )
        expected_id = _expected_checkpoint_dataset_id(previous_folder, old_rows)
        if checkpoint_payload.get("dataset_id") != expected_id:
            raise SystemExit(
                f"checkpoint dataset_id mismatch: expected {expected_id!r}, "
                f"got {checkpoint_payload.get('dataset_id')!r}"
            )
        unknown = sorted(set(checkpoint_payload["crops"]) - old_names)
        if unknown:
            raise SystemExit(f"checkpoint has paths outside the previous manifest: {unknown[:3]}")
        reviewed = {
            name: json.loads(json.dumps(record))
            for name, record in checkpoint_payload["crops"].items()
        }
        for name, record in reviewed.items():
            if not isinstance(record, dict) or record.get("status") not in CHECKPOINT_STATUSES:
                raise SystemExit(f"{name}: invalid reviewed record/status")
            old_image = old_crops / name
            if not old_image.is_file():
                raise SystemExit(f"missing reviewed crop {old_image}")
            _, _, pixel_hash = _pixel_identity(old_image)
            if record["status"] == "verified":
                verified_names.add(name)
                verified_hashes.add(pixel_hash)
            else:
                rejected_hashes.add(pixel_hash)
        rejected_hashes -= verified_hashes
        missing_verified = sorted(verified_names - set(names))
        if missing_verified:
            raise SystemExit(
                f"successor set dropped {len(missing_verified)} verified crops: "
                f"{missing_verified[:3]}"
            )
        for name in names:
            new_image = crops / name
            _, _, new_hash = _pixel_identity(new_image)
            if name in verified_names:
                _, _, old_hash = _pixel_identity(old_crops / name)
                if new_hash != old_hash:
                    raise SystemExit(f"{name}: verified pixels changed in successor set")
            elif new_hash in rejected_hashes:
                raise SystemExit(f"{name}: skipped/unusable pixels returned via an alias")
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    reusable: dict[str, dict[str, Any]] = {}
    if base_draft is not None:
        try:
            draft_payload = json.loads(base_draft.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot read base draft {base_draft}: {exc}") from exc
        if draft_payload.get("schema") != "aoi-joint-boxes/1.0":
            raise SystemExit(f"{base_draft}: unexpected schema")
        if draft_payload.get("coordinate_space") != "crop_pixels_top_left_origin":
            raise SystemExit(f"{base_draft}: unexpected coordinate space")
        if draft_payload.get("classes") != ["component"]:
            raise SystemExit(f"{base_draft}: expected classes ['component']")
        if previous_folder is not None and draft_payload.get("dataset") != previous_folder.name:
            raise SystemExit(f"{base_draft}: dataset does not match --previous-folder")
        if not isinstance(draft_payload.get("crops"), dict):
            raise SystemExit(f"{base_draft}: crops must be an object")
        for name, record in draft_payload["crops"].items():
            if not isinstance(record, dict) or record.get("status", "") != "":
                raise SystemExit(f"{base_draft}: {name} is not an unreviewed detector draft")
            if not isinstance(record.get("boxes", []), list):
                raise SystemExit(f"{base_draft}: {name} boxes must be a list")
            reusable[name] = json.loads(json.dumps(record))

    detector = None
    payload: dict[str, dict[str, object]] = {}
    drawn = dropped = reused = carried = inferred = 0
    for index, row in enumerate(manifest_rows, start=1):
        name = row["crop_path"]
        image_path = crops / name
        if name in verified_names:
            payload[name] = json.loads(json.dumps(reviewed[name]))
            carried += 1
            continue
        if name in reusable:
            old_image = previous_folder / "crops" / name  # guarded above
            if not old_image.is_file():
                raise SystemExit(f"base draft crop is missing: {old_image}")
            if _pixel_identity(old_image)[2] != _pixel_identity(image_path)[2]:
                raise SystemExit(f"{name}: cannot reuse draft on changed pixels")
            payload[name] = json.loads(json.dumps(reusable[name]))
            drawn += len(payload[name].get("boxes", []))
            reused += 1
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            raise SystemExit(f"cannot read crop {image_path}")
        if detector is None:
            detector = create_detector(
                str(model), model_config=ModelDetectorConfig(confidence=confidence)
            )
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
        inferred += 1
        payload[name] = {
            # Rỗng, và phải rỗng: app và packer đều đọc chuỗi này là "chưa ai
            # xem". Đặt 'verified' ở đây là biến phỏng đoán của model thành sự
            # thật, và dạy model sau đúng điểm mù của model trước.
            "status": "",
            "notes": "bản nháp do detector lượt 1 vẽ — cần duyệt",
            "boxes": boxes,
        }
        if index % 25 == 0:
            print(f"  {index}/{len(manifest_rows)} tile …")

    if dry_run:
        print(
            f"[dry-run] {carried} verified giữ nguyên, {reused} draft dùng lại, "
            f"{inferred} tile chạy detector; {drawn} box nháp"
        )
        return 0

    if set(payload) != set(names):
        raise RuntimeError("successor draft is missing manifest rows")
    for name in verified_names:
        if payload[name] != reviewed[name]:
            raise RuntimeError(f"{name}: verified checkpoint record changed")

    verified_semantic = {
        name: payload[name] for name in sorted(verified_names)
    }
    semantic_sha256 = hashlib.sha256(
        json.dumps(
            verified_semantic,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    result = {
        "schema": "aoi-joint-boxes/1.0",
        "dataset_id": dataset_id_for(folder, app_rows, ["component"]),
        "dataset": folder.name,
        "reviewer_id": "" if checkpoint is None else checkpoint_payload.get("reviewer_id", ""),
        "coordinate_space": "crop_pixels_top_left_origin",
        "classes": ["component"],
        "source_checkpoint_sha256": checkpoint_sha256 or None,
        "carried_verified_semantic_sha256": semantic_sha256,
        "note": (
            "Record verified được mang nguyên trạng từ checkpoint; mọi box còn lại "
            "chỉ là bản nháp detector với status rỗng và phải được người duyệt xác nhận."
        ),
        "crops": payload,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=out.parent,
        prefix=f".{out.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(result, handle, indent=1, ensure_ascii=False)
    try:
        temporary.replace(out)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    print(f"\nđã ghi {out}")
    print(f"  giữ nguyên {carried} record verified từ checkpoint")
    print(f"  dùng lại {reused} draft cũ; chạy detector trên {inferred} tile mới")
    print(f"  {drawn} box nháp (trung bình {drawn/max(1,reused+inferred):.1f}/tile chưa duyệt)")
    print(f"  bỏ {dropped} hộp tỉ lệ cạnh >{max_aspect:g} (hộp đặt trên dãy chân, không phải gói)")
    # Đừng bảo người dùng "Nạp file" nữa: từ vòng 2, app được SEED sẵn bằng
    # chính bản nháp này, và nạp tay là đúng thao tác mà README của vòng 2 cấm
    # (checkpoint cũ khác dataset_id, app từ chối). Chỉ dựng app rồi mở.
    print(f"\nBước tiếp: python scripts/build_joint_box_app.py {folder} \\")
    print(f"    --classes component --seed-json {out}")
    print("rồi mở label_boxes.html. App đã có sẵn bản nháp — KHÔNG bấm 'Nạp file'.")
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
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--previous-folder", type=Path)
    parser.add_argument("--base-draft", type=Path,
                        help="tái dùng box nháp status rỗng cho crop cũ")
    parser.add_argument("--output", type=Path,
                        help="mặc định <folder>/draft_boxes.json; không ghi đè")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return prelabel(args.folder.resolve(), args.model.resolve(),
                    confidence=args.confidence, max_aspect=args.max_aspect,
                    checkpoint=args.checkpoint.resolve() if args.checkpoint else None,
                    previous_folder=(args.previous_folder.resolve()
                                     if args.previous_folder else None),
                    base_draft=args.base_draft.resolve() if args.base_draft else None,
                    output=args.output.resolve() if args.output else None,
                    dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
