"""Turn board tiles into a labelling folder for retraining the PASS-1 detector.

Pass 1 finds component *bodies* on a board, and the shipped model does that
badly on large fine-pitch packages: measured on a tile of pcb7, its largest box
is 231x219 at confidence 0.25 and 251x250 even at 0.10, while the QFPs there are
about 350 px. So it never boxes them as a package -- it boxes a lead comb
(aspect ratio 4.5 and 5.6 observed) or the printed text area, and step 5.5 then
derives ROIs from a box that is not the part.

The downloaded generic sources do not fully fix this for a one-class body
detector. The deduplicated RF100 ``printed-circuit-board`` export carries 2,263
IC boxes but only 86 with a short side over 250 px. (Its legacy local directory
is named ``fpic_boards_rf100``, but the dataset is not FPIC.) The official PCB
DSLR annotations are IC-only, so merging whole images directly would make every
other, unlabelled component body a false negative. The two IC-focused Roboflow
sets that do have large boxes (``ic-hpvk3``, ``integrated-circuit-ic``) turned
out to be isolated chips on a tray and on white paper, which is the wrong
problem -- pass 1 has to find a part *among* other parts on a board.

What does have it is already on disk: ``datasets/test_images/tiles_1024``, cut
from 16 MP whole-board photographs and full of QFP and PLCC packages at 300-400
px. This prepares those tiles for the box tool that already exists.

**Box the BODY, not the leads.** That is not a preference, it is what step 5.5
assumes: ``lead_inner_ratio`` reaches 0.14 of the short side back INTO the box
and ``lead_outer_ratio`` reaches 0.26 OUT, so the lead band straddles the box
edge. Measured on the ten hand-checked parts in ``tests/data``, pads sit 42%
inside the detector box and 58% outside, and only 2 of 28 fall wholly inside. A
box drawn around the leads would push that band 0.26 x span past them, onto bare
laminate.

    python scripts/prepare_component_labelling.py --output <thư mục vòng mới> \\
        --checkpoint "~/Downloads/joint_boxes (N).json" \\
        --checkpoint-root <thư mục vòng trước>
    python scripts/prelabel_component_bodies.py <thư mục vòng mới> \\
        --model models/active/detector/best.onnx \\
        --checkpoint "~/Downloads/joint_boxes (N).json" \\
        --previous-folder <thư mục vòng trước> \\
        --base-draft <thư mục vòng trước>/draft_boxes.json
    python scripts/build_joint_box_app.py <thư mục vòng mới> \\
        --classes component --seed-json <thư mục vòng mới>/draft_boxes.json

Bỏ ba cờ ``--checkpoint``/``--previous-folder``/``--base-draft`` khi làm vòng
ĐẦU TIÊN; có chúng thì mọi record đã duyệt được mang sang nguyên trạng và
detector chỉ chạy trên tile mới.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def prepare(
    tiles: Path,
    output: Path,
    *,
    limit: int = 0,
    max_dark: float = 0.6,
    checkpoint: Path | None = None,
    checkpoint_root: Path | None = None,
    max_per_board: int = 6,
    dry_run: bool = False,
) -> int:
    from PIL import Image

    from scripts.pack_component_detection_dataset import (
        CHECKPOINT_STATUSES,
        LOCAL_TAG,
        _expected_checkpoint_dataset_id,
        canonical_board_id,
        read_boxes,
    )

    Image.MAX_IMAGE_PIXELS = None

    if limit < 0:
        raise SystemExit("--limit phải >= 0")
    if not 0.0 <= max_dark <= 1.0:
        raise SystemExit("--max-dark-fraction phải nằm trong [0, 1]")
    if max_per_board < 1:
        raise SystemExit("--max-per-board phải >= 1")
    if checkpoint is not None and checkpoint_root is None:
        raise SystemExit("--checkpoint cần --checkpoint-root để xác thực bộ đã duyệt")
    if checkpoint is None and checkpoint_root is not None:
        raise SystemExit("--checkpoint-root chỉ có nghĩa khi dùng --checkpoint")
    if not dry_run and output.exists():
        raise SystemExit(f"output đã tồn tại; không ghi đè: {output}")

    manifest_path = tiles / "tiles_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"không thấy {manifest_path}; chạy scripts/tile_test_images.py trước")
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))

    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"{manifest_path} không có tile")

    def pixel_identity(path: Path) -> tuple[int, int, str]:
        try:
            with Image.open(path) as handle:
                rgb = handle.convert("RGB")
                width, height = rgb.size
                digest = hashlib.sha256()
                digest.update(width.to_bytes(8, "big"))
                digest.update(height.to_bytes(8, "big"))
                digest.update(rgb.tobytes())
        except (OSError, ValueError) as exc:
            raise SystemExit(f"không đọc được ảnh {path}: {exc}") from exc
        return width, height, digest.hexdigest()

    source_by_name: dict[str, dict[str, Any]] = {}
    by_hash: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for order, raw in enumerate(rows):
        if not isinstance(raw, dict) or not raw.get("file"):
            raise SystemExit(f"{manifest_path}: dòng tile sai định dạng: {raw!r}")
        row = dict(raw)
        name = str(row["file"])
        if name in source_by_name:
            raise SystemExit(f"{manifest_path}: crop_path trùng: {name}")
        image_path = tiles / name
        if not image_path.is_file():
            raise SystemExit(f"manifest trỏ tới ảnh không tồn tại: {image_path}")
        width, height, pixel_hash = pixel_identity(image_path)
        row["_width"] = width
        row["_height"] = height
        row["_pixel_hash"] = pixel_hash
        row["_order"] = order
        scene = str(row.get("source") or name).rsplit(".", 1)[0]
        row["_board"] = canonical_board_id(scene, LOCAL_TAG)
        source_by_name[name] = row
        by_hash[pixel_hash].append(row)

    checkpoint_payload: dict[str, Any] | None = None
    checkpoint_records: dict[str, dict[str, Any]] = {}
    previous_names: set[str] = set()
    verified_names: set[str] = set()
    rejected_names: set[str] = set()
    rejected_hashes: set[str] = set()
    checkpoint_sha256 = ""
    if checkpoint is not None and checkpoint_root is not None:
        previous_manifest = checkpoint_root / "manifest.csv"
        previous_crops = checkpoint_root / "crops"
        for required in (checkpoint, previous_manifest, previous_crops):
            if not required.exists():
                raise SystemExit(f"không thấy {required}")
        with previous_manifest.open(encoding="utf-8", newline="") as handle:
            previous_rows = list(csv.DictReader(handle))
        previous_names = {str(row.get("crop_path", "")) for row in previous_rows}
        if "" in previous_names:
            raise SystemExit(f"{previous_manifest}: có dòng thiếu crop_path")
        checkpoint_payload = read_boxes(checkpoint)
        if checkpoint_payload.get("dataset") != checkpoint_root.name:
            raise SystemExit(
                f"checkpoint thuộc dataset {checkpoint_payload.get('dataset')!r}, "
                f"không phải {checkpoint_root.name!r}"
            )
        expected_id = _expected_checkpoint_dataset_id(checkpoint_root, previous_rows)
        if checkpoint_payload.get("dataset_id") != expected_id:
            raise SystemExit(
                f"checkpoint dataset_id không khớp; cần {expected_id!r}, "
                f"nhận {checkpoint_payload.get('dataset_id')!r}"
            )
        raw_records = checkpoint_payload["crops"]
        unknown = sorted(set(raw_records) - previous_names)
        if unknown:
            raise SystemExit(
                f"checkpoint có {len(unknown)} ảnh ngoài manifest cũ; đầu tiên: {unknown[:3]}"
            )
        missing_source = sorted(set(raw_records) - set(source_by_name))
        if missing_source:
            raise SystemExit(
                f"checkpoint có {len(missing_source)} ảnh không còn trong kho tile; "
                f"đầu tiên: {missing_source[:3]}"
            )
        for name, raw_record in raw_records.items():
            if not isinstance(raw_record, dict):
                raise SystemExit(f"{name}: record checkpoint phải là object")
            status = raw_record.get("status")
            if status not in CHECKPOINT_STATUSES:
                raise SystemExit(
                    f"{name}: status checkpoint {status!r} không hợp lệ; "
                    f"cần một trong {sorted(CHECKPOINT_STATUSES)}"
                )
            old_image = previous_crops / name
            if not old_image.is_file():
                raise SystemExit(f"thiếu crop đã duyệt: {old_image}")
            _, _, old_hash = pixel_identity(old_image)
            source_hash = str(source_by_name[name]["_pixel_hash"])
            if old_hash != source_hash:
                raise SystemExit(f"{name}: pixel nguồn đã đổi sau khi duyệt")
            record = json.loads(json.dumps(raw_record))
            checkpoint_records[name] = record
            if status == "verified":
                boxes = record.get("boxes")
                if not isinstance(boxes, list):
                    raise SystemExit(f"{name}: boxes của record verified phải là list")
                verified_names.add(name)
            else:
                rejected_names.add(name)
                rejected_hashes.add(source_hash)
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    verified_by_hash: dict[str, list[str]] = collections.defaultdict(list)
    for name in sorted(verified_names):
        verified_by_hash[str(source_by_name[name]["_pixel_hash"])].append(name)
    duplicate_verified = [names for names in verified_by_hash.values() if len(names) > 1]
    if duplicate_verified:
        raise SystemExit(
            "checkpoint có nhiều record verified trên cùng pixel; không thể vừa giữ nguyên "
            f"mọi record vừa bỏ ảnh trùng: {duplicate_verified[:3]}"
        )

    def candidate_rank(row: dict[str, Any]) -> tuple[object, ...]:
        # Ưu tiên alias có tên chính thức PCB-DSLR để provenance/audit IC về sau
        # khớp trực tiếp; sau đó lấy tile dày linh kiện và ít nền tối hơn.
        name = str(row["file"])
        official_alias = 0 if name.lower().startswith("pcb_dslr_") else 1
        return (
            -int(row.get("components", 0)),
            float(row.get("dark_fraction", 0.0)),
            official_alias,
            name,
        )

    # Mỗi hash pixel chỉ được có một đại diện. Record verified luôn thắng alias;
    # nếu người duyệt đã loại một ảnh thì mọi alias pixel của nó cũng bị loại,
    # trừ khi chính hash đó có một record verified (trường hợp duplicate đã được
    # người dùng giữ một bản và đánh dấu bản kia là unusable).
    representatives: list[dict[str, Any]] = []
    for pixel_hash, copies in sorted(by_hash.items()):
        verified_copy = [row for row in copies if str(row["file"]) in verified_names]
        if verified_copy:
            representatives.append(verified_copy[0])
            continue
        if pixel_hash in rejected_hashes:
            continue
        eligible = [
            row for row in copies
            if str(row["file"]) not in rejected_names
            and float(row.get("dark_fraction", 0.0)) <= max_dark
        ]
        if eligible:
            representatives.append(min(eligible, key=candidate_rank))

    verified_rows = [
        source_by_name[name]
        for name in sorted(verified_names, key=lambda item: source_by_name[item]["_order"])
    ]
    if limit and len(verified_rows) > limit:
        raise SystemExit(
            f"checkpoint có {len(verified_rows)} ảnh verified, vượt --limit={limit}"
        )

    candidates = [row for row in representatives if str(row["file"]) not in verified_names]
    candidates.sort(key=candidate_rank)
    selected: list[dict[str, Any]] = list(verified_rows)
    selected_names = {str(row["file"]) for row in selected}
    board_counts: collections.Counter[str] = collections.Counter(
        str(row["_board"]) for row in selected
    )

    target = limit or (len(selected) + len(candidates))
    # Mỗi bo vật lý có ít nhất một tile trước khi tăng mật độ trên các bo đã có.
    # Điều này vừa tăng đa dạng miền vừa làm split theo bo có đủ cơ hội tạo
    # train/valid/test, thay vì 120 ảnh nhưng chỉ từ vài PCB.
    by_board: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in candidates:
        by_board[str(row["_board"])].append(row)
    for board in sorted(by_board):
        if len(selected) >= target:
            break
        if board_counts[board] or board_counts[board] >= max_per_board:
            continue
        row = by_board[board][0]
        selected.append(row)
        selected_names.add(str(row["file"]))
        board_counts[board] += 1

    for row in candidates:
        if len(selected) >= target:
            break
        name = str(row["file"])
        board = str(row["_board"])
        if name in selected_names or board_counts[board] >= max_per_board:
            continue
        selected.append(row)
        selected_names.add(name)
        board_counts[board] += 1

    if len(selected) < target:
        raise SystemExit(
            f"chỉ chọn được {len(selected)}/{target} tile độc nhất với "
            f"--max-per-board={max_per_board}; tăng giới hạn hoặc giảm --limit"
        )

    selected_hashes = [str(row["_pixel_hash"]) for row in selected]
    if len(selected_hashes) != len(set(selected_hashes)):
        raise RuntimeError("lỗi nội bộ: bộ chọn vẫn để lọt pixel trùng")

    out_rows: list[dict[str, object]] = []
    for row in selected:
        source = tiles / str(row["file"])
        width, height = int(row["_width"]), int(row["_height"])
        out_rows.append({
            "crop_path": source.name,
            # Ô này là cả một mảng board, không phải một linh kiện, nên không có
            # "lớp linh kiện" để hiện. Ghi số linh kiện lượt 1 tìm được để người
            # duyệt biết ảnh này dày hay thưa trước khi mở.
            "component_class": f"tile ~{row.get('components', '?')} linh kiện",
            "dataset_source": "tiles_1024",
            # Cảnh = ảnh board gốc. Chia tập theo nó, không theo tile: hai tile
            # cạnh nhau chồng lấn 256 px nên chia theo tile là rò rỉ.
            "scene_id": str(row.get("source", source.name)).rsplit(".", 1)[0],
            "crop_w": width,
            "crop_h": height,
            # Không có khung gợi ý: thứ cần khoanh là MỌI thân linh kiện trong
            # ảnh, không phải một cái đã biết trước. Số 0 tắt phần vẽ gợi ý.
            "body_x": 0, "body_y": 0, "body_w": 0, "body_h": 0,
            "roi_kind": "board_tile",
            "label_status": "", "reviewer_id": "", "notes": "",
        })

    if dry_run:
        print(f"[dry-run] {len(out_rows)} tile độc nhất sẽ được chuẩn bị")
        print(f"  giữ nguyên {len(verified_names)} tile verified từ checkpoint")
        print(f"  loại {sum(len(group) - 1 for group in by_hash.values())} bản sao pixel")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        crops = staging / "crops"
        crops.mkdir()
        for row in selected:
            source = tiles / str(row["file"])
            shutil.copy2(source, crops / source.name)
        with (staging / "manifest.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)

        retained_unreviewed = sum(
            str(row["file"]) in previous_names and str(row["file"]) not in verified_names
            for row in selected
        )
        newly_added = sum(str(row["file"]) not in previous_names for row in selected)
        provenance = {
            "schema": "aoi-component-labelling-preparation/2.0",
            "source": "datasets/test_images/tiles_1024",
            "tiles_available": len(rows),
            "unique_pixel_groups_available": len(by_hash),
            "exact_duplicate_rows_available": sum(len(group) - 1 for group in by_hash.values()),
            "tiles_prepared": len(out_rows),
            "unique_pixel_groups_prepared": len(set(selected_hashes)),
            "physical_boards_prepared": len(board_counts),
            "max_dark_fraction": max_dark,
            "max_tiles_per_physical_board": max_per_board,
            "selection_policy": (
                "verified checkpoint records first; reject skipped/unusable pixel groups; "
                "one decoded-pixel representative; cover physical boards; then component "
                "count descending under the per-board cap"
            ),
            "checkpoint": None if checkpoint_payload is None else {
                "file": checkpoint.name if checkpoint is not None else "",
                "sha256": checkpoint_sha256,
                "dataset": checkpoint_payload.get("dataset"),
                "dataset_id": checkpoint_payload.get("dataset_id"),
                "exported_at": checkpoint_payload.get("exported_at"),
                "verified_carried": len(verified_names),
                "verified_boxes_carried": sum(
                    len(checkpoint_records[name].get("boxes", [])) for name in verified_names
                ),
                "reviewed_rejected": len(rejected_names),
            },
            "retained_unreviewed_from_previous_set": retained_unreviewed,
            "new_tiles_from_source_pool": newly_added,
            "purpose": "gán nhãn THÂN linh kiện để train lại detector lượt 1",
            "box_convention": (
                "Khoanh THÂN (gói đen / thân gốm / vỏ can), KHÔNG bao chân. Bước 5.5 "
                "đặt dải chân vắt qua mép hộp (lead_inner_ratio 0.14 vào trong, "
                "lead_outer_ratio 0.26 ra ngoài). Đo trên 10 linh kiện kiểm tay: pad "
                "nằm 42% trong hộp, 58% ngoài."
            ),
        }
        (staging / "provenance.json").write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"đã chuẩn bị {len(out_rows)} tile → {output}")
    print(f"  giữ nguyên {len(verified_names)} tile verified từ checkpoint")
    print(f"  loại {sum(len(group) - 1 for group in by_hash.values())} bản sao pixel")
    print(f"  {len(board_counts)} bo vật lý; tối đa {max(board_counts.values(), default=0)} tile/bo")
    counts = sorted((int(r.get("components", 0)) for r in selected), reverse=True)
    if counts:
        print(f"  linh kiện/tile: nhiều nhất {counts[0]}, trung vị {counts[len(counts)//2]}")
    # Ba bước, không phải một: bỏ qua prelabel thì người duyệt phải VẼ thay vì
    # SỬA, và bỏ --seed-json thì app mở ra trắng trơn dù bản nháp đã có sẵn.
    print("\nBước tiếp:")
    print(f"  1. python scripts/prelabel_component_bodies.py {output} \\")
    print("       --model models/active/detector/best.onnx")
    print(f"  2. python scripts/build_joint_box_app.py {output} \\")
    print(f"       --classes component --seed-json {output / 'draft_boxes.json'}")
    print(f"  3. mở {output / 'label_boxes.html'} — app đã seed, KHÔNG bấm 'Nạp file'")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tiles", type=Path,
                        default=PROJECT_ROOT / "datasets" / "test_images" / "tiles_1024")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "datasets" / "labelling" / "component_bodies")
    parser.add_argument("--limit", type=int, default=120,
                        help="số tile chuẩn bị, ưu tiên tile nhiều linh kiện (0 = tất cả)")
    parser.add_argument("--max-dark-fraction", type=float, default=0.6)
    parser.add_argument("--checkpoint", type=Path,
                        help="checkpoint JSON mới nhất; chỉ record verified được mang sang")
    parser.add_argument("--checkpoint-root", type=Path,
                        help="thư mục labelling cũ mà checkpoint thuộc về")
    parser.add_argument("--max-per-board", type=int, default=6,
                        help="trần tile trên mỗi bo vật lý sau các record verified")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return prepare(args.tiles.resolve(), args.output.resolve(),
                   limit=args.limit, max_dark=args.max_dark_fraction,
                   checkpoint=args.checkpoint.resolve() if args.checkpoint else None,
                   checkpoint_root=(args.checkpoint_root.resolve()
                                    if args.checkpoint_root else None),
                   max_per_board=args.max_per_board,
                   dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
