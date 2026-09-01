"""Dựng lại chỗ làm việc gán nhãn trên máy một người mới, bằng một lệnh.

Ảnh crop **không nằm trong git**, và đó là chủ ý chứ không phải thiếu sót. Hai
lý do, lý do thứ hai mới là lý do chính:

1. 120 tile là 215 MB pixel, trong khi repo này công khai;
2. chúng cắt ra từ **CVL PCB-DSLR**, bộ mà chủ dữ liệu giới hạn **nghiên cứu phi
   thương mại**. Điều kiện đó đi theo cả tile phái sinh. Đăng lại chúng trong một
   repo công khai là phát hành lại dữ liệu của người khác sai điều khoản — nên
   mỗi người tự tải nguồn theo đúng điều khoản của nguồn.

Cái *được* commit là thứ làm cho việc dựng lại có thể kiểm chứng: `manifest.csv`
(120 tile nào), `crops.sha256` (đúng pixel nào), `draft_boxes.json` /
`draft_package_boxes.json` (box đã duyệt và box nháp). Script này ghép chúng lại.

    python scripts/setup_labelling_workspace.py \\
        datasets/labelling/component_bodies_round2_20260830

Chạy xong sẽ có `crops/` và trang gán nhãn, giống hệt máy người đã duyệt 16 tile
đầu — giống tới từng byte, và script chứng minh điều đó bằng sha256.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_TILES = PROJECT_ROOT / "datasets" / "test_images" / "tiles_1024"

#: Chuỗi lệnh dựng lại kho tile nguồn, in ra khi thiếu.
FETCH_STEPS = (
    "python scripts/fetch_test_board_images.py",
    "python scripts/fetch_pcb_dslr_diverse_set.py",
    "python scripts/tile_test_images.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_hashes(root: Path) -> dict[str, str]:
    listing = root / "crops.sha256"
    if not listing.is_file():
        return {}
    out: dict[str, str] = {}
    for line in listing.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        out[name.strip()] = digest.strip()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--tiles", type=Path, default=DEFAULT_TILES)
    parser.add_argument(
        "--classes",
        choices=("body", "package", "both"),
        default="both",
        help="dựng trang gán nhãn nào (mặc định: cả hai)",
    )
    parser.add_argument("--skip-pages", action="store_true")
    args = parser.parse_args(argv)

    root = args.workspace.expanduser().resolve()
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        raise SystemExit(f"không thấy {manifest}")

    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    names = [str(row["crop_path"]) for row in rows]
    expected = _expected_hashes(root)
    crops_dir = root / "crops"
    crops_dir.mkdir(exist_ok=True)

    missing = [name for name in names if not (args.tiles / name).is_file()]
    if missing:
        print(f"THIẾU {len(missing)}/{len(names)} tile nguồn trong {args.tiles}")
        for name in missing[:5]:
            print(f"    {name}")
        if len(missing) > 5:
            print(f"    ... và {len(missing) - 5} tile nữa")
        print("\nDựng lại kho tile trước, theo thứ tự:")
        for step in FETCH_STEPS:
            print(f"    {step}")
        print(
            "\nHai script fetch tải ảnh board từ CVL PCB-DSLR. Bộ đó chỉ cho phép "
            "\nNGHIÊN CỨU PHI THƯƠNG MẠI — đọc datasets/test_images/ATTRIBUTION.md "
            "\ntrước khi dùng, điều khoản đi theo cả tile bạn cắt ra."
        )
        return 1

    copied = verified = mismatched = 0
    bad: list[str] = []
    for name in names:
        source, target = args.tiles / name, crops_dir / name
        if not target.is_file() or _sha256(target) != _sha256(source):
            shutil.copy2(source, target)
            copied += 1
        if name in expected:
            if _sha256(target) == expected[name]:
                verified += 1
            else:
                mismatched += 1
                bad.append(name)

    print(f"crops/: {len(names)} tile ({copied} vừa chép)")
    if expected:
        print(f"  đối chiếu crops.sha256: {verified} khớp, {mismatched} lệch")
        if bad:
            print(
                "\nLỆCH PIXEL — đừng gán nhãn tiếp trên bộ này. Tile bạn dựng ra "
                "khác\ntile mà 16 record đã duyệt được vẽ lên, nên toạ độ box sẽ "
                "trỏ sai chỗ:"
            )
            for name in bad[:5]:
                print(f"    {name}")
            return 1
    else:
        print("  (không có crops.sha256 nên không kiểm chứng được pixel)")

    if args.skip_pages:
        return 0

    builder = PROJECT_ROOT / "scripts" / "build_joint_box_app.py"
    jobs = []
    if args.classes in ("body", "both") and (root / "draft_boxes.json").is_file():
        jobs.append((["component"], root / "draft_boxes.json", root / "label_boxes.html"))
    if args.classes in ("package", "both") and (root / "draft_package_boxes.json").is_file():
        from aoi_pipeline.config import PACKAGE_CLASSES

        jobs.append(
            (list(PACKAGE_CLASSES), root / "draft_package_boxes.json",
             root / "label_packages.html")
        )

    for classes, seed, output in jobs:
        command = [
            sys.executable, str(builder), str(root),
            "--classes", *classes,
            "--seed-json", str(seed),
            "--output", str(output),
        ]
        result = subprocess.run(command, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            raise SystemExit(f"dựng {output.name} thất bại")

    print("\nXong. Mở trang sau trong trình duyệt để bắt đầu gán nhãn:")
    for _classes, _seed, output in jobs:
        print(f"    {output}")
    print(
        "\nLƯU Ý cho người mới: trang đã được SEED sẵn — 16 tile đầu hiện trạng "
        "\n`verified` và không cần đụng vào. ĐỪNG bấm 'Nạp file' để nạp checkpoint "
        "\ncủa người khác; mỗi người xuất JSON của riêng mình khi dừng, rồi gộp lại."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
