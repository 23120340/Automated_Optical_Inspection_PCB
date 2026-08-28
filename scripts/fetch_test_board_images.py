"""Collect high-resolution whole-board photographs for testing the pipeline.

"High quality" here is not "sharpest". It is a **pixel-scale** question, and this
project has measured what happens when that is got wrong: SolDef_AI is macro
photography at 1-3 um/px and a model trained on it returned **zero** boxes on
this project's board at every magnification from 1x to 12x. A 640x640 board
export fails the other way -- a chip part lands on 10 px and nothing about its
joints survives.

So an image earns a place here by clearing three measured bars:

* **whole board in frame** -- the pipeline's first three steps localise and align
  a board, which needs the board, not a close-up of one component;
* **native resolution kept** -- no re-export through a 640 px letterbox;
* **components still resolvable** -- at this project's 46 um/px a chip part is
  about 62x58 px and its pad about 23 px, so a board photographed at 12+ MP
  leaves room to downscale to that scale rather than upscale toward it.

Sources, and why each is here:

``cvl_pcb_dslr``
    40 boards, 175 exposures at 4928x3280 (16.2 MP), DSLR. The highest-quality
    public whole-board set found. Zenodo DOI 10.5281/zenodo.3886553. The source
    text limits use to **non-commercial research**; that restriction travels
    with the files and is recorded in ATTRIBUTION.md.

The two archives are not downloaded whole: ``zipfile`` reads them through a
seekable HTTP range reader, so only the selected members and the central
directory cross the wire.

    python scripts/fetch_test_board_images.py --output datasets/test_images
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fetch_pcb_dslr_diverse_set import (  # noqa: E402
    ARCHIVES,
    RemoteRangeReader,
)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

IMAGE_MEMBER = re.compile(r"(pcb\d+)/(rec\d+)\.jpg$")

ATTRIBUTION = """# Nguồn ảnh test

## `cvl_pcb_dslr/` — {count} ảnh, 4928×3280 (16,2 MP)

CVL PCB DSLR Dataset, TU Wien.

* Trang chính thức: https://cvl.tuwien.ac.at/research/cvl-databases/pcb-dslr-dataset/
* Dữ liệu: https://zenodo.org/records/3886553 (DOI 10.5281/zenodo.3886553)
* Bài báo: C. Pramerdorfer, M. Kampel, *A Dataset for Computer-Vision-Based PCB
  Analysis*, MVA 2015.

**Giấy phép: nghiên cứu PHI THƯƠNG MẠI.** Điều kiện đó đi theo các file này.
Không đẩy lên repo công khai, không dùng thương mại. Trích dẫn bài MVA 2015 nếu
công bố kết quả.

Mỗi board được chụp nhiều lần (`rec1`…`rec5`) ở các vị trí/ánh sáng khác nhau —
đó là lý do số ảnh nhiều hơn số board, và cũng là thứ làm bộ này hữu ích để thử
bước 2 (align) và bước 3 (khoanh board).

Tên file: `<board>__<recording>.jpg`, ví dụ `pcb07__rec3.jpg`.
"""


def fetch(output: Path, *, limit: int | None = None, dry_run: bool = False) -> int:
    images = output / "cvl_pcb_dslr"
    if not dry_run:
        images.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    written = skipped = 0
    for spec in ARCHIVES:
        print(f"đọc {spec.filename} …")
        reader = RemoteRangeReader(spec.url)
        with reader, ZipFile(reader) as archive:
            members = [n for n in archive.namelist() if IMAGE_MEMBER.search(n)]
            members.sort(key=lambda n: (
                int(re.search(r"pcb(\d+)", n).group(1)),
                int(re.search(r"rec(\d+)", n).group(1)),
            ))
            for name in members:
                if limit is not None and written >= limit:
                    break
                board, recording = IMAGE_MEMBER.search(name).groups()
                target = images / f"{board}__{recording}.jpg"
                if target.exists():
                    skipped += 1
                    continue
                if dry_run:
                    written += 1
                    continue
                payload = archive.read(name)
                target.write_bytes(payload)
                manifest.append({
                    "file": target.name,
                    "board": board,
                    "recording": recording,
                    "source_archive": spec.filename,
                    "source_member": name,
                    "bytes": len(payload),
                    "sha256": sha256(payload).hexdigest(),
                })
                written += 1
                if written % 20 == 0:
                    print(f"  {written} ảnh …")
        if limit is not None and written >= limit:
            break

    if dry_run:
        print(f"[dry-run] sẽ lấy {written} ảnh, bỏ qua {skipped} đã có")
        return 0

    if manifest:
        record = output / "cvl_pcb_dslr_manifest.json"
        existing = (
            json.loads(record.read_text(encoding="utf-8")) if record.exists() else []
        )
        by_name = {row["file"]: row for row in existing}
        by_name.update({row["file"]: row for row in manifest})
        record.write_text(
            json.dumps(sorted(by_name.values(), key=lambda r: r["file"]),
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    total = len(list(images.glob("*.jpg")))
    (output / "ATTRIBUTION.md").write_text(
        ATTRIBUTION.format(count=total), encoding="utf-8"
    )
    print(f"\nđã ghi {written} ảnh mới, bỏ qua {skipped} đã có → {images}")
    print(f"tổng trong thư mục: {total} ảnh")
    return 0



# --------------------------------------------------------------------------- #
# Gom từ kho đã có trên đĩa
# --------------------------------------------------------------------------- #

#: Ngưỡng megapixel để một ảnh được coi là "chất lượng cao" cho việc test.
#:
#: Không phải con số tuỳ ý. Board dự án chụp ở 46 um/px cho linh kiện ~62x58 px;
#: một ảnh 6 MP trở lên còn dư chỗ để HẠ về tỉ lệ đó, còn ảnh dưới ngưỡng thì
#: muốn tới đó phải PHÓNG LÊN, mà phóng to bằng phần mềm không tạo ra chi tiết
#: chưa từng được chụp -- đã đo trên SolDef_AI: 0 box ở mọi mức 1x..12x.
MIN_MEGAPIXELS = 6.0

#: Kho trên đĩa và tên thư mục đích. Mỗi nguồn giữ giấy phép riêng, xem
#: ATTRIBUTION.md của chính nó trong ``datasets/``.
LOCAL_SOURCES = {
    "mpi_pcb_gas_pump": Path("datasets/reference_sets/mpi_pcb_gas_pump_same_board_30/images"),
    "pcb_dslr_diverse": Path("datasets/reference_sets/pcb_dslr_30_diverse/boards"),
}


def gather_local(output: Path, *, min_megapixels: float = MIN_MEGAPIXELS,
                 dry_run: bool = False) -> int:
    """Copy every locally held board photo that clears the resolution gate.

    Measured, not assumed: each candidate is opened and its real pixel count
    read. A folder's reputation is not evidence -- ``visa_pcb2_30`` sits in the
    same tree and its images are 1.5 MP, which is below anything useful here.
    """

    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    rows: list[dict[str, object]] = []
    for tag, folder in LOCAL_SOURCES.items():
        source = (PROJECT_ROOT / folder).resolve()
        if not source.is_dir():
            print(f"  bỏ qua {tag}: không thấy {source}")
            continue
        target = output / tag
        if not dry_run:
            target.mkdir(parents=True, exist_ok=True)
        kept = rejected = skipped_kind = 0
        for path in sorted(list(source.rglob("*.jpg")) + list(source.rglob("*.png"))):
            # Mask nhị phân và ảnh nhãn nằm CÙNG thư mục với ảnh chụp và cũng
            # thừa sức qua ngưỡng megapixel -- ngưỡng đo độ phân giải, không đo
            # "đây có phải ảnh chụp không". Lọc theo tên trước khi đo.
            if any(mark in path.stem.lower()
                   for mark in ("-mask", "_mask", "-annot", "_annot", "-label", "_label")):
                skipped_kind += 1
                continue
            try:
                with Image.open(path) as handle:
                    width, height = handle.size
            except Exception:
                continue
            megapixels = width * height / 1e6
            if megapixels < min_megapixels:
                rejected += 1
                continue
            # Tên phẳng, mang cả thư mục cha: pcb_dslr lưu mọi ảnh là ``rec1.jpg``
            # trong thư mục riêng, gộp phẳng mà không đặt lại tên thì chỉ còn một.
            stem = (f"{path.parent.name}__{path.stem}"
                    if path.parent != source else path.stem)
            destination = target / f"{stem}{path.suffix.lower()}"
            if not dry_run and not destination.exists():
                destination.write_bytes(path.read_bytes())
            rows.append({
                "file": f"{tag}/{destination.name}",
                "source": str(path.relative_to(PROJECT_ROOT)),
                "width": width,
                "height": height,
                "megapixels": round(megapixels, 1),
            })
            kept += 1
        print(f"  {tag:<22}{kept:>4} giữ, {rejected:>4} dưới {min_megapixels:g} MP, "
              f"{skipped_kind:>4} không phải ảnh chụp (mask/nhãn)")
    if rows and not dry_run:
        (output / "local_sources_manifest.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "datasets" / "test_images")
    parser.add_argument("--limit", type=int, default=None,
                        help="dừng sau ngần này ảnh mới; để trống là lấy hết")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-only", action="store_true",
                        help="chỉ gom từ kho đã có trên đĩa, không tải gì")
    parser.add_argument("--skip-local", action="store_true")
    parser.add_argument("--min-megapixels", type=float, default=MIN_MEGAPIXELS)
    args = parser.parse_args(argv)

    output = args.output.resolve()
    if not args.local_only:
        fetch(output, limit=args.limit, dry_run=args.dry_run)
    if not args.skip_local:
        print("")
        print("gom từ kho trên đĩa:")
        gather_local(output, min_megapixels=args.min_megapixels, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
