"""Liệt kê mọi model trong `models/`, kèm thứ phân biệt được chúng với nhau.

Mọi artifact trong dự án đều tên `best.onnx`, nên nhìn thư mục không biết bản
nào là bản nào. Thông tin ấy nằm trong `model_manifest.json` bên cạnh — script
này đọc ra và xếp thành bảng.

    python scripts/list_models.py
    python scripts/list_models.py --kind classifier
    python scripts/list_models.py --json          # cho script khác dùng

Manifest là nguồn đáng tin, không phải tên thư mục: tên do người đặt và có thể
lệch khỏi file, manifest do notebook sinh ra cùng lúc với trọng số.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aoi_pipeline.modelops.model_registry import (  # noqa: E402
    ACTIVE_ROOT,
    ARCHIVE_ROOT,
    LIBRARY_ROOT,
    MODELS_ROOT,
    discover_models,
)

ORIGIN_LABEL = {
    "active": "đang dùng",
    "archive": "bản cũ",
    "library": "của bạn",
}
ORIGIN_ORDER = {"active": 0, "library": 1, "archive": 2}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        choices=["detector", "classifier", "package", "package_classifier", "solder"],
                        help="chỉ liệt kê một bước")
    parser.add_argument("--json", action="store_true", help="in JSON thay vì bảng")
    parser.add_argument("--all", action="store_true",
                        help="kể cả model thiếu manifest (app sẽ không nạp được)")
    args = parser.parse_args()

    entries = discover_models(args.kind, require_manifest=not args.all)
    entries.sort(key=lambda e: (e.kind, ORIGIN_ORDER.get(e.origin, 9),
                                e.summary().created or ""))

    if args.json:
        payload = [
            {
                "kind": e.kind, "origin": e.origin,
                "path": e.model_path.relative_to(MODELS_ROOT.parent).as_posix(),
                "size_mb": round(e.size_mb, 1),
                **{k: v for k, v in vars(e.summary()).items()},
            }
            for e in entries
        ]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if not entries:
        print("Không có model nào." if args.kind is None
              else f"Không có model nào cho bước {args.kind}.")
        print(f"Thả file .onnx kèm model_manifest.json vào {LIBRARY_ROOT}")
        return 0

    def folder_of(entry) -> str:
        return entry.name.rsplit("/", 1)[0] if "/" in entry.name else entry.name

    # Co giãn theo tên dài nhất: quy ước <bước>-<kiến trúc>-<ngày> cho ra những
    # tên khá dài, và một cột cứng sẽ để chúng tràn sang cột bên cạnh.
    width = max([len("thư mục")] + [len(folder_of(e)) for e in entries]) + 2
    # Cột bước cũng phải co giãn: từ khi bước 6.2 tách vai trò, ``kind`` có thể
    # là ``solder_classifier`` (17 ký tự) và một cột cứng 11 sẽ đè lên cột kế.
    kind_width = max([len("bước")] + [len(e.kind) for e in entries]) + 2
    # Ô ``solder_segmenter`` nhận cả model detect lẫn model segment, và tên ô
    # không nói được đang là cái nào -- trong khi đổi giữa hai thứ đó đổi luôn
    # hành vi của bước 6.2. Task khai trong manifest là chỗ duy nhất nói thật.
    tasks = [(e.summary().task or "") for e in entries]
    task_width = max([len("task")] + [len(t) for t in tasks]) + 2
    header = (f"{'bước':<{kind_width}}{'nguồn':<11}{'thư mục':<{width}}"
              f"{'task':<{task_width}}"
              f"{'kiến trúc':<20}{'ngày':<12}{'điểm':<18}{'MB':>6}")
    print(header)
    print("-" * len(header))
    current = None
    for entry in entries:
        if entry.kind != current:
            if current is not None:
                print()
            current = entry.kind
        summary = entry.summary()
        print(
            f"{entry.kind:<{kind_width}}"
            f"{ORIGIN_LABEL.get(entry.origin, entry.origin):<11}"
            f"{folder_of(entry):<{width}}"
            f"{(summary.task or '—'):<{task_width}}"
            f"{(summary.architecture or '—'):<20}"
            f"{(summary.created or '—'):<12}"
            f"{(summary.metric or '—'):<18}"
            f"{entry.size_mb:>6.0f}"
        )

    print()
    print(f"đang dùng = app tự nạp    ({ACTIVE_ROOT.relative_to(MODELS_ROOT.parent)})")
    print(f"của bạn   = bạn tự thả vào ({LIBRARY_ROOT.relative_to(MODELS_ROOT.parent)}, git bỏ qua)")
    print(f"bản cũ    = giữ để đối chiếu, KHÔNG tự nạp ({ARCHIVE_ROOT.relative_to(MODELS_ROOT.parent)})")

    missing = [
        e for e in discover_models(args.kind, require_manifest=False)
        if not e.has_manifest
    ]
    if missing and not args.all:
        print()
        print(f"Còn {len(missing)} file .onnx KHÔNG có manifest bên cạnh nên app không "
              "nạp được (chạy với --all để xem).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
