"""Rewrite a detector manifest's head contract from the ONNX it ships with.

The v2 notebook used to copy ``CONFIG["end2end"]`` and ``CONFIG["max_det"]``
straight into the manifest. YOLO26 exports end-to-end -- output ``(1, 300, 6)``
with NMS inside the graph and a hard 300-detection cap -- regardless of
``nms=False`` at export time and ``end2end=False`` in the config, so the shipped
manifest claimed ``nms: external`` and ``max_det: 2000`` about an artifact that
does neither.

Nothing at runtime reads this file: ``create_detector`` takes a model path and a
config object, and the app only validates the step-6.1 and step-6.2 manifests.
So this is a documentation fix, which is exactly why it is worth a script rather
than a five-hour retrain -- the weights are fine.

    python scripts/fix_detector_manifest.py path/to/model_manifest.json

Pass ``--onnx`` when the ONNX does not sit beside the manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def read_output_shape(onnx_path: Path) -> list[int | str]:
    import onnx

    graph = onnx.load(str(onnx_path)).graph
    return [
        dim.dim_value or dim.dim_param
        for dim in graph.output[0].type.tensor_type.shape.dim
    ]


def head_contract(shape: list[int | str], declared: dict) -> dict:
    """Build the corrected contract, reading the *original* declared values.

    Must stay idempotent: on a second run ``declared`` is this function's own
    previous output, where the config's claim lives under ``declared_end2end``
    and ``max_det`` has already been overwritten with the real cap. Reading only
    the first-run key names would quietly turn the recorded claim into ``None``
    and lose what the config originally said.
    """

    already_rewritten = "declared_end2end" in declared
    contract = {
        "declared_end2end": (
            declared.get("declared_end2end") if already_rewritten
            else declared.get("end2end")
        ),
        "nms": "external",
        # A rewritten manifest no longer carries the config's max_det, so there
        # is nothing truthful to fall back to; leave it to the shape branch.
        "max_det": None if already_rewritten else declared.get("max_det"),
        "onnx_output_shape": shape,
        "actual_end2end": False,
    }
    if len(shape) == 3 and shape[-1] == 6:
        contract.update(
            {"nms": "internal", "actual_end2end": True, "max_det": int(shape[1])}
        )
    return contract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", help="model_manifest.json to rewrite.")
    parser.add_argument("--onnx", default=None, help="ONNX path (default: best.onnx beside it).")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show the change without writing."
    )
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.is_file():
        print(f"Không có manifest: {manifest_path}", file=sys.stderr)
        return 2
    onnx_path = (
        Path(args.onnx).expanduser().resolve()
        if args.onnx
        else manifest_path.parent / "best.onnx"
    )
    if not onnx_path.is_file():
        print(f"Không có ONNX: {onnx_path}", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    before = dict(manifest.get("head") or {})
    shape = read_output_shape(onnx_path)
    after = head_contract(shape, before)

    print(f"ONNX   : {onnx_path.name}  output {tuple(shape)}")
    print(f"trước  : {before}")
    print(f"sau    : {after}")
    if before == after:
        print("\nManifest đã đúng, không cần sửa.")
        return 0
    if args.dry_run:
        print("\n(dry-run: chưa ghi gì)")
        return 0

    manifest["head"] = after
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nĐã ghi lại {manifest_path}")
    if after.get("actual_end2end"):
        print(
            f"Lưu ý: trần thật là {after['max_det']} detection mỗi lần suy luận. "
            "Với board dày đặc hãy dựa vào chia tile — mỗi tile là một lần suy "
            "luận riêng."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
