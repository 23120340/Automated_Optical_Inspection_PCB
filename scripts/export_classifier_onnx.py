"""Xuất ONNX + manifest từ `best_state.pt` của notebook 6.1, chạy tại máy.

Vì sao cần script này: notebook xuất ONNX ở **cell cuối cùng**, sau khi mọi
tính toán đã xong. Cell đó hỏng thì cả run coi như không có artifact, dù trọng
số đã được lưu an toàn từ trước. Đúng chuyện đã xảy ra ngày 2026-08-22:
`torch.onnx.export` báo `ModuleNotFoundError: No module named 'onnxscript'`
trên Kaggle, và cái duy nhất còn lại là `best_state.pt`.

Checkpoint tự mô tả — nó mang `model_name`, `input_size`, `class_names`,
`epoch`, `score` — nên dựng lại model rồi xuất được mà không cần Kaggle, không
cần dataset, không cần GPU.

    python scripts/export_classifier_onnx.py best_state.pt ^
        --out models/library/classifier_v2 --temperature 0.60

Phần hiệu chỉnh (`--temperature`) không nằm trong checkpoint vì nó được tính ở
bước sau. Không truyền thì manifest ghi 1.0 **và ghi rõ là chưa hiệu chỉnh**,
chứ không lặng lẽ giả vờ đã hiệu chỉnh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Notebook 6.1 chuẩn hoá bằng thống kê ImageNet.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
LETTERBOX_VALUE = 114
SCHEMA = "pcb-component-classifier/1.0"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(name: str, num_classes: int):
    """Dựng đúng backbone mà checkpoint đã train, với số lớp của nó."""

    import torch.nn as nn
    import torchvision.models as models

    factory = getattr(models, name, None)
    if factory is None:
        raise SystemExit(f"torchvision không có backbone tên '{name}'")
    network = factory(weights=None)

    # torchvision đặt đầu phân loại ở ba chỗ khác nhau tuỳ họ model.
    if hasattr(network, "classifier"):
        head = network.classifier
        if isinstance(head, nn.Sequential):
            for index in range(len(head) - 1, -1, -1):
                if isinstance(head[index], nn.Linear):
                    head[index] = nn.Linear(head[index].in_features, num_classes)
                    break
            else:
                raise SystemExit(f"không tìm thấy lớp Linear trong classifier của {name}")
        else:
            network.classifier = nn.Linear(head.in_features, num_classes)
    elif hasattr(network, "fc"):
        network.fc = nn.Linear(network.fc.in_features, num_classes)
    elif hasattr(network, "head"):
        network.head = nn.Linear(network.head.in_features, num_classes)
    else:
        raise SystemExit(f"không nhận ra đầu phân loại của {name}")
    return network


def export_onnx(model, dummy, path: Path, opset: int) -> str:
    """Xuất ONNX, ưu tiên đường không cần thêm thư viện. Trả về tên bộ xuất.

    Từ torch 2.9, `torch.onnx.export` mặc định `dynamo=True`, và đường đó uỷ
    quyền cho `onnxscript`. Image Kaggle không có gói này, nên mặc định mới làm
    hỏng đúng cell cuối cùng của notebook.

    `dynamo=False` dùng bộ xuất TorchScript cũ: không cần `onnxscript`, và
    **giữ đúng opset được yêu cầu** — đường mới âm thầm nâng opset lên bản nó
    hỗ trợ (đo được: yêu cầu 12, nhận về 18).

    Bộ xuất cũ đã bị đánh dấu deprecated, nên nếu một ngày torch bỏ hẳn thì
    rơi về đường mới thay vì hỏng.
    """

    import torch

    common = dict(
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **common)
        return "torchscript"
    except Exception as exc:                      # noqa: BLE001
        print(f"  bộ xuất cũ không dùng được ({type(exc).__name__}), thử đường dynamo")
        torch.onnx.export(model, dummy, str(path), dynamo=True, **common)
        return "dynamo"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="best_state.pt từ notebook 6.1")
    parser.add_argument("--out", type=Path, default=Path("models/library/classifier_v2"))
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Nhiệt độ hiệu chỉnh in ra ở bước 5 của notebook. Bỏ trống thì "
             "manifest ghi 1.0 và đánh dấu là CHƯA hiệu chỉnh.",
    )
    parser.add_argument("--accept", type=float, default=0.85)
    parser.add_argument("--review", type=float, default=0.50)
    parser.add_argument(
        "--metrics", type=Path, default=None,
        help="File JSON tuỳ chọn, nội dung được nhét nguyên vào manifest['training'].",
    )
    args = parser.parse_args()

    import torch

    if not args.checkpoint.is_file():
        raise SystemExit(f"Không thấy checkpoint: {args.checkpoint}")

    print(f"Đọc {args.checkpoint} ({args.checkpoint.stat().st_size / 1e6:.1f} MB)")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise SystemExit(
            "Checkpoint không đúng dạng notebook 6.1 (thiếu khoá 'state_dict')."
        )

    class_names = list(checkpoint.get("class_names") or [])
    model_name = str(checkpoint.get("model_name") or "convnext_base")
    input_size = int(checkpoint.get("input_size") or 288)
    if not class_names:
        raise SystemExit("Checkpoint không mang class_names; không dựng được manifest.")

    print(f"  backbone {model_name} · input {input_size} · {len(class_names)} lớp"
          f" · epoch {checkpoint.get('epoch')} · macro recall "
          f"{float(checkpoint.get('score', 0.0)):.4f}")

    model = build_model(model_name, len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    # 350 MB trọng số đã nằm trong model; giữ thêm một bản trong dict là vô ích
    # và máy chạy script này thường không dư bộ nhớ.
    del checkpoint["state_dict"]

    args.out.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out / "best.onnx"
    dummy = torch.zeros(1, 3, input_size, input_size)

    print(f"Xuất ONNX -> {onnx_path}")
    exporter = export_onnx(model, dummy, onnx_path, args.opset)
    print(f"  bộ xuất: {exporter} · {onnx_path.stat().st_size / 1e6:.1f} MB")

    # Đối chiếu: một ONNX nạp được nhưng ra số khác torch còn tệ hơn một ONNX
    # hỏng hẳn, vì nó sai một cách im lặng.
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"input": dummy.numpy()})[0]
    with torch.no_grad():
        torch_out = model(dummy).numpy()
    difference = float(np.max(np.abs(onnx_out - torch_out)))
    if difference >= 1e-3:
        raise SystemExit(f"ONNX lệch torch {difference:.2e} — không dùng được.")
    print(f"  khớp torch (lệch {difference:.2e})")

    batch = session.run(None, {"input": np.zeros((3, 3, input_size, input_size),
                                                 np.float32)})[0]
    print(f"  batch động OK ({batch.shape})")

    calibrated = args.temperature is not None
    training = {
        "source_checkpoint": args.checkpoint.name,
        "epoch": checkpoint.get("epoch"),
        "best_macro_recall": checkpoint.get("score"),
        "used_ema": checkpoint.get("used_ema"),
        "exported_by": "scripts/export_classifier_onnx.py",
        "onnx_exporter": exporter,
    }
    if args.metrics is not None:
        training.update(json.loads(args.metrics.read_text(encoding="utf-8")))

    manifest = {
        "schema_version": SCHEMA,
        "task": "component_family_classification",
        "model_format": "onnx",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "class_names": class_names,
        "input": {
            "name": "input",
            "size": [input_size, input_size],
            "color_space": "RGB",
            "resize_mode": "letterbox",
            "letterbox_value": LETTERBOX_VALUE,
            "normalization": {"mean": MEAN, "std": STD},
        },
        "output": {"name": "logits", "type": "raw_logits"},
        "calibration": {
            "temperature": float(args.temperature) if calibrated else 1.0,
            # Nói thẳng khi chưa hiệu chỉnh. 1.0 trông y hệt "đã hiệu chỉnh và
            # hoá ra bằng 1.0", và hai thứ đó rất khác nhau.
            "calibrated": calibrated,
        },
        "decision_thresholds": {
            "accept": args.accept, "review": args.review, "accept_by_class": {},
        },
        "model": {
            "version": f"classifier-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}",
            "architecture": model_name,
            "sha256": sha256_of(onnx_path),
        },
        "training": training,
    }
    manifest_path = args.out / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    print(f"Ghi {manifest_path}")
    if not calibrated:
        print("  CẢNH BÁO: chưa truyền --temperature, manifest ghi 1.0 và "
              "đánh dấu calibrated=false.")
    print(f"\nXong. Chọn thư mục này trong app: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
