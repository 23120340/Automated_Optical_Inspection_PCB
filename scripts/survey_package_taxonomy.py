"""Khảo sát cây family -> package: crop, contact sheet, đặc trưng hình học, cụm.

Chỉ KHẢO SÁT. Không viết luật phân loại, không train gì, không đặt tên package
công nghiệp cho cụm — cụm chỉ được đánh số, người xem ảnh mới gán tên thật.

Hai nguồn công khai bù nhau đúng chỗ, cả hai đều CC BY 4.0:

* ``fpic_boards_rf100`` (Roboflow-100 printed-circuit-board v4) — 23 lớp ở mức
  **HỌ**: IC, Transistor, Connector, Resistor... và quan trọng nhất là
  ``Electrolytic Capacitor``, tức tụ trụ đứng — thứ mà bộ winnies không có một
  ví dụ nào.
* ``pcb_packages_winnies`` (pcb-components-wc8ms v3) — 24 lớp ở mức **PACKAGE**
  với tên công nghiệp thật: SOT23, SOIC-16, TSSOP-14, SOD123... Đây là chỗ tra
  tên chuẩn thay vì tự nghĩ ra.

Đọc thẳng từ file zip, không giải nén: hai bộ cộng lại hơn 100 MB ảnh và máy
làm việc chỉ có 8 GB RAM.

    python scripts/survey_package_taxonomy.py --out datasets/survey/<tên>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCES = {
    "fpic": {
        "zip": "datasets/public/fpic_boards_rf100/export_yolov8_v4.zip",
        "level": "family",
        "attribution": "Roboflow-100 printed-circuit-board v4, CC BY 4.0",
    },
    "winnies": {
        "zip": "datasets/public/pcb_packages_winnies/export_yolov8_v3.zip",
        "level": "package",
        "attribution": "winnies-workspace pcb-components-wc8ms v3, CC BY 4.0",
    },
}

#: Dưới ngưỡng này thì contour là nhiễu chứ không phải hình dạng. Vẫn ghi vào
#: CSV nhưng các cột hình học để trống, để không ai lỡ gộp chúng vào một cụm.
MIN_SIDE_FOR_SHAPE = 12
#: Trần số crop GHI RA ĐĨA cho mỗi lớp. Đặc trưng thì tính cho *mọi* box (rẻ),
#: chỉ ảnh mới bị chặn — contact sheet chỉ cần 64 ảnh, và 134k PNG là vô ích.
MAX_CROPS_PER_CLASS = 400
SHEET_GRID = 8
SHEET_CELL = 96


@dataclass(slots=True)
class BoxFeature:
    source: str
    level: str
    label: str
    image: str
    index: int
    x: int
    y: int
    w: int
    h: int
    long_side: int
    short_side: int
    aspect: float
    area_px: int
    area_frac: float
    crop_path: str
    # Các cột dưới đây rỗng khi box quá nhỏ để có hình dạng thật.
    poly_vertices: int | None = None
    solidity: float | None = None
    extent: float | None = None
    circularity: float | None = None
    hull_defects: int | None = None
    fill_ratio: float | None = None
    #: Gán ở bước 4. Rỗng khi lớp quá ít mẫu để phân cụm.
    cluster: str = ""


def _shape_features(crop: np.ndarray) -> dict[str, float | int]:
    """Đặc trưng contour của thân, sau khi tách nền bằng Otsu.

    ``circularity`` là cột đắt giá nhất ở đây: kế hoạch cần nó để tách tụ hoá
    trụ đứng (nhìn từ trên là hình tròn) khỏi tụ chip (chữ nhật), mà cho tới
    giờ chưa có bộ nào để hiệu chuẩn ngưỡng.
    """

    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (3, 3), 0)
    _, mask = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Thân có thể sáng hơn hoặc tối hơn nền tuỳ mặt board. Chọn cực nào chạm
    # biên ảnh ít hơn: thân nằm giữa crop, nền mới là thứ chạm mép.
    border = np.concatenate(
        [mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]]
    )
    if border.mean() > 127:
        mask = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {}
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if area <= 0 or perimeter <= 0:
        return {}

    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    _, _, cw, ch = cv2.boundingRect(contour)

    defects_count = 0
    if len(contour) > 3:
        hull_idx = cv2.convexHull(contour, returnPoints=False)
        if hull_idx is not None and len(hull_idx) > 3:
            order = np.sort(hull_idx.reshape(-1))[::-1].astype(np.int32)
            try:
                defects = cv2.convexityDefects(contour, order[:, None])
            except cv2.error:
                defects = None
            if defects is not None and defects.size:
                # Chỉ đếm chỗ lõm ĐỦ SÂU. Rìa răng cưa do threshold tạo ra
                # hàng chục defect nông; ngưỡng theo cạnh dài của contour.
                # convexityDefects trả (N,1,4) ở bản này và (N,4) ở bản
                # khác, nên ép về (N,4) thay vì tin vào một dạng.
                depths = defects.reshape(-1, 4)[:, 3]
                floor = 0.06 * max(cw, ch) * 256.0
                defects_count = int((depths > floor).sum())

    return {
        "poly_vertices": int(len(approx)),
        "solidity": round(area / hull_area, 4) if hull_area > 0 else None,
        "extent": round(area / float(cw * ch), 4) if cw * ch > 0 else None,
        "circularity": round(4.0 * math.pi * area / (perimeter * perimeter), 4),
        "hull_defects": defects_count,
        "fill_ratio": round(float(mask.mean()) / 255.0, 4),
    }


def _iter_boxes(zip_path: Path):
    """Sinh (tên lớp, tên ảnh, ảnh BGR, danh sách box pixel) cho từng ảnh."""

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        meta = yaml.safe_load(
            archive.read(next(n for n in names if n.endswith("data.yaml")))
        )
        class_names = list(meta["names"])
        images = sorted(
            n for n in names
            if "/images/" in n and n.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        for image_name in images:
            label_name = (
                image_name.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            )
            if label_name not in names:
                continue
            buffer = np.frombuffer(archive.read(image_name), dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if image is None:
                continue
            height, width = image.shape[:2]
            rows = []
            for line in archive.read(label_name).decode("utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cid, cx, cy, bw, bh = (
                    int(parts[0]), *(float(v) for v in parts[1:5])
                )
                x = int(round((cx - bw / 2) * width))
                y = int(round((cy - bh / 2) * height))
                w = int(round(bw * width))
                h = int(round(bh * height))
                x, y = max(0, x), max(0, y)
                w, h = max(1, min(w, width - x)), max(1, min(h, height - y))
                rows.append((class_names[cid], x, y, w, h))
            yield image_name, image, rows


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def survey(out_root: Path, sources: dict[str, dict]) -> list[BoxFeature]:
    features: list[BoxFeature] = []
    written = Counter()
    for source, spec in sources.items():
        zip_path = PROJECT_ROOT / spec["zip"]
        if not zip_path.is_file():
            print(f"bỏ qua {source}: không thấy {zip_path}")
            continue
        print(f"--- {source} ({spec['level']}) {zip_path.name}")
        for image_name, image, rows in _iter_boxes(zip_path):
            ih, iw = image.shape[:2]
            for index, (label, x, y, w, h) in enumerate(rows):
                crop = image[y : y + h, x : x + w]
                if crop.size == 0:
                    continue
                key = (source, label)
                crop_path = ""
                if written[key] < MAX_CROPS_PER_CLASS:
                    folder = out_root / "crops" / source / _safe(label)
                    folder.mkdir(parents=True, exist_ok=True)
                    target = folder / f"{written[key]:04d}.png"
                    cv2.imwrite(str(target), crop)
                    crop_path = str(target.relative_to(out_root)).replace("\\", "/")
                    written[key] += 1

                item = BoxFeature(
                    source=source,
                    level=spec["level"],
                    label=label,
                    image=image_name,
                    index=index,
                    x=x, y=y, w=w, h=h,
                    long_side=max(w, h),
                    short_side=min(w, h),
                    aspect=round(max(w, h) / max(1, min(w, h)), 4),
                    area_px=w * h,
                    area_frac=round(w * h / float(iw * ih), 8),
                    crop_path=crop_path,
                )
                if min(w, h) >= MIN_SIDE_FOR_SHAPE:
                    for field, value in _shape_features(crop).items():
                        setattr(item, field, value)
                features.append(item)
        print(f"    {sum(1 for f in features if f.source == source)} box")
    return features


def contact_sheets(out_root: Path, features: list[BoxFeature], subdir: str,
                   group) -> int:
    """Lưới ảnh 8x8 cho mỗi nhóm, để người duyệt bằng mắt."""

    buckets: dict[str, list[BoxFeature]] = defaultdict(list)
    for item in features:
        if item.crop_path:
            buckets[group(item)].append(item)

    folder = out_root / subdir
    folder.mkdir(parents=True, exist_ok=True)
    made = 0
    rng = random.Random(42)
    for name, items in sorted(buckets.items()):
        picked = items if len(items) <= SHEET_GRID ** 2 else rng.sample(
            items, SHEET_GRID ** 2
        )
        sheet = np.full(
            (SHEET_GRID * SHEET_CELL + 26, SHEET_GRID * SHEET_CELL, 3),
            32, dtype=np.uint8,
        )
        cv2.putText(
            sheet, f"{name}  (n={len(items)})", (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA,
        )
        for position, item in enumerate(picked):
            crop = cv2.imread(str(out_root / item.crop_path))
            if crop is None:
                continue
            scale = min(
                (SHEET_CELL - 6) / crop.shape[1], (SHEET_CELL - 6) / crop.shape[0]
            )
            resized = cv2.resize(
                crop,
                (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST,
            )
            row, col = divmod(position, SHEET_GRID)
            oy = 26 + row * SHEET_CELL + (SHEET_CELL - resized.shape[0]) // 2
            ox = col * SHEET_CELL + (SHEET_CELL - resized.shape[1]) // 2
            sheet[oy : oy + resized.shape[0], ox : ox + resized.shape[1]] = resized
        cv2.imwrite(str(folder / f"{_safe(name)}.png"), sheet)
        made += 1
    return made


def _kmeans(matrix: np.ndarray, k: int, *, seed: int = 42,
            iterations: int = 100) -> np.ndarray:
    """KMeans++ đủ dùng cho 5 cột và vài nghìn dòng.

    Tự viết thay vì kéo scikit-learn vào: repo này chỉ khai báo những phụ thuộc
    thật sự cần cho pipeline, và một script khảo sát dùng một lần không đáng
    thêm một thư viện nặng vào requirements.
    """

    rng = np.random.default_rng(seed)
    n = len(matrix)
    # KMeans++: tâm đầu ngẫu nhiên, các tâm sau lấy theo xác suất tỉ lệ với
    # bình phương khoảng cách tới tâm gần nhất đã chọn.
    centres = [matrix[rng.integers(n)]]
    for _ in range(1, k):
        gaps = np.min(
            ((matrix[:, None, :] - np.array(centres)[None]) ** 2).sum(axis=2), axis=1
        )
        total = gaps.sum()
        if total <= 0:
            centres.append(matrix[rng.integers(n)])
            continue
        centres.append(matrix[rng.choice(n, p=gaps / total)])
    centre_matrix = np.array(centres)

    labels = np.zeros(n, dtype=int)
    for _ in range(iterations):
        distances = ((matrix[:, None, :] - centre_matrix[None]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for index in range(k):
            members = matrix[labels == index]
            if len(members):
                centre_matrix[index] = members.mean(axis=0)
    return labels


def cluster(features: list[BoxFeature], max_k: int = 4, min_samples: int = 40
            ) -> dict[tuple[str, str], dict]:
    """KMeans nhỏ trong TỪNG lớp. Cụm chỉ đánh số, không đặt tên package.

    Đặt tên chuẩn công nghiệp cho một cụm là việc phải nhìn ảnh mới làm được;
    máy đoán tên ở đây chỉ tạo ra một cái nhãn trông như đã xong.
    """

    report: dict[tuple[str, str], dict] = {}
    by_class: dict[tuple[str, str], list[BoxFeature]] = defaultdict(list)
    for item in features:
        if item.circularity is not None:
            by_class[(item.source, item.label)].append(item)

    for key, items in sorted(by_class.items()):
        if len(items) < min_samples:
            continue
        matrix = np.array(
            [
                [
                    math.log10(max(1, item.area_px)),
                    item.aspect,
                    item.circularity or 0.0,
                    item.solidity or 0.0,
                    float(item.poly_vertices or 0),
                ]
                for item in items
            ],
            dtype=np.float64,
        )
        matrix = (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1e-9)
        k = min(max_k, max(2, len(items) // 60))
        labels = _kmeans(matrix, k)
        for item, tag in zip(items, labels):
            item.cluster = f"cluster_{int(tag) + 1}"
        report[key] = {
            "n": len(items),
            "k": k,
            "sizes": dict(Counter(f"cluster_{int(v) + 1}" for v in labels)),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-k", type=int, default=4)
    args = parser.parse_args(argv)

    out_root = (PROJECT_ROOT / args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    features = survey(out_root, SOURCES)
    if not features:
        raise SystemExit("không đọc được box nào; kiểm tra datasets/public/")

    report = cluster(features, max_k=args.max_k)

    columns = list(BoxFeature.__dataclass_fields__)
    with (out_root / "features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in features:
            writer.writerow(asdict(item))

    sheets = contact_sheets(
        out_root, features, "contact_sheets", lambda i: f"{i.source}__{i.label}"
    )
    cluster_sheets = contact_sheets(
        out_root,
        [f for f in features if f.cluster],
        "cluster_sheets",
        lambda i: f"{i.source}__{i.label}__{i.cluster}",
    )

    summary = {
        "sources": {
            name: {**spec, "boxes": sum(1 for f in features if f.source == name)}
            for name, spec in SOURCES.items()
        },
        "total_boxes": len(features),
        "boxes_with_shape": sum(1 for f in features if f.circularity is not None),
        "min_side_for_shape": MIN_SIDE_FOR_SHAPE,
        "max_crops_written_per_class": MAX_CROPS_PER_CLASS,
        "contact_sheets": sheets,
        "cluster_sheets": cluster_sheets,
        "clusters": {f"{s}/{c}": v for (s, c), v in report.items()},
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n{len(features)} box, {summary['boxes_with_shape']} có đặc trưng hình học")
    print(f"{sheets} contact sheet lớp, {cluster_sheets} contact sheet cụm")
    print(f"ghi -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
