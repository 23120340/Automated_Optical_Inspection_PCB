"""Cổng nghiệm thu cho ĐƯỜNG LUẬT 5.2 — so ROI trước và sau khi bật luật.

Khác hẳn ``evaluate_package_roi_gate.py``: script đó nhận **ma trận nhầm lẫn
của một model**, mà đường luật không sinh ra thứ đó, nên nó không dùng được ở
đây (ghi ở kế hoạch package §10.3).

**Cổng chính là bất đối xứng, và cố ý.** Mất một pad baseline là **fail ngay**,
bất kể bao nhiêu pad khác được cải thiện: pad mất nghĩa là mối hàn đó không ai
kiểm, còn ROI thừa chỉ tốn một cái liếc mắt. Đo trên 364 box gán tay của dự án,
đổi 13 ca lọt lưới lấy 33 ca đỡ phải xem là sai chiều với bài toán kiểm tra.

    python scripts/evaluate_package_rule_gate.py tests/data/solder_geometry
    python scripts/evaluate_package_rule_gate.py <thư mục> --families model
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aoi_pipeline import (  # noqa: E402
    BoundingBox, Detection, SolderJointConfig, derive_solder_joints,
)
from aoi_pipeline.config import (  # noqa: E402
    PackageRulesConfig, PipelineConfig, PreprocessConfig, terminal_geometry,
)
from aoi_pipeline.imaging.preprocessing import ImagePreprocessor  # noqa: E402
from aoi_pipeline.classification.package_rules import _edge_of  # noqa: E402
from aoi_pipeline.models import ClassProbability, ComponentClassification  # noqa: E402
from aoi_pipeline.pipeline import AOIPipeline  # noqa: E402
from aoi_pipeline.solder.geometry import deconflict_joint_rois  # noqa: E402
from aoi_pipeline.solder.leads import assign_leads_to_components  # noqa: E402
from aoi_pipeline.solder.lead_detection import (  # noqa: E402
    detect_leads_in_components,
)

#: Cùng ngưỡng với ``tests/inspection/test_solder_geometry_real_board.py``: một
#: nửa diện tích pad phải nằm trong một ROI thì mới coi là phủ.
MIN_PAD_COVERAGE = 0.50


@dataclass(slots=True)
class BoardResult:
    board: str
    pads_total: int = 0
    covered_before: int = 0
    covered_after: int = 0
    lost: list[str] = field(default_factory=list)
    gained: list[str] = field(default_factory=list)
    roi_before: int = 0
    roi_after: int = 0
    roi_area_before: float = 0.0
    roi_area_after: float = 0.0
    decided: int = 0
    abstained: int = 0
    leads_found: int = 0
    #: Chan duoc gan cho mot than nhung TAM nam TRONG box than. Nhung chan
    #: do khong dong gop duoc canh nao, nen nhanh ``ic`` cua luat khong chay.
    leads_inside_body: int = 0
    #: Vi sao luat bo qua. "Bo qua 15" khong dung duoc; "bo qua 3 con
    #: ic vi chi thay chan o 1 canh" thi dung.
    abstain_reasons: Counter = field(default_factory=Counter)
    by_package: Counter = field(default_factory=Counter)


def _coverage(roi: BoundingBox, pad: list[int]) -> float:
    px1, py1, px2, py2 = pad
    iw = max(0.0, min(roi.x2, px2) - max(roi.x1, px1))
    ih = max(0.0, min(roi.y2, py2) - max(roi.y1, py1))
    area = float((px2 - px1) * (py2 - py1))
    return (iw * ih) / area if area > 0 else 0.0


def _rois(image, detections, config: SolderJointConfig) -> list[BoundingBox]:
    height, width = image.shape[:2]
    joints: list = []
    for detection in detections:
        joints.extend(
            derive_solder_joints(detection, width, height, config=config, image=image)
        )
    return [j.bbox for j in deconflict_joint_rois(joints, detections, config)
            if j.kind == "joint"]


def _families(detections, raw_rows, mode: str, pipeline: AOIPipeline, image):
    """Nguồn nhãn họ cho luật.

    ``truth`` dùng nhãn của chính fixture, tách hẳn lỗi của 6.1 ra khỏi phép
    đo — cổng này đo LUẬT, không đo classifier. ``model`` chạy 6.1 thật, để
    thấy con số đầu-cuối.
    """

    if mode == "model":
        crops = pipeline.make_crops(image, detections)
        return pipeline.classify_components(crops)
    return [
        ComponentClassification(
            crop_id=f"crop_{index:04d}",
            detection_id=detection.detection_id,
            family=row["label"],
            probability=1.0,
            top_k=[ClassProbability(row["label"], 1.0)],
            unknown_score=0.0,
            decision="accept",
            model_version="fixture-truth",
        )
        for index, (detection, row) in enumerate(zip(detections, raw_rows))
    ]


def evaluate_board(truth_path: Path, families_mode: str,
                   lead_model: Path | None) -> BoardResult:
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    raw = cv2.imread(str(truth_path.parent / truth["image"]))
    if raw is None:
        raise SystemExit(f"không đọc được {truth['image']}")
    image = ImagePreprocessor(PreprocessConfig()).process(raw).image

    rows = truth["detections"]
    detections = [
        Detection(row["label"], row["confidence"], BoundingBox(*row["box"]))
        for row in rows
    ]
    result = BoardResult(board=truth_path.stem)

    solder = SolderJointConfig()
    before = _rois(image, detections, solder)

    config = PipelineConfig(package_rules=PackageRulesConfig(enabled=True))
    if lead_model is not None:
        config.lead_detection = replace(
            config.lead_detection, enabled=True, model_path=str(lead_model)
        )
    pipeline = AOIPipeline(config)
    families = _families(detections, rows, families_mode, pipeline, image)

    # Chan PHAI co that. Khong co chan thi nhanh ``ic`` cua luat -- phan dang
    # kiem nhat -- khong bao gio chay, va cong chi do may anh xa ho tam thuong.
    leads = (
        detect_leads_in_components(
            image, detections, pipeline.lead_detector, config.lead_detection
        )
        if lead_model is not None else []
    )
    result.leads_found = len(leads)
    packages = pipeline._append_package_rules(detections, leads, [], families)

    # Vi sao tung than bi bo qua. Dung chinh cac ham cua bo luat de con so nay
    # khong troi khoi hanh vi that.
    decided_ids = {p.detection_id for p in packages}
    assigned = assign_leads_to_components(detections, leads, config.lead_fusion)
    family_of = {f.detection_id: f.family for f in families}
    for detection in detections:
        if detection.detection_id in decided_ids:
            continue
        own = assigned.get(detection.detection_id, [])
        counts = Counter(
            edge for edge in (_edge_of(lead, detection) for lead in own) if edge
        )
        edges = sum(
            1 for e in ("left", "right", "top", "bottom")
            if counts[e] >= config.package_rules.min_leads_per_edge
        )
        inside = sum(1 for lead in own if _edge_of(lead, detection) is None)
        result.leads_inside_body += inside
        family = family_of.get(detection.detection_id, "?")
        result.abstain_reasons[f"{family} - {edges} canh co dai chan"] += 1
    by_id = {p.detection_id: p for p in packages}
    result.decided = len(by_id)
    result.abstained = len(detections) - len(by_id)
    for item in packages:
        result.by_package[item.package_class] += 1
    after_detections = pipeline.apply_package_classifications(detections, packages)
    after = _rois(image, after_detections, solder)

    result.roi_before, result.roi_after = len(before), len(after)
    result.roi_area_before = sum(r.area for r in before)
    result.roi_area_after = sum(r.area for r in after)

    for name, entry in truth["components"].items():
        for index, pad in enumerate(entry["pads"]):
            result.pads_total += 1
            was = max((_coverage(r, pad) for r in before), default=0.0) >= MIN_PAD_COVERAGE
            now = max((_coverage(r, pad) for r in after), default=0.0) >= MIN_PAD_COVERAGE
            result.covered_before += was
            result.covered_after += now
            if was and not now:
                result.lost.append(f"{name} pad{index}")
            elif now and not was:
                result.gained.append(f"{name} pad{index}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="thư mục chứa các *.json fixture")
    parser.add_argument("--families", choices=("truth", "model"), default="truth")
    parser.add_argument(
        "--lead-model", type=Path,
        default=Path("models/active/lead_detector/best.onnx"),
        help="de trong thi nhanh `ic` cua luat KHONG chay duoc",
    )
    parser.add_argument("--no-leads", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    fixtures = sorted((PROJECT_ROOT / args.root).glob("*.json"))
    if not fixtures:
        raise SystemExit(f"không thấy fixture nào trong {args.root}")

    lead_model = None if args.no_leads else (PROJECT_ROOT / args.lead_model)
    if lead_model is not None and not lead_model.is_file():
        raise SystemExit(f"khong thay lead detector {lead_model}; dung --no-leads")
    results = [evaluate_board(path, args.families, lead_model)
               for path in fixtures]
    print(f"nguồn nhãn họ: {args.families}\n")
    header = (f"{'board':24s} {'pad':>10s} {'mất':>5s} {'thêm':>5s} "
              f"{'ROI':>13s} {'bỏ qua':>8s} {'chân':>5s}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.board[:24]:24s} {r.covered_before:3d}->{r.covered_after:3d}/{r.pads_total:<3d} "
              f"{len(r.lost):5d} {len(r.gained):5d} "
              f"{r.roi_before:5d}->{r.roi_after:<6d} {r.abstained:5d}/{r.decided + r.abstained:<3d}"
              f" {r.leads_found:5d}")

    lost = [(r.board, name) for r in results for name in r.lost]
    packages = Counter()
    for r in results:
        packages.update(r.by_package)
    print()
    print(f"gói luật quyết được: {dict(packages) or 'khong cai nao'}")
    reasons = Counter()
    for r in results:
        reasons.update(r.abstain_reasons)
    if reasons:
        print("luật BỎ QUA, theo lý do:")
        for key, count in reasons.most_common():
            print(f"    {count:3d}  {key}")
    inside = sum(r.leads_inside_body for r in results)
    if inside:
        print()
        print(f"⚠️  {inside} chân được gán cho một thân nhưng TÂM nằm TRONG box")
        print("    thân, nên chúng không đóng góp cạnh nào và nhánh `ic` của luật")
        print("    không chạy tới. Hai nguyên nhân đều có thể, cổng KHÔNG phân")
        print("    biệt được:")
        print("      (a) box theo quy ước CŨ, bao cả chân — khi đó chân thật sự")
        print("          nằm trong box;")
        print("      (b) lượt 2 cắt một cửa sổ QUANH linh kiện rồi tìm mối hàn")
        print("          bên trong đó, nên nó vẫn trả về mối hàn nằm trên/dưới")
        print("          thân — đúng thiết kế, không phải lỗi.")
        print("    Dù nguyên nhân nào, nhánh `ic` vẫn chưa được kiểm ở đây.")

    boards = len(results)
    pads = sum(r.pads_total for r in results)
    print(f"\n{boards} board, {pads} pad đếm tay")
    if boards < 3:
        # Rule of three: 0 lỗi trên n mẫu vẫn để lại cận trên ~3/n. Với một
        # board thì con số đó vô nghĩa, nên cổng phải nói thẳng chứ không để
        # người đọc tưởng "0 mất pad" là bằng chứng mạnh.
        print("  ⚠️  QUÁ ÍT BOARD. Không mất pad nào ở đây KHÔNG chứng minh được")
        print("      luật an toàn — nó chỉ nói luật không hỏng trên chỗ đã thử.")
    if pads:
        print(f"  cận trên xấp xỉ cho tỉ lệ mất pad (rule of three, 95%): "
              f"{300.0 / pads:.2f}% nếu không mất cái nào")

    if lost:
        print(f"\nFAIL — mất {len(lost)} pad baseline:")
        for board, name in lost:
            print(f"    {board}: {name}")
        print("\n  Mất pad là mối hàn không ai kiểm. Không có số cải thiện nào")
        print("  bù lại được điều đó, nên cổng này không cân nhắc đánh đổi.")
    else:
        print("\nPASS — không pad baseline nào bị mất.")

    if args.output:
        payload = {
            "families_mode": args.families,
            "boards": [
                {
                    "board": r.board, "pads_total": r.pads_total,
                    "covered_before": r.covered_before, "covered_after": r.covered_after,
                    "lost": r.lost, "gained": r.gained,
                    "roi_before": r.roi_before, "roi_after": r.roi_after,
                    "roi_area_before": round(r.roi_area_before, 1),
                    "roi_area_after": round(r.roi_area_after, 1),
                    "decided": r.decided, "abstained": r.abstained,
                    "leads_found": r.leads_found,
                    "abstain_reasons": dict(r.abstain_reasons),
                    "leads_inside_body": r.leads_inside_body,
                    "by_package": dict(r.by_package),
                }
                for r in results
            ],
            "passed": not lost,
        }
        out = (PROJECT_ROOT / args.output).resolve()
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nghi -> {out}")
    return 1 if lost else 0


if __name__ == "__main__":
    raise SystemExit(main())
