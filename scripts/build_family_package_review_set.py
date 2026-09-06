"""Dựng tập kiểm phân tầng để gán nhãn HỌ + PACKAGE bằng mắt.

Tập này phục vụ hai phép đo cùng lúc (xem
``docs/evaluation/danh_gia_classifier_6_1.md`` §6):

* tỉ lệ trúng của bộ luật package;
* **độ chính xác thật của 6.1 trên miền ảnh dự án** — hiện chưa ai biết, vì
  9.486 box đã khoanh chỉ mang một lớp ``component``.

Vì mục đích thứ hai, **không được điền sẵn nhãn bằng chính 6.1**: lấy dự đoán
của model làm nhãn thì phép đo tự xác nhận chính nó, và người duyệt bị neo theo
nhãn có sẵn. Script này chỉ cắt ảnh và xếp lưới; nhãn do người nhìn mà gán.

Crop cắt **rộng hơn box thân** rất nhiều. Quy ước khoanh của dự án là "chỉ thân,
loại chân/pad", mà lớp package lại được định nghĩa bằng *vị trí chân* — cắt sát
thân thì chính thứ cần nhìn nằm ngoài khung.

    python scripts/build_family_package_review_set.py --out <thư mục>
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOXES = (PROJECT_ROOT / "datasets/labelling/component_bodies_round2_20260830"
         / "joint_boxes_cleaned.json")
TILES = PROJECT_ROOT / "datasets/test_images/tiles_1024"

#: Lề quanh thân, theo tỉ lệ cạnh DÀI, nhưng không bao giờ dưới ``MIN_PAD_PX``.
#: Chip 0402 chỉ 12 px nên lề theo tỉ lệ thuần sẽ ra vài pixel và không thấy
#: được pad hai đầu — mà pad chính là thứ phân biệt gói.
#: Lề phải CHẶN TRÊN. Lề thuần theo tỉ lệ làm crop của một box 600 px phình
#: lên 2,8 lần, và linh kiện chỉ còn chiếm một góc — nhìn tờ lưới đầu tiên là
#: thấy ngay. Chân linh kiện thò ra một quãng gần như CỐ ĐỊNH chứ không tỉ lệ
#: với thân, nên trần tuyệt đối mới đúng bản chất.
PAD_RATIO = 0.45
MIN_PAD_PX = 14
MAX_PAD_PX = 70

#: Khung thứ HAI, rộng hơn nhiều. Lề ở trên được chỉnh để thấy **pad hai đầu**,
#: và chính vì nó đúng cho câu hỏi *gói* mà nó sai cho câu hỏi *họ*: dấu hiệu
#: mạnh nhất để biết một chip là điện trở hay tụ là **silkscreen designator**
#: (``R902``, ``C450``, ``FB19``, ``L501``) in ngay cạnh nó trên bo — mà chữ đó
#: nằm cách thân 20-60 px, tức LUÔN ngoài khung chặt.
#:
#: Đo được: trên 140 box mà khung chặt không kết luận nổi họ, cắt lại bằng khung
#: rộng giải thêm **57 ca (41%)** — trong đó 22 ca đọc thẳng được designator sát
#: box, ở cỡ nhỏ nhất là **9 px**. Đọc được vì *chữ silkscreen to hơn linh kiện*.
#:
#: Không nâng ``PAD_RATIO`` để làm việc này: nâng lên thì linh kiện lớn chìm
#: trong crop khổng lồ, đúng cái bẫy ghi ở ``MAX_PAD_PX``. Hai câu hỏi khác nhau
#: cần hai khung khác nhau, không phải một con số dung hoà.
WIDE_HALF_MIN_PX = 95
WIDE_HALF_RATIO = 1.5
#: Hai chế độ lưới. Box lớn cần ô to mới thấy được chân; box nhỏ thì ô to chỉ
#: phóng to nhiễu, nên xếp dày hơn để đỡ số tờ phải xem.
GRID_LARGE, CELL_LARGE = 4, 224
GRID_SMALL, CELL_SMALL = 6, 150


def strata(long_side: float, aspect: float) -> str:
    """Bốn tầng của kế hoạch §7.2, dịch sang ngưỡng đo được từ chính dữ liệu."""

    if long_side >= 60 and aspect >= 2.5:
        return "dai_nhieu_chan"      # ứng viên connector
    if long_side >= 60:
        return "lon_vuong"           # ứng viên ic_hai_ben / bon_ben / khong_chan
    if long_side < 60:
        return "nho"                 # ứng viên hai_chan / tru_dung / goi_nho
    return "khac"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-lon-vuong", type=int, default=250)
    parser.add_argument("--n-dai", type=int, default=150)
    parser.add_argument("--n-nho", type=int, default=250)
    parser.add_argument("--n-ngau-nhien", type=int, default=100)
    args = parser.parse_args(argv)

    out = (PROJECT_ROOT / args.out).resolve()
    if out.exists():
        raise SystemExit(f"đã có {out}; không ghi đè")

    payload = json.loads(BOXES.read_text(encoding="utf-8"))
    pool: list[dict] = []
    for tile_name, record in payload["crops"].items():
        if record.get("status") != "verified":
            continue
        for index, box in enumerate(record.get("boxes", [])):
            w, h = float(box["w"]), float(box["h"])
            if w <= 0 or h <= 0:
                continue
            long_side, short_side = max(w, h), min(w, h)
            pool.append({
                "tile": tile_name,
                "box_index": index,
                "board": tile_name.split("__")[0],
                "x": float(box["x"]), "y": float(box["y"]), "w": w, "h": h,
                "long_side": long_side,
                "aspect": round(long_side / max(1.0, short_side), 3),
                "stratum": strata(long_side, long_side / max(1.0, short_side)),
            })

    rng = random.Random(args.seed)
    wanted = {"lon_vuong": args.n_lon_vuong, "dai_nhieu_chan": args.n_dai,
              "nho": args.n_nho}
    chosen: list[dict] = []
    taken: set[tuple[str, int]] = set()
    for name, count in wanted.items():
        group = [item for item in pool if item["stratum"] == name]
        # Trải đều theo BO trước rồi mới bốc, để một bo dày linh kiện không
        # chiếm hết một tầng.
        by_board: dict[str, list[dict]] = {}
        for item in group:
            by_board.setdefault(item["board"], []).append(item)
        for items in by_board.values():
            rng.shuffle(items)
        picked: list[dict] = []
        boards = sorted(by_board)
        cursor = 0
        while len(picked) < min(count, len(group)):
            board = boards[cursor % len(boards)]
            cursor += 1
            if by_board[board]:
                picked.append(by_board[board].pop())
        for item in picked:
            item["stratum_reason"] = name
            chosen.append(item)
            taken.add((item["tile"], item["box_index"]))

    rest = [i for i in pool if (i["tile"], i["box_index"]) not in taken]
    rng.shuffle(rest)
    for item in rest[:args.n_ngau_nhien]:
        item["stratum_reason"] = "ngau_nhien"
        chosen.append(item)

    rng.shuffle(chosen)
    for order, item in enumerate(chosen):
        item["id"] = order

    (out / "crops").mkdir(parents=True, exist_ok=True)
    (out / "crops_wide").mkdir(parents=True, exist_ok=True)
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    (out / "sheets_wide").mkdir(parents=True, exist_ok=True)

    cache: dict[str, np.ndarray] = {}
    for item in chosen:
        tile = cache.get(item["tile"])
        if tile is None:
            tile = cv2.imread(str(TILES / item["tile"]))
            if tile is None:
                raise SystemExit(f"không đọc được tile {item['tile']}")
            cache[item["tile"]] = tile
        height, width = tile.shape[:2]
        pad = min(MAX_PAD_PX, max(MIN_PAD_PX, PAD_RATIO * item["long_side"]))
        x1 = int(max(0, item["x"] - pad))
        y1 = int(max(0, item["y"] - pad))
        x2 = int(min(width, item["x"] + item["w"] + pad))
        y2 = int(min(height, item["y"] + item["h"] + pad))
        patch = tile[y1:y2, x1:x2].copy()
        # Vẽ khung thân lên chính crop: không có nó thì ở vùng dày linh kiện
        # không biết đang phải gán nhãn cho cái nào.
        # Nét vẽ phải dày theo crop: nét 1 px biến mất khi tờ lưới thu nhỏ,
        # và ở vùng dày linh kiện thì không biết đang gán nhãn cho cái nào.
        thickness = max(1, round(max(patch.shape[:2]) / 110))
        cv2.rectangle(
            patch,
            (int(item["x"] - x1), int(item["y"] - y1)),
            (int(item["x"] + item["w"] - x1), int(item["y"] + item["h"] - y1)),
            (60, 220, 60), thickness,
        )
        cv2.imwrite(str(out / "crops" / f"{item['id']:04d}.png"), patch)
        item["crop"] = f"crops/{item['id']:04d}.png"

        # Khung RỘNG: để trả lời câu HỌ, không phải câu gói. Xem WIDE_HALF_*.
        half = max(WIDE_HALF_MIN_PX, WIDE_HALF_RATIO * item["long_side"])
        cx, cy = item["x"] + item["w"] / 2, item["y"] + item["h"] / 2
        wx1, wy1 = int(max(0, cx - half)), int(max(0, cy - half))
        wx2, wy2 = int(min(width, cx + half)), int(min(height, cy + half))
        wide = tile[wy1:wy2, wx1:wx2].copy()
        # Khung ĐỎ chứ không xanh lá như crop chặt: ở khung rộng phần lớn ảnh là
        # mặt bo màu xanh lá, nét xanh chìm mất. Đổi màu cũng nhắc người duyệt
        # rằng đang nhìn khung khác.
        cv2.rectangle(
            wide,
            (int(item["x"]) - wx1, int(item["y"]) - wy1),
            (int(item["x"] + item["w"]) - wx1, int(item["y"] + item["h"]) - wy1),
            (60, 60, 235), 1,
        )
        cv2.imwrite(str(out / "crops_wide" / f"{item['id']:04d}.png"), wide)
        item["crop_wide"] = f"crops_wide/{item['id']:04d}.png"

    # Lưới: xếp theo TẦNG rồi tới cỡ, để mỗi tờ là một loại bài toán.
    ordered = sorted(chosen, key=lambda i: (i["stratum_reason"], -i["long_side"]))
    sheets = 0
    start = 0
    while start < len(ordered):
        large = ordered[start]["long_side"] >= 100
        grid, cell = ((GRID_LARGE, CELL_LARGE) if large else (GRID_SMALL, CELL_SMALL))
        per_sheet = grid * grid
        block = ordered[start:start + per_sheet]
        # Không trộn hai chế độ trong một tờ: cắt tờ ngay khi cỡ đổi hạng.
        for cut, item in enumerate(block):
            if (item["long_side"] >= 100) != large:
                block = block[:cut]
                break
        if not block:
            block = ordered[start:start + 1]
        start += len(block)
        GRID, CELL = grid, cell
        sheet = np.full((GRID * CELL + 30, GRID * CELL, 3), 28, dtype=np.uint8)
        cv2.putText(sheet, f"{block[0]['stratum_reason']}  id {block[0]['id']}..{block[-1]['id']}",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        for position, item in enumerate(block):
            patch = cv2.imread(str(out / item["crop"]))
            if patch is None:
                continue
            scale = min((CELL - 26) / patch.shape[1], (CELL - 26) / patch.shape[0])
            resized = cv2.resize(
                patch,
                (max(1, int(patch.shape[1] * scale)), max(1, int(patch.shape[0] * scale))),
                interpolation=cv2.INTER_NEAREST if scale > 1 else cv2.INTER_AREA,
            )
            row, col = divmod(position, GRID)
            oy = 30 + row * CELL + 18 + (CELL - 18 - resized.shape[0]) // 2
            ox = col * CELL + (CELL - resized.shape[1]) // 2
            sheet[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
            cv2.putText(sheet, f"{item['id']}", (col * CELL + 4, 30 + row * CELL + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(sheet, f"{item['long_side']:.0f}px",
                        (col * CELL + 46, 30 + row * CELL + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170, 170, 170), 1, cv2.LINE_AA)
        cv2.imwrite(
            str(out / "sheets" / f"{sheets:02d}_{block[0]['stratum_reason']}.png"), sheet)
        sheets += 1

    # Tờ lưới cho khung RỘNG. Ô to hơn hẳn và mỗi tờ ít ảnh hơn: thứ phải đọc ở
    # đây là chữ silkscreen quanh linh kiện, không phải hình dạng linh kiện, nên
    # xếp dày là hỏng mục đích.
    WIDE_GRID, WIDE_CELL = 3, 330
    wide_sheets = 0
    for begin in range(0, len(ordered), WIDE_GRID * WIDE_GRID):
        block = ordered[begin:begin + WIDE_GRID * WIDE_GRID]
        sheet = np.full((WIDE_GRID * (WIDE_CELL + 26), WIDE_GRID * WIDE_CELL, 3),
                        250, dtype=np.uint8)
        for position, item in enumerate(block):
            patch = cv2.imread(str(out / item["crop_wide"]))
            if patch is None:
                continue
            scale = min((WIDE_CELL - 8) / patch.shape[1], (WIDE_CELL - 8) / patch.shape[0])
            resized = cv2.resize(
                patch,
                (max(1, int(patch.shape[1] * scale)), max(1, int(patch.shape[0] * scale))),
                interpolation=cv2.INTER_NEAREST if scale > 1 else cv2.INTER_AREA,
            )
            row, col = divmod(position, WIDE_GRID)
            oy = row * (WIDE_CELL + 26) + 26
            ox = col * WIDE_CELL
            sheet[oy + (WIDE_CELL - resized.shape[0]) // 2:
                  oy + (WIDE_CELL - resized.shape[0]) // 2 + resized.shape[0],
                  ox + (WIDE_CELL - resized.shape[1]) // 2:
                  ox + (WIDE_CELL - resized.shape[1]) // 2 + resized.shape[1]] = resized
            cv2.putText(sheet, f"#{item['id']}  {item['long_side']:.0f}px",
                        (ox + 6, oy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / "sheets_wide" / f"{wide_sheets:02d}.png"), sheet)
        wide_sheets += 1

    (out / "sample.json").write_text(
        json.dumps({
            "source": str(BOXES.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "seed": args.seed,
            "pad_ratio": PAD_RATIO,
            "min_pad_px": MIN_PAD_PX,
            "wide_half_min_px": WIDE_HALF_MIN_PX,
            "wide_half_ratio": WIDE_HALF_RATIO,
            "note": "Nhãn KHÔNG được điền sẵn bằng 6.1: tập này dùng để đo 6.1.",
            "note_hai_khung": (
                "crops/ = khung CHẶT, trả lời câu GÓI (thấy pad hai đầu). "
                "crops_wide/ = khung RỘNG, trả lời câu HỌ (thấy silkscreen "
                "designator R/C/D/L cạnh linh kiện). Một con số lề không phục "
                "vụ được cả hai câu."
            ),
            "items": chosen,
        }, ensure_ascii=False, indent=1), encoding="utf-8")

    counts: dict[str, int] = {}
    for item in chosen:
        counts[item["stratum_reason"]] = counts.get(item["stratum_reason"], 0) + 1
    boards = len({item["board"] for item in chosen})
    print(f"{len(chosen)} box, {boards} bo, {sheets} tờ chặt + {wide_sheets} tờ rộng")
    for key, value in sorted(counts.items()):
        print(f"    {key:16s} {value}")
    print(f"ghi -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
