# Xếp hạng model — bước này nên bật cái nào

> Cập nhật 2026-08-23. Hợp nhất số đo từ `so_sanh_model_v2.md`,
> `danh_gia_model_6_2.md` và `khao_sat_model_huggingface.md`.
>
> **Xem cách đọc ở mục cuối trước khi so hai con số với nhau.** mAP50 của
> detector và accuracy của classifier không cùng thang đo; một bảng tổng gộp
> tất cả sẽ mời người đọc so nhầm, nên ở đây chia theo bước.

## Tóm tắt: bật cái gì ngay bây giờ

| Bước | Đang bật | Nên bật | Vì sao |
|---|---|---|---|
| 4 · detect linh kiện | `detector` (20260820) | **`detector-yolo26s-20260817`** | Bản đang bật tìm được **36** linh kiện, bản cũ tìm **64**, trên cùng một ảnh |
| 6.1 · phân loại | `classifier` (efficientnet_b0) | tuỳ thời gian chu kỳ | convnext tốt hơn về chất lượng nhưng **chậm gấp 26 lần** |
| 6.2 · chấm mối hàn | `solder` (mobilenet_v3_small) | **giữ lớp luật, đừng tin model** | Model chưa dùng để ra quyết định được |

---

## 1. Bước 4 — Detect linh kiện

Đo trên tile 1024² cắt giữa board, lặp trên 4 ảnh khác nhau.

| Hạng | Model | Linh kiện tìm được | mAP50 | Thời gian | Ghi chú |
|---|---|---|---|---|---|
| **1** | `detector-yolo26s-20260817` *(archive)* | **64** | **0.579** | 1.8–2.3 s | Tốt hơn ở mọi phép đo lấy được |
| 2 | `detector-yolov8-huggingface-20260704` *(archive)* | 61 | — | **0.24–0.45 s** | Nhanh gấp ~7 lần, confidence cao nhất — nhưng **không có số đo val nào**, không train bởi dự án |
| 3 | `detector` *(đang bật, 20260820)* | 36 | 0.505 | 2.5–3.3 s | Bỏ sót nhiều nhất, chậm nhất |

Lặp trên nhiều ảnh, cùng một hướng:

| Ảnh | ver 08-17 | ver 08-20 |
|---|---|---|
| golden.png | **64** | 36 |
| sample.jpg | **67** | 49 |
| golden-image.jpg | **64** | 36 |
| 9e6aa662….jpg | **8** | 0 |

Hai bản **không bất đồng về cái chúng cùng thấy**: 27 box trùng nhau, trong đó
25/27 (93%) cùng nhãn. Vấn đề là bản mới **không thấy** — 37 box chỉ bản cũ có,
9 box chỉ bản mới có. Hạ ngưỡng không cứu được (đã quét 0.25/0.35/0.50), và bản
mới còn sinh ảo giác `display` trên tụ điện.

**Khuyến nghị: đưa bản 2026-08-17 về `models/active/detector/`.**

> ⏳ **Đang chờ quyết định.** Tôi đã nêu việc này ba lượt và chưa nhận được trả
> lời, nên chưa tự đổi — đổi model đang chạy là việc của người, không phải của
> agent.

Chưa bản nào giúp được lượt 2: cả ba đều cho **0 box** lớp `pads`/`pins`.

## 2. Bước 6.1 — Phân loại linh kiện

| | `classifier` *(đang bật)* | `classifier-convnext_base-20260822` |
|---|---|---|
| kiến trúc | efficientnet_b0 | convnext_base |
| accuracy (test) | 0.9580 | 0.9539 |
| macro | macro_f1 **0.8903** | macro_recall **0.9326** |
| **ms mỗi crop** | **13.6** | 353.6 |
| **mỗi board (108 linh kiện)** | **1.5 s** | **38.2 s** |
| tự động chấp nhận (≥0.85) | 33/64 (52%) | **42/64 (66%)** |
| đồng ý với detector | 43/64 (67%) | **46/64 (72%)** |
| dung lượng | **16 MB** | 351 MB |
| số lớp | 16 | 17 |

**Đừng đọc "0.9580 > 0.9539" thành "bản cũ tốt hơn".** Hai bản dùng **taxonomy
khác nhau** — chỉ chung 8 trong 16/17 lớp — nên chúng không giải cùng một bài
toán. Và `macro_f1` với `macro_recall` là hai đại lượng khác nhau.

Cái so được thì rõ ràng: ở những ca hai bản bất đồng, bản mới hợp lý hơn hẳn.

| detector nói | bản cũ nói | convnext nói |
|---|---|---|
| `ic` | `relay` 0.46 | **`ic` 0.98** |
| `ic` | `display` 0.51 | **`ic` 0.99** |
| `diode` | `discrete_semiconductor` 0.51 | **`diode` 0.74** |

**Khuyến nghị:** đổi nếu **38 s/board** chấp nhận được. Nếu không, train lại
notebook v2 với `convnext_tiny` — **giữ nguyên taxonomy 17 lớp**, đó mới là
phần giá trị, không phải backbone.

Bản cũ có thêm lớp `false_crop_background` mà bản mới không có. Nghe như mất
mát, nhưng đã đo: lớp đó chỉ bắn vào vùng xám phẳng; với nhiễu ngẫu nhiên nó
gọi `connector` 98%. Nó không phải bộ lọc "không phải linh kiện" đáng tin.

## 3. Bước 6.2 — Chấm lỗi mối hàn

| Model | Trạng thái |
|---|---|
| `solder` *(đang bật, mobilenet_v3_small)* | **Chưa dùng để ra quyết định được** |
| keremberke `yolov8m-pcb-defect-segmentation` | **Chưa kết luận** — cần một board lỗi thật |

Bản đang bật, ba bằng chứng theo thứ tự nặng dần:

| Kiểm bằng gì | Kết quả |
|---|---|
| Tập val của chính nó | **46% mối hàn phải xem tay** ở ngưỡng đang chạy |
| 119 ROI thật | **61.3% bị gọi `bridge`**, chỉ 9.2% gọi `good` |
| Nhiễu ngẫu nhiên | `bridge` **70.0%** — chồng lấn **80.4%** với ROI thật |

Nguyên nhân đọc từ manifest: mất cân bằng lớp 10,6×; ba lớp học từ **một** nguồn
duy nhất; và một nguồn là SolDef_AI, macro 1–3 µm/px so với 46 µm/px của board.

**Lớp luật vật lý hiện là thứ đáng tin nhất ở bước này.** Ba chốt mặc định
(`model_accept_probability` 0.80, `escape_guard_enabled`,
`disagreement_is_review`) đang giữ model đúng chỗ nó xứng đáng.

Chi tiết: `Docs/danh_gia_model_6_2.md`.

## 4. Model ngoài, chưa tích hợp

| Model / nguồn | Trạng thái | Lý do |
|---|---|---|
| keremberke `yolov8m` | **Chưa kết luận** | Đặt **36/36 box đúng vùng lỗi** ở 46 µm/px, nhưng mới thử trên ảnh của chính họ |
| keremberke `yolov8n` | Chưa kết luận | Như trên, yếu hơn (25/31) |
| SolDef_AI | Đã loại | Sai tỉ lệ 20 lần; ở 46 µm/px chỉ ra 6 box, toàn `spike` |
| Ulger solder-joint | Đã loại | Đúng tỉ lệ (~20–25 µm/px) nhưng **không có box**, không có ảnh board gốc |
| PCB-SAID | Đã loại | Ảnh cào web 640×480, nhãn theo linh kiện, không có link tải |
| Dukeb DETR | Đã loại | 24 lớp tên `LABEL_0`…`LABEL_23`, **không có tên lớp** |
| Roboflow Universe | **Chưa kiểm chứng được** | Có nhãn `Dry_joint`/`Cold Solder` nhưng cần API key |

Chi tiết và cách kiểm lại: `Docs/khao_sat_model_huggingface.md`.

---

## 5. Cách đọc bảng này

**Cái gì so được với cái gì.** Trong cùng một bước thì so được. Giữa các bước
thì không: mAP50 đo chất lượng khoanh vùng, accuracy đo chất lượng gán nhãn.

**Số nào từ đâu.** `mAP50` lấy từ manifest, tức đo trên tập val của lần train
đó — hai model có thể dùng hai tập val khác nhau. Các cột "linh kiện tìm được",
"ms mỗi crop" là **tôi đo trên board của dự án**, cùng ảnh cho mọi model, nên so
được thẳng.

**Điểm yếu lớn nhất của bảng này:** mọi số đều đo trên **một board chuẩn**. Nó
nói được model nào bỏ sót nhiều hơn, nhưng không nói được model nào sai ở *loại
linh kiện nào*, hay sai ở *điều kiện chiếu sáng nào* — vì chỉ có một board.

## 6. Cách làm bảng này chính xác dần

Trong app, mỗi bước có một mục **"Đánh giá model"** ở cuối trang. Người vận
hành đánh dấu chỗ model sai và ghi chú; bản ghi giữ **toạ độ**, không giữ ảnh.

Mỗi bản ghi gắn với **sha256 của chính file trọng số**, nên:

- đổi model không làm hỏng đánh giá cũ
- hai model **so được với nhau trên cùng những lỗi đã báo** — đây mới là phép
  so đáng tin, vì nó dùng board thật của dây chuyền chứ không phải một board mẫu

Đọc lại bằng code:

```python
from aoi_pipeline.model_feedback import load_feedback, group_by_model

entries, _ = load_feedback()
for key, group in group_by_model(entries).items():
    print(key[:12], len(group))
```

Xem `feedback/README.md`. Khi đã có vài chục bản ghi, cập nhật lại các bảng ở
trên bằng số từ đó — chúng đáng tin hơn mọi phép đo trong tài liệu này.
