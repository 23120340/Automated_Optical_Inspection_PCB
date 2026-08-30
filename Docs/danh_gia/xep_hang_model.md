# Xếp hạng model — bước này nên bật cái nào

> **Đo lại toàn bộ 2026-08-23.** Mọi con số dưới đây đến từ **một lần chạy duy
> nhất**, cùng 5 ảnh board, cùng một tiến trình, cùng một máy:
>
> ```bash
> python scripts/benchmark_models.py <thư-mục-kết-quả> <ảnh1> <ảnh2> ...
> ```
>
> Kết quả thô của lần đo này được commit kèm: **`Docs/bench/bench_20260823.json`**.
> Mọi con số trong tài liệu phải truy được về file đó, và chỉ về nó.
>
> Bản trước của tài liệu này ghép số từ nhiều lần chạy khác nhau — thời gian
> detector lấy cả "1.79 s" của hôm trước lẫn "2.26 s" của hôm sau. Thời gian đo
> trên máy có tải khác nhau thì **không so được**, và một bảng trộn như thế mời
> người đọc so nhầm. Bản này bỏ hết số cũ và đo lại từ đầu.

**Giao thức:** ô 1024² cắt giữa mỗi ảnh · `conf = 0.25` · 5 ảnh board thật ·
CPU. Cùng ảnh cho mọi model, cùng crop cho mọi classifier.

## Tóm tắt

| Bước | Đang bật | Kết luận |
|---|---|---|
| 4 · detect | `detector` (20260820) | **Kém nhất trong ba bản** — tìm ít nhất, chậm nhất, và **mù hoàn toàn trên 2/5 ảnh** |
| 6.1 · phân loại | `classifier` (efficientnet_b0) | Cả hai bản đều **yếu trên crop thật**; convnext nhỉnh hơn nhưng chậm ~33× |
| 6.2 · mối hàn | `solder` | **Không phân biệt được mối hàn với mảnh board bất kỳ** |

---

> **Cập nhật 2026-08-24 — khuyến nghị ở mục 1 đã được thực hiện.**
> `models/active/detector/` giờ là bản `detector-yolo26s-kaggle-ver1`
> (2026-08-17, mAP50 0.579). Bản cũ chuyển sang
> `models/archive/detector-yolo26s-kaggle-ver2/`.
>
> Tên thư mục đã đổi theo cấu trúc `<bước>-<kiến trúc>[-<nguồn>]-ver<N>`; tài
> liệu này dùng tên mới, còn `Docs/bench/bench_20260823.json` giữ nguyên tên
> lúc chạy vì nó là **bản ghi của một lần đo**, sửa vào đó là làm sai bản ghi.
> Đối chiếu: `…-20260817` → `…-kaggle-ver1`, `…-20260820` → `…-kaggle-ver2`,
> `…-huggingface-20260704` → `…-huggingface-ver1`.
>
> **Một chỗ cần đọc kèm.** Bảng dưới đo trên 5 ảnh của bộ benchmark. Đo riêng
> trên `pcb03.jpg` — board thật của bạn — thì hai bản gần như trùng nhau:
> ghép cặp được **67/67 box, cùng nhãn cả 67**. Bản mới bật (ver1) bỏ sót đúng
> một linh kiện thật (R20, `resistor` 0.27) nhưng **nhanh gấp đôi** (9,2 s so
> với 17,5 s). Nói cách khác chỗ "mù trên 2/5 ảnh" là thật, nhưng nó không xuất
> hiện trên board này — một board không đại diện cho cả năm.

## 1. Bước 4 — Detect linh kiện

| Hạng | Model | Tổng box (5 ảnh) | mAP50 | s/ảnh | conf trung vị |
|---|---|---|---|---|---|
| 1 | `detector-yolov8-huggingface-ver1` | **213** | — | **0.17** | **0.618** |
| 2 | `detector-yolo26s-kaggle-ver1` | 203 | **0.579** | 1.00 | 0.367 |
| 3 | `detector` *(đang bật)* | 121 | 0.505 | 2.45 | 0.427 |

Số box từng ảnh — đây là chỗ chênh lệch lộ rõ nhất:

| Ảnh | HF | v0817 | đang bật |
|---|---|---|---|
| golden.png | 61 | **64** | 36 |
| sample.jpg | 65 | **67** | 49 |
| golden-image.jpg | 61 | **64** | 36 |
| 9e6aa662….jpg | **20** | 8 | **0** |
| ae9d7157….jpg | **6** | 0 | **0** |

**Bản đang bật không tìm được gì trên 2 trong 5 ảnh.** Đó là điểm nặng nhất, và
nó không phải chuyện ngưỡng — hai bản kia vẫn tìm ra 20 và 8 box trên cùng ảnh.

Bản đang bật cũng báo **12 `display`** trong khi hai bản kia báo 2 và 0. Board
này không có màn hình nào; đó là ảo giác trên tụ điện.

### "Nhiều box hơn" có phải "tốt hơn" không — chưa chắc

Không có ground truth nên số box **không tự nó** chứng minh gì. Bảng đồng thuận
nói thêm được một chút:

| Cặp | Trùng box | Cùng nhãn | Chỉ A | Chỉ B |
|---|---|---|---|---|
| đang bật ↔ v0817 | 87 | **81/87 (93%)** | 34 | 116 |
| v0817 ↔ HF | 110 | 70/110 (64%) | 93 | 103 |
| đang bật ↔ HF | 70 | 36/70 (51%) | 51 | 143 |

Hai bản **của dự án** đồng ý 93% về nhãn của những gì cùng thấy — nên chúng
không bất đồng về cách đọc, chỉ khác nhau ở chỗ bản đang bật **không thấy**.

Bản HuggingFace thì khác: nó chỉ cùng nhãn 51–64% với hai bản kia. Nó tìm nhiều
nhất và nhanh nhất, nhưng **không có số đo val nào**, không do dự án train, và
cách gán nhãn của nó lệch rõ so với hai bản còn lại.

### Khuyến nghị

**Bỏ bản đang bật.** Nó thua ở mọi phép đo lấy được và mù trên 2/5 ảnh — điều
này chắc chắn.

Giữa hai bản còn lại: **`detector-yolo26s-kaggle-ver1`** là lựa chọn an toàn — nó
là bản duy nhất có mAP50 đo trên tập val của chính dự án (0.579), và đồng thuận
93% với bản đang bật về nhãn. Bản HuggingFace đáng thử nếu thời gian chu kỳ căng
(nhanh gấp **14 lần**), nhưng phải kiểm nhãn bằng mắt trước vì không ai đo nó.

Không bản nào giúp được lượt 2: cả ba đều cho **0 box** lớp `pads`/`pins`.

## 2. Bước 6.1 — Phân loại linh kiện

Đo trên **cùng 213 crop** cắt từ 5 ảnh.

| | `classifier` *(đang bật)* | `classifier-convnext_base-ver1` |
|---|---|---|
| kiến trúc | efficientnet_b0 | convnext_base |
| **ms mỗi crop** | **9.1** | 305.6 |
| conf trung vị | 0.511 | **0.628** |
| conf phân vị 10 | 0.259 | **0.297** |
| **tự động chấp nhận (≥0.85)** | **9/213 (4.2%)** | **26/213 (12.2%)** |
| đồng ý với detector | 113/213 (53.1%) | 112/213 (52.6%) |
| dung lượng | **16 MB** | 350 MB |
| số lớp | 16 | 17 |
| accuracy trên tập val của nó | 0.958 | 0.954 |

**Con số quan trọng nhất và cũng khó chịu nhất: cả hai bản chỉ tự động chấp
nhận 4–12% số crop.** Trên tập val của chúng, cả hai đều báo accuracy ~0.95.
Trên crop thật của board thì gần như mọi crop rơi vào hàng chờ xem tay.

Hai lý do có thể, và tôi **chưa tách được**:
- crop được cắt bằng detector HuggingFace (bản tìm nhiều box nhất), nên trong đó
  có thể có nhiều box rác mà classifier không phân loại nổi — đúng ra thì nó
  *nên* không tự tin
- hoặc chính hai classifier yếu trên phân bố ảnh này

Tỉ lệ đồng ý với detector (53%) cũng thấp cho cả hai — nhưng detector đó dùng
taxonomy khác, nên con số này khó đọc.

**Cái so được là tương đối, và nó rõ ràng:** convnext tốt hơn ở mọi chỉ số chất
lượng, đổi lại **chậm ~33 lần** và nặng gấp 22 lần.

> Lưu ý về thời gian: đo hai lần cho 9.9/248.7 ms và 9.1/305.6 ms. Tỉ lệ giữa
> hai model ổn định trong khoảng **25–34 lần**; con số tuyệt đối phụ thuộc tải
> máy, đừng trích dẫn nó như một hằng số.

**Khuyến nghị:** giữ bản đang bật cho tới khi giải thích được tỉ lệ chấp nhận
4–12%. Đổi sang convnext chỉ giải quyết được một phần (4.2% → 12.2%) mà trả giá
33× thời gian — không đáng khi cả hai đều còn xa mức dùng được.

## 3. Bước 6.2 — Chấm lỗi mối hàn

664 ROI mối hàn từ 5 ảnh, so với **664 mảnh board ngẫu nhiên cùng phân bố kích
thước** (không phải ROI):

| Nhãn | ROI mối hàn thật | Mảnh board ngẫu nhiên |
|---|---|---|
| `bridge` | **333 (50.2%)** | **334 (50.3%)** |
| `shift_component` | 144 (21.7%) | 46 (6.9%) |
| `excess` | 49 | 122 |
| `insufficient` | 48 | 24 |
| `good` | 43 (6.5%) | 48 |
| `missing_solder` | 36 | 49 |
| `cold` | 11 (1.7%) | 41 |

**Model gọi `bridge` cho 50.2% mối hàn thật và 50.3% mảnh board bất kỳ.** Với
lớp chiếm một nửa đầu ra của nó, model không phân biệt được hai thứ đó.

| | |
|---|---|
| chồng lấn hai phân bố | **0.816** |
| vượt ngưỡng chấp nhận | **29/664 (4.4%)** |
| confidence trung vị | 0.470 |
| review_rate trên tập val của nó | **0.457** |
| mất cân bằng lớp lúc train | 10,6× |
| lớp học từ MỘT nguồn | `cold`, `insufficient`, `shift_component` |

**Khuyến nghị: giữ lớp luật vật lý, đừng để model quyết định.** Ba chốt mặc định
(`model_accept_probability` 0.80, `escape_guard_enabled`,
`disagreement_is_review`) đang giữ nó đúng chỗ.

Phân tích nguyên nhân đầy đủ: `Docs/danh_gia/danh_gia_model_6_2.md`.

## 4. Model ngoài, chưa tích hợp

| Model / nguồn | Trạng thái | Lý do |
|---|---|---|
| keremberke `yolov8m` | **Chưa kết luận** | Chịu được thang chụp 46 µm/px (recall 0.544 so với 0.595 ở gốc) và đặt 36/36 box đúng vùng lỗi — nhưng mới thử trên ảnh của chính họ, chưa có board lỗi thật của dây chuyền |
| keremberke `yolov8n` | Chưa kết luận | Như trên, yếu hơn (25/31) |
| SolDef_AI | Đã loại | Ở 46 µm/px chỉ ra 6 box, toàn `spike` |
| Ulger solder-joint | Đã loại | Đúng tỉ lệ nhưng **không có box**, không có ảnh board gốc |
| PCB-SAID | Đã loại | Ảnh cào web 640×480, nhãn theo linh kiện, không có link tải |
| Dukeb DETR | Đã loại | 24 lớp tên `LABEL_0`…`LABEL_23`, **không có tên lớp** |
| Roboflow Universe | **Chưa kiểm chứng được** | Có nhãn `Dry_joint`/`Cold Solder` nhưng cần API key |

Chi tiết: `Docs/khao_sat/khao_sat_model_huggingface.md`.

---

## 5. Cách đọc bảng này

**Số nào đo được, số nào lấy từ manifest.** Các cột "tổng box", "s/ảnh", "ms mỗi
crop", "tự động chấp nhận", "chồng lấn" là **đo trong lần chạy này**, cùng ảnh
cho mọi model. Các cột `mAP50` và `accuracy` lấy từ manifest, tức đo trên tập
val của **lần train đó** — hai model có thể dùng hai tập val khác nhau, nên
chúng chỉ là tham khảo.

**Cái gì không so được.** mAP50 của detector và accuracy của classifier là hai
đại lượng khác nhau. `macro_f1` và `macro_recall` cũng vậy. Và hai classifier
dùng **taxonomy khác nhau** (16 so với 17 lớp), nên chúng không giải cùng một
bài toán.

**Không có ground truth.** 5 ảnh này không có nhãn người, nên "tìm được nhiều
box hơn" **không** tự chứng minh là tốt hơn — nó có thể là báo nhầm nhiều hơn.
Chỗ duy nhất kết luận chắc được là khi một model tìm **0** trong khi hai model
khác tìm ra hàng chục trên cùng ảnh.

**Muốn kiểm lại:** chạy `scripts/benchmark_models.py` trên board của bạn. Mọi
con số ở đây phải khớp, hoặc bảng này sai.

## 6. Cách làm bảng này chính xác dần

Điểm yếu lớn nhất còn lại là **không có nhãn người**. Trong app, mỗi bước có mục
**"Đánh giá model"** ở cuối trang: người vận hành đánh dấu chỗ model sai và ghi
chú, bản ghi giữ toạ độ và gắn với **sha256 của chính file trọng số**.

Khi đã có vài chục bản ghi, chúng trả lời được đúng câu mà bảng này không trả
lời được: model nào sai ở *loại linh kiện nào*, trong *điều kiện nào*.

```python
from aoi_pipeline.modelops.model_feedback import load_feedback, group_by_model
entries, _ = load_feedback()
for key, group in group_by_model(entries).items():
    print(key[:12], len(group))
```

Xem `feedback/README.md`.
