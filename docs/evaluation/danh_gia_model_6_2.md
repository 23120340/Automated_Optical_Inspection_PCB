# Model chấm mối hàn 6.2 có dùng được không?

> Đo 2026-08-22 trên `models/active/solder_classifier/best.onnx`
> (lúc đo thư mục còn tên `solder/classifier`; đổi phẳng 2026-09-03)
> (`solder-joint-20260820T180939Z`, mobilenet_v3_small, 7 lớp).
>
> Câu hỏi không phải "nó có nạp được không" — nó nạp được. Câu hỏi là đầu ra
> của nó có đủ tín hiệu để ra quyết định không.

## Kết luận

**Chưa dùng để ra quyết định được.** Giữ nguyên ở vai trò hiện tại: một lớp ý
kiến bị lớp vật lý phủ quyết, và mọi bất đồng đi vào hàng chờ xem tay.

**Không phải vì nó hỏng** — nó có tín hiệu thật, kiểm định thống kê xác nhận.
Nhưng tín hiệu quá yếu so với thứ nó phải quyết, và nó có một **thiên lệch
`bridge`** bắn vào gần như mọi ảnh có kết cấu.

## Bằng chứng 1 — chính tập val của nó đã nói

Đọc thẳng `threshold_sweep` trong manifest, tại đúng ngưỡng đang chạy (0.85):

| Chỉ số | Giá trị |
|---|---|
| escape (lỗi lọt lưới) | 0.0098 |
| false_call (báo động nhầm) | **0.2276** |
| **review_rate** | **0.4569** |

**46% mối hàn phải xem tay** — trên chính tập val của nó, tức trong điều kiện
thuận lợi nhất có thể. Một dây chuyền phải mở tay gần một nửa số mối hàn thì
mức tự động hoá gần bằng không.

Không có ngưỡng nào cứu được. Nhìn cả bảng sweep: hạ xuống 0.5 thì review còn
3.8% nhưng escape tăng gấp đôi (0.0196); lên 0.9 thì escape còn 0.0049 nhưng
false_call vọt lên **38.6%**. Đây là hình dạng của một model **chưa tách được
lớp**, không phải một model chỉ cần chỉnh ngưỡng.

## Bằng chứng 2 — mất cân bằng và nguồn đơn lẻ

Cũng từ manifest:

- Mất cân bằng lớp **10,6×**: `insufficient` 396 mẫu, `shift_component` 4192.
- `single_source_classes` = **`cold`, `insufficient`, `shift_component`** —
  mỗi lớp học từ **một** dataset duy nhất, nên model học được đặc trưng của
  *nguồn ảnh* chứ chưa chắc của *khuyết tật*.
- Một trong ba nguồn train là **`soldef_ai`**, mà SolDef_AI đã đo được là chụp
  macro **1–3 µm/px**, trong khi board dự án là **46 µm/px** — chênh khoảng 20
  lần. Xem `docs/surveys/dataset_lead_detection.md`.

## Bằng chứng 3 — chạy thật trên mối hàn của dự án

Board thật 1832×2560 → 36 linh kiện → 119 ROI mối hàn (hình học bước 5.5):

| Độ tin cậy cao nhất mỗi ROI | |
|---|---|
| p10 | 0.290 |
| p50 | **0.547** |
| p90 | 0.833 |
| vượt ngưỡng accept 0.85 | **11/119 (9.2%)** |

Nhãn nó gán:

| Lớp | Số | Tỉ lệ |
|---|---|---|
| **`bridge`** | 73 | **61.3%** |
| `shift_component` | 17 | 14.3% |
| `good` | 11 | 9.2% |
| còn lại | 18 | 15.2% |

**61% mối hàn bị gọi là chập, 9% được gọi là tốt.** Một board mà 61% mối hàn
chập thì đã là phế phẩm trước khi vào máy. Đây không phải một cách đọc hợp lý.

## Bằng chứng 4 — thiên lệch `bridge`, không phải lỗi của ROI

Có hai cách giải thích con số 61%, và chúng cần cách sửa khác nhau:

- **(a)** ROI của bước 5.5 sai — chúng lấn sang linh kiện bên cạnh, nên ROI
  *thật sự* chứa hai mối hàn và `bridge` là cách đọc hợp lý. Sửa được bằng
  lượt 2.
- **(b)** Model có thiên lệch và nói `bridge` với gần như mọi thứ. Lượt 2 không
  cứu được.

Cho model ăn thứ **không chứa mối hàn nào**:

| Đầu vào | Nhãn hay gặp nhất | Tỉ lệ |
|---|---|---|
| nhiễu ngẫu nhiên | `bridge` | **70.0%** |
| mảnh board ngẫu nhiên (không theo ROI) | `bridge` | **68.3%** |
| **ROI mối hàn thật** | `bridge` | **61.3%** |
| đen tuyền / trắng tuyền / xám phẳng | `excess` | 100% |

So sánh chặt hơn — 119 ROI thật với 357 mảnh board ngẫu nhiên **cùng phân bố
kích thước**:

```
chi-square = 32.91, dof = 6, ngưỡng 0.05 = 12.59  =>  hai phân bố CÓ khác nhau
Mann-Whitney z = +2.71                             =>  ROI thật tự tin hơn chút
phần chồng lấn hai phân bố = 80.4%
```

Đọc cho đúng: model **không** hoàn toàn suy biến — khác biệt là có thật và có ý
nghĩa thống kê. Nhưng **80,4% phân bố đầu ra của nó giống hệt nhau** dù bạn đưa
một mối hàn thật hay một mảnh board bất kỳ, và lớp trội là `bridge` ở cả hai
trường hợp với tỉ lệ gần bằng nhau. Tín hiệu thật, nhưng quá yếu để dựa vào.

Giả thuyết (b) là chính. Lượt 2 sẽ giúp, nhưng không đủ.

### Đo lại trên mẫu lớn hơn — 2026-08-23

Mọi con số ở trên đo trên **một** ảnh (119 ROI). Lần đo lại dùng **5 ảnh board,
664 ROI**, cùng giao thức, qua `scripts/benchmark_models.py`:

| | 1 ảnh (119 ROI) | **5 ảnh (664 ROI)** |
|---|---|---|
| `bridge` trên ROI thật | 61,3% | **50,2%** |
| `bridge` trên mảnh ngẫu nhiên | 68,3% | **50,3%** |
| chồng lấn hai phân bố | 0,804 | **0,816** |
| vượt ngưỡng chấp nhận | 9,2% | **4,4%** |
| confidence trung vị | 0,547 | **0,470** |

Kết luận **không đổi, và mạnh hơn**: trên mẫu lớn, `bridge` chiếm 50,2% ROI thật
so với 50,3% mảnh board ngẫu nhiên — với lớp chiếm một nửa đầu ra, model không
phân biệt được hai thứ đó. Tỉ lệ vượt ngưỡng chấp nhận còn tụt xuống 4,4%.

Dùng số của lần 5 ảnh khi trích dẫn; số 1 ảnh giữ lại để thấy chúng cùng hướng.

## Hiện tại pipeline đang xử lý thế nào — và đang đúng

`SolderGradingConfig` mặc định đã giữ model ở đúng chỗ nó xứng đáng:

| Chốt | Giá trị | Tác dụng |
|---|---|---|
| `model_accept_probability` | 0.80 | Verdict dưới mức này không bao giờ được chấp nhận một mình |
| `escape_guard_enabled` | `True` | Sàn vật lý: model tin là tốt đến mấy thì vẫn phải có đủ thiếc |
| `disagreement_is_review` | `True` | Hai lớp bất đồng thì vào hàng chờ, không chọn bên |

Với 9.2% ROI vượt 0.85 và thiên lệch `bridge`, hệ quả thực tế là **gần như mọi
mối hàn đi vào hàng chờ xem tay**. Đó là hành vi **an toàn nhưng không có giá
trị tự động hoá** — và quan trọng là nó **không đẩy board tốt thành phế phẩm**,
vì `disagreement_is_review` chặn đúng chỗ.

`tests/inspection/test_solder_model_assessment.py` khoá ba chốt này lại. Muốn
làm model thành thẩm quyền thì phải sửa test, và test sẽ hỏi bằng chứng.

## Hướng khắc phục, theo thứ tự đáng làm

### 1. Không train lại bằng chính ba nguồn cũ

Đây là điều quan trọng nhất. Vấn đề không nằm ở kiến trúc, số epoch hay ngưỡng
— nằm ở **dữ liệu sai tỉ lệ chụp**. Đổi mobilenet sang resnet, train 100 epoch
thay vì 30, quét lại ngưỡng: cả ba đều không chạm vào nguyên nhân.

### 2. Gán nhãn mối hàn trên board của chính dây chuyền

Cùng một kết luận với lượt 2, và **nên làm chung một lượt gán nhãn**: khi đã
khoanh box từng chân cho lượt 2 rồi thì gán thêm nhãn tốt/xấu cho chính box đó
rẻ hơn nhiều so với làm hai đợt.

Mốc tối thiểu: **10–20 board khác lô**, và với mỗi lớp khuyết tật cần ít nhất
vài chục ví dụ **chụp bằng chính camera của dây chuyền**.

### 3. Thu hẹp số lớp trước khi mở rộng

7 lớp là quá tham vọng cho lượng dữ liệu hiện có. `cold` chỉ có 1/119 lần được
gọi, và đã đo trước đây rằng **nén JPEG một mình đã đổi `specular_ratio` 0.0381
so với ngưỡng `cold_specular_ratio` 0.010** — tức `cold` không thể phát hiện
qua bất kỳ luồng IP camera nào. Bắt đầu bằng **`good` / `not_good`** nhị phân,
mở rộng khi mỗi lớp đủ vài trăm mẫu thật.

### 4. Trong lúc chờ: lớp luật vật lý vẫn là thứ đáng tin nhất

Bước 6.2 chạy được **không cần model nào**. Các ngưỡng vật lý
(`missing_solder_ratio`, `insufficient_solder_ratio`, …) đo trực tiếp lượng
thiếc trong ROI, không phụ thuộc dataset ngoài. Đó là lớp nên hiệu chỉnh trước
— dùng `scripts/calibrate_solder_thresholds.py` trên board thật đã biết kết quả.

### 5. Nếu vẫn muốn dùng model này

Được, nhưng chỉ như **bộ lọc một chiều**: chỉ tin khi nó nói `good` với xác
suất ≥ 0.85 **và** lớp luật cũng nói tốt. 11/119 ROI thoả — tức nó tiết kiệm
được khoảng 9% công xem tay, không hơn. Đừng bao giờ để nó tự mình gọi một mối
hàn là lỗi: với 61% `bridge`, làm vậy là biến mọi board thành phế phẩm.
