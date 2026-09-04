# Classifier 6.1 còn chạy được không? — 2026-09-04

Câu hỏi đặt ra sau khi đổi detector lượt 1: 6.1 được hiệu chuẩn trên crop cắt
từ box của detector cũ, giờ crop đến từ box của detector mới.

**Trả lời ngắn: model KHÔNG hỏng, và KHÔNG nên train lại lúc này.** Nhưng nó
đang bỏ 30–43% linh kiện vào hàng chờ người xem trên ảnh của dự án, và nguyên
nhân lớn nhất **không phải** detector mới.

---

## 1. Chuẩn của chính nó

`models/active/classifier/model_manifest.json`: efficientnet_b0, 16 họ,
accuracy **0,958**, macro-F1 0,890, ngưỡng accept **0,8168** sau hiệu chuẩn
nhiệt độ, `accepted_precision` **0,980**, `accepted_coverage` **0,949**.

Tức trên tập test của nó, 94,9% crop được `accept` và trong số đó 98,0% đúng.

## 2. Đo lại trên ảnh thật

| ảnh | detector cắt crop | crop | accept | độ tin trung vị |
|---|---|---:|---:|---:|
| bo fixture `solder_geometry` | cũ 22 lớp | 39 | **94,9%** | 0,981 |
| bo fixture `solder_geometry` | mới 1 lớp | 62 | 75,8% | 0,959 |
| tile PCB-DSLR (11 tile, 3 bo) | **cũ 22 lớp** | 232 | **70,3%** | 0,968 |
| tile PCB-DSLR (11 tile, 3 bo) | mới 1 lớp | 838 | 57,0% | 0,885 |
| tile PCB-DSLR, **box tay** | — | 640 | 63,9% | 0,911 |

Dòng thứ ba là dòng quan trọng nhất: **cùng một detector cũ**, chỉ đổi ảnh, và
accept tụt từ 94,9% xuống 70,3%.

**⇒ Nguyên nhân lớn nhất là LỆCH MIỀN ẢNH (−24,6 điểm), không phải detector mới
và không phải khung cắt.** 6.1 được train trên bộ consolidated của Kaggle, còn
ảnh dự án là PCB-DSLR.

## 3. Ba giả thuyết bị chính phép đo bác bỏ

**"Detector mới làm hỏng 6.1."** Không. Ghép theo tâm 38 linh kiện xuất hiện ở
cả hai bộ box, rồi so nhãn dưới hai khung cắt: **đổi nhãn 1/38 = 2,6%**. Nhãn
rất ổn định; thứ đổi là độ tin.

*(Báo cáo box §7.2 đo được 22,3% — đó là phép so khác: crop-detector với
crop-box-tay trên tile gán nhãn, không phải hai detector trên cùng bo. Hai con
số không mâu thuẫn vì chúng đo hai thứ.)*

**"Linh kiện nhỏ là vấn đề."** Không. Accept theo dải cỡ trên 640 box tay gần
như phẳng:

| dải cạnh dài | n | accept | tin trung vị |
|---|---:|---:|---:|
| <24 px | 337 | 66,5% | 0,915 |
| 24–40 px | 78 | 64,1% | 0,909 |
| 40–64 px | 85 | 55,3% | 0,909 |
| 64–120 px | 47 | 63,8% | 0,904 |
| ≥120 px | 93 | 67,7% | 0,956 |

**"Nó nhầm linh kiện thành nền."** Không. Trên 640 box tay — mọi box đều là
linh kiện thật — chỉ **1,2%** bị đoán là `false_crop_background`.

## 4. Vì sao accept tụt thêm với detector mới

Trên bo fixture, tách 62 linh kiện của detector mới làm hai nhóm:

| nhóm | n | accept | tin trung vị | cạnh dài trung vị |
|---|---:|---:|---:|---:|
| detector cũ cũng tìm ra | 38 | **89,5%** | 0,977 | 78,1 px |
| **chỉ detector mới tìm ra** | 24 | **54,2%** | 0,881 | 50,7 px |

Detector mới tìm ra 838 crop trên tile PCB-DSLR so với 232 của bản cũ. Phần
tụt thêm chủ yếu là **thành phần**: nó đưa ra ánh sáng những linh kiện mà bản
cũ chưa từng cho classifier nhìn thấy, và chúng khó hơn.

Rơi vào `review` là **hành vi đúng** của một model không chắc — nó không đoán
liều. Cái đổi là khối lượng người phải xem.

---

## 5. Có cần sửa gì không

**Đã làm:** `run()` giờ so tỉ lệ accept quan sát được với `accepted_coverage`
trong manifest và cảnh báo khi lệch quá 15 điểm. Lệch miền không báo lỗi ở đâu
cả — nhãn vẫn ra, ROI vẫn dựng — nên nếu không đo thì không ai thấy.

**Chưa làm, và chưa nên làm:**

- **Hạ ngưỡng accept.** Nó sẽ kéo coverage lên ngay, nhưng ngưỡng 0,8168 đi
  kèm nhiệt độ hiệu chuẩn trên tập validation *của bộ cũ*. Trên miền đã lệch,
  hiệu chuẩn đó không còn đúng, nên hạ ngưỡng là **mua coverage bằng một mức
  precision không đo được**. `accepted_precision` 0,980 hiện là thứ đang bảo vệ
  cả đường ống phía sau.
- **Train lại.** Đây mới là chỗ mấu chốt: **không có nhãn HỌ nào cho ảnh của dự
  án.** 9.486 box đã khoanh chỉ mang một lớp `component`. Không có nhãn thì
  không train được, và cũng **không đo được độ chính xác thật** — mọi con số ở
  §2 là *độ tin*, không phải *độ đúng*.

## 6. Đường rẻ nhất để gỡ

Tập kiểm 600–800 box phân tầng ở
[kế hoạch package §7](../ke_hoach/ke_hoach_phan_nhom_package.md) vốn được lập
để đo bộ luật package. **Gán thêm nhãn họ cho cùng những box đó** thì cùng một
lượt công việc trả lời được cả ba câu:

1. tỉ lệ trúng của luật package;
2. **độ chính xác thật của 6.1 trên miền ảnh của dự án** — hiện chưa ai biết;
3. có nên hiệu chuẩn lại nhiệt độ/ngưỡng cho miền này không.

Chỉ sau khi có (2) mới nói được 6.1 cần train lại hay chỉ cần hiệu chuẩn lại.
Hiệu chuẩn lại rẻ hơn train lại rất nhiều, và với accuracy 0,958 trên miền gốc
thì khả năng cao đó mới là thứ cần.

---

## 7. Đo được rồi — 2026-09-04

§6 nói độ chính xác thật "hiện chưa ai biết". Giờ biết: **364 box** trong tập
kiểm phân tầng đã được gán nhãn họ **bằng mắt**, không dùng 6.1 (dùng nó thì
phép đo tự xác nhận chính nó).

> ⚠️ Đây là nhãn **tiền gán, người dùng CHƯA duyệt**. Con số dưới đây sẽ đổi
> sau lượt duyệt. Nhưng thứ tự giữa các tầng thì khó đảo.

### 7.1. Độ tin có thông tin thật

| tầng | n | đúng **họ** | đúng **hình học chân** |
|---|---:|---:|---:|
| `accept` | 263 | **88,2%** | **95,1%** |
| `review` | 64 | 54,7% | 68,8% |
| `unknown` | 37 | 40,5% | 83,8% |
| tổng | 364 | 77,5% | 89,3% |

Ở mức **họ**, thứ tự sạch và đơn điệu: 88,2 > 54,7 > 40,5. Nên câu "confidence
không để làm gì" không đúng — nó xếp hạng được.

Nhưng ở mức **hình học chân** — thứ bước 5.5 thật sự tiêu thụ — thứ tự **vỡ**:
`unknown` (83,8%) tốt hơn `review` (68,8%). Lý do: phần lớn nhầm lẫn của 6.1 là
giữa các họ **cùng dẫn về một hình học** (`capacitor`↔`led`↔`resistor` đều là
`two_terminal`). Ngưỡng đang hiệu chuẩn cho *độ mịn họ*, trong khi thứ dùng nó
cần *độ mịn topology*.

### 7.2. Vậy có nên bỏ cổng `accept` không? **Không.**

101 box bị cổng từ chối hiện lùi về `multi_pin`. So hai chính sách trên đúng
364 box đó:

| chính sách | **bỏ ROI** (lọt lưới) | thừa ROI | đúng hình học |
|---|---:|---:|---:|
| **hiện tại** — chỉ dùng nhãn khi `accept` | **6** | 53 | 83,8% |
| bỏ cổng — dùng nhãn ở mọi tầng | **19** | 20 | **89,3%** |

Bỏ cổng **làm tổng độ chính xác tăng** 83,8% → 89,3%, và trên bo 28 pad thì
độ phủ cũng nhích 11/28 → 13/28. Nhìn hai con số đó thì bỏ cổng có vẻ đúng.

Nhưng cột đầu mới là cột phải đọc: nó **gấp ba số ca bỏ ROI, từ 6 lên 19**. Bỏ
ROI nghĩa là mối hàn đó không ai kiểm — bo lỗi đi ra khỏi chuyền. Thừa ROI chỉ
tốn một cái liếc mắt. Đổi 13 ca lọt lưới lấy 33 ca đỡ phải xem là sai chiều với
bài toán kiểm tra.

Đã thử tìm chính sách theo tầng ăn được cả hai (nhận `unknown` nhưng chặn
`review`, hoặc chỉ nhận nhãn khi model nói `multi_pin`): **không có** — mọi
biến thể đều ra đúng 6 bỏ / 53 thừa của chính sách hiện tại, hoặc tệ hơn.

**Một điều tôi ghi lại vì nó bác chính lý lẽ tôi từng viết trong code:**
`multi_pin` KHÔNG phải mặc định an toàn miễn phí. Trong 101 box bị từ chối,
**46 box thật sự là `two_terminal`** — đẩy chúng sang `multi_pin` là dựng dải
quanh cả 4 cạnh của linh kiện 2 chân, đúng cái bệnh mà kế hoạch package đang
chữa. Nó an toàn theo nghĩa *không bỏ sót*, không an toàn theo nghĩa *đặt ROI
đúng chỗ*.

### 7.3. Việc đáng làm, theo thứ tự

1. **Người dùng duyệt 750 nhãn tiền gán.** Mọi con số ở §7 phụ thuộc vào chúng.
2. **Hiệu chuẩn lại nhiệt độ + ngưỡng trên miền này**, không phải train lại.
   `accept` đã đúng 95,1% ở mức hình học; vấn đề là nó chỉ phủ 72% số box. Một
   lần hiệu chuẩn lại có thể kéo coverage lên mà không đụng vào 6 ca bỏ ROI.
3. **Chỉ tính train lại nếu (2) không đủ.** Accuracy 0,958 trên miền gốc và
   95,1% hình học ở tầng `accept` cho thấy model không hỏng — nó bị lệch miền.
