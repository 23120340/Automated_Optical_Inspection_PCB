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
