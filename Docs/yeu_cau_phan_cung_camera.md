# Yêu cầu tối thiểu về phần cứng chụp ảnh (AOI PCB)

> Soạn 2026-08-21. Mọi con số ở đây **đo từ chính ảnh của dự án** hoặc tính ra
> từ số đo đó, không lấy từ catalogue nhà sản xuất.

## 1. Hiện trạng: ảnh đang chụp ở 46 µm/px

Không cần biết thông số máy đã dùng — bản thân tấm ảnh tự nói ra, vì bao bì linh
kiện có kích thước tiêu chuẩn. Đo trên `00001__1024__1648___4120.png`:

| Căn cứ | Kích thước chuẩn | Đo được | Suy ra |
|---|---|---|---|
| SOT-23 D202, 2 chân cùng cạnh | 1.90 mm | 40.7 px | 46.7 µm/px |
| SOIC U201, pitch chân (cạnh trên) | 1.27 mm | 28.1 px | 45.2 µm/px |
| SOIC U201, pitch chân (cạnh dưới) | 1.27 mm | 27.2 px | 46.7 µm/px |

Kiểm chéo bằng kích thước thân: điện trở mảng đo ra 2.12 × 1.47 mm (chuẩn 0805
là 2.00 × 1.25); thân D202 đo 2.85 mm (chuẩn SOT-23 là 2.9). Ba phép đo độc lập
chênh nhau dưới 3%.

**Kết luận: ảnh hiện tại ở khoảng 46 µm/px.**

## 2. Ở 46 µm/px làm được gì và không làm được gì

| Loại lỗi | Ở 46 µm/px | Lý do |
|---|---|---|
| Thiếu linh kiện | **Được** | linh kiện 0805 chiếm ~46 × 27 px, quá rõ |
| Sai linh kiện / sai chiều | **Được** | đọc được thân và vạch chỉ cực |
| Lệch vị trí (shift, tombstone) | **Được** | đo bằng trọng tâm, không cần chi tiết |
| Cầu thiếc (bridge) | **Được** | thiếc nối hai pad là vệt lớn |
| Định vị pad/chân cho lượt 2 | **Được, sát ngưỡng** | pad chừng 10–15 px |
| Thiếu thiếc / thừa thiếc | **Miễn cưỡng** | đo theo diện tích, không theo hình |
| **Cold solder (hàn nguội)** | **Không** | phải đọc *độ dốc và độ bóng* của fillet |
| **Hình dạng fillet** | **Không** | fillet chỉ vài pixel, không còn hình |

Ngưỡng thường dùng cho kiểm tra fillet là **15–25 µm/px**. Hiện đang thiếu
khoảng **2 lần**.

Để so sánh: dataset SolDef_AI (ảnh macro, mỗi ảnh một linh kiện) ở khoảng
1–3 µm/px. Model train trên đó cho **0 box** khi chạy lên ảnh 46 µm/px của dự
án, ở mọi mức phóng to từ 1× đến 12×. **Phóng to bằng phần mềm không tạo ra chi
tiết chưa từng được chụp** — đây là giới hạn vật lý, không phải giới hạn thuật
toán.

## 3. Yêu cầu tối thiểu

### 3.1 Độ phân giải — chọn theo mục tiêu

| Mục tiêu | µm/px cần | Ghi chú |
|---|---|---|
| Giữ nguyên năng lực hiện tại | 46 | không cần đổi gì |
| **Khuyến nghị: định vị chân + lỗi thô** | **≤ 25** | đủ cho lượt 2 và phần lớn lỗi |
| Chấm hình dạng fillet, cold solder | ≤ 15 | kèm điều kiện chiếu sáng ở 3.3 |

Số điểm ảnh cần, theo trường nhìn:

| FOV (mm) | 46 µm/px | 25 µm/px | 15 µm/px |
|---|---|---|---|
| 200 × 150 | 4348 × 3261 (14 MP) | 8000 × 6000 (48 MP) | 13333 × 10000 (133 MP) |
| 100 × 75 | 2174 × 1630 (3.5 MP) | 4000 × 3000 (12 MP) | 6667 × 5000 (33 MP) |
| 50 × 38 | 1087 × 815 (0.9 MP) | 2000 × 1500 (3.0 MP) | 3333 × 2500 (8.3 MP) |
| 35 × 26 | 761 × 565 (0.4 MP) | 1400 × 1040 (1.5 MP) | 2333 × 1733 (4.0 MP) |

Chụp cả board 200 × 150 mm trong một khung ở 15 µm/px cần **133 MP** — không có
cảm biến công nghiệp phổ thông nào làm được. Nên **giảm trường nhìn và chụp
nhiều lần**, đúng cách các máy AOI thương mại vẫn làm.

### 3.2 Số lần chụp để phủ board 200 × 150 mm

Đã trừ 15% chồng biên giữa các khung.

| µm/px | Cảm biến | Trường nhìn | Số lần chụp |
|---|---|---|---|
| 46 | 12 MP (4024 × 3036) | 185 × 140 mm | **4** |
| 25 | 12 MP | 101 × 76 mm | **9** |
| 25 | 20 MP (5472 × 3648) | 137 × 91 mm | **4** |
| 15 | 12 MP | 60 × 46 mm | **16** |
| 15 | 20 MP | 82 × 55 mm | **12** |

**Cấu hình khuyến nghị: cảm biến 20 MP + trường nhìn ~137 × 91 mm → 25 µm/px,
4 lần chụp mỗi board.** Đây là điểm cân bằng tốt nhất giữa chi phí, thời gian
chu kỳ và năng lực phát hiện lỗi.

### 3.3 Ánh sáng — quan trọng ngang độ phân giải

Chi tiết fillet nằm ở **hướng phản xạ**, không nằm ở độ phân giải. Một mối hàn
tốt có mặt cong bóng phản xạ theo góc nhất định; mối hàn nguội xỉn và phản xạ
tán. Đèn chiếu phẳng thì hai thứ đó trông như nhau **ở bất kỳ độ phân giải nào**.

Tối thiểu:

- **Đèn vòng nhiều tầng góc** (thường gọi RGB tri-level / three-level ring):
  tầng thấp, tầng giữa, tầng cao chiếu ở ba góc tới khác nhau. Đây là điều kiện
  **bắt buộc** để phân biệt hàn nguội, không phải tuỳ chọn.
- Ánh sáng khuếch tán, **không có điểm loá cứng** trên mặt board.
- Cường độ và nhiệt độ màu **cố định**, có thể lặp lại giữa các lần chụp. Model
  học màu; đèn đổi là miền dữ liệu đổi.
- Chặn ánh sáng môi trường: hộp che hoặc buồng kín.

### 3.4 Ống kính

- **Ống kính telecentric** nếu cần đo vị trí chính xác. Ống thường gây méo phối
  cảnh: linh kiện ở rìa khung nhìn nghiêng, làm sai phép đo lệch vị trí.
- Độ phân giải ống kính phải **theo kịp cảm biến**: ống 5 MP gắn trên cảm biến
  20 MP thì chỉ thu được chi tiết mức 5 MP.
- **Chiều sâu trường ảnh (DOF)** phải phủ được chênh cao linh kiện trên board —
  từ chip 0402 (0.35 mm) tới tụ can (10 mm trở lên). Thiếu DOF thì tụ nét mà
  chip mờ, hoặc ngược lại.
- Khoảng cách làm việc phải chừa chỗ cho đèn và cho việc gá board.

### 3.5 Cơ khí

- **Cố định board**: mọi phép đo vị trí đều so với gốc toạ độ; board xê dịch
  giữa các lần chụp là hỏng.
- Nếu chụp nhiều khung: bàn XY có **độ lặp lại tốt hơn 1 pixel** ở µm/px mục
  tiêu (tức < 25 µm cho cấu hình khuyến nghị), hoặc dùng fiducial để ghép ảnh
  bằng phần mềm.
- Chống rung: chụp trên bàn có giảm chấn, hoặc dừng hẳn trước khi chụp.

### 3.6 Định dạng ảnh

- **Không nén mất dữ liệu.** PNG hoặc TIFF. JPEG tạo nhiễu khối đúng ở vùng
  tương phản cao — chính là mép fillet.
- **Không auto-exposure, không auto-white-balance.** Phơi sáng cố định, cân bằng
  trắng cố định. Máy tự chỉnh nghĩa là mỗi tấm một miền dữ liệu khác nhau.
- Giữ ảnh gốc; mọi khâu tiền xử lý làm ở phần mềm để còn quay lại được.

## 4. Bảng kiểm khi nghiệm thu thiết bị

- [ ] Chụp một board mẫu, đo lại µm/px bằng chính cách ở mục 1 (pitch SOIC hoặc
      khoảng cách chân SOT-23). **Đừng tin thông số ghi trên máy — hãy đo.**
- [ ] Fillet của một mối hàn 0805 chiếm **ít nhất 15 px** ở cạnh ngắn.
- [ ] Chụp cùng một board 10 lần liên tiếp: độ lệch vị trí giữa các lần **dưới
      1 px**.
- [ ] Chụp một mối hàn tốt và một mối hàn nguội đã biết: **nhìn bằng mắt trên
      ảnh phải phân biệt được**. Nếu mắt người không phân biệt nổi thì model
      cũng không, và không có phần mềm nào cứu được.
- [ ] Không có vùng loá cháy trắng nào trên pad.
- [ ] Chạy `scripts/compare_preprocessing_ab.py --isolate` trên ảnh mới để chốt
      bật/tắt từng bước tiền xử lý dựa trên số đo.

## 5. Lưu ý về chi phí

Nâng từ 46 lên 25 µm/px là **cải thiện 2 lần theo mỗi chiều, tức 4 lần số điểm
ảnh** — kéo theo số lần chụp và thời gian chu kỳ. Nâng lên 15 µm/px là 9 lần số
điểm ảnh so với hiện tại.

Nếu ngân sách chỉ đủ một hạng mục: **chọn hệ chiếu sáng nhiều góc trước, không
phải cảm biến**. Độ phân giải cao hơn dưới ánh sáng phẳng vẫn không phân biệt
được hàn nguội, còn ánh sáng đúng ở 46 µm/px thì đã bắt được thêm một loạt lỗi
mà hiện giờ đang bỏ sót.
