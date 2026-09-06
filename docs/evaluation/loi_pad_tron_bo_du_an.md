# Lượt 1 khoanh pad tròn trên bo dự án — chẩn đoán

Đo 2026-09-06 trên `00001__1024__1648___4120.png` (ảnh mẫu của **bo dự án**).

> **Kết luận ngắn: không phải lỗi nhãn.** Bộ nhãn đạt recall **94%** trên chính
> miền của nó. Lỗi là **một lớp thị giác duy nhất** — pad xuyên lỗ / lỗ mạ tròn —
> trên một bo **không có trong tập huấn luyện**.

---

## 1. Triệu chứng

Lượt 1 sinh **62 box** trên tile này. Cắt từng box ra xem bằng mắt:

| | số box | |
|---|---:|---|
| linh kiện thật | 42 | 68% |
| **pad tròn / lỗ mạ, không phải linh kiện** | **20** | **32%** |

20 box sai **cùng một kiểu**: vòng đồng tròn, lỗ mạ, hoặc pad trống sáng. Không
rải rác nhiều kiểu lỗi — một lớp thị giác duy nhất.

## 2. Không phải do bộ nhãn

Chạy detector trên 6 tile **có nhãn tay**, so bằng IoU ≥ 0,4:

| | box nhãn | box detect | trùng | recall | thừa |
|---|---:|---:|---:|---:|---:|
| 6 tile trong miền | 584 | 660 | 548 | **94%** | 112 (17%) |

Nếu bộ nhãn có khoanh cả pad và mối hàn thì detector đã nổ trên **mọi** pad và
số thừa phải cao hơn hẳn. Vẽ nhãn lên tile huấn luyện xem tận mắt cũng thấy:
chúng khoanh **thân linh kiện**, box nhỏ là chip 0402/0603 thật, không có box
nào đặt lên pad trống.

**Nguyên nhân thật:** tile `00001__1024__1648___4120` **không nằm trong tập
huấn luyện**. Tập là 120 tile của bo PCB-DSLR; đây là bo của chính dự án, tức
**ngoài miền**. Bo PCB-DSLR gần như toàn linh kiện dán, nên detector chưa từng
thấy pad xuyên lỗ lộ thiên cỡ này làm **ví dụ âm**.

## 3. Nâng ngưỡng tin cậy KHÔNG cứu được

| ngưỡng | giữ box đúng | còn box sai |
|---|---:|---:|
| 0,25 *(hiện tại)* | 41/42 (98%) | 19/20 (95%) |
| 0,45 | 33/42 (79%) | 12/20 (60%) |
| 0,55 | 28/42 (67%) | 8/20 (40%) |
| **0,75** | **19/42 (45%)** | **0/20 (0%)** |

Muốn sạch hết box sai thì mất **hơn một nửa linh kiện thật**. Với AOI thì bỏ sót
linh kiện tệ hơn hẳn một box thừa, nên đây là đánh đổi sai chiều.

**Lọc theo hình dạng cũng không được**: tụ hoá nhìn từ trên **cũng tròn**, và
trong 42 box đúng có ít nhất 3 cái là tụ trụ tròn. Tròn không tách được tròn.

## 4. Lượt 2 phần lớn vô can

| chạy lượt 2 trên | chân sinh ra | ROI mối hàn |
|---|---:|---:|
| cả 62 box | 73 | 220 |
| chỉ 42 box **đúng** | 72 | 161 |
| chỉ 20 box **sai** | **1** | **59** |

Lead detector **từ chối** gần hết box sai: 72/73 chân đến từ box đúng. Nhưng
**59/220 = 27% ROI mối hàn** vẫn được **suy ra bằng hình học** từ 20 box sai đó,
và chúng nằm trên pad trống.

⇒ Cái nhìn giống "lượt 2 sai chân hàn nghiêm trọng" thực ra là **ROI suy ra từ
box sai của lượt 1**. Sửa lượt 1 thì 27% ROI rác biến mất theo.

## 5. Vì sao KHÔNG nên lọc nhãn bằng kết quả model

Có đề xuất: phủ kết quả detect lên nhãn tay, giữ box trùng, xoá box không trùng.
**Không nên**, ba lý do đo được:

1. **Nó xoá đúng phần model đang sai.** Recall 94% nghĩa là 6% (36 box trong
   584) là linh kiện thật mà model **bỏ sót**. Lọc theo cách trên xoá đúng 36
   cái đó — tức xoá những ca khó nhất, thứ duy nhất dạy được model.
2. **Nó làm nhãn đồng ý với model.** Lần train sau điểm sẽ đẹp hơn và thực tế
   thì tệ hơn. Đây đúng loại tự xác nhận mà dự án đã cấm ở chỗ khác: *"dùng
   model để điền nhãn cho tập dùng để đo chính model đó"*.
3. **Nó không chạm được vào lỗi đang gặp.** 20 pad tròn kia **không có trong bộ
   nhãn** — chúng là nền. Xoá nhãn không xoá được thứ không nằm trong nhãn.

## 6. Hướng đúng

**Train lại, nhưng bằng cách THÊM dữ liệu chứ không xoá nhãn.**

Cần khoảng **10–20 tile từ chính bo dự án**, khoanh cùng quy ước (chỉ thân, loại
chân/pad). Khi đó pad tròn xuất hiện trong ảnh mà **không có nhãn**, tức thành
ví dụ âm — đó là thứ duy nhất dạy được "vòng đồng tròn ≠ linh kiện".

Đang có: `00001__1024__1648___1648.png`, `00001__1024__1648___4120.png`, và
`real_pcb/phone/whole_pcb.jpg`. **Cần thêm ảnh gốc độ phân giải đầy đủ của bo
dự án** để cắt đủ 10–20 tile.

Thứ tự đề xuất:

1. Cắt tile từ ảnh bo dự án, chọn tile **có pad xuyên lỗ** — đó là lớp đang hỏng.
2. Khoanh tay bằng chính app đang dùng, cùng quy ước.
3. Gộp vào tập hiện có, train lại.
4. Đo lại **trên tile bo dự án giữ riêng** (không dùng để train), báo cáo số box
   sai kiểu pad tròn trước/sau.

Bước 4 là bắt buộc: không có tập giữ riêng thì không biết đã sửa được hay chỉ
làm model thuộc lòng thêm vài tile.
