# Khảo sát model "PCB defect" trên Hugging Face

> Khảo sát 2026-08-23 trên 52 kết quả của `huggingface.co/models?search=pcb+de`.
> Đã tải về chạy thật hai ứng viên; ba nhóm còn lại loại được mà không cần tải.
>
> Quy ước: mục nào **đã đo** thì ghi rõ đo bằng cách nào. Mục nào chỉ đọc tài
> liệu thì ghi là chưa kiểm chứng.

## Kết luận

**keremberke YOLOv8 — chưa loại, đang chờ một board lỗi thật.** Nó chịu được
thang chụp của dự án và bắt được lỗi hàn thật ở thang đó. Điều chưa chứng minh
được là nó có chạy trên board *của bạn*, chụp bằng camera *của bạn* hay không.

Mọi model còn lại đã loại, mỗi cái vì một lý do khác nhau — xem mục 3.

## 1. keremberke YOLOv8n/m — đã tải, đã chạy

Nhóm duy nhất trong 52 kết quả có mAP công bố và dataset rõ ràng. Lớp
`Dry_joint` và `Short_circuit` đúng là hai thứ bước 6.2 đang thiếu.

`best.pt` **nạp được bằng ultralytics 8.4.104 của dự án** dù train bằng
8.0.21/8.0.23. **Đừng** cài `ultralyticsplus` / `ultralytics==8.0.23` theo model
card — nó sẽ hạ cấp thư viện và phá bước 4.

### Thang chụp: đo, không đoán

Đo bằng pitch chân SOIC 1.27 mm, đúng thước đã dùng cho board dự án:

| | µm/px |
|---|---|
| Ảnh keremberke | **~33** |
| Board dự án | 46 |
| Camera mục tiêu | 25 |

Ảnh của họ **nằm giữa** hiện tại và mục tiêu — không phải macro.

### Model chịu được thang chụp tới đâu

Lấy tập test của họ (79 lỗi có nhãn), thu nhỏ dần để mô phỏng camera thô hơn.
Thu nhỏ một ảnh mịn mô phỏng đúng một camera thô; **phóng to thì không tạo ra
chi tiết**, nên chỉ làm được một chiều.

| Hệ số | µm/px | Recall |
|---|---|---|
| 1.00 | 33 (gốc) | 0.595 |
| 0.72 | **46 (board bạn)** | **0.544** |
| 0.50 | 66 | 0.494 |
| 0.36 | 92 | 0.430 |

Suy giảm rất từ tốn — ở 46 µm/px vẫn giữ 0.544, chỉ kém gốc 9%. Nếu thang chụp
là rào cản thì recall đã sụp ở đây.

### Có bắt được lỗi thật ở thang của dự án không

6 ảnh chụp thật chứa 38 lỗi thật, thu về 46 µm/px, dán vào giữa board dự án:

| Model | box | trong vùng có lỗi |
|---|---|---|
| **keremberke yolov8m** | 36 | **36 (100%)** |
| keremberke yolov8n | 31 | 25 |
| SolDef_AI | 6 | 6 (toàn `spike`) |
| detector dự án *(đối chứng)* | 280 | 41 — nó tìm linh kiện, đúng vai |

Nó cũng **có** gọi `Dry_joint` (5 box ở conf 0.25 trên một ảnh).

### Điều chưa chứng minh được

Mọi phép đo trên dùng **ảnh của chính họ** — cùng camera, cùng ánh sáng, cùng
loại board. Đó là câu hỏi **miền ảnh**, và không có board lỗi thật của dây
chuyền thì không trả lời được.

Dataset của họ cũng nhỏ và lệch: **326 instance trên 128 ảnh**, mất cân bằng
17× (`incorrect_installation` 220 so với `dry_joint` 44), và **không khai giấy
phép** (`license: None`, export từ Roboflow). Đó là lý do để **kỳ vọng vừa
phải**, không phải lý do để loại.

### Việc nên làm — 10 phút, và nó quyết định

Chụp **một board có lỗi thật** bằng camera hiện tại:

```bash
curl -L -o yolov8m-pcb.pt \
  https://huggingface.co/keremberke/yolov8m-pcb-defect-segmentation/resolve/main/best.pt
```

```python
from ultralytics import YOLO
YOLO("yolov8m-pcb.pt").predict("board_loi.jpg", conf=0.25, imgsz=640)[0].save("kq.jpg")
```

- **Khoanh trúng lỗi** → có điểm khởi đầu miễn phí, và việc gán nhãn chuyển từ
  "vẽ từ đầu" sang "sửa box có sẵn"
- **Không thấy gì** → mới là lúc kết luận miền ảnh chặn

`.pt` mang pickle: nạp là chạy mã. Repo này 1000+ lượt tải/tháng nên rủi ro
thấp, nhưng đó là lý do bộ chọn model của app **không liệt kê `.pt`**.

## 2. Dukeb DETR — loại bằng một file 194 byte

`config.json` khai 24 lớp, tên là **`LABEL_0` … `LABEL_23`**.

Model **không mang tên lớp**, và bảng ánh xạ không công bố. Cả đường ống của dự
án khoá theo tên lớp — hình học chân 5.5, phân loại 6.1, đối chiếu BOM. Không
cần tải `pytorch_model.bin`, không cần cài `transformers`.

## 3. Các nhóm còn lại

| Nhóm | Vì sao loại |
|---|---|
| DeepVisionXplain (6 backbone) | **Không có model card** — không xác nhận được task, dataset hay hiệu năng |
| 8 model không tài liệu | Như trên |
| DeepSeek-R1-\*-PCB (8B) | **LLM văn bản**, không phải thị giác máy tính. Model card là template mặc định của Unsloth |
| gemma-pcb_product-description | Sinh mô tả bán hàng, không liên quan phát hiện lỗi |

Nhóm 5 và 6 nhận **văn bản** vào và trả **văn bản** ra — không có chỗ nào nhận
được một tấm ảnh board.

## 4. Bốn nguồn đã kiểm chứng, tổng kết

| Nguồn | Trạng thái |
|---|---|
| **keremberke** | **Chưa loại** — chạy được ở 46 µm/px, cần board lỗi thật |
| SolDef_AI | Loại — macro 1–3 µm/px; ở 46 µm/px chỉ ra `spike` |
| PCB-SAID | Loại — ảnh cào web 640×480, nhãn theo linh kiện, tải theo yêu cầu |
| Ulger | Loại — đúng tỉ lệ (~20–25 µm/px) nhưng **không có box** |
| Roboflow Universe | **Chưa kiểm chứng được** — cần API key miễn phí |

Ràng buộc vẫn là vật lý: ảnh dự án ở **46 µm/px**, và chưa nguồn công khai nào
vừa đúng thang đó vừa có nhãn định vị **và** giấy phép rõ ràng. Việc gán nhãn
board của chính dây chuyền (C1–C3 trong `Docs/tien_do_detect_2_luot.md`) vẫn là
đường chắc chắn nhất.

## 5. Về camera

Nâng cấp camera vẫn đáng, nhưng **không phải để cứu các model này** — keremberke
đã chạy được ở 46 µm/px. Lý do nằm ở chỗ khác, đã ghi trong
`Docs/yeu_cau_phan_cung_camera.md`: kiểm tra fillet cần 15–25 µm/px, và hàn nguội
cần **hướng chiếu sáng** đúng chứ không chỉ độ phân giải — "đèn chiếu phẳng thì
mối hàn tốt và mối hàn nguội trông như nhau ở bất kỳ độ phân giải nào".

---

## Phụ lục — ba điều tôi kết luận sai lúc đầu

Giữ lại vì chúng là ba cách sai dễ lặp lại, không phải để trách móc.

**1. Nhầm số điểm ảnh với thang chụp.** Tôi xếp keremberke chung với PCB-SAID vì
"cùng 640×480". Nhưng 640×480 nói về *số điểm ảnh*, không nói về *µm/px* — một
khung 640×480 chụp một vùng nhỏ vẫn rất mịn. Đo bằng pitch SOIC mới ra ~33 µm/px.

**2. Đọc "0 box trên board chuẩn" thành thất bại.** `golden.png` là board
**chuẩn** — không có lỗi để tìm. Không ra box chính là hành vi *đúng*.

**3. Phép thử lỗi nhân tạo quá thô.** Tôi vẽ đường thẳng giả làm mối hàn chập.
Model học từ **ảnh chụp**, chưa từng thấy nét vẽ, nên nó không phản ứng — và
điều đó không nói gì về khả năng bắt lỗi thật. Phép thử đúng là dán một mảnh
**ảnh chụp thật có lỗi thật** đã thu về đúng thang chụp.

*Ghi chú: trang tìm kiếm còn 1 trang nữa (23 model) chưa khảo sát.*
