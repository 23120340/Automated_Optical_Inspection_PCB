# Báo cáo tiến độ — Dự án AOI PCB

> Cập nhật: 2026-08-20. Tóm tắt những gì đã làm được, chưa xong đến đâu là
> chính xác — không tô hồng. Chi tiết kỹ thuật từng phần xem các file khác
> trong `docs/` và `README.md` ở gốc repo.

## Pipeline tổng quan

```
0 import → 1 tiền xử lý → 2 align (golden image) → 3 khoanh PCB →
4 detect linh kiện → 5 crop → 5.5 suy ra ROI mối hàn → 6.1 phân loại họ
linh kiện → 6.2 chấm lỗi mối hàn
```

Streamlit + OpenCV + Ultralytics YOLO + ONNX Runtime. Bộ test hiện tại:
**243/243 pass**.

## Đã hoàn thành theo từng bước

**Bước 0–3 (import/tiền xử lý/align/khoanh board)** — ổn định, có test.
Sửa một lỗi thật do OpenCV 5 đổi shape trả về của corner detector, làm hỏng
hiệu chỉnh ống kính; đã có test chạy OpenCV thật để bắt lại nếu tái phát.

**Bước 4 (detect linh kiện)** — notebook v2
(`training/kaggle/pcb_detector_v2_kaggle.py`): oversample class hiếm
(pads/pins) có trần an toàn, `imgsz` 1536, `copy_paste`, cổng kiểm tra
accelerator (chặn sớm lỗi chọn nhầm GPU P100), cổng verdict cuối so sánh
recall với baseline. Thêm cơ chế **resume từ checkpoint** (`last.pt`) để một
lần train bị đứt giữa chừng (mất mạng, đóng tab) không phải chạy lại từ đầu.

**Bước 5 / 5.5 (crop + ROI mối hàn)** — điểm khó nhất của dự án: không dataset
công khai nào gán nhãn chân/pad riêng, nên ROI mối hàn được **suy ra hình
học** từ box linh kiện + topology chân (`aoi_pipeline/solder.py`),
không phải detect trực tiếp. Ba lớp hợp nhất chồng lên trên, theo thứ tự ưu
tiên và đều đã đo, không chỉ lý thuyết:

1. **Lead/pad detection thật** (`inspection/leads.py`) thắng khi có, theo
   *từng chân* chứ không theo cả linh kiện — model chỉ thấy một đầu thì đầu
   kia vẫn giữ ROI suy ra.
2. **CAD fusion** (`inspection/cad.py`, `fusion.py`) — hợp nhất khi có file
   CAD board, hiệu chỉnh cục bộ từng linh kiện, không thay thế detector.
3. **Siết theo kim loại thật** (`refine_to_metal`) — thu ROI về đúng vùng kim
   loại bên trong nó, đo được cải thiện IoU 0.24→0.70 trên board tổng hợp và
   16/24 ROI trên board thật; chỉ tìm trong ROI đã dự đoán, không mở rộng ra
   lân cận (đã thử và đo là tệ hơn trên board thật).

**Bước 6.1 (phân loại họ linh kiện)** — notebook v2
(`pcb_classifier_v2_kaggle.py`): bỏ ràng buộc Raspberry Pi, đổi sang
ConvNeXt-Base (8 backbone khác để so sánh), input 288px, layer-wise LR decay
(sửa một lỗi khiến decay không thực sự chạy — đo được macro recall
0.731→0.883 sau khi sửa), EMA, TTA 4-view lúc train (0.929→0.942).

**Bước 6.2 (chấm lỗi mối hàn)** — kiến trúc 3 tầng
(`aoi_pipeline/grading/`): đo vật lý (không cần train) → luật kèm lý do →
model ONNX tuỳ chọn → hợp nhất. `escape_guard` là sàn vật lý cuối: model nói
đạt nhưng đo được ít thiếc thì vẫn ép về hàng đợi kiểm, không confidence nào
vượt qua được. Chạy được ngay cả khi chưa có model train.

**Ghép dataset cho 6.2** — không nguồn công khai nào đủ một mình, nên notebook
tự dò layout (folder-per-class/COCO/CSV/YOLO/**LabelMe**) và ghép nhiều
nguồn, cưỡng chế 3 nguyên tắc: nhãn lạ bị bỏ và đếm chứ không đoán, chia tập
theo board không theo crop (tránh rò rỉ), lớp thiếu dữ liệu bị loại khỏi
`class_names` thay vì train một head chưa từng thấy nó. Sửa một lỗi thật khiến
toàn bộ 428 ảnh SolDef_AI đọc ra 0 record (thiếu reader LabelMe); sau khi sửa,
lần chạy thật cho thấy nhãn thô thật sự khác thuật ngữ bài báo
(`no_good`/`poor_solder`/`spike`/`exc_solder`) — đã map phần chắc chắn
(`exc_solder`, `spike` → `excess`), cố tình chưa map phần mơ hồ
(`no_good`, `poor_solder`) và thêm cell tự vẽ ảnh mẫu để quyết định bằng mắt
thay vì đoán. Nguồn Hugging Face (`hf_soldering_boarding`) giờ tải tự động
qua thư viện `datasets`, không cần thao tác tay.

## Tăng độ chính xác không cần train lại

- **TTA lúc suy luận**: cả hai classifier (6.1, 6.2) và detector đều có tuỳ
  chọn trung bình 4 góc nhìn / augment tích hợp Ultralytics, mặc định tắt.
  Với classifier 6.1 đã có số đo thật (0.929→0.942 macro recall).
- **Nghi vấn lệch miền tiền xử lý**: bước 1 mặc định bật denoise/CLAHE/
  white-balance/normalize/sharpen, nhưng không notebook train nào áp dụng
  chuỗi này lên ảnh train — rủi ro thật, xác minh được qua code, nhưng
  **chưa đo được** vì thiếu ảnh board thật để test. Có sẵn công cụ đo:
  `scripts/compare_preprocessing_ab.py --isolate`.

## Trạng thái model hiện tại

| Model | Kiến trúc | Trạng thái |
|---|---|---|
| Detector (bước 4) | YOLO26s | v1 đã có trong `models/detector/`; v2 (oversample + imgsz 1536) đang/đã train trên Kaggle |
| Classifier (bước 6.1) | ConvNeXt-Base | v1 đã có trong `models/classifier/`; v2 đã đo macro recall 0.942 với TTA |
| Solder grading (bước 6.2) | MobileNetV3-Small | Chưa có model thật trong `models/` — pipeline đang chạy bằng tầng luật; notebook train đã chạy được hết tới ONNX hợp lệ trên dữ liệu ghép |

## Giới hạn còn tồn tại — nói thẳng

- **Độ phủ dataset detector chỉ 3/22 class đạt chuẩn** (capacitor, resistor,
  ic = 81% dữ liệu); 9 class dưới 100 instance; 3 class không đo được recall
  vì 0 mẫu val.
- **Dataset 6.2 vẫn thiếu nhiều lớp**: `bridge`, `cold`, `tombstone`,
  `wrong_polarity` chưa có nguồn công khai đủ tin cậy; `no_good`/
  `poor_solder` (145 mẫu SolDef_AI) đang chờ người xem ảnh để quyết định map.
- **Hai nút thắt vật lý không dataset nào bù được**: cold solder cần đèn vòng
  RGB đa góc mới tách được khỏi mối hàn tốt; fillet cần ~15–25 µm/px trong
  khi ảnh nhập vào hiện thấp hơn nhiều.
- **Domain gap tiền xử lý**: xem mục trên, cần ảnh board thật để đo.
- **Model production thật sự** vẫn cần fine-tune trên chính board/camera/ánh
  sáng của dây chuyền — mọi model hiện tại học từ dataset công khai, khác
  miền với thiết bị thật.

## Việc tiếp theo, theo thứ tự ưu tiên

1. Chờ detector v2 + classifier v2 train xong trên Kaggle, so kết quả với
   baseline (cổng verdict đã có sẵn trong notebook).
2. Chạy `compare_preprocessing_ab.py` trên ảnh board thật ngay khi có model,
   quyết định giữ/tắt từng bước tiền xử lý dựa trên số đo thật.
3. Xem ảnh mẫu cho `no_good`/`poor_solder` (cell đã có trong notebook 6.2),
   bổ sung `LABEL_MAPS` nếu xác định được — thêm gần gấp đôi dữ liệu hiện có.
4. Tự thu thập ảnh từ chính dây chuyền/board thật — nguồn dữ liệu giá trị
   nhất, không dataset công khai nào thay thế được
   (`scripts/export_solder_dataset.py --overlays` hỗ trợ bootstrap nhãn).
