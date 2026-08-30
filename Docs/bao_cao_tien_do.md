# Báo cáo tiến độ — Dự án AOI PCB

> Cập nhật: 2026-08-31. Tóm tắt những gì đã làm được, chưa xong đến đâu là
> chính xác — không tô hồng. Chi tiết kỹ thuật từng phần xem các file khác
> trong `Docs/` và `README.md` ở gốc repo.

## Pipeline tổng quan

```
0 import → 1 tiền xử lý → 2 align (golden image) → 3 khoanh PCB →
3.5 Golden Inspection → 4 detect linh kiện → 5 crop →
5.5 ROI mối hàn (lượt 2 → CAD → hình học) → 6.1 phân loại họ linh kiện →
6.2 chấm lỗi mối hàn
```

Streamlit + OpenCV + Ultralytics YOLO + ONNX Runtime. Bộ test hiện tại:
**1012/1012 pass, 0 skip** (đo 2026-08-31, 211 s).

Golden Inspection **không còn là workspace riêng**: từ 23/08 nó là **bước 3.5**
trong chính đường ống, ngay sau khi khoanh vùng board. Bước **6.2 là mục riêng**
trong điều hướng, không phải tab con của bước 4.

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
học** từ box linh kiện + topology chân (`aoi_pipeline/solder/geometry.py`),
không phải detect trực tiếp. Ba lớp hợp nhất chồng lên trên, theo thứ tự ưu
tiên và đều đã đo, không chỉ lý thuyết:

1. **Lead/pad detection thật** (`aoi_pipeline/solder/leads.py`) thắng khi có, theo
   *từng chân* chứ không theo cả linh kiện — model chỉ thấy một đầu thì đầu
   kia vẫn giữ ROI suy ra.
2. **CAD fusion** (`aoi_pipeline/solder/cad.py`, `cad_fusion.py`) — hợp nhất khi có file
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

Bảng dưới dựng lại **từ chính bốn `model_manifest.json` trong
`models/active/`** (đo 2026-08-31), không chép lại từ CONFIG của notebook — bản
trước mô tả model **archive** chứ không phải model app đang nạp.

| Ô model | Kiến trúc | Phiên bản trong manifest | Số đo |
|---|---|---|---|
| `detector` (bước 4) | YOLO26s, **imgsz 1280** | `detector-yolo26s-kaggle-ver1` | val mAP50 **0.579**, mAP50-95 **0.287**; test mAP50 0.558. Macro recall **0.522** |
| `classifier` (bước 6.1) | **EfficientNet-B0**, 224px | `classifier-efficientnet_b0-kaggle-ver1` | test accuracy **0.958**, macro F1 **0.890**, accept coverage 0.949 |
| `lead_detector` (lượt 2 của 5.5) | YOLO11s, 640px | `pcb-joint-locator-yolo11s-solderjoint` | test mAP50 **0.9912**, mAP50-95 0.560. **Opt-in, không tự nạp** |
| `solder/classifier` (bước 6.2) | MobileNetV3-Small, 128px | `solder-mobilenet_v3_small-ver1` | 7 lớp; ở ngưỡng 0.85: review **45.7%**, escape **0.98%**, false call 22.8% |

Hai điều chỉnh so với bản 21/08, cả hai đều là **bản trước ghi nhầm**, không phải
model bị thay:

- Detector đang chạy là **ver1 @1280**, không phải ver2 @1536. ver2
  (mAP50 0.505) nằm ở `models/archive/detector-yolo26s-kaggle-ver2/`.
- Classifier đang chạy là **EfficientNet-B0**. Bản ConvNeXt-Base nằm ở
  `models/library/`, tức **không tự nạp** — đó là thư mục model cá nhân.

**Về con số 89.9% của 6.2**: lần chạy trước báo 97.65%, nhưng đó là số **ảo** do
Roboflow sinh nhiều bản augment cho cùng một ảnh và chúng bị tách thành các
group khác nhau nên rơi vào cả train lẫn val. Sau khi gộp về ảnh gốc
(2334→1185 group) thì còn 89.9% — đây mới là số đúng.

Điểm yếu rõ nhất của 6.2: **`insufficient` recall chỉ 48.7%** (37/76), trong đó
18 mẫu bị đọc thành `good` — tức escape thật. Nguyên nhân là dữ liệu (320 mẫu
train, lệch 11:1 so với `shift_component`), không phải thiếu epoch: loss đã
phẳng từ epoch 21. Tầng luật (`escape_guard`) vẫn bắt độc lập với model.

`cold` đạt 100% (64/64) và `shift_component` 99.2% — cả hai **chỉ từ một
nguồn** và chưa kiểm chứng leave-one-source-out, nên chưa nên tin.

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

> Cập nhật 31/08. Ba việc đầu của bản 21/08 đã xong hoặc không còn là đường đi;
> danh sách dưới đây là ưu tiên thật hiện nay.

1. **Duyệt thêm 2 bo trong bộ gán nhãn vòng 2** — đây là **việc chặn duy nhất**
   giữa dự án và dataset train detector một lớp `component`. Packer từ chối ghi
   khi chưa đủ 10 bo và khi bucket `valid` còn trống; hiện 8 bo, `valid` trống.
   Packer tự nêu tên bo cần duyệt khi chạy (`pcb_dslr:017`, `pcb_dslr:030` ở
   checkpoint 30/08). Không có dòng code nào thay thế được việc này.
2. **Chạy `compare_preprocessing_ab.py --isolate`** trên ảnh board thật, quyết
   định giữ/tắt từng bước tiền xử lý bằng số đo. Việc này chặn cả kế hoạch
   phân nhóm package (các lớp dựa vào màu).
3. **Fine-tune model lượt 2 trên ảnh dây chuyền.** Manifest của nó tự khai
   `bootstrap_only`; phủ pad trung vị tụt 0.97 → 0.79 khi bật, pad yếu nhất
   0.52 so với cổng 0.50.
4. **Quyết hai kế hoạch mới đang chờ duyệt**:
   [phân nhóm package](ke_hoach_phan_nhom_package.md) và
   [lỗi toàn mạch](ke_hoach_pcb_defect_toan_mach.md).
5. Xem ảnh mẫu cho `no_good`/`poor_solder` (cell đã có trong notebook 6.2),
   bổ sung `LABEL_MAPS` nếu xác định được — thêm gần gấp đôi dữ liệu 6.2.
6. Tự thu thập ảnh từ chính dây chuyền/board thật — nguồn dữ liệu giá trị nhất,
   không dataset công khai nào thay thế được. Repo hiện có **3 ảnh điện thoại**
   ở `real_pcb/`; 235 ảnh toàn board đang dùng đều là ảnh công khai.
