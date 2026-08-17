# Automated Optical Inspection PCB

Ứng dụng local thử nghiệm luồng AOI từ bước 0 đến bước 5:

```text
0. Import ảnh
   → 1. Tiền xử lý
   → 2. Căn chỉnh với Golden Image (tùy chọn)
   → 3. Khoanh vùng PCB
   → 4. Phát hiện linh kiện
   → 5. Crop và xuất dữ liệu linh kiện
```

Hiện bước 0 dùng upload ảnh để có thể phát triển khi chưa gắn camera. Adapter
camera/RTSP sẽ được thêm sau mà không thay đổi pipeline xử lý.

## Chạy ứng dụng trên Windows

Yêu cầu: Python 3.12 đã được cài qua Python Launcher (`py`).

```powershell
cd D:\repos\Internship\Automated_Optical_Inspection_PCB
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup.ps1
.\scripts\run_app.ps1
```

Sau đó mở địa chỉ Streamlit hiển thị trong terminal, mặc định là
`http://127.0.0.1:8501`.

Chạy test:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Trạng thái bước 4

Khi chưa có model, app dùng OpenCV để đề xuất các vùng `component_candidate`.
Chế độ này chỉ giúp thử luồng UI/crop/export; nó **không nhận dạng loại linh kiện
đáng tin cậy** và được đánh dấu `CV DEMO` trong giao diện.

Để train detector thật trên Kaggle:

1. Đọc [hướng dẫn Kaggle](training/kaggle/README.md).
2. Import notebook
   [pcb_component_detection_kaggle.ipynb](training/kaggle/pcb_component_detection_kaggle.ipynb).
3. Gắn dataset YOLO, bật GPU và chạy `Run All`.
4. Tải `/kaggle/working/pcb_component_detector_artifacts.zip`.
5. Gửi lại bundle đó cùng 10–20 ảnh PCB test chưa dùng trong training.

Notebook tạo `best.pt`, `best.onnx`, class map, manifest SHA-256, dataset audit,
metric theo class, confusion matrix và ảnh prediction. App hỗ trợ model
Ultralytics `.pt`/`.onnx` khi đã cài dependency model tùy chọn
(không cần cài để chạy `CV DEMO`):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-model.txt
```

Chỉ nạp `.pt` từ nguồn tin cậy. Với triển khai inference, ưu tiên `.onnx` cùng
manifest/class map. Giao diện yêu cầu xác nhận tin cậy trước khi thực thi `.pt`.

Notebook của dự án đánh giá và export YOLO26 với head one-to-many
(`end2end=False`) để dùng NMS/IoU nhất quán. UI local ghim cùng chế độ này; khi
dùng `UltralyticsDetector` như một thư viện với model khác, mặc định `end2end=None`
sẽ giữ nguyên head do artifact khai báo.

Manifest và CSV export ghi rõ `coordinate_space`, kích thước ảnh analysis và quy
ước box. Tọa độ bước 3–5 thuộc ảnh preprocessed/aligned được export kèm gói, không
mặc định thuộc ảnh input gốc nếu pipeline đã resize hoặc warp.

Tài liệu đã chuẩn bị cho các bước tiếp theo:

- [Khảo sát dataset linh kiện PCB](Docs/pcb_aoi_component_datasets.md).
- [Kế hoạch pre-train cho bước 6.1](Docs/ke_hoach_pretrain_6_1_classification.md).

## Cấu trúc dự án

```text
app/                 Streamlit UI và bridge
aoi_pipeline/        Pipeline OpenCV/model cho bước 0–5
tests/               Unit tests
training/kaggle/     Notebook train detector bước 4
models/              Nơi đặt model local (weights không commit Git)
Docs/                Khảo sát dataset và kế hoạch pre-train 6.1
scripts/             Setup/chạy app trên Windows
```

## Giới hạn hiện tại

- Khoanh PCB ở bước 3 đang dùng contour fallback, chưa có PCB detector riêng.
- CV proposal ở bước 4 không thay thế model đã train.
- Baseline dùng `max_det=2000` cho board dày linh kiện; cần tune lại theo SKU/tốc độ.
- Căn chỉnh chính xác cần một Golden Image/reference cùng board side.
- Model cuối phải được đánh giá trên camera, lens, ánh sáng và PCB của dây chuyền.
- Mỗi ảnh import được giới hạn 64 MB/50 MP; upload Streamlit tối đa 256 MB/file.
- Ultralytics công bố lựa chọn AGPL-3.0/Enterprise; cần duyệt license trước khi
  đưa framework/model vào sản phẩm thương mại hoặc phần mềm đóng nguồn.
