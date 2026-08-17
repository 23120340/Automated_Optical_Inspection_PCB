# Huấn luyện bước 4 trên Kaggle: phát hiện linh kiện PCB

Notebook [`pcb_component_detection_kaggle.ipynb`](./pcb_component_detection_kaggle.ipynb) huấn luyện detector của bước 4 trong sơ đồ AOI. Nó nhận dataset object detection theo định dạng YOLO, kiểm tra dữ liệu trước khi dùng GPU, fine-tune model pretrained, đánh giá, trực quan hóa và tạo một gói kết quả để đưa về app local.

## Kết quả notebook tạo ra

Sau khi chạy hết notebook, tải file:

```text
/kaggle/working/pcb_component_detector_artifacts.zip
```

Gói này chứa tối thiểu:

- `best.onnx`: artifact inference ưu tiên; fixed batch 1/fixed `imgsz`, raw one-to-many output, không nhúng NMS.
- `best.pt`: checkpoint Ultralytics để debug/fallback trong môi trường tin cậy.
- `model_manifest.json`: cấu hình train, phiên bản thư viện, class map, metric và SHA-256.
- `metrics_summary.json` và `per_class_metrics.csv`.
- `dataset_audit.json`, `class_distribution.csv` và `invalid_labels.csv` nếu có.
- `data_resolved.yaml`: cấu hình dataset với đường dẫn đã được notebook chuẩn hóa.
- `onnx_verification.json`: kết quả ONNX checker, input/output graph và smoke-test CPU.
- `results.csv`, các confusion matrix/PR curve và ảnh dự đoán do Ultralytics sinh ra nếu có.

## 1. Chuẩn bị dataset

Notebook dùng đúng định dạng YOLO detect chính thức:

```text
pcb-dataset/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/                 # không bắt buộc
└── labels/
    ├── train/
    ├── val/
    └── test/                 # không bắt buộc
```

Mỗi ảnh có một file `.txt` cùng stem. Mỗi dòng nhãn là:

```text
class_id x_center y_center width height
```

`class_id` bắt đầu từ 0; bốn tọa độ là `xywh` đã chuẩn hóa về khoảng 0–1. Ảnh không có đối tượng (negative image) nên có file `.txt` cùng stem nhưng rỗng. Mặc định notebook coi file label bị thiếu là lỗi để tránh biến ảnh chưa gán nhãn thành false negative. Ví dụ `data.yaml`:

```yaml
path: .
train: images/train
val: images/val
test: images/test
names:
  0: resistor
  1: capacitor
  2: ic
  3: diode
  4: inductor
  5: connector
```

Khuyến nghị chia dữ liệu theo **board/SKU/lot**, không random các crop hoặc ảnh gần giống của cùng một board vào nhiều split. Notebook chỉ tìm được ảnh trùng byte hoàn toàn; nó không thể tự phát hiện mọi trường hợp leakage do ảnh chụp gần giống.

### Hai cách đưa dữ liệu vào Kaggle

**Cách A — Kaggle Dataset (khuyến nghị)**

1. Nén cả thư mục dataset thành `.zip`, hoặc upload cấu trúc thư mục trực tiếp khi tạo Kaggle Dataset.
2. Tạo Notebook, chọn **Add Input**, gắn dataset vừa tạo.
3. Import notebook `.ipynb` trong thư mục này.
4. Giữ khóa `"dataset_source": None` trong dict `CONFIG` để tự tìm. Nếu có nhiều dataset/YAML, điền đường dẫn rõ ràng, ví dụ:

   ```python
   "dataset_source": "/kaggle/input/pcb-components-yolo/pcb_dataset.zip",
   "data_yaml": None,
   ```

**Cách B — một file ZIP trong Kaggle Input**

Gắn Dataset chứa file ZIP rồi đặt `dataset_source` tới file đó. Notebook chặn path traversal và mặc định không giải nén quá 12 GB để chừa dung lượng cho checkpoint/report trong `/kaggle/working`. Chỉ tăng `max_extract_gb` sau khi kiểm tra quota của session.

Kaggle Input là vùng chỉ đọc; model và báo cáo được ghi vào `/kaggle/working`. Baseline đặt `cache=False`; nếu dataset được đọc trực tiếp từ Input thì Ultralytics có thể bỏ qua file cache do vùng này read-only.

## 2. Cấu hình Kaggle

Trong **Notebook options**:

- Accelerator: chọn **GPU**.
- Internet: bật trong lần đầu để cài/đồng bộ Ultralytics và tải checkpoint pretrained. Nếu tổ chức không cho bật Internet, hãy gắn wheel/package và checkpoint `yolo26s.pt` như Kaggle Input rồi sửa cell cài đặt/model path.
- Chạy `Run All` sau khi chỉnh duy nhất cell `CONFIG`.

Notebook pin `ultralytics==8.4.104` (bản stable đã đối chiếu khi tạo notebook) và ghi lại toàn bộ phiên bản runtime trong manifest. Chỉ nâng phiên bản sau khi đã chạy lại export + smoke-test vì output ONNX/API có thể thay đổi.

Baseline đã đặt:

```python
"model_family": "yolo26",
"model": "yolo26s.pt",
"imgsz": 1280,
"epochs": 100,
"batch": -1,
"patience": 25,
"save_period": 10,
"end2end": False,
"max_det": 2000,
"allow_negative_images": False,
"max_missing_label_ratio": 0.0,
```

`YOLO26s` là baseline PoC; `1280` giữ nhiều chi tiết hơn cho linh kiện nhỏ. Nếu hết VRAM, giảm `imgsz` xuống `960` hoặc dùng `yolo26n.pt`. `batch=-1` để Ultralytics tự chọn batch theo VRAM. `save_period=10` tạo checkpoint định kỳ phòng khi session bị gián đoạn.

`model_family` là metadata tường minh; notebook còn kiểm tra head thật có `one2one`/`one2many`, vì checkpoint có thể mang tên tùy ý như `/kaggle/input/.../best.pt`. Training loss, validation nội bộ/early stopping, final val/test, prediction và export đều dùng one-to-many (`end2end=False`) và `max_det=2000`. Mức 2000 tránh truncate board AOI dày có thể vượt 1000 linh kiện; chỉ giảm khi đã biết chắc mật độ board và đã đo RAM/thời gian NMS.

Với Ultralytics 8.4.104, notebook dùng một subclass nhỏ của `DetectionTrainer` để đặt head ngay khi trainer rebuild model, trước loss/EMA/validator. Lý do: public argument `end2end=False` được ghi vào trainer config nhưng không tự đổi YOLO26 head trong version đã pin. Post-train assert dừng notebook nếu trainer args, model head hoặc `best.pt` lệch contract.

Không đổi taxonomy giữa các lần train nếu app đã bắt đầu dùng model.

## 3. Các gate kiểm tra dữ liệu

Notebook audit:

- YAML có `train`, `val`, `names` hợp lệ.
- Split/list phải resolve được; mọi path ảnh trong file list phải tồn tại và có extension được hỗ trợ.
- Mỗi ảnh phải có label; file `.txt` rỗng là negative image tường minh; nhãn không rỗng phải có đúng 5 cột.
- Class ID là số nguyên, bắt đầu từ 0 và nằm trong class map.
- Tọa độ hữu hạn, tâm nằm trong 0–1, chiều rộng/cao lớn hơn 0 và box không vượt biên.
- Dòng annotation bị lặp.
- Ảnh trùng byte giữa train/val/test.
- Phân bố class, class không xuất hiện, ảnh âm tính và tỷ lệ box rất nhỏ.
- Giải mã thử ảnh để phát hiện file hỏng.

`strict_audit=True` khiến notebook dừng trước train khi có lỗi nghiêm trọng. Hãy sửa dataset; không nên chuyển sang `False` chỉ để ép chạy.

Chỉ khi dataset quy ước rõ rằng “không có file label = negative”, đặt `allow_negative_images=True` và chọn `max_missing_label_ratio` dương phù hợp. Notebook vẫn dừng nếu tỷ lệ thiếu vượt ngưỡng. Cách an toàn hơn là tạo file `.txt` rỗng.

Audit ước tính cạnh box theo `imgsz` và cảnh báo box dưới 8 px. Nếu ảnh nguyên board làm linh kiện quá nhỏ, 1280 px không tự giải quyết được; hãy bổ sung crop/tiling có overlap và chia split theo board gốc để tránh leakage.

## 4. Contract ONNX

Baseline export `best.onnx` với batch 1, input vuông cố định theo `imgsz`, one-to-many raw output và `nms=False`. App có thể dùng `YOLO("best.onnx", task="detect")` để wrapper thực hiện letterbox/postprocess/NMS, hoặc tự cài cùng contract bên runtime ONNX. Không coi tensor output là box cuối cùng khi chưa decode/NMS.

Notebook chạy `onnx.checker`, khởi tạo ONNX Runtime CPU, assert shape input và smoke-test một ảnh validation. Chi tiết nằm trong `onnx_verification.json`.

## 5. Những file cần gửi lại để tích hợp bước 4

Gửi file `pcb_component_detector_artifacts.zip`. Nếu giới hạn dung lượng, gửi tối thiểu:

1. `best.onnx` (ưu tiên cho inference an toàn hơn) và `best.pt` nếu cần debug/tích
   hợp Ultralytics Python. App hiện hỗ trợ độc lập từng định dạng.
2. `model_manifest.json`, `metrics_summary.json`, `per_class_metrics.csv`.
3. `dataset_audit.json`, `data_resolved.yaml` và `class_distribution.csv`.
4. Khoảng 10–20 ảnh PCB nguyên bản chưa dùng trong train, ưu tiên ảnh giống camera AOI thực tế.
5. Nguồn/license của dataset và mô tả cách chia train/val/test.

Không cần gửi toàn bộ dataset nếu không có quyền chia sẻ. Tuy nhiên phải gửi class map chính xác và vài ảnh test thực tế để kiểm tra mapping box từ bước 4 sang bước 5.

Notebook đặt `end2end=False`; train/early stopping, validation, prediction và ONNX export đều dùng head one-to-many. UI local phải áp dụng NMS cùng `iou`/`max_det` trong `model_manifest.json`.

## 6. Diễn giải metric

- Gate ban đầu nên theo dõi `mAP50-95`, precision và recall; không chỉ nhìn `mAP50`.
- Xem metric theo từng class. Macro score đẹp vẫn có thể che class hiếm bị bỏ sót.
- Với AOI, recall của linh kiện nhỏ và class quan trọng thường đáng ưu tiên hơn precision tổng.
- Đánh giá cuối phải chạy trên ảnh từ camera/dàn sáng thật; metric public dataset chỉ là baseline.

Notebook không huấn luyện định danh như `MCU`, `PMIC`, `ADC` chỉ từ hình dáng package. Bước 4 nên dùng lớp hình thái ổn định (`ic`, `resistor`, `capacitor`, `diode`, `connector`, ...); mã linh kiện cụ thể được xử lý tiếp ở OCR + BOM (6.4–6.5).

## Tài liệu chính thức đã đối chiếu

- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/): model hiện hành, checkpoint, head end-to-end và hỗ trợ vật thể nhỏ.
- [Ultralytics Detect](https://docs.ultralytics.com/tasks/detect/): Python API cho train, val và export.
- [Ultralytics dataset format](https://docs.ultralytics.com/datasets/detect/): cấu trúc YAML và nhãn normalized `xywh`.
- [Ultralytics validation](https://docs.ultralytics.com/modes/val/): `mAP50`, `mAP75`, `mAP50-95` và metric theo class.
- [Ultralytics configuration](https://docs.ultralytics.com/usage/cfg/): `batch=-1`, `imgsz`, `patience`, seed và tham số train.
- [Ultralytics installation](https://docs.ultralytics.com/quickstart/): cài stable package qua pip.
- [DetectionTrainer source v8.4.104](https://github.com/ultralytics/ultralytics/blob/v8.4.104/ultralytics/models/yolo/detect/train.py): `get_model`, `set_model_attributes` và validator được đối chiếu cho one-to-many trainer contract.
- [BaseTrainer source v8.4.104](https://github.com/ultralytics/ultralytics/blob/v8.4.104/ultralytics/engine/trainer.py): thứ tự rebuild model, loss/EMA/validation và checkpoint selection.
- [Kaggle Python image](https://github.com/Kaggle/docker-python): môi trường notebook/GPU chính thức.

> Lưu ý license: tài liệu Ultralytics hiện công bố lựa chọn AGPL-3.0 và Enterprise. Trước khi đưa model/framework vào sản phẩm thương mại hoặc phần mềm đóng nguồn, cần để bộ phận phụ trách xác nhận phương án license phù hợp. License dataset cũng phải được kiểm tra độc lập.
