# Huấn luyện bước 6.1 trên Kaggle: phân loại family linh kiện

Notebook [pcb_component_classification_kaggle.ipynb](./pcb_component_classification_kaggle.ipynb)
nhận annotation YOLO detect, cắt từng bounding box thành ảnh phân loại, fine-tune
EfficientNet-B0, hiệu chỉnh confidence và export ONNX cho app local. File
[pcb_component_classification_kaggle.py](./pcb_component_classification_kaggle.py) là nguồn
percent-format của notebook; chạy `scripts/build_classification_notebook.py` sau khi sửa nguồn.

## Dataset đã chọn

Notebook đã cấu hình sẵn, không dùng đường dẫn placeholder:

- Kaggle slug: `aryanstein/pcb-component-detection-consolidated-dataset`.
- Version preset đã kiểm tra: **1**.
- Kaggle Input: `/kaggle/input/pcb-component-detection-consolidated-dataset`.
- YOLO YAML ưu tiên: `components_data_uncropped/data.yaml`.
- Trang dữ liệu: [PCB Component Detection Consolidated Dataset](https://www.kaggle.com/datasets/aryanstein/pcb-component-detection-consolidated-dataset/data).

Trong Kaggle chọn **Add Input**, tìm đúng slug trên, bật GPU và bật Internet lần
đầu để tải ImageNet weights. Cell tìm dataset thử đường dẫn đầy đủ trước; nếu
Kaggle đổi tên mount, nó chỉ chọn YAML YOLO có class map phù hợp và báo rõ khi có
nhiều YAML mơ hồ.

### Lỗi `no kernel image is available` trên Tesla P100

Nếu weights 20,5 MB đã tải xong nhưng lần gọi model đầu tiên báo lỗi trên, nguyên
nhân là GPU P100 có compute capability `sm_60` trong khi PyTorch wheel của phiên
Kaggle không chứa kernel tương ứng. Đây không phải lỗi EfficientNet hay dataset.

Notebook chạy một CUDA probe ngay ở phần khởi tạo và dừng sớm với tên GPU,
capability và danh sách kiến trúc mà PyTorch hỗ trợ. Trong Kaggle, vào **Settings >
Accelerator**, chọn **GPU T4 x2** hoặc GPU mới hơn, lưu cấu hình, **restart session**
rồi Run All. Chạy lại riêng cell train trong chính phiên P100 không giải quyết được
lỗi. Notebook không tự hạ phiên bản PyTorch vì việc đó có thể làm lệch cặp
Torch/TorchVision và các dependency export ONNX; CPU chỉ được phép khi chủ động đặt
`CONFIG["allow_cpu_training"] = True`.

### Lỗi `Inference tensors cannot be saved for backward` ở cell calibration

Temperature scaling cần gradient theo biến nhiệt độ dù model inference không cần
gradient. Notebook dùng `torch.no_grad()` cho hàm `predict` và clone logits thành
tensor thường trước khi chạy LBFGS. Nếu lỗi xuất hiện trong notebook cũ đã chạy
cell train, dùng notebook mới rồi chạy lại cell định nghĩa `predict` và cell
calibration; không cần train lại nếu `best.pt` vẫn còn trong phiên Kaggle.

Một số bản mount của bộ này đặt `data.yaml` trong `components_data_uncropped`
nhưng vẫn ghi split là `../train/images`. Resolver không ghép cứng chuỗi đó: nó
thử cả `components_data_uncropped/train/images`, thư mục split ở gốc dataset và
hai cấu trúc YOLO phổ biến `train/images`/`images/train`. Khi lỗi, thông báo liệt
kê toàn bộ đường dẫn đã thử để phân biệt mount sai với dataset thật sự thiếu.

Kaggle uploader khai báo Apache 2.0. Vì đây là bộ hợp nhất nhiều nguồn, vẫn phải
kiểm tra license/provenance của từng nguồn thành phần trước khi dùng thương mại.

## Taxonomy baseline

22 nhãn detector nguồn được gom thành 17 family ngoại quan:

```text
acoustic, battery_power_input, capacitor, connector, diode,
discrete_semiconductor, display, false_crop_background, ic, led, magnetic,
protection, relay, resistor, switch_control, thermal_mechanical, timing
```

Các quy tắc đáng chú ý:

- `button`, `switch`, `potentiometer` → `switch_control`.
- `inductor`, `transformer` → `magnetic`.
- `pads`, `pins` → `false_crop_background` để hỗ trợ reject crop sai, không coi
  là family linh kiện dương.
- `transducer` bị bỏ qua ở baseline vì preset không có train support đủ tin cậy.
- Không suy đoán chức năng/part number IC chỉ từ hình package; phần đó dành cho
  OCR + BOM matching ở các bước sau.

Mapping đầy đủ nằm trong cell `CONFIG` và được ghi lại vào manifest. Không đổi
mapping giữa train và runtime.

## Backbone dành cho Raspberry Pi

Classifier mặc định là **EfficientNet-B0 pretrained ImageNet, input 224×224**.
Theo TorchVision, model có khoảng 5,29 triệu tham số, 0,39 GFLOPs và file weights
20,5 MB. So với EfficientNetV2-S trước đó (21,46 triệu tham số, 8,37 GFLOPs), B0
giảm khoảng 4 lần số tham số và hơn 21 lần lượng tính toán danh nghĩa. Đây là lựa
chọn cân bằng theo yêu cầu: ưu tiên độ chính xác, không cần real-time, nhưng vẫn
có thể triển khai ONNX Runtime trên Raspberry Pi ARM64.

Không dùng MobileNetV3-Small làm mặc định vì nó ưu tiên latency mạnh hơn và có
năng lực biểu diễn thấp hơn. MobileNetV3-Large vẫn là ứng viên dự phòng nếu đo
thực tế trên Pi cho thấy EfficientNet-B0 quá chậm; quyết định cuối phải dựa trên
macro-F1 của chính dataset PCB và benchmark latency/RAM trên đúng phiên bản Pi.

Artifact mặc định giữ FP32. Chỉ chuyển INT8 sau khi calibration bằng crop camera
thật và xác nhận macro-F1/accepted precision không giảm quá gate của dự án.

## Các gate notebook thực hiện

- Audit YAML, class ID, 5 cột YOLO, tọa độ hữu hạn và ảnh hỏng.
- Clip box partial-object ở biên; dòng sai thật được ghi vào `invalid_labels.csv`
  và chặn train khi `strict_audit=True`.
- Bỏ dòng annotation trùng và ảnh trùng byte chéo split theo ưu tiên
  `train > val > test` mà không sửa Kaggle Input.
- Chia validation gốc thành `val` và `calibration` theo ảnh cha, nên crop của cùng
  ảnh không lọt sang cả hai phía. Test gốc chỉ dùng đánh giá cuối.
- Theo dõi macro-F1, metric từng class, confusion matrix, temperature scaling và
  precision/coverage của confidence gate.
- Kiểm tra ONNX bằng `onnx.checker`, ONNX Runtime CPU và parity với PyTorch.

Reject hiện tại dựa trên `1 - max_probability`; đây chưa phải đánh giá OOD đầy
đủ. Trước production phải bổ sung crop ngoài taxonomy và ảnh từ camera/đèn thật.

## Artifact cần đưa vào app

Sau `Run All`, tải:

```text
/kaggle/working/pcb_component_classifier_artifacts.zip
```

Giải nén và nạp cùng lúc trong sidebar **Model phân loại 6.1**:

1. `best.onnx` — model raw-logit, dynamic batch.
2. `model_manifest.json` — class order, RGB/letterbox/ImageNet normalization,
   temperature, ngưỡng accept/review, dataset và SHA-256.

App không dùng nhãn detector thay classifier và không tự tạo kết quả khi thiếu
một trong hai file. `best.pt` chỉ để resume/debug trong môi trường tin cậy; runtime
bước 6.1 chủ động không nạp pickle.
