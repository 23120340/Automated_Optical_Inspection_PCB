# Train lại detector (bước 4) và classifier (bước 6.1)

Hai notebook v2:

- [pcb_detector_v2_kaggle.ipynb](pcb_detector_v2_kaggle.ipynb) — detector thân linh kiện **và** chân/pad
- [pcb_classifier_v2_kaggle.ipynb](pcb_classifier_v2_kaggle.ipynb) — phân loại family, ConvNeXt-Base

Nguồn percent-format là hai file `.py` cùng tên; sửa `.py` rồi chạy
`python scripts/build_notebook.py --all`.

---

## Trước tiên: kết luận khảo sát dataset

Bạn muốn detector nhận cả chân linh kiện. Tôi đã tra cứu, và đây là câu trả lời
thẳng:

**Không có dataset công khai nào gán nhãn chân/pad linh kiện thành object riêng.**

| Dataset | Thực tế gán nhãn gì |
|---|---|
| [FPIC / FICS-PCB](https://www.trust-hub.org/#/data/pcb-images) (261 ảnh, 93 board, Nikon D850, **CC BY 4.0**) | Contour linh kiện SMD + text/silkscreen. Bài báo nói thẳng: *"Future releases will include annotations of vias, traces, and pins"* — tức **bản hiện tại KHÔNG có** |
| FPIC-Component (bản dẫn xuất, 6260 ảnh, 25 class) | Chỉ mức linh kiện. Và license là **CC BY-NC-ND 4.0** — *NonCommercial* **và** *NoDerivatives*, train model trên đó là rủi ro pháp lý thật |
| [SolDef_AI](https://www.kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection) | Một polygon bao **cả linh kiện lẫn pad**, nhãn nhị phân đúng/sai vị trí — không phải box từng pad |
| [RF100 printed-circuit-board](https://universe.roboflow.com/roboflow-100/printed-circuit-board) | Có class Pads/Pins, nhưng class list gần trùng khớp dataset bạn đang dùng ⇒ nhiều khả năng chính là nguồn gốc của nó |

Nên notebook detector không hứa điều nó không làm được. Nó tấn công vào chỗ thật
sự sửa được: **mất cân bằng cực đoan**.

---

## Trước tiên: chọn đúng accelerator

**Không dùng GPU P100.** Wheel PyTorch trên Kaggle đã bỏ kernel cho `sm_60`, nên
P100 sẽ chạy bình thường tới lúc model được đẩy lên GPU rồi nổ:

```
AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

Traceback rơi sâu trong `trainer._setup_train()` nên trông như lỗi Ultralytics,
nhưng nguyên nhân chỉ là lựa chọn accelerator.

**Chọn: GPU T4 x2** (hoặc L4 / A100). Cả hai notebook v2 có cell preflight kiểm
tra ngay từ đầu và dừng kèm hướng dẫn, thay vì để bạn tốn thời gian setup rồi mới
biết.

## Detector v2

### Vấn đề là dữ liệu, không phải kiến trúc

Model hiện tại của bạn:

| Class | Instance train | Recall val | mAP50-95 |
|---|---:|---:|---:|
| capacitor | 7775 | 0.654 | 0.375 |
| resistor | 7133 | 0.446 | 0.307 |
| ic | 2220 | 0.836 | 0.514 |
| **pins** | **261** | **0.145** | 0.106 |
| **pads** | **186** | **0.000** | 0.0019 |

`pads` có ít hơn capacitor **42 lần** và mỗi pad chỉ vài chục pixel. **Đổi sang
RT-DETR, D-FINE hay YOLOv12 không sửa được 186 instance.**

Thêm nữa, YOLO26 đã có sẵn **STAL** (small-target-aware label assignment): ép tối
thiểu 4 anchor cho object nhỏ hơn 8 px, và tách hình học chọn ứng viên khỏi hình
học hồi quy. Đó đúng là cơ chế cho bài toán này. Nên khuyến nghị: **giữ YOLO26**.

### Notebook làm gì khác v1

| Thay đổi | Vì sao |
|---|---|
| imgsz 1280 → **1536** | Pad vài chục pixel ở 1280, sau stride 8 chỉ còn 2–3 pixel đặc trưng |
| **Oversample ảnh chứa pads/pins** | Ultralytics không có sampler theo class; cách duy nhất tác động được là nhân bản đường dẫn trong train list |
| **Trần oversample 35%** | Nhân bản quá tay thì model học thuộc đúng vài ảnh đó. Notebook tự hạ hệ số và in ra tỉ lệ thực tế |
| `copy_paste` 0.30 | Dán object hiếm sang ảnh khác — đòn bẩy trực tiếp nhất cho mất cân bằng mức instance |
| epochs 100 → **150**, `close_mosaic` 10 → **25** | Class hiếm cần nhiều epoch có mosaic mới gặp đủ biến thể |
| **Cổng verdict cuối notebook** | So recall pads/pins với baseline. Không đạt thì nói thẳng là dữ liệu, kèm 3 phương án |

### Nếu recall pads/pins vẫn kẹt

Notebook sẽ in ra, nhưng tóm tắt ở đây theo thứ tự chi phí:

1. **Giữ ROI suy ra** (rẻ nhất, đã chạy được). Bước 5.5 đã suy ROI từ box thân +
   topology chân. Detector yếu ở pads/pins **không chặn** bước 5.5.
2. **Gán nhãn bootstrap** (hiệu quả nhất). Chạy
   `scripts/export_solder_dataset.py --overlays` — nó sinh sẵn ROI chân ứng viên.
   Người chỉ **sửa** box thay vì vẽ từ đầu. Vài trăm board là vượt xa 186 instance.
3. **Thêm dataset** (không chắc ăn). Kiểm tra trùng lặp với RF100 trước.

---

## Classifier v2

### Backbone: tối ưu cho độ chính xác

Không còn ràng buộc Raspberry Pi, nên chọn theo accuracy thuần. Đo được trên cùng
dữ liệu, cùng số epoch:

| Backbone | Params | TEST macro recall |
|---|---:|---:|
| EfficientNetV2-S | 20.2M | 0.883 |
| **ConvNeXt-Base** ✅ mặc định | 87.6M | **0.929** |
| ConvNeXt-Base + TTA | — | **0.942** |

Notebook hỗ trợ 8 backbone qua `CONFIG["model_name"]`: `convnext_base`,
`convnext_small`, `convnext_tiny`, `efficientnet_v2_m`, `efficientnet_v2_s`,
`efficientnet_b0`, `swin_t`, `mobilenet_v3_small`. Đã kiểm cả 8 dựng được,
forward đúng, và cho 9–14 param group phân tầng.

**Cảnh báo:** model to hơn ≠ chính xác hơn khi dữ liệu ít. Trong khảo sát 7
dataset, **ConvNeXt-Tiny thắng mọi model khác ở hầu hết dataset ảnh tự nhiên**.
Với vài nghìn crop, Base (87.6M) rất dễ overfit. Notebook ghi `best_macro_recall`
vào manifest nên **so được giữa các lần chạy** — chạy Base/Small/Tiny rồi chọn
theo số đo, đừng chọn theo tên.

Hai đòn bẩy khác đã bật sẵn: **input 288px** (crop nhỏ, fine-grained nên độ phân
giải giúp nhiều) và **TTA khi đánh giá** (+1.25 điểm đo được).

### Công thức train

| Kỹ thuật | Vì sao |
|---|---|
| **Chia theo ảnh cha** | Nhiều crop từ cùng một board; chia theo crop là rò rỉ và cho điểm ảo |
| **Freeze head 3 epoch rồi mở khoá** | Fine-tune toàn bộ từ epoch 1 phá weight pretrain |
| **Layer-wise LR decay 0.80** | Tầng gần input mang đặc trưng tổng quát; LR lớn ở đó là vứt bỏ pretrain |
| **RandAugment + Mixup/CutMix** | Nhóm augmentation tác động lớn nhất với fine-grained ít dữ liệu |
| **EMA weight** | Gần như luôn +0.5–1.5%, không tốn gì |
| **Class-balanced sampler** | Không cân thì class hiếm bị capacitor/resistor nhấn chìm |
| **Macro recall để chọn model** | Accuracy tổng bị class đa số che; macro recall không |
| **Calibration tách riêng** | Ngưỡng đo trên tập đã dùng chọn model luôn lạc quan |
| **Cảnh báo temperature chạm biên** | Chạm mép dải quét nghĩa là tối ưu nằm ngoài dải; ghi số đó vào manifest là làm lệch mọi xác suất sau này |

### Ràng buộc phải giữ

Notebook cắt crop theo **đúng công thức app dùng**: `pad = 0.15 * max(w,h)`, cắt
theo biên, **không ép vuông**. Sửa `CropConfig` trong app mà không train lại là
lệch phân bố đầu vào — biểu hiện thành accuracy tụt không rõ lý do.

`pads`/`pins` bị loại khỏi taxonomy family: chúng là vùng hàn, không phải loại
linh kiện. Nhầm một pad thành resistor tệ hơn là biết đó không phải linh kiện.

---

## Sau khi tải về

```powershell
# detector -> models/detector/
# classifier -> nạp ở sidebar "Model phân loại 6.1"

# classifier có thể kiểm tra bằng cùng công cụ của 6.2? Không — schema khác.
# Nhưng cả hai manifest đều được app kiểm tra khi nạp, và từ chối nếu sai.
```

Sau khi nạp detector mới, chạy một board và xem cảnh báo:

> `Bước 5.5: dùng N ROI từ detection chân/pad thật và M ROI suy ra.`

**N > 0** nghĩa là detector mới thực sự đang đóng góp ROI chân. **N = 0** nghĩa là
pads/pins vẫn chưa học được, và pipeline đang chạy hoàn toàn bằng ROI suy ra —
vẫn hoạt động, chỉ là detector chưa giúp được gì cho phần này.

## Cách kết hợp thuật toán và model ở bước 5.5

Đã cài trong [`aoi_pipeline/inspection/leads.py`](../../aoi_pipeline/inspection/leads.py).
Quy tắc: **ưu tiên detection thật, quay về hình học suy ra ở chỗ không có** — và
điều quan trọng là chọn **theo từng chân, không theo từng linh kiện**:

| Tình huống | ROI dùng |
|---|---|
| Detector tìm được chân, chồng lên ROI suy ra | Dùng box detect (`source=detected`) |
| Detector tìm được chân ở vị trí ROI suy ra không đoán (thermal pad…) | Thêm vào, giữ nguyên ROI suy ra |
| Detector tìm được 1 trong 2 đầu | Đầu tìm được dùng detect, **đầu còn lại vẫn giữ ROI suy ra** |
| Detector không tìm được gì | Toàn bộ dùng ROI suy ra |
| Detection chân nằm quá xa mọi linh kiện | **Bỏ và báo**, không gán bừa cho linh kiện gần nhất |

Dòng thứ ba là dòng quan trọng nhất: chuyển cả linh kiện sang "detected" khi model
chỉ thấy một đầu sẽ **âm thầm mất đầu kia** — mà đó thường đúng là đầu có lỗi.
