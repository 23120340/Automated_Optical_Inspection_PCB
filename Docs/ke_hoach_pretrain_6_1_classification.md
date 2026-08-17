# Kế hoạch pre-train bước 6.1 — Phân loại linh kiện PCB

> Trạng thái triển khai: khung runtime ONNX + manifest và notebook baseline đã
> được tạo. Xem [hướng dẫn Kaggle](../training/kaggle/README_classification.md).

> Model trong sơ đồ: `03_classify_component`  
> Phiên bản kế hoạch: `v1.0-draft` — 2026-08-16  
> Tài liệu dữ liệu liên quan: [Khảo sát dataset PCB AOI](./pcb_aoi_component_datasets.md)

## 1. Quyết định đề xuất

Xây bước 6.1 thành một **bộ phân loại phân cấp có quyền từ chối**, nhận crop từ bước 5 và trả về:

- họ linh kiện theo hình thái (`family`);
- subtype nhìn thấy được, nếu dữ liệu đủ (`visual_subtype`);
- kiểu lắp (`mount_type`: SMD/THT/module/mechanical);
- top-k, xác suất đã calibration và `unknown/reject`;
- embedding để tìm ảnh tương tự và điều tra lỗi.

Không giao cho 6.1 nhiệm vụ đoán chức năng điện tử không thể quan sát chắc chắn. Ví dụ MCU, op-amp, ADC, PMIC, EEPROM hay CAN transceiver đều có thể dùng cùng package IC. Các nhãn đó phải được xác minh ở bước 6.4–6.5 bằng **OCR top-marking + BOM/centroid/part database**.

Lựa chọn model ban đầu:

| Vai trò | Model đề xuất | Mục đích |
|---|---|---|
| Baseline bắt buộc/deployment | **EfficientNet-B0**, pretrained ImageNet, 224 px | Cân bằng accuracy và Raspberry Pi: khoảng 5,3M tham số, 0,39 GFLOPs, weights 20,5 MB; xuất ONNX đơn giản. |
| Đối chứng ARM nhẹ hơn | **MobileNetV3-Large**, 224 px | Dùng khi benchmark thật trên Pi cho thấy B0 chưa đạt latency/RAM; không chọn chỉ vì nhanh hơn. |
| Upper-bound offline, không phải deployment mặc định | **EfficientNetV2-S**, 320/384 px | Chỉ đo xem backbone lớn cải thiện macro-F1 bao nhiêu; khoảng 21,5M tham số và 8,37 GFLOPs nên quá nặng cho Raspberry Pi nếu lợi ích nhỏ. |
| Teacher/R&D | **DINOv3 ViT-S/16**: linear probe rồi partial fine-tune | Representation mạnh, phù hợp thử nghiệm ít nhãn và unknown bằng embedding. Phải duyệt license DINOv3 trước khi dùng sản phẩm. |
| Đối chứng kiến trúc | **ConvNeXt-Tiny** | Xác nhận kết quả không phụ thuộc riêng EfficientNet. |
| Model triển khai | EfficientNet-B0 hoặc model nhỏ thắng benchmark | Chọn theo macro-F1/accepted precision trước, sau đó latency/RAM trên đúng Raspberry Pi; dự án không yêu cầu real-time. |

Nguồn kỹ thuật: [TorchVision EfficientNet-B0](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html), [TorchVision MobileNetV3-Large](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_large.html), [TorchVision EfficientNetV2-S](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_s.html), [ONNX Runtime trên Raspberry Pi](https://onnxruntime.ai/docs/tutorials/iot-edge/rasp-pi-cv.html), [DINOv3 paper](https://arxiv.org/abs/2508.10104), [mã và license DINOv3](https://github.com/facebookresearch/dinov3).

## 2. Phạm vi và hợp đồng với hệ thống

### 2.1 Giả định

- Ảnh RGB top-view đã qua resize/white balance/căn chỉnh PCB ở bước 1–3.
- Bước 4 trả bbox hoặc polygon linh kiện; bước 5 crop từ ảnh gốc/ảnh đã căn chỉnh với đủ độ phân giải.
- Chưa có yêu cầu cứng về GPU, latency và số SKU. Vì vậy latency trong kế hoạch là **metric phải đo**, không đặt số ms tùy ý.
- Model 6.1 phân loại **linh kiện hiện diện**. `missing component` thuộc bước 6.2/6.8; crop trống được nhận biết như `false_crop/background`, không coi là một loại linh kiện.

### 2.2 Input contract

Mỗi sample phải mang cả ảnh và metadata:

```yaml
crop_id: string
image_id: string
board_id: string
sku_id: string | null
side: top | bottom
bbox_xyxy: [x1, y1, x2, y2]
detector_class_hint: string | null
detector_confidence: float | null
crop_rgb: uint8[H,W,3]
source: internal_aoi | fpic | fics_pcb | wacv | pcb_vision | ...
```

Tiền xử lý mặc định cần thử nghiệm:

1. Nới bbox 10–15% để giữ chân/pad và một ít context.
2. Giữ aspect ratio, letterbox thay vì kéo méo.
3. Dùng kích thước 320 px làm mốc; so với 224 và 384 px.
4. Lấy crop từ ảnh có độ phân giải cao nhất, không crop sau khi toàn board đã bị resize nhỏ.
5. Lưu `crop_quality`: kích thước gốc, blur, saturation, clipping, occlusion và tỷ lệ bbox nằm ngoài ảnh.

### 2.3 Output contract

```json
{
  "family": "capacitor",
  "family_probability": 0.982,
  "visual_subtype": "electrolytic",
  "subtype_probability": 0.947,
  "mount_type": "smd",
  "top_k": [["capacitor", 0.982], ["battery_power", 0.010]],
  "unknown_score": 0.041,
  "decision": "accept",
  "embedding_version": "pcb-emb-v1",
  "model_version": "classify-component-v1.3.0"
}
```

Ba quyết định hợp lệ:

- `accept`: đủ chắc chắn để chuyển kết quả sang fusion engine;
- `review`: crop có thể biết nhưng dưới ngưỡng hoặc detector/classifier bất đồng;
- `unknown`: ngoài taxonomy/out-of-distribution; không ép vào lớp gần nhất.

## 3. Taxonomy v1 cho pre-training

### 3.1 Head bắt buộc: `family`

Taxonomy đầu tiên nên đủ rộng để học từ dataset công khai nhưng vẫn có ý nghĩa AOI:

| ID | Family | Nhãn nguồn có thể gom vào | Ghi chú |
|---:|---|---|---|
| 0 | `chip_passive` | tiny resistor/capacitor/inductor/ferrite không phân biệt được | Dùng cho 0201/0402 hoặc crop không có marking/refdes. Không ép model đoán R/C/L từ pixel giống nhau. |
| 1 | `resistor` | resistor | Subtype SMD-marked/THT/network ở head phụ nếu đủ dữ liệu. |
| 2 | `capacitor` | capacitor | Subtype chip/electrolytic/other ở head phụ. |
| 3 | `magnetic` | inductor, ferrite bead, common-mode choke, transformer, EMI filter | Cho phép subtype khi hình thái và nhãn đáng tin cậy. |
| 4 | `diode` | diode, bridge diode | Schottky/Zener/TVS là chức năng; chỉ tách khi có BOM/marking và không dùng như nhãn thuần thị giác. |
| 5 | `led` | LED | RGB/IR không luôn phân biệt được bằng ngoại hình. |
| 6 | `discrete_semiconductor` | transistor, MOSFET | BJT/JFET/MOSFET/IGBT không được suy ra từ package đơn thuần. |
| 7 | `power_semiconductor` | power MOSFET, rectifier, power package | Lớp hình thái cho DPAK/D2PAK/TO-220...; identity do BOM/OCR. |
| 8 | `ic` | IC | Chỉ là IC/package; không phải chức năng. |
| 9 | `timing` | clock, crystal, oscillator, resonator | RTC/clock generator dạng IC vẫn thuộc `ic`. |
| 10 | `protection` | fuse, PTC, MOV, NTC, surge component | Tách visual subtype khi đủ dữ liệu. |
| 11 | `relay` | electromechanical relay, SSR package nếu nhìn thấy rõ | SSR dạng IC không nên ép vào relay nếu không có marking. |
| 12 | `connector` | header, terminal, board/wire connector, FFC/FPC, socket | Head phụ phân loại hình thái connector. |
| 13 | `physical_port` | USB, HDMI, RJ45, jack, VGA/DVI | Tách subtype khi đủ instance và đủ biến thể nhà cung cấp. |
| 14 | `switch_control` | button, tactile/slide/toggle/DIP/rotary switch, encoder, potentiometer | Head phụ cho subtype. |
| 15 | `display` | LCD/OLED/TFT/7-segment/matrix/e-paper | Module hiển thị lớn có thể đồng thời gắn tag `module`. |
| 16 | `acoustic` | buzzer, microphone, speaker | Audio amplifier/codec vẫn là `ic`. |
| 17 | `battery_power_input` | battery, holder, DC/power connector | Pin/cell rời chỉ học nếu thực sự xuất hiện trong ảnh AOI. |
| 18 | `thermal_mechanical` | heatsink, fan, shield can, screw, spacer/standoff | Có thể tách sau; mounting hole không phải linh kiện. |
| 19 | `module` | wireless/GNSS/camera/power/sensor/MCU/RF module | Chức năng module do OCR/BOM; 6.1 chỉ nhận dạng hình thái module. |
| 20 | `other_component` | linh kiện hợp lệ ngoài taxonomy | Lớp đóng tạm thời, phải được review định kỳ. |
| 21 | `false_crop_background` | pad, silkscreen, via, trace, solder blob, vùng trống | Giúp chịu lỗi từ detector; không thay thế OOD score. |

### 3.2 Head phụ

- `mount_type`: `smd`, `tht`, `module`, `mechanical`, `unknown`.
- `visual_subtype`: chỉ tồn tại dưới từng family, ví dụ `capacitor/electrolytic`, `connector/ffc_fpc`, `switch_control/dip_switch`.
- `package_family`: `chip_0402_like`, `sot23`, `soic`, `qfn_qfp`, `bga`, `to220`, `radial`, `axial`, `custom`, `unknown`; nhãn từ BOM/footprint nếu có.
- `quality_flag` không phải ground-truth lỗi chi tiết: `clean`, `blurred`, `partial`, `occluded`, `severe_defect`. Lỗi hàn chính thức thuộc 6.6–6.9.

Danh sách leaf ứng viên để audit dữ liệu, chưa mặc định mở toàn bộ ngay từ run đầu:

```text
chip_passive, resistor_smd, resistor_tht,
capacitor_chip, capacitor_electrolytic, capacitor_other,
inductor, ferrite_bead, transformer_choke,
diode, led, transistor_small, power_semiconductor, ic,
crystal_oscillator, fuse_ptc, protection_disc, relay,
connector, physical_port, switch_button, potentiometer,
display, acoustic, battery_holder, heatsink, module
```

Nhãn public chỉ có `capacitor` được dùng cho loss của parent `capacitor`, không tự động biến thành `capacitor_chip`. Tương tự, `transistor`, `clock`, `battery` hoặc `connector` chỉ đi xuống leaf sau khi ảnh/BOM đã được audit.

### 3.3 Quy tắc mở một subtype mới

Chỉ tách subtype khỏi parent khi đồng thời đạt:

- định nghĩa dựa trên đặc điểm quan sát được hoặc ground truth BOM đáng tin cậy;
- tối thiểu 500 observation sạch, ít nhất 5 board và 3 part/package variant; P0 nên hướng tới 2.000+ observation và 10+ board;
- cùng một linh kiện chụp lặp không được tính là sample độc lập;
- audit cho thấy người gán nhãn thống nhất; sample mơ hồ được đặt `ambiguous`, không ép nhãn;
- benchmark leave-one-board/leave-one-variant-out cho thấy subtype tốt hơn việc giữ parent.

Nếu không đạt, giữ parent class và để BOM/OCR giải quyết identity.

## 4. Kế hoạch dữ liệu

### 4.1 Vai trò của từng nguồn

| Dữ liệu | Vai trò trong pre-train | Có được vào test cuối không? |
|---|---|---|
| Consolidated/WACV/FICS/FPIC | Supervised pre-train ở nhãn family; tăng độ đa dạng PCB | **Không**; chỉ train/dev theo source. |
| PCB-Vision | Bổ sung `ic`, `capacitor`, `connector`; segmentation chuyển thành crop | Không. |
| PCB DSLR | Bổ sung đa dạng IC và top-marking | Không; còn giới hạn non-commercial research. |
| PCB-SAID | Normal SMD + crop bị lệch/nhấc/thiếu cho robustness/quality flag | Không; kiểm tra license và taxonomy release. |
| Ảnh board nội bộ chưa nhãn | Domain-adaptive self-supervised pre-training; hard-negative mining | Không nếu ảnh được dùng để học representation. |
| Ảnh AOI nội bộ có BOM/centroid | Fine-tune và supervision đáng tin cậy | Chia theo board/lot; một phần khóa làm test. |
| Lot/SKU tương lai, không dùng tune | Qualification và shadow test | **Có; đây là test quyết định**. |

Tách hai nhánh pháp lý:

- `research_track`: được phép thử mọi nguồn có điều khoản nghiên cứu phù hợp;
- `production_track`: chỉ dùng dữ liệu có license đã được duyệt và dữ liệu nội bộ.

Không đưa checkpoint từ `research_track` sang sản phẩm chỉ bằng cách fine-tune thêm; trọng số vẫn là derivative của nguồn pre-train.

### 4.2 Manifest bắt buộc

Mỗi crop là một dòng Parquet/CSV:

```text
crop_id, sha256, parent_image_sha256, source_dataset, source_version,
license_id, physical_board_id, sku_id, board_side, lot_id, capture_session,
camera_id, component_refdes, part_number_hash, footprint, bbox_xyxy,
family, visual_subtype, mount_type, package_family, label_source,
label_confidence, annotator_id, crop_quality, split
```

`label_source` nên là một trong: `public_annotation`, `bom_centroid`, `ocr_lookup`, `human_verified`, `pseudo_label`. Không trộn pseudo-label với human label mà mất provenance.

Thêm tier chất lượng nhãn:

- **Gold:** ảnh AOI nội bộ, join BOM/refdes hoặc human-verified;
- **Silver:** annotation trực tiếp từ nguồn học thuật, taxonomy rõ và đã audit;
- **Bronze:** community/web/remap/proxy hoặc taxonomy mơ hồ; chỉ dùng SSL, parent head hoặc trọng số thấp sau review, không dùng làm benchmark cuối.

Consolidated dataset có thể chứa lại WACV/FICS/PCB-Vision. Chọn một bản canonical hoặc dedup source trước khi trộn, tránh tính một ảnh nhiều lần.

### 4.3 Tạo crop

Với mỗi bbox/polygon hợp lệ, sinh và lưu metadata của ba view:

1. `tight`: bbox + 5% padding;
2. `context`: bbox + 15% padding;
3. `detector_like`: jitter theo phân phối lỗi thật của model bước 4.

Khi chưa có detector đủ tốt, dùng sweep IoU 0,6–1,0 để mô phỏng; sau đó thay bằng thống kê residual `(dx, dy, dw, dh)` trên validation của detector. Không dùng jitter tùy ý mãi mãi.

Negative/unknown phải gồm:

- pad trống, trace/via, silkscreen/text, solder joint;
- bbox cắt cụt linh kiện, hai linh kiện trong một crop, phản sáng/blur nặng;
- linh kiện ngoài taxonomy, package mới, foreign object;
- ảnh ngoài PCB để đo OOD, nhưng ngưỡng cuối phải tune bằng lỗi thực tế từ cell AOI.

### 4.4 Dedup và split chống leakage

Thứ tự dedup:

1. SHA-256 tìm file giống hệt;
2. perceptual hash tìm resize/compress/rotate;
3. embedding nearest-neighbor để review near-duplicate;
4. nhóm mọi crop của cùng `parent_image`, `physical_board`, burst chụp và augmentation vào cùng split.

Split nội bộ đề xuất khi dữ liệu đủ lớn:

| Split | Tỷ lệ tham khảo | Quy tắc |
|---|---:|---|
| `train` | 70% | Board/lot dùng train; có thể nhận public data. |
| `val_model` | 10% | Chọn checkpoint/hyperparameter; board không trùng train. |
| `calibration` | 10% | Chỉ fit temperature/threshold/reject; không chọn backbone bằng tập này. |
| `test_locked` | 10% | Board hoặc SKU không trùng; khóa cho đến khi chốt model. |
| `future_lot_shadow` | ngoài split | Lot/thời gian chụp mới; chỉ dùng qualification/monitoring. |

Nếu có dưới 20 board độc lập, dùng **grouped cross-validation/leave-one-board-out** thay vì cố chia nhiều tập nhỏ. Mọi tile/crop của cùng board phải đi chung một fold.

Nên duy trì năm lát cắt benchmark:

- `T0-public-board-held-out`: đo chất lượng checkpoint public pre-train;
- `T1-seen-SKU-new-lot`: đúng SKU nhưng unit/lot/session mới;
- `T2-unseen-SKU-package`: board design, supplier hoặc package chưa gặp;
- `T3-open-set-background`: linh kiện novel, pad/via/text và false crop;
- `T4-end-to-end`: crop do detector bước 4 thật tạo ra.

### 4.5 Mục tiêu số lượng

- P0 (`chip_passive`, resistor, capacitor, magnetic, diode, LED, discrete semiconductor, IC, connector, timing): mục tiêu 2.000+ observation độc lập/family, 10+ board, 5+ variant.
- P1: 500+ observation, 5+ board, 3+ variant.
- Rare class dưới 100–500 sample: huấn luyện ở parent class; giữ nhãn subtype để mở lớp sau.
- `false_crop_background` và unknown/outlier: chiếm khoảng 10–20% batch ở thí nghiệm open-set, nhưng tỷ lệ cuối phải được tune theo lỗi detector thật.
- Unlabeled domain pool: nếu có từ 50.000–100.000 crop/board-tile **độc lập và đa dạng**, thử continued self-supervised learning; ít hơn thì ưu tiên foundation backbone + supervised fine-tune để tiết kiệm compute.

Các con số trên là ngưỡng khởi động, không phải bảo đảm accuracy. Board/variant diversity quan trọng hơn việc nhân bản bằng augmentation.

### 4.6 Quality gate trước training

- 100% sample có provenance, version và license state.
- Không có parent-image/board leakage qua split.
- Audit ngẫu nhiên tối thiểu 200 crop/class lớn và toàn bộ class hiếm.
- Hai người review tối thiểu 10% sample mơ hồ; có hàng đợi `needs_review`.
- Báo cáo class/source/board/package distribution và gallery 50 sample/class.
- Không dùng lớp có tên mơ hồ như `cap1`, `cap2`, `clock` trước khi mapping được duyệt.
- Validation/calibration/test cuối chỉ dùng Gold hoặc Silver đã human-review; Bronze không làm ground truth nghiệm thu.

## 5. Chuỗi pre-training và fine-tuning

### Cấu hình khởi điểm có thể chạy ngay

Đây là config cho E02, dùng để tạo baseline tái lập rồi mới tune:

```yaml
model: efficientnet_b0
init: torchvision_imagenet_default
input_size: 224
heads: [family, mount_type]
optimizer: adamw
lr_head: 3.0e-4
lr_backbone: 5.0e-5
weight_decay: 0.05
schedule: cosine
warmup_ratio: 0.05
epochs: 40
freeze_backbone_epochs: 3
effective_batch_size: 128       # dùng gradient accumulation nếu cần
amp: true
loss_family: cross_entropy
label_smoothing: 0.05
sampler: balanced_by_family_source_board
early_stopping_patience: 8
seed: 42
select_checkpoint_by: [macro_f1, p0_min_recall]
```

`lr`, epoch và batch size là điểm bắt đầu, phải xác nhận bằng learning curves. Chạy thêm seed `17` và `73` chỉ cho hai cấu hình vào vòng cuối. Nếu batch vật lý quá nhỏ với BatchNorm, đánh giá freeze BN statistics hoặc SyncBatchNorm thay vì giả định gradient accumulation sửa được thống kê BN.

### Giai đoạn W0 — Khởi tạo foundation

Tạo hai nhánh song song:

- `W0-CNN`: EfficientNet-B0 pretrained ImageNet;
- `W0-VFM`: DINOv3 ViT-S/16 pretrained; chỉ dùng sau khi chấp nhận license.

Kiểm tra linear probe trên internal validation trước khi unfreeze. Nếu VFM không cải thiện macro-F1/unknown separation đủ để bù latency và độ phức tạp, không đưa vào deployment.

Không pre-train backbone từ random initialization ở MVP. Chỉ cân nhắc training from scratch khi có cỡ hàng triệu crop PCB độc lập, đủ biến thể board/camera/package và có ngân sách đối chứng với foundation weights.

### Giai đoạn W1 — Domain-adaptive pre-training, tùy điều kiện

Dùng ảnh nội bộ chưa nhãn (full-board tiles + component crops) với self-supervised objective kiểu DINO/masked-image modeling. Mục tiêu là học solder mask, silkscreen, package, marking và ánh sáng của cell AOI mà không cần nhãn.

Chỉ chạy W1 khi:

- pool đủ lớn/đa dạng;
- dedup theo board/session hoàn tất;
- có ngân sách GPU và thí nghiệm `with/without W1`;
- không cho ảnh thuộc `test_locked`/`future_lot_shadow` đi vào pre-training.

Output: `W1-PCB-SSL`, chưa phải classifier.

### Giai đoạn W2 — Supervised pre-train trên public + internal-train

- Remap nhãn nguồn sang `family`; sample mơ hồ dùng `ignore` hoặc parent label.
- Batch cân bằng đồng thời theo `family` và `source`, tránh một batch chứa quá nhiều crop từ cùng board.
- Baseline loss: Cross Entropy + label smoothing nhỏ (bắt đầu 0,05).
- So sánh class-weighted CE hoặc Balanced Softmax khi long-tail; không vừa oversample cực mạnh vừa dùng weight cực mạnh ở run đầu.
- Huấn luyện head trước khi unfreeze backbone; dùng learning rate backbone thấp hơn head.

Output: `W2-PCB-PUBLIC-SUP`.

### Giai đoạn W3 — Fine-tune trên AOI nội bộ

- Tăng tỷ trọng ảnh đúng camera/recipe đích.
- Trộn clean GT crops và detector-like crops; batch cuối epoch không bị một SKU thống trị.
- Multi-task loss tham khảo:

```text
L = 1.0 * L_family
  + 0.5 * L_visual_subtype(masked)
  + 0.2 * L_mount_type
  + 0.2 * L_package(masked)
  + lambda_ood * L_ood
```

Các hệ số phải ablate; head không có nhãn dùng mask, không điền `unknown` giả làm ground truth.

Output: `W3-AOI-FINETUNED`.

### Giai đoạn W4 — Calibration, reject và hard-negative mining

1. Fit temperature scaling trên split `calibration`; phương pháp này là baseline calibration đơn giản được mô tả trong [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html).
2. So sánh maximum calibrated probability, energy score và khoảng cách embedding tới prototype. Energy score là baseline OOD theo [Liu et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html).
3. Chọn threshold **theo class/risk**, không dùng một ngưỡng 0,5 cho mọi lớp.
4. Chạy trên ảnh AOI không nhãn; cluster các crop reject/sai; human review; thêm hard negative và package mới vào vòng train tiếp theo.

Ví dụ logic quyết định:

```text
if crop_quality is unusable:
    review
elif ood_score > threshold_ood:
    unknown
elif calibrated_p_top1 < threshold[class]:
    review
elif detector_hint conflicts with classifier and neither is dominant:
    review
else:
    accept
```

Output: `W4-CALIBRATED` + `calibration.json` + `thresholds.yaml`.

### Giai đoạn W5 — Distill và deploy

- Teacher tốt nhất có thể là DINOv3/ConvNeXt; student là EfficientNet-B0 hoặc model edge đã benchmark.
- Distill logits + embedding, sau đó fine-tune supervised trên internal data.
- Xuất ONNX FP32/FP16; chỉ thử INT8 sau khi có tập calibration đại diện và kiểm tra lại per-class recall, calibration và unknown detection.
- Benchmark cả batch 1 và batch nhiều crop/board trên đúng thiết bị đích.

Tài liệu TensorRT nhấn mạnh tập calibration INT8 phải đại diện cho phân phối inference; quy trình quantization hiện hành cần đối chiếu đúng phiên bản TensorRT đang triển khai: [NVIDIA quantized types](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html).

## 6. Augmentation policy

Chỉ mô phỏng biến thiên thật đã đo từ cell AOI:

| Augmentation | Điểm bắt đầu | Lưu ý |
|---|---|---|
| Rotate | 0/90/180/270° + jitter nhỏ khoảng ±5° | Family classifier nên gần rotation-invariant; không dùng chung policy này cho polarity model 6.3. |
| Translation/scale | khoảng ±5%, 0,9–1,1 | Thay bằng phân phối residual của detector khi có số liệu. |
| Brightness/contrast/gamma | mức nhẹ, khoảng ±10–15% | Ước lượng từ lot/camera thật; giữ color cue của LED/capacitor. |
| White-balance/channel shift | nhẹ | Không biến solder mask hoặc marking thành màu phi thực tế. |
| Blur/noise/compression | mức tương ứng camera | Thêm cả quality flag; blur quá nặng nên reject. |
| Specular/shadow | mô phỏng nhẹ hoặc lấy ảnh thật | Synthetic phải được kiểm chứng bằng nearest-neighbor/gallery. |
| CutMix/MixUp/random erasing | chỉ là ablation | Có thể tạo linh kiện phi vật lý và phá marking/pin; không bật mặc định. |

Không bật mirror flip mặc định: nó có thể tạo marking/polarity hoặc cấu hình cơ khí không tồn tại. Nếu family classifier được chứng minh bất biến với flip, chỉ mở sau một ablation riêng.

Validation/test không augmentation ngoài preprocessing deterministic.

Để phát hiện shortcut từ silkscreen/refdes, chạy ba test bổ sung trên cùng checkpoint: `component-only`, `context-only` và `refdes-masked`. Nếu `context-only` vẫn dự đoán tốt hoặc `refdes-masked` tụt mạnh bất thường, model đang học chữ `R/C/U` thay vì hình thái linh kiện.

## 7. Ma trận thí nghiệm tối thiểu

| Run | Backbone/khởi tạo | Input | Thay đổi chính | Câu hỏi cần trả lời |
|---|---|---:|---|---|
| E00 | EfficientNet-B0/ImageNet | 224 | Overfit 200–500 crop sạch | Pipeline/label/loss có đúng không? Phải gần như memorise tập nhỏ. |
| E01 | EfficientNet-B0/ImageNet | 224 | Flat family baseline | Mốc deployment chính cho Raspberry Pi. |
| E02 | MobileNetV3-Large/ImageNet | 224 | Giữ recipe E01 | Giảm compute có làm mất macro-F1/accepted precision quá mức không? |
| E03 | EfficientNet-B0/ImageNet | 320 | Giữ mọi thứ khác E01 | Resolution cao có tăng accuracy đủ để bù compute trên Pi không? |
| E04 | EfficientNetV2-S/ImageNet | 320 | Upper-bound offline | Backbone nặng cải thiện bao nhiêu; không mặc định triển khai lên Pi. |
| E05 | DINOv3 ViT-S/16 | 224/320 | Frozen backbone + linear head | Representation zero/few-shot tốt đến đâu? |
| E06 | DINOv3 ViT-S/16 | 224/320 | Unfreeze 2–4 block cuối | Partial fine-tune có thắng E03? |
| E07 | Model thắng E01–E06 | như cũ | Có/không W1 domain SSL | Unlabeled internal data có lợi thật không? |
| E08 | Model tốt nhất | như cũ | Flat vs hierarchical/multi-task | Head mount/package có giảm confusion không? |
| E09 | Model tốt nhất | như cũ | Tight/context/detector-like crop | Độ bền với sai số bước 4–5. |
| E10 | Model tốt nhất | như cũ | CE vs weighted CE/Balanced Softmax | Xử lý tail class mà không phá head class. |
| E11 | Model tốt nhất | như cũ | Calibration + energy/embedding reject | Risk–coverage/OOD tốt đến đâu? |
| E12 | Student + distillation | 224/320 | FP32/FP16/INT8 | Điểm Pareto accuracy–latency–memory. |

Nếu các package rất giống nhau, thêm một ablation `Supervised Contrastive + CE`; không thay baseline trước khi chứng minh lợi ích. Tham khảo [Supervised Contrastive Learning, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html).

Mỗi run lưu seed, commit, config, data-manifest hash, split hash, checkpoint và error gallery. Chạy ít nhất 3 seed cho các ứng viên cuối; không cần 3 seed cho mọi sanity run.

## 8. Metric và gate

### 8.1 Metric bắt buộc

- `macro-F1`, balanced accuracy và per-class precision/recall;
- confusion matrix, đặc biệt `chip_passive/resistor/capacitor/magnetic`, `ic/discrete_semiconductor`, `connector/physical_port`;
- top-1 và top-3 accuracy chỉ là metric phụ;
- ECE, NLL/Brier score và reliability diagram sau calibration;
- OOD AUROC, AUPR, FPR@95TPR;
- selective risk–coverage: tỷ lệ lỗi khi model chỉ auto-accept sample đủ tin cậy;
- chênh lệch giữa GT crop và detector-generated crop;
- per-board/SKU/lot/camera/package performance với confidence interval bootstrap theo **board**, không bootstrap crop độc lập;
- latency p50/p95/p99, throughput, peak VRAM/RAM, model size trên hardware đích.
- giá trị thực của pre-train: so với ImageNet-only, checkpoint PCB nên tăng target-domain macro-F1 có ý nghĩa (mốc sàng lọc +2 điểm phần trăm) hoặc đạt cùng chất lượng với không quá 50% nhãn nội bộ; nếu không, bỏ công đoạn đắt tiền đó.

### 8.2 Gate đề xuất cho PoC

Đây là mục tiêu khởi đầu, phải thay bằng chi phí false accept/false reject của dây chuyền:

| Gate | Điều kiện đề xuất |
|---|---|
| Data-ready | Không leakage; mapping/license/provenance đạt 100%; audit error dưới 2% ở sample đã review. |
| Baseline-ready | E00 overfit thành công; E02 macro-F1 ≥ 0,90 trên internal grouped validation; không class P0 nào recall < 0,90. |
| PoC-ready | Macro-F1 ≥ 0,95 trên `test_locked`; P0 recall ≥ 0,98; gap GT-crop → detector-crop ≤ 2 điểm phần trăm. |
| Confidence-ready | ECE ≤ 0,03; threshold được chọn trên calibration split; báo cáo risk–coverage và OOD thay vì chỉ softmax confidence. |
| Pilot-ready | Ở operating point đã chọn, auto-accept coverage ≥ 90% và classification error trong phần accepted ≤ 0,5%; không có regression lớn ở rare/critical classes. |
| Deploy-ready | Thỏa SLA latency/memory trên hardware thật; FP16/INT8 không làm giảm P0 recall quá mức đã duyệt; vượt shadow test trên lot/SKU mới. |

Không dùng một con số “accuracy 99%” làm gate duy nhất. Với mục tiêu escape rất thấp, phải tính cỡ mẫu qualification và confidence bound dựa trên mức rủi ro mong muốn.

## 9. Kết nối các bước khác trong sơ đồ

```text
4. detector
   -> bbox + detector_hint + detector_score
5. crop
   -> tight/context crop + crop_quality
6.1 classifier
   -> family/subtype/top-k/calibrated confidence/OOD/embedding
6.4 + 6.5
   -> OCR marking + BOM/centroid lookup -> part identity
7. fusion
   -> accept/review/NG cùng lý do và provenance
```

Quy tắc tích hợp:

- Nếu bước 4 đã là multi-class detector, giữ output như một prior; không cho detector và classifier dùng hai taxonomy không mapping.
- Đánh giá riêng `classifier with GT crops` và `end-to-end with predicted crops` để biết lỗi thuộc bước 4 hay 6.1.
- `wrong part` không được kết luận chỉ từ family classifier nếu BOM yêu cầu part number cụ thể.
- Silkscreen/refdes có thể hỗ trợ fusion, nhưng cần thí nghiệm crop tight/context để phát hiện model đang “đọc C/R/U” thay vì nhìn linh kiện.
- Lưu top-k và embedding để người vận hành xem nearest examples; không chỉ lưu nhãn cuối.

## 10. Lịch thực hiện 6 tuần

| Tuần | Công việc | Đầu ra/gate |
|---:|---|---|
| 1 | Chốt contract, taxonomy v1, mapping nguồn, license; tạo manifest; dedup/split | `taxonomy.yaml`, `label_mapping.yaml`, `dataset_manifest.parquet`, báo cáo audit. |
| 2 | Pipeline crop + gallery + E00/E01; đo distribution input | Sanity pass, baseline nhẹ, danh sách lỗi nhãn. |
| 3 | E02–E06: MobileNetV3/EfficientNetV2/ConvNeXt/DINO đối chứng | Bảng benchmark backbone/resolution; B0 vẫn là deployment mặc định nếu model lớn không thắng rõ. |
| 4 | E07–E10: domain SSL nếu đủ dữ liệu, hierarchy, detector jitter, imbalance | Model ứng viên và ablation report. |
| 5 | E11: calibration, OOD/reject, hard-negative round 1; test locked | Risk–coverage, OOD, confusion/error gallery; quyết định PoC. |
| 6 | E12: distill/export/FP16–INT8; benchmark hardware và shadow mode | ONNX/engine, model card, thresholds, deployment benchmark, rollback plan. |

Nếu dữ liệu nội bộ/BOM chưa sẵn sàng, không coi tuần 5–6 là qualification; chỉ hoàn thành research baseline.

## 11. Ước lượng compute để lập ngân sách

Các khoảng dưới đây chỉ dùng để chuẩn bị tài nguyên; thời gian thật phụ thuộc số crop và GPU:

- CNN baseline: 1 GPU 12–24 GB, mixed precision; thường có thể hoàn thành mỗi run trong vài giờ đến khoảng 1–2 ngày.
- DINOv3 partial fine-tune: nên có 24 GB trở lên; teacher lớn hoặc resolution cao cần gradient accumulation/multi-GPU.
- Continued self-supervised pre-training: tốn nhất; chỉ chạy sau khi E02/E06 và data audit chứng minh đáng đầu tư.
- Dự trù 12–20 run có ý nghĩa gồm sanity/ablation; giữ một ngân sách riêng cho 3-seed và hard-negative retraining của 2 ứng viên cuối.

Không khóa GPU theo phỏng đoán: chạy profiling 1.000 step đầu để đo samples/s, VRAM và ngoại suy chi phí.

## 12. Artefact phải bàn giao

```text
classification_6_1/
├── taxonomy.yaml
├── label_mapping.yaml
├── dataset_manifest.parquet
├── splits/
│   ├── train.csv
│   ├── val_model.csv
│   ├── calibration.csv
│   └── test_locked.csv
├── configs/
├── checkpoints/
├── calibration.json
├── thresholds.yaml
├── model.onnx
├── model_card.md
├── data_card.md
├── benchmark_hardware.md
├── metrics.json
├── confusion_matrix.png
├── risk_coverage.png
└── error_gallery/
```

Model card tối thiểu ghi: taxonomy/version, nguồn và license dữ liệu, split strategy, camera/SKU đã kiểm tra, metric theo class, known limitations, unknown rule, preprocessing checksum, output schema, hardware benchmark và rollback model.

## 13. Việc cần làm trong 10 ngày đầu

1. Chốt danh sách SKU/camera/hardware đích và chi phí của false accept/false reject.
2. Trích toàn bộ bbox từ dataset công khai thành manifest, chưa resize crop vĩnh viễn.
3. Join ảnh nội bộ với BOM/centroid; hash part number nếu cần bảo mật.
4. Gom nhãn về 22 family v1; đánh dấu `ambiguous` thay vì ép nhãn.
5. Dedup và split theo physical board/lot/source.
6. Sinh gallery và audit; sửa mapping trước khi train.
7. Hoàn thiện E00 để bắt lỗi loader/loss/label index.
8. Chạy E01 và E02 trên cùng split/config.
9. Đo GT-crop so với detector-like crop.
10. Chỉ sau các bước trên mới mở DINO/domain SSL và subtype head.

## 14. Tiêu chí hoàn thành bước pre-train 6.1

Pre-train được coi là hoàn thành khi có `W2-PCB-PUBLIC-SUP` tái lập được, nhưng bước 6.1 chỉ sẵn sàng PoC khi có thêm:

- `W3-AOI-FINETUNED` trên dữ liệu đúng cell;
- `W4-CALIBRATED` và rule `accept/review/unknown`;
- báo cáo grouped holdout không leakage;
- test end-to-end bằng bbox thật từ bước 4;
- model/card/config/manifest hash và checkpoint có thể tái tạo;
- danh sách lớp chưa đủ dữ liệu và đường lui về parent/BOM/OCR.

Nói ngắn gọn: **public data tạo khả năng nhận dạng hình thái; dữ liệu AOI nội bộ, calibration và reject policy mới tạo độ tin cậy để vận hành.**
