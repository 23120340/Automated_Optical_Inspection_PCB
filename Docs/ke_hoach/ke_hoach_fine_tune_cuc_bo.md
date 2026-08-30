# Fine-tune ngay trên máy này — có làm được không, và làm thế nào

> Soạn 2026-08-24, trả lời câu hỏi: *"phần detect mối hàn khá ổn ngoại trừ với
> chân của IC, có thể train fine tune lại ở trên máy local sau khi tôi feedback
> model và dùng với ảnh đã có không?"*
>
> Mọi con số về tốc độ ở đây **đo trên chính máy này**, không phải ước lượng.

## Trả lời ngắn

**Được, và đúng ở tầm sức của máy này** — nhưng cần đính chính một chỗ trong
cách đặt câu hỏi trước đã, vì nó đổi hẳn việc phải làm.

## Trước hết: "detect mối hàn" hiện KHÔNG phải một model

Bước 5.5 (tách ROI chân hàn) chạy bằng **hình học**, không bằng model: nó lấy
box linh kiện của lượt 1 rồi suy ra vị trí chân theo topology của lớp linh kiện
(`two_terminal` / `multi_pin` / `pad_only`).

Nên **không có gì để fine-tune ở đó**. Và điều đó giải thích đúng cái bạn quan
sát được:

| Loại linh kiện | Vì sao hình học làm được / không làm được |
|---|---|
| Điện trở, tụ (2 chân) | Hai đầu, đối xứng, suy từ box là ra — **nên nó ổn** |
| **Chân IC (nhiều chân)** | Phải đoán *cạnh nào có chân*, *bao nhiêu chân*, *pitch bao nhiêu*. Box không chứa thông tin đó — **nên nó sai** |

Hình học không biết trên board có gì; nó chỉ biết cái box. Với IC thì cái box
không đủ. Đây là lý do lượt 2 tồn tại trong kế hoạch từ đầu
(`Docs/bao_cao/tien_do_detect_2_luot.md`, mục B/C).

**Vậy việc thật sự phải làm không phải "fine-tune model cũ" mà là "train model
lượt 2", và ưu tiên chân IC trước.** Tin tốt: thứ vẫn thiếu suốt là **dữ liệu
gán nhãn**, và mục "Đánh giá model" vừa thêm vào app chính là công cụ gán nhãn
đó.

## Máy này làm được tới đâu — số đo thật

| | |
|---|---|
| CPU | AMD Ryzen 7 5800H, 16 luồng |
| GPU | **Không có** — torch bản `2.13.0+cpu` |
| RAM | 7.5 GB tổng, **còn trống ~1.5 GB** |

Tôi đã tưởng RAM trống là ràng buộc thật. **Đo lại thì không phải** — xem
bảng bộ nhớ dưới đây. Ràng buộc thật là **thời gian CPU**.

Đo `yolo11n` (2.6M tham số) trên chính máy này:

| imgsz | forward có autograd | ước một bước train | ảnh/giây |
|---|---|---|---|
| **256** | 25.6 ms/ảnh | **~77 ms/ảnh** | ~13 |
| 640 | 243 ms/ảnh | ~730 ms/ảnh | ~1.4 |

*(một bước train ≈ 3× forward: xuôi + ngược + cập nhật. Đây là ước lượng có
nêu giả định, không phải đo trực tiếp.)*

Bộ nhớ, đo trên cùng máy (tổng tiến trình, `imgsz=256`):

| batch | RAM tiến trình | tăng so với nền |
|---|---|---|
| 4 | 378 MB | 131 MB |
| 8 | 403 MB | 156 MB |
| **16** | **452 MB** | 205 MB |

Thấp hơn tôi lo nhiều: batch 16 vẫn thoải mái trong 1,5 GB trống. Con số này đo
lượt xuôi; một bước train đủ còn thêm gradient và momentum (~10 MB mỗi thứ với
2,6M tham số) — vẫn không đáng kể. **Thứ có thể làm tràn là dataloader**, xem
`cache=False` ở bước 3.

Quy ra thời gian thật:

| Dữ liệu | Epoch | imgsz 256 | imgsz 640 |
|---|---|---|---|
| 2.000 crop | 30 | **~1,3 giờ** | ~12 giờ |
| 5.000 crop | 50 | **~5,3 giờ** | ~2 ngày |
| 10.000 crop | 100 | ~21 giờ | không khả thi |

**Kết luận: khả thi ở `imgsz=256`, chạy qua đêm. Không khả thi ở 640.** Và 256
đã là cấu hình notebook lượt 2 đang dùng — vì lý do độc lập, xem
`Docs/danh_gia/xep_hang_model.md`.

## Kế hoạch, theo thứ tự

### Bước 1 — Gán nhãn bằng chính app (việc tốn công nhất)

Mục **"Đánh giá model"** ở cuối trang bước 6.2. Với mỗi board:

- Chân IC bị khoanh sai → **"Bấm vào box bị sai"** → loại `ROI sai chỗ`
- Chân IC bị bỏ sót → **"Bấm vào chỗ model bỏ sót"**, chỉnh cỡ ô cho vừa chân
  rồi bấm vào giữa chân đó

Cỡ ô được ghi lại (`box_size`) chính là để lượt train này dùng — nó là nhãn
kích thước, không chỉ là vùng xem lại.

**Mốc tối thiểu:** 10–20 board **khác lô**, ưu tiên nhiều IC. Nút thắt không
phải số chân mà là **số board**: chân từ cùng một board có cùng ánh sáng, cùng
lô hàn, thường cùng loại IC — và model sẽ học thuộc board thay vì học hình dạng
chân.

Ước lượng: một board có ~20 IC × ~14 chân ≈ 280 chân. 15 board ≈ 4.200 chân.
Đủ cho mốc "5.000 crop" ở bảng trên.

### Bước 2 — Xuất nhãn ra định dạng train

Bản ghi đánh giá là JSON Lines có toạ độ; cần chuyển sang YOLO. Chưa có script
này — **đây là phần tôi cần viết**, và nó nhỏ:

```
feedback/*.jsonl  →  scripts/feedback_to_yolo.py  →  datasets/leads_v1/
```

Việc nó phải làm: lọc theo `stage="solder"`, cắt crop linh kiện từ ảnh gốc theo
`source_sha256`, đổi toạ độ board sang toạ độ crop, ghi nhãn YOLO. `EvidenceViewer`
đã lo phần cắt lại và phần **từ chối khi ảnh gốc đã đổi**.

### Bước 3 — Train tại chỗ

```bash
yolo detect train model=yolo11n.pt data=datasets/leads_v1/data.yaml \
     imgsz=256 epochs=30 batch=16 device=cpu workers=4 cache=False
```

Ba tham số quan trọng, và lý do:

- `imgsz=256` — xem bảng trên. 640 là 12 lần thời gian mà **không thêm thông
  tin nào**: crop trung vị chỉ 48 px, phóng lên 640 là nội suy.
- `batch=16` — đo được là chỉ tốn 452 MB, thoải mái trong 1,5 GB trống. Tôi
  ban đầu định khuyên batch 8 vì lo tràn RAM; đo lại thì lo đó không có cơ sở.
- `cache=False` — `cache=True` nạp cả dataset vào RAM. Với 1.5 GB trống thì đó
  là cách chắc chắn nhất để hỏng.

Chạy nền, ghi log ra file, để qua đêm.

### Bước 4 — Xuất và nạp

```bash
python scripts/export_classifier_onnx.py ...   # cho classifier
# hoặc, với YOLO:
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=256 opset=18
```

Kèm `model_manifest.json` rồi bỏ vào `models/library/<bước>-<kiến trúc>-<ngày>/`.
Xem `models/README.md`.

## Ba điều tôi nghĩ bạn nên biết trước khi bắt đầu

**1. Fine-tune từ `yolo11n.pt` (COCO), không phải từ model 6.2 hiện tại.**
Model 6.2 hiện tại là bộ **phân loại** mối hàn (mobilenet), còn cái đang thiếu
là bộ **định vị** chân. Hai bài toán khác nhau; không fine-tune cái này thành
cái kia được. Và đánh giá cho thấy model 6.2 hiện tại không đáng làm điểm khởi
đầu — nó gọi `bridge` cho 50,2% ROI thật và 50,3% mảnh board ngẫu nhiên.

**2. Chỉ nhãn "sai" thì không train được.** Bản ghi đánh giá cho biết model
sai *ở đâu*, nhưng train một bộ định vị cần cả **chân đúng**. Hai nguồn bổ sung:
- `scripts/bootstrap_lead_labels.py` vẽ sẵn box từ hình học — **sửa** nhanh hơn
  **vẽ** nhiều lần, và với chân IC thì phần sửa chính là phần giá trị nhất
- Chân 2 đầu (điện trở, tụ) hình học vốn làm đúng — dùng thẳng làm nhãn dương

**3. Có một ứng viên miễn phí chưa loại.** keremberke `yolov8m` đã đo là chịu
được thang chụp 46 µm/px của dự án. Thử nó trên **một board có lỗi thật** mất
10 phút, và nếu nó khoanh trúng thì việc gán nhãn chuyển từ "vẽ từ đầu" sang
"sửa box có sẵn". Xem `Docs/khao_sat/khao_sat_model_huggingface.md`.

## Việc còn thiếu ở phía tôi

- [ ] `scripts/feedback_to_yolo.py` — chuyển bản ghi đánh giá sang dataset YOLO
- [ ] Notebook/`scripts` chạy train tại chỗ với đúng ba tham số ở bước 3
- [x] ~~Đo đỉnh bộ nhớ~~ — đã đo, xem bảng ở trên. Không phải ràng buộc.

Cả hai gạch còn lại là việc nhỏ và làm được ngay khi bạn có board gán nhãn.
