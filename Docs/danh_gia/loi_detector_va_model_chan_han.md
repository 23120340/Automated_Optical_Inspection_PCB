# Hai câu hỏi ngày 2026-08-24: lỗi của detector, và model cho chân mối hàn

> Mọi con số dưới đây đo trên `pcb03.jpg` — **đúng tấm ảnh bạn đã ghi nhận lỗi**,
> qua đúng cấu hình tiền xử lý ghi trong bản ghi đó (digest ảnh khớp:
> `1ea6908a…f9bc7e`). Chạy lại được bằng các bước ghi ở cuối bài.

---

## Phần 1 — Cái box `diode` bạn đánh dấu

Bạn ghi nhận một box tại `(738, 334, 777, 444)`, model nói **`diode` 0.32**,
loại lỗi *sai nhãn*.

### Nó là cái gì

Không phải một con diode bị nhận nhầm. Nó là **36 px cuối bên phải của con
SOIC-14 U1** — cái box đó nằm đè lên rìa phải của chính con IC.

### Vì sao

Ảnh 2000×1333 nên bước 4 chia làm 2 tile, **đường ranh rơi vào x = 737**. Con
U1 trải từ x 601 đến 773, tức là bị đường ranh cắt đôi. Chạy riêng từng cửa sổ:

| Cửa sổ nhìn thấy gì | Model trả lời |
|---|---|
| Toàn board (1 lần) | `ic` 0.434 — **không có diode** |
| Tile trái (thấy trọn U1) | `ic` 0.62 — **không có diode** |
| **Tile phải (chỉ thấy 36 px cuối)** | **`diode` 0.32** *và* `ic` 0.276 |

Cho xem đúng một mẩu chữ nhật sẫm có kim loại hai đầu thì model trả lời
"diode" — một câu trả lời hợp lý cho một câu hỏi sai.

### Vì sao bộ lọc trùng không bắt được

Hai cơ chế lẽ ra phải bắt, cả hai đều trượt:

1. **Lọc lồng nhau** (`containment_ios_threshold` 0.80). Box `diode` nằm lọt
   **89,7 %** trong box `ic` — thừa ngưỡng. Nhưng dòng đầu của nó là
   `if not same_class: return False`. `diode` ≠ `ic`, nên không xét.
2. **NMS khác lớp** (`cross_class_iou_threshold` 0.70). IoU thật là **0,169**.

Và điểm 2 **không phải chuyện hạ ngưỡng cho xong**. Một box nằm lọt trong box
khác có IoU bị chặn trên bởi tỉ lệ diện tích:

```
IoU ≤ dt_nhỏ / dt_lớn = (39×110) / (172×130) = 0,19
```

Ngưỡng 0,70 là **không thể với tới** với mọi mảnh vụn kiểu này. Còn hạ ngưỡng
xuống 0,19 thì mọi linh kiện nằm cạnh nhau sẽ bị gộp.

Trớ trêu nhất: tile phải trả về **cả hai** nhãn cho cùng mẩu đó — `diode` 0.32
và `ic` 0.276. Lọc lồng nhau cùng lớp đã xoá cái `ic` (đúng), và giữ lại cái
`diode` (sai). **Nhãn đúng bị xoá, nhãn sai sống sót.**

### Đã sửa thế nào

Thêm `TilingConfig.drop_cross_class_edge_fragments` (mặc định bật). Một box bị
bỏ khi **cả ba** điều sau cùng đúng:

- nó từ một tile, và **chạm mép tile đó** → tile chỉ nhìn thấy vật bị cắt;
- **tâm nó nằm ngoài vùng sở hữu** của tile → cửa sổ khác mới là cửa sổ có
  quyền trả lời ở đó;
- nó nằm trong một box khác quá `containment_ios_threshold`.

Điều kiện giữa là điều kiện giữ an toàn. Chính detector này có hai lớp `pads`
và `pins` **sinh ra để nằm lồng trong** box `ic`; thiếu phép thử sở hữu thì luật
mới sẽ xoá đúng những thứ lượt 2 cần.

### Kết quả

| | trước | sau |
|---|---|---|
| số box | 69 | **68** |
| `diode` giả | 1 | **0** |
| `resistor` / `capacitor` / `ic` | 43 / 21 / 4 | **43 / 21 / 4 — y nguyên** |

Đúng một box bị bỏ, và đó là đúng box bạn đánh dấu.

### Điều nên nói rõ

Soi cả 69 box thì **chỉ có một box** hội đủ bốn dấu hiệu (từ tile, chạm mép,
ngoài vùng sở hữu, lồng khác lớp) — và đó chính là nó. Con U2 cũng là SOIC-14
y hệt, nằm cách xa đường ranh, và **không** sinh box giả nào.

Nghĩa là: **trên tấm ảnh này detector chạy tốt**, và cái bạn bắt được là lỗi
của khâu ghép tile chứ không phải của model. Nhưng đây là **một** board — một
board không đủ để nói lỗi này hiếm hay thường. Ghi nhận thêm vài board nữa sẽ
trả lời được.

---

## Phần 2 — Chân mối hàn có dùng model không

**Không.** Bước 5.5 suy ROI chân từ box linh kiện cộng topology của lớp, bằng
hình học thuần. Đúng như bạn đoán.

### Đã lấy model Hugging Face về thử

Rà toàn bộ Hugging Face: `solder`, `pad`, `lead`, `pin`, `smd`, `smt`, `bga`,
`qfp`, `aoi`, `pcb` — 199 kết quả cho "pcb", 26 cái là model thị giác. Gần như
tất cả là **lỗi bo trần** (DeepPCB/HRIPCB: `missing_hole`, `mouse_bite`,
`short`…) hoặc lỗi hàn (keremberke), **không cái nào định vị chân**.

Trừ đúng một cái: **`JcProg/PCBInspect-AI`** (YOLOv12m, MIT) — có hẳn lớp
`Lead` và `SolderJoint`. Đã tải về chạy thật.

> Model card ghi lớp là `0=Background, 1=MountingHole…`; checkpoint thật là
> `0=MountingHole, 1=ComponentBody, 2=SolderJoint, 3=Lead`. Card lệch một bậc —
> đọc từ file, đừng đọc từ card.

### Thước đo

"Chân có nằm trong ROI nào không" là thước đo vô nghĩa: một dải trùm cả 7 chân
vẫn tính là trúng cả 7 — mà đúng cái dải đó mới là thứ khiến không chấm điểm
được từng mối hàn. Nên thước đo là: **một chân chỉ được tính khi có một ROI phủ
nó mà ROI đó không phủ chân nào khác.**

Mốc thật: 4 con IC trên `pcb03.jpg`, 50 chân, nội suy đều trên hai hàng chân của
box rồi kiểm lại bằng mắt trên ảnh chồng box.

### Đầu tiên tôi đo sai, và nó suýt dẫn tới kết luận ngược

Chạy model ở `imgsz=896` — **đúng kích thước nó được train** — chỉ được 58 %.
Quét lại mới thấy đó là thiết lập tệ nhất trong ba cái:

| imgsz | conf 0.10 | 0.15 | 0.25 | 0.40 |
|---|---|---|---|---|
| **640** | **86 %** | 82 % | 78 % | 70 % |
| 896 *(cỡ train)* | 70 % | 60 % | 58 % | 48 % |
| 1280 | 8 % | 4 % | 2 % | 2 % |

Crop linh kiện chỉ ~230 px; kéo lên 896 làm chân to hơn hẳn thang model từng
thấy, kéo lên 1280 thì sụp hẳn. Lấy cỡ train làm mặc định là phản xạ đúng trong
hầu hết trường hợp, ở đây thì sai.

### Kết quả — và cái quan trọng hơn model

| | model HF (640/0.10) | **hình học, `split_pins` BẬT** | hình học, mặc định cũ |
|---|---|---|---|
| U3 (SOIC-14) | 13/14 | **14/14** | 0/14 |
| U2 (SOIC-14) | 13/14 | **14/14** | 1/14 |
| SOIC-8 (dựng đứng) | 4/8 | 4/8 | 0/8 |
| U1 (SOIC-14) | 13/14 | **14/14** | 1/14 |
| **tổng** | 43/50 = **86 %** | **46/50 = 92 %** | 2/50 = **4 %** |

**Thứ cần sửa không phải là tải model về, mà là một cái cờ đã có sẵn trong dự
án và đang tắt.** `split_pins` bật lên là ROI-một-chân đi từ 4 % lên 92 % — hơn
model, mà không tốn thêm 40 MB lẫn thời gian suy luận nào.

### Vì sao trước đây nó tắt, và vì sao lý do đó không đứng được

Comment trong code ghi: *"cầu chì nối hai chân kề nhau, nên dải thường là đơn
vị kiểm tốt hơn một chân lẻ"*. Đo lại thì:

| | tắt | **bật** |
|---|---|---|
| ROI chứa **đúng một** chân | 3 | **48** |
| ROI trùm nhiều chân | 9 | 5 |
| **khe giữa hai chân được phủ** | 42/42 | **42/42** |
| ROI cho 20 linh kiện 2 chân | 60 | **60** |

`pin_padding_ratio` (0.35) đã nới mỗi ROI chân sang tới khe với chân bên cạnh,
nên **cầu chì không mất chỗ nào** — nỗi lo kia đã được xử lý sẵn ở chỗ khác.
Linh kiện 2 chân không đổi một ROI nào, vì tách chân chỉ áp cho hình học
`multi_pin`.

**Đã bật mặc định**, kèm test giữ lại cả hai nửa phép đo (một ROI một chân, và
khe vẫn được phủ) để lần sau không ai tắt lại mà không có bằng chứng.

### Model đó có nên dùng không

**Chưa.** Trên board này nó không cho thêm gì so với cái cờ vừa bật, mà đắt
hơn. Nhưng chưa nên loại hẳn:

- 86 % là con số thật của một bộ dò *đã học*, không phải suy hình học — nó có
  thể trụ được ở chỗ hình học gãy (đế lạ, linh kiện xoay, chân bị che);
- nó cắm vừa `LeadDetector` của lượt 2 mà không phải sửa gì;
- weights nằm sẵn ở `scratchpad/hf_lead/pcbinspect_yolov12m.pt`
  (sha256 `ed31d287…86ed5`), muốn thử lại là chạy được ngay.

### Chỗ cả hai đều gãy

**Con SOIC-8 dựng đứng: cả hai chỉ được 4/8.** Model dán nhãn chân của nó thành
`ComponentBody`; hình học thì đặt dải sai trục. Đây là ứng viên rõ nhất cho lượt
gán nhãn đầu tiên trong `Docs/ke_hoach/ke_hoach_fine_tune_cuc_bo.md` — và có liên quan
tới `estimate_orientation` (hiện tắt, xem `config.py`).

---

## Chạy lại

```bash
# Phần 1 — mảnh vụn khác lớp
.venv/Scripts/python -m pytest tests/test_tiling.py -q

# Phần 2 — tách chân và khe cầu chì
.venv/Scripts/python -m pytest tests/inspection/test_solder_joints.py -q
```

Bản ghi đánh giá gốc: `feedback/model_feedback.jsonl`, mục
`0fbd27864df143d1bd423e3f161bb480`.
