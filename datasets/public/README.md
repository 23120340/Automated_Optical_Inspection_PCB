# Dataset công khai cho bước 6.2

Thư mục này giữ dữ liệu **tải về từ nguồn ngoài**. Ảnh không được commit — xem
`.gitignore`. Mỗi nguồn có tình trạng giấy phép riêng, đọc trước khi dùng.

Tải lại bằng: `python scripts/fetch_public_solder_datasets.py --list`

---

## Đã tải

### `roboflow_solder_leadjoints/` — 2.761 ảnh, 11.033 box ⭐ TỐT NHẤT

`universe.roboflow.com/pcb-vndkd/solder-dbcbh` v3 · **CC BY 4.0**

| | |
|---|---|
| Ảnh | 2.761 (train 2.401 / valid 243 / test 117), **1.257 cảnh gốc** |
| Box | 11.033 — `Bad_podu` 8.745 · `Bad_qiaojiao` 2.288 |
| Tỉ lệ đo được | box trung vị **15 px = 2,3 % bề rộng khung** |
| Màu | chroma 55,9 — **ảnh màu** |

**Đây là nguồn công khai đầu tiên có ĐỦ CẢ HAI: đúng tỉ lệ và có box.**
Board dự án: pad **23 px ≈ 2,2 %** khung. Khớp gần như hoàn hảo — khác hẳn
SolDef_AI (lệch 20×) và Ulger (đúng tỉ lệ nhưng không box).

Kiểm bằng mắt: ảnh màu, IC gull-wing (SOIC/QFP) trên board xanh, **box đặt trên
từng mối hàn chân riêng lẻ** — đúng độ hạt bước 6.2 cần, đúng loại linh kiện như
U201 của dự án.

Hai lớp là tiếng Trung phiên âm: `Bad_podu` (坡度 — thấm thiếc kém/dốc fillet),
`Bad_qiaojiao` (翘脚 — chân vênh, không tiếp xúc pad).

> ⚠️ **Chỉ khoanh mối hàn LỖI.** Một ảnh 28 chân chỉ có 2–7 box. Không có lớp
> `good`, nên train thẳng sẽ ra detector *chỉ tìm lỗi* — dùng được cho 6.2
> (khoanh vùng nghi ngờ), **không** dùng được cho lượt 2 định vị chân (cần mọi
> chân, kể cả chân lành). Muốn có `good` thì tự thêm bằng
> `scripts/build_solder_label_app.py`.

### `fpic_boards_rf100/` — 199 board độ phân giải gốc ⭐ NGUỒN CROP TỐT NHẤT

`universe.roboflow.com/roboflow-100/printed-circuit-board` **v4** · **CC BY 4.0**

Thượng nguồn gần như chắc chắn là **FPIC** (Lu et al., ACM JETC 2023,
arXiv:2202.08414 — 261 ảnh / 93 board, CC BY 4.0): tên board trùng
(`Arty_Top`, `ATTIOT_Bottom`, `XLVDSproSupply_Bottom2`) và taxonomy lớp đặc
trưng cũng trùng (`Capacitor Jumper`, `Resistor Network`, `Test Point`).

| | |
|---|---|
| Ảnh | 672 file, **199 cảnh gốc** (mỗi cảnh lưu 2–4 bản, MAD 0.7 ⇒ chỉ khác nén) |
| Box | 134.047 hộp linh kiện, 20+ lớp |
| Kích thước ảnh | **504…5985 px bề rộng**, native — v4 là bản duy nhất `resize=None` |
| Chroma | 39.1 — ảnh màu |

**Vì sao lấy:** đây là bộ công khai duy nhất tìm được vừa chụp **toàn board**,
vừa có **box từng linh kiện**, vừa giữ **độ phân giải gốc**. Đo cạnh ngắn tuyệt
đối của box: **IC 68% ≥48 px** (trung vị 62), **Connector 77%** (75),
Button 94%, Switch 94%, Pins 50%. Kiểm bằng mắt: chân gull-wing của IC và
transistor, fillet hai đầu điện trở chip, pad xuyên lỗ của connector — đều đọc
được. Board đủ màu (xanh dương, xanh lá, đen, trắng), tức là nguồn đa dạng chứ
không phải một dây chuyền.

> ⚠️ Tụ và điện trở chiếm phần lớn số box nhưng **8% và 3%** qua ngưỡng 48 px.
> Đó là giới hạn quang học, không phải giới hạn của ngưỡng.

### `pcb_packages_winnies/` — 73 cảnh, phân lớp theo KIỂU VỎ

`universe.roboflow.com/winnies-workspace-0yaec/pcb-components-wc8ms` v3 · **CC BY 4.0**

| | |
|---|---|
| Ảnh | 173 file, **73 cảnh gốc**, đồng nhất **1536×2048** |
| Box | 16.632 · **20,9% ≥48 px** |
| Lớp | `SOT23, SOT96, SOT753, SOT223, SOT143, SOT457, SOD123, SOD128, SOD323, SOIC-12/14/16, TSSOP-14/16, MOSFET, Polyfuse_*, Resistor rond, …` |

**Vì sao lấy:** phân lớp theo **kiểu vỏ**, không phải theo chức năng — mà kiểu vỏ
mới là thứ quyết định hình học chân hàn. `SOD123`/`SOD323` là đúng dạng diode
D201/D202 đang lỗi, `Resistor rond` là **MELF hình trụ** — đúng bài toán "tụ/diode
tròn" mà luật hình học 5.5 phải xử lý. Ảnh đủ nét để thấy fillet ở cả hai đầu.

> ⚠️ Chỉ **73 ảnh gốc**. Đủ để bổ sung dạng vỏ, không đủ làm nền chính.

### `roboflow_solder_extra/` — 837 ảnh, 2.511 box

`universe.roboflow.com/pcb-defect/solder-f8m5i` · **CC BY 4.0** · lớp `extra__solder`

Box trung vị **78 px = 3,1 %** khung, ảnh màu (chroma 66,9). Gần dải của dự án,
nhưng chỉ **109 cảnh gốc** sau khi bỏ augment ⇒ đa dạng thấp. Một lớp duy nhất
(thừa thiếc). Dùng bổ sung, không dùng làm nền chính.

---

### `ulger_solder_joints/` — 3.389 crop, 5 lớp

`github.com/furkanulger/solder-joint-dataset` · IEEE TIM 2023, doi 10129988

| Lớp | Số ảnh | Kích thước đo (W med × H med) |
|---|---:|---|
| `normal` | 2.735 | 26 × 27 px |
| `short` | 300 | 82 × 89 px |
| `insufficient_solder` | 148 | 79 × 69 px |
| `shifted_component` | 114 | 136 × 101 px |
| `excessive_solder` | 92 | 51 × 87 px |

**Tỉ lệ: đây là nguồn công khai gần board của dự án nhất.** Crop `normal` trung
vị 26 px cho một mối hàn, so với pad **23 px** đo trên board dự án; crop linh
kiện 136×101 so với 62×58 ⇒ board của họ mịn hơn khoảng 2 lần, suy ra
**~20–25 µm/px** (dự án: 46 µm/px). Đây là bằng chứng công khai cho thấy
46 µm/px không phải ngoại lệ dị thường.

**Dùng được vào gì**
- ✅ Bộ **phân loại** chất lượng mối hàn 6.2 (`good/insufficient/excess/...`)
- ✅ Pretrain backbone đúng dải tỉ lệ
- ✅ Nguồn copy-paste augmentation

**KHÔNG dùng được để train detect**
Chỉ có crop đã cắt sẵn — **không file nhãn, không box, không toạ độ, và không có
ảnh board gốc** nên không dựng lại box được. Ép dùng thì box = toàn khung ảnh,
model học "vật thể luôn chiếm 100% khung", tức không học gì về định vị.

> ⚠️ **Giấy phép: KHÔNG CÓ.** Repo gốc không có file LICENSE (kiểm 2026-08-25:
> GitHub API trả `"license": null`). Mặc định pháp lý là **tác giả giữ toàn
> quyền**. Dùng nghiên cứu nội bộ thì bình thường; **không redistribute, không
> commit lên repo công khai, không dùng thương mại** khi chưa xin phép tác giả.
> Trích dẫn bài IEEE TIM 2023 nếu công bố kết quả.

---

## Đã kiểm và LOẠI

### Khảo sát 2026-08-26 — tìm nguồn để CẮT ẢNH LINH KIỆN

Câu hỏi khác lần trước: không tìm ảnh mối hàn đã cắt sẵn, mà tìm **ảnh board
toàn cảnh có box linh kiện** để tự cắt rồi tự gắn nhãn. Quét 132 project Universe
qua 12 truy vấn, bỏ sơ đồ mạch và breadboard còn 93, rồi lọc theo **độ phân giải
export**.

**Phát hiện quyết định: gần như mọi project Roboflow export ở `640×640 Stretch to`.**
Board 4000×3000 bị nén còn 640×640 (linh kiện 60 px → 10 px) **và méo tỉ lệ** —
tụ tròn thành elip, pad vuông thành chữ nhật. Với bài toán hình học mối hàn thì
điều đó phá hỏng chính thứ cần đo. Chỉ giữ bản `resize=None`.

| Bộ | Ảnh / cảnh | ≥48 px | Phán quyết |
|---|---:|---:|---|
| **roboflow-100/printed-circuit-board v4** | 672 / **199** | 12,1% | ✅ **lấy** |
| **winnies-workspace/pcb-components-wc8ms v3** | 173 / **73** | 20,9% | ✅ lấy |
| clara-y7ocp/pcb-component-detection-odem1 v2 | 855 / 855 | 17,8% | ✗ **trộn lại**: 174 tile trùng y hệt Consolidated + ảnh của FPIC |
| pcb-test-neely/pcb-component-detection v8 | 328 / **40** | 24,7% | ✗ tập con FPIC, chỉ 40 cảnh, đã hạ xuống 1280 |
| obudai-egyetem-nik/pcb-components v6 | 2275 / 43 | 4,8% | ✗ ảnh **400×270**, 43 cảnh |
| rf100-vl-fsod/smd-components v7 | 380 / 380 | **60,6%** | ✗ xem bên dưới |
| marco-filippozzi/smd-component-detection v9 | 5003 native | — | ⏳ **không tải được**: Roboflow không sinh nổi export (trả rỗng sau 60 phút) |

**`rf100-vl-fsod` là cái bẫy đáng nhớ.** Số của nó đẹp nhất bảng — 60,6% box
≥48 px, 380 cảnh riêng biệt, 1280×720 đồng nhất. Nhưng nhìn ảnh thì đó là
**linh kiện rời chụp trên thảm xanh trước khi hàn**: lớp `IC Bottom`,
`Resistor Bottom` là mặt dưới linh kiện chưa gắn, còn `*_Footprint` là **pad
trống chưa có linh kiện**. Không có một mối hàn nào. Chỉ đọc số thì đã lấy nhầm.

### PCB Component Detection Consolidated — đã đo lại 2026-08-26

Kết luận cũ ("sai bài toán") vẫn đúng cho việc **detect chân hàn**, nhưng đo lại
cho việc **cắt ảnh linh kiện** thì: 917 ảnh, 29.669 box, và
`components_data_uncropped/train/images/00001__1024__1648___4120.png`
**trùng từng pixel** với `tests/data/solder_geometry/board_smd_00001.png`
(`np.array_equal → True`) — tức Consolidated chính là nguồn ảnh test của dự án,
đúng tỉ lệ làm việc. Nhưng chỉ **17,1% box** qua ngưỡng 48 px, và
`clara`/`neely` cho thấy nó bị trộn lẫn khắp Universe. `fpic_boards_rf100` phủ
cùng nguồn ở độ phân giải gốc chưa cắt tile, nên lấy bộ đó thay.

### FPIC-Component (bản Dataset Ninja) — giấy phép chặn

19.158 crop linh kiện 768×768, nhưng bản đóng gói của Dataset Ninja ghi
**CC BY-NC-ND 4.0**. ND cấm phân phối bản phái sinh. Bài báo gốc ghi CC BY 4.0;
khi hai nguồn mâu thuẫn thì lấy bản Roboflow (CC BY 4.0, khớp bài báo) và tự cắt.


### PHME Data Challenge 2022 — không có ảnh nào

`github.com/PHME-Datachallenge/Data-Challenge-2022` · CC BY-NC-SA 4.0

Kiểm 2026-08-25, ba lý do độc lập, mỗi lý do đủ để loại:

1. **Không phải dữ liệu ảnh.** Toàn bộ `data/` là CSV nén:
   `SPI_training_{0..3}.csv.zip`, `AOI_training.csv.zip`. Cột theo
   `SPI-ColumnDescription.txt`: `PanelID, ComponentID, PinNumber, PadID,
   PosX(mm), Volume(%), Height(um), Area(%), OffsetX(%), SizeX…` — **số đo, không
   phải pixel**. Không train detector ảnh được.
2. **Sai giai đoạn.** SPI = Solder Paste Inspection, đo **bột hàn trên pad trước
   khi gắn linh kiện**. Bước 6.2 kiểm **mối hàn sau khi hàn**. Cùng lý do đã loại
   PCB-AoI (KubeEdge-Ianvs).
3. **Truy cập + giấy phép.** Experiment khoá mật khẩu, phải đăng ký tại
   phm-europe.org. Giấy phép **NC** (phi thương mại).

### PCB Component Detection Consolidated — sai bài toán

`kaggle.com/datasets/aryanstein/...` — đây là detect **linh kiện**, không phải
mối hàn. Lớp `pads`/`pins` chỉ **186/261 instance trên ~30 ảnh** trong 670 ảnh
train; recall đo được **0.072**. Chi tiết: `docs/dataset_lead_detection.md`.

### SolDef_AI — sai tỉ lệ 20 lần, và đã được dùng tự động rồi

`kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection`

Chụp macro **1–3 µm/px** so với **46 µm/px** của dự án. Đo: model train trên nó
đạt Box mAP50 0.771 trên chính nó nhưng ra **0 box** trên board dự án ở mọi mức
phóng đại 1×–12×. Phóng to bằng phần mềm không tạo ra chi tiết chưa từng chụp.

**Không cần gắn nhãn tay:** `training/kaggle/pcb_solder_defect_v2_kaggle.py` ở
chế độ `public_bootstrap` **tự tải và tự dùng** SolDef_AI qua KaggleHub.

---

## Chưa kiểm được

### ~~Roboflow Universe~~ — ĐÃ KIỂM XONG 2026-08-25

Có API key thì `api.roboflow.com/universe/search?q=...` chạy được (trang web vẫn
403 vì Cloudflare, nhưng API thì không). Đã quét 6 truy vấn, 35 project, đo 5 bộ
đứng đầu bằng `scripts/probe_roboflow_solder.py`:

| Project | Ảnh | Cảnh | box % | chroma | Phán quyết |
|---|---:|---:|---:|---:|---|
| **pcb-vndkd/solder-dbcbh** | 2.761 | 1.257 | **2,3 %** | 55,9 | ✅ **lấy** |
| **pcb-defect/solder-f8m5i** | 837 | 109 | 3,1 % | 66,9 | ✅ lấy, đa dạng thấp |
| nuruls/cold-joint-defect | 806 | 673 | 9,6 % | 15,2 | ✗ gần gấp 5, gần đơn sắc |
| work-6qkmv/pcb-solder-joint | 3.908 | 1.626 | 18,5 % | **0,0** | ✗ macro + **đơn sắc** |
| stuti-garg/joint-hzvjg | 695 | 695 | 50,6 % | 11,1 | ✗ một mối hàn lấp nửa khung |

Bẫy từ khoá: Universe khớp `solder` với `soldier` (ảnh lính), `smd` với biển/nước
và lon 7UP. Script tự lọc bằng danh sách nhiễu.

### Kaggle (SolDef_AI, Consolidated)

Máy chưa có `~/.kaggle/kaggle.json`. Nếu muốn tải về xem tay:
Kaggle → Account → Create New API Token → lưu vào `~/.kaggle/kaggle.json`.

---

## Kết luận cho mục tiêu "model detect mối hàn tốt"

| Nguồn | Đúng tỉ lệ | Có box | Ảnh màu |
|---|---|---|---|
| **roboflow_solder_leadjoints** | ✅ **2,3 %** vs 2,2 % | ✅ **11.033** | ✅ |
| roboflow_solder_extra | ✅ 3,1 % | ✅ 2.511 | ✅ |
| Ulger | ✅ ~20–25 µm/px | ❌ | ✅ |
| SolDef_AI | ❌ lệch 20× | ✅ | ✅ |
| Consolidated | — | ✅ nhưng sai bài toán | ✅ |
| PHME 2022 | — | ❌ không có ảnh | — |

Trước 2026-08-25 kết luận là *"không nguồn nào có cả hai"*. Roboflow Universe đổi
điều đó: **`roboflow_solder_leadjoints` có đủ cả hai**, và là nền train tốt nhất
hiện có cho bước 6.2.

Vẫn còn hai khoảng trống mà dữ liệu công khai không lấp được:

1. **Không có lớp `good`.** Cả hai bộ Roboflow chỉ khoanh mối hàn lỗi. Model học
   từ đó biết "lỗi trông thế nào" nhưng chưa từng thấy nhãn cho mối hàn lành.
2. **Không phải camera của bạn.** Ánh sáng, ống kính, lớp mạ và loại lỗi của dây
   chuyền bạn vẫn là miền chưa được đại diện.

Nên đường đi đúng là **train nền trên `roboflow_solder_leadjoints`, rồi fine-tune
trên ảnh board của chính bạn** gắn nhãn bằng `scripts/build_solder_label_app.py`
— không phải chọn một trong hai.
