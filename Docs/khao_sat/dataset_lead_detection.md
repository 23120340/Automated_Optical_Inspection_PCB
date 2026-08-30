# Khảo sát dataset cho lượt 2 — detect chân/pad

> Soạn 2026-08-21. Khác với `Docs/khao_sat/pcb_aoi_component_datasets.md` (khảo sát nguồn
> nhận dạng **linh kiện**), file này tìm nguồn có nhãn **vị trí chân/pad/mối
> hàn** ở mức board — thứ mà lượt 2 cần.
>
> Quy ước: mục nào **đã kiểm chứng** thì ghi rõ kiểm bằng cách nào. Mục nào chỉ
> đọc được mô tả thì ghi là **chưa kiểm chứng** — đừng coi hai loại như nhau.

> ## ⚠️ Cập nhật 2026-08-28 — kết luận chính của file này ĐÃ BỊ LẬT
>
> Câu "không có dataset công khai nào đủ dùng cho lượt 2" **không còn đúng**.
> Nó đúng với câu hỏi *"có bộ nào gán nhãn sẵn chân hàn không"* — vẫn không có.
> Nhưng nó bỏ sót một đường thứ ba: **lấy ảnh board công khai đúng tỉ lệ rồi tự
> gán nhãn**.
>
> Đã làm xong đường đó: `datasets/train/solder_joint_v1` — **2.054 crop,
> 9.232 box vẽ tay, một lớp `solder_joint`**, cắt từ `fpic_boards_rf100` và
> `pcb_packages_winnies` (cả hai CC BY 4.0) bằng
> `scripts/crop_components_for_labelling.py`, gắn nhãn bằng
> `scripts/build_joint_box_app.py`, đóng gói bằng
> `scripts/pack_joint_detection_dataset.py`. Notebook train:
> `training/kaggle/pcb_joint_locator_kaggle.py`.
>
> Cái ràng buộc vật lý mà file này đo được **vẫn đúng và vẫn là thứ quyết định**
> — chỉ có điều nó là tiêu chí *lọc*, không phải bản án. Ngưỡng dùng để lọc:
> linh kiện phải có cạnh ngắn ≥ 48 px thì mối hàn mới ≥ ~24 px, tức bằng pad đo
> trên board dự án. Ở ngưỡng đó FPIC còn 12,1 % số box và winnies còn 20,9 %.
>
> Phần còn lại của file vẫn đúng và vẫn đáng đọc: các số đo về `pads` recall
> 0.072, về SolDef_AI lệch tỉ lệ 20 lần, và về Ulger không có box đều được dựng
> lại được. Xem `datasets/public/README.md` cho khảo sát mới nhất.

## Kết luận trước (2026-08-21 — xem ô cảnh báo ở trên)

**Không có dataset công khai nào đủ dùng cho lượt 2.** Đường duy nhất đi tới
model là gán nhãn trên board của chính dây chuyền. Các nguồn công khai chỉ có
giá trị làm pretrain hoặc đối chứng.

Lý do không phải "chưa tìm kỹ" mà là ràng buộc vật lý đã đo được: nhãn chân hàn
công khai hoặc **quá hiếm** (30 ảnh), hoặc **sai tỉ lệ chụp** (macro 1–3 µm/px
so với 46 µm/px của dự án).

## Đã kiểm chứng bằng số đo

### 1. `pads`/`pins` trong PCB Component Detection Consolidated

Đọc thẳng từ artifact của dự án (`models/active/detector/model_manifest.json`):

| Lớp | Số instance train | So với |
|---|---|---|
| `pads` | **186** | `capacitor` 7775 |
| `pins` | **261** | `resistor` 7135 |

`rare_image_fraction` = 0.2195 **sau khi đã nhân bản 6 lần**, tức nguyên bản chỉ
khoảng **30 ảnh** có hai lớp này. Kết quả huấn luyện: `pads` recall **0.265**,
`pins` **0.595** — và đó là sau khi notebook v2 đã tăng imgsz lên 1536, bật
`copy_paste`, oversample và kéo dài lịch train.

**Dùng được không:** làm pretrain thì có (đúng lớp, đúng tỉ lệ board). Làm nguồn
chính thì không — 30 ảnh không thể dạy một lớp.

### 2. SolDef_AI — đúng bài toán, sai tỉ lệ

Notebook `soldef-ai.ipynb` (bạn đã tải): YOLO11m-seg, 428 ảnh, val Box mAP50
**0.771**, Mask mAP50 0.766. Nhãn là **từng mối hàn**, có cả box lẫn mask — đúng
loại nhãn cần.

Nhưng đã đo: chạy `best_soldef_AI.pt` lên tile board của dự án cho **0 box** ở
conf 0.25 và 0.10; ở conf 0.05–0.01 ra 1–5 box, toàn nhãn `spike`, tức nhiễu.
Phóng to từng linh kiện **1×, 2×, 4×, 8×, 12× — vẫn 0 box**. Đối chứng: chạy
trên chính ảnh SolDef_AI ra 1–3 box mỗi ảnh, nhãn hợp lý. Model không hỏng.

Nguyên nhân: SolDef_AI chụp macro, mỗi ảnh một linh kiện, khoảng **1–3 µm/px**.
Board của dự án **46 µm/px**. Chênh khoảng 20 lần, và nội suy không tạo ra chi
tiết chưa từng được chụp.

**Dùng được không:** không, cho tới khi có camera macro. Xem
`Docs/thiet_ke/yeu_cau_phan_cung_camera.md`.

### 3. Board của chính dự án — nguồn khả thi nhất

Chạy `scripts/bootstrap_lead_labels.py` lên một tile 1024² thật rồi áp logic
chuyển crop của notebook lượt 2:

| Chỉ số | Đo được |
|---|---|
| Linh kiện lượt 1 tìm được | 38 |
| Crop sinh ra (đều có pad) | 38 |
| Crop trống / quá nhỏ | 0 / 0 |
| Kích thước crop | trung vị **62 × 58 px** |
| **Pad, cạnh ngắn** | trung vị **23 px**, phân vị 10 là 19 px |
| Pad dưới 8 px | **2/90 (2.2%)** |

**Đây là con số đáng mừng nhất trong cả khảo sát.** Pad 23 px là học được. Ngưỡng
cảnh báo của notebook (quá nửa số pad dưới 8 px) còn rất xa.

Lưu ý: đây là box **suy ra từ hình học**, không phải ground truth — kích thước
phản ánh công thức hình học chứ không phải pad thật. Nhưng nó cho biết **thang
đo** đủ để làm việc.

## Tìm được nhưng CHƯA kiểm chứng

Ghi lại để bạn tự kiểm, tôi không xác minh được nội dung.

| Nguồn | Mô tả theo tài liệu | Vì sao chưa kiểm chứng |
|---|---|---|
| [Roboflow Universe — lớp `solder`](https://universe.roboflow.com/search?q=class:solder) | Nhiều dataset cộng đồng có lớp liên quan mối hàn | Roboflow trả HTTP 403 khi fetch. Chỉ đọc được tiêu đề từ kết quả tìm kiếm |
| [smd-component-detection (Roboflow)](https://universe.roboflow.com/marco-filippozzi-siwjn/smd-component-detection) | Detect linh kiện SMD | Như trên. Nhãn nhiều khả năng là **linh kiện**, không phải chân |
| [PCBA-Dataset (GitHub)](https://github.com/ismh16/PCBA-Dataset) | Object detection cho lỗi PCBA | Chưa mở. Khảo sát cũ ghi PCBA-DET chủ yếu nhãn vít/quạt/dây/xước, không phải chân hàn |

## Nguồn đã loại, kèm lý do

- **DeepPCB, PKU PCB defect**: lỗi bare-board (`open`, `short`, `mouse bite`…),
  không có linh kiện đã gắn, không có chân hàn.
- **FPIC / FICS-PCB**: polygon cho **linh kiện** và text, không phải pad. Có ích
  cho lượt 1, không cho lượt 2.
- **PCB-Vision**: mask 3 lớp (IC / tụ / connector), mức linh kiện.
- **PCB DSLR**: chỉ bbox IC, và chỉ cho nghiên cứu phi thương mại.

## Khuyến nghị

1. **Bắt buộc:** gán nhãn board của chính bạn. `scripts/bootstrap_lead_labels.py`
   đã vẽ sẵn box (79 box trên một tile) — **sửa** nhanh hơn **vẽ** nhiều lần.
   Giá trị lớn nhất khi sửa là **thêm box ở chân mà hình học bỏ sót**; đó chính
   là thứ model cần học mà hình học không biết.
2. **Nên làm:** pretrain trên tập con `pads`/`pins` của Consolidated (30 ảnh)
   rồi fine-tune trên board của bạn. Rẻ, và đúng tỉ lệ board.
3. **Nên xin phép:** liên hệ tác giả Ulger để dùng 2.735 crop `normal` làm
   nguồn **copy-paste augmentation** — dán lên nền board của bạn, box biết
   trước chính là biên crop. Đây là cách duy nhất biến một dataset không có
   box thành tín hiệu định vị. Repo không có LICENSE nên mặc định là giữ toàn
   quyền; phải hỏi trước khi dùng thương mại.
4. **Chưa cần:** SolDef_AI, cho tới khi có camera macro.
5. **Đã loại, không cần quay lại:** PCB-SAID — xem mục 4 của bản cập nhật.

## Cần bao nhiêu board?

Không có con số chắc chắn, nhưng có thể ước lượng từ chính số đo trên: một tile
1024² cho **38 crop**. Một board 5144² cho khoảng **960 crop**.

Nút thắt **không phải số crop mà là số board**. Crop từ cùng một board có cùng
ánh sáng, cùng lô hàn, thường cùng loại linh kiện. Notebook vì thế chia tập
**theo board** và chặn nếu có dưới 3 board.

Mốc thực tế để bắt đầu: **10–20 board khác nhau**, ưu tiên khác lô, khác loại
board, khác điều kiện chiếu sáng — hơn là nhiều ảnh của cùng một board.

---

# Cập nhật 2026-08-22 — khảo sát vòng 2

Kết luận cũ **giữ nguyên**: không có dataset công khai nào đủ dùng cho lượt 2,
tự gán nhãn vẫn là đường duy nhất. Nhưng vòng này đóng được mục treo lớn nhất
và tìm thêm một nguồn đáng giá.

## 4. PCB-SAID — đã kiểm chứng, LOẠI HẲN

Tải được PDF bằng `curl` kèm User-Agent trình duyệt (WebFetch bị 403, curl 200).
Nguyên văn mục 3.1:

> "PCB-SAID comprises 175 high-resolution RGB images **(native resolution
> 640 × 480)** aggregated from multiple public sources, including **electronics
> enthusiast forums, open-source hardware repositories, and automated web
> crawls** filtered for PCB content."

| Tiêu chí | Thực tế |
|---|---|
| Tỉ lệ chụp | **Không có camera setup nào.** Ảnh cào từ web, 640×480. Không có µm/px và không suy ra được |
| Đơn vị gán nhãn | **Theo linh kiện.** 66 lớp = mỗi lớp một cặp (loại linh kiện, trạng thái lắp). Chỉ `Short Circuit` (56 instance) là box ở mức mối hàn |
| Tải về | **Không link, không DOI, không giấy phép.** Bài ghi: *"will be made publicly available upon request"* |

Đây là dataset **cùng loại với lượt 1**, không phải lượt 2. Khảo sát vòng 1 xếp
nó là "ứng viên đáng xem nhất" — **sai**, và nay đã kiểm chứng là ngõ cụt.

## 5. Ulger solder-joint-dataset — ĐÚNG TỈ LỆ, nhưng KHÔNG CÓ BOX

`github.com/furkanulger/solder-joint-dataset` (bài IEEE TIM 2023, doc 10129988).
Đã tải ảnh thật và **đo kích thước pixel**, không đọc mô tả:

| Lớp | Số ảnh | Kích thước đo (n=12/lớp, min/trung vị/max) |
|---|---|---|
| `normal` | 2.735 `.tiff` | **11/29/39 px** — mỗi ảnh một mối hàn đơn |
| `excessive_solder` | 92 | 43/66/147 × 66/90/155 |
| `insufficient_solder` | 149 | 32/54/133 × 32/60/169 |
| `shifted_component` | 114 | 77/**139**/198 × 44/**100**/159 — cả linh kiện |
| `short` | 300 | 35/85/113 × 26/94/110 |
| **Tổng** | **3.390** | khớp chính xác README |

Crop `normal` trung vị **29 px** cho một mối hàn — cùng bậc với pad **23 px**
đo được trên board của dự án. Crop linh kiện trung vị 139×100 px so với 62×58 px
của dự án ⇒ board của họ mịn hơn khoảng 2 lần, suy ra **~20–25 µm/px**.

**Đây là bằng chứng công khai đầu tiên cho thấy 46 µm/px không phải ngoại lệ dị
thường** — có người khác đã làm bài toán mối hàn ở đúng dải này.

**Vì sao vẫn không train lượt 2 được:** cấu trúc là thư mục-theo-lớp, **không có
file nhãn nào** — không box, không mask, không toạ độ. Và **không có ảnh board
gốc**, chỉ có crop đã cắt sẵn, nên không dựng lại box được. Ép dùng thì box =
toàn bộ ảnh, model học được "vật thể luôn chiếm 100% khung", tức không học gì về
định vị. Repo **không có LICENSE** (đã kiểm: HTTP 404) ⇒ mặc định giữ toàn quyền.

**Dùng được vào:** pretrain backbone đúng tỉ lệ · nguồn copy-paste augmentation ·
sau này làm bộ phân loại chất lượng mối hàn cho bước 6.2.

## 6. Các nguồn khác

| Nguồn | Vì sao không dùng |
|---|---|
| **PCB-AoI (KubeEdge-Ianvs)** | 1.271 ảnh **SPI** — bột hàn trên pad, **chưa gắn linh kiện**. Sai giai đoạn |
| **PCB-Defect (Mendeley, DOI 10.17632/vdj74sngvn.1)** | ~15,9 µm/px (scanner 1600 dpi), CC BY 4.0 — đúng dải tỉ lệ, nhưng **bare board**. `missing pad` là *lỗi*, không phải *vị trí pad* |
| **IEEE DataPort `10.21227/fped-0p25`** | Trang ghi nguyên văn *"Files have not been uploaded for this dataset"*. Bản ghi rỗng |
| **HF `aimmifm/PCBA_Standard-to-Real`** | **VQA**, không phải detection. API trả 401 restricted |
| **openAOI** | Chỉ có code + `yolov8s.pt`, **không có dataset** |
| **Zenodo** | Truy vấn `"solder joint"` lọc `type=dataset` trả về **đúng 1 kết quả**, là dataset hồng ngoại công-tơ điện. Đã tra cạn |
| **Papers-with-Code** | **Đã đóng cửa 24/07/2025**. Không còn là nguồn tra cứu |
| **Roboflow Universe** | **Vẫn chưa kiểm chứng được.** WebFetch 403, curl 403 (Cloudflare "Just a moment…"), API 401 thiếu key. Đây là khoảng trống duy nhất còn lại |

**Bẫy:** kết quả tìm kiếm mô tả "MF-PCBA là dataset phân cấp cho PCB defect ở
mức pin/component/board" — **sai hoàn toàn**. MF-PCBA là dataset **hoá học**
(Multi-Fidelity PubChem BioAssay). Đừng mất thời gian.

### Nếu muốn tự kiểm Roboflow

Cần một API key miễn phí, mất 2 phút:

```
curl "https://api.roboflow.com/<workspace>/<project>?api_key=<KEY>"
```

Trả về số ảnh, danh sách lớp, số instance mỗi lớp.

## 7. Xác nhận kiến trúc — đáng giá hơn cả dataset

Bài *"Deep learning-based solder joint defect detector"* (Int. J. Adv. Manuf.
Technol. 137:5133–5147, 2025), open access tại `https://d-nb.info/1370145357/34`,
mô tả **đúng kiến trúc lượt 2 của dự án**, đã chạy trên dây chuyền thật:

> "the ROI cannot be exactly adjusted to the solder joint to absorb the position
> uncertainties, which means that an additional step needs to be included: **the
> finding of the solder joint bounding boxes in the cropped images**. […] since
> **the search space is already restricted by an initial ROI, a lightweight NN
> like YOLO V4 tiny is well-suited**."

Ba điều rút ra:

1. **"Hình học không biết chân nằm đâu" là vấn đề công nghiệp đã được thừa
   nhận**, không phải đặc thù dự án này. Họ gọi box thô là *padded ROI*, box tinh
   là *adjusted ROI* — đúng cặp khái niệm của bước 5.5.
2. **Họ chọn model tí hon**, lý do y hệt: không gian tìm kiếm đã bị crop thu hẹp.
3. Dataset của họ **không phát hành** (`Materials availability: Not applicable`).

## 8. Chọn model cho lượt 2 — đã đo trên board thật

> Mục này **sửa** đề xuất ban đầu của khảo sát (imgsz 128–160, gom lô 64–256).
> Cả hai con số đó đều sai khi đem đo.

### imgsz = 256

Chạy detector lượt 1 lên board thật (1832×2560, 36 linh kiện), áp
`component_crop_window`, rồi xem một `imgsz` cố định làm gì với pad 23 px:

| imgsz | crop bị **thu nhỏ** | pad tụt dưới 8 px | ms/crop |
|---|---|---|---|
| 128 | 4/36 | **1** | — |
| 160 | 4/36 | **1** | — |
| **256** | **2/36** | **0** | **19.6** |
| 640 | 0/36 | 0 | 123.8 |

Cạnh dài của crop trải từ **25 px đến 462 px** — trung vị 48, p90 **168**, p99
**428**. Vì thế **không được chọn imgsz theo trung vị**: imgsz nhỏ thì phóng to
crop bé (vô hại, chỉ phí tính toán) nhưng **thu nhỏ crop lớn** — mà crop lớn
chính là IC và connector, những thứ **nhiều chân nhất**. 256 là mức nhỏ nhất còn
giữ mọi pad trên 8 px, và **nhanh hơn 640 khoảng 5,2 lần**.

### Gom lô = 4, không phải 64

64 crop qua yolo11n @ imgsz 256, CPU máy này:

| Cỡ lô | ms/crop |
|---|---|
| 1 | 28.9 |
| 2 | 22.6 |
| **4** | **19.6** |
| 8 | 23.5 |
| 16 | 28.6 |
| 32 | 31.1 |
| 64 | 33.2 |

Lô nhỏ khấu hao được chi phí thiết lập mỗi lần gọi; lô lớn mất vào lưu lượng bộ
nhớ nhiều hơn phần thắng, vì backend CPU vốn đã chia luồng **bên trong** một
ảnh. **Lô 64 là cấu hình tệ nhất đo được** — đúng thứ khảo sát ban đầu đề xuất.

1,47× ở lô 4 đáng lấy, nhưng **không phải nút thắt**: hạ imgsz 640→256 một mình
đã được 5,2×. GPU sẽ đẩy điểm tối ưu này lên cao nhiều; **phải đo lại** trước
khi tăng.

Đã cài: `LeadDetectionConfig.batch_size = 4`, `LeadDetector` nhận thêm
`detect_batch` **tuỳ chọn**. Detector không có nó vẫn chạy từng crop.

### yolo11n, không phải yolo11s

Với 10–20 board tự gán nhãn: ~10–20k crop nhưng chỉ **10–20 cảnh độc lập**.
2,6M tham số đã thừa cho hai lớp vật thể dạng đốm. Model lớn hơn sẽ học thuộc
danh tính board thay vì hình dạng pad. Họ DETR đặc biệt sai chỗ: cần lịch train
dài và dữ liệu lớn.

**Không chọn:** YOLOv4-tiny (backbone cũ hơn, dù bài 2025 dùng) · YOLO11n-seg
(mask fillet có ích cho *chấm điểm* 6.2 nhưng đắt gấp nhiều lần khi gán nhãn —
định vị trước, phân đoạn sau) · bất kỳ model nào >5M tham số.

**Phương án B nếu box mAP không lên:** model heatmap/keypoint nhỏ ở 96–128².
Pad là đốm kích thước gần cố định khi đã biết lớp linh kiện từ lượt 1, nên dự
đoán **tâm** + kích thước cố định thường tiết kiệm mẫu hơn hồi quy box kiểu
anchor/DFL ở mức 23 px, và bỏ hẳn NMS. Nhược điểm: adapter Ultralytics không
dùng được, phải viết code riêng.

**Đòn bẩy rẻ:** lượt 1 đã biết lớp linh kiện. Dùng làm tiên nghiệm — một model
chung + hậu xử lý theo **số pad kỳ vọng** suy từ topology lớp. Nên làm trước vì
không nhân số nhãn phải gán.

---

## Bổ sung 2026-08-23 — 52 model "PCB defect" trên Hugging Face

Chi tiết ở `Docs/khao_sat/khao_sat_model_huggingface.md`. **Kết luận đã được sửa trong
cùng ngày** — bản đầu loại keremberke là sai, xem mục "ĐÍNH CHÍNH" ở cuối file
đó.

**keremberke YOLOv8m: chưa loại, đang chờ một board lỗi thật để kết luận.** Đo
lại cho thấy ảnh của họ ở **~33 µm/px** (không phải macro — trước đó tôi nhầm
"640×480" là thang chụp, trong khi đó chỉ là số điểm ảnh). Model suy giảm rất
từ tốn theo thang: recall 0.595 ở 33 µm/px xuống 0.544 ở 46 µm/px. Và trên 6
ảnh chụp thật chứa 38 lỗi thật, thu về 46 µm/px, nó đặt **36/36 box vào đúng
vùng có lỗi**.

Nó không ra box trên board của dự án vì **board đó là board chuẩn, không có
lỗi**. Điều chưa chứng minh được là **miền ảnh**: board của bạn, camera của
bạn, lỗi thật của dây chuyền.

Bảng tổng kết các nguồn đã kiểm chứng:

| Nguồn | Trạng thái |
|---|---|
| SolDef_AI | Loại — sai tỉ lệ 20 lần; ở 46 µm/px chỉ ra 6 box, toàn `spike` |
| PCB-SAID | Loại — không có link tải, nhãn theo linh kiện |
| Ulger | Loại — đúng tỉ lệ nhưng **không có box** |
| **keremberke** | **Chưa loại** — chạy được ở 46 µm/px, cần thử trên board lỗi thật |
| Roboflow Universe | **Chưa kiểm chứng** — có nhãn `Dry_joint`/`Cold Solder` nhưng cần API key |
