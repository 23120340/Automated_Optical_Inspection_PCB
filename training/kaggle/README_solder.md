# Train model bước 6.2 trên Kaggle

Notebook: [pcb_solder_defect_kaggle.ipynb](pcb_solder_defect_kaggle.ipynb)
(nguồn percent-format: [pcb_solder_defect_kaggle.py](pcb_solder_defect_kaggle.py),
build lại bằng `python scripts/build_notebook.py training/kaggle/pcb_solder_defect_kaggle.py`).

Đầu ra: `best.onnx` + `model_manifest.json` trong
`/kaggle/working/pcb_solder_defect_artifacts.zip`. Tải về, nạp vào app ở sidebar
**Model kiểm tra mối hàn 6.2**, không cần đổi gì khác.

## Đã sửa: SolDef_AI đọc ra 0 record

Lần chạy đầu trên Kaggle dừng ở cell "Ma trận phủ taxonomy" với:

```
SystemExit: Chỉ còn 0 lớp đủ dữ liệu ([]).
```

Nguyên nhân: `SolDef_AI/Labeled/` chứa 428 cặp `<tên>.jpg` + `<tên>.json` —
sidecar **LabelMe** (đúng như bài báo mô tả: *"manually annotated using
LabelMe... a JSON file containing all the created masks"*). Bộ đọc dataset ban
đầu chỉ nhận 4 layout (folder-per-class, COCO, CSV, YOLO), không nhận dạng
được LabelMe nên coi cả 428 ảnh là "unknown" và bỏ qua toàn bộ, kéo theo mọi
lớp về 0 mẫu.

Đã thêm bộ đọc LabelMe (nhận cả `rectangle`, `polygon`, `circle`) vào cả
`aoi_pipeline/grading/datasets.py` và bản nhúng trong notebook. Đã kiểm chứng
bằng fixture mô phỏng đúng cấu trúc `SolDef_AI/Labeled/*.json` +
`Dataset/CS1..CS7`: notebook chạy trọn 12 cell, export ra `best.onnx` hợp lệ.

## Đã sửa: 4192 nhãn `component misalignment` bị vứt

Lần chạy 3 nguồn đầu tiên cho 7635 record / 6 lớp, nhưng cell đọc dữ liệu in:

```
!! KHÔNG ÁNH XẠ ĐƯỢC: {'component misalignment': 4192, 'solder residue': 619,
                       'charred solder': 275}
```

`component misalignment` (có **dấu cách**, normalize thành
`component_misalignment`) là **nhóm lớn nhất trong toàn bộ lần ghép** và là
nguồn `shift_component` đáng kể duy nhất tìm được ở bất cứ đâu — trong khi
`shift_component` đang có **0 mẫu** và bị loại khỏi `class_names`. Nó bị bỏ
lặng lẽ chỉ vì thiếu một dòng trong `LABEL_MAPS`. Đã thêm.

Hai nhãn còn lại vào `IGNORE` **kèm lý do**, không map bừa:
- `solder residue` (619): cặn flux là lỗi **vệ sinh**, không phải lỗi hình dạng
  mối hàn; gộp vào `excess` là giấu nhiễm bẩn sau một nhãn nói về lượng thiếc.
- `charred solder` (275): thiếc cháy là lỗi **quá nhiệt**; lớp trông giống nhất
  là `cold` lại là lỗi **thiếu nhiệt** — map vào đó là dạy model ngược hẳn
  nguyên nhân vật lý.

## Đã sửa: rò rỉ val do Roboflow nhân bản ảnh augment

Roboflow đổi tên mọi ảnh export thành `<gốc>_<ext>.rf.<md5>` và sinh **một file
cho mỗi bản augment**. Ba bản augment của cùng một tấm ảnh có ba stem khác nhau,
nên với `group = image.stem` chúng thành **ba group riêng** — bản 1 có thể vào
train trong khi bản 2 vào val. Đó đúng là kiểu rò rỉ mà việc chia theo group
sinh ra để ngăn, và nó thổi phồng mọi con số báo cáo mà không để lại dấu vết.

Đã thêm `source_group()` gộp các bản augment về đúng ảnh gốc (vô hại với mọi
nguồn khác — tên không khớp mẫu Roboflow được giữ nguyên). Cell đọc dữ liệu giờ
**in ra bằng chứng**: nếu số file ảnh nhiều hơn số group, nó báo tỉ lệ
bản/ảnh. Nếu dataset của bạn không có augment thì con số không đổi và không mất
gì.

## Nhãn thật của SolDef_AI

**Cập nhật (2026-08): nhãn chữ thật đã quan sát được trên một lần Run All
thật**, khác hẳn suy đoán ban đầu từ thuật ngữ bài báo:

| Nhãn thô trong JSON | Số mẫu | Map vào |
|---|---:|---|
| `good` | 136 | `good` |
| `no_good` | 114 | **chưa map** — xem lý do dưới |
| `poor_solder` | 31 | **chưa map** — xem lý do dưới |
| `spike` | 29 | `excess` |
| `exc_solder` | 43 | `excess` |

`misalignment` chưa từng xuất hiện trong 353 annotation đọc được, nên
`shift_component` hiện có 0 mẫu từ nguồn này dù `LABEL_MAPS` có sẵn đồng nghĩa
— không phải lỗi, chỉ là annotator không dùng thuật ngữ đó (hoặc lệch vị trí
không nằm trong 428 ảnh này).

`no_good` (nhóm lớn nhất) và `poor_solder` **cố tình chưa map**: `no_good` đọc
như nhãn "không đạt" chung chung — rất có thể được dùng cho bất kỳ lỗi nào
không khớp cụ thể `poor_solder`/`spike`/`exc_solder`, tức là hỗn hợp nhiều loại
lỗi khác nhau trộn dưới một tên chứ không phải một lỗi riêng; map bừa vào một
lớp sẽ làm bẩn chính lớp đó. `poor_solder` mơ hồ giữa `insufficient` và `cold`
(IPC gọi cả hai là "poor wetting"). Notebook giờ có cell mới **"Xem mẫu ảnh
cho nhãn chưa map được"** ngay sau bước đọc dataset — tự cắt theo đúng bbox
(kể cả polygon/circle của LabelMe) và vẽ vài ảnh mẫu cho mỗi nhãn này. Nhìn
ảnh rồi quyết định map vào đâu, sau đó thêm vào `LABEL_MAPS["soldef_ai"]` ở
cell trước và chạy lại — đừng đoán tiếp từ tên nhãn.

Với 2 lớp map được (`good` 136 + `excess` 72), pipeline chạy trót lọt hết 12
cell tới `best.onnx` hợp lệ — trước đây bị chặn ở cell "Ma trận phủ taxonomy"
vì chỉ có 1 lớp `good` đủ dữ liệu. Nếu bạn giải quyết được `no_good`/`poor_solder`,
tổng dữ liệu train tăng đáng kể (thêm tới 145 mẫu, gần gấp đôi hiện tại).

Thư mục `Dataset/CS1..CS7` vẫn chưa khám phá — có thể chứa thêm ảnh chưa được
đưa vào. Nếu Run All cho thấy `soldef_ai` chỉ đọc được một phần nhỏ trong 428
ảnh, đó là nơi cần nhìn tiếp.

## Trước tiên: không có dataset công khai nào đủ

Khảo sát tháng 8/2026. Đây là kết luận trung thực, không phải lời mở đầu:

| Nguồn | Phủ | Trạng thái |
|---|---|---|
| [SolDef_AI](https://www.kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection) | good, insufficient, excess, **shift_component** | Tốt nhất. 1150 ảnh linh kiện SMT, 3 góc nhìn, từ bài [MDPI JMMP 2024](https://doi.org/10.3390/jmmp8030117). Nguồn peer-reviewed **duy nhất** tìm được có gán nhãn lệch vị trí linh kiện |
| [ouvic215/Soldering-Data-Annotation-boarding](https://huggingface.co/datasets/ouvic215/Soldering-Data-Annotation-boarding) | bridge, excess, missing_solder | 1522 ảnh 512×512. **Không license, không nguồn gốc.** Repo anh em tên `...-ControlNet` ⇒ nghi dữ liệu sinh |
| [AndyLiu0104/Soldering-Data-Tiny-…](https://huggingface.co/datasets/AndyLiu0104/Soldering-Data-Tiny-More-Data-with-appearance-hole-micro-bridge-0801) | bridge, excess, missing_solder | 10469 ảnh nhưng chỉ **36–144 px** — dưới ngưỡng đọc được fillet. Cùng nghi vấn dữ liệu sinh |
| [Roboflow soldering-defects](https://universe.roboflow.com/search?q=class:solder) | **cold**, bridge, insufficient | Nguồn công khai duy nhất có cold solder, nhưng chỉ vài trăm ảnh, chất lượng không đồng đều |

**Bị loại thẳng — đừng nối nhầm:**

| Dataset | Vì sao loại |
|---|---|
| DeepPCB, HRIPCB / PKU-Market-PCB, DsPCBSD+, `akhatova/pcb-defects` | Lỗi **board trần** (open/short/mousebite/spur/copper/pinhole). Board chưa gắn linh kiện, không có mối hàn nào. Bài toán khác hẳn nhưng rất hay bị trích dẫn nhầm |
| AXI_PCB | Ảnh **X-quang**, không dùng được cho AOI quang học |
| [PCBSPDefect](https://github.com/cairs-project-5/PCBSPDefect) | Chưa phát hành: "will be available once the paper is published" |
| PCB-AoI (KubeEdge) | Kiểm tra **kem hàn trước reflow** (SPI), không phải mối hàn sau hàn |

Vì vậy phải **ghép nhiều nguồn**. Notebook làm việc đó và bắt buộc quá trình ghép
phải kiểm toán được.

## Chuẩn bị trên Kaggle

1. Tạo notebook mới, import `pcb_solder_defect_kaggle.ipynb`.
2. **Add Input** ít nhất một dataset. Bắt đầu bằng SolDef_AI:
   `mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection`.
3. **Roboflow** (`roboflow_soldering`): mở project trên
   [universe.roboflow.com/search?q=class:solder](https://universe.roboflow.com/search?q=class:solder),
   xem preview — có khung box khoanh lỗi thì Export **YOLO** (bất kỳ biến thể
   nào, `read_yolo()` đọc được mọi bản YOLO vì định dạng .txt annotation không
   đổi giữa các phiên bản), không có box mà chỉ gán nhãn cả ảnh thì Export
   **Folder Structure**. Tải zip về, upload thành Kaggle Dataset, Add Input.
4. **Hugging Face** (`hf_soldering_boarding`,
   [ouvic215/Soldering-Data-Annotation-boarding](https://huggingface.co/datasets/ouvic215/Soldering-Data-Annotation-boarding)):
   không có file rời để tải tay — dataset lưu dạng bảng `{"image", "text"}`
   (đã xác minh qua Hugging Face datasets-server API: cột `text` đúng là nhãn
   thô `bridge`/`excess_solder`/`empty`/...). Đặt `enabled: True` cho nguồn
   này ở CONFIG cell đầu, cell **"1b. Tải nguồn Hugging Face"** ngay sau đó tự
   tải qua thư viện `datasets` và ghi ra ảnh theo layout folder-per-class —
   không cần Add Input tay. **Nhắc lại**: nguồn này không license, nghi dữ
   liệu sinh (repo anh em cùng tác giả tên `...-ControlNet`) — chỉ dùng bổ
   sung, đừng để nó chiếm đa số một lớp.
5. Chọn **GPU T4 x2** hoặc mới hơn. Bật **Internet** (bắt buộc nếu dùng nguồn
   Hugging Face; cũng cần để TorchVision tải ImageNet weights lần đầu).
6. Sửa `SOURCES` ở cell 1: bật nguồn nào có, sửa `root` cho khớp đường dẫn
   `/kaggle/input/...` (Roboflow/local_export) — riêng Hugging Face không cần
   sửa `root`, cell 1b tự điền.
7. Run All.

## Ba thứ notebook cưỡng chế, đừng tắt

**1. Nhãn không map được thì bỏ và đếm, không đoán.** Gộp `solder_ball` vào
`excess` là giấu một loại lỗi model chưa từng thấy sau một nhãn đạt. Nhãn lạ sẽ
hiện ở cell 5 kèm số lượng; thêm nó vào `LABEL_MAPS` hoặc vào `IGNORE` **kèm lý
do**, rồi chạy lại.

**2. Chia tập theo board, không theo crop.** Các crop cùng một board dùng chung
ánh sáng, tiêu cự và thao tác của cùng một người. Chia theo crop đặt các mẫu gần
trùng nhau ở cả hai phía và cho ra điểm số dây chuyền không bao giờ thấy.

**3. Lớp không đủ dữ liệu bị loại khỏi `class_names`.** Cell 6 in ma trận phủ và
loại lớp dưới `min_per_class`. Một head xuất ra lớp nó chưa từng thấy sẽ cho dự
đoán tự tin mà không có gì đằng sau.

Lớp bị loại **không mất trắng**: tầng luật của bước 6.2 vẫn bắt được chúng mà
không cần model — `escape_guard` cho thiếu thiếc, luật cặp chân cho bridge, luật
so hai đầu cho dựng bia.

## Đọc kết quả

Notebook báo **escape rate** và **false call rate**, không báo accuracy. Dây
chuyền 99.5% đạt thì cứ gọi tất cả là đạt đã được 99.5% accuracy trong khi bỏ lọt
toàn bộ lỗi.

Cell 8 quét ngưỡng accept. **Chọn ngưỡng có escape chấp nhận được trước, rồi mới
xét false_call** — bỏ lọt lỗi thì giao hàng lỗi, báo nhầm chỉ tốn 10 giây của
người kiểm. Sửa `decision_thresholds` trong manifest theo lựa chọn đó.

## Sau khi tải về

```powershell
# 1. Kiểm tra artifact nạp được trước khi tin
.\.venv\Scripts\python.exe scripts\verify_solder_model.py `
  models\solder\best.onnx models\solder\model_manifest.json

# 2. Hiệu chỉnh ngưỡng tầng luật theo board của bạn
.\.venv\Scripts\python.exe scripts\calibrate_solder_thresholds.py D:\board_dat `
  --model models\detector\kaggle\best.onnx --output config\solder_thresholds.json
```

`verify_solder_model.py` nạp cặp file qua **đúng runtime app dùng**, nên pass ở
đó nghĩa là app nạp được. Nó cũng bắt hai lỗi hay gặp: export bị tách trọng số ra
file `.data` riêng (app chỉ copy `.onnx` nên sẽ hỏng), và model gọi một land trơn
là "good" với độ tự tin cao (dấu hiệu thứ tự class bị hoán vị).

## Giới hạn lớn nhất: khoảng cách miền

Model train từ dataset công khai học camera, ống kính và ánh sáng của **người
khác**. Nó là điểm khởi đầu để fine-tune trên board của bạn, không phải model
production.

Bước 6.2 hợp nhất model với tầng luật đo chính vì lý do này: khi hai tầng bất
đồng, ROI đi vào hàng đợi kiểm tra thay vì tin model. Nguồn dữ liệu giá trị nhất
vẫn là export từ chính dây chuyền của bạn — xem
`scripts/export_solder_dataset.py`, rồi bật nguồn `local_export` trong `SOURCES`.

Và nhắc lại hai nút thắt vật lý đã nêu ở mục bước 5.5: **cold solder cần đèn vòng
RGB đa góc** mới tách được khỏi mối hàn tốt, và fillet cần ~15–25 µm/px mới đọc
được hình dạng. Không dataset nào bù được hai thứ đó.
