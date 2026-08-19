# Automated Optical Inspection PCB

Ứng dụng local thử nghiệm luồng AOI từ bước 0 đến bước 6.1:

```text
0. Import ảnh
   → 1. Undistort tùy chọn + tiền xử lý
   → 2. Căn chỉnh với Golden Image (tùy chọn)
   → 3. Khoanh vùng PCB
   → 4. Phát hiện linh kiện
   → 5. Crop và xuất dữ liệu linh kiện
   → 5.5. Suy ra ROI mối hàn + hợp nhất CAD nếu có (dữ liệu cho 6.2)
   → 6.1. Phân loại family (accept/review/unknown)
   → 6.2. Kiểm tra mối hàn (luật đo + model, accept/review/reject)
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

## Vì sao bước 1 và bước 2 trông như bị khoá

Cả hai bước đều **không hỏng và không thiếu thư viện**. Chúng bị khoá vì thiếu
*file đầu vào* mà không thứ gì trong repo tạo thay được:

| Chỗ bị khoá | Thiếu gì | Cần camera không? | Mở khoá bằng |
|---|---|---|---|
| Bước 1 · ô **Sửa méo ống kính** | Profile hiệu chỉnh `.json` | **Có** — phải chụp bàn cờ bằng đúng camera/lens/tiêu cự sẽ dùng | [Hiệu chỉnh méo ống kính camera](#hiệu-chỉnh-méo-ống-kính-camera) |
| Bước 2 · nút **Căn chỉnh với reference** | Ảnh Golden Image | **Không** — dùng được ảnh board đạt chuẩn có sẵn | Sidebar → *Golden Image / Reference* |
| Bước 2 · ô **Phương pháp** | — | — | Không mở được: ORB + ECC là phương pháp duy nhất được nối vào core, nên không có gì để chọn |

Không có Golden Image thì bấm **Bỏ qua căn chỉnh**; pipeline vẫn chạy hết bước
3–6.1, chỉ là toạ độ không được đưa về hệ của board chuẩn.

Mọi tuỳ chọn còn lại của bước 1 (resize, khử nhiễu, white balance, CLAHE,
normalize, sharpen) dùng được ngay mà không cần gì thêm.

### Cài đặt: kiểm tra nhanh

Nếu nghi thiếu thư viện, chạy lệnh này. Chỉ cần `requirements.txt` là bước 1 và 2
chạy đủ; `requirements-model.txt` chỉ cần cho bước 4/6.1 khi nạp model.

```powershell
.\.venv\Scripts\python.exe -c "import cv2, numpy, pandas, streamlit; print('core OK', cv2.__version__)"
.\.venv\Scripts\python.exe -m pytest -q
```

Test chạy sạch nghĩa là phần calibration và alignment đã hoạt động; lúc đó thứ
còn thiếu chắc chắn là file đầu vào, không phải cài đặt.

## Hiệu chỉnh méo ống kính camera

Bước 1 hỗ trợ profile camera OpenCV để sửa méo radial/tangential **trước khi
resize**. Đây là lớp xử lý khác với homography ở bước 2: undistort sửa méo lens,
còn homography đưa mặt phẳng PCB về Golden Image.

**Bước 0 — in bàn cờ.** Tải một mẫu chessboard bất kỳ (ví dụ 10×7 ô), in ra và
dán phẳng lên bìa cứng. Bàn cờ cong sẽ làm sai toàn bộ phép hiệu chỉnh. Đo cạnh
một ô bằng thước, đó là `--square-size`.

**Bước 1 — chụp.** Cần ít nhất 10 ảnh dùng được, nên chụp 15–25 ảnh bằng **đúng**
camera, lens, tiêu cự, focus và độ phân giải sẽ dùng khi chạy AOI. Đổi bất kỳ thứ
nào trong số đó là profile hết giá trị. Cho bàn cờ xuất hiện ở giữa khung, ở bốn
góc, sát cạnh ảnh, và nghiêng nhiều góc khác nhau — nghiêng là thứ tách được tiêu
cự khỏi khoảng cách, chụp toàn ảnh chính diện sẽ ra profile kém.

**Bước 2 — chạy script.** `--columns`/`--rows` là số **giao điểm bên trong**,
không phải số ô. Bàn cờ 10×7 ô có 9×6 giao điểm trong:

```powershell
.\.venv\Scripts\python.exe scripts\calibrate_camera.py `
  D:\camera_calibration\cam01 `
  --columns 9 `
  --rows 6 `
  --square-size 20 `
  --camera-id CAM01 `
  --lens-id 12mm-fixed `
  --output camera_profiles\cam01_12mm.json
```

Đơn vị `--square-size` có thể là mm hoặc đơn vị khác miễn nhất quán.

Script in ra số ảnh nhận/loại và reprojection error. **RMS dưới ~0.5 px là tốt,
trên ~1.0 px thì nên chụp lại** — thường là do bàn cờ cong, ảnh mờ, hoặc thiếu
góc nghiêng. Ảnh nào không tìm thấy bàn cờ sẽ được liệt kê ở mục Rejected; nếu
loại quá nhiều thì kiểm tra lại `--columns`/`--rows` trước đã, đếm nhầm giao điểm
là lỗi hay gặp nhất.

**Bước 3 — dùng trong app:**

1. Mở sidebar **Camera calibration** và tải file JSON.
2. Ở bước 1 bật **Sửa méo ống kính** (ô này chỉ bật lên sau khi có profile).
3. Để `Giữ vùng biên sau undistort = 0` nếu muốn ít viền đen; tăng dần về `1` nếu
   cần giữ trường nhìn.
4. Nếu dùng Golden Image, ảnh chuẩn phải là ảnh raw từ cùng camera/lens/recipe;
   pipeline sẽ undistort cả ảnh hiện tại và Golden Image trước khi homography.

Profile ghi độ phân giải calibration và reprojection error. Ảnh cùng tỉ lệ khung
hình có thể dùng độ phân giải khác vì ma trận nội tại được scale; ảnh crop hoặc
sai tỉ lệ sẽ bị từ chối để tránh hiệu chỉnh sai mà không báo.

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

## Adaptive tiling cho ảnh PCB lớn

Bước 4 mặc định dùng chế độ `auto`: `tile_size=1280` là giới hạn trên, còn cửa sổ
detail được chọn thích ứng từ 640–1280 px. Vì vậy ảnh 1000×750 sẽ thực sự chạy bốn
tile khoảng 640 px thay vì bị coi là nhỏ hơn một tile 1280. Các tile overlap 20%
và dùng confidence detail mặc định 0.20. Pipeline có thể chạy thêm
một lượt toàn board để giữ ngữ cảnh cho linh kiện lớn, sau đó đổi mọi box về tọa
độ ảnh analysis và dùng class-aware global NMS để loại detection trùng. Box nằm ở
đường nối tile nhận một penalty ưu tiên nhỏ, nhờ đó box nhìn thấy linh kiện đầy đủ
hơn ở tile bên cạnh được giữ lại. Ảnh nhỏ vẫn chỉ chạy đúng một lượt như trước.

UI bước 4 cho phép chọn `Tự động / Luôn bật / Tắt`, tile size tối đa, overlap,
confidence detail, full-board pass, merge IoU và lưới debug. Metadata/CSV ghi thêm `frame_id`, `inference_pass`,
`tile_id` và trạng thái chạm biên. CV candidate demo không dùng tiling vì các
ngưỡng area ratio của nó gắn với toàn ROI và không phải model nhận dạng thật.

Mỗi tile còn có một vùng ownership nằm giữa phần overlap. Detection có tâm trong
ownership được ưu tiên hơn box sát đường cắt, nên linh kiện nhỏ không bị chọn theo
một crop cụt ở mép. Sau same-class NMS, hai box khác class nhưng IoU trên 0.70 được
coi là hai giả thuyết cho cùng vật thể và chỉ giữ box có ưu tiên cao hơn. Lưới debug
vẽ cả cửa sổ inference và ownership để kiểm tra trực quan.

Client chặn ảnh toàn PCB dưới **1280×960 px (1,23 MP)** ngay tại bước import và
kiểm tra lại trước preprocessing. Thông báo yêu cầu chụp/gửi ảnh khác; upscale ảnh
cũ không được coi là đạt chất lượng vì không tạo thêm chi tiết quang học. Ngưỡng
này là gate ban đầu, cần tăng theo kích thước linh kiện nhỏ nhất của camera thật.

Detector class trên overlay chỉ là gợi ý. Nếu một capacitor/LED có box nhưng bị
gán thành resistor, tiling/NMS không thể sửa class một cách đáng tin cậy; bước 6.1
phải phân loại lại crop hoặc detector cần fine-tune bằng dữ liệu capacitor/LED từ
đúng camera và recipe ánh sáng của hệ thống.

Bước tiền xử lý của app giữ mặc định cạnh dài tối đa 4096 px. Nếu ảnh bị thu nhỏ
quá mạnh, bước 4 cảnh báo vì tiling không thể khôi phục pixel đã mất. Crop cho
classifier được lấy từ ảnh analysis độ phân giải cao bằng box đã gộp, không lấy
từ tensor tile đã bị detector letterbox/resize.

Với camera nhiều khung sau này, mỗi frame dùng chính contract `frame_id` và tọa độ
frame-local này: undistort, detect từng frame, chiếu box qua homography sang hệ PCB
chung rồi global merge. Panorama chỉ cần cho hiển thị, không phải đầu vào detector.

## Bước 5.5 — ROI mối hàn cho bước 6.2

Detector ở bước 4 được train trên dataset gán nhãn **thân linh kiện**. Fillet
mối hàn nằm *ngoài* silhouette đó, nên không có box nào của detector — và không
có lần train lại nào trên cùng bộ nhãn — chứa được mối hàn. Đi bắt detector tìm
thẳng mối hàn cũng không phải lựa chọn: model hiện tại đạt recall **0.0** cho
class `pads` trên cả val lẫn test (186 instance train), và 0.14/0.21 cho `pins`.

Vì vậy ROI mối hàn được **suy ra** chứ không phải detect. Từ box cộng với
topology chân của class, vị trí mọi mối hàn là bài toán hình học — đúng cách các
hệ AOI cửa sổ vẫn đặt vùng kiểm tra:

| Topology | Class | ROI sinh ra |
|---|---|---|
| `two_terminal` | resistor, capacitor, diode, led, inductor, fuse | 2 ROI ở hai đầu trục dài |
| `multi_pin` | ic, connector, transistor, relay, switch… và mọi class chưa biết | 1 ROI dải cho mỗi cạnh có chân |
| `pad_only` | pads | chính box đó, nới nhẹ |

Mỗi linh kiện còn có thêm một ảnh `body` gồm thân **và** toàn bộ chân — đây là
view "nhìn thấy cả mối hàn" mà box gốc không cho.

Với board `multi_pin`, cạnh nào không có kim loại chân sẽ bị loại bằng năng
lượng Laplacian tương đối giữa 4 cạnh của cùng linh kiện, nên SOIC chỉ còn hai
dải trái/phải thay vì bốn. Nếu không truyền ảnh vào, bộ lọc bị bỏ qua và cả 4
cạnh được giữ: loại bỏ một cạnh khi không có bằng chứng sẽ làm mất mối hàn mà
không báo.

Tùy chọn `split_pins` cắt mỗi dải thành một ROI cho từng chân, dùng profile 1-D
đã khử nền. Mặc định **tắt**, vì lỗi bridge nằm giữa hai chân nên ROI dải thường
là đơn vị kiểm tra tốt hơn. Khi bật, dải nào không đọc được hàng chân đáng tin
(số chân hoặc pitch bất thường) sẽ giữ nguyên là một dải thay vì bịa ra chân.

### Sinh dataset cho bước 6.2

```powershell
.\.venv\Scripts\python.exe scripts\export_solder_dataset.py D:\anh_board `
  --output D:\datasets\solder_v1 `
  --model models\detector\best.onnx `
  --overlays
```

Kết quả là `crops/` phẳng cộng `solder_dataset.csv` có cột `defect_class` bỏ
trống. Hình học đã giải quyết xong, nên gán nhãn chỉ còn là phán quyết theo từng
dòng chứ không phải đi khoanh box lại. Script cảnh báo nếu ROI quá nhỏ để chấm
được fillet. Trong app, bước 5 có tab **ROI mối hàn (6.2)** để xem overlay,
gallery và tải cùng bảng nhãn đó.

### Hai điều kiện vật lý quyết định bước 6.2

Trước khi gán nhãn cả lô, hãy xem `overlays/` và một mẫu `crops/`:

- **Ánh sáng.** AOI soi mối hàn thật dùng đèn vòng RGB đa góc để độ dốc fillet
  được mã hóa thành màu. Với đèn trắng phẳng hoặc coaxial, mối hàn tốt và cold
  joint gần như giống hệt nhau, và không model nào sửa được điều đó.
- **Độ phân giải.** Fillet cần cỡ 10–15 px ngang mới đọc được hình dạng. Với
  linh kiện 0402 tức khoảng 15–25 µm/px, tức board 100 mm cần ảnh cỡ 5000–6000
  px chiều ngang. Gate import hiện tại là 1280×960, thấp hơn nhiều bậc so với
  yêu cầu đó.

Nếu có Gerber/CAD hoặc file pick-and-place thì phần dưới đây hợp nhất toạ độ
land thật vào chính hình học này. Không có CAD thì mọi thứ trên vẫn chạy nguyên
vẹn.

### Hợp nhất sơ đồ CAD (đã dựng sẵn, chưa cần file)

Toàn bộ đường CAD đã có trong code. Khi có sơ đồ thì chỉ cần nạp file vào, không
phải sửa gì. **Chưa có file thì pipeline chạy đúng như phần trên**, không thêm
bước nào.

Quan trọng: đây là **kết hợp**, không phải chọn một trong hai. CAD biết land nằm
đâu và linh kiện *phải* là gì, nhưng không biết board đang nằm đâu dưới camera và
không biết linh kiện nào đặt sai hay thiếu. Detector thì ngược lại. Nên:

| Tình huống | ROI sinh ra | `source` |
|---|---|---|
| Hai bên cùng chỉ vào một land | ROI hợp nhất | `cad+derived` |
| CAD có land, detector không có ROI ở đó | ROI theo land CAD | `cad` |
| Detector có ROI, CAD không liệt kê land (thermal pad, shield…) | Giữ ROI suy ra | `derived` |
| CAD chỉ có vị trí đặt (pick-and-place) | Hình học suy ra, neo trên tâm/góc CAD | `cad+derived` |
| CAD có linh kiện, ảnh không thấy | ROI theo land + finding `missing_component` | `cad` |
| Ảnh có linh kiện, CAD không có | Giữ ROI suy ra + finding `unexpected_component` | `derived` |

Hai điểm làm nên phần "kết hợp":

- **Hiệu chỉnh cục bộ.** CAD cho footprint, detector cho biết linh kiện *này*
  thực tế nằm đâu. Mỗi linh kiện được dịch theo sai lệch giữa hai vị trí, nên một
  phép căn chỉ gần đúng trên toàn board vẫn cho ROI chính xác tại từng linh kiện.
- **Topology chân lấy từ số pad thật.** Một linh kiện 4 chân bị detector đọc nhầm
  thành `resistor` vẫn ra 4 ROI, thay vì 2 theo suy đoán từ class.

Nhờ đối chiếu, có thêm bốn loại lỗi phát hiện được **không cần model nào** — ghi
vào `cad_findings.csv`: `missing_component` (defect), `shifted_component`
(review, ngưỡng mặc định 0.5 mm), `unexpected_component` và `class_mismatch`.

Định dạng nhận được: bảng pad CSV, file pick-and-place/centroid, IPC-D-356A, và
`cad_json` đã lưu. Nhận dạng tự động. Thêm định dạng mới chỉ là viết một hàm rồi
đăng ký vào `CAD_LOADERS`.

```powershell
.\.venv\Scripts\python.exe scripts\export_solder_dataset.py D:\anh_board `
  --output D:\datasets\solder_v1 `
  --model models\detector\kaggle\best.onnx `
  --cad D:\cad\board_pads.csv `
  --save-registration D:\cad\reg_sku01.json
```

Trong app: sidebar có mục **Sơ đồ CAD (tuỳ chọn)**, bước 5 có tab **Đối chiếu CAD**.

**Phép căn sai trông y hệt phép căn đúng nếu chỉ nhìn residual**, nên hệ thống báo
ra thay vì im lặng áp dụng: phép căn kém chất lượng bị **từ chối** và quay về ROI
suy ra; phép căn mơ hồ (layout đối xứng, hoặc detector không cho class) bị đánh
dấu `ambiguous` kèm cảnh báo. Chốt chắc chắn bằng fiducial hoặc file
`registration.json` lưu một lần cho mỗi SKU/đồ gá.

Chi tiết định dạng, cách căn và cách xử lý khi phép căn không đáng tin:
[docs/cad_formats.md](docs/cad_formats.md), template:
[docs/cad_pads_template.csv](docs/cad_pads_template.csv).

### Crop bước 5 vẫn giữ nguyên hợp đồng với 6.1

Bước 5.5 có ROI riêng đúng để không phải nới crop của bước 5. Classifier 6.1
được train với `pad = 0.15 × max(w, h)`, cắt theo biên ảnh, **không** ép vuông;
`CropConfig` và config UI nay khớp đúng công thức đó. `CropConfig.solder_aware_padding`
cho phép chuyển sang padding theo trục/theo class, nhưng mặc định tắt vì nó làm
lệch phân bố đầu vào của classifier.

## Trạng thái bước 6.1

Khung phân loại đã có trong pipeline và Streamlit. Khi chưa có classifier, bước
6.1 hiển thị trạng thái chờ và **không** lấy nhãn detector làm kết quả giả. Để
train baseline:

1. Đọc [hướng dẫn phân loại Kaggle](training/kaggle/README_classification.md).
2. Import [pcb_component_classification_kaggle.ipynb](training/kaggle/pcb_component_classification_kaggle.ipynb).
3. Add Input `aryanstein/pcb-component-detection-consolidated-dataset`, bật GPU
   và chạy `Run All`.
4. Tải `pcb_component_classifier_artifacts.zip`, rồi nạp đồng thời `best.onnx`
   và `model_manifest.json` trong sidebar **Model phân loại 6.1**.

Runtime đọc class order, preprocessing, calibration và ngưỡng quyết định từ
manifest, đồng thời kiểm tra SHA-256 của ONNX. Kết quả `review`/`unknown` được đưa
vào hàng đợi kiểm tra; ZIP export có thêm `classifications.csv`. Backbone mặc
định là EfficientNet-B0/224 để cân bằng accuracy với triển khai ONNX Runtime trên
Raspberry Pi ARM64; YOLO chỉ thuộc bước phát hiện 4, không dùng cho phân loại 6.1.

Tài liệu đã chuẩn bị cho các bước tiếp theo:

- [Khảo sát dataset linh kiện PCB](Docs/pcb_aoi_component_datasets.md).
- [Kế hoạch pre-train cho bước 6.1](Docs/ke_hoach_pretrain_6_1_classification.md).

## Bước 6.2 — Kiểm tra mối hàn

Chấm từng ROI do bước 5.5 sinh ra. **Chạy được ngay khi chưa có model.**

### Ba tầng, dùng cùng nhau

| Tầng | Là gì | Cần train? | Vai trò |
|---|---|---|---|
| A · `grading/features.py` + `rules.py` | Đo đặc trưng vật lý rồi phán quyết theo ngưỡng | Không | Chạy từ ngày đầu, giải thích được, sinh nhãn mồi |
| B · `grading/classifier.py` | CNN trên crop mối hàn (ONNX) | Có | Bắt cái ngưỡng không bắt được: mối hàn nguội, hình dạng tinh vi |
| C · `grading/inspector.py` | Hợp nhất A và B | — | Quyết định cuối, kèm chốt chặn |

Vì sao không nhảy thẳng vào CNN: tầng A chạy khi chưa có nhãn nào, biến việc gán
nhãn thành *xác nhận/sửa* thay vì gán từ đầu, và ở lại làm chốt chặn. Một call
mà người vận hành không truy được về con số sẽ bị bỏ qua ngay trong tuần đầu.

### Quy tắc hợp nhất

Bất đối xứng có chủ ý: **bỏ lọt lỗi thì giao hàng lỗi, báo nhầm chỉ tốn 10 giây
của người kiểm.**

| Tình huống | Kết quả |
|---|---|
| Model và luật đồng ý, model đủ tự tin | `accept` · source `model+rules` |
| Model và luật đồng ý nhưng model thiếu tự tin | `review` |
| Hai bên **bất đồng** | `review` · source `conflict` — không chọn bên nào thắng |
| Model nói đạt nhưng lượng thiếc dưới sàn vật lý | `review` · source `escape_guard` — không confidence nào vượt qua được |
| Chưa có model | Verdict của luật · source `rules` |

Chốt chặn chạy **sau cùng** nên không bước nào ở trên hoàn tác được nó. Tắt bằng
`SolderGradingConfig.escape_guard_enabled=False` nếu thật sự muốn model toàn quyền.

### Phân loại lỗi

Bám đúng hai loại ROI bước 5.5 đã sinh:

- **ROI mối hàn** (`kind="joint"`): `good`, `insufficient`, `excess`, `bridge`,
  `cold`, `missing_solder`
- **ROI linh kiện** (`kind="body"`): `ok`, `missing`, `tombstone`, `shifted`,
  `wrong_polarity`

Hai lỗi chỉ nhìn thấy khi so ROI với nhau, nên được xử lý riêng: **bridge** cần
hai chân cạnh nhau cùng phủ kín biên chung; **dựng bia** cần so hai đầu của cùng
một linh kiện hai chân.

### Bước 1 — hiệu chỉnh ngưỡng (làm trước, kể cả khi định train)

Ngưỡng mặc định trong `SolderGradingConfig` **là số khởi đầu, không phải số
đúng**. Chúng phụ thuộc ống kính, ánh sáng, hình dạng land và độ rộng ROI của
bước 5.5. Chạy mặc định trên một dây chuyền mới thường gắn cờ gần hết board.

Đo thay vì đoán — chỉ vào các board bạn **đã chấp nhận**:

```powershell
.\.venv\Scripts\python.exe scripts\calibrate_solder_thresholds.py D:\board_dat `
  --model models\detector\kaggle\best.onnx `
  --output config\solder_thresholds.json `
  --dump-features config\solder_features.csv
```

Script in phân bố từng đặc trưng và đề xuất ngưỡng theo phân vị. Dán khối
`solder_grading` vào config. Xem lại trước khi dùng: board đưa vào mà có lỗi thì
ngưỡng sẽ nới rộng đúng chỗ đáng lẽ phải bắt.

### Bước 2 — train model (khi đã có nhãn)

```powershell
# 1. sinh dataset ROI (đã có từ bước 5.5)
.\.venv\Scripts\python.exe scripts\export_solder_dataset.py D:\anh_board `
  --output D:\datasets\solder_v1 --model models\detector\kaggle\best.onnx

# 2. điền cột defect_class trong solder_dataset.csv

# 3. train và export
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
.\.venv\Scripts\python.exe training\train_solder_classifier.py D:\datasets\solder_v1 `
  --output models\solder --epochs 30
```

Ba điểm trong script không nên đổi nếu không có lý do:

- **Chia tập theo board, không theo ROI.** Hai mối hàn cùng board dùng chung ánh
  sáng, tiêu cự và thao tác của cùng một người. Chia theo ROI đặt các mẫu gần
  trùng nhau ở cả hai phía và cho ra điểm số dây chuyền sẽ không bao giờ thấy.
- **Chỉ số báo là escape rate và false call rate, không phải accuracy.** Dây
  chuyền 99.5% đạt thì cứ gọi tất cả là đạt đã được 99.5%. Cái cần biết là bao
  nhiêu lỗi bị gọi thành đạt, và bao nhiêu mối hàn tốt bị gọi thành lỗi.
- **Class weight bật sẵn.** Không có nó, loss bị lớp `good` chi phối và model học
  cách không bao giờ đánh trượt thứ gì.

### Bước 3 — thả model vào là chạy

Cần đúng hai file, do script train xuất ra:

```text
best.onnx            model logit thô
model_manifest.json  thứ tự class, tiền xử lý, calibration, ngưỡng
```

Nạp cả hai ở sidebar **Model kiểm tra mối hàn 6.2**, hoặc trỏ bằng config:

```python
config.solder_grading.model_path = "models/solder/best.onnx"
config.solder_grading.manifest_path = "models/solder/model_manifest.json"
```

Không cần đổi gì khác. Thiếu một trong hai thì bước 6.2 vẫn chạy bằng luật và
báo rõ. Runtime **từ chối** manifest sai schema thay vì đoán — đoán sai thứ tự
class là biến mọi lỗi thành "đạt". Schema đầy đủ:
[docs/solder_model_manifest_template.json](docs/solder_model_manifest_template.json).

### Kết quả xem ở đâu

Trong app: bước 5 → tab **Chấm mối hàn (6.2)**, có overlay theo quyết định, bảng
kết quả và chi tiết từng ROI kèm lý do. Trong gói ZIP export:
`solder_joints/solder_verdicts.csv` và `images/06_solder_verdicts.png`. Mỗi dòng
CSV mang theo số đo đã dẫn tới quyết định, để tranh luận được là ngưỡng sai hay
mối hàn sai.

### Giới hạn phải biết trước

Tầng A tìm thiếc bằng "sáng và ít bão hoà", tức giả định fillet phản xạ nhiều
hơn nền quanh nó. **Dưới đèn trắng phẳng, mối hàn nguội và mối hàn tốt đo ra gần
như nhau** — đó là giới hạn quang học, không ngưỡng nào cứu được. Đây cũng chính
là hai nút thắt đã nêu ở mục bước 5.5: ánh sáng và độ phân giải.

## Cấu trúc dự án

```text
aoi_pipeline/        Thư viện pipeline, chia theo bước
  core/                Nền tảng: models, exceptions, image_io (không phụ thuộc ai)
  imaging/             Bước 0–1: calibration, preprocessing
  board/               Bước 2–3: alignment, localization
  detection/           Bước 4: detectors, tiling
  inspection/          Bước 5–5.5: cropping, solder, cad, fusion
  export/              Đóng gói: exporters, overlays
  grading/             Bước 6.2: features, rules, classifier, inspector
  classification.py    Bước 6.1
  config.py            Toàn bộ knob của mọi bước, một chỗ
  pipeline.py          Facade 0 → 6.1
app/                 Streamlit UI và bridge
tests/               Unit tests, soi gương cấu trúc aoi_pipeline/
training/            Notebook train bước 4 / 6.1, script train bước 6.2
models/              Nơi đặt model local (weights không commit Git)
docs/                Khảo sát dataset, kế hoạch 6.1, hướng dẫn nạp CAD
scripts/             Setup/chạy app, calibrate camera, export dataset 6.2
legacy/              Prototype cũ (KCS_Inspec_PCBA_V2.exe), không commit Git
```

Phụ thuộc trong `aoi_pipeline/` phân tầng nghiêm ngặt và không có vòng: `core/`
không phụ thuộc ai, mỗi package theo bước chỉ phụ thuộc `core/` cộng `config.py`,
và chỉ `pipeline.py` biết tới tất cả. Vì vậy đọc một bước không cần đọc bước khác.

API công khai vẫn nguyên: `from aoi_pipeline import ...` không đổi gì. Chỉ khi
import thẳng submodule mới cần dùng đường dẫn mới, ví dụ
`from aoi_pipeline.inspection.cad import load_cad`.

## Giới hạn hiện tại

- Khoanh PCB ở bước 3 đang dùng contour fallback, chưa có PCB detector riêng.
- ROI bước 5.5 suy ra từ box nên chỉ đúng khi class của detector đúng; class sai
  kéo theo topology chân sai. Nạp CAD sẽ lấy topology từ số pad thật.
- Tự căn CAD cần detector cho được class thật; với CV demo (mọi thứ đều là
  `component_candidate`) phép căn chỉ dựa trên hình học và bị đánh dấu mơ hồ trên
  layout đối xứng. Dùng fiducial hoặc registration đã lưu cho sản xuất.
- Chưa có loader cho KiCad `.kicad_pcb`, ODB++ hay Gerber; quy về bảng pad CSV
  hoặc thêm loader vào `CAD_LOADERS`.
- Ước lượng góc xoay linh kiện (`estimate_orientation`) mặc định tắt: góc sai làm
  lệch mọi ROI suy ra từ nó, đắt hơn là để ROI axis-aligned hơi rộng.
- Ngưỡng mặc định của bước 6.2 là số khởi đầu, không phải số đúng cho dây chuyền
  của bạn; phải chạy `calibrate_solder_thresholds.py` trước khi tin kết quả.
- Tầng đo của bước 6.2 phụ thuộc mạnh vào ánh sáng: đèn trắng phẳng không tách
  được mối hàn nguội khỏi mối hàn tốt.
- CV proposal ở bước 4 không thay thế model đã train.
- Baseline dùng `max_det=2000` cho board dày linh kiện; cần tune lại theo SKU/tốc độ.
- Adaptive tiling tăng recall cho linh kiện nhỏ nhưng tăng thời gian gần tỷ lệ với
  số tile; cần benchmark tile size/overlap trên máy chạy thật và Raspberry Pi.
- Căn chỉnh chính xác cần một Golden Image/reference cùng board side.
- Model cuối phải được đánh giá trên camera, lens, ánh sáng và PCB của dây chuyền.
- Mỗi ảnh import được giới hạn 64 MB/50 MP; upload Streamlit tối đa 256 MB/file.
- Ultralytics công bố lựa chọn AGPL-3.0/Enterprise; cần duyệt license trước khi
  đưa framework/model vào sản phẩm thương mại hoặc phần mềm đóng nguồn.
