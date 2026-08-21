# Automated Optical Inspection PCB

Ứng dụng local có hai workspace: Golden Inspection dùng recipe cố định cho
Position/Appearance, và pipeline thử nghiệm từ bước 0 đến bước 6.2:

```text
0. Import ảnh
   → 1. Undistort tùy chọn + tiền xử lý
   → 2. Căn chỉnh với Golden Image (tùy chọn)
   → 3. Khoanh vùng PCB
   → 4. Phát hiện linh kiện
   → 5. Crop và xuất dữ liệu linh kiện
   → 5.5. Suy ra ROI chân/mối hàn (hình học + CAD + detection chân)
   → 6.1. Phân loại family (accept/review/unknown)
   → 6.2. Chấm lỗi mối hàn (luật đo + model + hợp nhất)
```

Bước 6.2 là **mục riêng** trong điều hướng, không phải tab con của bước 4: nó
có ROI riêng, hợp đồng model riêng và từ vựng verdict riêng, và nó mới là bước
quyết định board có xuất xưởng được hay không.

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

## Golden Inspection

Workspace mặc định của UI là **Golden Inspection**:

1. Nạp Golden Image ở sidebar. Ảnh JPEG đầu vào demo được chấp nhận, nhưng
   recipe luôn lưu lại Golden, anchor, template và mask dưới dạng PNG lossless.
2. Nạp ảnh board cần kiểm tra. Detector nạp ở sidebar; bản mới nhất là
   `models/detector/kaggle/ver2/best.onnx`. Chưa nạp thì UI chạy CV demo và
   nói rõ đó không phải nhận dạng đáng tin cậy.
3. Trong tab **Build Recipe**, nhập `Board ID`, chọn `top`/`bottom`, đặt
   calibration/tolerance rồi tạo fixed slot ROI và strict-alignment anchors.
4. Trong tab **Inspect Board**, chạy core inspector. UI hiển thị Alignment,
   Position và Appearance ở ba bảng riêng, kèm overlay và JSON portable.
5. Recipe có thể tải thành ZIP gồm `recipe.json` và toàn bộ asset tương đối;
   kết quả inspection tải thành `inspection_result.json`.

Anchor lưới tự động chỉ phục vụ enrollment demo và vẫn phải qua residual,
inlier, scale, rotation và canvas-overlap gate. Không đánh dấu calibration là
verified nếu chưa đo thật. Recipe demo vẫn inspect được khi tắt production gate,
nhưng không được coi là acceptance production.

Recipe hiện dùng schema `aoi-inspection-recipe/1.1`: Golden, mọi anchor,
template và mask lossless đều có SHA-256 riêng, và các digest đó được gắn vào
recipe hash. Sửa một asset cùng kích thước cũng làm validation thất bại. Recipe
schema 1.0 không có bảo đảm này nên phải build lại thay vì được nâng cấp ngầm.
Khi camera calibration được bật, Golden lẫn ảnh test đều chỉ đi qua undistort
(không resize/letterbox); profile hoặc alpha khác recipe sẽ dừng inspection.

File detector `.pt` chỉ được chạy sau khi người dùng xác nhận tin cậy trong
sidebar vì định dạng này có thể thực thi pickle khi nạp. Recipe production chỉ
được xét khi detector runtime khớp identifier/hash đã lưu, metrology đã verified
và anchor có provenance fiducial/hole/stable-patch được phê duyệt; anchor lưới
tự động không thể tự nâng recipe lên production.

## Hiệu chỉnh méo ống kính camera

Bước 1 hỗ trợ profile camera OpenCV để sửa méo radial/tangential **trước khi
resize**. Đây là lớp xử lý khác với homography ở bước 2: undistort sửa méo lens,
còn homography đưa mặt phẳng PCB về Golden Image.

Chuẩn bị ít nhất 10 ảnh bàn cờ calibration ở đúng camera, lens, tiêu cự, focus và
độ phân giải sẽ dùng khi chạy AOI. Nên chụp 15–25 ảnh với bàn cờ xuất hiện ở giữa,
các góc, cạnh ảnh và nhiều góc nghiêng. `--columns`/`--rows` là số **giao điểm bên
trong**, không phải số ô. Ví dụ bàn cờ 10×7 ô có 9×6 giao điểm trong:

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

Đơn vị `--square-size` có thể là mm hoặc đơn vị khác miễn nhất quán. Sau khi tạo
profile:

1. Mở sidebar **Camera calibration** và tải file JSON.
2. Ở bước 1 bật **Sửa méo ống kính**.
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
| A · `aoi_pipeline/grading/features.py` + `rules.py` | Đo đặc trưng vật lý rồi phán quyết theo ngưỡng | Không | Chạy từ ngày đầu, giải thích được, sinh nhãn mồi |
| B · `aoi_pipeline/grading/classifier.py` | CNN trên crop mối hàn (ONNX) | Có | Bắt cái ngưỡng không bắt được: mối hàn nguội, hình dạng tinh vi |
| C · `aoi_pipeline/grading/inspector.py` | Hợp nhất A và B | — | Quyết định cuối, kèm chốt chặn |

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

### Dataset: không có nguồn công khai nào đủ, phải ghép

Khảo sát tháng 8/2026. **Không dataset công khai nào phủ đủ taxonomy 6.2.** Nguồn
tốt nhất mỗi nguồn phủ một phần, nên chúng được ghép lại:

| Nguồn | Phủ | Ghi chú |
|---|---|---|
| **SolDef_AI** (Kaggle, [MDPI JMMP 2024](https://doi.org/10.3390/jmmp8030117)) | good, insufficient, excess, **shift_component** | 1150 ảnh, 3 góc nhìn. Nguồn peer-reviewed duy nhất có nhãn lệch vị trí linh kiện |
| HF `ouvic215` / `AndyLiu0104` | bridge, excess, missing_solder | **Không license, không nguồn gốc**; nghi dữ liệu sinh |
| Roboflow soldering-defects | **cold**, bridge | Nguồn công khai duy nhất có cold solder, nhưng vài trăm ảnh |
| Export từ board của bạn | tất cả | Nguồn **duy nhất** khớp camera/ánh sáng của bạn |

**Đừng nối nhầm:** DeepPCB, HRIPCB/PKU-Market-PCB, DsPCBSD+, `akhatova/pcb-defects`
là lỗi **board trần** — board chưa gắn linh kiện, không có mối hàn nào. AXI_PCB là
X-quang. PCBSPDefect chưa phát hành. `aoi_pipeline/grading/datasets.py` liệt kê
chúng trong `BARE_BOARD_DATASETS` để không ai nối vào nhầm.

Module ghép ([datasets.py](aoi_pipeline/grading/datasets.py)) cưỡng chế ba điều:
tự **dò** layout thư mục thay vì đoán (layout lạ thì dừng và báo), nhãn không map
được thì **bỏ và đếm** chứ không gộp vào lớp gần nhất, và mỗi record mang `group`
là board gốc để chia tập giữ nguyên board một phía.

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

**Trên Kaggle** (ghép dataset công khai, dùng khi chưa gán nhãn đủ):
[training/kaggle/README_solder.md](training/kaggle/README_solder.md) và notebook
[pcb_solder_defect_kaggle.ipynb](training/kaggle/pcb_solder_defect_kaggle.ipynb).
Notebook in ma trận phủ taxonomy trước khi train và **loại lớp không đủ dữ liệu
khỏi `class_names`** thay vì train một head xuất ra lớp nó chưa từng thấy.

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

Kiểm tra trước khi tin:

```powershell
.\.venv\Scripts\python.exe scripts\verify_solder_model.py `
  models\solder\best.onnx models\solder\model_manifest.json
```

Lệnh này nạp cặp file qua **đúng runtime app dùng**, nên pass ở đó nghĩa là app
nạp được. Nó bắt cả trường hợp export bị tách trọng số ra file `.data` riêng —
lúc đó `best.onnx` một mình sẽ hỏng dù trên máy vừa train vẫn chạy.

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

## Train lại detector và classifier (v2)

Hai notebook v2 ở [training/kaggle/README_v2_models.md](training/kaggle/README_v2_models.md).

### Detector: vấn đề là dữ liệu, không phải kiến trúc

Model hiện tại có `pads` recall **0.000** và `pins` recall **0.145**. Nguyên nhân
đo được: `pads` có **186** instance train so với **7775** của capacitor — ít hơn
42 lần — và mỗi pad chỉ vài chục pixel.

**Đổi sang RT-DETR / D-FINE / YOLOv12 không sửa được 186 instance.** YOLO26 vốn đã
có STAL (small-target-aware label assignment) ép tối thiểu 4 anchor cho object
nhỏ hơn 8 px — đúng cơ chế cho bài toán này. Khuyến nghị: **giữ YOLO26**, sửa dữ
liệu và công thức.

[pcb_detector_v2_kaggle.ipynb](training/kaggle/pcb_detector_v2_kaggle.ipynb) làm:
imgsz 1280→1536, oversample ảnh chứa pads/pins (có trần 35% và tự hạ hệ số),
`copy_paste` 0.30, 150 epoch với `close_mosaic` muộn, và một **cổng verdict** so
recall với baseline rồi nói thẳng nếu vẫn kẹt.

#### Kết quả v2 (đã chạy đủ 150 epoch): vẫn kẹt, và biết vì sao

| | baseline | v2 | |
|---|---:|---:|---|
| `pads` recall | 0.000 | **0.072** | precision 0.712 |
| `pins` recall | 0.145 | **0.333** | precision 0.332 |
| mAP50 tổng | — | 0.457 | mAP50-95 0.166 |

`pads` precision **0.712** với recall **0.072** là chữ ký của việc **học thuộc**,
không phải học lớp: model nhận ra đúng vài pad nó đã thấy, và bỏ lỡ 93% số còn
lại. Con số giải thích điều đó nằm ở dữ liệu — **chỉ 30 trong 670 ảnh train có
chứa pads/pins**. Nhân bản 30 ảnh đó lên 6 lần (22% train list) không tạo thêm
sự đa dạng nào, chỉ lặp lại đúng 30 tấm.

Kết luận đo được, không phải phỏng đoán: **thêm epoch hoặc đổi kiến trúc không
sửa được 30 ảnh.** Chỉ ảnh mới mới sửa được — xem mục bootstrap ngay dưới.

#### Ba chế độ của notebook detector

`CONFIG` trong [pcb_detector_v2_kaggle.ipynb](training/kaggle/pcb_detector_v2_kaggle.ipynb),
xét theo thứ tự này:

| CONFIG | Làm gì | Dùng khi |
|---|---|---|
| `export_from` | **Bỏ qua train**, nạp thẳng file `.pt` | Đã có checkpoint tốt, chỉ cần `best.onnx` + manifest |
| `resume_from` | Train tiếp đúng epoch còn dở | Bị đứt giữa chừng, muốn chạy nốt |
| (không đặt) | Train mới từ đầu | Lần đầu |

**Còn giữ `best.pt` từ session đã chết thì dùng `export_from`, đừng resume.**
`best.pt` là bản fitness đỉnh đã được chọn sẵn; resume chỉ chạy nốt các epoch
còn lại và không có gì đảm bảo chúng vượt được đỉnh đó — đúng lý do sau resume
thường không sinh ra `best.pt` mới. Export-only mất vài phút thay vì hơn một
giờ, và cell val vẫn đo metric thật trên chính trọng số được export.

Cả ba chế độ **vẫn cần Add Input dataset gốc**, vì cell val chấm điểm trên tập
val của nó để sinh metric cho manifest.

Hai lỗi đã sửa ở đường này, cả hai đều chỉ lộ ra sau nhiều giờ train:
- Cell export chết `FileNotFoundError: best.pt` sau resume — nay lùi về
  `last.pt` kèm giải thích, và ghi `exported_from`/`trained_in_this_run` vào
  manifest để sau còn truy được.
- `hasattr(model, "trainer")` **luôn** trả True vì Ultralytics đặt sẵn
  `self.trainer = None` trong `__init__` — kiểm chứng trực tiếp: dòng cũ ném
  `AttributeError: 'NoneType' object has no attribute 'save_dir'` ngay khi có
  đường không train. Nay kiểm tra giá trị thay vì sự tồn tại của thuộc tính.

#### Kích thước ảnh do chính file ONNX quyết định

Export `dynamic=False` khoá cứng input, và các model đang có khoá ở ba mức khác
nhau: ver1 **1280**, ver2 **1536**, huggingface **640**. App từng hardcode 1280
nên chỉ ver1 chạy được — nạp ver2 vào là nổ
`INVALID_ARGUMENT ... Got: 1280 Expected: 1536`. Nay `UltralyticsDetector` tự
đọc shape từ file (`onnx.load`, ~0.09s) và dùng đúng kích thước đó; `.pt` và
ONNX `dynamic=True` vẫn theo config vì chúng thật sự resize được.

YOLO26 còn export ra dạng end-to-end `(1, 300, 6)` — NMS nằm **trong** graph —
bất kể `nms=False`. App vẫn chạy đúng vì `non_max_suppression` của Ultralytics
nhận dạng theo shape (`if prediction.shape[-1] == 6 or end2end`), và đo trên
artifact thật thì conf runtime vẫn tác dụng bình thường. Ràng buộc thật duy
nhất là **trần 300 detection** mỗi lần suy luận, nên với board dày đặc hãy dựa
vào chia tile thay vì một lượt toàn ảnh.

Manifest của detector **không được app đọc** (`create_detector` chỉ nhận
`model_path`), nên nếu nó ghi sai head thì sửa file JSON, đừng train lại:

```powershell
.\.venv\Scripts\python.exe scripts\fix_detector_manifest.py `
  <thư mục artifact>\model_manifest.json
```

Script đọc shape từ chính ONNX đi kèm, ghi lại `nms`/`max_det` cho đúng và
chạy lại nhiều lần vẫn an toàn.

#### Bootstrap nhãn chân từ board của bạn

[`scripts/bootstrap_lead_labels.py`](scripts/bootstrap_lead_labels.py) chạy
bước 0–5.5 trên ảnh board của bạn rồi xuất ROI chân/pad ra **định dạng YOLO**
kèm ảnh khung phân tích, mở thẳng được bằng LabelImg/CVAT/Roboflow:

```powershell
.\.venv\Scripts\python.exe scripts\bootstrap_lead_labels.py D:\anh_board `
  --output datasets\leads_v1 --model models\detector\kaggle\best.onnx --overlays
```

Box đã vẽ sẵn nên việc của người là **sửa**, không phải vẽ từ đầu — nhanh hơn
nhiều lần. Giá trị lớn nhất là **thêm box ở chân mà hình học bỏ sót**, vì đó
đúng là thứ model đang không thấy.

Hai điều script tự cưỡng chế, vì cả hai đều là bẫy im lặng:
- Ảnh ghi ra là **khung phân tích** (sau tiền xử lý + căn chỉnh), không phải file
  gốc — toạ độ box khớp khung này. Ghi nhầm ảnh gốc thì mọi box lệch theo tỉ lệ
  resize mà nhãn vẫn "đúng định dạng", không có gì báo lỗi.
- Output đánh dấu `PSEUDO_LABELS_NEED_REVIEW` và kèm `README_FIRST.md`. **Train
  thẳng lên nhãn chưa sửa chỉ dạy model lặp lại đúng công thức hình học đã sinh
  ra nó** — recall sẽ đẹp trên chính tập đó mà không thêm thông tin nào so với
  việc gọi thẳng hàm suy ra ROI.

**Không có dataset công khai nào gán nhãn chân/pad thành object riêng.** FPIC —
dataset PCB uy tín nhất — ghi rõ trong bài báo rằng annotation pin là *future
work*. Bản dẫn xuất FPIC-Component chỉ mức linh kiện và mang license **CC BY-NC-ND**
(NonCommercial **và** NoDerivatives — rủi ro thật khi train model). Chi tiết ở
README_v2_models.

### Classifier: EfficientNetV2-S + công thức train kỹ

[pcb_classifier_v2_kaggle.ipynb](training/kaggle/pcb_classifier_v2_kaggle.ipynb):
chia tập **theo ảnh cha** (không theo crop), freeze head rồi mở khoá với
layer-wise LR decay, RandAugment + Mixup/CutMix, EMA, class-balanced sampler,
chọn model theo **macro recall** (không phải accuracy), và tách riêng tập
calibration để chọn ngưỡng.

Đổi backbone bằng `CONFIG["model_name"]`: `efficientnet_v2_s` (mặc định),
`mobilenet_v3_small` (nhanh nhất CPU, dùng nếu Pi không kham nổi), `convnext_tiny`,
`efficientnet_b0`.

### Siết ROI theo kim loại thật (đã đo)

`SolderJointConfig.refine_to_metal` (mặc định bật). Các tỉ lệ hình học quyết định
ROI nằm **ở đâu**; chúng không thể biết land dưới đó **rộng bao nhiêu**. Đo cái đó
từ chính pixel bên trong ROI:

| Cách | Mean IoU với pad thật | Định vị được (IoU≥0.5) |
|---|---:|---:|
| Hình học theo tỉ lệ | 0.236 | 2/26 (8%) |
| Tìm kim loại trong **cả vùng quanh linh kiện** | 0.701 | 25/26 (96%) |
| **Siết trong chính ROI đã dự đoán** ✅ | **0.701** | **26/26 (100%)** |

Hai cách sau ngang nhau trên board tổng hợp, nhưng **cách giữa hỏng trên board
thật**: nó bám vào đường mạch đồng, via và pad của linh kiện bên cạnh. Cách được
chọn chỉ tìm trong ROI đã dự đoán, nên nhiễu ở xa không lọt vào được. Trên ảnh
PCB thật nó siết 16/24 ROI và giảm **33% diện tích ROI trung vị**.

Không siết khi bằng chứng yếu: ROI không có kim loại, hoặc đốm quá nhỏ. Mối hàn
thiếu thiếc **phải giữ nguyên ROI rỗng lớn** — chính sự trống đó là bằng chứng
bước 6.2 cần; co nó lại quanh vài pixel sáng là giấu mất lỗi.

### Kết hợp thuật toán và model ở bước 5.5

[`aoi_pipeline/solder/leads.py`](aoi_pipeline/solder/leads.py) — **ưu tiên
detection thật, quay về hình học suy ra ở chỗ không có**, và chọn **theo từng
chân, không theo từng linh kiện**:

| Tình huống | ROI dùng | `source` |
|---|---|---|
| Detector tìm được chân, chồng ROI suy ra | Box detect | `detected` |
| Detector tìm được chân ở chỗ ROI suy ra không đoán | Thêm vào, giữ cả hai | `detected` + `derived` |
| Detector tìm được 1 trong 2 đầu | Đầu đó detect, **đầu kia vẫn giữ ROI suy ra** | trộn |
| Detector không tìm được gì | Toàn bộ ROI suy ra | `derived` |
| Detection chân quá xa mọi linh kiện | **Giữ làm ROI độc lập**, không gán bừa vào linh kiện gần nhất | `detected` |

Dòng thứ ba là quan trọng nhất: chuyển cả linh kiện sang "detected" khi model chỉ
thấy một đầu sẽ **âm thầm mất đầu kia** — thường đúng là đầu có lỗi. Cấu hình ở
`LeadFusionConfig`; inert hoàn toàn khi detector không báo class `pads`/`pins`.

Dòng cuối từng là một lỗi thật, đã sửa: pad không thuộc linh kiện nào bị **vứt
hẳn**, chỉ để lại warning. Nhưng "không thuộc linh kiện nào" và "không phải mối
hàn" là hai chuyện khác nhau — test point, footprint chưa gắn linh kiện, hoặc pad
mà detector bỏ sót mất thân đều rơi vào đây. Với `pads` đo được precision **0.712**
/ recall **0.072**, model bắn rất ít nhưng bắn thì thường đúng, nên vứt một
detection tự tin là vứt đúng thứ hiếm nhất. Giờ nó thành ROI độc lập
(`terminal_geometry="pad_only"`); tắt bằng `keep_unassigned_leads=False`.

Sau khi nạp detector mới, chạy một board và xem cảnh báo *"dùng N ROI từ detection
chân/pad thật và M ROI suy ra"*. N > 0 nghĩa là detector mới thực sự đóng góp.

## Độ phủ dataset: đo được, và chưa đủ

Số liệu thật từ `pcb_component_detector_artifacts/class_distribution.csv` —
**21.160 instance train, 22 class**:

| Nhóm | Số class | Class |
|---|---:|---|
| Đủ (≥2000) | **3** | capacitor 7775, resistor 7133, ic 2220 |
| Mỏng (500–1999) | 4 | connector 889, transistor 572, diode 551, led 549 |
| Rất mỏng (100–499) | 5 | switch 283, **pins 261**, inductor 213, **pads 186**, fuse 111 |
| Không dùng được (<100) | **9** | clock 89, relay 66, display 63, button 56, potentiometer 50, buzzer 45, battery 42, heatsink 4, transformer 2 |
| Rỗng | 1 | transducer (0 train, 4 test) |

**Ba class chiếm 81% toàn bộ dữ liệu.** Và `heatsink`, `transformer`, `transducer`
có **0 instance trong val** — recall của chúng về mặt kỹ thuật là không đo được,
nên mọi con số báo cho chúng đều vô nghĩa.

Kết luận thẳng: dataset này **phủ đúng 3 trên 22 class**. Với 9 class dưới 100
instance, model không học mà chỉ ghi nhớ. Đây là lý do notebook v2 loại class
dưới `min_per_class` khỏi `class_names` thay vì giả vờ train chúng.

### Với bước 6.2 (mối hàn)

| Lớp | Nguồn | Đủ chưa |
|---|---|---|
| good, insufficient, excess | SolDef_AI | Tạm được, nhưng khác camera/ánh sáng của bạn |
| shift_component | SolDef_AI | Nguồn công khai duy nhất có |
| bridge, missing_solder | HF (không license, nghi dữ liệu sinh) | **Không tin được cho production** |
| **cold** | Roboflow, vài trăm ảnh | **Yếu nhất.** Và cold cần đèn RGB đa góc mới tách được — dataset không bù được quang học |
| tombstone, wrong_polarity | không nguồn nào | **Không có** |

### Cần bao nhiêu mới đủ

Ước lượng thực tế cho AOI: **≥500 instance/class** để học được, **≥2000** để ổn
định, và tối thiểu **50–100 instance/class trong val** mới đo được recall đáng
tin. Chiếu vào bảng trên: cần thêm dữ liệu cho **19 trên 22 class** của detector,
và cho gần như mọi lớp lỗi của 6.2.

Nguồn dữ liệu giá trị nhất vẫn là board của chính bạn — nó là nguồn duy nhất khớp
camera, ống kính và ánh sáng thật. `scripts/export_solder_dataset.py --overlays`
sinh sẵn ROI ứng viên để người **sửa** thay vì vẽ từ đầu.

## Tăng độ chính xác không cần train lại

Rà lại toàn bộ khung (tiền xử lý bước 1, detector, hai classifier) trong lúc chờ
model train trên Kaggle. Hai việc thật, không phải lý thuyết:

### 1. Nghi vấn: tiền xử lý bước 1 có thể đang lệch miền so với model đã train

`ImagePreprocessor` ([aoi_pipeline/imaging/preprocessing.py](aoi_pipeline/imaging/preprocessing.py))
mặc định bật cả 5 bước: denoise, white-balance, CLAHE, normalize luminance,
unsharp mask — chạy trên **mọi** ảnh trước khi đưa vào cả detector lẫn crop cho
hai classifier. Đã kiểm tra cả 3 notebook train (detector v1/v2, classifier v2,
solder v2): **không notebook nào áp dụng chuỗi này lên ảnh train** — chúng train
trên ảnh thô của dataset cộng augmentation chuẩn (mosaic/HSV cho YOLO,
RandAugment cho classifier). Nghĩa là ảnh đưa vào model lúc suy luận có thể khác
thống kê pixel so với ảnh model từng thấy lúc train — đặc biệt CLAHE (viết lại
tương phản cục bộ) và unsharp mask (khuếch đại biên) là hai phép biến đổi mà một
CNN/detector backbone rất nhạy.

**Chưa đo được trên ảnh board thật** — repo này không có sẵn ảnh board thật nào
(chỉ có fixture tổng hợp), nên chưa thể kết luận chiều nào đúng, chỉ nêu đúng
mức độ rủi ro đã xác minh được qua code. Dùng script mới
[`scripts/compare_preprocessing_ab.py`](scripts/compare_preprocessing_ab.py) để
tự đo trên board thật của bạn ngay khi có model:

```powershell
.\.venv\Scripts\python.exe scripts\compare_preprocessing_ab.py D:\anh_board `
  --model models\detector\kaggle\best.onnx --isolate
```

So khớp detection giữa "ảnh thô" và "ảnh đã tiền xử lý" theo IoU+class, báo số
box mất/thêm và độ lệch confidence trung bình. Cờ `--isolate` bật lần lượt từng
bước (denoise/white_balance/clahe/normalize/sharpen) để biết chính xác bước nào
là thủ phạm nếu có, thay vì đổ cho "tiền xử lý" nói chung. Nếu kết quả cho thấy
mất detection/giảm confidence rõ rệt: tắt bớt trong `PreprocessConfig`, hoặc để
lâu dài thì train lại trên ảnh đã qua đúng chuỗi tiền xử lý này.

### 2. Đã thêm: TTA (test-time augmentation) lúc suy luận

Ba "helper" chạy model — `ONNXComponentClassifier`, `ONNXSolderClassifier`,
`UltralyticsDetector` — trước đó chỉ chạy đúng 1 lượt forward mỗi ảnh dù notebook
train bước 6.1 đã **đo được** lợi ích của việc trung bình 4 góc nhìn (gốc + lật
ngang + lật dọc + lật cả hai): macro recall 0.9292 → 0.9417 trên cùng một bộ
trọng số. Đó là lợi ích miễn phí bị bỏ lại ở lúc deploy — không cần train lại.

Đã bật tuỳ chọn (mặc định **tắt**, vì tăng ~4x thời gian suy luận của bước đó):

```python
config.classification.tta = True     # bước 6.1, đã đo lợi ích ở trên
config.solder_grading.tta = True     # bước 6.2, cùng kỹ thuật nhưng chưa đo trên model này
config.model_detector.tta = True     # bước 4, augment=True có sẵn của Ultralytics
```

`classification.tta`/`solder_grading.tta` trung bình softmax qua đúng 4 view mà
notebook 6.1 đã kiểm chứng. `model_detector.tta` bật `augment=True` tích hợp sẵn
của Ultralytics (đa tỉ lệ + lật, tự hợp nhất trước NMS) — kỹ thuật chuẩn, thường
+0.5–2 mAP, chưa đo riêng trên model của dự án này. Test ở
[tests/test_classification.py](tests/test_classification.py),
[tests/grading/test_solder_grading.py](tests/grading/test_solder_grading.py),
[tests/test_detectors.py](tests/test_detectors.py) kiểm cả
việc 4 view thật sự được lật đúng chiều lẫn xác suất được trung bình đúng.

## Cấu trúc dự án

Thư mục nhóm theo **bước của pipeline** — cùng cách cả dự án vẫn được mô tả, nên
tìm code của một bước là mở đúng thư mục mang tên bước đó.

```text
app/                    Streamlit UI và bridge sang pipeline
aoi_pipeline/
  ├─ models.py          Kiểu dữ liệu và hàm hình học dùng chung (BoundingBox, IoU…)
  ├─ config.py          Toàn bộ dataclass cấu hình của mọi bước
  ├─ exceptions.py      Cây ngoại lệ
  ├─ pipeline.py        Facade AOIPipeline: ghép các bước lại
  │
  ├─ imaging/           Bước 0–3 · từ file ảnh tới vùng board đã căn chỉnh
  │    image_io · preprocessing · calibration · alignment · board
  ├─ detection/         Bước 4–5 · tìm linh kiện rồi cắt ra
  │    detectors · tiling · cropping
  ├─ solder/            Bước 5.5 · chân hàn nằm ở đâu
  │    geometry · leads · lead_detection · cad · cad_fusion
  ├─ classification.py  Bước 6.1 · phân loại họ linh kiện
  ├─ grading/           Bước 6.2 · đo vật lý → luật → model ONNX → hợp nhất
  │    features · rules · classifier · inspector · datasets
  ├─ golden/            Golden Inspection · workspace riêng, không thuộc 0–6.2
  │    recipe · inspector · position · compare
  ├─ exporters.py       Xuất JSON/CSV/ZIP
  └─ overlays.py        Vẽ chồng lên ảnh

tests/                  Unit test, gương theo cấu trúc trên
training/kaggle/        Notebook train: detector (4), classifier (6.1),
                        solder (6.2), lead detector (lượt 2)
models/                 Model local — chỉ commit .onnx + model_manifest.json
Docs/                   Khảo sát dataset, kế hoạch, báo cáo, yêu cầu phần cứng
scripts/                Setup/chạy app, bootstrap nhãn chân, kiểm artifact
```

### Vì sao trước đây là layout phẳng, và vì sao đổi

Bản trước ghi *"dùng layout phẳng, đừng tạo lại subpackage"*. Lời cảnh báo đó có
lý do thật: một lần merge từng tạo thư mục `inspection/` trong khi đã có
`inspection.py`, và hai thứ cùng tên ở cùng cấp làm hỏng import. Làm phẳng là
cách chữa nhanh, không phải kết luận rằng phẳng thì tốt hơn.

Lần này gom nhóm nhưng **đổi tên để không còn cặp nào trùng**: `solder.py` thành
`solder/geometry.py`, `inspection.py` thành `golden/inspector.py`,
`golden_compare.py` thành `golden/compare.py`. Không có module nào cùng tên với
thư mục chứa nó.

Bề mặt công khai không đổi: `aoi_pipeline/__init__.py` vẫn export đúng **145
tên** như trước, và một test đối chiếu để không ai làm rơi tên nào. Import trong
package dùng dạng tương đối theo tầng: `from ..models import ...` từ trong một
subpackage, `from .leads import ...` giữa các module cùng nhóm.

## Giới hạn hiện tại

- Khoanh PCB ở bước 3 đang dùng contour fallback, chưa có PCB detector riêng.
- CV proposal ở bước 4 không thay thế model đã train.
- Baseline dùng `max_det=2000` cho board dày linh kiện; cần tune lại theo SKU/tốc độ.
- Adaptive tiling tăng recall cho linh kiện nhỏ nhưng tăng thời gian gần tỷ lệ với
  số tile; cần benchmark tile size/overlap trên máy chạy thật và Raspberry Pi.
- Căn chỉnh chính xác cần một Golden Image/reference cùng board side.
- Model cuối phải được đánh giá trên camera, lens, ánh sáng và PCB của dây chuyền.
- Mỗi ảnh import được giới hạn 64 MB/50 MP; upload Streamlit tối đa 256 MB/file.
- Ultralytics công bố lựa chọn AGPL-3.0/Enterprise; cần duyệt license trước khi
  đưa framework/model vào sản phẩm thương mại hoặc phần mềm đóng nguồn.
