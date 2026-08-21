# Automated Optical Inspection PCB

Ứng dụng local có hai workspace: Golden Inspection dùng recipe cố định cho
Position/Appearance, và pipeline thử nghiệm từ bước 0 đến bước 6.1:

```text
0. Import ảnh
   → 1. Undistort tùy chọn + tiền xử lý
   → 2. Căn chỉnh với Golden Image (tùy chọn)
   → 3. Khoanh vùng PCB
   → 4. Phát hiện linh kiện
   → 5. Crop và xuất dữ liệu linh kiện
   → 6.1. Phân loại family (accept/review/unknown)
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

## Golden Inspection

Workspace mặc định của UI là **Golden Inspection**:

1. Nạp Golden Image ở sidebar. Ảnh JPEG đầu vào demo được chấp nhận, nhưng
   recipe luôn lưu lại Golden, anchor, template và mask dưới dạng PNG lossless.
2. Nạp ảnh board cần kiểm tra và dùng `models/detector/best.onnx` có sẵn, hoặc
   chọn detector khác trong sidebar.
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

## Cấu trúc dự án

```text
app/                 Streamlit UI và bridge
aoi_pipeline/        Pipeline OpenCV/model cho bước 0–6.1
tests/               Unit tests
training/kaggle/     Notebook train detector bước 4 và classifier bước 6.1
models/              Nơi đặt model local (weights không commit Git)
Docs/                Khảo sát dataset và kế hoạch pre-train 6.1
scripts/             Setup/chạy app trên Windows
```

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
