# Automated Optical Inspection PCB

Ứng dụng local thử nghiệm luồng AOI từ bước 0 đến bước 6.1:

```text
0. Import ảnh
   → 1. Undistort tùy chọn + tiền xử lý
   → 2. Căn chỉnh với Golden Image (tùy chọn)
   → 3. Khoanh vùng PCB
   → 4. Phát hiện linh kiện
   → 5. Crop và xuất dữ liệu linh kiện
   → 5.5. Suy ra ROI mối hàn (dữ liệu cho 6.2)
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

Nếu có Gerber/CAD hoặc file pick-and-place thì chiếu thẳng tọa độ land qua
homography ở bước 2 sẽ chính xác hơn hẳn cách suy ra từ bounding box; bước 5.5
là phương án cho board không có dữ liệu CAD.

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

## Cấu trúc dự án

```text
app/                 Streamlit UI và bridge
aoi_pipeline/        Pipeline OpenCV/model cho bước 0–6.1 (kèm 5.5 ROI mối hàn)
tests/               Unit tests
training/kaggle/     Notebook train detector bước 4 và classifier bước 6.1
models/              Nơi đặt model local (weights không commit Git)
Docs/                Khảo sát dataset và kế hoạch pre-train 6.1
scripts/             Setup/chạy app, calibrate camera, export dataset 6.2
```

## Giới hạn hiện tại

- Khoanh PCB ở bước 3 đang dùng contour fallback, chưa có PCB detector riêng.
- ROI bước 5.5 suy ra từ box nên chỉ đúng khi class của detector đúng; class sai
  kéo theo topology chân sai. Có CAD thì nên chiếu land qua homography.
- Ước lượng góc xoay linh kiện (`estimate_orientation`) mặc định tắt: góc sai làm
  lệch mọi ROI suy ra từ nó, đắt hơn là để ROI axis-aligned hơi rộng.
- Chưa có model bước 6.2; bước 5.5 chỉ tạo ROI và bảng nhãn, không chấm mối hàn.
- CV proposal ở bước 4 không thay thế model đã train.
- Baseline dùng `max_det=2000` cho board dày linh kiện; cần tune lại theo SKU/tốc độ.
- Adaptive tiling tăng recall cho linh kiện nhỏ nhưng tăng thời gian gần tỷ lệ với
  số tile; cần benchmark tile size/overlap trên máy chạy thật và Raspberry Pi.
- Căn chỉnh chính xác cần một Golden Image/reference cùng board side.
- Model cuối phải được đánh giá trên camera, lens, ánh sáng và PCB của dây chuyền.
- Mỗi ảnh import được giới hạn 64 MB/50 MP; upload Streamlit tối đa 256 MB/file.
- Ultralytics công bố lựa chọn AGPL-3.0/Enterprise; cần duyệt license trước khi
  đưa framework/model vào sản phẩm thương mại hoặc phần mềm đóng nguồn.
