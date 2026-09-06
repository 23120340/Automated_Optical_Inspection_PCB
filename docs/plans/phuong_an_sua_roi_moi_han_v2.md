# Phương án V2 — tách miền ảnh, định vị chân IC tại mép thân và chặn nhiễu lụa

> Trạng thái: thiết kế để triển khai và A/B; chưa phải kết luận production.
>
> Phạm vi: lỗi ROI mối hàn nhận nhầm chữ lụa/OCR và viền trắng, đặc biệt ở IC,
> SOT-23 D201/D202 và ảnh đầu vào đã được tiền xử lý từ dataset.

## 1. Kết luận ngắn

Không nên xóa hoặc inpaint chữ và viền trắng trực tiếp trên ảnh đưa vào model.
Hướng đó có ba rủi ro: xóa nhầm chân thật, tạo biên giả sau inpaint, và làm ảnh runtime
khác miền ảnh đã train.

Phương án V2 tách bài toán thành ba phần độc lập:

1. **Tách miền ảnh**: detector/classifier và phép đo ROI không còn buộc dùng chung một
   ảnh đã qua toàn bộ CLAHE/normalize/sharpen.
2. **Tách định vị khỏi đo**: với IC, pha tìm tâm hàng chân chỉ nhìn một dải hẹp sát mép
   thân, không mở rộng sâu ra ngoài. Sau khi đã có tâm từng chân mới dựng ROI đo vươn
   ra pad/fillet theo giới hạn vật lý.
3. **Không xóa pixel nhiễu**: tạo `silkscreen_likelihood_mask` riêng cho chữ lụa và
   đường viền trắng. Mask chỉ là bằng chứng âm khi chấm ứng viên ROI; crop gốc cho
   model vẫn được giữ nguyên.

Điểm quan trọng: đặt `lead_outer_ratio = 0` ngay trên code hiện tại **không phải** bản
hoàn chỉnh của ý “IC không mở box”. Hiện `_multi_pin_rects` dùng cùng một dải cho cả
tìm chân lẫn tạo ROI cuối. Đặt bằng 0 có thể làm sạch chữ nhưng đồng thời cắt mất pad,
fillet và khoảng giữa hai chân cần để phát hiện bridge. Cần tách hai pha trước.

## 2. Những gì đã được kiểm chứng

Nguồn đối chiếu:

- `ĐÁNH GIÁ MODEL DETECT SOLDER VÀ TÌM HƯỚNG GIẢI QUYẾT.docx`;
- `phuong_an_sua_roi_moi_han.md`;
- lịch sử Claude của workspace, gồm yêu cầu cuối chưa được thực hiện do lỗi 529:
  “Test thử chỉnh IC không mở box xem, tôi thấy khá ổn đối với IC”;
- đường chạy hiện tại từ Streamlit đến `AOIPipeline`, ROI và bước 6.2.

Các kết quả cũ vẫn có giá trị:

- chữ lụa `HDL01` bị `segment_solder` xem là kim loại **49,5%**, cao hơn hai dải
  chân IC thật **38,3%** và **29,9%**;
- comb/evenness đã sửa đúng ca SOIC-8 trên ảnh tham chiếu: 15 ROI/7 ROI chữ thành
  8 ROI/0 ROI chữ;
- D201/D202 vẫn còn ROI giả ở viền vì dải chỉ có một hoặc hai đốm; comb chủ động
  không kết luận khi dưới ba đốm;
- PnP trong phép thử cũ được dựng từ detection để kiểm đường ống, chưa phải tọa độ
  pick-and-place thật.

Cần sửa hai cách gọi trong DOCX để không chữa nhầm tầng:

- box vàng/cyan trong các ảnh chi tiết là overlay ROI bước 5.5, không phải output của
  model YOLO solder-defect độc lập; cyan là `body view` để hiển thị, còn box vàng
  `joint` mới là ROI đo;
- D201/D202 đi vào topology `multi_pin` vì **component detector bước 4** gọi chúng là
  `ic`. Classifier 6.1 chạy sau khi ROI solder đã được tạo, nên classifier không làm
  lệch các ROI này trong pipeline hiện tại.

Chạy lại end-to-end với code hiện tại trên đúng ảnh IC trong DOCX cho bbox khoảng
`[731, 659, 843, 788]` đã sinh **8 chân và 0 ROI trên chữ `U201`**. Comb/evenness hiện
đã sửa được đúng ca này; ảnh DOCX phản ánh code/config cũ. Vì vậy không được bật
“không mở rộng IC” toàn cục chỉ để chữa lại một lỗi hiện không còn tái hiện. Khoảng
trống còn tái hiện hợp lý là D201/D202 ít chân, capacitor gần vuông và miền ảnh bị xử
lý lại.

Đã chạy lại A/B detector trên
`C:\Users\Acer\Downloads\00001__1024__1648___4120.png`, cùng model ONNX và cùng
ngưỡng:

| Nhánh | Số detection | Mất/thêm box | Chênh confidence trung bình so với source-as-received |
|---|---:|---:|---:|
| Không tăng cường quang học lần hai | 39 | — | 0 |
| Chuỗi mặc định hiện tại | 39 | 0/0 | **-0,0225** |
| Chỉ white balance | 39 | 0/0 | **-0,0227** |
| Chỉ denoise | 39 | 0/0 | -0,0114 |

Phép đo này chỉ nói detector không mất box trên một ảnh. Nó không bác bỏ lỗi ROI:
phép đo trước đó cho thấy CLAHE + normalize + sharpen làm pixel sáng của chữ lụa tăng
từ 50,3% lên 87,2%, còn solder tăng từ 53,1% lên 100%. Thông tin màu cần để phân biệt
hai vật liệu đã bị mất trước khi `segment_solder`, refine và rule 6.2 đọc ảnh.

Probe hình học cô lập trên U201 (cố định một detection, tắt refine, chưa fusion với
linh kiện lân cận) cho kết quả:

| `lead_outer_ratio` | Source-as-received | Ảnh qua pipeline mặc định |
|---:|---:|---:|
| 0,26 hiện tại | 9 ROI | 12 ROI |
| 0,10 | 8 ROI | 8 ROI |
| 0,05 | 8 ROI | 8 ROI |
| 0,00 | 8 ROI | 8 ROI |

Điều này xác nhận strip ít/không vươn ra ngoài là locator tốt trên ca đó. Nó **không**
xác nhận `outer=0` là ROI đo tốt: các box ở arm 0,00 dừng tại mép body bbox và không
phủ hết fillet/pad. Kết quả end-to-end 8/0 ở trên cũng cho thấy P2 hiện tại có thể loại
chữ mà không cần đổi ROI cuối.

## 3. Đính chính provenance của ảnh

Tên “ảnh gốc” trong tài liệu cũ phải được hiểu là **source-as-received**: file đã qua
tiền xử lý khi tạo dataset, không phải sensor RAW. Pipeline hiện không gọi nhầm
`ImagePreprocessor` hai lần trên cùng test board; nhưng nó áp thêm một chuỗi quang học
toàn cục lên file vốn đã được xử lý, rồi mỗi model lại áp transform riêng của model.

Vì vậy có hai vấn đề khác nhau:

- **double photometric processing** theo provenance: dataset đã xử lý, pipeline xử lý
  quang học thêm một lần;
- **double spatial normalization** ở classifier: cropper letterbox trước, classifier
  lại letterbox theo manifest. Khi hai kích thước trùng nhau nó gần như idempotent;
  khi UI đổi kích thước, ảnh bị resample/padding thật sự hai lần.

Không được dùng kết quả trên file này để suy ngược chất lượng camera RAW tương lai.
Khi có camera, phải lưu cả RAW/source và ảnh dẫn xuất, đồng thời ghi rõ profile xử lý.

## 4. Kiến trúc ảnh mới

```text
source-as-received
        |
        +-- geometric transform duy nhất (undistort/resize/alignment) --> radiometric_analysis
        |                                                               |-- ROI geometry
        |                                                               |-- segment/refine/rules
        |                                                               +-- nuisance mask
        |
        +-- profile đúng contract từng model --------------------------> model_input
                                                                        |-- detector
                                                                        +-- classifier
```

Đề xuất kiểu dữ liệu `ImageDomains`/`FrameBundle` mang tối thiểu:

- `source_bgr`;
- `radiometric_analysis_bgr`: cùng kích thước và tọa độ với ảnh phân tích, nhưng không
  qua WB/CLAHE/normalize/sharpen ngoài contract;
- `enhanced_analysis_bgr` nếu một model đã được A/B chứng minh cần nó;
- ma trận source → analysis, valid mask;
- `input_profile_id`, `preprocess_count`, hash cấu hình và provenance
  (`camera_raw`, `dataset_preprocessed`, `unknown`).

Tất cả nhánh phải có cùng H×W và cùng `coordinate_space_id`; tuyệt đối không cắt bbox
đã align trực tiếp trên source chưa align.

Chính sách ban đầu:

- ảnh dataset hiện tại: bỏ denoise/WB/CLAHE/normalize/sharpen lần hai;
- ROI geometry, pin split, `segment_solder`, `refine_to_metal` và rule 6.2 đọc
  `radiometric_analysis_bgr`;
- learned model đọc nhánh đúng với `input_profile_id` trong manifest. Nếu notebook
  train trên source dataset + transform chuẩn của model, runtime cũng phải như vậy;
- giữ ảnh ROI chưa letterbox cho classifier; chỉ classifier theo manifest được resize,
  normalize và letterbox một lần.

## 5. IC: không mở rộng ở pha định vị, vẫn mở có kiểm soát ở ROI cuối

### 5.1 Pha A — định vị hàng chân

Từ bbox thân IC, tạo bốn strip ứng viên nhưng strip chỉ ôm sát mép thân:

- phần trong thân nhỏ để bắt điểm chân nối với package;
- phần ngoài bằng 0 hoặc rất nhỏ trong A/B đầu tiên;
- không kéo strip qua bốn góc;
- chấm theo profile 1-D dọc cạnh.

Một cặp cạnh được nhận là hàng chân khi nhiều tín hiệu đồng thuận:

- hai cạnh đối diện;
- số đốm gần nhau hoặc phù hợp package/CAD;
- pitch đều, kích thước đốm tương tự;
- đốm chạm `attachment corridor` sát mép thân;
- hướng đốm vuông góc với cạnh package;
- không trùng vùng nuisance có độ tin cậy cao.

Với cạnh chỉ có một/hai chân như SOT-23, pitch không đủ căn cứ. Khi đó ưu tiên theo
thứ tự: pad CAD > footprint/PnP + package > lead detector lượt 2 > hình học và đưa
ca mơ hồ vào review. Không dùng nhãn `ic` từ detector/classifier để tự sửa D201 vì hai
tầng từng cùng nhận sai khi được đánh giá độc lập; trong runtime hiện tại phải sửa từ
label bước 4/topology hoặc footprint, không trông chờ classifier 6.1 chạy phía sau.

### 5.2 Pha B — dựng ROI đo từng chân

Sau khi có tâm chân, tạo ROI riêng:

- `inner_depth` phủ đoạn chân nối vào package;
- `outer_depth` vươn ra pad/fillet, có cap theo mm hoặc pad CAD;
- chiều dọc ROI dựa trên pitch;
- padding phải phủ một phần khoảng inter-pin để bridge không biến mất;
- lưu `locator_bbox` và `measurement_bbox` riêng để overlay/audit.

Như vậy ý tưởng “không mở box IC” được dùng đúng chỗ: ngăn chữ `U201/HDL01` tham gia
pha tìm tâm, nhưng không làm mù phần vật lý nằm ngoài thân IC.

## 6. Chữ OCR và viền trắng: mask phụ, không phải eraser

Tên đúng của vấn đề hiện tại là **silkscreen/white-marking hard negative**, không phải
lỗi OCR; pipeline chưa có OCR engine. Không cần nhận đúng ký tự mới chặn được nó.

Tạo `silkscreen_likelihood_mask` trên ảnh radiometric từ các nhóm bằng chứng:

- connected components sáng, ít bão hòa;
- stroke width gần ổn định, nhiều component chung baseline/chiều cao → giống dòng chữ;
- component dài, mảnh, liên tục qua khoảng xa → giống viền/đường lụa;
- không chạm attachment corridor hoặc không theo pitch hàng chân;
- khác hình học pad trong CAD/footprint.

Mask được dùng như sau:

- không tô đen/inpaint ảnh model;
- trừ điểm candidate trong pha chọn band/pin;
- hard reject chỉ khi nuisance confidence cao **và** candidate không chạm hành lang nối
  chân/pad whitelist;
- nếu tín hiệu mâu thuẫn, giữ ROI nhưng đánh `review` và ghi reason;
- giai đoạn đầu chạy shadow mode: vẽ mask/score nhưng chưa xóa ROI.

Một classifier nhỏ bảy đặc trưng có thể bổ sung score, nhưng số 78,8% held-out-region
trước đây chỉ đến từ vài vùng trên cùng một ảnh; chưa đủ làm hard gate production.

## 7. Thứ tự triển khai

### P0 — Quan sát và khóa contract

- thêm provenance, `input_profile_id`, `preprocess_count` và hash profile;
- export overlay gồm body bbox, locator strip, measurement ROI, nuisance mask;
- báo lỗi khi model manifest yêu cầu profile khác profile runtime.

### P1 — Tách miền ảnh, chưa đổi quyết định

- dựng `radiometric_analysis_bgr` bằng cùng transform hình học;
- luồn nó đến ROI/rule dưới feature flag;
- bỏ letterbox thứ nhất khỏi đường model, giữ preview riêng;
- kiểm pixel/coordinate identity bằng test.

### P2 — Bỏ photometric pass thứ hai cho ảnh dataset

- cấu hình source là `dataset_preprocessed`;
- detector/classifier dùng đúng contract;
- geometry/refine/rule dùng radiometric branch;
- A/B paired với pipeline cũ.

### P3 — Tách locator và measurement ROI cho multi-pin

- thêm `lead_locator_inner_ratio`, `lead_locator_outer_ratio`;
- mặc định A/B `outer = 0`, không thay `lead_outer_ratio` của ROI cuối;
- fit cặp hàng chân và dựng lại ROI từ pin centers;
- fallback về hành vi cũ khi locator abstain, không trả về rỗng im lặng.

### P4 — Nuisance mask chạy shadow

- log từng candidate bị mask đề nghị hạ điểm;
- chỉ bật reject sau khi hard-negative test đạt gate;
- ưu tiên pad CAD/PnP làm positive whitelist khi có dữ liệu thật.

### P5 — Camera và fine-tune

- lưu camera RAW/source và ảnh dẫn xuất riêng;
- khóa ánh sáng, scale, exposure và profile;
- fine-tune trên ảnh cùng dây chuyền; split theo board/SKU, không random theo crop.

## 8. A/B bắt buộc trước khi bật mặc định

| Arm | Ảnh dùng cho ROI/rule | Locator IC | Nuisance mask |
|---|---|---|---|
| A | global enhanced hiện tại | dải hiện tại | tắt |
| B | radiometric/source-as-received | dải hiện tại | tắt |
| C | radiometric | strip sát mép, outer=0 | tắt |
| D | radiometric | strip sát mép | shadow |
| E | radiometric | strip sát mép | reject có điều kiện |

Giữ cố định model, manifest, threshold, NMS, detection và CAD giữa các arm. Bộ test phải
tách theo parent image/board/SKU và có hard negatives: chữ lụa, viền trắng, via/trace,
tụ vỏ kim loại, linh kiện sát nhau.

Chỉ số cần ghi:

- recall một ROI cho mỗi terminal thật, pin-count error, pad IoU;
- coverage fillet và khoảng inter-pin;
- false ROI trên chữ/viền theo từng linh kiện;
- số lần locator abstain/fallback;
- defect recall, escape, false call, review rate của 6.2;
- clipped-pixel fraction và phân bố `solder_ratio` trước/sau;
- confidence/bbox detector để tách lỗi detector khỏi lỗi ROI.

Gate tối thiểu:

- không giảm terminal/defect recall trên locked test;
- không làm mất vùng inter-pin dùng phát hiện bridge;
- giảm rõ rệt false ROI chữ/viền, mục tiêu ban đầu ≥80%;
- không có ROI biến mất mà không mang reason/fallback;
- trên tile trong tài liệu: U201 phải giữ đủ 8 chân và 0 ROI chữ; D201/D202 phải giữ
  đúng các chân thật và 0 ROI viền. Đây chỉ là smoke test, không phải validation set.

## 9. Vị trí code dự kiến

- `aoi_pipeline/imaging/preprocessing.py`: sinh các domain ảnh và profile metadata;
- `aoi_pipeline/pipeline.py`: luồn riêng detection/model/evidence image;
- `app/streamlit_app.py`, `app/pipeline_bridge.py`: giữ source và analysis domains;
- `aoi_pipeline/solder/geometry.py`: tách locator strip khỏi measurement ROI;
- `aoi_pipeline/grading/inspector.py`: rule đọc radiometric ROI;
- `aoi_pipeline/detection/cropping.py`, `aoi_pipeline/classification/family.py`,
  `aoi_pipeline/grading/classifier.py`: loại double letterbox;
- module mới `aoi_pipeline/solder/nuisance.py`: mask/score và reason codes.

Test mới phải khóa: cùng hệ tọa độ giữa các domain; photometric branch không làm đổi
radiometric pixels; một lần resize cho classifier; IC giữ đủ pin khi outer locator bằng
0; `refine_to_metal` không snap vào chữ/viền; fallback không làm linh kiện biến mất.

## 10. Quyết định đề xuất

Triển khai P0 → P1 → P2 trước. Đây là phần sửa đúng nguyên nhân double-processing và
cho phép đo sạch mà chưa hard-delete ROI. Sau đó triển khai P3 bằng feature flag để A/B
ý tưởng không mở rộng IC. Chỉ thêm P4 reject khi shadow data chứng minh mask không xóa
chân thật. Không thêm OCR engine hoặc inpaint ở vòng đầu.
