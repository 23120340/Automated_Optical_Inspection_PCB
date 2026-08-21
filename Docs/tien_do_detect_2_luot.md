# Tiến độ — Detect mối hàn theo 2 lượt

> File này là **bảng công việc sống**. Quy ước bắt buộc cho mọi agent/người làm:
> ghi hạng mục ra **trước khi** bắt tay, đổi trạng thái sang `ĐANG LÀM` khi bắt
> đầu, và tích `HOÀN THÀNH` **ngay khi xong** kèm bằng chứng đã đo.
>
> Cập nhật lần cuối: 2026-08-21

## Quy ước trạng thái

| Ký hiệu | Nghĩa |
|---|---|
| `[ ] CHƯA LÀM` | Đã lên kế hoạch, chưa ai đụng vào |
| `[~] ĐANG LÀM` | Có người đang làm dở — đừng làm trùng |
| `[x] HOÀN THÀNH` | Xong, đã kiểm chứng, có số đo kèm theo |
| `[!] CHẶN` | Không làm tiếp được, ghi rõ đang chờ gì |

---

## Bối cảnh: vì sao đổi sang 2 lượt

Detect thẳng mối hàn trên ảnh board rộng bị nhiễu. Hướng mới: lượt 1 detect
thân linh kiện trên ảnh lớn (như hiện tại), lượt 2 detect chân/pad **bên trong
crop của từng linh kiện**.

### Số đo nền, đo trên board thật của dự án

Tile `00001__1024__1648___4120.png`, detector `models/active/detector/best.onnx`,
38 detection.

| Hạng mục | Số đo | Nguồn |
|---|---|---|
| Độ phân giải thật của ảnh | **~46 µm/px** | 3 phép đo độc lập, xem `Docs/yeu_cau_phan_cung_camera.md` |
| Cần cho kiểm tra fillet | 15–25 µm/px | báo cáo tiến độ dự án |
| Ảnh crop so với toạ độ | **nặng gấp 414 lần** | 11.6 MB ảnh so với 0.028 MB toạ độ |
| Crop so với chính ảnh gốc | **gấp 3.7 lần** | board 5144²: 292 MB crop / 79 MB ảnh gốc |
| Dùng lại detector cho lượt 2 | **0/24 cấu hình ra `pads`/`pins`** | 8 linh kiện × 3 mức biên |

**Kết luận nền:** kiến trúc 2 lượt khả thi và dự án đã có sẵn phần lớn đường
ống. Nút thắt **không phải kiến trúc mà là model cho lượt 2** — detector hiện
tại không dùng lại được, đã đo.

---

## Giai đoạn A — Bộ nhớ: lưu toạ độ thay vì lưu ảnh

Mục tiêu: bỏ việc giữ mảng ảnh cho từng ROI; cắt lại từ ảnh phân tích khi cần
hiển thị. Không phụ thuộc model nào, làm được ngay.

- `[x] HOÀN THÀNH` **A1.** `SolderCropRecord.image` nay là `np.ndarray | None`;
  bridge có cờ `keep_images` (mặc định `False`). Không đổi thứ tự trường nên
  test dựng record theo vị trí vẫn chạy.
- `[x] HOÀN THÀNH` **A2.** Record dựng không kèm ảnh. **Phát hiện thêm:** ảnh bị
  giữ *hai lần* — một lần ở `image`, một lần nữa ở `raw.image`. Đã bỏ cả hai.
- `[x] HOÀN THÀNH` **A3.** `_roi_pixels_for_display(source, crop)` cắt lại từ ảnh
  phân tích; nếu record có sẵn ảnh thì dùng ảnh đó, không có ảnh lẫn khung thì
  trả `None` chứ không làm sập tab.
- `[x] HOÀN THÀNH` **A4.** Đo trên board thật (119 ROI):

  | | ảnh ở `record` | ảnh ở `.raw` | tổng |
  |---|---|---|---|
  | CŨ | 5.85 MB | 5.85 MB | **11.70 MB** |
  | MỚI | 0.00 MB | 0.00 MB | **0.00 MB** |

  Ngoại suy: board 5144² (~958 linh kiện, 2874 ROI) **283 MB → ~0 MB**;
  board 8192² (~2432 linh kiện, 7296 ROI) **717 MB → ~0 MB**.
  Verdict giữ nguyên 119/119, toạ độ giữ nguyên từng pixel.
- `[x] HOÀN THÀNH` **A5.** 5 test mới: record không giữ pixel; bỏ pixel không đổi
  bất kỳ verdict nào; UI cắt ra đúng vùng ROI; `keep_images=True` vẫn dùng được;
  không có khung lẫn ảnh thì không sập.
- `[ ] CHƯA LÀM` **A6.** *(mới, phát sinh)* Crop linh kiện bước 5 vẫn giữ ảnh
  trong `CropRecord.image` **và** `CropRecord.raw.image` (147 KB/cái). Bước 6.1
  đang dùng `raw` nên phải cắt lại lúc phân loại — đụng chạm hơn A1–A5, tách
  riêng. Ước tính tiết kiệm thêm ~141 MB cho board 5144².

## Giai đoạn B — Đường ống lượt 2

Mục tiêu: có chỗ cắm model detect chân, chạy được ngay cả khi **chưa có model**
(khi đó rơi về hình học suy ra như hiện nay).

- `[x] HOÀN THÀNH` **B1.** `LeadDetectionConfig` trong `aoi_pipeline/config.py`:
  `enabled`, `crop_margin_ratio` 0.35, `crop_margin_min_px` 6, `min_crop_px` 24,
  `confidence` 0.25, `min_lead_px` 3. Chỉ đọc khi caller đưa section vào, không
  để key lạ bật nhầm.
- `[x] HOÀN THÀNH` **B2.** Module mới `aoi_pipeline/solder/lead_detection.py`:
  `component_crop_window` (crop kèm biên để lộ fillet ra ngoài thân),
  `to_board_coordinates` (quy đổi), `detect_leads_in_components` (chạy cả loạt).
  **Toàn bộ câu chuyện toạ độ chỉ là một phép cộng** — crop là cửa sổ cắt thẳng
  từ ảnh phân tích, không resize, nên `global = local + crop_origin`. Không cần
  hướng phân cấp.
- `[x] HOÀN THÀNH` **B3.** Nối vào `AOIPipeline.make_solder_crops`: chân lượt 2
  nhập chung với chân từ lượt 1 rồi đi qua `fuse_detected_leads` sẵn có. Có test
  khẳng định chân đo được **thắng** hình học suy ra và ROI rơi đúng chỗ (±2 px).
- `[x] HOÀN THÀNH` **B4.** Không model / tắt stage = no-op tuyệt đối. Test so
  sánh từng ROI trước và sau khi thêm stage: giống hệt. Detector còn không được
  gọi khi stage tắt.
- `[x] HOÀN THÀNH` **B5.** 15 test, gồm: quy đổi đúng ở cả 4 góc crop; **cắt
  crop và cắt board tại toạ độ đã quy đổi phải ra pixel giống hệt nhau**; biên
  bị chặn ở mép ảnh; lead confidence thấp/quá nhỏ bị loại; một linh kiện lỗi
  inference không làm mất các linh kiện còn lại.
  Đã kiểm chứng test **bắt được lỗi thật**: ngắt dây nối ở pipeline thì
  `test_detected_leads_reach_the_fusion_stage` fail đúng như mong đợi.
- `[ ] CHƯA LÀM` **B6.** *(mới, phát sinh)* Chỗ nạp model lượt 2 trên sidebar +
  đường truyền qua `PipelineBridge`. **Cố tình chưa làm**: chưa có model nào để
  kiểm chứng, dựng UI lúc này là dựng thứ không test được.

## Giai đoạn C — Model cho lượt 2

Khảo sát và notebook đã xong. Việc còn lại là **gán nhãn**, không code được.

- `[x] HOÀN THÀNH` **C0a.** Khảo sát → `Docs/dataset_lead_detection.md`.
  **Kết luận: không nguồn công khai nào đủ dùng.** `pads`/`pins` trong
  Consolidated chỉ 186/261 instance trên ~30 ảnh (đọc từ manifest ver2);
  SolDef_AI đúng loại nhãn nhưng sai tỉ lệ 20 lần (đo: 0 box). Ứng viên đáng
  xem nhất là **PCB-SAID** (ICCVW 2025) — nhưng openaccess trả 403 khi fetch,
  bạn phải tự mở bài báo lấy link và giấy phép.
- `[x] HOÀN THÀNH` **C0b.** Chọn `yolo11s` detection, `imgsz=640`. Lý do: crop
  chỉ có một linh kiện nên bài dễ hơn detect trên board rộng, mà lượt 2 chạy
  **một lần cho mỗi linh kiện** (~1000 forward pass mỗi board) nên tốc độ quan
  trọng hơn sức mạnh. Không cần segmentation: lượt 2 trả lời "chân ở đâu".
- `[x] HOÀN THÀNH` **C0c.** `training/kaggle/pcb_lead_detector_kaggle.py` +
  `.ipynb` (28 cell, 14 code, mọi cell parse OK).
  Nguyên tắc số một của notebook: **train trên crop, vì lúc chạy nhìn crop** —
  không lặp lại sai lầm train-trên-board-chạy-trên-crop.
  Có cổng chặn nhãn giả chưa sửa, chia tập **theo board**, cell vẽ ngược nhãn
  lên crop để bắt lỗi toạ độ trước khi train, và cổng phán quyết cuối
  (recall ≥ 0.70) vì hình học không bao giờ bỏ sót chân — model recall thấp hơn
  thì tệ hơn thứ nó thay thế.
- `[x] HOÀN THÀNH` **C0d.** *(phát sinh)* `bootstrap_lead_labels.py` nay xuất
  thêm `components/<stem>.json` — box linh kiện của lượt 1. Không có nó thì
  notebook không cắt được crop, mà detect lại lúc train sẽ dùng box khác với box
  đã sinh ra nhãn. Đã chạy thật trên board: 38 linh kiện, 79 box chân.
- `[x] HOÀN THÀNH` **C0e.** *(phát sinh)* Test chống lệch giữa notebook và thư
  viện: notebook mang bản chép `component_crop_window` vì Kaggle không có repo,
  nên test so hai bản trên 6 hình dạng box. **Đã bắt được lệch thật** — bản chép
  clamp thiếu một chiều, đã sửa.

**Số đo trên board thật, chạy hết chuỗi bootstrap → crop:**

| Chỉ số | Đo được |
|---|---|
| Linh kiện → crop | 38 → 38 (0 trống, 0 quá nhỏ) |
| Kích thước crop | trung vị 62 × 58 px |
| **Pad, cạnh ngắn** | **trung vị 23 px**, phân vị 10 là 19 px |
| Pad dưới 8 px | 2/90 (2.2%) |

Pad 23 px là **học được**. Ngưỡng cảnh báo của notebook (quá nửa dưới 8 px) còn
rất xa. Đây là dấu hiệu tốt nhất cho thấy lượt 2 khả thi ở độ phân giải hiện tại.

- `[ ] CHƯA LÀM` **C1.** Chụp/thu thập ảnh board thật của dây chuyền.
- `[ ] CHƯA LÀM` **C2.** Chạy `scripts/bootstrap_lead_labels.py` để xuất ROI suy
  ra thành dataset YOLO cho người sửa (sửa box nhanh hơn vẽ box).
- `[ ] CHƯA LÀM` **C3.** Sửa nhãn bằng LabelImg/CVAT/Roboflow. **Đây là phần tốn
  công nhất và không có cách nào bỏ qua.**
- `[ ] CHƯA LÀM` **C4.** Train bằng `training/kaggle/pcb_lead_detector_kaggle.ipynb`
  (đã soạn ở C0c). Notebook `soldef-ai.ipynb` giữ làm tham khảo khung xử lý.
- `[ ] CHƯA LÀM` **C5.** Export ONNX + `model_manifest.json`, nạp qua sidebar.

## Giai đoạn D — Phần cứng

- `[x] HOÀN THÀNH` **D1.** Soạn yêu cầu tối thiểu về camera/ống kính/ánh sáng.
  → `Docs/yeu_cau_phan_cung_camera.md`. Chốt: hiện 46 µm/px (3 phép đo độc lập,
  lệch <3%); khuyến nghị cảm biến 20 MP + FOV 137×91 mm → 25 µm/px, 4 lần chụp
  mỗi board. Nếu chỉ đủ tiền một hạng mục thì **mua đèn nhiều góc trước, không
  phải cảm biến**.
- `[x] HOÀN THÀNH` **D3.** *(phát sinh)* Đánh giá camera có sẵn Hikvision
  DS-2CD5026WZ-YD 11–40 mm → mục 6 của `Docs/yeu_cau_phan_cung_camera.md`.
  Tóm tắt: 2 MP, pixel pitch 3.74 µm. **Quang học đủ** (40 mm @ 30 cm →
  24 µm/px) nếu ống kính lấy nét được ở khoảng đó — chưa biết MOD, phải đo.
  **Nút chặn là nén JPEG**: nhiễu Δspecular 0.038 so với ngưỡng
  `cold_specular_ratio` 0.010 — **gấp 3.8 lần**, ở mọi mức chất lượng. Dùng để
  thu ảnh gán nhãn cho giai đoạn C thì được; chấm hàn nguội thì không.
- `[ ] CHƯA LÀM` **D2.** Quyết định mua/thuê thiết bị — việc của người, không
  phải của code.

---

## Về notebook `soldef-ai.ipynb` đã tải về

**Dùng lại được, nhưng phải đổi dữ liệu chứ không phải đổi model.**

Notebook đó là YOLO11m-seg train trên SolDef_AI, kết quả val Box mAP50 **0.771**,
Mask mAP50 0.766. Chạy lên board của dự án thì ra **0 box** ở mọi mức phóng đại
từ 1× đến 12×, vì SolDef_AI là ảnh macro 1–3 µm/px còn board của ta là 46 µm/px
— chênh khoảng 20 lần. Đổi sang model "chuẩn hơn" (yolo11l, RT-DETR…) **không
sửa được chuyện này**: nút thắt là miền dữ liệu, không phải sức mạnh model.

Cái đáng giữ ở notebook là **khung xử lý**: nhận diện format annotation theo nội
dung, parser LabelMe/COCO/VOC/**YOLO**/mask, split stratified theo lớp hiếm,
kiểm tra integrity `data.yaml`, đường cong loss, export ONNX/TorchScript/OpenVINO
có try/except riêng từng format.

Đã kiểm chứng: notebook **tự nhận dạng format và có sẵn parser YOLO**, nên
output của `bootstrap_lead_labels.py` (định dạng YOLO) nạp thẳng vào được, chỉ
cần trỏ `DATA_ROOT_OVERRIDE` và đặt `FORMAT_OVERRIDE = "yolo"`.

**Việc cần làm với notebook, theo thứ tự:**

1. Chạy lại Cell 3 (import) rồi Cell 22–28. Cell 22 đang lỗi `NameError: YOLO`
   do kernel restart giữa chừng — không phải lỗi code. Kéo theo chưa có số test,
   chưa có confusion matrix, **chưa có ONNX**.
2. Đổi `DATA_ROOT_OVERRIDE` sang dataset chân hàn của chính dây chuyền.
3. Đổi bộ lớp: SolDef_AI dùng `good/no_good/exc_solder/poor_solder/spike`; lượt
   2 cần lớp **vị trí chân** (`pads`, `pins`) chứ không phải lớp lỗi.
4. Cân nhắc bỏ segmentation, chỉ cần detection: lượt 2 trả lời "chân ở đâu", còn
   "chân tốt hay xấu" là việc của bước 6.2.

Gợi ý model nền: `yolo11s`/`yolo11m` detection ở `imgsz=640` cho crop. Không cần
model lớn hơn — crop chỉ chứa một linh kiện, bài toán dễ hơn ảnh board rộng
nhiều. Dữ liệu mới là thứ quyết định.

---

## Đợt việc 2026-08-22

Danh sách người dùng giao. Ghi ra trước khi làm, tích ngay khi xong.

### E — Lưu dữ liệu lỗi (làm TRƯỚC lượt 2, theo yêu cầu)

- `[x] HOÀN THÀNH` **E1.** Đo trước rồi mới chốt. Số thật trên board của dự án:

  | | mỗi board (~960 linh kiện) | mỗi năm (100 board/ngày) |
  |---|---|---|
  | Lưu ảnh từng lỗi (PNG) | 7.1 MB | ~260 GB |
  | Lưu toạ độ + số đo (JSON) | 1.39 MB | ~50 GB |

  Tỉ lệ chỉ **5 lần**, không phải hàng trăm — vì chỉ lưu phần LỖI, và crop nhỏ
  nén PNG rất tốt. Nên lý do chọn toạ độ không hẳn là dung lượng, mà là **số thì
  tra cứu/thống kê/so sánh được, còn ảnh thì không**.

  Chi phí đổi lại, cũng đo được:

  ```
  dựng lại khung ảnh từ file gốc : 219 ms   (một lần cho cả board)
  cắt một ROI từ khung đã có     : 0.004 ms
  ```

  → Giữ **đúng một khung** trong RAM cho mỗi board đang xem. Công nhân mở một
  board rồi lật qua các lỗi của nó, nên 219 ms trả một lần, mọi crop sau đó gần
  như miễn phí. Dựng lại cho từng ROI sẽ đắt gấp **~59.000 lần**.
- `[x] HOÀN THÀNH` **E2.** `aoi_pipeline/evidence.py`: `EvidenceBundle` lưu
  toạ độ + số đo + **vân tay nguồn** (đường dẫn, SHA-256, kích thước khung, cấu
  hình tiền xử lý). `EvidenceViewer` giữ tối đa một khung, `release()` để trả
  RAM.

  Phần quan trọng hơn tiết kiệm: **khung dựng lại phải đúng là khung cũ.** Toạ
  độ ghi theo một công thức tiền xử lý sẽ trỏ vào pixel khác dưới công thức
  khác, và cho công nhân xem nhầm pixel còn tệ hơn không cho xem gì. Nên viewer
  **từ chối** khi digest lệch, khi kích thước khung lệch, hoặc khi mất ảnh gốc —
  báo lỗi rõ ràng thay vì cắt bừa.
- `[x] HOÀN THÀNH` **E3.** 10 test, phần lớn là test các ca **từ chối**: file
  nguồn đổi nội dung, tiền xử lý đổi làm khung khác kích thước, mất ảnh gốc,
  schema cũ.

### F — Nhập BOM

- `[ ] CHƯA LÀM` **F1.** Đọc BOM (vị trí, toạ độ, kích thước từng linh kiện).
- `[ ] CHƯA LÀM` **F2.** Đối chiếu detect ↔ BOM. **Linh kiện detect được ở
  toạ độ mà BOM không có cũng là một lỗi** — không chỉ thiếu linh kiện.
- `[ ] CHƯA LÀM` **F3.** UI nạp BOM + hiển thị đối chiếu.

### G — Thư mục model

- `[ ] CHƯA LÀM` **G1.** Tách thư mục model dùng chính, copy model hiện dùng
  sang, giữ riêng một chỗ cho model người dùng tự lưu.
- `[ ] CHƯA LÀM` **G2.** App nạp từ thư mục đó thay vì bắt upload mỗi lần.

### H — UI

- `[ ] CHƯA LÀM` **H1.** Font Montserrat.
- `[ ] CHƯA LÀM` **H2.** Thao tác dễ hơn.

### I — Model 6.2 hiện tại có dùng được không

- `[ ] CHƯA LÀM` **I1.** Kiểm bằng test, kết luận rõ ràng, nêu hướng khắc phục
  nếu không đạt.

### J — Lượt 2 cho chân mối hàn

- `[ ] CHƯA LÀM` **J1.** Agent con khảo sát dataset + chọn model.
- `[ ] CHƯA LÀM` **J2.** Hoàn thiện đường ống theo kết quả khảo sát.

---

## Nhật ký

| Ngày | Việc | Kết quả |
|---|---|---|
| 2026-08-21 | Lập kế hoạch, đo số nền | Ghi ở mục "Số đo nền" |
| 2026-08-21 | D1 — yêu cầu phần cứng | `Docs/yeu_cau_phan_cung_camera.md` |
| 2026-08-21 | Giai đoạn A xong (A1–A5) | 11.70 MB → 0.00 MB mỗi tile; 419/419 test pass |
| 2026-08-21 | Giai đoạn B xong (B1–B5) | `lead_detection.py` + 15 test; 434/434 test pass |
| 2026-08-21 | C0a–C0e: khảo sát + notebook | `Docs/dataset_lead_detection.md`, notebook lượt 2; 443/443 test pass |
| 2026-08-21 | D3 — đánh giá camera Hikvision sẵn có | Quang học đủ, nén JPEG chặn phần hàn nguội |
| 2026-08-21 | Cho `refine_to_metal` quan sát được từ app | Nút bật/tắt + cột `refined`/`shrink_pct`; đo trên board: siết 78/81 ROI, trung vị 16.1% |
