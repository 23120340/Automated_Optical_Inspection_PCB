# Tiến độ — Detect mối hàn theo 2 lượt

> File này là **bảng công việc sống**. Quy ước bắt buộc cho mọi agent/người làm:
> ghi hạng mục ra **trước khi** bắt tay, đổi trạng thái sang `ĐANG LÀM` khi bắt
> đầu, và tích `HOÀN THÀNH` **ngay khi xong** kèm bằng chứng đã đo.
>
> Cập nhật lần cuối: 2026-08-30

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
| Độ phân giải thật của ảnh | **~46 µm/px** | 3 phép đo độc lập, xem `Docs/thiet_ke/yeu_cau_phan_cung_camera.md` |
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
- `[x] HOÀN THÀNH` **B6.** *(28/08)* Ô nạp model lượt 2 trên sidebar +
  đường truyền qua `PipelineBridge`. Trước đó cố tình chưa làm vì chưa có model;
  giờ có model nên dựng được thứ kiểm chứng được. `app/streamlit_app.py` khối
  "Lượt 2 · Detect chân trong crop linh kiện" + `_render_model_picker(
  "lead_detector")`.
  **Một tính chất không giống bất kỳ ô model nào khác: nó không bao giờ tự nạp**
  (`_NO_AUTO_ADOPT`). Con số 26/28 trong `1447ed5` là của bản locator **cũ**;
  bản đã promote ở `0b8c34d` phủ **28/28 pad, 0 ca lọt lưới**. Cái vẫn chưa an
  toàn là **độ ôm** chứ không phải độ phủ: mức phủ trung vị tụt 0.97 → **0.79**
  và pad yếu nhất còn **0.52** so với cổng 0.50 — sát mép, 4 pad tệ đi
  (R254 1.00→0.52, U201 pad0/pad6 0.97→0.52, C220 1.00→0.57) đổi lấy 3 pad tốt
  lên. Đó là lý do nó opt-in: bảng chọn chỉ hiện mAP50 0.9912, và riêng con số
  đó là một lời mời. Sidebar in phần đo trên board ra trước.
  Kèm theo: sửa `KeyError` làm app chết **đúng lúc bấm vào chính ô này** —
  `_use_model_entry` ghi `config["lead_detection"]` trong khi `_default_config`
  không tạo khoá đó.

## Giai đoạn C — Model cho lượt 2

> **Cập nhật 28–30/08 — hai câu dưới đây đã sai, giữ lại để thấy vì sao.**
> Câu "việc còn lại là gán nhãn, không code được" hoá ra ngược: gán nhãn được
> là **nhờ** viết công cụ. Riêng `22a7e5d` thêm 5 script và một app HTML gán
> nhãn chạy offline; `c0ccfd3` thêm packer và notebook. Không có chúng thì
> 9.089 box đã không tồn tại.

Khảo sát và notebook đã xong. Việc còn lại là **gán nhãn**, không code được.

- `[x] HOÀN THÀNH` **C0a.** Khảo sát → `Docs/khao_sat/dataset_lead_detection.md`.
  ~~**Kết luận: không nguồn công khai nào đủ dùng.**~~ **Kết luận này đã bị bác
  bỏ ngày 28/08** (`07bbe99`, banner ở đầu `Docs/khao_sat/dataset_lead_detection.md`).
  Khảo sát chỉ xét hai đường — tìm dataset *đã có nhãn chân*, hoặc tự chụp — và
  bỏ sót đường thứ ba đã thành công: **lấy ảnh board công khai ĐÚNG TỈ LỆ PIXEL
  rồi tự gán nhãn**. Phần vật lý của khảo sát vẫn đúng (lọc cạnh ngắn 48 px:
  RF100 còn 12,1%, Winnies 20,9%); phần suy ra từ nó thì sai.
  Nguyên văn kết luận cũ: `pads`/`pins` trong
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

- `[x] HOÀN THÀNH` **C1a.** *(28/08)* Thu ảnh board **công khai** đúng tỉ lệ
  pixel: 235 ảnh toàn board (`2df01b3`) cắt thành 310 tile mức linh kiện
  (`171bd97`). Đây là thứ đã mở khoá cả giai đoạn C.
- `[!] CHẶN` **C1b.** Ảnh từ **chính dây chuyền** vẫn chưa có (repo mới có 3 ảnh
  điện thoại ở `real_pcb/`). Chặn cùng lý do với D2. Manifest của model lượt 2
  tự khai `bootstrap_only`: camera/ánh sáng/lớp mạ của dây chuyền không nằm
  trong tập train, nên mọi con số dưới đây là số của miền công khai.
- `[x] HOÀN THÀNH` **C2.** Không dùng `bootstrap_lead_labels.py` như kế hoạch.
  Đường thật sự chạy: `crop_components_for_labelling.py` →
  `prelabel_joint_boxes.py` → `build_joint_box_app.py` (`22a7e5d`). Lý do đổi:
  bootstrap xuất ROI **suy ra** nên người duyệt sửa lại đúng công thức hình học
  đã sinh ra nó; cắt crop từ ảnh công khai rồi vẽ mới thì nhãn độc lập với hình
  học.
- `[x] HOÀN THÀNH` **C3.** Gán nhãn **hai lượt**, bằng app HTML trong repo chứ
  không phải LabelImg/CVAT/Roboflow. Kết quả ở `datasets/train/solder_joint_v2`:
  **2.031 crop / 9.089 box** (train 1.498/6.625, valid 246/1.042, test 287/1.422).
  Lượt 2 là lượt **gán nhãn lại** (`105b10f`): box lượt 1 chỉ ôm phần thiếc, box
  lượt 2 phủ cả fillet — diện tích box trung vị 581 → 848 px², gấp 1,39 lần.
- `[x] HOÀN THÀNH` **C4.** Train bằng `training/kaggle/pcb_joint_locator_kaggle.ipynb`
  (không phải `pcb_lead_detector_kaggle.ipynb` của C0c). Đo trên test:
  mAP50 **0.9912**, mAP50-95 **0.5598**, P/R **0.973/0.977**, khoảng cách
  train–test **0.0018** ⇒ không học thuộc board.
- `[x] HOÀN THÀNH` **C5.** ONNX + `model_manifest.json` ở
  `models/active/lead_detector/`, nạp qua sidebar (xem B6). **Mặc định TẮT.**

## Giai đoạn D — Phần cứng

- `[x] HOÀN THÀNH` **D1.** Soạn yêu cầu tối thiểu về camera/ống kính/ánh sáng.
  → `Docs/thiet_ke/yeu_cau_phan_cung_camera.md`. Chốt: hiện 46 µm/px (3 phép đo độc lập,
  lệch <3%); khuyến nghị cảm biến 20 MP + FOV 137×91 mm → 25 µm/px, 4 lần chụp
  mỗi board. Nếu chỉ đủ tiền một hạng mục thì **mua đèn nhiều góc trước, không
  phải cảm biến**.
- `[x] HOÀN THÀNH` **D3.** *(phát sinh)* Đánh giá camera có sẵn Hikvision
  DS-2CD5026WZ-YD 11–40 mm → mục 6 của `Docs/thiet_ke/yeu_cau_phan_cung_camera.md`.
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
- `[x] HOÀN THÀNH` **E2.** `aoi_pipeline/reporting/evidence.py`: `EvidenceBundle` lưu
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

- `[x] ĐÃ HOÀN THÀNH` **F1.** Đọc BOM — `aoi_pipeline/placement/bom.py`. Nhận cả hai
  dạng đều bị gọi là "BOM": một dòng mỗi linh kiện kèm toạ độ/kích thước, và
  một dòng mỗi **loại** với danh sách designator `"R1, R2, R5"` kèm Quantity.
  Dòng gộp được tách ra thành từng linh kiện. Quantity lệch với chính danh
  sách của nó thì báo cảnh báo và **tin theo designator** — chúng gọi tên vị
  trí thật. `CadComponent` được bổ sung `width`/`height` (trước chỉ `CadPad`
  có kích thước).
- `[x] ĐÃ HOÀN THÀNH` **F2.** Đối chiếu detect ↔ BOM. **Linh kiện detect được ở
  toạ độ mà BOM không có cũng là một lỗi** — không chỉ thiếu linh kiện.
- `[x] ĐÃ HOÀN THÀNH` **F3.** UI nạp BOM ở sidebar + tab "Đối chiếu BOM" ở
  bước 4, kèm 4 ô số (BOM / khớp / thiếu / không có trong BOM) và bảng phát
  hiện xếp **lỗi lên đầu** — người vận hành đọc từ trên xuống, lỗi nằm dưới ba
  dòng ghi nhận là lỗi bị bỏ qua. 8 test giao diện + 19 test logic.

**Ranh giới quan trọng nhất của mục này:** BOM và CAD trả lời hai câu hỏi khác
nhau. File CAD nói *land nằm ở đâu* và thường thiếu (thermal pad, shield, land
cơ khí), nên `cad_fusion` coi detection không có trong CAD là **ghi nhận**.
File BOM nói *board phải có những linh kiện nào* — nó đầy đủ theo định nghĩa,
nên detection không có trong BOM là **LỖI**. `BillOfMaterials.complete` mang
ranh giới này và **từ chối đoán** nó: đọc từ file thì mặc định đủ, nguồn một
phần phải tự khai.

Khi BOM không có toạ độ, đối chiếu chuyển sang **đếm theo lớp** — yếu hơn và
thành thật về điều đó: nói được board thiếu một con trở, không nói được thiếu
con nào. Vẫn bắt được thiếu/thừa, tức phần lớn công dụng của BOM.

### G — Thư mục model

- `[x] ĐÃ HOÀN THÀNH` **G1.** Tách thư mục model dùng chính, copy model hiện dùng
  sang, giữ riêng một chỗ cho model người dùng tự lưu.
- `[x] ĐÃ HOÀN THÀNH` **G2.** App nạp từ thư mục đó thay vì bắt upload mỗi lần.

`models/` chia theo **ai sở hữu file**: `active/` là model dự án đang dùng
(commit kèm, clone về là chạy được), `archive/` là bản cũ giữ để đối chiếu và
**không bao giờ tự nạp**, `library/` là chỗ bạn bỏ model riêng vào (gitignore,
không đánh nhau khi pull). `aoi_pipeline/modelops/model_registry.py` chỉ liệt kê `.onnx`
**có manifest bên cạnh** — thiếu manifest thì 6.1/6.2 từ chối nạp, nên chào mời
một file không manifest chỉ dời thất bại sang một cú click sau. `.pt` mang
pickle nên cũng không liệt kê.

### H — UI

- `[x] ĐÃ HOÀN THÀNH` **H1.** Font Montserrat, **tự phục vụ từ repo**. App này
  chạy ở xưởng, nơi một request tới `fonts.googleapis.com` nghĩa là trang hiện
  bằng font dự phòng, hoặc treo chờ timeout. File `.woff2` nằm trong
  `app/static/fonts/`, khai báo qua `[[theme.fontFaces]]` của Streamlit 1.61 và
  bật `server.enableStaticServing`. Chỉ lấy hai subset `vietnamese` + `latin`
  (228 KB cho 4 weight); các dải Cyrillic/Greek chiếm phần lớn 8 file của Google
  và không có chữ nào dùng đến. `tests/test_theme_font.py` kiểm **cả file thật**,
  vì chỗ dễ hỏng nhất là file font biến mất — khi đó config vẫn đúng, test cũ vẫn
  xanh, và app im lặng rơi về Arial.
- `[x] ĐÃ HOÀN THÀNH` **H2.** Sidebar trước đây vẽ danh sách 8 bước **hai lần**
  — một khối HTML có trạng thái nhưng không bấm được, và một radio nhạt bên
  dưới để điều hướng; người dùng phải đọc ở bảng trên rồi tìm lại đúng dòng đó
  ở bảng dưới. Nay mỗi bước một dòng, bấm thẳng, dấu trạng thái đứng trước tên.
  Mô tả ngắn chuyển vào tooltip. Gỡ 12 khối CSS chết.

### I — Model 6.2 hiện tại có dùng được không

- `[x] ĐÃ HOÀN THÀNH` **I1.** Kiểm bằng test, kết luận rõ ràng, nêu hướng khắc phục
  nếu không đạt.

**Kết luận: chưa dùng để ra quyết định được.** Đầy đủ ở
`Docs/danh_gia/danh_gia_model_6_2.md`, chốt lại bằng `tests/inspection/test_solder_model_assessment.py`.

Ba bằng chứng, theo thứ tự nặng dần:

| Nguồn | Số đo |
|---|---|
| Chính tập val của nó (manifest) | **46% mối hàn phải xem tay** ở ngưỡng 0.85 đang chạy; false_call 0.228 |
| 664 ROI mối hàn thật (5 ảnh, đo lại 2026-08-23) | **50.2% bị gọi `bridge`**; chỉ **4.4%** vượt ngưỡng chấp nhận |
| Cho ăn thứ không có mối hàn nào | nhiễu ngẫu nhiên → `bridge` **70.0%**; mảnh board bất kỳ → **68.3%** |

Đọc cho đúng: model **không** hoàn toàn suy biến — chi-square 32.91 (dof 6,
ngưỡng 12.59) bác được giả thuyết "hai phân bố là một", Mann-Whitney z = +2.71.
Nhưng **81,6% phân bố đầu ra của nó giống hệt nhau** dù đưa mối hàn thật hay
một mảnh board bất kỳ. Tín hiệu có thật, quá yếu để dựa vào.

Nguyên nhân đọc thẳng từ manifest: mất cân bằng lớp **10,6×**; ba lớp (`cold`,
`insufficient`, `shift_component`) học từ **một** nguồn duy nhất; và một trong
ba nguồn train là **SolDef_AI**, đã đo là macro 1–3 µm/px so với 46 µm/px của
board dự án.

**Hướng khắc phục — điều quan trọng nhất là KHÔNG train lại bằng ba nguồn cũ.**
Vấn đề không nằm ở kiến trúc, số epoch hay ngưỡng mà ở dữ liệu sai tỉ lệ chụp.
Nên gán nhãn tốt/xấu **chung một lượt** với việc khoanh box cho lượt 2 (C3), và
thu từ 7 lớp xuống nhị phân `good`/`not_good` trước khi mở rộng.

Trong lúc chờ, ba chốt mặc định đang giữ model đúng chỗ nó xứng đáng
(`model_accept_probability` 0.80, `escape_guard_enabled`,
`disagreement_is_review`) — hệ quả là gần như mọi mối hàn vào hàng chờ xem tay:
an toàn, không có giá trị tự động hoá, và **không đẩy board tốt thành phế phẩm**.

### J — Lượt 2 cho chân mối hàn

- `[x] ĐÃ HOÀN THÀNH` **J1.** Agent con khảo sát dataset + chọn model. Kết quả
  và **các chỗ tôi đo lại khác với đề xuất của nó** ghi ở
  `Docs/khao_sat/dataset_lead_detection.md`, mục "Cập nhật 2026-08-22".
- `[x] HOÀN THÀNH (bootstrap)` **J2.** Đường ống chạy, dữ liệu có, model có.
  **Nút chặn đã dời chỗ**, không còn là "chờ dữ liệu gán nhãn".
- `[!] CHẶN` **J3.** *(mới)* Model lượt 2 vẫn opt-in vì ba lý do đã đo, ghi
  trong `models/active/lead_detector/model_manifest.json`:
  (a) phủ pad trung vị **0.97 → 0.79** khi bật model, pad yếu nhất 0.52 so với
  cổng 0.50 — sát mép;
  (b) `bootstrap_only`: train toàn ảnh công khai, chưa có ảnh dây chuyền (C1b);
  (c) **chưa quyết được** box ôm sát thiếc hay box rộng trùm cả land thì bước
  6.2 chấm chuẩn hơn — muốn quyết phải có ground truth lỗi mà board này không
  có. Xem trường `what_is_NOT_settled` trong manifest.
  Ngoài ra `49b26fb` đo được: ROI kỳ quặc trên IC là lỗi **lượt 1**, không phải
  lượt 2 — box lớn nhất detector cho ra 251×250 px trong khi QFP thật ~350 px.
- `[x] ĐÃ SỬA 2026-09-04` **J4.** Lỗi lượt 1 ở trên đã hết. Detector thân
  linh kiện 1 lớp (`0477cd3`) đo trên 640 box tay của tập test, tách dải cỡ:

  | dải cạnh dài | detector cũ 22 lớp | detector mới 1 lớp |
  |---|---:|---:|
  | <32 px | 4,2% | **93,3%** |
  | 32–96 px | 47,0% | 84,1% |
  | 96–250 px | 30,8% | 76,9% |
  | **≥250 px** *(dải của QFP)* | **4,9%** | **92,7%** |

  Box thừa không khớp GT nào: cũ 28/129 (21,7%), mới 160/776 (20,6%) — tỉ lệ
  tương đương, và chỉ **1** box thừa thuộc dạng dài-mảnh 'lược chân'.

- `[!] MỚI 2026-09-04` **J5. Lượt 2 không còn là tuỳ chọn.** Đo trên bo
  `tests/data/solder_geometry` (28 pad đếm tay, khung sau bước 1, có nạp 6.1):

  | | chỉ lượt 1 | lượt 1 + lượt 2 |
  |---|---:|---:|
  | detector cũ 22 lớp | 28/28 | 28/28 |
  | detector mới 1 lớp | **11/28** | **27/28** |

  Nguyên nhân đo được: trên bo NÀY box của detector mới lớn hơn hẳn — cạnh
  dài trung vị 76 px so với 35 px, diện tích 2701 so với 718 px². Hình học
  suy ra ở 5.5 đặt ROI theo *tỉ lệ* của box, nên box đổi thang thì ROI đặt
  sai chỗ. Lượt 2 miễn nhiễm vì nó tìm mối hàn từ pixel, không từ tỉ lệ box.

  ⚠️ Bo này là **ngoài miền** ("public SMD board crop", không thuộc 28 bo
  PCB-DSLR), đúng giới hạn manifest tự khai. Trên tile trong miền, box của
  detector mới khớp nhãn tay ở IoU 0.5 với recall trong bảng J4. Nên con số
  11/28 đo **hành vi ngoài miền**, không phải hành vi chung.

  Cách chữa, chưa làm: hiệu chuẩn lại `PadProfile` theo quy ước box mới
  ("chỉ thân, loại chân/pad"), hoặc bật lượt 2 mặc định — nhưng lượt 2 còn
  `bootstrap_only` nên chưa bật production được.

- `[!] MỚI 2026-09-04` **J6.** Cùng lần đo: **15/62 thân** không nhận được
  nhãn họ từ 6.1 (`review`/`unknown`/`false_crop_background`) nên rơi về
  `multi_pin`. `run()` đã cảnh báo số này thay vì im lặng.

- `[x] ĐÃ SỬA 2026-09-04` **J7. Cỡ tile bám theo artifact.** ONNX khoá cứng
  shape chỉ nhận đúng một cỡ, nên tile lớn hơn bị letterbox thu nhỏ và linh
  kiện xuất hiện nhỏ hơn lúc train. Đo trên 640 box tay, artifact native
  1024 với tile 1280:

  | dải cạnh dài | tile 1024 | tile 1280 (0,8×) |
  |---|---:|---:|
  | **tổng** | **89,7%** | 83,3% |
  | <32 px | 93,3% | 87,8% |
  | 32–96 px | 84,1% | 76,5% |
  | 96–250 px | 76,9% | 72,3% |
  | **≥250 px** | **92,7%** | **78,0%** |

  Precision không khá hơn để bù (74,0% so với 74,6%) — mất trắng 6,4 điểm
  recall tổng. `detect_components` giờ lấy cỡ từ `detector.image_size` chứ
  không gán cứng, nên model 22 lớp cũ (native 1280) giữ nguyên hành vi.

- `[x] ĐÃ ĐO 2026-09-04` **J8. Điểm vận hành conf/iou đã đúng sẵn.** Quét
  trên cùng 640 box tay: F1 đạt đỉnh ở `conf=0.40`, nhưng F1 sai cho AOI —
  bỏ sót linh kiện là mối hàn không ai kiểm, dựng thừa ROI thì xem lại được.
  Tính bằng **F2** thì đỉnh rơi đúng vào `conf=0.25` đang chạy. **Không đổi.**
  Kết quả âm kèm theo: `iou` **không có tác dụng gì** với artifact YOLO26 —
  0.45 và 0.70 cho số y hệt ở cả bảy mức conf, vì kiến trúc đó NMS-free và
  top-k nằm trong graph.

- `[!] CHƯA ĐO` **J9. `include_full_image` với ONNX khoá cỡ.** Tiler chạy
  thêm một lượt trên NGUYÊN ảnh; với bo 3072 px và artifact 1024 thì lượt đó
  thu nhỏ 3×, gần như không còn linh kiện nào đủ lớn. Chưa đo được nó đóng
  góp hay chỉ tốn thời gian, nên chưa đụng.

- `[ ] CHỜ DỮ LIỆU` **J10. Fine-tune trên ảnh dây chuyền.** Cả hai lượt đều
  train trên ảnh công khai. Lượt 2 tự khai `bootstrap_only: true` và
  `safe_to_enable_in_production: false`; manifest lượt 1 khai *"phải fine-tune
  trên ảnh thật trước khi tin số đo ở production"*. Bằng chứng đo được cho
  thấy điều này không phải hình thức: trên bo NGOÀI MIỀN
  (`tests/data/solder_geometry`), box của lượt 1 phình gấp đôi (cạnh dài
  trung vị 76 px so với 35) và độ phủ pad chỉ-lượt-1 tụt còn 11/28. Trong
  miền thì recall theo dải ở J4 hoàn toàn khoẻ. **Không sửa được bằng code**
  — cần ảnh từ camera dây chuyền.

**Hai tham số trong kế hoạch ban đầu đều sai khi đem đo.**

`imgsz`: khảo sát đề nghị 128–160. Đo trên board thật (36 linh kiện, ảnh
1832×2560) thì cạnh dài của crop trải từ **25 px đến 462 px** — trung vị 48,
p90 **168**, p99 **428**. Chọn theo trung vị là sai: imgsz nhỏ **thu nhỏ** crop
lớn, mà crop lớn chính là IC và connector, những thứ **nhiều chân nhất**.

| imgsz | crop bị thu nhỏ | pad tụt dưới 8 px | ms/crop |
|---|---|---|---|
| 128 | 4/36 | **1** | — |
| 160 | 4/36 | **1** | — |
| **256** | **2/36** | **0** | **19.6** |
| 640 | 0/36 | 0 | 123.8 |

⇒ đổi notebook lượt 2 sang **`imgsz=256`** (nhanh hơn 640 khoảng **5,2 lần**,
không mất pad nào) và **`yolo11n`** thay vì `yolo11s` — với 10–20 board tự gán
nhãn bạn có ~10–20k crop nhưng chỉ **10–20 cảnh độc lập**, model to hơn sẽ học
thuộc danh tính board thay vì hình dạng pad.

> **Đính chính 28/08 — model đã ship là `yolo11s` ở `imgsz=640`.** Lời khuyên
> trên đúng với giả định "10–20 cảnh độc lập", và giả định đó không còn: dữ liệu
> thật có **165 cảnh** (train 115 / valid 25 / test 25). Nỗi lo học thuộc đã
> được đo chứ không phỏng đoán — khoảng cách train–test 0.0018,
> `scene_memorisation_warning` là `false`. Phần đo cỡ lô (lô 4 tối ưu trên CPU)
> không bị ảnh hưởng, vẫn giữ nguyên.

Gom lô: khảo sát đề nghị 64–256. Đo thì **lô 64 là cấu hình tệ nhất**:

| Cỡ lô | 1 | 2 | **4** | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| ms/crop | 28.9 | 22.6 | **19.6** | 23.5 | 28.6 | 31.1 | 33.2 |

Backend CPU vốn đã chia luồng **bên trong** một ảnh, nên lô lớn mất vào lưu
lượng bộ nhớ nhiều hơn phần thắng. Đã cài `LeadDetectionConfig.batch_size = 4`
và `detect_batch` **tuỳ chọn** trên protocol; detector không có nó vẫn chạy
từng crop. GPU sẽ đổi điểm tối ưu này — **phải đo lại** trước khi tăng.

Test bảo vệ **hợp đồng vị trí**: kết quả thứ *i* phải thuộc crop thứ *i*. Lệch
một bậc là gán chân sang nhầm linh kiện mà **không có gì báo lỗi**. Lô trả về
sai số lượng thì quay về gọi từng cái, chứ không ghép theo may rủi.

---

## Nhật ký

| Ngày | Việc | Kết quả |
|---|---|---|
| 2026-08-21 | Lập kế hoạch, đo số nền | Ghi ở mục "Số đo nền" |
| 2026-08-21 | D1 — yêu cầu phần cứng | `Docs/thiet_ke/yeu_cau_phan_cung_camera.md` |
| 2026-08-21 | Giai đoạn A xong (A1–A5) | 11.70 MB → 0.00 MB mỗi tile; 419/419 test pass |
| 2026-08-21 | Giai đoạn B xong (B1–B5) | `lead_detection.py` + 15 test; 434/434 test pass |
| 2026-08-21 | C0a–C0e: khảo sát + notebook | `Docs/khao_sat/dataset_lead_detection.md`, notebook lượt 2; 443/443 test pass |
| 2026-08-21 | D3 — đánh giá camera Hikvision sẵn có | Quang học đủ, nén JPEG chặn phần hàn nguội |
| 2026-08-21 | Cho `refine_to_metal` quan sát được từ app | Nút bật/tắt + cột `refined`/`shrink_pct`; đo trên board: siết 78/81 ROI, trung vị 16.1% |
| 2026-08-22 | G1–G2 — thư mục model + bộ chọn | `models/active|archive|library`, `model_registry.py`; 487 test pass, 0 skip |
| 2026-08-22 | J1 — khảo sát vòng 2 | PCB-SAID đã kiểm chứng là ngõ cụt; tìm được Ulger (đúng tỉ lệ, không có box) |
| 2026-08-22 | J — đo lại tham số lượt 2 | imgsz 640→256 (5,2× nhanh hơn), gom lô 4; 492 test pass |
| 2026-08-22 | H1–H2 — Montserrat + gộp stepper | Font tự phục vụ 228 KB, sidebar hết vẽ 8 bước hai lần; 504 test pass |
| 2026-08-22 | I1 — đánh giá model 6.2 | `Docs/danh_gia/danh_gia_model_6_2.md`: chưa dùng để quyết định được, 61% mối hàn bị gọi `bridge` |
| 2026-08-22 | F1–F3 — nhập BOM + đối chiếu | `aoi_pipeline/placement/bom.py` + UI; linh kiện ngoài BOM = LỖI; 531 test pass |
| 2026-08-23 | Đánh giá model trong app | `aoi_pipeline/modelops/model_feedback.py` + mục ở cuối trang bước 4/6.1/6.2; lưu toạ độ, gắn sha256 model; 605 test |
| 2026-09-03 | Lượt 1: promote detector thân linh kiện 1 lớp | `0477cd3`; test recall 0.844, cận dưới theo bo 0.744 > incumbent 0.54 |
| 2026-09-04 | Đo lại lượt 1 theo dải cỡ | J4: dải ≥250 px từ 4,9% lên 92,7% — lỗi QFP của `49b26fb` đã hết |
| 2026-09-04 | J7 — tile bám theo artifact | recall tổng 83,3% → 89,7%; dải ≥250px 78,0% → 92,7% |
| 2026-09-04 | J8 — quét conf/iou | conf 0.25 đã là đỉnh F2, giữ nguyên; `iou` vô tác dụng với YOLO26 |
| 2026-09-04 | Đo hai lượt chạy chung | J5: lượt 2 hết là tuỳ chọn với detector mới (11/28 → 27/28 trên bo ngoài miền) |
| 2026-08-23 | Gộp Golden vào đường ống | Bỏ workspace riêng; Golden = bước 3.5, ngay sau khoanh vùng board; 610 test |
| 2026-08-23 | Đo lại TOÀN BỘ model | `scripts/benchmark_models.py` + `Docs/bench/bench_20260823.json`; bảng xếp hạng dựng lại từ một lần chạy duy nhất |
| 2026-08-24 | Feedback bằng chuột | Bấm thẳng vào box sai / chỗ bỏ sót; ghi kèm `box_size` cho lượt train sau; 615 test |
| 2026-08-24 | Gộp hồ sơ board | Golden + BOM + CAD/pick-and-place vào một ô multiselect; lộ CAD ra UI lần đầu; 619 test |
| 2026-08-24 | Fiducial cho bước 3 | `aoi_pipeline/imaging/fiducials.py`; bấm tay là đường tin cậy, dò tự động chỉ đề xuất; 630 test |
| 2026-08-24 | Kế hoạch fine-tune tại chỗ | `Docs/ke_hoach/ke_hoach_fine_tune_cuc_bo.md`; đo trên máy này: khả thi ở imgsz 256, ~1–5 giờ |
| 2026-08-24 | Bản đồ kiểm tra + kế hoạch chụp | `aoi_pipeline/placement/inspection_map.py`; board 197×148 mm → 4 khung, 0 linh kiện lọt; 646 test |
| 2026-08-25 | Sửa luật hình học 6.2 + verify bộ reference | `041b99b` |
| 2026-08-27 | Bộ công cụ gán nhãn: 5 script + app HTML offline | `22a7e5d`; khảo sát nguồn công khai vào `datasets/public/README.md` |
| 2026-08-28 | Thu 235 ảnh toàn board, cắt 310 tile | `2df01b3`, `171bd97` |
| 2026-08-28 | Gán nhãn lại mối hàn để box phủ cả fillet | `105b10f`; diện tích box trung vị 581 → 848 px² |
| 2026-08-28 | Dataset `solder_joint_v2` | `c0ccfd3`; 2.031 crop / 9.089 box |
| 2026-08-28 | **Gỡ tầng detect lỗi hàn toàn board**, thay bằng ô model lượt 2 | `1447ed5`; xem B6 |
| 2026-08-28 | Model lượt 2 lên `models/active/lead_detector` | `0b8c34d`; test mAP50 0.9912, opt-in, không tự nạp |
| 2026-08-28 | Bộ gán nhãn thân linh kiện lượt 1 + bản nháp để SỬA thay vì VẼ | `49b26fb`, `956ddda` |
| 2026-08-30 | Khử trùng bộ gán nhãn → vòng 2 | 310 tile chỉ có 170 nhóm pixel độc nhất; `component_bodies_round2_20260830`: 120 tile, 120 hash khác nhau, 30 bo, giữ nguyên 16 tile đã duyệt / 1.595 box |
| 2026-08-30 | Packer dataset một lớp `component` | `scripts/pack_component_detection_dataset.py`; audit: 8/10 bo, thiếu bucket `valid` ⇒ **chưa pack được** |
| 2026-08-30 | Sửa hai chốt 6.2 im lặng bỏ qua từ 24/08 | `MANIFEST_PATH` trỏ vào thư mục model đã dời; 4+2 skip → 7 pass |
| 2026-08-30 | Chống mất việc đã duyệt khi nạp file vào app gán nhãn | nạp checkpoint mâu thuẫn nay huỷ toàn bộ import thay vì ghi đè im lặng |
| 2026-08-31 | Trang gán nhãn trên đĩa dựng từ template CŨ, thiếu chốt chống ghi đè | trang nằm trong .gitignore nên trôi lệch không ai thấy; đã dựng lại cả 4 bộ + thêm vân tay template để test bắt được |
| 2026-08-31 | Packer nêu TÊN bo cần duyệt khi một bucket trống | trước chỉ báo "thiếu bucket valid", mà bucket là hàm băm nên không ai đoán được |
| 2026-08-31 | Kế hoạch phân nhóm package | `Docs/ke_hoach/ke_hoach_phan_nhom_package.md`; đo được 13,5% linh kiện mang 31,2% mối hàn và đều chỉ có chân ở 2/4 cạnh |
| 2026-09-01 | Triển khai package 5.2 | Parser footprint ưu tiên BOM/PnP/CAD; 7 topology + cờ mismatch; classifier ONNX opt-in/no-op khi thiếu artifact; editor migration giữ nguyên xywh; packer chia theo board, notebook và gate 28 pad. **Còn thủ công:** giải quyết 3.847 `unknown`, train, chạy gate rồi mới promote. |
| 2026-08-31 | Kế hoạch lỗi toàn mạch | `Docs/ke_hoach/ke_hoach_pcb_defect_toan_mach.md` — **chờ duyệt**; VisA pcb1–4 là nguồn có xước trên board ĐÃ LẮP, script fetch đã có sẵn |
