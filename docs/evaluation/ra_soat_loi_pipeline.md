# Rà soát lỗi pipeline — 2026-08-27

Rà toàn bộ `aoi_pipeline/`, `app/`, `scripts/` (20.762 dòng). Mỗi phát hiện dưới
đây đều **đã được chứng minh bằng cách chạy code**, không suy luận từ đọc hiểu.

Cách rà: quét mẫu lỗi (nuốt exception, tham số mặc định khả biến, `is` với
literal, chia không bảo vệ, index rỗng, IO thiếu encoding), chạy pipeline
đầu-cuối trên board thật, rồi soi các hợp đồng giữa module.

## Những gì KHÔNG phải lỗi

Ghi lại để lần sau khỏi rà lại:

- **Tham số mặc định khả biến, `except: pass`, `is` với literal, file mở không
  đóng** — không có trường hợp nào.
- **IO thiếu `encoding=`** — 13 chỗ bị AST scan báo, tất cả đều là dương tính
  giả: đều mở nhị phân (`"rb"`/`"ab"`) hoặc `Image.open`. Scan đọc nhầm vị trí
  tham số `mode` khi gọi ở dạng method.
- **Pipeline trả cả `good` lẫn `ok`** — đúng thiết kế, không phải trùng nghĩa:
  `ok` thuộc `COMPONENT_CLASSES` (mức linh kiện), `good` thuộc `JOINT_CLASSES`
  (mức mối hàn). Chạy thật cho 39 body view + 90 joint = 129 ROI, khớp chính xác.
- **`_axis_starts` chia cho `nominal_stride`** — `_validate_policy` đã ép
  `0 <= overlap_ratio < 0.5` nên mẫu số luôn dương.
- **Khoá file trong `model_feedback.py`** — lập luận kỹ, có đo thực nghiệm trên
  Windows ghi trong docstring.

---

## Lỗi 1 — `exporters.py` xuất CSV không thoát công thức ⚠️ NGHIÊM TRỌNG NHẤT

**File:** `aoi_pipeline/reporting/exporters.py` — `solder_joints_csv`, `solder_verdicts_csv`,
`cad_findings_csv`

Ba hàm xuất của pipeline ghi thẳng `designator`, `net`, `label`,
`component_label`, `model_label`, `expected_class`, `observed_class`, `message`
vào CSV **không thoát ký tự công thức**.

`designator` và `net` **đến thẳng từ file người dùng nạp**: `load_pads_csv`,
`load_placement_csv`, `load_ipc356`, `load_cad_json` (`aoi_pipeline/solder/cad.py`)
đọc nguyên văn từ CSV/IPC-356 rồi đưa xuống. File CSV xuất ra là thứ **kỹ thuật
viên mở bằng Excel**.

Chứng minh (chạy thật):

```
d0_joint00,d0,capacitor,joint,terminal_a,,two_terminal,0.00,...,
  "=HYPERLINK(""http://evil"",""click"")",,+1+1,joints/x.png,

ô bắt đầu bằng ký tự công thức, KHÔNG được thoát:
  ['=HYPERLINK("http://evil","click")', '+1+1']
```

Nghịch lý: **đã có ba bản sao hàm thoát** trong repo (xem Lỗi 2) nhưng đúng
module chuyên trách xuất file thì không dùng bản nào.

**Sửa:** đưa hàm thoát về `exporters.py` làm bản chuẩn, áp cho cả ba hàm xuất.

---

## Lỗi 2 — Hàm thoát CSV bị chép ba bản

**File:** `aoi_pipeline/placement/digitizer.py:834` (`_safe_csv_text`),
`app/streamlit_app.py:4043` (`_csv_cell`),
`scripts/build_reference_bundle.py:611` (`_safe_csv_text`)

Ba bản thân hàm giống hệt nhau từng ký tự. Test của chính dự án
(`tests/test_no_duplicate_helpers.py`) đang **fail** vì hai bản đầu — đây là lỗi
duy nhất còn đỏ trong suite.

Nguy hiểm không nằm ở chỗ lặp code mà ở chỗ **ba bản đang đồng ý với nhau**:
không có gì báo khi một bản được siết chặt còn hai bản kia thì không. Đúng lý do
docstring của test đó nêu ra.

Chỉ có bản UI được test (`test_csv_cells_neutralize_spreadsheet_formulas`).

**Sửa:** một bản chuẩn trong `exporters.py`, ba nơi kia import về. Cùng một sửa
với Lỗi 1.

---

## Lỗi 3 — Thư mục `solder/segmenter` chứa model DETECT

> **Kết cục (2026-09-03).** Sửa nửa vời một lần: thư mục đổi tên
> `segmenter` → `defect`, nhưng model vẫn nằm trong `active/` trong khi
> registry đã bỏ hẳn ô cho nó — tức một model không đường nào nạp lại
> nằm trong thư mục có nghĩa là *app tự nạp*. Đã chuyển sang
> `models/archive/solder-defect-detector-wholeboard-ver1/`, và
> `tests/test_models_layout.py` giờ canh để không tái diễn.

**File:** `models/active/solder/segmenter/`, `aoi_pipeline/modelops/model_registry.py:57`

Thư mục tên `segmenter`, nhưng manifest bên trong ghi:

```json
"task": "solder_defect_detection",
"required_ultralytics_task": "detect",
"model_family": "yolo11"
```

Toàn bộ từ vựng registry gọi ô này là `solder_segmenter`, và comment tại
`model_registry.py:72` còn lập luận *"It segments defects, so it is a segmenter"*
— câu đó **đúng với artifact cũ** (`yolov8m-seg`, 4 lớp `Dry_joint`/`Short_circuit`)
nhưng **sai với artifact hiện tại** (`yolo11s` detect, 2 lớp `Bad_podu`/`Bad_qiaojiao`).

Hậu quả nhìn thấy được — `scripts/list_models.py` in ra:

```
solder_segmenter   đang dùng  solder\segmenter                   yolo11s      2026-08-26
solder_segmenter   bản cũ     solder_segmenter_yolov8m_20260824  yolov8m-seg  2026-08-24
```

Hai dòng cùng nhãn, hai task khác nhau, xếp cạnh nhau — đúng chỗ dễ nhầm nhất.
Người dùng đã yêu cầu đổi tên ô này từ phiên trước.

**Sửa:** in cột `task` trong bảng liệt kê để phân biệt được hai hình thái, và
sửa lại comment sai. **Không** đổi định danh `solder_segmenter` — lý do ở phần
"Đã sửa" bên dưới.

---

## Lỗi 4 — Registry không đọc được điểm số của chính model đang chạy

**File:** `aoi_pipeline/modelops/model_registry.py:119` (`_HEADLINE_METRICS`)

Danh sách có `("reported_metrics", "map50_mask")` và
`("reported_metrics", "map50_box")` nhưng **không có `("reported_metrics", "map50")`**
— đúng khoá mà manifest detect 6.2 dùng.

Chứng minh:

```
manifest có: reported_metrics.map50 = 0.8561
_summarise(...).metric -> None
```

Nên `list_models.py` in `—` ở cột điểm cho model đang dùng, còn bản cũ kém hơn
thì lại hiện `mask mAP50 0.557`. Người chọn model nhìn vào bảng sẽ tưởng bản mới
chưa từng được đo.

**Sửa:** thêm `("reported_metrics", "map50")` vào cuối nhóm mAP50.

---

## Lỗi 5 — Bộ phân loại vai trò không đọc được tên thư mục của chính nó

**File:** `aoi_pipeline/modelops/model_registry.py:455` (`_TASK_TO_KIND`),
`:466` (`_solder_role_from_hint`)

Hai lỗ hổng cùng chỗ:

1. `_TASK_TO_KIND` **không có** `solder_defect_detection` — task mà notebook
   `training/kaggle/pcb_solder_detector_kaggle.py` của chính dự án sinh ra.
2. `_solder_role_from_hint("solder/segmenter")` trả **`None`**. Danh sách token
   nhận diện có `"segmentation"` nhưng **không có `"segmenter"`**, trong khi
   `STAGE_FOLDERS["solder_segmenter"] = "solder/segmenter"` — registry không
   đọc ngược được tên thư mục do chính nó đặt.

Chứng minh:

```
_TASK_TO_KIND có 'solder_defect_detection' ? False
_solder_role_from_hint('solder/segmenter')  -> None
_solder_role_from_hint('solder/detector')   -> solder_segmenter
```

Hiện chưa gây hỏng vì model active được ánh xạ thẳng qua `STAGE_FOLDERS`, nhưng
một bản copy thả vào `models/library/` sẽ không được phân loại — đúng luồng
"của bạn" mà `list_models.py` quảng cáo.

**Sửa:** thêm task còn thiếu vào `_TASK_TO_KIND`, thêm `"segmenter"` vào danh
sách token. `"defect"` thì **không** — lý do ở phần dưới.

---

# Đã sửa

| Lỗi | Sửa ở đâu |
|---|---|
| 1 · CSV không thoát công thức | `exporters.py`: thêm `csv_cell()` làm bản chuẩn, áp cho cả `solder_joints_csv`, `solder_verdicts_csv`, `cad_findings_csv` — **chỉ ô chữ, không đụng ô số** |
| 2 · Ba bản sao hàm thoát | Gỡ cả ba, `digitizer` / `streamlit_app` / `build_reference_bundle` import bản chuẩn |
| 3 · Không phân biệt được detect với segment | `list_models.py` thêm cột **task**; sửa comment sai trong registry |
| 4 · Không đọc được điểm | `_HEADLINE_METRICS` thêm `("reported_metrics", "map50")` |
| 5 · Bộ phân loại vai trò | `_TASK_TO_KIND` thêm `solder_defect_detection`; token thêm `"segmenter"` |

## Về Lỗi 3 — vì sao KHÔNG đổi tên `solder_segmenter`

Cân nhắc rồi bác. Định danh đó xuất hiện ở **63 chỗ trong `streamlit_app.py`,
15 chỗ trong `pipeline_bridge.py` và 28 chỗ trong test**, phần lớn là **khoá
`st.session_state`**. Đổi tên sẽ **xoá lựa chọn model người dùng đã lưu** mà
không được gì về chức năng.

Cái gây hại thật không phải cái tên, mà là bảng liệt kê **không cho biết đang
nạp hình thái nào** — trong khi đổi giữa detect và segment thì đổi luôn hành vi
bước 6.2. Nên sửa đúng chỗ đó:

```
bước               nguồn      thư mục                     task                                 kiến trúc     điểm
solder_segmenter   đang dùng  solder\segmenter            solder_defect_detection              yolo11s       mAP50 0.856
solder_segmenter   bản cũ     solder_segmenter_yolov8m..   solder_defect_instance_segmentation  yolov8m-seg   mask mAP50 0.557
```

Hai dòng giờ phân biệt được, và bản đang chạy đã hiện điểm.

Một cám dỗ đã bị bác và ghi lại thành test: thêm `"defect"` vào danh sách token
detector. `"solder_defect_classification"` chứa **cả** `"classification"` lẫn
`"defect"`, nên hai hint cùng đúng và resolver sẽ im lặng ở đúng vai trò nó
đang giải đúng.

## Test

Trước khi sửa: `899 passed, 1 failed` (lỗi đỏ chính là Lỗi 2).
Sau khi sửa: **`947 passed, 2 skipped, 0 failed`** — thêm 47 test mới:

- `tests/test_csv_formula_escaping.py` (32) — kiểm ở **văn bản CSV xuất ra**,
  không kiểm hàm rời; gồm cả một test bảo vệ chiều ngược lại: toạ độ âm
  (`-12.50`) **không được** thoát, nếu không số sẽ thành chuỗi.
- `tests/test_model_registry.py` (+15) — điểm số, task, phân loại vai trò.

Mọi sửa đổi đều được **mutation-test**: gỡ từng sửa ra thì đúng những test
tương ứng fail (3/3 cho registry), gắn lại thì pass.
