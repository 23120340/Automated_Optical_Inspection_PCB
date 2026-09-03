# Phân nhóm package cho linh kiện

Cập nhật 2026-09-03. Viết lại toàn bộ — bản trước là ba lượt sửa chồng lên
nhau nên không đọc được.

---

## 1. Một trang tóm tắt

**Mục tiêu.** Cho bước 5.5 biết linh kiện có mấy chân và chân nằm ở cạnh nào,
để nó thôi phải đoán bằng pixel.

**Hướng đã chốt.** Dùng **luật hình học**, không train model package. Luật
**nối sau** classifier 6.1: 6.1 cho *họ*, luật chỉ chia nhỏ *bên trong* một họ.

**Vị trí.** Bước **5.2**, giữa bước 5 (cắt crop) và 5.5.

**Đang ở đâu.**

| | trạng thái |
|---|---|
| Thứ tự pipeline (6.1 và lead detector chạy trước 5.2) | ✅ xong, có test canh |
| Bộ luật cho họ `ic` + 5 họ ánh xạ thẳng | ✅ xong, mặc định TẮT |
| Chia họ `capacitor` (trụ đứng ↔ chip) | ❌ **đặc trưng đề xuất đã đo và KHÔNG chạy** — §6.3 |
| Tập kiểm gán tay để đo tỉ lệ trúng | ❌ chưa có |
| Taxonomy đầy đủ family → package | ⏳ **chờ bạn duyệt** — §6 |

**Việc kế tiếp cần bạn:** duyệt taxonomy ở §6.5 dựa trên contact sheet đã sinh.

---

## 2. Bảy lớp package

Tiêu chí giữ một lớp: *người gán nhãn phân biệt được trong một giây* **và**
*lớp đó làm 5.5 hành xử khác đi*. Không thoả cả hai thì bỏ.

| # | Lớp | Nhìn thế nào | 5.5 làm gì khác |
|---|---|---|---|
| 1 | `hai_chan` | hộp nhỏ chữ nhật, hai đầu kim loại ở hai cạnh ngắn | 2 ROI ở hai đầu trục dài *(đường đang chạy, đã đúng)* |
| 2 | `tru_dung` | hình tròn nhìn từ trên, nắp nhôm có rãnh chữ thập | 2 ROI, nhưng **không** đoán trục bằng kim loại — vỏ can chính là kim loại |
| 3 | `goi_nho` | hộp đen nhỏ, vài chân to bản | dải chân trên 2 cạnh đối, ít chân, chân dày |
| 4 | `ic_hai_ben` | hai hàng chân mảnh ở hai cạnh đối | dải chân trên **đúng 2 cạnh**; 2 cạnh kia không dựng |
| 5 | `ic_bon_ben` | chân ra cả bốn phía | dải chân trên **cả 4 cạnh** |
| 6 | `ic_khong_chan` | mép nhẵn, không chân nào ló ra | **KHÔNG sinh ROI.** Đánh dấu "không kiểm được bằng ảnh trên xuống" |
| 7 | `connector` | dãy chân thẳng hàng, thân nhựa, hoặc chân xuyên lỗ | dải chân 1–2 hàng, bước chân lớn, ROI to hơn |

Bảy chuỗi này được ghi ra bởi app gán nhãn, manifest model và export runtime,
nên chúng là **giá trị ổn định**, không đổi tên tuỳ tiện.

---

## 3. Vì sao cần: 5.5 đang đoán bằng pixel

`aoi_pipeline/config.py` giữ toàn bộ tri thức hiện có về hình dạng chân trong
ba nhóm, mà nhóm thứ ba là cái sọt đựng phần còn lại:

```python
TWO_TERMINAL_CLASSES = {"capacitor", "resistor", "diode", "led", "inductor", "fuse"}
PAD_ONLY_CLASSES     = {"pads"}
DEFAULT_TERMINAL_GEOMETRY = "multi_pin"   # ic, connector, transistor, relay, ...
```

Đo trên bộ winnies (16.632 box, 24 kiểu vỏ):

| | box | % linh kiện | % mối hàn |
|---|---:|---:|---:|
| nhánh `two_terminal` | 14.379 | **86,5%** | 68,8% |
| nhánh `multi_pin` | 2.253 | **13,5%** | **31,2%** |

Và con số quyết định: **2.247/2.253 gói trong nhánh `multi_pin` chỉ có chân
trên đúng 2 cạnh**, nhưng `_multi_pin_rects` dựng dải quanh **cả 4 cạnh** rồi
mới lọc bớt bằng năng lượng pixel.

Tức 13,5% linh kiện mang 31,2% mối hàn, và với gần như tất cả chúng pipeline
đang dựng thừa hai dải trên nền trống rồi nhờ pixel nói hộ dải nào là thật.
**Tiền nằm ở nhóm 13,5% đó.**

---

## 4. Kiến trúc

```
5    make_crops()          → crop
6.1  classify_components() → HỌ (capacitor / ic / resistor / ...)
        │
5.2     ├─ họ = ic         → luật đọc vị trí chân → 2 bên | 4 bên | không chân
        ├─ họ = capacitor  → luật phân biệt trụ đứng | chip      (CHƯA CÓ)
        └─ họ khác         → gói mặc định của họ đó
        │
5.5  make_solder_crops()   → ROI mối hàn, dùng terminal_geometry(package)
```

**Vì sao khoá theo họ chứ không luật toàn cục.** Đo trên winnies: chip 2 chân
chiếm 86,5%, nên "luôn đoán chip" đã đúng 86,5%. Luật ngưỡng tốt nhất tìm được
trên **tỉ lệ cạnh** chỉ đạt **84,5% — tệ hơn baseline**; trên diện tích được
88,7%, tức +2,2 điểm. Luật hình học toàn cục trên box thân gần như vô dụng.
Biết trước họ là thứ xoá đi mất cân bằng 86:9:5 đó.

Đây cũng là lời giải thích bằng số cho luật cũ bị gỡ ở `73ce2aa`: nó dùng đúng
đặc trưng đo được là *dưới* baseline.

**Vì sao ở 5.2 chứ không sau 6.2.** Đặt sau 6.2 thì nhãn ra đời sau khi mọi ROI
đã dựng xong — chỉ còn giá trị báo cáo, không cải thiện được gì.

**Điểm nối trong code.** `terminal_geometry()` (`config.py:54`) nhận
`package: str | None` với thứ tự ưu tiên **footprint → package → họ detector**.
Nó không quan tâm chuỗi package đến từ model hay từ luật, nên không phải sửa
kiến trúc, chỉ thay nguồn.

---

## 5. Đã làm gì

- `aoi_pipeline/classification/package_rules.py` — bộ luật, nối vào pipeline
  qua `_append_package_rules`. **Nối sau, không thay**: thân nào đã có kết quả
  từ model thì luật không đụng. `PackageRulesConfig` **mặc định TẮT**.
  - 5 họ ánh xạ thẳng: `resistor`/`led`/`diode` → `hai_chan`,
    `discrete_semiconductor` → `goi_nho`, `connector` → `connector`.
  - Họ `ic` chia theo cạnh có dải chân, dùng `assign_leads_to_components()`.
  - Chốt an toàn: chỉ kết luận `ic_khong_chan` khi lead detector đã tìm được
    chân ở linh kiện **khác** trên cùng board. Gói này có `PadProfile(0, 0)`
    nên kết luận sai làm 5.5 bỏ hẳn ROI — mất mối hàn mà không báo.
  - 19 test, phần lớn canh những ca luật phải trả về **rỗng**.
- Thứ tự pipeline (`99b962f`): 6.1 chạy trước 5.2/5.5, lead detector chạy trước
  5.2. `tests/test_pipeline_stage_order.py` canh cả hai — trước đó không test
  nào canh thứ tự và cả 1087 test đều xanh với thứ tự sai.
- `scripts/survey_package_taxonomy.py` — khảo sát ở §6.
- Hạ tầng cũ vẫn dùng được: parser footprint BOM/PnP/CAD, bảy topology ở 5.5,
  đường ONNX 5.2 (no-op tuyệt đối khi thiếu artifact), ô model
  `models/active/package/` với `_NO_AUTO_ADOPT`.

**Classifier 6.1 KHÔNG train lại.** Nó đo được 22,3% ca đổi nhãn khi đổi khung
cắt, nhưng ở độ mịn package phần lớn sai số đó biến mất: `capacitor → led` (91
ca) và `resistor → capacitor` (33 ca) đều dẫn về cùng gói `hai_chan`. Chỉ
`ic ↔ thụ động` (12 ca) mới hỏng ROI — **124/136 ca được nêu tên là vô hại**.

---

## 6. Khảo sát taxonomy — 2026-09-03

`python scripts/survey_package_taxonomy.py --out datasets/survey/package_taxonomy_20260903`

Hai nguồn công khai, cả hai CC BY 4.0, bù nhau đúng chỗ:

| nguồn | mức | lớp | ảnh | box | có gì đặc biệt |
|---|---|---:|---:|---:|---|
| `fpic_boards_rf100` | **họ** | 23 | 672 | 134.047 | có `Electrolytic Capacitor` — tụ trụ đứng |
| `pcb_packages_winnies` | **package** | 24 | 173 | 16.632 | tên công nghiệp thật: SOT23, SOIC-16, TSSOP-14 |

Kết quả: **150.679 box**, 106.895 đủ lớn để có đặc trưng hình học (cạnh ngắn
≥ 12 px), 47 contact sheet theo lớp, 140 contact sheet theo cụm.

### 6.1. Đầu ra để bạn duyệt

```
datasets/survey/package_taxonomy_20260903/
  contact_sheets/    47 lưới 8x8, một cái mỗi lớp  ← xem cái này trước
  cluster_sheets/   140 lưới, một cái mỗi cụm      ← rồi tới cái này
  crops/            ảnh crop lẻ, tối đa 400 mỗi lớp
  features.csv      150.679 dòng đặc trưng hình học
  summary.json      thống kê + cấu hình lần chạy
```

### 6.2. Kích thước tách được họ, hình dạng thì không

Trung vị theo lớp, nguồn fpic (mức họ):

| lớp | box | cạnh dài | tỉ lệ cạnh | độ tròn |
|---|---:|---:|---:|---:|
| `Connector` | 4.551 | **223** | 2,39 | 0,22 |
| `Switch` | 188 | 157 | 1,65 | 0,15 |
| `Button` | 277 | 124 | 1,35 | 0,23 |
| `Clock` | 146 | 99 | 1,27 | 0,35 |
| `Electrolytic Capacitor` | 856 | **95** | 1,75 | 0,42 |
| `IC` | 7.452 | **84** | 1,22 | 0,28 |
| `Diode` | 283 | 81 | 1,63 | 0,34 |
| `Transistor` | 4.164 | 35 | 1,19 | 0,43 |
| `Resistor` | 3.146 | 32 | 1,76 | 0,45 |
| `Capacitor` | 21.021 | **29** | 1,59 | 0,48 |

Cạnh dài trải từ 29 tới 223 px và xếp đúng thứ tự trực giác. Tỉ lệ cạnh thì
gần như dẫm chân nhau (1,2–2,4 cho tất cả), và độ tròn còn tệ hơn.

### 6.3. Đặc trưng kế hoạch đề xuất cho họ `capacitor` KHÔNG chạy

Kế hoạch trước đề xuất tách tụ trụ đứng khỏi tụ chip bằng **độ tròn contour**
`4πA/P²`. Giờ có dữ liệu để đo, và kết quả là âm:

| đặc trưng | ngưỡng tốt nhất | độ chính xác **cân bằng** | recall tụ hoá | recall tụ chip |
|---|---:|---:|---:|---:|
| `long_side` | ≥ 43 px | **85,8%** | 98,1% | 73,5% |
| `area_px` | ≥ 1.044 | 84,7% | 96,7% | 72,7% |
| `area_frac` | ≥ 0,0004 | 79,6% | 95,0% | 64,3% |
| `aspect` | ≥ 2,08 | 62,8% | 37,1% | 88,5% |
| **`circularity`** | ≥ 0,683 | **53,9%** | 22,4% | 85,3% |

*(cân bằng = trung bình recall hai lớp; đoán bừa = 50%)*

Trung vị độ tròn: tụ hoá **0,420**, tụ chip **0,477** — tụ hoá còn **kém tròn
hơn**, ngược hẳn với lý thuyết. Nguyên nhân nhiều khả năng là Otsu trên crop
nhỏ, tương phản thấp không bắt được đường bao thân mà bắt vào chữ in và vệt
sáng trên nắp.

**Kết luận: bỏ ngưỡng độ tròn, dùng kích thước.** Nhưng có một điều kiện —
`long_side` là **pixel**, tức phụ thuộc độ phóng đại. Muốn dùng được trên dây
chuyền thì phải quy về **mm**, và px/mm chỉ có khi đăng ký CAD chạy. Không có
px/mm thì `area_frac` (79,6%) là phương án lùi, kém 6 điểm.

Đây là câu hỏi 3 ở §9.

### 6.4. Ba điều đáng ngờ trong dữ liệu công khai

1. **`Resistor Network` = 20.515 box** trong fpic, nhiều thứ nhì sau
   `Capacitor`, mà trung vị cạnh dài 26 px và tỉ lệ cạnh 1,64 — **không phân
   biệt được với `Capacitor`** (29 px, 1,59). Mạng điện trở thật là gói dài
   nhiều chân. Đây gần như chắc chắn là nhãn sai hàng loạt của RF100.
   Kiểm bằng `contact_sheets/fpic__Resistor_Network.png`.
2. **Cụm nào tách bằng `circularity` thì đáng ngờ.** §6.3 vừa cho thấy đặc
   trưng này gần vô dụng, nên cụm dựa vào nó nhiều khả năng là hiện vật của
   phép tách nền chứ không phải khác biệt gói thật. Ví dụ 4 cụm của
   `winnies/capacitor` có tỉ lệ cạnh gần y hệt (1,81–1,94) và chỉ khác độ tròn
   (0,13–0,56) — đó là ánh sáng, không phải kiểu vỏ.
3. **Cụm nào tách bằng kích thước thì đáng tin.** `fpic/Connector` chia thành
   4 cụm với cạnh dài 125 / 148 / 488 / **871** px và tỉ lệ cạnh tới **12,1** —
   đó là header dài thật, khác gói thật.

### 6.5. Taxonomy đề xuất — CHỜ BẠN DUYỆT

Cụm chỉ được đánh số, cố ý không đặt tên công nghiệp: đặt tên là việc phải
nhìn ảnh mới làm được, và máy đoán tên chỉ tạo ra cái nhãn trông như đã xong.

| họ 6.1 | cần chia? | gói | căn cứ |
|---|---|---|---|
| `resistor`, `led`, `diode` | ❌ | `hai_chan` | đã cài |
| `discrete_semiconductor` | ❌ | `goi_nho` | đã cài |
| `connector` | ❌ | `connector` | đã cài — nhưng xem §6.4 mục 3, có thể đáng chia tiếp |
| `magnetic`, `protection`, `timing`, `acoustic` | ⏳ | **chưa ánh xạ** | không lớp nào trong bảy lớp tả đúng |
| `relay`, `display`, `switch_control`, `battery_power_input` | ⏳ | **chưa ánh xạ** | hộp lớn nhiều chân, chưa có lớp |
| `capacitor` | ✅ | `tru_dung` / `hai_chan` | dùng **kích thước**, không phải độ tròn (§6.3) |
| `ic` | ✅ | 2 bên / 4 bên / không chân | vị trí chân — đã cài |

Bốn cụm của `fpic/IC` có cạnh dài 68 / 71 / 118 / 180 px — nhiều khả năng ứng
với SOT/SOIC nhỏ, SOIC lớn, và QFP. Xem `cluster_sheets/fpic__IC__cluster_*.png`
rồi cho tôi biết cụm nào là cái gì.

---

## 7. Còn thiếu gì

### 7.1. Không nguồn công khai nào có đủ

Trong 24 kiểu vỏ của winnies: **mọi IC đều là chân 2 bên** — không một QFP hay
QFN nào. Tức lớp 5 (`ic_bon_ben`) và lớp 6 (`ic_khong_chan`) có **0 ví dụ**.
fpic có `IC` nhưng ở mức họ, không nói chân nằm đâu.

Đúng hai phép chia mà kế hoạch cần đều thiếu ví dụ công khai.

### 7.2. Tập kiểm gán tay: 600–800 box

Không phải để train, mà để **đo tỉ lệ trúng của luật**. Lấy phân tầng chứ không
ngẫu nhiên — lớp 1 chiếm 86,5% nên mẫu đều sẽ toàn lớp 1:

| tầng | box | vì sao |
|---|---:|---|
| thân lớn, vuông (ứng viên lớp 4/5/6) | ~250 | cặp nhầm `ic ↔ thụ động` phải bằng 0 |
| thân dài, nhiều chân (ứng viên lớp 7) | ~150 | connector, bước chân lớn |
| thân nhỏ (ứng viên lớp 1/2/3) | ~250 | đo ngưỡng kích thước ở §6.3 |
| ngẫu nhiên nền | ~100 | bắt ca luật chưa nghĩ tới |

Không phải vẽ lại box — vị trí đã có trong 9.486 box đã duyệt, package chỉ thêm
một nhãn lớp, app có phím tắt `1`–`7`.

⚠️ **Draft package trên đĩa đang cũ.** `draft_package_boxes.json` mang **3.855**
box, sinh lúc mới có 16 tile verified; bộ thân hoàn chỉnh là **9.486 box / 95
tile** (`joint_boxes_cleaned.json`, sha `f4719695…`). Phải sinh lại, **sang thư
mục round mới**: `dataset_id` = sha256(*tên thư mục | số crop | crop đầu | tên
lớp*) — cố tình không gồm hình học — nên ghi đè tại chỗ tạo ra draft mới mang
đúng id cũ, và localStorage cũ sẽ trộn vào hình học mới mà không cảnh báo.

### 7.3. Không có CAD thì mất gì

Ba thứ, package chỉ bù được một:

| CAD cho | Ai bù được khi không có CAD |
|---|---|
| **số chân thật** | ✅ IPC-356, footprint trong BOM/PnP, golden lúc enroll, hoặc luật package |
| toạ độ land theo mm | ✅ chỉ IPC-356 |
| land không có linh kiện (test point, thermal pad) | ✅ IPC-356; ⚠️ golden nếu người enroll khoanh |

Bộ đọc footprint (`SOIC-16` → 16 chân, 2 cạnh) cho số chân **chính xác, không
cần model, không cần gán nhãn** — rẻ hơn luật package nhiều. Nếu BOM/PnP của
bạn có cột `footprint` thì làm nó trước (câu hỏi 5 ở §9).

---

## 8. Cổng nghiệm thu và rủi ro

**Ba cổng, theo thứ tự:**

1. **Nhầm `ic` ↔ thụ động phải bằng 0** trên tập kiểm §7.2. Đây là cặp duy nhất
   làm ROI *tệ đi thật*.
2. **Đo lại ROI trên board thật** — `tests/data/solder_geometry`, 28 pad đếm
   tay: bật luật phải **không giảm** độ phủ pad. Đây là cổng thật; cổng 1 chỉ
   là điều kiện cần.
3. **Mặc định TẮT** cho tới khi vượt cổng 2 trên board của chính dây chuyền.

**Rủi ro, theo khả năng xảy ra:**

1. **Luật hỏng có hệ thống, không hỏng dần.** Lead detector recall kém trên một
   board ⇒ mọi IC thành `ic_khong_chan` ⇒ mất ROI im lặng. *Đã chặn:* điều kiện
   ngữ cảnh ở §5.
2. **Ngưỡng kích thước không chuyển được giữa các độ phóng đại** (§6.3). *Giảm
   thiểu:* quy về mm khi có px/mm, lùi về `area_frac` khi không.
3. **Lớp 5/6 không có ví dụ nào** (§7.1). *Giảm thiểu:* nếu tập kiểm vẫn quá ít
   mẫu thì gộp chúng thành một lớp "IC lớn — cần người xem", dùng như cờ chuyển
   review chứ không như lớp đo đạc. Thà thành thật là không biết.
4. **Nhãn công khai sai** (§6.4). *Giảm thiểu:* duyệt contact sheet trước khi
   tin bất kỳ con số nào rút từ chúng.
5. **Không có CAD ⇒ không còn lưới an toàn thứ hai.** Package là *một* nguồn,
   không phải hai nguồn kiểm chéo. *Giảm thiểu:* cờ `review` khi số ROI dựng
   được lệch hạng kỳ vọng.

---

## 9. Câu hỏi cần bạn quyết

1. **Duyệt taxonomy ở §6.5 chứ?** Cụ thể: 4 cụm của `fpic/IC` ứng với những
   package nào, và có chia tiếp `connector` không (§6.4 mục 3).
2. **Bốn họ chưa ánh xạ** (`magnetic`/`protection`/`timing`/`acoustic` và
   `relay`/`display`/`switch_control`/`battery_power_input`) — thêm lớp thứ 8
   cho "hộp lớn nhiều chân", hay để chúng lùi về đường `multi_pin` như hôm nay?
3. **Ngưỡng kích thước cho `capacitor` quy về mm hay dùng `area_frac`?** (§6.3)
   Quy về mm chính xác hơn 6 điểm nhưng cần px/mm từ đăng ký CAD.
4. **Tập kiểm 600–800 box phân tầng ở §7.2 — đồng ý cỡ đó chứ?** Nhỏ hơn thì
   không đo nổi cặp nhầm `ic ↔ thụ động`.
5. **BOM/pick-and-place của bạn có cột `footprint` không?** Câu rẻ nhất trong
   danh sách: **có** thì làm bộ đọc footprint trước và hạ luật package xuống ưu
   tiên thấp; **không** thì luật lên đầu.
6. **Có xin được file IPC-D-356 từ bên gia công không?** Repo đọc được sẵn, và
   nó cho *từng pad một* — gần bằng có CAD.
7. **Lớp 6 (`ic_khong_chan`): chấp nhận kết luận "không kiểm được bằng ảnh 2D
   trên xuống" chứ?** QFN/BGA mà vẫn phải kiểm là bài toán X-quang.

---

Xem thêm: `Docs/bao_cao/tien_do_detect_2_luot.md` (bảng công việc sống),
`Docs/danh_gia/danh_gia_khoanh_box_than_linh_kien.md` (báo cáo box),
`Docs/ke_hoach/ke_hoach_pcb_defect_toan_mach.md` (kế hoạch lỗi toàn mạch).
