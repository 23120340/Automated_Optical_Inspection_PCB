# Kế hoạch phân nhóm package cho linh kiện

> Soạn 2026-08-31, sửa lần 3: **giả định không có CAD**, **rút gọn còn các loại
> package cơ bản phân biệt bằng mắt thường**, và **xét lại theo tình huống chỉ có
> golden image hoặc BOM/pick-and-place** — tình huống đó làm §4 phải viết lại.
> **Đã triển khai phần code ngày 2026-09-01.** Con số có chữ *đo được* đều chạy
> ra từ dữ liệu trong repo; chỗ nào là suy đoán thì nói thẳng là suy đoán.
> Phần cần người/model được tách riêng ở mục 0 dưới đây và chưa được giả vờ là xong.

## 0. Trạng thái triển khai và phần phải làm thủ công

- `[x]` Parser footprint cho BOM/PnP/CAD; thứ tự bằng chứng là pad CAD →
  footprint → package model → family legacy.
- `[x]` Bảy topology ở 5.5, gồm `ic_khong_chan` không sinh ROI giả và cờ
  review khi số cạnh/chân kỳ vọng lệch ROI thực tế.
- `[x]` Bước 5.2 ONNX + manifest riêng, chỉ `accept` mới tác động và không đè
  footprint/CAD. Thiếu artifact là no-op tuyệt đối.
- `[x]` Registry/Streamlit có slot package riêng và `_NO_AUTO_ADOPT`; đặt file
  vào `models/active/package/` không tự bật.
- `[x]` App gán nhãn 7 lớp đã dựng tại
  `datasets/labelling/component_bodies_round2_20260830/label_packages.html`.
  Migration giữ nguyên tọa độ box cũ, reset nhãn cũ thành sentinel `unknown`,
  giữ key localStorage cũ và ghi receipt SHA-256.
- `[x]` Packer dataset chia theo board, notebook Kaggle xuất ONNX/manifest và
  gate nghiệm thu 28 pad; mọi bước fail-closed khi taxonomy/split sai.
- `[ ]` **DRAFT ĐANG CŨ:** `draft_package_boxes.json` mang 3.855 box, sinh lúc
  mới có 16 tile verified; bộ thân hoàn chỉnh là **9.486 box / 95 tile**. Phải
  sinh lại, **sang thư mục round mới** (lý do ở §7).
- `[ ]` **CẦN NGƯỜI GÁN NHÃN:** cả 3.855 box đều `unknown` (8 prelabel hình
  học đã bị gỡ ở `73ce2aa` vì ngưỡng ngược). Với hướng luật thì chỉ cần
  **600–800 box phân tầng** để ĐO, không phải cả bộ để train — §7.
- `[ ]` **CẦN VIẾT LUẬT, KHÔNG TRAIN MODEL** (đổi 2026-09-03, §8): luật chạy
  SAU 6.1 và chỉ chia nhỏ trong hai họ `capacitor`/`ic`, điền vào đúng tham
  số `package` mà `terminal_geometry()` đã nhận sẵn. Đường ONNX ở 5.2 giữ
  nguyên làm dự phòng — nó vốn no-op tuyệt đối khi thiếu artifact.
- `[ ]` **PHẢI ĐỔI THỨ TỰ PIPELINE (hai chỗ, §8.5):** (a) đưa
  `classify_components` từ `pipeline.py:709` lên trước 5.5; (b) đưa lead
  detector từ dòng 507 lên trước bước phân package — detector mới là một lớp
  `component`, không sinh pads/pins nên `leads` sẽ rỗng đúng chỗ luật cần.
- `[x]` **Classifier 6.1 giữ nguyên, không train lại** — lý do đo được ở §8.6.

Các lựa chọn triển khai đã chốt: đúng 7 lớp, đặt ở 5.2, package ẩn chân là
không kiểm được bằng ảnh 2D trên xuống, footprint làm trước. **Thay đổi
2026-09-03: bước 5.2 do LUẬT quyết, không do CNN** — lý do và cách chặn rủi
ro ở §8.

## 1. Kết luận nhanh

- **Package NỐI SAU classifier, không thay classifier.** (Chốt 2026-09-03.)
  6.1 vẫn cho họ linh kiện; luật hình học chỉ chia nhỏ tiếp **bên trong một
  họ** — `capacitor` → tròn/vuông, `ic` → chân 2 bên/4 bên/không chân. Không
  train model package. §8.
- **Điều kiện hoá theo họ là bắt buộc, và đo được vì sao.** Luật hình học
  *toàn cục* trên box thân gần như vô dụng: chip 2 chân chiếm 86,5% nên
  "luôn đoán chip" đã đúng 86,5%, ngưỡng diện tích tốt nhất chỉ đạt 88,7%
  (**+2,2 điểm**), còn ngưỡng tỉ lệ cạnh đạt 84,5% — **tệ hơn baseline**.
  Biết trước họ là thứ xoá đi cái mất cân bằng đó. §8.1.
- **Chỉ HAI họ cần chia**, không phải bảy lớp tự do: `capacitor` và `ic`. §8.2.
- **Classifier 6.1 KHÔNG phải train lại.** Ở độ mịn package, 124/136 ca đổi
  nhãn được nêu tên trong báo cáo box (91%) là vô hại — `capacitor`, `led`,
  `resistor` đều dẫn về cùng một gói. Chỉ `ic ↔ thụ động` mới hại. §8.6.
- **Nhưng phải đổi THỨ TỰ:** 6.1 đang chạy ở `pipeline.py:709`, tức **sau**
  5.5. Luật khoá theo họ nên 6.1 phải lên trước. Đổi thứ tự thuần tuý. §8.5.
- **Gán nhãn co từ 9.486 xuống 600–800 box**, đổi mục đích thành đo tỉ lệ
  trúng của luật. Không bỏ được: đúng hai phép chia mà kế hoạch cần đều có
  **0 ví dụ** trong dữ liệu công khai. §7, §8.8.
- **Package là phương án CUỐI, không phải phương án duy nhất.** `pad_count` —
  số chân *kỳ vọng* — hiện chỉ sinh ra ở **đúng một dòng**: `cad_fusion.py:734`.
  Nhưng **BOM và pick-and-place cũng trả lời được**, và rẻ hơn nhiều: cả hai
  reader trong repo **đã đọc cột `footprint`** rồi **vứt đi không dùng**. Một
  bộ đọc tên footprint (`SOIC-16` → 16 chân, 2 cạnh) cho số chân **chính xác,
  không cần model, không cần gán nhãn**. Bảng đầy đủ ở §4.
- **Vì vậy thứ tự làm nên đổi:** bộ đọc footprint trước (rẻ, chính xác),
  classifier package sau — cho board **không có hồ sơ gì cả**.
- **Giá trị nằm ở hình học, không ở cái nhãn.** Package nói cho bước 5.5 biết
  linh kiện có mấy chân và chân ở cạnh nào. Hiện 5.5 phải **đoán điều đó từ
  pixel**, và mã nguồn đã ghi lại một ca đoán sai.
- **Vị trí đúng KHÔNG phải "luồng tiếp theo sau 6.2".** Đặt sau 6.2 thì nhãn ra
  đời sau khi mọi ROI đã dựng xong — quá muộn. Đề xuất **bước 5.2**, ngay sau
  bước 5 (cắt crop) và trước 5.5. §5.
- **Chỉ 7 lớp, tất cả nhìn một cái là biết.** Rút từ bản trước xuống, theo đúng
  góp ý. Tiêu chí giữ một lớp: *người gán nhãn phân biệt được trong một giây*
  **và** *lớp đó làm 5.5 hành xử khác đi*. Lớp nào không thoả cả hai thì bỏ. §3.
- **Phạm vi thật nhỏ hơn cảm giác:** đo được **86,5% linh kiện là loại 2 chân**
  và đường 2-chân của 5.5 đã đúng sẵn. Chỉ **13,5%** đi vào nhánh "nhiều chân" —
  nhưng chúng mang **31,2% tổng số mối hàn**. Tiền nằm ở nhóm 13,5% đó.

---

## 2. Vì sao cần package: 5.5 đang đoán, và đoán bằng pixel

`aoi_pipeline/config.py:30-50` giữ toàn bộ tri thức hiện có về hình dạng chân:

```python
TWO_TERMINAL_CLASSES = {"capacitor", "resistor", "diode", "led", "inductor", "fuse"}
PAD_ONLY_CLASSES     = {"pads"}
DEFAULT_TERMINAL_GEOMETRY = "multi_pin"   # ic, connector, transistor, relay, ...
```

Ba nhóm, và nhóm thứ ba là **cái sọt đựng tất cả những gì còn lại**. Đo trên bộ
Winnies (16.632 box, 24 kiểu vỏ — nguồn package-labelled duy nhất repo có):

| | số box | % linh kiện | mối hàn | % mối hàn |
|---|---:|---:|---:|---:|
| Vào nhánh `two_terminal` | 14.379 | **86,5%** | 28.758 | 68,8% |
| Vào nhánh `multi_pin` | 2.253 | **13,5%** | 13.017 | **31,2%** |

Và điều quan trọng nhất, cũng đo được: **2.247/2.253 gói trong nhánh `multi_pin`
chỉ có chân trên ĐÚNG 2 cạnh**. Nhưng `_multi_pin_rects` dựng dải quanh **cả 4
cạnh**, rồi mới lọc bớt bằng năng lượng pixel và độ đều của "cái lược" chân.

Tức với gần như mọi linh kiện nhiều chân, pipeline **dựng thừa 2 dải trên nền
trống rồi nhờ pixel nói hộ dải nào là thật**. Phép nhờ đó có lúc sai — chính mã
nguồn ghi lại một ca:

> *"letting it run anyway is how D201 lost the single lead on its top edge: the
> silkscreen brackets in the band corners dragged the comb score to 0.938 against
> a 0.95 gate and the whole band went, real lead included."*
> — `aoi_pipeline/solder/geometry.py`

Biết package là biết trước câu trả lời: **IC chân hai bên ⇒ hai cạnh dài có
chân, hai cạnh ngắn là nền, chấm hết.** Không cần đo năng lượng, không có cửa
cho vệt silkscreen làm hỏng phép đo.

> **Ranh giới:** package **không** làm mối hàn dễ chấm hơn. Nó làm ROI *nằm đúng
> chỗ* và *đủ số lượng*. Chấm tốt/xấu vẫn là việc của 6.2.

---

## 3. Bảy lớp cơ bản

Quy ước: `T` = hành vi mà 5.5 sẽ áp dụng.

| # | Tên | slug | Nhìn thế nào là biết | T — 5.5 làm gì khác đi |
|---|---|---|---|---|
| 1 | **Hai chân** | `hai_chan` | hộp nhỏ chữ nhật (hoặc trụ nằm), **hai đầu kim loại ở hai cạnh ngắn**, giữa là thân | 2 ROI ở hai đầu trục dài. *Đường đang chạy, đã đúng* |
| 2 | **Trụ đứng** | `tru_dung` | **hình tròn nhìn từ trên**, nắp nhôm có rãnh chữ thập, thân cao | 2 ROI, nhưng **không đoán trục bằng kim loại** — vỏ can chính là kim loại |
| 3 | **Gói nhỏ 3–5 chân** | `goi_nho` | hộp đen nhỏ, **vài chân to bản**, thường 2 bên 1 bên | dải chân trên **2 cạnh đối**, ít chân, chân dày |
| 4 | **IC chân hai bên** | `ic_hai_ben` | thân đen **dài**, hai hàng chân mảnh ở **hai cạnh dài** | dải chân trên **đúng 2 cạnh dài**; 2 cạnh ngắn không dựng dải |
| 5 | **IC chân bốn bên** | `ic_bon_ben` | thân **vuông**, chân ra **cả bốn phía** | dải chân trên **cả 4 cạnh** — đây mới là lúc dùng đúng |
| 6 | **IC không thấy chân** | `ic_khong_chan` | thân vuông/chữ nhật, **mép nhẵn, không có chân nào ló ra** | **KHÔNG sinh ROI 2D.** Đánh dấu "không kiểm được bằng ảnh trên xuống" |
| 7 | **Connector / xuyên lỗ** | `connector` | dãy chân **thẳng hàng**, thân nhựa, hoặc chân **xuyên qua lỗ** với thiếc thành vòng khuyên | dải chân **1 hoặc 2 hàng**, bước chân lớn, ROI to hơn |

**Vì sao đúng bảy lớp này, không hơn không kém.** Mỗi lớp trả lời một câu 5.5
đang phải đoán, và mỗi lớp là một hành vi khác nhau — bỏ lớp nào cũng mất một
hành vi. Ngược lại, mọi lớp tôi đã cắt khỏi bản trước đều **không** đổi hành vi
5.5:

- Tách *tụ gốm* khỏi *điện trở chip*: cả hai đều là 2 chân, 5.5 làm y hệt nhau.
  Lại còn phân biệt bằng **màu**, mà bước 1 đang bật white-balance + CLAHE +
  normalize chưa từng được đo A/B. Bỏ.
- Tách *SOIC* khỏi *TSSOP*: cùng là chân hai bên. Khác nhau ở bước chân, mà ở
  46 µm/px TSSOP đã ở mức 14,1 px/bước — sát mép đọc được. Bỏ.
- Tách *MELF* khỏi *chip chữ nhật*: cùng 2 chân, cùng trục dài. Bỏ.
- Tách *diode có vạch catot* để bắt ngược cực: vạch cực rộng ~0,2–0,3 mm ⇒ 4–6 px.
  Thấy *có vạch* thì được, đọc *vạch ở đầu nào* thì **chưa đủ pixel**. Bỏ, và
  đừng hứa kiểm tra phân cực ở độ phân giải hiện tại.
- Tách *SOT có tab tản nhiệt*: **cái này tiếc nhất.** Tab là một pad to gấp
  nhiều lần chân tín hiệu, đưa qua cùng ngưỡng `solder_ratio` là so hai thứ khác
  nhau. Nhưng nó là chi tiết bên trong lớp 3, nên để **đợt sau**, khi lớp 3 đã
  chạy đúng.

**Số học độ phân giải, ở 46 µm/px của dự án** — quyết định lớp 5 và 6 phải là
*nhận dạng hình dáng*, không phải *đếm chân*:

| Loại | Bước chân | px/bước | Đếm chân được? |
|---|---|---:|---|
| SOIC / SO | 1,27 mm | 27,6 | ✅ thoải mái |
| TSSOP / SSOP | 0,65 mm | 14,1 | ⚠️ sát mép |
| QFP | 0,50 mm | 10,9 | ❌ |
| QFN | 0,40 mm | 8,7 | ❌ |
| BGA | 0,35 mm | 7,6 | ❌ |

*(cần ~3 px cho khe + ~3 px cho chân + biên ⇒ ~12 px là sàn thực dụng)*

Nên "IC 6 chân / IC 8 chân" như bạn ví dụ **làm được với lớp 4** (SOIC/SO) và
**không làm được với lớp 5** (QFP). Với lớp 5 và 6, cái nhìn thấy là **hình
dáng** — vuông có chân bốn bên, hay vuông nhẵn không chân — và đó đã đủ cho việc
5.5 cần.

---

## 4. Không có CAD thì mất gì, và package bù được đến đâu

Đây là phần viết lại theo giả định của bạn: **sẽ không có file CAD.**

`aoi_pipeline/solder/cad_fusion.py` mở đầu bằng đúng phép chia vai:

> *"CAD supplies the land geometry and the true terminal count; the detector
> supplies a per-component position correction."*

Bỏ CAD đi thì vế trái biến mất. Cụ thể, **ba** thứ mất — và package chỉ bù được
**một**, nhưng đúng cái quan trọng nhất:

| CAD cho | Không có CAD | Ai bù được? |
|---|---|---|
| **Số chân thật của linh kiện** | `pad_count` chỉ sinh ra ở `cad_fusion.py:734` | ✅ **IPC-356**, hoặc **footprint trong BOM/PnP**, hoặc **golden lúc enroll**, hoặc package classifier — xem §4.1 |
| Toạ độ land theo mm, chính xác từng pad | mất hẳn | ✅ **IPC-356** cho đúng cái này; ❌ các nguồn còn lại không |
| Land **không có linh kiện** (test point, thermal pad, lỗ bắt vít) | mất hẳn | ✅ **IPC-356**; ⚠️ golden thấy được nếu người enroll khoanh; ❌ package không |

### 4.1. Xếp hạng nguồn cho SỐ CHÂN, khi không có CAD

Đọc từ code, không phải suy đoán:

| # | Nguồn | Cho được gì | Trạng thái trong repo |
|---|---|---|---|
| 1 | **IPC-D-356 netlist** | **Từng pad một**: toạ độ + số hiệu chân ⇒ `pad_count` chính xác, gần bằng CAD | ✅ **đã đọc được** — `load_ipc356()`. CM thường có sẵn vì đây là file test điện |
| 2 | **BOM hoặc PnP có cột `footprint`** | Chuỗi `SOIC-16` / `0603` / `QFP-64` **mã hoá sẵn gói và số chân** | ⚠️ **đã đọc, chưa dùng** — `BomEntry.footprint`, `CadComponent.footprint`; grep cho thấy 3 chỗ, cả 3 chỉ để ghi ra `to_dict()` |
| 3 | **Pick-and-place không có footprint** | Vị trí + **góc xoay** + mặt. Docstring `load_placement_csv` nói thẳng: *"no lands... the terminal topology then still comes from the derived geometry"* | ✅ đã dùng — góc xoay là `axis_known` |
| 4 | **Tiền tố RefDes** (R/C/U/Q/D/J…) | Chỉ **họ**, đúng bằng mức `terminal_geometry()` hôm nay | ✅ đã có — `designator_to_class()` |
| 5 | **Golden image** | Số chân **từng ô**, chốt lúc enroll, người xác nhận một lần cho mỗi SKU. Nói được cái **thực tế nằm trên board**, không phải cái bản vẽ định | ⚠️ hạ tầng có (bước 3.5), chưa có bước đếm chân |
| 6 | **Package classifier** (kế hoạch này) | Hạng số chân (2 / 3–5 / nhiều-hai-bên / nhiều-bốn-bên / không đếm được) | ❌ chưa có |

**Trả lời thẳng câu hỏi "có đủ tài nguyên để detect đủ số chân không":**

- **Chỉ có golden image:** **có**, nhưng phải trả một lượt **người xác nhận khi
  enroll**, cho mỗi SKU. Đổi lại nó chính xác cho đúng SKU đó và phản ánh board
  thật. Đây là đường CAD-free tự nhiên nhất cho một dây chuyền ít SKU.
- **Chỉ có BOM/PnP:** **tuỳ có cột `footprint` hay không.** Có → **chính xác,
  miễn phí, không cần model**. Không có → chỉ được họ + góc xoay, tức **không**
  đủ số chân.
- **Xin được IPC-356:** **có, chính xác nhất**, và repo đã đọc được sẵn. Nếu
  hỏi được CM thì hỏi cái này trước mọi thứ khác.
- **Không có gì cả:** lúc đó mới cần classifier package ở §3.

⇒ **Việc rẻ nhất nên làm trước kế hoạch này:** viết bộ đọc tên footprint và cho
`terminal_geometry()` ưu tiên nó. Không model, không gán nhãn, dùng lại đúng
chuỗi mà hai reader đang đọc rồi bỏ. Classifier package vẫn đáng làm, nhưng nó
là lưới cuối chứ không phải lưới duy nhất.

**Dù đi đường nào, cái bẫy dưới đây vẫn phải chặn — repo đã có sẵn một test
đặt tên đúng nó:**

```
tests/inspection/test_cad_fusion.py:457
test_cad_pad_count_overrides_the_class_topology_guess
    """A four-pad part labelled 'resistor' must not be treated as two-terminal."""
```

Một linh kiện 4 chân bị detector gọi nhầm là `resistor` sẽ được xử lý như loại 2
chân — **sinh 2 ROI thay vì 4, im lặng bỏ sót 2 mối hàn.** Hôm nay **CAD là thứ
duy nhất chặn được chuyện đó**. Không có CAD thì **không gì chặn được**, và bỏ
sót một mối hàn không để lại dấu vết nào: không cảnh báo, không cột trống, chỉ
là hai ROI ít hơn mức đáng có.

Chuỗi kiểm tra đó sống lại mà không cần CAD — bằng **bất kỳ** nguồn nào ở
§4.1, không nhất thiết phải là classifier:

- Lớp package nói **hạng số chân kỳ vọng** (2 / 3–5 / nhiều-hai-bên / nhiều-bốn-bên
  / không đếm được). Đây là con số thô hơn CAD, nhưng đủ để bắt đúng ca nguy
  hiểm: *nhãn nói 2 chân mà gói là IC*.
- 5.5 đếm được ROI thật nó dựng ra. **Lệch hạng ⇒ cờ `review`**, không im lặng.
- Ba thứ CAD-only còn lại chuyển sang **Golden Inspection (bước 3.5)**: nó
  enroll từ một board chuẩn thật chứ không cần file thiết kế, nên nó là đường
  CAD-free duy nhất biết được "chỗ này lẽ ra có gì".

**Hai hệ quả cho phần còn lại của kế hoạch, do bỏ CAD:**

0. **Làm bộ đọc footprint TRƯỚC.** Nó rẻ hơn, chính xác hơn và không cần dữ
   liệu gán nhãn nào. Chỉ khi board không có hồ sơ thì classifier mới vào cuộc.
1. **Không được dùng CAD làm ground truth khi nghiệm thu.** Cổng ở §8 phải đo
   trên **28 pad đếm tay** đã có sẵn ở `tests/data/solder_geometry`, không phải
   trên pad_count của một file CAD giả lập.
2. **`pad_only` và `keep_unassigned_leads` quan trọng hơn trước.** Không có CAD
   thì test point và pad trống không ai khai báo; đường duy nhất chúng lọt vào
   kết quả là qua detection chân không thuộc linh kiện nào —
   `leads.py:187` giữ chúng lại thành ROI độc lập. **Giữ mặc định `True`.**

`cad.py` và `cad_fusion.py` **cứ để nguyên**: chúng đã tự khai *"Nothing here is
required. With no CAD file the pipeline behaves exactly as it did before, and
every function in this module is simply never called."* Không cần gỡ, và nếu mai
kia có file CAD thật thì hai nguồn kiểm chéo lẫn nhau.

---

## 5. Vị trí trong pipeline — vì sao **không** đặt sau 6.2

Thứ tự thật, đọc từ `aoi_pipeline/pipeline.py:568-578`:

```
4   detect_components()   → detections (mang nhãn HỌ của detector)
5   make_crops()          → crops              ← crop đã có ở đây
5.5 make_solder_crops()   → ROI mối hàn        ← dùng terminal_geometry(detection.label)
6.1 classify_components(crops)
6.2 grade_solder()
```

Mấu chốt: **5.5 lấy topology từ nhãn của bước 4, và chạy TRƯỚC 6.1.**

| Phương án | Cái được | Cái mất |
|---|---|---|
| **(a) Bước riêng sau 6.2** *(đề xuất ban đầu)* | không đụng gì đang chạy | **nhãn ra đời sau khi ROI đã dựng xong ⇒ 0 tác dụng lên 5.5**; chỉ còn giá trị thống kê |
| (b) Thêm head package vào detector bước 4 | không tốn thêm lần suy luận | phải train lại detector, mà detector đang là nút thắt riêng (8/10 bo, chưa pack nổi dataset). Trộn hai việc khó vào một |
| (c) Đảo 6.1 lên trước 5.5 | tái dùng model 6.1 | 6.1 phân loại **họ**, không phải package; vẫn phải thêm head, lại đụng `_invalidate_after` của UI |
| **(d) Bước 5.2, giữa 5 và 5.5** ⭐ | crop **đã có sẵn**; 5.5 **đã nhận sẵn tham số `geometry=`**; không đảo thứ tự gì | thêm một lần suy luận nhỏ mỗi linh kiện |
| (e) Bảng tra `họ → package` cứng | không cần model | không phân biệt được lớp 4 với lớp 6 — cùng họ `ic`, topology ngược nhau. Đây đúng là cái đang sai hôm nay |

**Khuyến nghị: (d), bước 5.2.**

Cửa vào đã có sẵn. Nguyên văn docstring của `derive_solder_joints()`:

> *"``frame`` and ``geometry`` let a caller that knows better override what the
> box alone can say. CAD fusion uses them..."*

Cơ chế "một nguồn bên ngoài biết rõ hơn cái hộp" **đã được thiết kế sẵn**. Điểm
đáng nói: nó được thiết kế **cho CAD** — và nếu không có CAD thì package sẽ là
**khách hàng duy nhất** của cửa đó, chứ không phải khách hàng thứ hai. Việc phải
làm: mở `terminal_geometry()` từ 3 giá trị lên 7, thêm `PadProfile` tương ứng,
và cho 5.5 ưu tiên `geometry` do 5.2 cấp.

**Lập luận mạnh nhất CHỐNG lại đề xuất của chính tôi:** thêm 5.2 là thêm model
thứ sáu vào một pipeline đã có **hai model chưa đạt** (detector macro recall
0,52; classifier 6.2 chưa dùng để quyết được). Một classifier package sai sẽ đưa
topology sai vào 5.5 và làm ROI **tệ hơn** đường đoán-bằng-pixel hiện tại. Vì
vậy §8 đặt cổng nghiệm thu theo đúng lệ của repo: **phải hơn thứ nó thay thế, đo
trên board thật, mặc định TẮT cho tới khi hơn.**

---

## 6. Dữ liệu: có sẵn bao nhiêu, thiếu bao nhiêu

Đo trực tiếp từ `datasets/public/pcb_packages_winnies/export_yolov8_v3.zip`,
gộp 24 kiểu vỏ của nó về 7 lớp ở §3:

| Lớp | Gộp từ | box | Đủ train? |
|---|---|---:|---|
| 1 `hai_chan` | resistor, capacitor, LED, SOD123/128/323, Resistor rond, feriet kraal, Polyfuse_GR/Z | **14.379** | ✅ thừa |
| 3 `goi_nho` | SOT23/143/223/457/753, MOSFET, MOSFET-2 | **1.629** | ✅ |
| 4 `ic_hai_ben` | SOT96 (SO-8), SOIC-12/14/16, TSSOP-14/16 | **618** | ✅ |
| 2 `tru_dung` | — | **0** | ❌ **phải tự gán** |
| 5 `ic_bon_ben` | — | **0** | ❌ **phải tự gán** |
| 6 `ic_khong_chan` | — | **0** | ❌ **phải tự gán** |
| 7 `connector` | — | **0** | ❌ **phải tự gán** |

*(14.379 + 1.629 + 618 + 6 nhãn `CHIP` mơ hồ = 16.632, khớp tổng)*

**Đây là điều phải nói thẳng: bốn trong bảy lớp không có lấy một mẫu.** Và ba
trong bốn lớp đó (2, 5, 6) chính là những lớp đổi hành vi 5.5 nhiều nhất. Bộ
Winnies dạy được cái pipeline **đã làm đúng rồi** và không dạy được cái đang sai.

Hai hạn chế nữa của Winnies:
1. **Chỉ 73 ảnh nguồn** (173 file là biến thể lật/xoay của Roboflow). Chia
   train/val theo file là rò rỉ — đúng cái bẫy đã thổi accuracy 6.2 từ 89,9% lên
   97,65%. Phải gom theo ảnh nguồn.
2. **Sai miền** — không phải camera/ánh sáng dây chuyền của bạn.

⇒ **Bốn lớp còn thiếu phải gán từ chính tile của dự án**, và đó là việc ở §7.
Tin tốt: tile của dự án có sẵn IC lớn — chính chúng là lý do bộ vòng 2 được tạo
(box lớn nhất detector cho ra 251×250 px trong khi QFP thật ~350 px).

---

## 7. Kế hoạch gán nhãn — để ĐO luật, không để train

Hướng luật (§8) đổi hẳn quy mô việc này. Không còn cần một tập train; cần một
tập **kiểm** đủ để đo tỉ lệ trúng của luật.

| | model (bản cũ) | luật (bản này) |
|---|---:|---:|
| box phải gán nhãn | **9.486** | **600–800** |
| mục đích | train | đo tỉ lệ trúng |
| ước tính | 8–12 giờ | **1–1,5 giờ** |

**Chọn mẫu thế nào cho 600–800 box đó.** Không lấy ngẫu nhiên đều: lớp 1 chiếm
86,5% nên mẫu ngẫu nhiên sẽ gần như toàn lớp 1 và không nói gì về nhóm đắt
tiền. Lấy **phân tầng**:

| tầng | số box | vì sao |
|---|---:|---|
| 6.1 nói `ic` | ~300 | chia chân 2 bên / 4 bên / không chân — 0 ví dụ công khai |
| 6.1 nói `capacitor` | ~250 | đo ngưỡng độ tròn tròn/vuông — 0 ví dụ công khai |
| 6.1 nói họ khác | ~150 | xác nhận "không cần chia" là đúng, và bắt ca `ic ↔ thụ động` |

**Không phải vẽ lại box.** Vị trí đã có sẵn trong 9.486 box đã duyệt; package
chỉ thêm một nhãn lớp. App có phím tắt `1`–`7` ⇒ bấm một phím mỗi box.

⚠️ **Draft package trên đĩa đang cũ.** `draft_package_boxes.json` mang **3.855**
box, sinh ra lúc mới có 16 tile verified. Bộ thân hoàn chỉnh là **9.486 box trên
95 tile** (`joint_boxes_cleaned.json`, sha `f4719695…`, khớp `pack_manifest`).
Phải sinh lại trước khi gán nhãn, **và sinh sang thư mục round mới**:
`dataset_id` = sha256(*tên thư mục | số crop | crop đầu | tên lớp*) — cố tình
không gồm hình học — nên ghi đè tại chỗ sẽ tạo ra draft mới mang **đúng id cũ**,
và localStorage cũ sẽ trộn vào hình học mới mà không cảnh báo.

---

## 8. Kế hoạch: LUẬT nối SAU classifier, không train model package

**Quyết định (2026-09-03, theo yêu cầu).** Package **không thay** classifier.
Classifier 6.1 vẫn chạy và vẫn cho họ linh kiện; package chỉ **chia nhỏ tiếp
bên trong một họ**, bằng luật hình học:

```
6.1 classifier  ──►  họ (capacitor / ic / resistor / ...)
                        │
                        ├─ họ = capacitor ──► luật ──► tụ tròn | tụ vuông
                        ├─ họ = ic        ──► luật ──► chân 2 bên | 4 bên | không chân
                        └─ họ khác        ──► gói mặc định của họ đó
```

### 8.1. Vì sao điều kiện hoá theo họ là điều BẮT BUỘC, không phải tuỳ chọn

Đo trên `datasets/public/pcb_packages_winnies/export_yolov8_v3.zip` — 24 kiểu
vỏ, 16.632 box có nhãn footprint thật:

| tầng gói | box | diện tích p10 / trung vị / p90 (×10⁻⁴) | tỉ lệ cạnh trung vị |
|---|---:|---|---:|
| chip 2 chân | 14.379 | 1,7 / 4,8 / 15,0 | 1,90 |
| gói nhỏ 3–5 chân | 1.433 | 3,4 / 9,7 / 33,5 | 1,29 |
| IC | 814 | 11,6 / 38,2 / 146,3 | 1,24 |

Chip chiếm **86,5%**, nên luật "luôn đoán chip" đã đúng 86,5% mà không cần nghĩ.
Luật ngưỡng tốt nhất tìm được trên toàn bộ:

| luật toàn cục | đúng | so với baseline 86,5% |
|---|---:|---|
| hai ngưỡng trên **diện tích** | 88,7% | **+2,2 điểm** |
| hai ngưỡng trên **tỉ lệ cạnh** | 84,5% | **−2,0 điểm — tệ hơn** |

**Luật hình học toàn cục trên box thân gần như vô dụng.** Đây cũng là lời giải
thích bằng số cho vụ luật cũ bị gỡ ở `73ce2aa`: nó dùng **tỉ lệ cạnh**, đúng cái
đặc trưng đo được là *dưới* baseline.

Nhưng cả hai con số trên đều là hệ quả của việc **trộn ba tầng vào một bài
toán**. Cho biết trước họ là `ic`, câu hỏi còn lại không còn là "chip hay IC"
(đã biết) mà là "IC kiểu nào" — và ba tầng mất cân bằng 86:9:5 biến thành một
bài toán cân bằng hơn nhiều. **Đó chính là giá trị mà classifier mang lại cho
bước này, và là lý do không được bỏ nó.**

### 8.2. Chỉ HAI họ cần chia, không phải bảy lớp tự do

Trong 16 họ mà 6.1 xuất ra, phần lớn ứng đúng một kiểu vỏ nên **không cần luật
gì cả**:

| họ 6.1 | cần chia? | gói |
|---|---|---|
| `resistor`, `led`, `diode` | ❌ | chip 2 chân |
| `discrete_semiconductor` | ❌ | gói nhỏ 3–5 chân |
| `connector` | ❌ | đã là lớp riêng |
| `magnetic`, `protection`, `timing`, `acoustic` | ❌ | theo mặc định của họ |
| `relay`, `display`, `switch_control`, `battery_power_input` | ❌ | nhiều chân, hộp lớn |
| **`capacitor`** | ✅ | **tròn (trụ đứng) / vuông (chip)** |
| **`ic`** | ✅ | **chân 2 bên / 4 bên / không thấy chân** |

Phạm vi thật của phần luật vì thế là **hai họ**, không phải bảy lớp. Nhỏ hơn hẳn
so với bản kế hoạch trước.

### 8.3. Luật cho từng họ

**Họ `capacitor` — tròn hay vuông.** Tụ hoá trụ đứng nhìn từ trên là hình
**tròn**; tụ chip là chữ nhật. Tỉ lệ cạnh không phân biệt được (cả hai đều gần
1–2), nên phải dùng **độ tròn của contour thân**: `4πA / P²` từ
`cv2.findContours`, xấp xỉ 1 cho tròn, ~0,78 cho vuông. Kèm điều kiện diện tích
(tụ hoá lớn hơn tụ chip nhiều bậc). Đây là luật trên pixel, **ngưỡng phải đo**,
không được đoán.

**Họ `ic` — chân ở đâu.** Ba gói:

| gói | dấu hiệu |
|---|---|
| chân 2 bên (SOIC/SOP/TSSOP) | dải lead trên **đúng 2 cạnh đối** |
| chân 4 bên (QFP) | dải lead trên **cả 4 cạnh** |
| không thấy chân (QFN/BGA) | **không lead nào** — kèm điều kiện an toàn ở §8.4 |

Không dùng tỉ lệ cạnh: đo được SOIC-16 = 1,17 và TSSOP-16 = 1,08, tức các IC
trong bộ dữ liệu này **gần vuông hết**, không "dài" như trực giác. Dấu hiệu thật
là **vị trí chân**, và chân nằm ngoài box thân theo đúng quy ước
(`visible component body only; exclude leads, pads and test points`).

Nguyên liệu đã có sẵn: `split_lead_detections()` và
`assign_leads_to_components()` trong `aoi_pipeline/solder/leads.py`.

### 8.4. Rủi ro riêng của luật, và cách chặn

CNN hỏng thì hỏng dần; **luật hỏng thì hỏng có hệ thống**. Ca xấu nhất: lead
detector recall kém trên một board ⇒ **mọi IC thành "không thấy chân"** ⇒ 5.5
không dựng ROI ⇒ mất mối hàn mà không ai biết.

Chặn bằng điều kiện ngữ cảnh, không bằng ngưỡng tin cậy:

> Chỉ kết luận "không thấy chân" khi lead detector **đã tìm được lead ở linh
> kiện khác trên cùng board**. Không có bằng chứng nó đang chạy được thì trả
> `unknown`, và `terminal_geometry()` tự lùi về đường họ-detector như hôm nay.

### 8.5. Hai thay đổi bắt buộc trong pipeline

**(a) Đưa 6.1 lên TRƯỚC 5.5.** Hôm nay `classify_components(crops)` nằm ở
`pipeline.py:709`, tức **sau** lời gọi 5.5. Luật package khoá theo họ nên phải
có họ trước. Đây là đổi thứ tự thuần tuý — `crops` đã tồn tại từ bước 5 và đang
được truyền vào chính lời gọi 5.5 đó (`component_crops=crops`), nên không sinh
phụ thuộc mới.

**(b) Đưa lead detector lên trước bước phân package.** Ở `pipeline.py:476`,
`leads` chỉ đến từ lớp `pads`/`pins` của detector 22 lớp đang chạy. Detector mới
đang train là **một lớp `component`** — không sinh pads/pins ⇒ `leads` rỗng đúng
chỗ luật cần. Lead detector chuyên dụng (`models/active/lead_detector`) có sinh,
nhưng chạy ở **dòng 507, sau**. Quên chỗ này thì luật im lặng gán `unknown` cho
tất cả IC.

### 8.6. Classifier 6.1 có phải train lại không? **Không.**

Lo ngại hợp lý: luật khoá theo đầu ra của classifier, mà classifier đã đo được
**22,3% ca đổi nhãn khi đổi khung cắt**
([báo cáo box §7.2](../danh_gia/danh_gia_khoanh_box_than_linh_kien.md)). Nhãn
sai ⇒ áp nhầm luật.

Nhưng ở **độ mịn package**, phần lớn sai số đó **biến mất**:

| cặp đổi nhãn | số ca | cùng gói? |
|---|---:|---|
| `capacitor → led` | 91 | ✅ cả hai là chip 2 chân |
| `resistor → capacitor` | 33 | ✅ cả hai là chip 2 chân |
| `ic → resistor` | 12 | ❌ **khác gói — đây mới là ca hại** |

**124 trong 136 ca được nêu tên (91%) là vô hại** ở độ mịn này: luật chỉ cần
biết "chip 2 chân", mà cả `capacitor`, `led`, `resistor` đều dẫn về đúng gói đó.
Chỉ nhóm `ic ↔ thụ động` mới làm hỏng ROI.

**Kết luận:** không train lại 6.1 cho kế hoạch này. Việc phải làm là **đo lại
22,3% đó ở độ mịn package** — 56 ca còn lại chưa được phân loại trong báo cáo cũ
— và chỉ khi tỉ lệ *qua ranh giới gói* còn cao mới tính đến train lại. Đổi lại,
`ic ↔ thụ động` trở thành cặp phải theo dõi, thay cho cặp `4 ↔ 6` của bản cũ.

### 8.7. Cổng nghiệm thu

1. **Nhầm `ic` ↔ thụ động phải bằng 0** trên tập kiểm (thay cổng 4↔6 cũ).
2. **Đo lại ROI trên board thật** — `tests/data/solder_geometry`, 28 pad đếm
   tay: bật luật package phải **không giảm** độ phủ pad.
3. **Mặc định TẮT** cho tới khi vượt cổng 2 trên board của chính dây chuyền.

Cổng "macro recall ≥ 0,85" bỏ — đó là cổng của model. Thay bằng **tỉ lệ trúng
của luật trên tập kiểm gán tay** (§7).

### 8.8. Điều dữ liệu công khai KHÔNG kiểm được

Trong 24 kiểu vỏ của bộ winnies: mọi IC đều là **chân 2 bên** (SOT96, SOIC-12/
14/16, TSSOP-14/16) — **không có một QFP hay QFN nào**. Và không có tụ hoá trụ
đứng nào.

Tức **đúng hai phép chia mà kế hoạch này cần đều có 0 ví dụ công khai.** Ngưỡng
độ tròn và luật đếm cạnh chỉ đo được trên tập gán tay ở §7. Đây là lý do §7
không bỏ được, dù đã co từ 9.486 xuống 600–800 box.

---

## 9. Rủi ro, xếp theo khả năng xảy ra

1. **Bốn lớp không có dữ liệu công khai** (§6), mà ba trong số đó là ba lớp đáng
   giá nhất. Giảm thiểu: P3 gán riêng từ tile dự án; và nếu lớp 5/6 vẫn quá ít
   mẫu thì **gộp chúng thành một lớp "IC lớn — cần người xem"**, tức dùng như cờ
   chuyển review chứ không như lớp đo đạc. Thà thành thật là không biết.
2. **Nhãn package sai làm ROI tệ hơn đường đoán-bằng-pixel.** Giảm thiểu: cổng 2
   và 3 ở §8, mặc định tắt.
3. **Model học thuộc 73 ảnh Winnies.** Đúng cái đã xảy ra với detector (30/670
   ảnh chứa pads/pins ⇒ precision cao, recall 0,072). Giảm thiểu: chia theo bo,
   và **test phải là tile của dự án**, không phải Winnies.
4. **Đổi số lớp giữa chừng làm mất tiến độ đã duyệt** — `dataset_id` băm cả danh
   sách lớp. Giảm thiểu: quyết bộ lớp trước khi duyệt tiếp (§10 câu 1).
5. **Không có CAD ⇒ không có lưới an toàn thứ hai.** Hôm nay CAD là thứ duy nhất
   bắt được "nhãn nói 2 chân mà gói là IC". Sau kế hoạch này, package thay chỗ
   đó — nhưng nó là **một** nguồn, không phải hai nguồn kiểm chéo. Giảm thiểu:
   cờ `review` khi số ROI dựng được lệch hạng kỳ vọng (§4), để chỗ sai nổi lên
   thay vì lặn mất.

---

## 10. Câu hỏi cần bạn quyết

1. **Chốt 7 lớp ở §3 chứ?** Cần quyết **trước khi bạn duyệt tiếp vòng 2**, vì đổi
   bộ lớp sau đó sẽ phải chuyển 16 tile đã duyệt sang bộ mới.
2. **Tập kiểm 600–800 box phân tầng theo ĐẦU RA 6.1 (§7) — đồng ý chứ?** Phân
   tầng theo họ chứ không ngẫu nhiên, vì hai họ cần chia (`capacitor`, `ic`)
   là thiểu số và mẫu ngẫu nhiên sẽ gần như toàn chip 2 chân.
3. **Đồng ý đặt ở bước 5.2 thay vì sau 6.2 không?** Giữ ở cuối thì nó vẫn chạy
   nhưng **chỉ còn giá trị báo cáo**, không cải thiện ROI.
4. **Đồng ý đổi thứ tự pipeline hai chỗ chứ (§8.5)?** Đưa `classify_components`
   lên trước 5.5, và đưa lead detector lên trước bước phân package. Cả hai là
   đổi thứ tự thuần tuý, không sinh phụ thuộc mới.
5. **Lớp 6 (`ic_khong_chan`): chấp nhận kết luận "không kiểm được bằng ảnh 2D
   trên xuống" chứ?** Nếu dây chuyền có QFN/BGA mà vẫn phải kiểm, đó là bài toán
   X-quang, nằm ngoài phạm vi dự án.
6. **Xác nhận là sẽ KHÔNG có CAD chứ?** Nếu sau này có, tôi không phải bỏ gì cả —
   hai nguồn sẽ kiểm chéo nhau. Nhưng nếu chắc chắn không có, thì **cờ `review`
   khi lệch số chân kỳ vọng** (§4) trở thành bắt buộc chứ không còn là tuỳ chọn.
7. **BOM/pick-and-place của bạn có cột `footprint` (hay `package`/`pattern`)
   không?** Đây là câu hỏi rẻ nhất trong danh sách và đổi được cả thứ tự công
   việc: **có** thì làm bộ đọc footprint trước và hạ bộ luật package xuống
   ưu tiên thấp; **không** thì bộ luật lên đầu.
8. **Có xin được file IPC-D-356 từ bên gia công không?** Repo đọc được sẵn, và
   nó cho *từng pad một* — gần bằng có CAD. Đây là đường tắt lớn nhất còn lại.

---

Xem thêm: `Docs/bao_cao/tien_do_detect_2_luot.md` (bảng công việc sống),
`Docs/ke_hoach/ke_hoach_pcb_defect_toan_mach.md` (kế hoạch lỗi toàn mạch),
`datasets/public/README.md` (khảo sát nguồn ảnh công khai).
