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
- `[ ]` **CẦN NGƯỜI GÁN NHÃN:** 3.847/3.855 box hiện còn `unknown`; 8 prelabel
  chỉ là gợi ý bảo thủ. `unknown` không phải lớp thứ tám và chặn export.
- `[ ]` **CẦN TRAIN MODEL:** sau khi nhãn hoàn tất, chạy
  `pack_package_classification_dataset.py`, notebook
  `pcb_package_classification_kaggle.ipynb`, rồi
  `evaluate_package_roi_gate.py`. Chỉ promote thủ công nếu toàn bộ gate đạt.

Các lựa chọn triển khai đã chốt theo khuyến nghị của kế hoạch: đúng 7 lớp,
package đi cùng lượt duyệt thân, đặt ở 5.2, package ẩn chân là không kiểm được
bằng ảnh 2D trên xuống, footprint làm trước và model chỉ là lưới cuối.

## 1. Kết luận nhanh

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

## 7. Kế hoạch gán nhãn

**Không phải vẽ lại box.** 1.595 box thân linh kiện bạn đã duyệt là **vị trí**;
package chỉ thêm một **nhãn lớp** lên box đã có. App đã hỗ trợ nhiều lớp và có
phím tắt `1`–`9` ⇒ thao tác là **bấm một phím trên mỗi box**, không phải vẽ.

Với 7 lớp thì `1`–`7` phủ đúng hết, không cần cuộn menu.

| Giai đoạn | Việc | Số box | Ước tính |
|---|---|---:|---|
| P0 | Dựng app 7 lớp, điền sẵn đoán từ họ + tỉ lệ cạnh + diện tích | — | code, ~nửa ngày |
| P1 | Gán package cho 1.595 box đã duyệt | 1.595 | **1,5–3 giờ** (4–7 s/box) |
| P2 | Gán tiếp khi duyệt nốt 104 tile còn lại của vòng 2 | ~9.000 | **gộp cùng một lượt** với việc duyệt thân |
| P3 | Bổ sung riêng lớp 2/5/6/7 — lọc tile có box lớn | ~200 | ~1 giờ |

**Thứ tự quan trọng nhất:** dự án đang nợ **ba** việc gán nhãn — thân linh kiện
(lượt 1), chân/mối hàn (lượt 2), và package. **Package phải đi chung một lượt
với thân linh kiện**: cùng nhìn một crop, cùng một cái box. Tách ra là bắt bạn
xem lại đúng những ảnh đó lần thứ hai.

⚠️ **Nhưng vòng 2 đang chạy với một lớp `component`.** Đổi sang 7 lớp sẽ đổi
`dataset_id` ⇒ **mất localStorage của 16 tile đã duyệt**. Nên phải quyết **trước
khi bạn duyệt tiếp** (câu hỏi 1 ở §10). Nếu bạn chọn đổi, tôi sẽ chuyển 16 record
đó sang bộ mới bằng đúng đường carry-forward đã dùng cho vòng 2 — nó có chốt
semantic SHA-256 nên mất mát sẽ báo lỗi chứ không im lặng.

**Bootstrap để chỉ phải sửa thay vì gán:** với lớp 1 (86,5% số box), "hộp nhỏ tỉ
lệ ~2:1 + họ resistor/capacitor" gần như chắc chắn đúng, nên phần lớn công việc
điền sẵn được. Tỉ lệ điền đúng **tôi chưa đo được** và không nên đoán — đo bằng
cách gán tay 100 box rồi so.

---

## 8. Kế hoạch train

- **Model:** một classifier ảnh nhỏ trên crop — cùng khuôn với 6.1. Đề xuất
  `efficientnet_b0` hoặc `mobilenet_v3_small`, input **128×128** (crop trung vị
  chỉ 17 px cạnh ngắn; 224 chỉ phóng to nhiễu). Không cần detector: vị trí đã có
  từ bước 4.
- **Chia tập theo BO**, không theo crop; gom biến thể Roboflow về ảnh nguồn.
- **Notebook** theo khuôn `training/kaggle/` hiện có; xuất ONNX +
  `model_manifest.json`; ô model mới `models/active/package/`.
- **Cổng nghiệm thu — không dùng CAD làm ground truth** (§4):
  1. Macro recall ≥ 0,85 trên test chia theo bo.
  2. **Đo lại ROI trên board thật** — `tests/data/solder_geometry`, **28 pad đếm
     tay**: bật 5.2 phải **không giảm** độ phủ pad so với đường hiện tại. Đây là
     cổng thật; mục 1 chỉ là điều kiện cần.
  3. Nhầm **lớp 4 ↔ lớp 6** (`ic_hai_ben` ↔ `ic_khong_chan`) phải **bằng 0** trên
     test. Đây là cặp nhầm duy nhất làm ROI *tệ đi thật*: dựng dải chân trên một
     gói không có chân, hoặc bỏ ROI của một gói có chân.
  4. **Mặc định TẮT** (`_NO_AUTO_ADOPT`) cho tới khi vượt cổng 2 trên board của
     chính dây chuyền — đúng như ô `lead_detector` đang làm.
- Khi chưa có model: 5.2 là **no-op tuyệt đối**, 5.5 chạy y như hôm nay.

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
2. **Gán package chung một lượt với thân linh kiện, hay tách lượt riêng sau?**
   *(Tôi khuyên chung.)*
3. **Đồng ý đặt ở bước 5.2 thay vì sau 6.2 không?** Giữ ở cuối thì nó vẫn chạy
   nhưng **chỉ còn giá trị báo cáo**, không cải thiện ROI.
4. **Lớp 6 (`ic_khong_chan`): chấp nhận kết luận "không kiểm được bằng ảnh 2D
   trên xuống" chứ?** Nếu dây chuyền có QFN/BGA mà vẫn phải kiểm, đó là bài toán
   X-quang, nằm ngoài phạm vi dự án.
5. **Xác nhận là sẽ KHÔNG có CAD chứ?** Nếu sau này có, tôi không phải bỏ gì cả —
   hai nguồn sẽ kiểm chéo nhau. Nhưng nếu chắc chắn không có, thì **cờ `review`
   khi lệch số chân kỳ vọng** (§4) trở thành bắt buộc chứ không còn là tuỳ chọn.
6. **BOM/pick-and-place của bạn có cột `footprint` (hay `package`/`pattern`)
   không?** Đây là câu hỏi rẻ nhất trong danh sách và đổi được cả thứ tự công
   việc: **có** thì làm bộ đọc footprint trước và hạ classifier package xuống
   ưu tiên thấp; **không** thì classifier lên đầu.
7. **Có xin được file IPC-D-356 từ bên gia công không?** Repo đọc được sẵn, và
   nó cho *từng pad một* — gần bằng có CAD. Đây là đường tắt lớn nhất còn lại.

---

Xem thêm: `Docs/bao_cao/tien_do_detect_2_luot.md` (bảng công việc sống),
`Docs/ke_hoach/ke_hoach_pcb_defect_toan_mach.md` (kế hoạch lỗi toàn mạch),
`datasets/public/README.md` (khảo sát nguồn ảnh công khai).
