# Kế hoạch phân nhóm package cho linh kiện

> Soạn 2026-08-31. **Đây là bản để bạn duyệt, chưa code gì cả.** Mọi con số có
> chữ *đo được* đều chạy ra từ dữ liệu trong repo và ghi rõ chạy trên cái gì;
> chỗ nào là suy đoán thì nói thẳng là suy đoán.

## 1. Kết luận nhanh

- **Nên làm, nhưng không phải để "phân loại chi tiết hơn".** Giá trị thật của
  package không nằm ở cái nhãn, mà ở chỗ nó nói cho bước 5.5 biết **linh kiện có
  mấy chân và chân nằm ở cạnh nào**. Hiện 5.5 phải *đoán điều đó từ pixel*, và
  đã có một ca đoán sai được ghi lại trong chính mã nguồn.
- **Vị trí đúng KHÔNG phải "luồng tiếp theo sau 6.2".** Đặt sau 6.2 thì nhãn
  package ra đời sau khi mọi ROI mối hàn đã được dựng xong — quá muộn để có
  tác dụng. Đề xuất: **bước 5.2, ngay sau bước 5 (cắt crop) và trước 5.5**.
  Chi tiết và các phương án khác ở §5.
- **Phạm vi hợp lý là nhỏ hơn bạn nghĩ.** Đo được: **86,5% linh kiện là loại 2
  chân** và đường 2-chân của 5.5 đã đúng sẵn. Chỉ **13,5% linh kiện** đi vào
  nhánh "nhiều chân" — nhưng chúng mang **31,2% tổng số mối hàn**. Tiền nằm ở
  nhóm 13,5% đó.
- **Ở 46 µm/px, đếm chân chỉ khả thi tới SOIC.** SOIC (bước chân 1,27 mm) =
  27,6 px/bước: đếm được. TSSOP (0,65 mm) = 14,1 px: sát mép. QFP 0,5 mm =
  10,9 px, QFN 0,4 mm = 8,7 px: **không đếm được**. Nên "IC 6 chân / IC 8 chân"
  như bạn ví dụ là **đúng và làm được** với SOIC/SOT, còn với QFP/QFN thì phải
  nhận dạng bằng *hình dáng gói*, không phải bằng cách đếm.
- **Dữ liệu đã có sẵn 16.632 box package-labelled ngay trong repo** —
  `datasets/public/pcb_packages_winnies/`, 24 lớp KIỂU VỎ. Nhưng chỉ từ **73 ảnh
  nguồn**, và 11/24 lớp dưới 110 instance. Đủ để bootstrap, không đủ để nghiệm thu.

---

## 2. Vì sao cần package: bước 5.5 đang đoán, và đoán bằng pixel

Đây là chỗ duy nhất trong pipeline mà nhãn package đổi được kết quả, nên nói kỹ.

`aoi_pipeline/config.py:30-50` giữ toàn bộ tri thức hiện có về hình dạng chân:

```python
TWO_TERMINAL_CLASSES = {"capacitor", "resistor", "diode", "led", "inductor", "fuse"}
PAD_ONLY_CLASSES     = {"pads"}
DEFAULT_TERMINAL_GEOMETRY = "multi_pin"   # ic, connector, transistor, relay,
                                          # switch, display, clock, buzzer, ...
```

Ba nhóm, và nhóm thứ ba là **cái sọt đựng tất cả những gì còn lại**. Hệ quả đo
được trên bộ Winnies (16.632 box, 24 kiểu vỏ):

| | số box | % linh kiện | mối hàn | % mối hàn |
|---|---:|---:|---:|---:|
| Vào nhánh `two_terminal` | 14.379 | **86,5%** | 28.758 | 68,8% |
| Vào nhánh `multi_pin` | 2.253 | **13,5%** | 13.017 | **31,2%** |

Và điều quan trọng nhất, cũng đo được: **2.247/2.253 gói trong nhánh `multi_pin`
chỉ có chân trên ĐÚNG 2 cạnh** (SOT, SOIC, TSSOP đều là gullwing hai cạnh dài).
Nhưng `_multi_pin_rects` dựng dải quanh **cả 4 cạnh**, rồi mới lọc bớt bằng năng
lượng pixel và độ đều của "cái lược" chân.

Tức là với gần như mọi linh kiện nhiều chân, pipeline **dựng thừa 2 dải trên nền
trống rồi nhờ pixel nói hộ dải nào là thật**. Phép nhờ đó có lúc sai — chính mã
nguồn ghi lại một ca:

> *"letting it run anyway is how D201 lost the single lead on its top edge: the
> silkscreen brackets in the band corners dragged the comb score to 0.938 against
> a 0.95 gate and the whole band went, real lead included."*
> — `aoi_pipeline/solder/geometry.py`

Biết package là biết trước câu trả lời: **SOIC-16 ⇒ 8 chân mỗi bên trên hai cạnh
dài, 2 cạnh ngắn là nền, chấm hết.** Không cần đo năng lượng, không có cửa cho
vệt silkscreen làm hỏng phép đo.

Ba thứ khác mà package cho không, hiện đang thiếu:

1. **Số chân kỳ vọng.** Đếm được 14 ROI trên một TSSOP-16 là *phát hiện thiếu 2
   mối hàn* — hiện không ai nói được con số kỳ vọng là bao nhiêu nên không ai
   phát hiện được.
2. **Tab tản nhiệt.** SOT-223 và DPAK có một pad to ở lưng, diện tích thiếc lớn
   gấp nhiều lần chân tín hiệu. Đưa nó qua cùng ngưỡng `solder_ratio` với một
   chân SOT-23 là so hai thứ khác nhau.
3. **Gói không có chân nhìn thấy (QFN/DFN/BGA).** Pad nằm dưới bụng. Dựng dải
   quanh chu vi cho chúng là dựng ROI trên nền. Đúng ra phải **không sinh ROI 2D**
   và đánh dấu "không kiểm được bằng ảnh trên xuống" — một câu trả lời trung thực
   thay vì một ROI vô nghĩa.

> **Ranh giới cần nói rõ:** package **không** làm mối hàn dễ chấm hơn. Nó làm ROI
> *nằm đúng chỗ* và *đủ số lượng*. Chấm tốt/xấu vẫn là việc của 6.2.

---

## 3. Bảng phân nhóm đề xuất

Quy ước: `họ` là 1 trong 16 lớp của bước 6.1 hiện tại. `T` = topology mà 5.5 sẽ
dùng. Cột "đọc được ở 46 µm/px" nói người gán nhãn **có phân biệt được bằng mắt
trên chính ảnh của dự án hay không**.

### 3.1 Nhóm 2 chân (86,5% linh kiện) — chia theo *hình dáng*, không theo chân

| Tên tiếng Việt | slug | họ | dấu hiệu nhận dạng bằng mắt | T | 46 µm/px |
|---|---|---|---|---|---|
| Chip chữ nhật | `chip_2t` | resistor, capacitor | hộp chữ nhật phẳng, 2 đầu bạc ở 2 cạnh ngắn | 2 đầu | ✅ |
| Tụ gốm (MLCC) | `mlcc` | capacitor | thân **nâu/be/xám ngà**, không chữ, bóng mờ | 2 đầu | ✅ |
| Điện trở chip | `res_chip` | resistor | thân **đen**, mặt trên có số/mã in trắng | 2 đầu | ✅ |
| Tụ hoá (can nhôm) | `elec_can` | capacitor | **trụ tròn cao**, nắp nhôm có rãnh chữ thập, vạch cực | nhiều chân¹ | ✅ |
| Tụ tantalum | `tantalum` | capacitor | hộp **vàng cam/nâu**, một đầu có **vạch cực đậm** | 2 đầu | ✅ |
| MELF (trụ nằm) | `melf` | resistor, diode | **hình trụ nằm ngang**, bóng, 2 mũ kim loại | 2 đầu | ✅ |
| Diode SOD | `sod` | diode | hộp nhỏ đen, **vạch catot** ở một đầu | 2 đầu | ✅ vạch: ⚠️ |
| LED | `led` | led | thân **trắng đục/trong**, có thấu kính, thường có góc vát | 2 đầu | ✅ |
| Cuộn cảm / hạt ferrite | `inductor_2t` | magnetic | khối **đen xám xù xì** hoặc có lõi dây quấn nhìn thấy | 2 đầu | ✅ |
| Cầu chì / polyfuse | `fuse_2t` | protection | hộp vuông, thường **xanh lá / cam nhạt** | 2 đầu | ✅ |

¹ Tụ hoá đứng có 2 chân nhưng pad rất to và **vỏ can chính là kim loại** — đây
là ca `axis_known` đã ghi trong `geometry.py`: phép đoán trục dựa vào kim loại
không phân biệt được "ngoài thân" với "thân". Vì vậy tách riêng chứ không gộp
vào `chip_2t`.

### 3.2 Nhóm nhiều chân (13,5% linh kiện, 31,2% mối hàn) — **đây là phần đáng tiền**

| Tên tiếng Việt | slug | họ | dấu hiệu nhận dạng | chân | cạnh mang chân | 46 µm/px |
|---|---|---|---|---:|---|---|
| SOT 3 chân | `sot23` | discrete_semiconductor | hộp đen nhỏ, **2 chân một bên, 1 chân bên kia** | 3 | 2 | ✅ |
| SOT 4–6 chân | `sot_multi` | discrete_semiconductor | như trên nhưng chân dày hơn, đếm được | 4–6 | 2 | ✅ |
| SOT có tab (SOT-223/DPAK) | `sot_tab` | discrete_semiconductor | 3 chân một bên + **một tab to bản** bên kia | 3+tab | 2 | ✅ |
| SOIC / SO | `soic` | ic | thân đen dài, **chân cánh chim thưa**, đếm được từng chân | 8–16 | 2 dài | ✅ 27,6 px/bước |
| TSSOP / SSOP | `tssop` | ic | như SOIC nhưng **mỏng hơn và chân dày hơn** | 14–28 | 2 dài | ⚠️ 14,1 px/bước |
| QFP | `qfp` | ic | thân **vuông**, chân ra **cả 4 cạnh** | 32–100+ | 4 | ❌ không đếm được |
| QFN / DFN | `qfn` | ic | thân vuông/chữ nhật, **không thấy chân nào**, mép nhẵn | — | **dưới bụng** | ❌ |
| BGA | `bga` | ic | khối vuông dày, **hoàn toàn không thấy chân** | — | dưới bụng | ❌ |
| Connector 1 hàng | `conn_1row` | connector | dãy chân **thẳng một hàng**, thân nhựa trắng/đen | 2–20 | 1 | ✅ |
| Connector 2 hàng | `conn_2row` | connector | hai hàng chân song song, thân nhựa cao | 4–40 | 2 | ✅ |
| Xuyên lỗ (THT) | `tht` | mọi họ | chân **xuyên qua lỗ**, thiếc thành vòng khuyên quanh lỗ | thay đổi | vòng | ✅ |
| Khối lớn có chân riêng | `block` | relay, magnetic, acoustic, switch_control, battery_power_input, display, timing | khối to, chân ít và ở vị trí không theo quy luật | thay đổi | thay đổi | ⚠️ |

### 3.3 Tự phê bình bảng trên

Ba nhóm dưới đây **tôi khuyên bỏ khỏi bộ đầu tiên**, nêu ra để bạn thấy lý do:

- **`qfp` / `qfn` / `bga`** — không đếm được chân ở 46 µm/px, và với QFN/BGA thì
  ảnh 2D trên xuống **về nguyên tắc** không nhìn thấy mối hàn. Nhưng vẫn phải
  **nhận ra** chúng, để 5.5 *ngừng sinh ROI giả* và đánh dấu "không kiểm được".
  Tức là chúng có giá trị như một **cờ từ chối**, không phải như một lớp đo đạc.
- **`mlcc` vs `res_chip`** — phân biệt bằng **màu**, mà bước 1 đang bật
  white-balance + CLAHE + normalize. Chính repo đã ghi nghi vấn lệch miền tiền
  xử lý này (`scripts/compare_preprocessing_ab.py`). Một lớp dựa vào màu là lớp
  dựa vào thứ chưa đo. **Phải chạy A/B tiền xử lý trước khi tin nhóm này.**
- **`sod` vạch catot / `tantalum` vạch cực** — vạch cực rộng ~0,2–0,3 mm ⇒
  4–6 px ở 46 µm/px. Nhìn thấy *có vạch* thì được; đọc *vạch ở đầu nào* để bắt
  lỗi ngược cực thì **chưa đủ pixel**. Đừng hứa kiểm tra phân cực ở độ phân giải
  hiện tại.

Ba lớp **quá hiếm để train nổi**, đo trên Winnies: `SOIC-12` (4 box / 2 ảnh),
`CHIP` (6/2), `SOIC-14` (8/4). Gộp `SOIC-12/14/16` thành một `soic` là bắt buộc,
không phải tuỳ chọn.

---

## 4. Bộ tối thiểu và bộ đầy đủ

**Bộ tối thiểu — 7 lớp, khuyến nghị làm trước:**

`chip_2t` · `elec_can` · `sot23` · `sot_tab` · `soic_tssop` · `no_lead` (QFN/DFN/BGA
gộp) · `conn_tht`

Vì sao 7: đây là **tập nhỏ nhất thay đổi được hình học 5.5**. Mỗi lớp trả lời
đúng một câu 5.5 đang phải đoán — trục ở đâu, chân ở mấy cạnh, có tab không, có
kiểm được bằng ảnh 2D không. Bảy lớp phủ khoảng 95% linh kiện trên bộ Winnies
*(ước tính từ bảng §2, không phải phép đo trực tiếp trên board dự án).*

**Bộ đầy đủ — 22 lớp ở §3.** Thêm được: phân biệt vật liệu (gốm/tantalum/MELF),
tách connector 1 hàng và 2 hàng, tách SOIC khỏi TSSOP.
Chi phí thêm: dữ liệu cho các lớp hiếm, và độ chính xác của mọi lớp dựa vào màu
phụ thuộc kết quả A/B tiền xử lý chưa chạy.

**Khuyến nghị: làm bộ tối thiểu trước, đo, rồi mới mở rộng.** Lý do rất cụ thể:
bảy lớp đó chỉ cần **hình dáng và số cạnh mang chân** — đều là thứ nhìn thấy
được ở 46 µm/px và không phụ thuộc màu.

---

## 5. Vị trí trong pipeline — vì sao **không** đặt sau 6.2

Thứ tự thật, đọc từ `aoi_pipeline/pipeline.py:568-578`:

```
4  detect_components()   → detections (có label họ của detector)
5  make_crops()          → crops            ← crop đã có ở đây
5.5 make_solder_crops()  → ROI mối hàn      ← dùng terminal_geometry(detection.label)
6.1 classify_components(crops)
6.2 grade_solder()
```

Điểm mấu chốt: **5.5 lấy topology từ nhãn của bước 4, và chạy TRƯỚC 6.1.** Nên:

| Phương án | Cái được | Cái mất |
|---|---|---|
| **(a) Bước riêng sau 6.2** *(đề xuất ban đầu của bạn)* | không đụng gì đang chạy | **nhãn ra đời sau khi ROI đã dựng xong ⇒ 0 tác dụng lên 5.5.** Chỉ còn giá trị thống kê/báo cáo |
| (b) Thêm head package vào detector bước 4 | không tốn thêm lần suy luận | phải train lại detector; mà detector đang là nút thắt riêng (8/10 board, chưa pack nổi dataset). Trộn hai việc khó vào một |
| (c) Đảo 6.1 lên trước 5.5 | tái dùng đúng model 6.1 | 6.1 phân loại **họ**, không phải package; vẫn phải thêm head. Và đảo thứ tự đụng vào `_invalidate_after` của UI |
| **(d) Bước 5.2: classifier package trên crop, giữa 5 và 5.5** ⭐ | crop **đã có sẵn** ở bước 5; 5.5 **đã nhận sẵn tham số `geometry=`**; không đảo thứ tự gì | thêm ~1 lần suy luận nhỏ mỗi linh kiện |
| (e) Bảng tra `họ → package` cứng | không cần model | không phân biệt được SOIC với QFN — hai thứ cùng họ `ic` mà topology ngược nhau. Đây đúng là cái đang sai hôm nay |

**Khuyến nghị: (d), bước 5.2.**

Điều làm phương án này rẻ bất ngờ: `derive_solder_joints()` **đã có sẵn đường
vào**. Nguyên văn docstring:

> *"``frame`` and ``geometry`` let a caller that knows better override what the
> box alone can say. CAD fusion uses them to keep this exact ROI geometry while
> anchoring it on a registered placement and a real pad count."*

Nghĩa là cơ chế "một nguồn bên ngoài biết rõ hơn cái hộp" **đã được thiết kế và
đã có một khách hàng là CAD fusion**. Package classifier chỉ là khách hàng thứ
hai, đi đúng cửa đó. Việc phải làm: mở rộng `terminal_geometry()` từ 3 giá trị
lên bộ topology mới, thêm `PadProfile` cho từng loại, và cho 5.5 ưu tiên
`geometry` do 5.2 cung cấp trên `geometry` suy từ nhãn detector.

**Lập luận mạnh nhất CHỐNG lại đề xuất của chính tôi:** thêm 5.2 là thêm một
model nữa vào một pipeline đã có 5 model, trong đó **2 model chưa đạt** (detector
recall 0.52 macro; classifier 6.2 chưa dùng để quyết được). Một classifier
package sai sẽ đưa topology sai vào 5.5 và làm ROI **tệ hơn** đường đoán-bằng-pixel
hiện tại. Vì vậy §8 đặt cổng nghiệm thu theo đúng lệ của repo: **phải hơn thứ nó
thay thế, đo trên board thật, mặc định TẮT cho tới khi hơn.**

---

## 6. Dữ liệu: có sẵn bao nhiêu, thiếu bao nhiêu

Đo trực tiếp từ `datasets/public/pcb_packages_winnies/export_yolov8_v3.zip`
(nguồn package-labelled **duy nhất** repo đang có):

| gói | box | ảnh nguồn | cạnh ngắn (trung vị) | đủ train? |
|---|---:|---:|---:|---|
| resistor | 5.989 | 72 | 26,1 px | ✅ |
| capacitor | 5.406 | 70 | 28,6 px | ✅ |
| LED | 916 | 28 | 24,9 px | ✅ |
| SOT23 | 671 | 44 | 48,5 px | ✅ |
| SOD323 | 636 | 27 | 31,0 px | ✅ |
| feriet kraal | 490 | 43 | 30,0 px | ✅ |
| SOT96 (SO-8) | 405 | 50 | 66,4 px | ✅ |
| SOT753 (SOT23-5) | 334 | 37 | 72,3 px | ✅ |
| Resistor rond (MELF) | 305 | 19 | 33,1 px | ⚠️ ít ảnh |
| SOT457 · SOD123 · Polyfuse_Z/GR | 179–227 | 10–22 | 34–50 px | ⚠️ |
| SOIC-16 · MOSFET-2 · SOT143 · SOT223 · MOSFET · TSSOP-16 | 65–115 | 10–18 | 59–179 px | ❌ dưới 120 |
| SOD128 · TSSOP-14 · SOIC-14 · CHIP · SOIC-12 | **4–33** | **2–6** | — | ❌❌ |

**Ba hạn chế phải nói thẳng:**

1. **Chỉ 73 ảnh nguồn.** 173 file là do Roboflow sinh biến thể lật/xoay. Chia
   train/val theo file là rò rỉ — đúng cái bẫy đã làm accuracy 6.2 phồng từ
   89,9% lên 97,65%. Phải gom theo ảnh nguồn.
2. **Sai miền.** Winnies không phải camera/ánh sáng của dây chuyền bạn. Giống hệt
   tình trạng model lượt 2 (`bootstrap_only` trong manifest của nó).
3. **Không có QFP/QFN/BGA trong bộ này.** Ba lớp mà tôi khuyên dùng làm *cờ từ
   chối* lại **không có một mẫu nào**. Phải tự gán nhãn từ tile của dự án.

Nguồn thứ hai đã có sẵn: **RF100 `printed-circuit-board`**, 177 cảnh — nhưng nhãn
ở mức **họ**, không phải package. Dùng được làm ảnh, không dùng được làm nhãn.

---

## 7. Kế hoạch gán nhãn

Tin tốt: **không phải vẽ lại box.** 1.595 box thân linh kiện bạn đã duyệt là
**vị trí**; package chỉ thêm một **nhãn lớp** lên box đã có. App gán nhãn hiện đã
hỗ trợ nhiều lớp và có phím tắt `1`–`9`, nên thao tác là *bấm một phím trên mỗi
box* chứ không phải vẽ.

| Giai đoạn | Việc | Số box | Ước tính |
|---|---|---:|---|
| P0 | Dựng app gán nhãn 7 lớp, seed bằng đoán từ họ + tỉ lệ cạnh | — | code, ~nửa ngày |
| P1 | Gán package cho 1.595 box đã duyệt | 1.595 | **1,5–3 giờ** ở 4–7 giây/box |
| P2 | Gán tiếp khi duyệt nốt vòng 2 (104 tile còn lại) | ~9.000 | gộp **cùng một lượt** với việc duyệt thân |
| P3 | Bổ sung riêng QFP/QFN/BGA từ tile có IC lớn | ~200 | ~1 giờ |

**Điểm quan trọng nhất về thứ tự:** dự án đang nợ **ba** việc gán nhãn — thân
linh kiện (lượt 1), chân/mối hàn (lượt 2), và giờ là package. **Package phải đi
chung một lượt với thân linh kiện**, vì cùng nhìn một crop và cùng một cái box.
Tách ra là bắt người duyệt xem lại đúng những ảnh đó lần thứ hai.

⚠️ Nhưng vòng 2 hiện đang chạy với **một lớp `component`**. Đổi sang 7 lớp giữa
chừng sẽ đổi `dataset_id` ⇒ **mất localStorage của 16 tile đã duyệt**. Nên phải
quyết **trước khi bạn duyệt tiếp** (câu hỏi 1 ở §10).

**Bootstrap để chỉ phải sửa thay vì gán:** đoán package từ `(họ 6.1, tỉ lệ cạnh,
diện tích, số dải chân 5.5 tìm được)`. Với nhóm 2 chân, "hộp nhỏ tỉ lệ ~2:1 +
họ resistor/capacitor" gần như chắc chắn là `chip_2t`, và nhóm đó là 86,5% số
box — nên phần lớn công việc có thể điền sẵn đúng. Con số "điền đúng bao nhiêu %"
**tôi chưa đo được** và không nên đoán; đo bằng cách gán tay 100 box rồi so.

---

## 8. Kế hoạch train

- **Model:** một classifier ảnh nhỏ trên crop — cùng khuôn với 6.1. Đề xuất
  `efficientnet_b0` hoặc `mobilenet_v3_small`, input **128×128** (crop trung vị
  chỉ 17 px cạnh ngắn; 224 chỉ là phóng to nhiễu). Không cần detector: vị trí đã
  có từ bước 4.
- **Chia tập theo BOARD**, không theo crop, và gom biến thể Roboflow về ảnh nguồn.
- **Notebook** theo đúng khuôn `training/kaggle/` hiện có, xuất ONNX +
  `model_manifest.json`, đăng ký ô model mới `models/active/package/`.
- **Cổng nghiệm thu — theo đúng lệ đã có với model lượt 2:**
  1. Macro recall ≥ 0,85 trên test chia theo board.
  2. **Đo lại ROI trên board thật** (`tests/data/solder_geometry`, 28 pad đã đếm
     tay): bật 5.2 phải **không giảm** độ phủ pad so với đường hiện tại. Đây là
     cổng thật; con số ở mục 1 chỉ là điều kiện cần.
  3. Nhầm lẫn **`soic` ↔ `qfn`** phải bằng 0 trên test — đây là cặp nhầm duy nhất
     làm ROI *tệ đi thật* (dựng dải chân trên gói không có chân).
  4. **Mặc định TẮT** (`_NO_AUTO_ADOPT`) cho tới khi vượt cổng 2 trên board của
     chính dây chuyền.
- Khi chưa có model: 5.2 là **no-op tuyệt đối**, 5.5 chạy y như hôm nay.

---

## 9. Rủi ro, xếp theo khả năng xảy ra

1. **Nhãn package sai làm ROI tệ hơn đường đoán-bằng-pixel.** Giảm thiểu: cổng 2
   ở §8 và mặc định tắt. Đây là rủi ro số một và nó có thật.
2. **Lớp dựa vào màu chết vì tiền xử lý.** `mlcc` vs `res_chip` phân biệt bằng
   màu, mà bước 1 bật 5 phép biến đổi màu chưa từng được đo A/B. Giảm thiểu:
   chạy `compare_preprocessing_ab.py --isolate` **trước**, hoặc bỏ hai lớp này
   khỏi bộ tối thiểu (đã bỏ).
3. **Model học thuộc 73 ảnh Winnies.** Đúng cái đã xảy ra với detector (30/670
   ảnh chứa pads/pins ⇒ precision cao, recall 0,072). Giảm thiểu: chia theo
   board, và **test phải là tile của dự án**, không phải Winnies.
4. **Đổi số lớp giữa chừng làm mất tiến độ đã duyệt.** `dataset_id` băm cả danh
   sách lớp. Giảm thiểu: quyết bộ lớp **trước** khi duyệt tiếp (§10 câu 1).
5. **Lớp hiếm không bao giờ đủ mẫu.** 5 lớp dưới 35 box. Giảm thiểu: gộp
   (`SOIC-12/14/16` → `soic`) và chấp nhận bộ tối thiểu.

---

## 10. Câu hỏi cần bạn quyết

1. **Bộ tối thiểu 7 lớp hay bộ đầy đủ 22 lớp?** *(Tôi khuyên 7.)* Cần quyết
   **trước khi bạn duyệt tiếp vòng 2**, vì đổi bộ lớp sau đó sẽ mất 16 tile đã
   duyệt.
2. **Gán package chung một lượt với thân linh kiện, hay tách thành lượt riêng
   sau?** *(Tôi khuyên chung — tách ra là xem lại cùng những ảnh đó lần hai.)*
3. **Đồng ý đặt ở bước 5.2 thay vì sau 6.2 không?** Nếu bạn muốn giữ nó ở cuối
   như ý ban đầu, nó vẫn chạy được nhưng **chỉ còn giá trị báo cáo**, không cải
   thiện ROI — tôi cần biết bạn chọn cái nào.
4. **QFN/BGA: chấp nhận kết luận "không kiểm được bằng ảnh 2D trên xuống" chứ?**
   Nếu dây chuyền có loại này và vẫn phải kiểm, thì đó là bài toán X-quang, nằm
   ngoài phạm vi dự án.
5. **Có cần phân cực (ngược chiều tụ/diode) trong phạm vi này không?** Ở 46 µm/px
   vạch cực chỉ 4–6 px — tôi khuyên **để riêng**, gắn với đợt nâng cấp camera
   25 µm/px, không nhét vào đợt package này.

---

Xem thêm: `Docs/tien_do_detect_2_luot.md` (bảng công việc sống),
`Docs/ke_hoach_pcb_defect_toan_mach.md` (kế hoạch lỗi toàn mạch),
`datasets/public/README.md` (khảo sát nguồn ảnh công khai).
