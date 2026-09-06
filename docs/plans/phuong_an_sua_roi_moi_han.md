# Phương án cho ba lỗi trong "Đánh giá model detect solder"

> Đọc `docs/evaluation/ĐÁNH GIÁ MODEL DETECT SOLDER VÀ TÌM HƯỚNG GIẢI QUYẾT.docx`
> (2026-08-24) rồi **dựng lại đúng kết quả đó** trên ảnh nguồn để đo, chứ không
> phán đoán từ ảnh chụp màn hình.
>
> Ảnh nguồn: `00001__1024__1648___4120.png` — tìm ra bằng template matching với
> ảnh gốc nhúng trong file docx, khớp **1.000**. Chạy lại qua đường ống hiện tại
> ra đúng ba hiện tượng bạn chụp.
>
> Thang đo dùng suốt bài: **22 px/mm** (≈45 µm/px), đo từ pitch chân SOIC-8
> 1,27 mm = 28 px.

## Tóm tắt trước khi đi vào chi tiết

Ba triệu chứng, nhưng **hai trong ba cùng một gốc**. Ba trong bốn hướng tôi thử
đều hỏng — kể cả một hướng của chính tôi; hướng còn lại (xử lý ảnh nhiều đặc
trưng) thì **có tác dụng thật**, và đó là chỗ tôi đã kết luận sai lúc đầu, xem
mục 2.1.

| Triệu chứng trong file | Nguyên nhân đo được |
|---|---|
| Tụ tròn: box mở quá lớn, không box nào đúng | ROI sâu **0,75 × chiều dài**. Với tụ dài 6,7 mm ra ROI sâu **5,0 mm** — to hơn cả linh kiện. Cộng thêm box gần vuông (tỉ lệ 1,09) nên thuật toán không chốt được trục và phát **cả hai** cặp |
| Diode: nhận nhầm viền hai bên là mối hàn | Dải chu vi nằm trên **chữ lụa trắng** sống sót qua bộ lọc năng lượng |
| IC: lấn sang chữ OCR | **Cùng một gốc với dòng trên** — dải cạnh phải nằm trên chữ `HDL01`, rồi bị tách thành 7 "chân" giả |

Gốc chung: hàm `segment_solder` nhận diện kim loại bằng **sáng + ít bão hoà
màu**. Trên board này có ba thứ cùng chữ ký đó.

## 1. Số đo: chữ lụa trắng "kim loại" hơn cả chân IC thật

Đo `segment_solder` trên các mảng cắt từ chính ảnh đó:

| Vùng | Tỷ lệ "kim loại" | H, S, V trung bình |
|---|---|---|
| **Chữ lụa `HDL01`** | **49,5 %** | 105,3 · 47,8 · 147,8 |
| Mối hàn thật của D201 | 52,7 % | 97,2 · 48,0 · 163,6 |
| **Chân IC thật (hàng trên)** | **38,3 %** | 105,1 · 56,5 · 135,8 |
| **Chân IC thật (hàng dưới)** | **29,9 %** | 104,5 · 63,3 · 142,7 |
| **Thân can nhôm tụ hoá** | **60,3 %** | 118,1 · 93,6 · 194,9 |
| Pad của chính con tụ đó | 64,0 % | 119,3 · 73,1 · 207,2 |
| Nền phủ xanh | 18,4 % | 48,7 · 91,1 · 109,0 |

Hai dòng cần đọc kỹ:

- **Chữ lụa (49,5 %) ghi điểm cao hơn chân IC thật (38,3 % và 29,9 %).** Bộ
  lọc năng lượng không thể loại nó — dưới thước đo đang dùng, nó *giống kim
  loại hơn cả kim loại*. Bão hoà màu gần như trùng khít: 47,8 so với 48,0.
- **Thân can nhôm (60,3 %) gần bằng pad của nó (64,0 %).** Đây không phải lỗi
  ngưỡng: vỏ tụ hoá **là kim loại trần thật**. Không ngưỡng màu nào tách được
  hai thứ cùng vật liệu.

Chú thích trong code hiện ghi *"Green mask, dark component bodies and silkscreen
all fail one of those"*. Vế về silkscreen **sai** — số đo ở trên là bằng chứng.

## 2. Bốn hướng đã thử: ba hỏng, một được

Phần này quan trọng hơn phần đề xuất. Nó cũng chứa **một đính chính lớn** với
bản đầu của tài liệu này.

### 2.1. "Tiền xử lý ảnh để tách viền / thân linh kiện ra màu khác"

> **Đính chính 2026-08-24, sau khi bạn phản biện.** Mục này ban đầu tôi viết là
> "không được". **Kết luận đó sai**, và phép đo của tôi lúc đó quá yếu: tôi so
> **trung bình cả mảng**, mà mảng chữ lụa lại gồm cả nền tối xen giữa các nét.
> Đo lại ở mức từng pixel thì bức tranh khác hẳn — xem mục 2.1b.

Phần đúng của kết luận cũ: **một ngưỡng đơn lẻ thì không tách được.** Thử hết
11 đặc trưng một chiều, cái tốt nhất (`b*`) chỉ đạt **65,2 %** trên bài toán hai
lớp cân bằng, và độ bão hoà thì trùng khít: S = 47,8 (lụa) so với 48,0 (hàn).
Đó là lý do `segment_solder` hiện tại — vốn là một ngưỡng đôi trên (sáng, bão
hoà) — không có cách nào loại được chữ lụa.

### 2.1b. Nhưng nhiều đặc trưng cùng lúc thì **tách được** — bạn đúng

Lấy 7 đặc trưng (H, S, V, L\*, a\*, b\*, độ lệch chuẩn cục bộ 5×5) và một bộ
phân loại phi tuyến:

| Phép thử | Độ chính xác |
|---|---|
| Ngưỡng đơn lẻ tốt nhất | 65,2 % |
| 7 đặc trưng, trộn lẫn pixel cùng vùng | 85,7 % |
| **7 đặc trưng, kiểm trên vùng CHƯA TỪNG THẤY** | **78,8 %** |

Phép thử thứ ba là phép thử thật (train trên 3 vùng lụa + 3 vùng hàn, đo trên
cặp vùng bị giữ lại). 78,8 % nghĩa là nó **học được quy luật**, không phải học
thuộc ánh sáng cục bộ.

Có một điều đáng lo trong chi tiết: nó nhận **chữ lụa rất tốt (88–97 %)** nhưng
nhận **mối hàn kém (61–74 %)**. Ở mức từng pixel, cái giá của việc loại chữ lụa
là ném đi một phần tư đến một phần ba mối hàn thật — đúng cái chiều nguy hiểm.

**Nhưng quyết định cần lấy không phải ở mức pixel, mà ở mức dải.** Gộp hàng
trăm pixel lại thì sai số ngẫu nhiên trung bình hoá bớt:

| Dải | Có chân? | `segment_solder` hiện tại | Lọc 7 đặc trưng |
|---|---|---|---|
| SOIC-8 trên | **có** | 24 % | **45 %** |
| C239 cạnh trên | **có** | 40 % | **51 %** |
| SOIC-8 phải = chữ lụa | không | 21 % | **6 %** |
| SOIC-8 trái | không | 11 % | 13 % |
| D201 trái = viền lụa | không | 26 % | **2 %** |
| D201 phải | không | 25 % | 20 % |
| C239 cạnh trái | không | 20 % | 7 % |

Thước đo hiện tại: dải thật 24–40 %, dải giả 11–26 % — **chồng lấn**, không
ngưỡng nào tách được (chữ lụa 26 % còn cao hơn dải thật 24 %). Thước đo 7 đặc
trưng: dải thật 45–51 %, dải giả 2–20 % — **có khoảng trống thật**.

**Cái giá:** đó là một **bộ phân loại đã train**, không phải một luật. Tôi thử
tìm luật giải thích được thay cho nó — ngưỡng trên S, trên `b*`, và kết hợp hai
cái — **cả ba đều không tách được** (mọi khoảng cách đều âm trên 9 dải). Nên
muốn dùng cách này thì phải chấp nhận nó là một model nhỏ, cần dữ liệu gán nhãn
và cần kiểm lại khi đổi board hoặc đổi đèn.

### 2.1c. Và tiền xử lý hiện tại đang **làm hỏng** chính thông tin đó

Đo tỉ lệ pixel sáng bị chạy (≥254 ở ít nhất một kênh):

| | nét lụa | mối hàn |
|---|---|---|
| **Ảnh gốc** | 50,3 % | 53,1 % |
| **Sau tiền xử lý của dự án** | 87,2 % | **100,0 %** |

Sau CLAHE + normalize + sharpen, **100 % pixel sáng của mối hàn thành (255,255,
255)** — trắng tinh, không phân biệt được với mực lụa trắng. Không có phép xử lý
nào sau đó lấy lại được dữ liệu đã bị cắt.

Nên câu trả lời cho "sao không dùng xử lý ảnh" là: **xử lý ảnh đang là thứ làm
nó tệ hơn.** Việc rẻ nhất và có căn cứ nhất là **đo kim loại trên ảnh gốc**, chứ
không phải trên ảnh đã tiền xử lý — đo được 85,7 % so với 81,8 %.

### 2.2. Tách bằng **riêng** kết cấu (specular / phương sai cục bộ) — hỏng ngược

Đây là hướng của tôi, không có trong file. Lập luận: mối hàn phản xạ gương, chữ
lụa thì mờ. Dùng **một mình** thì ngược hẳn:

| Vùng | Phương sai Laplacian | p99/p50 của V |
|---|---|---|
| **Chữ lụa `HDL01`** | **16 004** | 1,78 |
| Chữ lụa `R239` | 13 681 | 2,28 |
| Chân IC thật | 4 737 | 2,73 |
| Mối hàn D201 | 5 205 | 1,12 |
| Thân can nhôm | 3 068 | 1,00 |

Chữ lụa có kết cấu **cao gấp ba** chân thật, vì nó là chữ nét sắc trên nền tối.
Lọc *chỉ* theo kết cấu sẽ giữ chữ và bỏ chân — tệ hơn hiện tại.

Lưu ý: độ lệch chuẩn cục bộ vẫn là **một trong 7 đặc trưng** ở mục 2.1b và ở đó
nó có ích. Cái sai là dùng nó một mình.

### 2.3. Chọn trục bằng vành kim loại **ngoài** hộp — hỏng 2/3

Ý tưởng: đừng đo kim loại ở dải chồng lên thân can, chỉ đo ở vành mỏng 1,5 mm
bên ngoài hộp, rồi chọn trục nào nhiều kim loại hơn.

| Tụ | trên | dưới | trái | phải | Trục chọn được | Đúng? |
|---|---|---|---|---|---|---|
| C239 | 13 % | 28 % | 9 % | 16 % | dọc (trên–dưới) | **đúng** |
| C232 | 27 % | 35 % | 50 % | 28 % | ngang | **sai** |
| C231 | 21 % | 40 % | 48 % | 44 % | ngang | **sai** |

Lý do hỏng nhìn thấy ngay trên ảnh: C232 và C231 có **điện trở chip nằm sát hai
bên**, và mối hàn của hàng xóm rơi vào đúng vành đang đo. Vành ngoài không phân
biệt được "chân của tôi" với "chân của người bên cạnh".

### 2.4. Chỉ số "lược" (chân là N đốm cách đều) — có tín hiệu, nhưng chưa đủ

Đo trên 8 dải chu vi:

| Dải | Có chân thật? | Số đốm | Độ đều |
|---|---|---|---|
| SOIC-8 trên | có | 4 | **0,99** |
| SOIC-8 dưới | có | 4 | **0,98** |
| SOIC-8 phải = **chữ lụa** | không | 5 | 0,87 |
| D201 trên | có | 3 | **0,97** |
| D201 trái = **viền lụa** | không | 3 | 0,73 |
| D201 dưới | có | **2** | 1,00 |
| SOIC-8 trái | không | **2** | 1,00 |

Tín hiệu là thật (0,97–0,99 so với 0,73–0,87). Nhưng luật "≥3 đốm và độ đều
≥0,95" **bỏ nhầm cạnh chỉ có 2 chân** — mà `min_pins_per_band = 2` tồn tại
đúng vì SOT-23/SOT-223 có cạnh 2 chân, và chính D201 là một con như vậy. Với 2
đốm thì "độ đều" luôn bằng 1,00 và vô nghĩa.

Kết luận: dùng được như **một tín hiệu phụ**, không dùng được làm cổng chặn.

## 3. Một đính chính về chẩn đoán trong file

**Mục Diode.** File ghi *"classifier nhận nhầm con này thành con IC"*. Đúng —
nhưng chưa đủ: **cả detector cũng nhầm**. Đo được:

| | detector | classifier |
|---|---|---|
| D201 | `ic` 0.69 | `ic` 0.50 → *unknown* |
| D202 | `ic` 0.85 | `ic` 0.84 → *accept* |

Hai tầng đồng thuận vào cùng một câu trả lời sai, nên **không thể lấy tầng này
sửa cho tầng kia**. Điểm sáng duy nhất: D201 đã bị hạ xuống `unknown`, tức
đường ống có ngờ.

Nhưng nhìn kỹ ảnh thì **`multi_pin` lại không sai**: D201 là gói SOT-23 **3
chân** (1 pad trên, 2 pad dưới), không phải diode 2 đầu. Cả 3 pad thật đều đã
được khoanh đúng. Cái sai chỉ là **2–3 ROI thừa trên chữ lụa**. Nói cách khác
sửa nhãn cũng không cứu được mục này — vẫn phải sửa bộ lọc dải.

**Mục IC.** File ghi *"box mở bị lệch sang bên phải"*. Đo lại: hộp là
x 731–843, còn các ROI giả nằm ở **x 849–872**, tức **ngoài hộp**. Dải chu vi
vốn được thiết kế để với ra ngoài hộp một đoạn `outer` — nó không lệch, nó đang
làm đúng việc của nó. Thứ đáng lẽ phải chặn là bộ lọc năng lượng, và mục 1 cho
thấy vì sao nó không chặn được. Thu hộp lại cũng không chắc giải quyết, vì dải
vẫn với ra ngoài.

## 4. Phương án

Xếp theo **tỉ lệ ăn chắc trên công bỏ ra**, không theo thứ tự trong file.

### P0 — Nạp toạ độ pad từ CAD / pick-and-place *(việc đã dựng xong, chưa dùng)*

Đây là câu trả lời thật cho cả ba triệu chứng, và nó **đã nằm sẵn trong dự án**:

```
aoi_pipeline/solder/cad.py         CadPad, BoardCad.has_pads, pad_count
aoi_pipeline/solder/cad_fusion.py  _cad_pad_joints, fuse_solder_joints
docs/design/cad_pads_template.csv         designator,pin,x_mm,y_mm,width_mm,height_mm,…
```

Có file pad thì ROI **lấy thẳng từ toạ độ pad thật**, không suy từ hộp nữa. Khi
đó: tụ tròn hết đoán trục (pad ghi rõ ở đâu), chữ lụa hết bị nhận nhầm (không
có pad nào ở đó), và IC có đúng 8 ROI.

Mọi thứ ở mục 2 là **đoán từ pixel**, và ba phép đo đã cho thấy pixel trên board
này không đủ dữ kiện để đoán. Nút thắt không phải thuật toán — là **thiếu dữ
liệu đầu vào**.

**Việc cần làm:** xuất pad từ CAD của board (Altium/KiCad đều xuất được
`.csv`/IPC-356), đúng cột như file mẫu. Ô *Dữ liệu tham chiếu* trên sidebar đã
nhận file này.

### P1 — Giới hạn độ sâu ROI theo **mm**, thay vì theo tỉ lệ span *(rẻ, đo được)*

Luật hiện tại `terminal_inner + terminal_outer = 0,75 × span` giả định pad to
lên theo linh kiện. **Pad không như thế** — nó là kích thước vật lý gần cố định.

| Linh kiện | chiều dài | ROI sâu hiện tại |
|---|---|---|
| Tụ hoá C239 | 6,7 mm | **5,0 mm** |
| Tụ hoá C231/C232 | 5,4 mm | 4,0 mm |
| Điện trở chip 0805 | 2,1 mm | 1,6 mm |

Trần **2,0 mm**, không phải 1,5. Ở 1,5 mm trần bắt đầu cắn cả chip 0805 (7/39
linh kiện bị cắt, có những con vốn không sai gì). Ở 2,0 mm:

> **6/39** linh kiện bị cắt — đúng những con to vô lý. **33 con còn lại không
> đổi một pixel nào**, vì chúng vốn đã dưới trần. Tụ C239 đi từ 110 px xuống
> 45 px và hai ROI về đúng hai pad.

Đây là tính chất cần có ở một bản sửa: nó chỉ cắn vào đúng chỗ luật cũ vô lý.

**Lưu ý:** P1 sửa được "box quá lớn", **không** sửa được "chọn sai trục". Việc
đó là của P0.

### P2 — Bộ lọc dải bằng chỉ số lược, làm tín hiệu phụ

Ghép `độ đều` (mục 2.4) vào cạnh `lead_band_energy_ratio` hiện có, và **chỉ áp
dụng khi dải có ≥3 đốm** — dưới 3 thì chỉ số vô nghĩa, giữ nguyên hành vi cũ.
Bắt được dải chữ `HDL01` (0,87) và viền D201 (0,73) mà không đụng cạnh 2 chân.

Chưa đủ để làm cổng chặn một mình; cần đo trên vài chục dải nữa trước khi bật
mặc định.

### P3 — Train bộ định vị chân của lượt 2

Đúng kế hoạch đã có ở `docs/plans/ke_hoach_fine_tune_cuc_bo.md`. Ba phép đo hỏng ở
mục 2 chính là lập luận mạnh nhất cho việc này: câu hỏi "đâu là chân" không
giải được bằng ngưỡng thủ công trên board này.

Board trong file docx là **ứng viên gán nhãn tốt nhất** hiện có — nó chứa cả ba
ca khó (tụ hoá tròn, SOT-23, SOIC cạnh chữ lụa) trong một ảnh.

## 5. Đã làm — và kết quả đo được

P0, P1, P2 đã triển khai (`ad89694`, `86c74fd`). P3 vẫn là kế hoạch.

### Trên con SOIC-8 trong file docx

| | tổng ROI | ROI nằm trên chữ lụa |
|---|---|---|
| Trước | 15 | **7** |
| Sau P1+P2 | **8** — đúng 8 chân | **0** |

### Trên tụ hoá C239

| | số ROI | độ sâu | đúng trục? |
|---|---|---|---|
| Trước | 4 (hai cặp mơ hồ) | 110 px = 5,0 mm | không |
| Sau P1+P2 | 4 | 34–45 px = 1,5–2,0 mm | không |
| **Sau P0+P1+P2** *(có pick-and-place)* | **2** | **45 px = 2,0 mm** | **có** |

### P0 chỉ cần pick-and-place, không cần pad CSV

Đây là điều tôi hiểu sai lúc đầu và đã kiểm lại: `_reanchored_derived_joints`
vốn đã dùng `component.rotation`, nên **một file pick-and-place là đủ** để chốt
trục. Cái còn thiếu là đường ống vẫn *đoán lại* trục sau đó; giờ có `axis_known`
để nói rằng trục đã biết thì đừng đoán.

### Hai lỗi im lặng bắt được trong lúc làm

1. **Phép chiếu dùng sai khung.** Chỉ số lược chiếu theo kích thước của `rect`
   (khung cục bộ của linh kiện) thay vì của mảng cắt (khung ảnh). Hai khung
   hoán vị trục khi linh kiện xoay, nên dải `lead_left` của SOIC-8 cắt ra mảng
   112×34 nằm ngang lại bị chiếu theo chiều dọc — không đời nào thấy được cái
   lược. Đây là lý do lần chạy đầu chỉ giảm được 7 → 1 ROI giả; sau khi sửa là
   7 → 0.
2. **`FusionConfig` mang một `SolderJointConfig` RIÊNG.** `from_mapping` nối hai
   cái lại nhưng `PipelineConfig()` dựng trực tiếp thì không, và đường CAD đọc
   đúng cái nằm trong `fusion`. Thiếu chỗ nối này thì trần mm chỉ có tác dụng
   khi **không** nạp CAD — tức đúng lúc cần nó nhất thì lại không có.

## 6. Còn lại

**Chưa sửa được: D201/D202** vẫn còn 3 và 1 ROI trên viền lụa. Dải đó dưới 3
đốm nên chỉ số lược cố tình im lặng — và đó là lựa chọn có chủ đích, vì cạnh 2
chân là chuyện thường. Cách xử lý là P0 (pick-and-place cho đúng số chân qua
`footprint`) hoặc P3.

**Việc rẻ nhất còn chưa làm, và có căn cứ nhất:** đo kim loại trên **ảnh gốc**
thay vì ảnh đã tiền xử lý. Mục 2.1c đo được tiền xử lý làm chạy sáng 100 % pixel
mối hàn; chuyển sang ảnh gốc là 85,7 % so với 81,8 % mà không cần train gì.

**Cần từ bạn:** file **pick-and-place thật** của board. File dùng trong phép đo
ở mục 5 là file *sinh ra* từ chính vị trí detect được — nó kiểm được đường ống
có dùng đúng góc xoay không, chưa kiểm được toạ độ thật có khớp không.
