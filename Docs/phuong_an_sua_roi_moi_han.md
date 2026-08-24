# Phương án cho ba lỗi trong "Đánh giá model detect solder"

> Đọc `Docs/ĐÁNH GIÁ MODEL DETECT SOLDER VÀ TÌM HƯỚNG GIẢI QUYẾT.docx`
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

Ba triệu chứng, nhưng **hai trong ba cùng một gốc**, và cái gốc đó khiến hai
hướng xử lý bạn đề xuất trong file không đi tới đâu — tôi đã thử cả hai và đo
được là chúng hỏng.

| Triệu chứng trong file | Nguyên nhân đo được |
|---|---|
| Tụ tròn: box mở quá lớn, không box nào đúng | ROI sâu **0,75 × span**. Với tụ 6,2 mm ra ROI sâu **4,6 mm** — to hơn cả linh kiện. Cộng thêm box gần vuông (tỉ lệ 1,09) nên thuật toán không chốt được trục và phát **cả hai** cặp |
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

## 2. Ba hướng đã thử, cả ba đều hỏng

Phần này quan trọng hơn phần đề xuất, vì nó loại bỏ đúng hai hướng ghi trong
file docx.

### 2.1. "Tiền xử lý ảnh để tách viền / thân linh kiện ra màu khác" — không được

Đây là hướng bạn ghi cho cả mục Capacitor và mục Diode. Nó không đi tới đâu vì
**ba thứ cần tách vốn cùng một màu**: mối hàn, chữ lụa trắng và vỏ nhôm đều là
sáng + ít bão hoà. Bảng ở mục 1 là số đo trực tiếp: S = 47,8 (lụa) so với
S = 48,0 (mối hàn). Không có phép biến đổi màu nào tách được hai giá trị bằng
nhau.

### 2.2. Tách bằng kết cấu (specular / phương sai cục bộ) — **hỏng, và hỏng ngược**

Đây là hướng của tôi, không có trong file. Lập luận: mối hàn phản xạ gương, chữ
lụa thì mờ. Đo ra thì ngược hẳn:

| Vùng | Phương sai Laplacian | p99/p50 của V |
|---|---|---|
| **Chữ lụa `HDL01`** | **16 004** | 1,78 |
| Chữ lụa `R239` | 13 681 | 2,28 |
| Chân IC thật | 4 737 | 2,73 |
| Mối hàn D201 | 5 205 | 1,12 |
| Thân can nhôm | 3 068 | 1,00 |

Chữ lụa có kết cấu **cao gấp ba** chân thật, vì nó là chữ nét sắc trên nền tối.
Lọc theo kết cấu sẽ giữ chữ và bỏ chân — tệ hơn hiện tại.

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
Docs/cad_pads_template.csv         designator,pin,x_mm,y_mm,width_mm,height_mm,…
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

| Linh kiện | span | ROI sâu hiện tại |
|---|---|---|
| Tụ hoá C239 | 6,2 mm | **4,6 mm** |
| Tụ hoá C231/C232 | 4,8 mm | 3,6 mm |
| Điện trở chip 0805 | 1,4 mm | 1,1 mm |

Thêm trần theo mm (đề xuất **1,5 mm**, cần thang µm/px đã có ở bước hiệu chuẩn
camera). Đo trên chính ảnh này:

> **5/36** linh kiện two-terminal bị cắt bớt — đúng 3 con tụ hoá và 2 điện trở
> lớn. **31 con còn lại không đổi một pixel nào**, vì chúng vốn đã dưới trần.

Đây là tính chất cần có ở một bản sửa: nó chỉ cắn vào đúng chỗ luật cũ vô lý.

**Lưu ý:** P1 sửa được "box quá lớn", **không** sửa được "chọn sai trục". Hai
việc khác nhau.

### P2 — Bộ lọc dải bằng chỉ số lược, làm tín hiệu phụ

Ghép `độ đều` (mục 2.4) vào cạnh `lead_band_energy_ratio` hiện có, và **chỉ áp
dụng khi dải có ≥3 đốm** — dưới 3 thì chỉ số vô nghĩa, giữ nguyên hành vi cũ.
Bắt được dải chữ `HDL01` (0,87) và viền D201 (0,73) mà không đụng cạnh 2 chân.

Chưa đủ để làm cổng chặn một mình; cần đo trên vài chục dải nữa trước khi bật
mặc định.

### P3 — Train bộ định vị chân của lượt 2

Đúng kế hoạch đã có ở `Docs/ke_hoach_fine_tune_cuc_bo.md`. Ba phép đo hỏng ở
mục 2 chính là lập luận mạnh nhất cho việc này: câu hỏi "đâu là chân" không
giải được bằng ngưỡng thủ công trên board này.

Board trong file docx là **ứng viên gán nhãn tốt nhất** hiện có — nó chứa cả ba
ca khó (tụ hoá tròn, SOT-23, SOIC cạnh chữ lụa) trong một ảnh.

## 5. Việc tôi đề nghị làm ngay

1. **P1** — thêm trần mm, có test giữ lại số đo 5/36 ở trên.
2. Xin file **pad CSV** của board này. Có nó thì P0 chạy được ngay và hai mục
   còn lại trong file docx tự hết.
3. Dùng mục *Đánh giá model* để đánh dấu các ROI sai trên chính ảnh này — vừa
   là dữ liệu cho P3, vừa là mốc để đo P1/P2 có thật sự tốt lên không.

Chưa nên làm: sửa `segment_solder` theo hướng màu hoặc kết cấu. Mục 2.1 và 2.2
đã đo là hai ngõ cụt.
