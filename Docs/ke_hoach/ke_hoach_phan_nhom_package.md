# Phân nhóm package cho linh kiện

Viết lại toàn bộ 2026-09-03 — bản trước là ba lượt sửa chồng lên nhau nên không
đọc được. **Cập nhật gần nhất 2026-09-05 sau review lượt 3 của Codex**, lượt
đầu tiên tìm ra lỗi bằng code chạy được chứ không bằng đọc tài liệu: §8.3, §9.0,
§7.2b, §10.4 là mới hoặc viết lại.

---

## 1. Một trang tóm tắt

**Mục tiêu.** Cho bước 5.5 biết **chân nằm ở cạnh nào** và **có nhìn thấy chân
hay không**, để nó thôi phải đoán bằng pixel.

> Bản trước viết "biết linh kiện có **mấy chân**". Đó là nói quá, và Codex chỉ
> đúng: bảy lớp chỉ cho biết topology tương đối — `ic_hai_ben` phủ 6–256 chân,
> `ic_bon_ben` phủ 8–512, `connector` phủ 1–512. **Số chân thật vẫn phải lấy từ
> footprint/CAD hoặc đếm bằng pixel.**

**Hướng đã chốt.** Dùng **luật hình học**, không train model package. Luật
**nối sau** classifier 6.1: 6.1 cho *họ*, luật chỉ chia nhỏ *bên trong* một họ.

**Vị trí.** Bước **5.2**, giữa bước 5 (cắt crop) và 5.5.

**Đang ở đâu.**

| | trạng thái |
|---|---|
| Thứ tự pipeline (6.1 và lead detector chạy trước 5.2) | ✅ xong, có test canh |
| Bộ luật cho họ `ic` + 5 họ ánh xạ thẳng | ✅ xong, mặc định TẮT |
| Chốt an toàn cho `ic_khong_chan` và QFP nửa vời | ✅ xong 2026-09-04 — §8.1, §8.2 |
| Cổng nghiệm thu cho đường luật | ✅ **đã sửa 2026-09-05** — dùng chung đường runtime, §9.0b |
| Cạnh chân đo được → 5.5 | ⚠️ **vẫn bị vứt đi**, nhưng đã chặn không cho đặt ROI sai cạnh — §8.3 |
| Nhánh `ic` của luật | ⚠️ **chạy được với pad đếm tay**, chưa chạy được với chân pass-2 — §9.0c |
| Chia họ `capacitor` (trụ đứng ↔ chip) | ❌ **CHƯA có luật** — phép đo cũ sai phạm vi, §6.3b |
| Tập kiểm gán tay | ⏳ **750 box đã tiền gán, chờ bạn duyệt** — §6.7 |
| Ánh xạ họ → gói cho các họ còn lại | ⏳ đang quá rộng — §6.5 |

> ⚠️ **Cập nhật 2026-09-05 (review lượt 3).** Hai lỗi code đã được **tái hiện
> bằng mẫu tổng hợp**, không phải suy đoán: cổng nghiệm thu không nhìn thấy
> thay đổi hình học do luật gây ra (§9.0), và cạnh chân mà luật đo được không
> đi tới bước dựng ROI (§8.3).
>
> **Cả hai đã sửa cùng ngày** (§9.0b, §8.3), và việc sửa cổng làm lộ thêm hai
> lỗ hổng cùng họ: `--families model` chưa từng chạy được, và "luật không chạy
> được" in ra giống hệt "luật chạy rồi bỏ qua". Cổng chạy lại cho **0 mất pad ở
> cả hai chế độ** — nhưng nhánh `ic` **vẫn chưa được kiểm lần nào**, nên luật
> giữ mặc định **TẮT**. Chi tiết và ba lý do ở §9.0b.

**Việc kế tiếp cần bạn:** duyệt 750 nhãn tiền gán ở §6.7 — và **ưu tiên 103
mẫu họ `capacitor`**, vì đó là chỗ phép đo đang mỏng nhất (§6.3b).

**Việc kế tiếp của tôi:** chờ bạn quyết bước 6 ở §10.4 — pass 2 trả mối hàn
**đè lên thân**, và đó mới là thứ chặn nhánh `ic` trên dây chuyền. Fixture mới
không sửa được (§9.0c). Bước 1–3 và 5 đã xong.


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
  - **Hai chốt an toàn đã cài (§8.1, §8.2):** không suy `ic_khong_chan` từ sự
    vắng mặt của chân, và không nhận `ic_hai_ben` khi thân gần vuông. Cả hai
    chặn ca *hỏng im lặng* — ROI vẫn dựng, chỉ là dựng thiếu.
    **Chốt thứ ba (§8.3, 2026-09-05):** cặp cạnh có chân phải khớp cặp cạnh
    DÀI của thân, nếu không thì 5.5 đặt ROI sang đúng hai cạnh không có chân.
  - 15 test, phần lớn canh những ca luật phải trả về **rỗng**.
- Thứ tự pipeline (`99b962f`): 6.1 chạy trước 5.2/5.5, lead detector chạy trước
  5.2. `tests/test_pipeline_stage_order.py` canh cả hai — trước đó không test
  nào canh thứ tự và cả 1087 test đều xanh với thứ tự sai.
- `scripts/survey_package_taxonomy.py` — khảo sát dữ liệu công khai (§6.1–6.5).
- `scripts/build_family_package_review_set.py` — dựng tập kiểm 750 box (§6.7).
- **Sửa một lỗi báo cáo:** `run()` từng giữ danh sách `package_classifications`
  cũ trong biến cục bộ trong khi luật ghi kết quả vào
  `self.last_package_classifications` bên trong `make_solder_crops`. Chỉ chạy
  luật thì báo cáo nói "0 kết quả 5.2" **trong khi ROI đã bị đổi**. Codex phát
  hiện; giờ có test canh — gỡ bản vá ra thì test đỏ, còn trước đó cả 1128 test
  đều xanh với lỗi này.
- Hạ tầng cũ vẫn dùng được: parser footprint BOM/PnP/CAD, bảy topology ở 5.5,
  đường ONNX 5.2 (no-op tuyệt đối khi thiếu artifact), ô model
  `models/active/package_classifier/` với `_NO_AUTO_ADOPT`.

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

### 6.3. Phép đo `capacitor` của bản trước ĐO SAI BÀI TOÁN

> **Sửa 2026-09-04.** Bảng dưới đây trong bản trước dùng lớp
> `Electrolytic Capacitor` của FPIC làm đại diện cho "tụ trụ đứng". Mở
> `contact_sheets/fpic__Electrolytic_Capacitor.png` ra xem thì lớp đó **gần
> như toàn tantalum dán hình chữ nhật màu cam**, chỉ lác đác lon tròn. Mà
> tantalum dán là `hai_chan` theo đúng định nghĩa ở §2.
>
> Nên con số 85,8% ấy tách **nhãn FPIC này với nhãn FPIC kia**, không tách
> `tru_dung` với `hai_chan`. Codex chỉ ra chỗ này và kiểm lại thì đúng.

Bảng cũ, giữ lại để không ai đo lại lần nữa:

| đặc trưng | ngưỡng | cân bằng | ghi chú |
|---|---:|---:|---|
| `long_side` | ≥ 43 px | 85,8% | **không dùng được — sai target** |
| `circularity` | ≥ 0,683 | 53,9% | gần như đoán bừa |

### 6.3b. Đo lại trên nhãn tay — và một lỗi phạm vi của chính bản trước

> **Sửa 2026-09-05.** Bản trước ghi **97,0%** ở đây và đặt dấu ✅ vào §1. Con số
> đó **sai phạm vi**, Codex chỉ ra ở lượt review thứ hai và kiểm lại thì đúng.

Luật chỉ chạy khi **họ = `capacitor`**. Nhưng phép đo cũ so 82 `tru_dung` với
**toàn bộ 312** box `hai_chan` — mà trong 312 đó chỉ **21 box thuộc họ
`capacitor`**; phần còn lại là `XEM_KY` (265), resistor (18), timing (7),
diode (1). Tức nó đo một bài toán dễ hơn nhiều so với bài toán luật phải giải.

Cùng ngưỡng `aspect < 1,69 và area ≥ 1200 px²`, đo trên hai phạm vi:

| phạm vi | n | recall trụ | recall chip | cân bằng |
|---|---:|---:|---:|---:|
| mọi `hai_chan` *(bản trước — sai)* | 82/312 | 97,6% | 96,5% | **97,0%** |
| **chỉ họ `capacitor`** *(phạm vi thật)* | 82/**21** | 97,6% | **61,9%** | **79,7%** |

Hiệu chuẩn lại ngưỡng trên đúng phạm vi thì tốt nhất chỉ đạt **89,1%**
(`aspect < 1,24`, bỏ hẳn điều kiện diện tích) — và con số đó khớp trên **21
mẫu chip**, quá ít để tin.

**Kết luận: chưa cài luật `capacitor`.** Cần bạn duyệt ít nhất **103 mẫu họ
`capacitor`** (82 trụ + 21 chip) trước, và nên có thêm mẫu chip.

**Kết luận cũ về độ tròn vẫn SỐNG SÓT** — 66,5%, gần như đoán bừa. Lý do khác
phỏng đoán ban đầu: không phải Otsu hỏng, mà tụ hoá nhìn từ trên vốn không tròn
— nắp có rãnh chữ thập, mép lon phản sáng, contour bắt vào những thứ đó.

### 6.3c. Nhận ra tụ tròn vẫn CHƯA biết đặt ROI ở đâu

Ngay cả khi tách đúng `tru_dung`, thân **tròn thì không có trục dài** — mà 5.5
đặt hai ROI ở hai đầu trục. Runtime đã biết điều này và tự đánh dấu:
`aoi_pipeline/solder/geometry.py:221` gắn cờ **`package_axis_assumed`** đúng
cho trường hợp gói thẳng đứng + thân gọn + chưa biết trục.

Nên `tru_dung` chỉ nên phát ra khi **biết được hướng**: footprint/PnP, golden
recipe đã duyệt, hoặc lead detection thấy hai pad thật. Không biết hướng thì
đừng dựng hai ROI theo một trục đoán — giữ đường bảo toàn recall hiện tại.

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

### 6.5. Taxonomy — đã có bằng chứng, không còn phải đoán

| họ 6.1 | cần chia? | gói | căn cứ |
|---|---|---|---|
| `resistor`, `led`, `diode` | ⚠️ | `hai_chan` | đã cài, **nhưng ánh xạ đang quá rộng** — xem dưới |
| `discrete_semiconductor` | ⚠️ | `goi_nho` | đã cài, cùng vấn đề |
| `connector` | ❌ | `connector` | đã cài |
| `magnetic`, `protection`, `timing`, `acoustic` | ⏳ | **chưa ánh xạ** | không lớp nào trong bảy lớp tả đúng |
| `relay`, `display`, `switch_control`, `battery_power_input` | ⏳ | **chưa ánh xạ** | hộp lớn nhiều chân, chưa có lớp |
| `capacitor` | ❌ chưa | `tru_dung` / `hai_chan` | phép đo cũ sai phạm vi, §6.3b |
| `ic` | ✅ | 2 bên / 4 bên / không chân | vị trí chân, §8 |

**Bằng chứng cho các ánh xạ thẳng mỏng hơn nhiều so với vẻ ngoài.** Đếm trên
750 box tiền gán:

| ánh xạ đang cài | số mẫu ủng hộ | số mẫu phản bác |
|---|---:|---:|
| `connector` → `connector` | **86** | 0 |
| `resistor` → `hai_chan` | 18 | 0 |
| `diode` → `hai_chan` | **1** | 0 |
| `discrete_semiconductor` → `goi_nho` | **1** | 0 |
| `led` → `hai_chan` | **0** | 0 |

Chỉ `connector` có đủ mẫu để nói gì đó. `led` **không có lấy một mẫu nào** —
ánh xạ đó hiện chỉ dựa vào trực giác, đúng loại căn cứ đã tạo ra luật bị gỡ ở
`73ce2aa`.

**Ánh xạ họ đang quá rộng** *(Codex chỉ ra, và nhãn tay xác nhận)*. Không phải
mọi thành viên của một họ đều cùng một gói:

- `resistor` → còn có mạng điện trở nhiều chân, và điện trở cắm đứng;
- `led` → còn có LED RGB nhiều chân;
- `diode` → còn có cầu diode 4 chân;
- `discrete_semiconductor` → SOT, SO-8, DPAK, TO đều khác nhau;
- `connector` → nhãn tay tìm thấy khe PCI, jack RCA, cổng SCART, FPC, header
  hai hàng, đế cắm CPU — **code đang dựng một kiểu ROI duy nhất cho tất cả**.

Chưa sửa. Sửa đúng cách là thêm điều kiện hình học **bên trong** mỗi họ, giống
họ `capacitor` ở §6.3b, và mỗi điều kiện phải đo trước khi bật.

**KHÔNG còn hỏi bạn đặt tên 4 cụm IC nữa.** Bản trước nhờ bạn đối chiếu
`cluster_sheets/fpic__IC__cluster_*.png`. Codex xem cả bốn tờ và kết luận mỗi
cụm vẫn trộn SOT, SOIC, QFP/BGA lẫn crop nhiễu, cùng một loại còn nằm ở nhiều
cụm. KMeans ở đó là **thăm dò, không phải bằng chứng taxonomy** — nên đừng
dùng nó để quyết gói, và đừng tốn công đặt tên cho nó.

### 6.6. Việc mới từ phía detector: 15/62 thân không có nhãn họ

Detector thân linh kiện chỉ có một lớp `component`, nên nhãn dùng cho
`terminal_geometry()` phải đến từ 6.1 (`AOIPipeline.apply_family_labels`).
Đo trên `tests/data/solder_geometry`: **15 trong 62 thân** không nhận được
nhãn — 6.1 trả `review`/`unknown`, hoặc trả `false_crop_background`.

Chúng rơi về `multi_pin`, tức dựng dải quanh cả 4 cạnh. Đúng vấn đề §3 mô
tả, chỉ là nay có một nguồn mới. `run()` đã đếm và cảnh báo số này thay vì
im lặng.

**Đã đo 2026-09-04, và kết quả bác một lý lẽ của chính tài liệu này.**

Đối chiếu 364 box có nhãn họ gán tay với dự đoán của 6.1, tách theo tầng:

| tầng | n | đúng **họ** | đúng **hình học chân** |
|---|---:|---:|---:|
| `accept` | 263 | 88,2% | **95,1%** |
| `review` | 64 | 54,7% | 68,8% |
| `unknown` | 37 | 40,5% | 83,8% |

Độ tin **có** xếp hạng được ở mức họ (88,2 > 54,7 > 40,5). Nhưng ở mức hình học
chân — thứ 5.5 thật sự dùng — thứ tự **vỡ**: `unknown` hơn `review`. Ngưỡng
đang hiệu chuẩn cho *độ mịn họ*, còn thứ dùng nó cần *độ mịn topology*.

Hai chính sách cho 101 box bị cổng từ chối:

| | **bỏ ROI** (lọt lưới) | thừa ROI | đúng hình học |
|---|---:|---:|---:|
| **hiện tại** — lùi về `multi_pin` | **6** | 53 | 83,8% |
| dùng nhãn ở mọi tầng | **19** | 20 | 89,3% |

Bỏ cổng **làm tổng độ chính xác tăng**, nhưng gấp ba số ca bỏ ROI. Với AOI thì
đó là sai chiều — **giữ cổng**. Việc đáng làm là **hiệu chuẩn lại ngưỡng cho
miền mới**, không phải bỏ cổng hay train lại 6.1.

> Một lý lẽ trong `package_rules.py` bị chính số liệu này bác: `multi_pin`
> **không** phải mặc định an toàn miễn phí. Trong 101 box đó, **46 box thật sự
> là `two_terminal`** — đẩy chúng sang `multi_pin` là dựng dải quanh cả 4 cạnh
> của linh kiện 2 chân, đúng cái bệnh §3 mô tả. Nó an toàn theo nghĩa *không bỏ
> sót*, không an toàn theo nghĩa *đặt ROI đúng chỗ*.

Chi tiết ở [đánh giá 6.1 §7](../danh_gia/danh_gia_classifier_6_1.md).

### 6.7. Tập nhãn tay 750 box — 2026-09-04

`python scripts/build_family_package_review_set.py --out datasets/survey/family_package_review_20260904`

750 box phân tầng trên **34 bo**, cắt kèm lề rộng để nhìn thấy chân, xếp thành
33 tờ lưới. Nhãn **gán bằng mắt, không dùng 6.1** — dùng model để điền nhãn cho
tập dùng để đo chính model đó thì phép đo tự xác nhận chính nó, và người duyệt
bị neo theo nhãn có sẵn.

> ⚠️ **Bạn CHƯA duyệt.** Mọi con số rút từ tập này là tạm.

| gói | n | | họ | n |
|---|---:|---|---|---:|
| `hai_chan` | 312 | | `ic` | 131 |
| `connector` | 86 | | `capacitor` | 104 |
| `ic_hai_ben` | 84 | | `connector` | 86 |
| **`tru_dung`** | **82** | | `resistor` | 18 |
| **`ic_bon_ben`** | **33** | | `timing`/`relay` | 14 |
| `ngoai_taxonomy` | 19 | | còn lại | 11 |
| `goi_nho` | 1 | | **chưa chắc** | **386** |
| **chưa chắc** | **133** | | | |

**Điều này lấp đúng hai lỗ mà §7.1 nói là không nguồn công khai nào có**: 82 ví
dụ `tru_dung` (tụ hoá nhìn từ trên, nắp có rãnh chữ thập) và 33 ví dụ
`ic_bon_ben` (QFP thật — nhiều con đọc được mã `TMS320LC31**PQ**40`, PQ chính
là plastic quad flat pack).

Hai phép đo đã rút ra từ đây, cả hai đều lật một khẳng định cũ:

1. **§6.3b** — phép đo FPIC cũ sai target, và bản thay thế đầu tiên của tôi
   cũng sai phạm vi (97,0% -> **79,7%** khi lọc đúng họ `capacitor`). Chưa
   có luật nào cài được cho họ này.
2. **`ic_bon_ben` gần vuông, `ic_hai_ben` thuôn dài** — aspect trung vị **1,03**
   so với **1,95**. Bản trước viết "các IC trong bộ dữ liệu này gần vuông hết,
   không dài như trực giác"; điều đó đúng với bộ winnies và **sai trên bo dự
   án**. Con số này giờ là chốt an toàn trong `package_rules.py`.

`ngoai_taxonomy` (19 ca) là relay, module nguồn, pin cúc áo, tản nhiệt, chiết
áp xoay — **không lớp nào trong bảy lớp tả đúng chúng**. Đây là bằng chứng thật
cho câu hỏi 2 ở §9.

---

## 7. Còn thiếu gì

### 7.1. Không nguồn công khai nào có đủ

Trong 24 kiểu vỏ của winnies: **mọi IC đều là chân 2 bên** — không một QFP hay
QFN nào. fpic có `IC` nhưng ở mức họ, không nói chân nằm đâu.

> **Đã lấp một phần bằng nhãn tay (§6.7):** 33 ví dụ `ic_bon_ben` và 82 ví dụ
> `tru_dung`, trên chính bo của dự án. Hai phép đo quan trọng nhất của tài liệu
> này (§6.3b và ngưỡng aspect ở §8.2) đều rút từ đó, chứ không còn từ dữ liệu
> công khai.

**Còn thiếu hẳn: `ic_khong_chan`.** Không nguồn nào — công khai lẫn nhãn tay —
có một ví dụ QFN/BGA nào. Đó là lý do §8.1 **cấm suy lớp này từ ảnh**.

### 7.2. Tập kiểm gán tay: 600–800 box

Không phải để train, mà để **đo tỉ lệ trúng của luật**. Lấy phân tầng chứ không
ngẫu nhiên — lớp 1 chiếm 86,5% nên mẫu đều sẽ toàn lớp 1:

| tầng | box | vì sao |
|---|---:|---|
| thân lớn, vuông (ứng viên lớp 4/5/6) | ~250 | cặp nhầm `ic ↔ thụ động` phải bằng 0 |
| thân dài, nhiều chân (ứng viên lớp 7) | ~150 | connector, bước chân lớn |
| thân nhỏ (ứng viên lớp 1/2/3) | ~250 | đo ngưỡng kích thước ở §6.3 |
| ngẫu nhiên nền | ~100 | bắt ca luật chưa nghĩ tới |

**Gán thêm nhãn HỌ cho cùng những box đó.** Cùng một lượt công việc trả lời
thêm được một câu hiện chưa ai biết: **độ chính xác thật của 6.1 trên miền ảnh
của dự án**. Đo được là accept của nó tụt 94,9% → 70,3% khi đổi sang ảnh
PCB-DSLR *với cùng một detector*, nhưng đó là độ TIN, không phải độ ĐÚNG — và
không có nhãn họ thì không đo được độ đúng. Xem
[danh_gia_classifier_6_1.md](../danh_gia/danh_gia_classifier_6_1.md).

Không phải vẽ lại box — vị trí đã có trong 9.486 box đã duyệt, package chỉ thêm
một nhãn lớp, app có phím tắt `1`–`7`.

#### 7.2b. Chia HIỆU CHỈNH ↔ NGHIỆM THU — thêm 2026-09-05

Tập 750 box (§6.7) đang bị dùng cho **cả hai** việc: tìm ngưỡng *và* chứng minh
ngưỡng đó tốt. Ngưỡng aspect 1,3 ở §8.2 rút thẳng từ 117 IC trong chính tập
này, rồi §8.2 lại lấy chính tập này để báo "bắt được 64% QFP". Con số đó là
**tỉ lệ trúng trên tập đã dùng để chọn ngưỡng**, nên nó lạc quan.

> **Không train model không có nghĩa là không overfit.** Mỗi hằng số trong
> `PackageRuleConfig` — `two_sided_min_aspect`, `min_leads_per_edge`,
> `circularity_threshold` khi đo được — đều là một tham số khớp bằng tay. Ba
> tham số khớp trên 117 mẫu đủ để khớp nhiễu.

Quy tắc chia, chốt **trước** khi đo lại:

| | tập | dùng làm gì |
|---|---|---|
| **Hiệu chỉnh** | ~24 bo | chọn mọi ngưỡng. Được nhìn thoải mái. |
| **Nghiệm thu** | ~10 bo, **khoá** | chỉ chạy SAU khi ngưỡng đã đóng băng |

- **Chia theo BO, không theo box.** Hai box cùng một bo có cùng ánh sáng, cùng
  độ phóng đại, cùng người gán — chia theo box thì tập nghiệm thu chỉ đo lại
  tập hiệu chỉnh. Dự án đã chia theo bo ở detector và 6.1; giữ nguyên nguyên
  tắc đó.
- **Cố định danh sách bo nghiệm thu vào file**, không chọn lại mỗi lần đo.
- **Ngưỡng đóng băng trước, đo sau.** Đo rồi chỉnh ngưỡng rồi đo lại trên cùng
  tập nghiệm thu thì tập đó thành tập hiệu chỉnh thứ hai.

Và phải đo **hai đường, báo cáo riêng**:

| đường | nguồn nhãn họ | trả lời câu gì |
|---|---|---|
| `--families truth` | nhãn tay của fixture | *luật* đúng bao nhiêu, tách khỏi lỗi 6.1 |
| `--families model` | 6.1 chạy thật | **đầu-cuối**, tức thứ dây chuyền thật nhận |

Cổng đã có sẵn cả hai cờ. Chỉ có con số `model` mới dùng để quyết bật luật —
`truth` là chẩn đoán, vì luật khoá theo họ nên họ sai là luật sai theo.

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
| **số chân thật** | ✅ IPC-356, footprint trong BOM/PnP, golden lúc enroll. ❌ **luật package KHÔNG bù được** |
| toạ độ land theo mm | ✅ chỉ IPC-356 |
| land không có linh kiện (test point, thermal pad) | ✅ IPC-356; ⚠️ golden nếu người enroll khoanh |

> **Sửa 2026-09-05.** Ô đầu trước đây liệt kê cả "luật package" — sai, và sai
> đúng kiểu §1 đã cảnh báo. Luật chỉ cho **topology tương đối**: `ic_hai_ben`
> phủ 6–256 chân. Nó không đếm chân, nên không thay được cột này.

Bộ đọc footprint (`SOIC-16` → 16 chân, 2 cạnh) cho số chân **chính xác, không
cần model, không cần gán nhãn** — rẻ hơn luật package nhiều. Nếu BOM/PnP của
bạn có cột `footprint` thì làm nó trước (câu hỏi 5 ở §11).

---

## 8. Ba chốt an toàn của bộ luật

Cả ba đến từ review của Codex, và cả ba đều là ca **hỏng im lặng**: ROI vẫn
được dựng, chỉ là dựng thiếu hoặc dựng sai chỗ, nên không nhìn ra từ ảnh kết
quả. §8.1 và §8.2 cài 2026-09-04; §8.3 cài 2026-09-05.

> **Nguyên tắc chung, rút ra sau ba lần.** Luật ảnh được phép **thêm** hoặc
> **giữ** ROI; nó chỉ được **xoá** ROI baseline khi có bằng chứng dương
> (footprint/CAD/nhãn tay/golden recipe đã duyệt). Mọi lần vi phạm nguyên tắc
> này đều sinh ra đúng một loại lỗi: mối hàn không ai kiểm, không ai biết.

### 8.1. Không suy "gói ẩn chân" từ sự VẮNG MẶT của chân

`allow_hidden_from_absence` giờ mặc định **False**.

Chốt cũ ("lead detector có tìm được chân ở linh kiện khác trên board") chỉ
chứng minh detector không chết hẳn — nó **không** chứng minh detector không bỏ
sót đúng con IC đang xét. Mà `ic_khong_chan` có `PadProfile(0, 0)`, nên kết
luận sai làm 5.5 bỏ **sạch** ROI của linh kiện đó.

Gói ẩn chân phải đến từ **bằng chứng dương** — footprint/CAD hoặc nhãn tay —
chứ không từ một phép suy trên sự vắng mặt. Và §7.1 vừa nói: không có lấy một
ví dụ QFN/BGA nào để hiệu chuẩn.

### 8.2. QFP nhìn thấy một nửa trông y hệt SOIC

Nếu chỉ phát hiện được chân ở 2 trong 4 cạnh của một QFP, luật cũ kết luận
`ic_hai_ben` — và 5.5 **không dựng dải trên hai cạnh còn lại**, hai cạnh có
chân thật.

Chốt mới: chân ở 2 cạnh đối **cộng thân gần vuông** thì trả `None`. Ngưỡng đo
trên 117 IC gán tay (§6.7):

| | aspect trung vị | tỉ lệ dưới 1,3 |
|---|---:|---:|
| `ic_bon_ben` (QFP) | **1,03** | 64% |
| `ic_hai_ben` (SOIC/TSOP) | **1,95** | 13% |

Thân vuông là bằng chứng nghiêng về QFP, không nghiêng về SOIC.

> **Chốt này GIẢM rủi ro chứ chưa an toàn.** Nó chỉ bắt được 64% số QFP:
> **12/33 QFP trong tập tiền gán có aspect ≥ 1,3**, và nếu detector chỉ thấy
> hai cạnh thì chúng vẫn bị gọi thành `ic_hai_ben` rồi mất ROI ở hai cạnh còn
> lại.
>
> Nguyên tắc đúng, theo đề xuất của Codex: **không bao giờ thu từ bốn cạnh
> xuống hai cạnh chỉ vì detector không nhìn thấy hai cạnh kia.** Chỉ thu hẹp
> khi có footprint/nhãn tay/golden recipe. Luật ảnh nên *ưu tiên* ROI chứ
> không *xoá* ROI baseline.

### 8.3. Cạnh chân đo được KHÔNG đi tới bước dựng ROI — 2026-09-05

**(a) đã cài 2026-09-05; (b) chưa — xem bảng cuối mục.**

Luật đọc được chân nằm ở cạnh nào và ghi vào
`metadata["lead_edges"]`. Nhưng `PackageClassification` chỉ mang **tên lớp**;
thông tin cạnh dừng lại ở đó. Sang [`geometry.py:145-152`](../../aoi_pipeline/solder/geometry.py#L145-L152),
`ic_hai_ben` → `dual_sided` → luôn giữ `lead_top`/`lead_bottom` **trong hệ toạ
độ linh kiện**, tức **hai cạnh dài của thân** — không phải hai cạnh mà luật vừa
tìm thấy chân.

Ghép với `_ic_package`, vốn nhận **cả hai** cặp đối (`{left,right}` *hoặc*
`{top,bottom}`) mà **không kiểm cặp đó có trùng trục dài không**, ta được một
đường đi từ phân loại đúng tới ROI sai cạnh.

**Tái hiện** (mẫu tổng hợp, thân 100×200 nên trục dài **dọc**, chân đặt ở hai
cạnh **ngắn**):

```
luat quyet:  ic_hai_ben, lead_edges = ['bottom','top']
RUNTIME truoc: do phu pad = 4/4
RUNTIME sau  : do phu pad = 0/4
```

Hai ROI còn lại nằm ở `x∈[274,314]` và `x∈[386,426]` — hai cạnh **trái/phải**
của ảnh, trong khi chân ở `y≈185` và `y≈408`. Mất sạch.

**Nặng hơn dự đoán ban đầu: bật luật có thể TỆ HƠN không bật.** Đường
`multi_pin` hôm nay còn chạy `_dominant_edge_pair` đọc pixel để tự tìm cặp cạnh
mang chân; `ic_hai_ben` bỏ qua bước đó và chốt cứng theo trục dài. Nên với đúng
ca này, luật **lấy đi** một cơ chế đang hoạt động.

**Hai việc, tách hẳn nhau:**

| | việc | khối lượng | chặn được gì |
|---|---|---|---|
| **a** ✅ | `_ic_package` **bỏ qua** khi cặp cạnh phát hiện được không phải cặp trục dài (`require_leads_on_long_axis`) | 3 dòng | biến ROI sai cạnh im lặng thành abstain — lùi về hành vi hôm nay |
| **b** | truyền `lead_edges` + hướng xuyên suốt xuống 5.5 (§10.1) | vừa | dựng ROI đúng cạnh thật |

(a) **không thay được** (b) — nó chỉ đóng lỗ hổng, không dùng được thông tin đã
đo. Nhưng (a) rẻ và an toàn tuyệt đối theo nguyên tắc ở đầu §8, nên làm trước.

> **Điều đáng lo nhất tìm được khi cài (a):** cài xong thì **cả ba test
> happy-path của nhánh `ic_hai_ben` đều đỏ** — và cả ba đều đặt chân ở hai cạnh
> **ngắn** rồi assert `ic_hai_ben`. Một trong số đó có docstring viết thẳng
> *"Chân sát hai cạnh NGẮN của thân"*.
>
> Tức 15 test của bộ luật xanh hết, nhưng chúng viết ra từ **cùng một mô hình
> sai** với code, nên chúng khẳng định lại cái sai thay vì bắt nó. Đây là loại
> lỗi mà thêm test không tìm ra được — chỉ có chạy thật rồi đo mới ra.
>
> Ba test đã sửa để dùng hình dạng thật (chân SOIC mọc từ hai cạnh dài), cộng
> hai test mới: một canh ca cạnh ngắn phải bỏ qua, một canh cờ
> `require_leads_on_long_axis` tắt được — để bản vá tạm không hoá thành vĩnh
> viễn khi (b) xong.

> **Trên bo thật lỗi này hiếm hay thường?** Chưa biết — và §9.0 giải thích vì
> sao chưa đo được. Với SOIC/TSSOP thật thì chân gần như luôn ở hai cạnh dài,
> nên (a) hiếm khi kích hoạt. Nhưng "hiếm" là phỏng đoán từ hiểu biết về gói
> chứ không phải số đo, và lead detector trả sai cặp cạnh cũng dẫn tới đây.

---

## 9. Cổng nghiệm thu và rủi ro

### 9.0. Cổng đang MÙ với chính thứ nó phải đo — 2026-09-05

**Đây là lỗi nặng nhất trong tài liệu này, và nó là lỗi của cổng chứ không phải
của luật.**

[`evaluate_package_rule_gate.py:84-92`](../../scripts/evaluate_package_rule_gate.py#L84-L92)
gọi thẳng `derive_solder_joints(...)` mà **không truyền `geometry=`**. Khi đó
[`geometry.py:101`](../../aoi_pipeline/solder/geometry.py#L101) lấy
`terminal_geometry(detection.label)` — tức **nhãn detector**, bỏ qua sạch
`metadata["terminal_geometry_override"]` mà `apply_package_classifications()`
vừa ghi vào ngay dòng trên.

Đường runtime thật là
[`SolderJointCropper.derive`](../../aoi_pipeline/solder/geometry.py#L1481-L1500),
và nó **có** đọc metadata đó. Hai đường khác nhau ⇒ `before` và `after` của cổng
bằng nhau **theo cấu trúc**, không phải vì luật an toàn.

**Tái hiện** (cùng mẫu tổng hợp ở §8.3):

| | pad phủ trước | pad phủ sau |
|---|---:|---:|
| **cổng** | 4/4 | **4/4** ← không thấy gì |
| **runtime** | 4/4 | **0/4** |

Cổng bỏ qua thêm `axis_known` và `refine_to_metal`, nên **thêm `geometry=` là
chưa đủ**. Sửa đúng: cho cổng gọi thẳng `SolderJointCropper.derive` — một đường
duy nhất, không có bản sao nào để lệch.

> **Bài học, ghi lại để không lặp:** cổng nghiệm thu **không được dựng lại**
> đường mà nó đo. Mọi bản sao của đường runtime sẽ trôi khỏi bản gốc, và khi
> trôi thì nó trôi về phía im lặng báo PASS. Test canh: mẫu tổng hợp ở §8.3
> phải làm cổng **fail**; cổng nào không fail trên nó là cổng hỏng.

**Hệ quả phải nói thẳng:** con số "0 mất pad, ROI 90 → 90, pad 28 → 28" báo cáo
ngày 2026-09-05 (§10.3) **không có giá trị**. Nó không chứng minh luật an toàn;
nó chỉ chứng minh cổng không nhìn.

#### 9.0b. Đã sửa, và hai lỗ hổng nữa lộ ra khi sửa

**✅ Sửa 2026-09-05.** Cổng giờ gọi thẳng `SolderJointCropper.derive`. Kèm hai
test canh ở `tests/test_package_rule_gate.py`: một test trực tiếp (`hidden_terminals`
phải cho **0 ROI**) và một test đi trọn `evaluate_board()`, đối chiếu **diện
tích** ROI với chính API runtime — không đối chiếu **số** ROI, vì ở mẫu đó
`connector_rows` và `multi_pin` cùng cho 2 ROI, đúng loại tín hiệu quá yếu đã
để lọt lỗi lần đầu.

Sửa xong thì lộ thêm hai chỗ cùng họ "hỏng im lặng", cả hai đã sửa luôn:

1. **`--families model` chưa từng chạy.** `AOIPipeline` **không tự nạp** 6.1 từ
   `models/active/classifier/`; phải truyền vào. Cổng không truyền, nên chế độ
   này im lặng nhận danh sách họ **rỗng**, luật không chạy lần nào, và bảng in
   ra 39 dòng "bỏ qua" trông y hệt như luật đã chạy rồi từ chối. Mà theo §7.2b
   thì **chính con số này mới quyết định bật luật**.
2. **"Luật không chạy được" và "luật chạy rồi bỏ qua" in ra giống hệt nhau.**
   Giờ có `rule_skip` và một dòng ⛔ riêng. Tương tự, cổng tách "6.1 trả về
   `unknown` nên luật không được phép chạy" khỏi "luật tự từ chối" — hai cái sửa
   ở hai nơi khác nhau.

**Chạy lại, số thật (1 board, 28 pad đếm tay):**

| | quyết được | bỏ qua | ROI | pad phủ | mất |
|---|---|---:|---|---|---:|
| `--families truth` | `hai_chan` ×24 | 15/39 | 90 → 90 | 19 → 19 /28 | **0** |
| `--families model` | `hai_chan` ×23 | 16/39 | 90 → 90 | 19 → 19 /28 | **0** |

Chênh lệch giữa hai chế độ là **2 thân bị 6.1 trả `unknown`** (1 `ic`, 1
`resistor`), không phải luật sai.

> ⚠️ **PASS này vẫn CHƯA đủ để bật luật**, vì ba lý do độc lập:
>
> 1. **Nhánh `ic` vẫn chưa chạy lần nào** — 18/60 chân có tâm nằm trong box
>    thân, nên không cạnh nào đủ dải. Mọi quyết định trên bảng đều là ánh xạ
>    họ tầm thường (`hai_chan`), mà `terminal_geometry("resistor")` vốn đã trả
>    `two_terminal`. Đây vẫn là §10.4 bước 5.
> 2. **1 board.** Cận trên rule of three là **10,71%**.
> 3. **Chưa có tập nghiệm thu khoá** (§7.2b).
>
> Cái đã đổi so với hôm qua: **cổng giờ đo được**. Trước đó nó không đo gì.

**Số phụ, về bước 5.5 chứ không về luật:** trên đường runtime (có
`refine_to_metal`) độ phủ baseline là **19/28**, không phải 28/28 — cùng 90 ROI,
nhưng refine co chúng về mảng kim loại và 9 pad rơi xuống dưới ngưỡng 50%. Nó
không chặn cổng (refine tác động như nhau lên cả hai vế), nhưng **9/28 pad đếm
tay tuột khỏi ROI sau refine là việc phải đi đo riêng.**

#### 9.0c. Nhánh `ic` bị đói vì NGUỒN CHÂN, không vì quy ước box

**Đây là lý do bước 5 của §10.4 (dựng fixture mới) sẽ KHÔNG mở được nhánh `ic`.**
Đo trước khi làm, và phép đo lật lại giả định đã ghi hai lần trong tài liệu này.

Giả định cũ: chân rơi vào trong box thân là vì fixture dùng **quy ước box cũ**
(detector 22 lớp khoanh bao cả chân). Đo bằng detector **mới** (một lớp, chỉ
thân) trên bốn nguồn ảnh:

| ảnh | thân | chân NGOÀI | chân TRONG | ứng viên `ic_hai_ben` |
|---|---:|---:|---:|---:|
| `real_pcb/phone/whole_pcb.jpg` (bo dự án, cả bo) | 266 | 43 | **469** (92%) | 1 |
| MPI gas-pump 4096×2816 (cả bo) | 166 | 271 | **1170** (81%) | 2 |
| 12 tile 1024 cắt từ ảnh MPI đó | 200 | 206 | **967** (82%) | 3 |
| tile PCB-DSLR 1024 (`pcb11`) | 204 | **189** | 65 (26%) | 5 |

Quy ước box mới **không** sửa được chuyện đó. Bằng chứng quyết định nằm ở chính
fixture đang có — cùng một bộ hộp thân, hai nguồn chân khác nhau:

| nguồn chân | ngoài thân | trong thân |
|---|---:|---:|
| pass 2 (lead detector) | 42 | **18** |
| pad **đếm tay** của fixture | 24 | **4** |

Cùng hộp thân, nhưng pad đếm tay nằm ngoài 24/28 còn chân pass 2 thì không.
Nguyên nhân là **nguyên nhân (b)** mà cổng vẫn nói là không phân biệt được:
`detect_leads_in_components` cắt một **cửa sổ quanh** linh kiện rồi tìm mối hàn
bên trong, nên mối hàn nó trả về đè lên thân. `_edge_of` đòi **tâm** chân nằm
ngoài hộp, nên chúng không đóng góp cạnh nào.

**✅ Đã thêm `--leads truth|model|none`**, đối xứng với `--families`, và vì cùng
một lý do. Với `--leads truth` bộ luật nhận pad đếm tay, và nhánh `ic` **chạy
lần đầu tiên**:

```
chân trong thân: 18 -> 4
luật BỎ QUA:
   12  capacitor - 0 canh co dai chan
    2  ic - 1 canh co dai chan
    1  ic - 2 canh doi nhung than gan vuong (aspect 1.13 < 1.3), §8.2
```

Con `ic` duy nhất đi tới được nhánh hai-cạnh-đối bị **§8.2 chặn đúng thiết kế**
— thân gần vuông, nghi là QFP mới nhìn thấy một nửa. Không phải lỗi.

> ⚠️ `--leads truth` đo **LOGIC của luật với bằng chứng chân hoàn hảo**, đúng
> kiểu `--families truth` đo luật tách khỏi lỗi 6.1. Nó **không** thay được
> `--leads model` cho quyết định bật luật.

Cổng cũng đã tách ba ca "2 cạnh" vốn in ra giống hệt nhau nhưng sửa ở ba chỗ
khác nhau: cạnh **kề** (ngoài taxonomy), cạnh **đối + thân vuông** (§8.2), cạnh
**đối + lệch trục dài** (§8.3).

**Việc thật lộ ra từ đây, thay cho "dựng thêm fixture":**

1. **Lead detector / pass 2 trả mối hàn đè lên thân.** Đây mới là thứ chặn nhánh
   `ic` trên dây chuyền, và không có fixture nào sửa được nó. Cần quyết: nới
   `_edge_of` (ví dụ so theo mép chứ không theo tâm), hay sửa cửa sổ cắt của
   pass 2, hay chấp nhận rằng đường ảnh không nuôi nổi nhánh `ic` và topology
   phải đến từ recipe (§10.2).
2. **Fixture mới vẫn cần**, nhưng cho việc khác: đo **độ phủ pad** trên hộp thân
   quy ước mới. Không còn là đường để mở nhánh `ic`.
3. **Giấy phép ảnh fixture.** `datasets/test_images/` bị `.gitignore` chặn và
   CVL PCB-DSLR là **phi thương mại**, tile phái sinh mang theo ràng buộc đó —
   không commit được. **MPI-PCB gas pump là CC BY 4.0**, commit được kèm
   attribution; đó là nguồn đúng cho fixture tiếp theo.

### 9.1. Ba cổng, theo thứ tự

1. **Nhầm `ic` ↔ thụ động phải bằng 0** trên tập kiểm §7.2. Đây là cặp duy nhất
   làm ROI *tệ đi thật*.

   ⚠️ **Cổng trong code đang kiểm cặp KHÁC.**
   `scripts/evaluate_package_roi_gate.py:86` kiểm `ic_hai_ben ↔ ic_khong_chan`,
   không kiểm `ic ↔ thụ động`. Hai cặp đều nguy hiểm và **cổng phải kiểm cả
   hai**; hiện chỉ có một. Codex chỉ ra, chưa sửa vì script đó viết cho đường
   ONNX mà kế hoạch này đang bỏ.

   Nặng hơn: **cổng đó nhận ma trận nhầm lẫn của một MODEL, nên nó không chạy
   được đường luật.** Đường luật cần một cổng riêng, đo trên nhiều board và
   theo từng pad — chứ 28 pad trên **một** board thì không đủ để tuyên bố bất
   cứ điều gì bằng 0.

   Và cổng phải đo **recall từng pad**, không chỉ tổng số pad còn được phủ:
   0 lỗi trên 250 mẫu vẫn để lại cận trên ~1,2% ở mức tin cậy 95% (rule of
   three). Với AOI thì 1,2% ca bỏ ROI là nhiều.
2. **Đo lại ROI trên board thật** — `tests/data/solder_geometry`, 28 pad đếm
   tay: bật luật phải **không giảm** độ phủ pad. Đây là cổng thật; cổng 1 chỉ
   là điều kiện cần.

   ⚠️ **Cổng này đang MÙ — xem §9.0.** Nó dựng ROI bằng một đường khác đường
   runtime, nên nó không thể thấy thay đổi do luật gây ra. Phải sửa trước, rồi
   chạy lại; số cũ bỏ đi. Và đo trên cả hai nguồn nhãn họ (`truth` *và*
   `model`, §7.2b) — chỉ con số `model` mới dùng để quyết bật luật.
3. **Mặc định TẮT** cho tới khi vượt cổng 2 trên board của chính dây chuyền,
   với ngưỡng đã đóng băng trước khi đo (§7.2b).

### 9.2. Rủi ro, theo khả năng xảy ra

1. **Luật hỏng có hệ thống, không hỏng dần.** Lead detector recall kém trên một
   board ⇒ mọi IC thành `ic_khong_chan` ⇒ mất ROI im lặng. *Đã chặn:* điều kiện
   ngữ cảnh ở §5.
2. **Ngưỡng kích thước không chuyển được giữa các độ phóng đại** (§6.3). *Giảm
   thiểu:* quy về mm khi có px/mm, lùi về `area_frac` khi không.
3. ~~**Lớp 5/6 không có ví dụ nào**~~ → **Lớp 6 (`ic_khong_chan`) không có ví
   dụ nào.** *Sửa 2026-09-05:* lớp 5 (`ic_bon_ben`) giờ có **33 ví dụ** từ nhãn
   tay §6.7 — mục này còn sót lại từ trước lần gán đó; §7.1 đã sửa, §9 thì
   chưa. Lớp 6 vẫn đúng là **0 ví dụ**, và đó chính là lý do §8.1 cấm suy lớp
   này từ ảnh. *Giảm thiểu:* nếu tập kiểm vẫn quá ít mẫu thì gộp thành một lớp
   "IC lớn — cần người xem", dùng như cờ chuyển review chứ không như lớp đo
   đạc. Thà thành thật là không biết.
4. **Nhãn công khai sai** (§6.4). *Giảm thiểu:* duyệt contact sheet trước khi
   tin bất kỳ con số nào rút từ chúng.
5. **Không có CAD ⇒ không còn lưới an toàn thứ hai.** Package là *một* nguồn,
   không phải hai nguồn kiểm chéo. *Giảm thiểu:* cờ `review` khi số ROI dựng
   được lệch hạng kỳ vọng.
6. **Cổng nghiệm thu trôi khỏi đường runtime** — thêm 2026-09-05, và đây là
   rủi ro đã **thành hiện thực một lần** (§9.0). Nó tệ hơn mọi rủi ro ở trên vì
   nó làm hỏng chính công cụ dùng để phát hiện các rủi ro kia. *Giảm thiểu:*
   cổng gọi thẳng hàm runtime, không dựng lại; và có test canh bằng mẫu tổng
   hợp mà cổng **phải** fail.
7. **Khớp ngưỡng quá tay dù không train model** (§7.2b). Ba hằng số trong
   `PackageRuleConfig` khớp bằng tay trên 117 mẫu vẫn đủ để khớp nhiễu.
   *Giảm thiểu:* chia hiệu chỉnh/nghiệm thu theo bo, đóng băng ngưỡng trước
   khi đo.

---

## 10. Ba đề xuất kiến trúc từ review lượt 2

Codex đề xuất ba thay đổi kiến trúc. Dưới đây là bản đã đối chiếu với code thật
— vì hai trong ba thứ Codex mô tả **đã tồn tại một phần**, và biết phần nào đã
có thì khối lượng việc khác hẳn.

> **Cập nhật 2026-09-05.** Mục này mở đầu bằng "chúng không phải sửa lỗi mà là
> đổi thiết kế, nên chưa làm". **Điều đó không còn đúng với §10.1.** Review
> lượt 3 tái hiện được ca ROI đặt sai cạnh (§8.3), nên §10.1 **đổi từ đề xuất
> thành việc bắt buộc**. §10.2 và §10.3 giữ nguyên trạng thái.

### 10.1. Đổi hợp đồng nội bộ thành `terminal_topology` — **BẮT BUỘC**

> **Vì sao lên bắt buộc (2026-09-05).** Đây không còn là câu chuyện "hợp đồng
> đẹp hơn". §8.3 chứng minh bằng mẫu chạy được: luật **đã đo** cạnh chân rồi
> **vứt đi**, và 5.5 dựng lại theo trục dài của thân — 4/4 pad → 0/4. Chừng nào
> bước 5.2 còn trả về một chuỗi trần thì thông tin đó không có đường nào đi
> tiếp.
>
> Trường tối thiểu để đóng §8.3(b): **`lead_edges`** (cạnh nào) và
> **`orientation`** (hệ toạ độ nào — ảnh hay linh kiện). Đúng hai trường bảng
> dưới đang đánh ⚠️ và ❌.

**Đề xuất.** Thay vì một chuỗi trong bảy slug, bước 5.2 trả về một cấu trúc:
`lead_edges`, `pin_count`/`range`, `visibility`, `mount_type`, `orientation`,
`source`, `decision`. Bảy slug cũ giữ làm alias v1.

**Đã có gì.** `aoi_pipeline/placement/footprints.py::FootprintProfile` **đã nói
gần đúng hợp đồng này** cho nguồn footprint:

| Codex đề xuất | `FootprintProfile` đã có |
|---|---|
| `pin_count` / `range` | ✅ `expected_pin_count`, `expected_pin_count_range` — loại trừ nhau |
| `visibility` | ✅ mã hoá trong `lead_sides = 0` (gói ẩn chân, **không** sinh ROI) |
| `source` | ✅ `reason` |
| `decision`/độ tin | ✅ `confidence` — docstring nói rõ đây là độ tin **phân tích tên**, không phải độ tin model |
| `lead_edges` | ⚠️ mới có `lead_sides` (đếm số cạnh), chưa nói **cạnh nào** |
| `orientation` | ❌ chưa có |
| `mount_type` | ❌ chưa có |

**Nên việc thật không phải "thiết kế hợp đồng mới"** mà là **cho bộ luật nói
đúng cái hợp đồng mà parser footprint đã nói**. Hiện luật trả một chuỗi trần,
nên `terminal_geometry()` không phân biệt được "biết chắc 16 chân hai cạnh" với
"đoán từ ảnh, hai cạnh".

Bảy slug **phải** giữ: §2 ghi rõ chúng là giá trị ổn định, được app gán nhãn,
manifest model và export runtime cùng ghi ra.

**Khối lượng:** vừa. Thêm hai trường vào `FootprintProfile`, cho `package_rules`
trả `FootprintProfile` thay vì `str`, và cho `terminal_geometry()` nhận nó.

### 10.2. Thứ tự nguồn, và chốt topology một lần lúc tạo golden recipe

**Đề xuất.** Thứ tự bằng chứng:

```
IPC/CAD pads → footprint/PnP → golden recipe đã duyệt
             → bằng chứng lead DƯƠNG → heuristic ảnh → unknown/multi_pin
```

Và: với dây chuyền kiểm cùng một mẫu PCB, chốt topology + ROI **một lần** lúc
tạo golden recipe; luật ảnh chỉ *gợi ý* lúc enroll, **không xoá** ROI production.

**Đã có gì.** `aoi_pipeline/golden/recipe.py::SlotRecipe` lưu cho từng slot:
`expected_angle_deg`, `rotation_period_deg` — **tức hướng**, đúng thứ §6.3c
đang thiếu cho `tru_dung` — cùng `fixed_roi_xyxy`, `source`, `source_confidence`.
Nên khái niệm "chốt một lần rồi dùng lại" đã là thiết kế sẵn có.

**Còn thiếu gì, và đây là phần đắt:**

1. `SlotRecipe` **không có trường topology nào**. Nó biết ROI ở đâu nhưng không
   biết ROI đó thuộc gói gì, nên không kiểm chéo được với luật.
2. `aoi_pipeline/pipeline.py` **không hề đọc recipe** — golden chạy qua
   `app/pipeline_bridge.py` (frame_id `golden_enrollment`). Đường ROI mối hàn
   trong `pipeline.py` và đường golden hiện là **hai nhánh song song không nói
   chuyện với nhau**.

**Đây là điểm mạnh nhất của đề xuất, và cũng là lý do nó đắt.** Nếu dây chuyền
chỉ kiểm vài mẫu PCB thì chốt topology lúc enroll là đúng: người duyệt một lần,
sau đó không luật nào phải đoán nữa. Nhưng nó đòi nối hai nhánh đang rời nhau.

**Khối lượng:** lớn. Cần bạn xác nhận **dây chuyền có kiểm cố định vài mẫu PCB
không** — nếu mỗi lô một mẫu khác thì enroll không trả đủ công.

### 10.3. Cổng riêng cho đường luật

**Đề xuất.** Cổng đo theo từng pad/từng linh kiện, số và diện tích ROI thừa,
tỉ lệ abstain, tách theo từng topology, trên nhiều board. **Mất bất kỳ pad
baseline nào là fail.**

**Vì sao bắt buộc.** `scripts/evaluate_package_roi_gate.py` nhận **ma trận nhầm
lẫn của một model** — đường luật không sinh ra thứ đó, nên cổng hiện tại
**không chạy được cho luật**. Và thước đo duy nhất đang có là **28 pad trên
MỘT board**, mà board đó lại **ngoài miền** của detector mới (§ báo cáo hai
lượt J5). Không đủ để tuyên bố bất cứ điều gì bằng 0.

Rule of three: quan sát 0 lỗi trên 250 mẫu vẫn để lại cận trên **~1,2%** ở mức
tin cậy 95%.

**Khối lượng:** vừa, và **không phụ thuộc hai đề xuất kia** — viết được ngay.
Đây là thứ nên làm trước, vì không có nó thì mọi thay đổi ở 10.1/10.2 đều không
chứng minh được là tốt lên.

> **⚠️ ĐÃ VIẾT 2026-09-05, NHƯNG CỔNG BỊ MÙ — xem §9.0.** Giữ nguyên phần
> tường thuật bên dưới làm bản ghi, nhưng **mọi con số trong đó phải bỏ đi**:
> cổng dựng ROI bằng một đường khác đường runtime, nên "ROI 90 → 90, pad
> 28 → 28, 0 mất pad" chỉ nói rằng cổng không nhìn thấy gì, không nói rằng luật
> không đổi gì. Hai chỗ mù ở mục 1 và 2 dưới đây vẫn đúng và vẫn phải xử lý;
> riêng kết luận "luật không đổi gì trên board này" thì **chưa được chứng
> minh**.
>
> `scripts/evaluate_package_rule_gate.py`.
> So ROI **trước và sau** khi bật luật, trên cùng một board. Mất một pad
> baseline là **fail ngay**, không cân nhắc đánh đổi.
>
> Chạy lần đầu trên fixture đang có, và nó lập tức phát hiện **hai chỗ mù mà
> tôi không biết trước**:
>
> 1. **Nhánh `ic` chưa từng được kiểm.** 60 chân được lead detector tìm ra,
>    nhưng **18 chân có tâm nằm TRONG box thân**, nên `_edge_of` trả `None` và
>    không cạnh nào đủ dải chân. Nguyên nhân: fixture dùng box của **detector
>    22 lớp cũ**, vốn khoanh *bao cả chân*; luật thì đọc chân **ngoài** thân
>    theo quy ước mới. Cổng giờ in cảnh báo này thay vì im lặng báo PASS —
>    **PASS ở đây không có nghĩa là luật đã được kiểm.**
> 2. **Luật hiện không đổi gì trên board này.** ROI 90 → 90, pad 28 → 28. Vì
>    24 quyết định đều là `hai_chan`, mà `terminal_geometry("resistor")` vốn
>    đã trả `two_terminal` cùng `PadProfile`. Giá trị của luật chỉ hiện ra ở
>    nhánh `ic` và `capacitor` — đúng hai nhánh chưa chạy được.
>
> Cổng cũng tự khai giới hạn của chính nó: **1 board, 28 pad**, cận trên rule
> of three là **10,71%**. Nó nói thẳng *"không mất pad nào ở đây KHÔNG chứng
> minh được luật an toàn"*.
>
> ⇒ ~~Việc tiếp theo không phải sửa luật, mà là có thêm fixture.~~
> **Sửa 2026-09-05: việc tiếp theo là sửa CỔNG (§9.0).** Thêm fixture vào một
> cái cân hỏng thì chỉ có thêm số sai. Thứ tự đúng ở §10.4.

### 10.4. Thứ tự — cập nhật 2026-09-05 sau review lượt 3

Thứ tự cũ đặt "viết cổng riêng" ở vị trí 1 và coi như xong. Cổng đã viết nhưng
mù (§9.0), nên bảng dưới thay hẳn bảng cũ. Codex đề xuất thứ tự này và tôi đồng
ý, chỉ chèn thêm bước 2 vì nó rẻ và độc lập.

| | việc | phụ thuộc | vì sao ở vị trí này |
|---|---|---|---|
| 1 | ✅ **sửa cổng dùng chung `SolderJointCropper.derive`** (§9.0b) + 2 test canh | không | mọi số sau đó đều vô nghĩa nếu cân còn hỏng |
| 2 | ✅ **chốt trục dài trong `_ic_package`** (§8.3a) | không | 3 dòng, đóng lỗ hổng đang mở, an toàn tuyệt đối |
| 3 | ✅ chạy lại cổng, cả `truth` **và** `model` (§7.2b) | (1),(2) | lần đầu tiên có số thật về luật — §9.0b |
| 4 | duyệt 103 mẫu họ `capacitor` (§6.3b) | bạn | chạy song song được với 1–3 |
| 5 | ✅ **`--leads truth`** — nguồn chân thứ hai (§9.0c) | (1) | mở được nhánh `ic` lần đầu; fixture mới KHÔNG mở được |
| 6 | ⭐ **quyết: pass 2 trả mối hàn đè lên thân thì xử lý sao** (§9.0c) | (5) | đây mới là thứ chặn nhánh `ic` trên dây chuyền |
| 7 | fixture đúng quy ước box mới, ảnh **MPI CC BY 4.0** + chia hiệu chỉnh/nghiệm thu theo bo (§7.2b) | (1) | để đo ĐỘ PHỦ PAD, không còn để mở nhánh `ic` |
| 8 | hợp đồng `terminal_topology` (§10.1, §8.3b) | (1),(7) | dùng được cạnh chân đã đo, thay vì chỉ bỏ qua |
| 9 | nối golden recipe (§10.2) | (8), dây chuyền kiểm cố định — **đã xác nhận** | chốt topology một lần, người duyệt |

**Luật giữ mặc định TẮT suốt 1–9.** Điều kiện bật: vượt cổng ở bước 3 trên tập
nghiệm thu khoá của bước 5, với ngưỡng đóng băng trước khi đo.

**Điều tôi không đồng ý với Codex:** đề xuất bỏ hẳn `aspect-only 89,6%` cho
`capacitor`. Con số đó cũng đo sai phạm vi như 97,0% (§6.3b) nên đằng nào cũng
phải đo lại — chưa có cơ sở để loại bỏ *hay* giữ lại nó.

---

## 11. Câu hỏi cần bạn quyết

1. **Duyệt 750 nhãn tiền gán ở §6.7 chứ?** Đây là việc chặn mọi thứ khác: bốn
   phép đo mới trong tài liệu này đều rút từ chúng, và tôi gán bằng mắt nên
   chắc chắn có chỗ sai. 133 gói và 386 họ tôi đã đánh **chưa chắc** thay vì
   đoán bừa — những ô đó cần bạn nhất.
2. **Bốn họ chưa ánh xạ** (`magnetic`/`protection`/`timing`/`acoustic` và
   `relay`/`display`/`switch_control`/`battery_power_input`) — thêm lớp thứ 8
   cho "hộp lớn nhiều chân", hay để chúng lùi về `multi_pin` như hôm nay?
   Nhãn tay tìm được **19 ca thật** thuộc nhóm này (§6.7), nên câu hỏi không
   còn là giả định.
3. **Ánh xạ họ đang quá rộng (§6.5) — sửa ngay hay để sau?** `resistor` còn có
   mạng điện trở, `led` còn có RGB, `connector` gộp cả khe PCI lẫn jack RCA
   lẫn FPC mà code dựng một kiểu ROI duy nhất. Sửa đúng cách là thêm điều kiện
   hình học trong mỗi họ, và mỗi điều kiện phải đo trước khi bật.
4. **Duyệt sớm 103 mẫu họ `capacitor` được không?** (82 `tru_dung` + 21 chip.)
   Đây là chỗ mỏng nhất: cả hai phép đo trước đều sai phạm vi, và bản hiệu
   chuẩn lại chỉ dựa trên **21 mẫu chip**. Có thêm mẫu chip thì càng tốt.
   Và kể cả khi tách đúng, vẫn còn câu **hướng đặt ROI** ở §6.3c — thân tròn
   không có trục dài để đặt hai đầu.
5. **BOM/pick-and-place của bạn có cột `footprint` không?** Câu rẻ nhất trong
   danh sách: **có** thì làm bộ đọc footprint trước và hạ luật package xuống ưu
   tiên thấp; **không** thì luật lên đầu.
6. ~~Dây chuyền có kiểm cố định vài mẫu PCB không?~~ **ĐÃ TRẢ LỜI 2026-09-05:
   kiểm cố định vài mẫu, và thêm mẫu mới vào được.** ⇒ §10.2 đáng làm.

   Kèm theo một tiền đề: ảnh Golden hiện chỉ chính diện **một vùng**, các vùng
   khác bị nhìn nghiêng. Hướng **ghép SƠ ĐỒ** (không ghép ảnh) đã được chọn, và
   xung đột với ràng buộc "never blends... must remain traceable to a real
   acquisition" trong `golden/enrollment.py` **đã được giải theo hướng đó** —
   ảnh Golden vẫn là một file thật. Kế hoạch:
   [ke_hoach_golden_ghep_so_do.md](ke_hoach_golden_ghep_so_do.md).
   Phép đo chặn ở §4.2 của kế hoạch đó đã chạy: lệch ≤ 413 µm so với dung sai
   ROI ≥ 736 µm, nên việc ghép **không còn chặn** §10.2.
7. ~~Đồng ý thứ tự ở §10.4 chứ?~~ **ĐÃ ĐỒNG Ý 2026-09-05.** *Thứ tự đó đã được
   thay 2026-09-05 sau review lượt 3* — bắt đầu từ **sửa cổng** (§9.0), không
   phải từ viết cổng (§10.3). Bảng mới ở §10.4.
8. **Có xin được file IPC-D-356 từ bên gia công không?** Repo đọc được sẵn, và
   nó cho *từng pad một* — gần bằng có CAD. Đây cũng là **nguồn duy nhất** cho
   `ic_khong_chan`, vì §8.1 đã cấm suy lớp đó từ ảnh.
9. **Lớp 6 (`ic_khong_chan`): chấp nhận kết luận "không kiểm được bằng ảnh 2D
   trên xuống" chứ?** QFN/BGA mà vẫn phải kiểm là bài toán X-quang.
10. **Khoá bo nghiệm thu thế nào?** (§7.2b, mới 2026-09-05.) 34 bo trong tập
    nhãn tay cần chia hiệu chỉnh/nghiệm thu. Đề xuất của tôi: **~10 bo khoá
    lại**, chọn sao cho phủ đủ `ic` và `capacitor` chứ không ngẫu nhiên — bo
    toàn điện trở nằm trong tập khoá thì không đo được gì. Danh sách ghi vào
    file, không chọn lại mỗi lần đo. Bạn có bo nào muốn giữ riêng cho nghiệm
    thu không, hay để tôi chọn theo phân tầng?

---

Xem thêm: `Docs/bao_cao/tien_do_detect_2_luot.md` (bảng công việc sống),
`Docs/danh_gia/danh_gia_khoanh_box_than_linh_kien.md` (báo cáo box),
`Docs/ke_hoach/ke_hoach_pcb_defect_toan_mach.md` (kế hoạch lỗi toàn mạch).
