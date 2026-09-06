# Phân nhóm package cho linh kiện

**Cập nhật 2026-09-06**, tiếp tục lượt đánh giá bị gián đoạn tối 2026-09-05.
Đối chiếu code tại commit `b9b3b69`, nhánh `sua-cong-package-va-chot-truc-dai`.
§1 và §10 là trạng thái và lộ trình hiện tại; các phép đo ngày 03–05/09 bên
dưới được giữ làm bằng chứng khảo sát, không phải kết quả nghiệm thu production.

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

**Kết luận đánh giá.** Giữ hướng family → topology và không train model
package. Có thể triển khai từng phần theo §10.4; **chưa đủ bằng chứng để bật
luật thay đổi ROI production**. Vì dây chuyền đã xác nhận kiểm cố định vài mẫu
PCB, topology/ROI được duyệt trong Golden là hướng chính; chẩn đoán pass 2
và gợi ý từ ảnh chạy song song, không chặn thiết kế hợp đồng/recipe.

**Đang ở đâu.**

| | trạng thái |
|---|---|
| Thứ tự pipeline (6.1 và lead detector chạy trước 5.2) | ✅ xong, có test canh |
| Bộ luật cho họ `ic` + 5 họ ánh xạ thẳng | ✅ xong, mặc định TẮT |
| Chốt cho `ic_khong_chan` và QFP nửa vời | ✅ đã cài; QFP dài vẫn có thể lọt, chưa bảo đảm giữ ROI — §8.1, §8.2 |
| Cổng đo hình học của đường luật | ✅ dùng chung `SolderJointCropper.derive`; **chưa đo toàn bộ đầu ra 5.5** — §9.0b, §9.0d |
| Cạnh chân đo được → 5.5 | ⚠️ còn trong metadata dự đoán, nhưng consumer không dùng để dựng ROI — §8.3 |
| Nhánh `ic` trên fixture hiện có | ⚠️ chân bước 2 giờ **đến được** nhánh này (chân rơi vào trong thân 18 → 0, §9.0e); một IC tới chốt hai cạnh rồi bị §8.2 từ chối, nên **vẫn chưa có quyết định IC nào được áp dụng** — §9.0c, §9.0e |
| Chia họ `capacitor` (trụ đứng ↔ chip) | ✅ **xong 2026-09-06** — aspect < 1,17, nghiệm thu 90,5% vs baseline 68,2%, §6.3d |
| Tập kiểm gán tay | ⏳ **750 box, bạn đã duyệt xong; 667/750 có nhãn họ** — §6.7, §6.7b |
| Ánh xạ họ → gói cho các họ còn lại | ⏳ đang quá rộng — §6.5 |

> **Giới hạn quan trọng:** “0 pad baseline bị mất” mới là kiểm tra hồi quy
> trên tập đã chạy. Baseline hình học hiện chỉ phủ **19/28 pad**, không chứng
> minh 28/28 được kiểm. Cổng còn bỏ qua gán họ cho body generic, hợp nhất lead,
> CAD và xử lý ROI lấn sang linh kiện khác ở cuối 5.5. Vì vậy
> `--families model --leads model` hiện cũng chưa phải nghiệm thu đầu-cuối.

**Việc kế tiếp cần bạn:** còn **83 ô chưa kết luận được họ** (§6.7b). Chúng nằm
ở vùng không có designator trong tầm nhìn, nên ảnh đã hết thông tin — bỏ qua
chúng là hợp lý, miễn là ghi kèm **thiên lệch cỡ**: phần bỏ có trung vị cạnh dài
**17 px** so với **82 px** của phần giữ lại, nên mọi con số 6.1 đo trên phần còn
lại đều **lạc quan** và phải báo cáo kèm điều kiện đó.

**Việc kỹ thuật kế tiếp:** hoàn thiện cổng đo đầu ra cuối 5.5 và hợp đồng
topology; kiểm tra overlay/matching từng IC để xác định vì sao pass 2 thiếu
bằng chứng cạnh. Chưa có căn cứ để chọn nới `_edge_of` hay đổi cửa sổ crop.
Thứ tự và điều kiện hoàn thành ở §10.4.


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
5.2     ├─ họ = ic         → luật đọc vị trí chân → 2 bên | 4 bên | bỏ qua
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

`ic_khong_chan` chỉ nhận từ nguồn có thẩm quyền như footprint/CAD hoặc
recipe đã duyệt; luật ảnh mặc định không suy lớp này từ sự vắng mặt của lead.

**Điểm nối trong code.** `terminal_geometry()` (`config.py:54`) nhận
`package: str | None` với thứ tự ưu tiên **footprint → package → họ detector**.
Đường hiện tại đủ để ánh xạ slug, nhưng chưa truyền cạnh và hệ tọa độ xuống
consumer 5.5. Việc này cần bổ sung hợp đồng ở §10.1, không chỉ thay nguồn nhãn.

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

### 6.3d. ĐÃ TÁCH ĐƯỢC họ `capacitor` — 2026-09-06

Mục này bế tắc từ đầu vì chỉ có **21 mẫu chip**. Sau khi gán xong 286 ảnh còn
lại (§6.7b), họ `capacitor` có **85 `tru_dung` / 84 `hai_chan`** trên 32 bo —
gần như cân bằng, tức lần đầu tiên đo được tử tế.

**Cách đo.** Chia hiệu chỉnh/nghiệm thu **theo bo** (§7.2b), đóng băng ngưỡng
trước khi chạm tập nghiệm thu. Điểm số theo **tần suất thật**, cân lại theo tỉ
lệ lấy mẫu của từng tầng — tập 750 lấy phân tầng nên nó over-sample linh kiện
to; không cân lại thì mọi con số đều sai lệch có hệ thống.

| luật | ngưỡng | nghiệm thu (tần suất thật) |
|---|---|---:|
| **aspect** | **< 1,17** | **90,5%** |
| kích thước | > 36 px | 88,6% |
| độ tròn `4πA/P²` | > 0,88 | 68,2% |
| độ sáng | < 0,60 | 54,9% |
| *baseline "luôn đoán chip"* | — | *68,2%* |

Chọn **aspect**, không chọn kích thước dù hai số sát nhau: ngưỡng kích thước
không chuyển được giữa các độ phóng đại (§6.3), aspect thì có.

> Wilson 95% trên 42 mẫu nghiệm thu: **75,0%–94,8%**. Con số 90% là thật nhưng
> khoảng còn rộng — cần thêm bo trước khi coi là chốt.

**Độ tròn không phải "chưa đo được" mà là ĐÃ ĐO VÀ LOẠI.** §8.8 đề xuất
`4πA/P²`; đo ra nó **chỉ ngược**: tụ **trụ** có độ tròn trung vị **0,343**, tụ
**chip** **0,635**. Lý do: rãnh chữ thập trên nắp nhôm cộng phản quang làm vỡ
contour của cái lon, còn thân chip cho một hình chữ nhật sạch. Ngưỡng tốt nhất
đạt đúng bằng baseline, tức không thêm được gì.

### 6.3c-bis. Trục của thân tròn — và một lỗi mất ROI lộ ra từ đó

§6.3c hỏi: thân tròn thì đặt hai ROI theo trục nào? Giờ có số:
**47/85 = 55% tụ trụ có hai cạnh lệch nhau dưới 10%.** Ở mức đó, "cạnh nào dài
hơn" do vài pixel của hộp quyết định — **trục là nhiễu, không phải tín hiệu**.

Và đo tiếp thì lộ ra một lỗi đang nằm sẵn trong `solder/geometry.py`:

| hộp | `two_terminal` | `tru_dung` (trước khi sửa) |
|---|---:|---:|
| gần vuông (aspect 1,07) | **4 ROI** | **2 ROI** |
| thuôn dài (aspect 2,0) | 2 ROI | 2 ROI |

Tức **gán ĐÚNG `tru_dung` cho một tụ trụ lại xoá mất hai ROI thật** — ngược hẳn
nguyên tắc ở đầu §8. `_resolve_two_terminal_rects` vốn đã xử lý đúng ca này
(*"khi phép đo không dứt khoát thì phát cả hai trục"*), chỉ nhánh `tru_dung` là
bỏ qua cơ chế đó và luôn phát một cặp.

**✅ Đã sửa.** `tru_dung` giờ chỉ chốt một trục khi trục đến từ **bằng chứng
dương** (`axis_known`, tức góc xoay trong pick-and-place) hoặc khi thân đủ thuôn
dài; còn lại phát cả hai trục, đặt tên `_cross` để người duyệt biết đó là giả
thuyết thay thế. Vẫn **không** dùng phép dò kim loại — nó trả lời sai câu hỏi
trên một cái lon kim loại.

> Test cũ chỉ phủ hộp 100×40 (aspect 2,5), tức **chỉ ca thuôn dài** — nên lỗi
> này im lặng suốt. Đã thêm test cho ca gần vuông; gỡ bản vá ra thì test đỏ.

**Kết quả trên cổng, cả bốn tổ hợp `--families` × `--leads`:**

| | trước | sau |
|---|---:|---:|
| luật bỏ qua | 15–16 / 39 | **3–4 / 39** |
| ROI | 90 | **92** (chỉ thêm) |
| pad baseline mất | 0 | **0** |

Bật được nhánh `capacitor` là **nhờ** sửa hình học đi trước: nếu không, phân
loại càng đúng thì càng mất ROI.

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

86 mẫu `connector` chỉ ủng hộ việc dùng cùng nhãn, **không chứng minh cùng
một topology hoặc chính sách ROI**: đây là phép ánh xạ họ sang chính tên họ.
`led` chưa có mẫu để đánh giá. Chưa ánh xạ nào được miễn nghiệm thu theo pad
chỉ vì nhãn package khớp.

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
> sót so với một số cách thu hẹp*, không bảo đảm recall tuyệt đối: bảng trên
> vẫn có 6 ca bỏ ROI. Cần đo pad thực tế, không suy an toàn từ tên fallback.

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

### 6.7b. Một khung cắt không trả lời được cả hai câu — 2026-09-06

Duyệt 750 box xong thì còn **286 ô `XEM_KY`**, và chúng bế tắc vì một lý do
không ai để ý: **khung cắt bị chỉnh đúng cho câu hỏi *gói*, nên nó sai cho câu
hỏi *họ*.**

`build_family_package_review_set.py:44` đặt `MIN_PAD_PX = 14`. Với chip 10 px,
lề 14 px mỗi bên cho crop ~38 px — **đủ thấy pad hai đầu**, và comment trong code
nói rõ đó là chủ ý: *"pad chính là thứ phân biệt gói"*. Đúng.

Nhưng dấu hiệu mạnh nhất để biết một chip là điện trở hay tụ **không nằm trên
linh kiện** — nó là **silkscreen designator** (`R902`, `C450`, `FB19`, `L501`)
in trên mặt bo cạnh linh kiện, cách thân **20–60 px**. Tức nó **luôn** nằm ngoài
khung chặt.

**Đo được.** Cắt lại 140 ô bế tắc bằng khung rộng (nửa cạnh
`max(95, 1,5 × cạnh dài)`), từ chính tile gốc:

| | |
|---|---:|
| Giải thêm được | **57 / 140 = 41%** |
| — đọc thẳng designator sát box | 22 |
| — suy từ cụm (dãy linh kiện giống hệt cạnh IC đệm; cụm designator cùng chữ) | 35 |
| Cả tập có nhãn họ | 481 → **667 / 750 = 88,9%** |

Cỡ nhỏ nhất đọc được designator là **9 px** (#100, cạnh `R531`/`R540`). Đọc được
vì **chữ silkscreen to hơn linh kiện** — điều mà mọi phép đo dựa trên "linh kiện
bao nhiêu pixel" đều bỏ sót.

Vài ca lớn cũng chỉ bối cảnh mới giải được: thanh trắng mỏng #662/#629 hoá ra là
**thân rơ-le SIEMENS nhìn nghiêng**; khối bạc #604 là **vỏ kim loại cổng PS/2**
— trước đó tôi đoán nhầm là vỏ chắn.

**✅ Đã sửa và áp dụng 2026-09-06.** Script cắt **hai khung mỗi box**:

| thư mục | lề | trả lời câu |
|---|---|---|
| `crops/` | `MIN_PAD_PX = 14`, **không đổi** | **gói** — thấy pad hai đầu |
| `crops_wide/` | `max(95, 1,5 × cạnh dài)` | **họ** — thấy designator |

Không nâng `PAD_RATIO` để làm việc này: nâng lên thì linh kiện lớn chìm trong
crop khổng lồ, đúng cái bẫy đã ghi ở `MAX_PAD_PX`.

> **Chứng minh không mất gì, không phải hứa:** dựng lại toàn bộ tập với cùng
> seed 42 và cùng nguồn (`joint_boxes_cleaned.json`, sha `f4719695…`) rồi so
> băm: **0/750 crop chặt khác byte, 0/33 tờ lưới chặt khác byte**, và
> `sample.json` khớp từng trường cũ. Phần thêm vào là thuần cộng thêm.

Trang duyệt được **vá tại chỗ** (không dựng lại, để nhãn đã gán và localStorage
không suy suyển): bấm vào ảnh đổi khung chặt ↔ rộng, khung rộng viền đỏ.

**Bài học rộng hơn ô này:** khung cắt là một **tham số đo**, và nó phải được
chọn theo *câu hỏi*, không theo *đối tượng*. Cùng một linh kiện cần hai khung
khác nhau cho hai câu hỏi khác nhau. Chỗ nào còn cắt crop để hỏi một câu về
linh kiện thì đáng kiểm lại điều này.

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

**Cập nhật 2026-09-06:** 34 bo này đã tham gia khảo sát và chọn ngưỡng.
Tách lại một phần sau khi đã xem không biến chúng thành tập nghiệm thu chưa
từng được sử dụng. Có thể chia để kiểm tra nội bộ; nghiệm thu cần bo/nhóm
thiết kế chưa dùng để ra quyết định, hoặc phải khai rõ mức độ đã tiếp xúc.

Quy tắc chia cho dữ liệu mới, chốt **trước** khi đo:

| | tập | dùng làm gì |
|---|---|---|
| **Hiệu chỉnh** | 34 bo khảo sát hiện tại; có thể chia nội bộ | chọn ngưỡng, xây luật và xem lỗi |
| **Nghiệm thu** | nhóm bo mới, **khoá** | chỉ chạy SAU khi code, cấu hình và ngưỡng đã đóng băng |

- **Chia theo BO, không theo box.** Hai box cùng một bo có cùng ánh sáng, cùng
  độ phóng đại, cùng người gán — chia theo box thì tập nghiệm thu chỉ đo lại
  tập hiệu chỉnh. Dự án đã chia theo bo ở detector và 6.1; giữ nguyên nguyên
  tắc đó.
- **Cố định danh sách bo nghiệm thu vào file**, không chọn lại mỗi lần đo.
- Các tile, crop và ảnh chụp lặp của cùng một bo phải ở cùng nhóm. Khi đo
  khả năng thêm mẫu PCB mới, nhóm theo thiết kế/revision để tránh cùng
  footprint/layout xuất hiện ở cả hai phía. Với một recipe cố định, đánh giá
  riêng bằng các lần chụp/bo vật lý chưa dùng lúc enroll.
- **Ngưỡng đóng băng trước, đo sau.** Đo rồi chỉnh ngưỡng rồi đo lại trên cùng
  tập nghiệm thu thì tập đó thành tập hiệu chỉnh thứ hai.

Và phải đo **hai đường, báo cáo riêng**:

| đường | nguồn nhãn họ | trả lời câu gì |
|---|---|---|
| `--families truth` | nhãn tay của fixture | *luật* đúng bao nhiêu, tách khỏi lỗi 6.1 |
| `--families model` | 6.1 chạy thật | tác động của nguồn họ dự đoán lên quyết định luật |

Cổng có sẵn cả hai cờ, kết hợp với `--leads truth|model|none`. Dùng cùng pad
gán tay làm đầu vào `--leads truth` và mục tiêu đo chỉ là chẩn đoán khi đã
biết đáp án; không phải phép đánh giá độc lập. Dù cả hai nguồn là `model`,
script hiện vẫn chỉ đo nhánh hình học trước fusion (§9.0d). Quyết bật luật
cần cổng cuối pipeline với cả hai nguồn dự đoán thật; mọi chế độ `truth`
chỉ dùng để định vị lỗi.

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

**(a) cài 2026-09-05; (b) cài 2026-09-06 — CẢ HAI đã xong.**

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
| **b** ✅ | truyền `lead_edges` + hướng xuyên suốt xuống 5.5 (§10.1) | vừa | dựng ROI đúng cạnh thật |

(a) **không thay được** (b) — nó chỉ đóng lỗ hổng, không dùng được thông tin đã
đo. Nhưng (a) rẻ và an toàn tuyệt đối theo nguyên tắc ở đầu §8, nên làm trước.

> **✅ (b) xong 2026-09-06.** `lead_edges` đi xuyên suốt: bộ luật →
> `package_profile` → `SolderJointCropper.derive` → `derive_solder_joints`, và
> nhánh hai-cạnh giữ đúng dải nằm trên **cạnh đo được** thay vì hai cạnh dài.
>
> Ánh xạ cục bộ→ảnh tính bằng **pháp tuyến ngoài xoay theo `frame.angle`**, nên
> nó đúng cả với linh kiện xoay chéo — CAD fusion dựng frame từ toạ độ land đã
> đăng ký nên góc có thể bất kỳ. Khi cạnh đo được **không** quy về đúng một cặp
> đối thì trả `None` và lùi về hai cạnh dài — **không đoán**.
>
> Chính mẫu từng cho độ phủ pad 4/4 → **0/4** giờ cho 4/4 → **4/4**.
>
> ⇒ Chốt tạm (a) `require_leads_on_long_axis` **đã tắt mặc định**, đúng điều
> kiện gỡ mà mục này đặt ra từ đầu: *"tắt khi và chỉ khi 5.5 đã biết đọc cạnh
> thật"*. Cờ vẫn còn để bật lại nếu đường truyền cạnh bị gỡ.

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

**Hệ quả của lỗi cũ (đã sửa ở §9.0b):** con số "0 mất pad, ROI 90 → 90, pad 28 → 28" báo cáo
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

#### 9.0e. ĐÃ GIẢI — đo mép ngoài thay vì đo tâm (2026-09-06)

> Đánh số **9.0e** chứ không 9.0d: §9.0d đã được phiên song song tham chiếu tới với nghĩa khác (cổng mới đo nhánh hình học *trước* fusion), nên để trống chỗ đó cho họ.

Thử **cả năm cách** quy một chân về cạnh, trên 3–4 ảnh thật với lead detector
thật (1.972 chân đã gán về thân):

| cách | chân có cạnh | 2 cạnh đối | 4 cạnh |
|---|---:|---:|---:|
| tâm ra ngoài *(bản cũ)* | 683 | 16 | 0 |
| **mép ngoài ra ngoài** | **1.086** | **22** | **2** |
| ≥25% diện tích ra ngoài | 966 | 17 | 0 |
| ≥50% diện tích ra ngoài | 745 | 16 | 0 |
| cạnh gần nhất, luôn gán | 1.972 | 35 | 6 |

**Cổng không phân biệt được năm cách này** — cả năm đều PASS với 0 mất pad, vì
fixture chỉ có một bo và quá ít IC có chân. Chọn "cạnh gần nhất" chỉ vì nó phủ
cao nhất là đúng kiểu đổi hỏng lấy hỏng. Nên đo **thẳng rủi ro** thay vì đoán:

| chân nằm đâu | n | % |
|---|---:|---:|
| tâm đã ở ngoài thân | 683 | 34,6% |
| **tâm trong, mép VẮT QUA mép thân** | **403** | **20,4%** |
| nằm hẳn trong thân, nông | 663 | 33,6% |
| nằm hẳn trong thân, **sâu ≥25% nửa bề ngang** | 223 | 11,3% |

Nhóm "vắt qua mép" **đều thò ra ngoài thân thật**, tức đều là ứng viên fillet
hợp lý — đúng thứ bước 2 sinh ra khi nó cắt cửa sổ *quanh* linh kiện. Còn "cạnh
gần nhất" nhận thêm 886 chân **nằm hẳn trong** thân, trong đó **223 nằm sâu**:
không chân thật nào nằm sâu trong lòng thân, nên đó là nguồn **topology giả** —
mà topology giả thì 5.5 thu ROI về hai cạnh và mất vùng kiểm thật (§8.3).

**✅ Chọn "mép ngoài", đã áp dụng.** Kèm một chốt: hộp chân bao *trọn* thân thì
thò ra cả bốn phía và `max` chọn bừa, nên đòi thêm tâm chân phải lệch về đúng
phía cạnh đó.

**Kết quả trên fixture, với lead detector thật:**

| | trước | sau |
|---|---:|---:|
| chân rơi vào TRONG thân | **18 / 60** | **0** |
| luật bỏ qua | 15 / 39 | **3 / 39** |
| pad baseline mất | 0 | **0** |

Ba ca bỏ qua còn lại đều có nội dung: 2 con `ic` chỉ thấy chân ở **một** cạnh,
1 con bị **§8.2** chặn (thân gần vuông, nghi QFP nhìn nửa vời). Tức nhánh `ic`
**đã chạy thật** — lần đầu tiên.

> Con số 18 → 0 được chốt bằng test, không chỉ ghi ở đây: nó là ô đã chặn nhánh
> `ic` suốt, và nếu quay lại thì mọi kết luận về nhánh đó mất hiệu lực.

#### 9.0f. Soát lại — và một con số của tôi phải đính chính (2026-09-06)

Rà lại toàn bộ đường liên quan sau ba thay đổi trong ngày. Ba kết quả:

**1. "ROI chỉ thêm, không bớt" chỉ đúng Ở `derive`, không đúng đầu-cuối.**
Cổng báo ROI 90 → 92 trên fixture. Chạy thật đầu-cuối trên một tile 204 linh
kiện, đo ROI **sau fusion**:

| | luật TẮT | luật BẬT |
|---|---:|---:|
| ROI sau fusion | 988 | 985 |
| ROI cũ **không còn** trong tập mới | — | **82** |
| ROI mới xuất hiện | — | 79 |

Tức bật luật **dịch chỗ 82 ROI**, không phải chỉ thêm. Cổng không thấy vì nó đo
ở `derive`, trước `fuse_detected_leads`/`fuse_solder_rois`/`deconflict` — **đúng
điểm phiên song song đã nêu** (§9.0d, chỗ để trống cho họ). Câu "ROI chỉ thêm"
trong §6.3d và §9.0e phải đọc kèm phạm vi đó.

**2. Nhưng độ phủ mối hàn THẬT không đổi.** Đo trên 2 tile, 601 mối hàn do lead
detector tìm được, ngưỡng phủ 50%:

| | luật TẮT | luật BẬT |
|---|---:|---:|
| mối hàn được phủ | 599 / 601 (99,7%) | **599 / 601 (99,7%)** |

82 ROI dịch chỗ nhưng **không bỏ rơi mối hàn nào** — vì fusion neo ROI lên chân
đo được. Đây là phép đo đầu-cuối đầu tiên của đường luật trên bo thật, và nó là
thứ cổng hiện chưa làm được.

**3. Một lỗi ưu tiên tiềm ẩn, đã sửa.** `cad_fusion` thay `package_profile` và
`terminal_geometry_override` bằng bằng chứng CAD nhưng **không xoá**
`terminal_lead_edges` mà luật ảnh đã ghi. Trong một lượt chạy thì chưa lộ —
`derive` xong trước khi CAD ghi — nhưng detection bị sửa **tại chỗ** và
`last_package_detections` giữ chính object đó, nên gọi `make_solder_crops` lần
nữa trên chúng sẽ ghép **hình học của CAD với cạnh của luật ảnh**. Đã xoá, có
test canh (gỡ bản vá ra thì đỏ).

> Kèm theo: docstring của `build_pad_fixture.py` vẫn nói nhánh `ic` bế tắc **vì
> quy ước box cũ**. Nguyên nhân đó đã được chứng minh là sai (§9.0c/§9.0e) — đã
> sửa lại. Fixture mới vẫn đáng làm, nhưng để đo **độ phủ pad**, không phải để
> mở nhánh `ic`.

**Việc thật lộ ra từ đây, thay cho "dựng thêm fixture":**

1. ~~Lead detector / pass 2 trả mối hàn đè lên thân.~~ **✅ ĐÃ GIẢI 2026-09-06 —
   §9.0e.** Nới `_edge_of` sang đo **mép ngoài**; chân rơi vào trong thân
   18 → 0, luật bỏ qua 15 → 3.
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

## 10. Kế hoạch tiếp theo — cập nhật review 2026-09-06

Các sửa lỗi ở §9.0b, nguồn `--leads truth` ở §9.0c, luật capacitor ở §6.3d
và phép đo mép ngoài ở §9.0e đã có trong code. Việc còn lại là kiểm được tác
động trên ROI cuối cùng, xác minh chất lượng bằng chứng cạnh và truyền
topology tới nơi sử dụng. **Luật ảnh vẫn mặc định TẮT.**

### 10.1. Truyền topology có kiểu dữ liệu, giữ hợp đồng đang có

`resolve_packages_by_rule()` hiện trả `PackageClassification`, **không phải
chuỗi trần**. Kết quả đã có `source`, `decision` và `metadata["lead_edges"]`.
Thông tin cạnh bị mất khi `AOIPipeline.apply_package_classifications()` dựng
lại `FootprintProfile` chỉ từ `package_class`; 5.5 sau đó chỉ đọc kiểu hình học.
Vì vậy, không thay API bằng một phép đổi `str → FootprintProfile`.

Thêm một payload topology có kiểu dữ liệu, **tùy chọn**, nối qua
`PackageClassification` → detection/profile → bộ dựng ROI và recipe. Dùng
lại phần pin count đã có của `FootprintProfile`; xác định rõ các trường:

| trường | hợp đồng cần chốt |
|---|---|
| `lead_edges` | cạnh cụ thể, không chỉ số cạnh; chưa biết phải biểu diễn được |
| `coordinate_space` | cạnh đang thuộc hệ tọa độ ảnh Golden hay hệ cục bộ linh kiện |
| `angle_deg` / transform | phép đổi giữa hai hệ; có quy ước dấu và đơn vị rõ ràng |
| `source` | nguồn bằng chứng và định danh tương ứng: CAD, footprint, Golden đã duyệt, luật ảnh |
| `decision` | accept/review/unknown; độc lập với xác suất hoặc độ tin phân tích tên |
| quyền thay đổi ROI | bằng chứng này chỉ được gợi ý, được bổ sung hay được thay tập ROI đã duyệt |

`reason` là lời giải thích, **không thay cho nguồn**; `confidence` không thay
cho quyết định duyệt. Không suy hướng của linh kiện từ tên footprint nếu tên
đó không mang thông tin hướng đặt. Chưa cần thêm `mount_type` nếu không có
hành vi nào sử dụng nó.

Bảy slug §2 giữ nguyên làm alias v1. JSON cũ thiếu topology vẫn đọc được;
round-trip không mất cạnh/hướng/nguồn/quyền. Chỉ bỏ chốt trục dài §8.3 sau
khi test xuyên suốt chứng minh cạnh đo được đến đúng ROI, kể cả linh kiện
xoay, chân ở hai cạnh ngắn và dữ liệu cũ không có topology.

### 10.2. Chốt topology trong Golden đã duyệt, triển khai song song

Người dùng **đã xác nhận kiểm cố định vài mẫu PCB và có thể thêm mẫu mới**.
Do đó, xây đường topology đã duyệt cùng với chẩn đoán pass 2; không đợi luật
ảnh phân được IC hoặc capacitor mới bắt đầu phần Golden.

Khi enroll, lấy CAD/pad hoặc footprint/PnP có hướng làm bằng chứng nếu có;
người duyệt chốt topology cùng tập ROI cần kiểm. Luật ảnh có thể gợi ý và ghi
bất đồng. Khi inspection, recipe đã duyệt là cấu hình đang có hiệu lực:
thiếu chân quan sát được không được tự giảm tập ROI đó. Dữ liệu CAD mới mâu
thuẫn với recipe phải qua lần sửa/duyệt recipe, không tự thay giữa một lượt
kiểm. `unknown` giữ baseline và chuyển review theo policy.

Công việc cụ thể:

1. Bổ sung topology và nguồn duyệt vào contract slot/recipe, giữ tương thích
   schema cũ; lưu hệ tọa độ và quan hệ với ROI của từng slot.
2. Nối đường ROI mối hàn với dữ liệu recipe bằng API dùng chung; hiện
   `AOIPipeline` và Golden inspection chưa chia sẻ contract topology này.
3. Test load/save và inspection: biến mất bằng chứng ảnh, sai nhãn họ hoặc
   đổi số chân pass 2 không làm mất ROI đã duyệt; bất đồng được ghi lại.
4. Với package ẩn chân được xác nhận, báo phần mối hàn là **không kiểm được
   bằng ảnh trên xuống**, không suy thành kết quả mối hàn PASS.

Người dùng đã yêu cầu tiếp tục triển khai kế hoạch. Tiến hành phần contract,
Golden và kiểm thử; giữ luật ảnh mặc định tắt trong khi chưa đạt nghiệm thu.

### 10.3. Cổng phải đo ROI cuối cùng và mức độ đã kiểm

Lỗi bỏ qua `terminal_geometry_override` đã được sửa bằng cách gọi
`SolderJointCropper.derive`. Tuy nhiên, đây mới là **bước dựng ROI**: runtime
còn áp nhãn họ, hợp nhất chân phát hiện/CAD và xử lý ROI chồng lấn. Giữ phép
đo hiện tại để chẩn đoán riêng bước dựng; thêm phép đo qua đường xử lý chung
đến **tập ROI cuối cùng**. Không chép lại các bước runtime vào script.

Baseline và nhánh thử dùng cùng ảnh, detections, kết quả 6.1, chân, CAD và
cấu hình; chỉ thay can thiệp package đang đánh giá. Đặc biệt phải áp nhãn họ
như runtime trước khi so, kể cả detector chỉ trả nhãn `component`.

Cổng cần xuất được:

- Pad baseline nào bị mất, pad nào mới được phủ, recall tuyệt đối và số/diện
  tích ROI thừa; gắn pad với linh kiện để ROI của hàng xóm không che lỗi gán.
- Kết quả theo board, topology và nguồn họ/chân; số trường hợp luật thực sự
  quyết định, số trường hợp làm thay đổi ROI, lý do review/abstain/skip.
- Hai kết luận riêng: **không giảm độ phủ trên tập đã đo** và **đủ bằng chứng
  nghiệm thu hay chưa**. Thiếu nguồn hoặc nhánh cần kiểm chưa chạy phải là
  `inconclusive` về nghiệm thu; không dùng exit 0 hay `passed: true` đơn lẻ
  làm quyền bật luật.
- Manifest tái lập: ảnh/nhãn và split, commit, hash model, cấu hình/ngưỡng,
  nguồn họ và nguồn chân. Báo cáo 28/28 cũ tại
  `Docs/bench/package_rule_gate_20260905.json` là bản trước sửa cổng, không
  dùng làm baseline hiện hành.

**Điều kiện bắt buộc:** không mất bất kỳ pad nào đã được baseline phủ. Điều
kiện đó chưa đủ: phải có coverage tuyệt đối và mức thực thi nhánh đạt tiêu
chí đã ghi trước khi mở tập nghiệm thu. Test cổng phải có ca giảm pad thật
qua đường cuối, ca chạy nhưng abstain, ca thiếu đầu vào và ca không có mẫu
cho topology cần nghiệm thu.

### 10.4. Thứ tự và sản phẩm bàn giao

| bước | sản phẩm cụ thể | nghiệm thu / phụ thuộc |
|---|---|---|
| 1 | Bổ sung cổng ROI cuối và trạng thái đủ/chưa đủ bằng chứng (§10.3) | Bắt được mất pad sau fusion/deconflict; không nhầm skip với đã kiểm |
| 2A | Kiểm chứng từng IC sau phép đo mép ngoài §9.0e: overlay, box thân, crop, chân gán và cạnh | So với chân/cạnh đã duyệt; đếm cạnh tăng không thay cho precision/recall, kiểm thêm gán nhầm và trùng detection |
| 2B | Contract topology tối thiểu và đường Golden đã duyệt (§10.1–10.2) | Làm song song 2A; test serialization, biến đổi cạnh/hướng và giữ ROI đã duyệt |
| 3 | Fixture và manifest hiệu chỉnh/nghiệm thu theo **bo vật lý** | Nhiều ảnh/crop của cùng bo không được sang hai tập; nhãn/pad đủ để đo các topology mục tiêu |
| 4 | Đo tác động của phép đo mép ngoài và luật capacitor bằng chế độ chỉ ghi nhận, so baseline | Giữ những sửa lỗi đã có; chỉ đổi tiếp thuật toán sau khi 2A/fixture chỉ ra nguyên nhân; bất định thì giữ ROI cũ |
| 5 | Đóng băng cấu hình; chạy tập khoá trên cả nguồn họ và chân truth/model | Bốn tổ hợp tách lỗi luật/6.1/pass 2; kết quả model/model mới đại diện đường ảnh runtime |
| 6 | Báo cáo quyết định theo từng topology và đường bằng chứng | Chỉ kết luận trong phạm vi đã đo; chưa đạt hoặc chưa được đo thì tiếp tục TẮT |

Bước 2A phải kiểm **chân duy nhất và chủ sở hữu** khi đánh giá bằng chứng
cạnh sau sửa §9.0e. Code hiện đếm từng detection; hai box trùng cùng một mối hàn
có thể đủ ngưỡng “hai chân”. `parent_detection_id` do pass 2 ghi lại cũng
chưa được hàm gán chân sử dụng. Đây là các khả năng đã tái hiện bằng mẫu tổng
hợp; chưa phải kết luận rằng chúng gây ra lỗi trên fixture hiện có.

Khi so phương án, bằng chứng ảnh chưa được nghiệm thu chỉ đề xuất topology
hoặc báo review, không tự xóa ROI baseline/Golden. Giữ các chốt không suy
ẩn chân từ sự vắng mặt và không ép IC gần vuông thành hai cạnh. Những họ
chưa có ánh xạ an toàn giữ fallback hiện tại; chưa cần người dùng chọn thêm
lớp thứ tám để làm các bước trên.

Luật capacitor đã có kết quả khảo sát mới ở §6.3d với 169 nhãn tiền gán.
Cần duyệt nhãn và kiểm lại trên bo mới trước khi coi đó là bằng chứng nghiệm
thu độc lập; việc đó không chặn contract topology, cổng hay đường Golden.
Không dùng lại con số đo sai phạm vi ở §6.3b để chọn ngưỡng.

---

## 11. Đầu vào bên ngoài còn thiếu và quyết định đã có

**Đã có câu trả lời:** kiểm cố định vài mẫu PCB, cho phép thêm mẫu mới;
Golden vẫn gắn với ảnh chụp thật và dùng hướng ghép sơ đồ. Không hỏi lại các
quyết định này. Thứ tự xử lý lỗi, thiết kế payload và chọn phép chẩn đoán là
công việc kỹ thuật có thể tiếp tục ngay.

Những đầu vào còn cần thu thập cho nghiệm thu:

1. **Nhãn đã duyệt.** Bộ 750 nhãn là bản tiền gán, chưa phải ground truth.
   Ưu tiên các IC, toàn bộ 169 mẫu họ capacitor của đợt bổ sung §6.3d
   (85 trụ đứng, 84 chip theo bản tiền gán), các ca chưa chắc và pad của
   fixture dùng nghiệm thu. Mốc 103 mẫu là thống kê trước đợt bổ sung.
2. **Bo/ảnh dây chuyền cùng dung sai thực tế.** Cần biết bo vật lý nào, lần
   chụp nào và điều kiện ảnh để chia tập không rò rỉ; tiêu chí coverage và
   phạm vi topology cần nghiệm thu phải chốt trước khi đo tập khoá.
3. **Nguồn thiết kế nếu lấy được:** BOM/PnP có footprint và hướng, hoặc
   IPC-D-356/CAD pads. Đây là nguồn giúp giảm việc gán tay, không phải điều
   kiện bắt buộc để làm topology Golden đã duyệt bằng tay.
4. **Yêu cầu kiểm package ẩn chân trên bo thực.** Nếu cần kiểm mối hàn bị
   che khuất, phải xác định phương pháp kiểm bổ sung; ảnh trên xuống không
   cung cấp bằng chứng cho mối hàn đó.

Chưa có các đầu vào nghiệm thu này vẫn tiếp tục được bước 1 và 2; không
biến danh sách thành một vòng xin xác nhận trước khi làm công việc kỹ thuật.

---

Xem thêm: `Docs/bao_cao/tien_do_detect_2_luot.md` (bảng công việc sống),
`Docs/danh_gia/danh_gia_khoanh_box_than_linh_kien.md` (báo cáo box),
`Docs/ke_hoach/ke_hoach_pcb_defect_toan_mach.md` (kế hoạch lỗi toàn mạch).
