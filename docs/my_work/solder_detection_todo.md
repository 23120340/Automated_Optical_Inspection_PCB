# Detect mối hàn — sửa như thế nào, và việc bạn phải làm

> Viết 2026-09-07. Số liệu trong tài liệu này đều đo được, nguồn ghi ở từng chỗ.

## Trả lời ngắn

**Không phải model mối hàn kém.** Cái nhìn "lượt 2 sai chân hàn nghiêm trọng"
thực ra là ba chỗ hỏng khác nhau chồng lên nhau, và chỗ nặng nhất **không nằm ở
lượt 2**:

| # | chỗ hỏng | bằng chứng | sửa bằng | ai làm |
|---|---|---|---|---|
| 1 | **Lượt 1 khoanh nhầm pad tròn** → ROI mối hàn suy ra từ box rác | 20/62 box sai trên bo dự án; **59/220 = 27% ROI** sinh ra từ chúng | thêm ảnh bo dự án vào tập train | **bạn** (ảnh + khoanh) |
| 2 | **Model mối hàn chưa từng được đo ở miền đang chạy** | 9.089 box nhãn đều nằm trên crop `fpic`/`winnies`; **0 box** trên tile PCB-DSLR hay bo dự án | khoanh mối hàn trên crop của bo thật | **bạn** (khoanh) |
| 3 | **Hình học ROI đang chạy chế độ chung** vì luật package 5.2 tắt | tập nghiệm thu 750 box đã gán 92%, còn **121 ô khó** | duyệt nốt 121 ô rồi mới bật luật | **bạn** (duyệt) |

Cả ba đều nghẽn ở **dữ liệu**, không nghẽn ở code. Train lại model mối hàn mà
không có mục 1 thì ROI rác vẫn còn nguyên; không có mục 2 thì train xong cũng
không biết tốt lên hay xấu đi.

---

## 1. Phần lớn "lượt 2 sai" là lỗi của lượt 1

Đo trên `00001__1024__1648___4120.png`
([chi tiết](../evaluation/loi_pad_tron_bo_du_an.md)):

| chạy lượt 2 trên | chân sinh ra | ROI mối hàn |
|---|---:|---:|
| cả 62 box của lượt 1 | 73 | 220 |
| chỉ 42 box **đúng** | 72 | 161 |
| chỉ 20 box **sai** | **1** | **59** |

Model mối hàn **từ chối** gần hết box sai — 72/73 chân đến từ box đúng. Nhưng
ROI thì vẫn được **suy ra bằng hình học** quanh cả 20 box sai, và chúng nằm trên
pad trống. Đó chính là thứ trông như "chân hàn sai tùm lum".

⇒ **Sửa lượt 1 thì 27% ROI rác biến mất theo**, không cần đụng gì tới model mối
hàn.

Vì sao lượt 1 sai: tile đó là **bo của chính dự án**, không nằm trong 120 tile
PCB-DSLR đã train. Bo PCB-DSLR gần như toàn linh kiện dán, nên detector **chưa
từng thấy pad xuyên lỗ lộ thiên** làm ví dụ âm. Trên miền của nó, bộ nhãn đạt
recall 94% — nhãn không có lỗi.

## 2. Model mối hàn: số đẹp, nhưng đẹp ở miền khác

`models/active/lead_detector` — mAP50 **0,9912**, recall **0,9768** trên 25 cảnh
test khoá. Số thật, đo đàng hoàng. Nhưng:

- toàn bộ **9.089 box** nhãn nằm trên **crop từng linh kiện** cắt từ hai bộ công
  khai `fpic` (1.044 crop) và `winnies` (987 crop);
- **không có box nào** trên tile PCB-DSLR, **không có box nào** trên bo dự án.

Nên câu "ROI mối hàn tốt tới đâu trên bo đang chạy" hiện **không trả lời được** —
không phải vì chưa đo, mà vì **chưa có gì để đo bằng**.

Hai điều dễ nhầm, nói luôn:

- **Mặc định pipeline KHÔNG nạp model này** (`lead_detection.model_path = None`).
  Nếu bạn không chọn nó trong mục "Lượt 2" của app thì bước 5.5 đang chạy
  **thuần hình học**. Sai mà bạn thấy có thể chưa liên quan gì tới model.
- Model này khoanh **mọi** mối hàn kể cả mối hàn tốt. Nó **không** phán quyết
  đạt/hỏng — việc đó của bước 6.2.

## 3. Hình học ROI đang ở chế độ chung

Luật phân nhóm package (5.2) **mặc định tắt**, nên `terminal_geometry` rơi về
suy đoán chung: chip hai chân và tụ hoá đứng nhận cách rải ROI gần như nhau.
Luật đã đo được 90,5% trên tách họ tụ, nhưng tập nghiệm thu **chưa khoá theo
bo**, nên chưa đủ căn cứ bật.

Chặn ở đúng một chỗ: tập nghiệm thu 750 box còn **121 ô `XEM_KY`** — xem việc 5.

---

# Việc của bạn

Xếp theo thứ tự phụ thuộc. Việc 1 chặn việc 2 và 3. Việc 4 **đã xong**; việc 5
làm được ngay hôm nay, không phải chờ gì.

## 🟡 Việc 1 — Ảnh bo dự án — **đã chụp, mặt sau cần chụp lại**

**Cập nhật 07/09: bạn đã chụp 31 ảnh, cả hai mặt.** Đủ để bắt đầu, nhưng hai mặt
không cùng chất lượng:

| | số ảnh | kích thước | tile 1024 cắt được |
|---|---:|---|---|
| `real_pcb/phone/front_side/` | 16 | 1920×2560 (4,9 MP) | **104 tile**, trung vị 13 linh kiện |
| `real_pcb/phone/back_side/` | 15 | **960×1280 (1,2 MP)** | ~1 tile mỗi ảnh, và **không đủ 1024 px chiều ngang** |

**Mặt trước dùng được ngay.** Mặt sau thì không: 960 px chiều ngang **nhỏ hơn
chính khung tile 1024**, nên mỗi ảnh chỉ ra được một mảnh 960×1024 chứ không phải
tile vuông. Chi tiết trên mỗi linh kiện cũng chỉ bằng khoảng một nửa mặt trước.

**Việc còn lại:** chụp lại **mặt sau** ở cùng độ phân giải mặt trước (1920×2560
trở lên). Cùng máy, cùng cách chụp — chỉ là lần chụp mặt sau bị để ở chế độ ảnh
nhỏ.

**Bắt buộc có ảnh chứa pad xuyên lỗ** — đó đúng là lớp đang hỏng. Ảnh toàn linh
kiện dán không dạy được gì mới.

*Nghiệm thu:* mặt sau ra được ≥ 10 tile 1024 px vuông, mỗi tile ≥ 5 linh kiện.

### Đã cắt xong mặt trước, và **không có vùng cháy nào đáng cắt bỏ**

104 tile 1024 px từ 16 ảnh mặt trước → `datasets/test_images/project_board_tiles/`,
trung vị 13 linh kiện mỗi tile. Sẵn sàng cho việc 2.

Bạn nhờ cắt bỏ các vùng cháy sáng. Đã đo từng tile, và câu trả lời là **gần như
không có gì để cắt**:

| | |
|---|---:|
| tile cháy nhiều nhất | **4,53%** |
| trung vị | 0,03% |
| tile vượt 5% | **0** |
| ảnh cháy nhiều nhất trong 31 ảnh | 2,16% |

Xem tận mắt bằng ảnh chồng mặt nạ: chỗ bị chấm là **connector nhựa kem, vỏ kim
loại cổng USB, vòng pad mạ vàng và dải mạ ở mép bo** — toàn thứ vốn sáng, và
**không có vệt loá nào nằm trên thân linh kiện**. Đốm lớn nhất trong cả bộ
(58.840 px, ảnh `front_side/9`) là **nhãn giấy trắng dán trên bo**, không phải
loá.

Nên cắt bỏ theo tile là sai công cụ ở đây: tile "cháy" nhất chỉ cháy ở dải mạ mép
bo, nửa còn lại vẫn đầy linh kiện dùng được. Ném nó đi thì mất nhiều hơn được.

Phép đo vẫn được giữ lại và ghi vào manifest từng tile
(`blown_fraction`), kèm cổng loại `--max-blown-fraction` cho lần chụp nào tệ hơn.

> **Một lần đo hụt, ghi lại để khỏi lặp.** Bản đầu của phép đo chấm một tile là
> "cháy 16%" — hoá ra vùng đó là **tản nhiệt nhôm anod xanh**: kênh lam chạm trần
> trên mặt phẳng lì. Mặt màu bão hoà không phải mặt cháy. Đã thêm điều kiện độ
> sáng tổng; sau khi sửa, tile đó không còn nằm trong nhóm cháy nhất, và nửa dưới
> của nó — có BGA và pad xuyên lỗ mạ vàng — được giữ lại.

## Việc 2 — Khoanh **thân linh kiện** trên tile bo dự án

Sau khi tôi cắt tile và dựng app khoanh, bạn mở `label_boxes.html` trong trình
duyệt.

**Quy ước (giống hệt vòng trước, đừng đổi):** khoanh sát **THÂN / gói / vỏ** linh
kiện. **Không** bao chân, **không** bao pad, **không** bao thiếc.

- `Enter` = duyệt tile sau khi đã sửa đủ mọi box
- `C` = tile thật sự không có linh kiện nào
- Xong thì bấm **Xuất JSON**, giữ file làm checkpoint

**Điểm quan trọng nhất của việc này:** pad tròn xuất hiện trong ảnh mà **không có
nhãn** → nó thành **ví dụ âm**. Đó là thứ duy nhất dạy được model "vòng đồng tròn
≠ linh kiện". Nên **đừng** khoanh pad lại "cho đủ".

*Nghiệm thu:* 10–20 tile ở trạng thái `verified`.

## Việc 3 — Khoanh **mối hàn** trên crop cắt từ bo dự án

Việc này **phải chờ việc 2**: crop từng linh kiện được cắt ra từ chính box thân
bạn vừa khoanh.

**Quy ước:** box phải trùm **pad + fillet (chân thiếc loe ra)**, không chỉ khoanh
phần kim loại sáng. Lần khoanh đầu tiên của dự án chỉ khoanh phần kim loại, và
model học ra box chỉ bằng **0,76 lần** diện tích pad — cắt mất đúng thứ bước 6.2
cần chấm. Đã sửa rồi, đừng lặp lại.

Một lớp duy nhất: `solder_joint`, cho **mọi** mối hàn kể cả mối hàn tốt.

*Nghiệm thu:* ≥ 200 box mối hàn trên crop của bo dự án, ở trạng thái `verified`.

## ✅ Việc 4 — Khoanh thân trên tile PCB-DSLR — **ĐÃ XONG**

> **Đính chính 2026-09-07.** Bản đầu của tài liệu này ghi "16/120 tile, còn 104
> tile". **Sai.** Tôi đọc `draft_boxes.json` — đó là file *seed* mà app được dựng
> từ đó, không phải tiến độ. File tiến độ thật là `joint_boxes_cleaned.json`,
> cùng `dataset_id` `65af1c1a6b21e37d`, cùng 120 crop:
>
> | | tile | box | trạng thái |
> |---|---:|---:|---|
> | `draft_boxes.json` *(seed)* | 120 | 3.855 | 16 verified, 104 trống |
> | `joint_boxes_cleaned.json` *(thật)* | 120 | **9.486** | **95 verified + 25 unusable** |
>
> 120/120 tile đã xử lý hết. **Không còn việc gì ở mục này.**

## 🟡 Việc 5 — Gán package — **ĐÃ LÀM 92%, còn 121 ô khó**

> **Đính chính 2026-09-07.** Bản đầu ghi "0/120 tile, 3.855 box còn `unknown`",
> và ngụ ý chưa ai làm gì. **Sai ở cả hai vế.**

**Nhầm thứ nhất — nhầm sản phẩm.** `label_packages.html` /
`draft_package_boxes.json` (3.855 box `unknown`) được dựng từ **seed 16-verified**,
tức trên hình học đã cũ. Nó **đã bị thay thế** bởi cách làm phân tầng, và
**không nên gán tay 3.855 box đó** — công bỏ ra không đổi lấy được gì.

**Nhầm thứ hai — việc đã làm rồi.** Bộ thật là `datasets/survey/family_package_review_20260904`:
**750 box** rút phân tầng từ chính 9.486 box của việc 4, gán nhãn **HỌ + GÓI**.

| lượt | family còn `XEM_KY` | package còn `XEM_KY` |
|---|---:|---:|
| bạn duyệt (04–05/09) | 269 | 107 |
| tôi gán lượt 1 (06/09) | 140 | 59 |
| tôi gán lượt 2, cắt cửa sổ rộng hơn để thấy silkscreen | **83** | **59** |

464 ô do bạn gán, 286 ô do tôi. Kết quả **nằm ở `Downloads`, chưa vào repo** —
đúng như bạn nhớ là thất lạc. Đã cứu về:
`datasets/survey/family_package_review_20260904/labels/family_package_01_claude_v2.json`
(và bản bạn duyệt gốc là `family_package_00_ban_duyet.json`).

> `.gitignore` đang chặn nhầm: `datasets/survey/` loại cả thư mục, mà git
> **không cho include lại file nằm trong thư mục đã loại** — nên ba dòng `!` bên
> dưới im lặng vô tác dụng. Đã sửa; file nhãn mới thêm giờ mới thực sự vào repo.

### Còn lại của bạn: 121 ô

```
datasets/survey/family_package_review_20260904/review_con_lai_kho.html
```

Trang này **chỉ chứa 121 ô còn `XEM_KY`** (83 ở HỌ, 59 ở GÓI, 21 ô dính cả hai),
không phải cả 750. Mỗi ô mang sẵn nhãn đã gán, độ tin, và **lý do vì sao khó**,
nên bạn không phải đoán lại từ đầu. Mỗi ô có hai ảnh — bấm để đổi giữa bản cắt
sát và bản cắt rộng thấy silkscreen designator.

**Đừng điền bừa cho hết `XEM_KY`.** Tập này dùng để **đo** classifier 6.1; điền
nhãn đoán vào là làm hỏng chính phép đo. Ô nào thật sự không đọc được thì **để
nguyên `XEM_KY`** — đó là câu trả lời đúng, không phải việc còn dang dở.

Đây là thứ mở khoá mục 3 ở trên: có tập này mới dựng được nghiệm thu khoá theo
bo, có nghiệm thu mới đủ căn cứ bật luật 5.2.

---

# Việc của tôi (bạn không phải làm)

1. Cắt tile từ ảnh bạn chụp (`scripts/tile_test_images.py`).
2. Dựng app khoanh thân và app khoanh mối hàn cho bộ tile mới
   (`scripts/build_joint_box_app.py`, `scripts/crop_components_for_labelling.py`).
3. Đóng gói dataset YOLO, **tách tập giữ riêng theo bo** trước khi train
   (`scripts/pack_component_detection_dataset.py`,
   `scripts/pack_joint_detection_dataset.py`).
4. Train lại lượt 1 và lượt 2 trên Kaggle
   (`training/kaggle/pcb_component_detector_v3_kaggle.ipynb`,
   `training/kaggle/pcb_lead_detector_kaggle.ipynb`).
5. Đo lại **trên tập giữ riêng**, báo số box kiểu pad tròn trước/sau.

**Bước 5 là bắt buộc.** Không có tập giữ riêng thì không phân biệt được "đã sửa
được" với "model học thuộc thêm vài tile".

---

# Ba việc KHÔNG nên làm

**1. Đừng lọc nhãn bằng kết quả model.** (Ý tưởng: phủ kết quả detect lên nhãn
tay, giữ box trùng, xoá box không trùng.) Ba lý do đo được:

- recall 94% nghĩa là 6% — 36 box trong 584 — là linh kiện thật mà model **bỏ
  sót**; lọc kiểu đó xoá đúng 36 ca khó nhất, thứ duy nhất dạy được model;
- nó làm nhãn **đồng ý với model**: lần train sau điểm đẹp hơn, thực tế tệ hơn;
- 20 pad tròn kia **không nằm trong bộ nhãn** — chúng là nền. Xoá nhãn không xoá
  được thứ không có trong nhãn.

**2. Đừng nâng ngưỡng tin cậy để dọn box sai.** Đã đo: muốn sạch hết 20 box sai
phải lên ngưỡng 0,75, và khi đó chỉ còn **19/42 (45%)** linh kiện thật. Với AOI,
bỏ sót linh kiện tệ hơn hẳn một box thừa — đánh đổi sai chiều.
Lọc theo **hình dạng** cũng không được: tụ hoá nhìn từ trên **cũng tròn**.

**3. Đừng bật `lead_detector` làm mặc định.** Trên fixture 28 pad đo tay nó phủ
đủ 28/28 (bản trước lọt 2), nhưng độ phủ trung vị tụt 0,97 → 0,79 và pad thấp
nhất còn 0,52 — chỉ trên ngưỡng 0,50 đúng 0,02. Chưa đo được "box bám sát thiếc
hay box rộng trùm land tốt hơn cho 6.2", vì cần nhãn **lỗi** mà bo đó không có.

---

# Tóm tắt một dòng

Việc 1 (chụp ảnh) chặn tất cả. Làm được việc 1 → tôi cắt tile → bạn làm việc 2 →
tôi cắt crop → bạn làm việc 3 → tôi train và đo.

Việc 4 **đã xong**. Việc 5 còn **121 ô**, làm được ngay hôm nay.
