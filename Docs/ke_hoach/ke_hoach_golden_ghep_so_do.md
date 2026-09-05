# Kế hoạch: Golden một ảnh, sơ đồ ghép từ nhiều ảnh

> Cập nhật 2026-09-05. Thay cho `ghi_chu_golden_ghep_nhieu_anh.md` — ghi chú đó
> nêu ba hướng, hướng số 2 đã được chọn.

---

## 1. Quyết định

**Ảnh Golden vẫn là MỘT lần chụp thật**, chọn bằng `select_reference()` như
hiện tại. **Chỉ SƠ ĐỒ được ghép** từ nhiều lần chụp, mỗi lần đóng góp phần
board mà nó nhìn chính diện.

Ba hướng đã cân nhắc:

| hướng | giữ được truy xuất? | vì sao chọn/không |
|---|---|---|
| ghép ảnh, lưu kèm homography | ⚠️ gián tiếp | ảnh Golden thành ảnh tổng hợp; khi tranh chấp không chỉ được vào một lần chụp thật |
| **ghép SƠ ĐỒ, ảnh giữ nguyên** | ✅ **trọn vẹn** | **đã chọn** — ràng buộc của `enrollment.py` không bị đụng tới |
| đổi ống kính telecentric | ✅ | giải bằng phần cứng, đắt; vẫn nên hỏi giá song song |

Lý do hướng 2 thắng nằm ngay trong docstring của `aoi_pipeline/golden/enrollment.py`:

> *"The selector **deliberately** returns one of the supplied files. It **never
> blends**, stacks, or otherwise synthesises a reference image: a Golden image
> must remain **traceable to a real acquisition**."*

Ghép sơ đồ không vi phạm câu đó. Sơ đồ là **dẫn xuất có nguồn gốc từng phần** —
mỗi linh kiện ghi rõ nó đến từ khung chụp nào.

---

## 2. Vấn đề đang giải

Một lần chụp chỉ chính diện **một vùng** — vùng dưới trục quang. Càng ra xa
tâm, linh kiện càng bị nhìn nghiêng: thân che pad phía xa, và **tâm biểu kiến
lệch theo chiều cao linh kiện** (parallax).

Với AOI thì đó không phải chuyện thẩm mỹ: ROI mối hàn đặt theo hình học thân,
thân lệch thì ROI lệch, và 6.2 chấm một vùng không phải mối hàn.

---

## 3. Đã có gì, thiếu chiều nào

| | có | ở đâu |
|---|---|---|
| Nhận nhiều ảnh lúc enroll, **chọn** medoid | ✅ | `golden/enrollment.py`, `min_images = 3` |
| Cấu trúc sơ đồ: designator + mm + góc xoay + kích thước | ✅ | `placement/inspection_map.py::MapComponent` |
| Kế hoạch chụp nhiều khung, chồng lấn | ✅ | `plan_capture_regions()`, `CaptureRegion.center_mm` |
| Cắt crop theo khung chụp | ✅ | `crop_boxes_for_capture()` |
| Nắn về một hệ toạ độ chung | ⚠️ theo bo | `imaging/alignment.py` — nắn theo bo, **không** theo chiều cao linh kiện |
| **Dựng sơ đồ TỪ ẢNH** | ❌ | — |
| **Ghép sơ đồ từ nhiều khung** | ❌ | — |

`inspection_map.py` ghi thẳng trong bảng nguồn của nó:

> | Golden image | **Không trực tiếp.** Nó là ảnh, muốn ra toạ độ vẫn phải detect — mà detect chính là thứ ta đang muốn dẫn đường |

Đó là đúng cho bài toán *dẫn đường chụp*. Nhưng dự án này **không có CAD**, nên
chiều ngược lại — ảnh → sơ đồ — là chiều bắt buộc. Kế hoạch này bổ sung đúng
chiều đó, không sửa chiều cũ.

---

## 4. "Vùng chính diện" đo thế nào

Đây là phần cốt lõi, và là chỗ dễ làm ẩu nhất.

**Không phải một bán kính cố định.** Sai lệch parallax tỉ lệ với **chiều cao
linh kiện**:

```
d  ≈  h · r / WD
```

`d` = lệch tâm biểu kiến · `h` = chiều cao linh kiện · `r` = khoảng cách từ
trục quang · `WD` = khoảng cách chụp.

Nên một chip 0402 (h ≈ 0,35 mm) chịu được `r` lớn hơn nhiều so với một tụ hoá
đứng (h ≈ 10 mm) — **gấp gần 30 lần**. Đặt một bán kính chung cho cả board là
sai về bản chất.

Ngưỡng nhận: `d` phải nhỏ hơn **lề ROI** hiện tại, vì vượt lề nghĩa là ROI
trượt khỏi land.

### 4.1. Đo mà KHÔNG cần thông số ống kính

`Docs/thiet_ke/yeu_cau_phan_cung_camera.md` có 46 µm/px và FOV mục tiêu, nhưng
**không có `WD`**, nên công thức trên chưa thay số được. Không cần chờ: đo
thẳng từ ảnh.

**Phép đo:** chụp cùng một board **nhiều lần, dịch board** sao cho một linh
kiện cao (tụ hoá đứng) lần lượt xuất hiện ở tâm và ở các bán kính khác nhau.
Với mỗi lần chụp, đo **lệch giữa tâm thân và tâm cụm pad của chính nó** — pad
nằm sát mặt board nên gần như không parallax, thân thì có. Hiệu số đó chính là
`d`, đo trực tiếp, không cần biết `WD` hay tiêu cự.

Lặp cho vài mức chiều cao (chip mỏng, SOIC, tụ đứng) để dựng `d(h, r)` thực
nghiệm. Từ đó suy ngược `r_max(h)` cho ngưỡng lề ROI.

### 4.2. Câu phải hỏi TRƯỚC, vì nó có thể huỷ cả kế hoạch

**`d` lớn nhất trên board có thật sự vượt lề ROI không?**

Nếu không vượt, toàn bộ việc ghép sơ đồ **không đáng làm** — cứ chọn một ảnh
Golden và dùng nó cho cả board.

### ✅ ĐÃ ĐO 2026-09-05 — và câu trả lời là KHÔNG VƯỢT

Đo trên `tests/data/solder_geometry/board_smd_00001`, 10 linh kiện có ≥2 pad
đếm tay, **trong đó 3 con là `smd_electrolytic_can`** — tức đúng loại cao nhất,
nơi parallax lớn nhất. Quy đổi ở 46 µm/px:

| | µm |
|---|---:|
| lệch tâm thân ↔ tâm cụm pad, trung vị | **82** |
| lệch lớn nhất (D202, sot23) | **413** |
| **lề ROI nhỏ nhất** — pad mong manh nhất trong 28 pad | **736** |
| lề ROI trung vị | 1.587 |

*(Lề ROI đo bằng cách dịch dần từng ROI cho tới khi độ phủ pad tụt dưới 50% —
tức lề THẬT của đường hình học đang chạy, không phải một hằng số cấu hình.)*

**Lệch lớn nhất vẫn dưới lề nhỏ nhất 1,8 lần.**

Và lệch đó **không mang chữ ký của parallax**: nếu là parallax thì mọi véc-tơ
lệch phải cùng chỉ ra xa một điểm. Đo được cos trung bình với hướng xuyên tâm
là **+0,337**, trong khi cùng phép tìm tâm chạy trên hướng **ngẫu nhiên** cho
trung vị +0,326 và p95 +0,588. Tức **không phân biệt được với ngẫu nhiên** —
đây là sai số khoanh box, không phải parallax.

### Nhưng phép đo này CHƯA trả lời được câu hỏi ở quy mô cả board

Phải nói rõ giới hạn, vì kết luận trên dễ bị đọc quá tay:

- Tile chỉ **1024×1024**, cắt từ một board lớn hơn tại offset (1648, 4120).
  Parallax tăng theo bán kính tính từ trục quang; **trong một ô 1024 px thì
  bán kính gần như không đổi**, nên parallax ở đây xuất hiện dưới dạng một
  dịch chuyển gần đồng đều chứ không thành hoa văn xuyên tâm. Phép kiểm hướng
  vì thế **yếu**, và không kết luận được "không có parallax".
- Nhưng **cận trên độ lớn thì vẫn đúng bất kể nguyên nhân**: lệch thân↔pad
  ≤ 413 µm trên ô này, dù nó là parallax hay sai số box.
- Chỗ chưa đo là **góc ngoài cùng của board**, nơi bán kính lớn nhất. Cần một
  ảnh **nguyên board** có pad đếm tay — hiện repo **không có**; fixture duy
  nhất có pad là ô 1024 này.

**⇒ Kết luận tạm: chưa đủ căn cứ để làm việc 2–6.** Việc cần làm không phải
viết code ghép, mà là **có một ảnh nguyên board với vài chục pad đếm tay ở cả
tâm lẫn góc**, rồi chạy lại đúng phép đo này. Nếu ở góc board lệch vẫn dưới
736 µm thì kế hoạch này đóng lại.

---

## 5. Ghép: chọn nguồn nào cho mỗi linh kiện

Sau khi có `r_max(h)`:

1. **Mỗi khung chụp** → detect (lượt 1) → 6.1 → được một danh sách linh kiện
   kèm toạ độ **trong khung đó**.
2. **Quy về hệ toạ độ board** bằng fiducial/bo, dùng `imaging/alignment.py`.
3. **Lọc theo vùng chính diện:** một linh kiện chỉ được khung đó đóng góp khi
   `r < r_max(h)` của nó. Chưa biết `h` thì tra theo gói (§ kế hoạch package)
   hoặc **giả định cao** — giả định cao thì thu hẹp vùng nhận, tức an toàn.
4. **Xung đột:** một linh kiện nằm trong vùng chính diện của nhiều khung thì
   lấy khung có `r` **nhỏ nhất**. Không trung bình cộng: trung bình hai quan
   sát lệch khác hướng cho ra một toạ độ không quan sát nào ủng hộ.
5. **Không khung nào phủ** → ghi vào danh sách hở, **không** lấy đại một khung.
   `uncovered()` đã có sẵn mẫu cho việc này.

### 5.1. Thay đổi schema cần thiết

`MapComponent` hiện `frozen=True` và không có trường nguồn gốc. Cần thêm, đều
có mặc định để không phá dữ liệu cũ:

| trường | vì sao |
|---|---|
| `source_capture: str \| None` | linh kiện này đến từ khung nào — **đây là thứ giữ tính truy xuất** |
| `radial_mm: float \| None` | nó nằm cách trục quang bao xa lúc được đo; cho phép hậu kiểm |
| `height_mm: float \| None` | chiều cao dùng để tính `r_max`; ghi ra để biết đã giả định gì |

`InspectionMap.source` hiện là **một chuỗi**. Sơ đồ ghép cần danh sách khung
nguồn + hash nội dung từng ảnh, giống cách `enrollment.py` đang làm với ứng
viên Golden.

---

## 6. Cổng nghiệm thu — không cần CAD

Không có CAD thì không có toạ độ đúng để so. Nhưng **vùng chồng lấn tự kiểm
được**:

> Với mỗi linh kiện nằm trong vùng chính diện của **≥ 2 khung**, độ lệch toạ độ
> giữa các khung là **cận trên đo được** của sai số sơ đồ.

Đó là phép đo thật, không cần nguồn ngoài. Cổng đề xuất:

1. **Bất đồng giữa các khung** ở vùng chồng lấn: trung vị và p95, tính bằng mm.
   Vượt lề ROI là fail.
2. **Độ phủ:** bao nhiêu linh kiện không khung nào nhìn chính diện. Con số này
   quyết định cần thêm bao nhiêu khung.
3. **So với đường một-ảnh:** sơ đồ ghép phải phủ **không ít hơn** sơ đồ dựng từ
   riêng ảnh Golden. Cùng nguyên tắc bất đối xứng như
   `scripts/evaluate_package_rule_gate.py`: mất thì fail, thừa thì xem lại.

---

## 7. Thứ tự việc

| | việc | chặn bởi | ghi chú |
|---|---|---|---|
| 1 | ~~Đo `d` trên ảnh đang có~~ | — | ✅ **xong 2026-09-05: 413 µm so với lề 736 µm — chưa vượt** |
| 1b | **Có ảnh nguyên board + pad đếm tay ở tâm và ở góc** | bạn | phép đo ở (1) không phủ được góc board |
| 2 | Dựng `d(h, r)` thực nghiệm (§4.1) | (1) nếu (1) nói đáng làm | cần chụp có chủ đích |
| 3 | Ảnh → `InspectionMap` cho **một** khung | (2) | tái dùng detect + 6.1 đang có |
| 4 | Thêm trường nguồn gốc vào schema (§5.1) | (3) | đổi tương thích ngược |
| 5 | Ghép nhiều khung + giải xung đột (§5) | (4) | |
| 6 | Cổng bất đồng vùng chồng lấn (§6) | (5) | |

Việc 1 **đã làm và trả lời "chưa vượt"**, nên việc 2–6 **đang tạm dừng**. Chúng
chỉ khởi động lại nếu (1b) cho thấy ở góc board lệch vượt 736 µm.

---

## 8. Bo của dây chuyền — chụp thế nào để ghép được

> Mục này viết cho **bo thật của bạn**, không phải fixture. Điền kích thước bo
> vào §8.1 trước khi dùng bảng số ở §8.3.

### 8.1. Thông số bo — CẦN BẠN ĐIỀN

| | giá trị | ghi chú |
|---|---|---|
| Kích thước bo | ⬜ … × … mm | quyết định số khung ở §8.3 |
| Linh kiện cao nhất | ⬜ … mm | tụ hoá đứng thường 5–10 mm |
| Bước chân nhỏ nhất | ⬜ … mm | quyết định µm/px cần |
| Có fiducial không | ⬜ có / không | nếu không, xem §8.4 |

### 8.2. "Nhiều GÓC" không dùng được — phải là "nhiều VÙNG"

Đây là chỗ dễ hiểu ngược nhất, và làm sai thì cả bộ ảnh vứt đi.

| | nghiêng máy sang góc khác | giữ vuông góc, **dịch bo** |
|---|---|---|
| Phối cảnh | **tệ hơn** mỗi ảnh | không đổi |
| Thân che pad | **nhiều hơn** | ít nhất có thể |
| Ghép ra mặt nhìn thẳng? | ❌ không | ✅ có |
| Đo được toạ độ? | ❌ mỗi ảnh một phép chiếu khác | ✅ |

Lý do: mục tiêu là **mặt nhìn thẳng từ trên xuống**. Ảnh chính diện đã là thứ
gần nhất với nó; nghiêng đi là đi xa khỏi mục tiêu. Chụp nhiều góc chỉ có ích
khi muốn dựng hình **3D** (thấy mặt bên của mối hàn) — đó là bài toán khác, và
không phải bài toán đang giải ở đây.

**Cách đúng:** máy **vuông góc với mặt bo**, **khoảng cách không đổi**, dịch bo
(hoặc máy) theo lưới để mỗi khung phủ một vùng khác nhau, có chồng biên.

### 8.3. Bao nhiêu khung

Bảng đã có ở mục 3.2 của `Docs/thiet_ke/yeu_cau_phan_cung_camera.md`, đã trừ 15% chồng
biên, cho bo 200 × 150 mm:

| µm/px | Cảm biến | Trường nhìn | Số khung |
|---|---|---|---|
| 46 *(mức hiện tại)* | 12 MP | 185 × 140 mm | **4** |
| **25** *(khuyến nghị)* | **20 MP** | 137 × 91 mm | **4** |
| 15 | 20 MP | 82 × 55 mm | **12** |

Bo của bạn khác 200 × 150 thì số khung đổi theo diện tích. Điền §8.1 rồi tính
lại.

### 8.4. Sáu điều kiện để bộ ảnh ghép được

Thiếu bất kỳ điều nào thì ghép ra sơ đồ sai mà **không có gì báo**.

1. **Vuông góc và cùng khoảng cách.** Lệch khoảng cách làm tỉ lệ mm/px khác
   nhau giữa các khung, và sơ đồ ghép ra sẽ co giãn theo vùng.
2. **Chồng biên ≥ 15%.** Vùng chồng vừa để nối, vừa là **thước đo sai số** —
   §6 dùng bất đồng ở vùng chồng làm cổng. Không chồng thì không kiểm được.
3. **Khoá phơi sáng, cân bằng trắng, tiêu cự.** Máy tự động đổi giữa các khung
   là đổi ngưỡng ảnh, mà 6.2 chấm mối hàn bằng ngưỡng — hai khung khác sáng
   cho hai phán quyết khác nhau trên cùng một mối hàn.
4. **Ánh sáng không đổi.** Cùng lý do, và mạnh hơn: đổi hướng sáng là đổi hình
   dạng vệt bóng trên thiếc.
5. **Mỗi khung phải có ≥ 3 điểm nhận dạng chung với khung kề.** Fiducial là
   tốt nhất; không có thì lỗ ốc, góc bo, hoặc chữ in đặc trưng cũng được. Hai
   điểm chỉ đủ cho tịnh tiến + xoay; ba điểm mới bắt được sai tỉ lệ.
6. **Ít nhất một khung có vật chuẩn dài đã biết** — thước, hoặc một linh kiện
   đã đo bằng thước cặp. Đây là thứ cho **px/mm**, mà hiện **cả dự án chưa có
   con số đó đo trực tiếp** (46 µm/px ở tài liệu phần cứng là suy từ bước chân
   linh kiện, không phải đo bằng vật chuẩn).

### 8.5. Chụp bằng điện thoại cầm tay thì sao

Làm được, nhưng phải biết mất gì:

| | rig cố định | điện thoại cầm tay |
|---|---|---|
| Khoảng cách không đổi | ✅ | ❌ đổi mỗi lần bấm |
| Vuông góc | ✅ | ⚠️ lệch vài độ là thường |
| Khoá phơi sáng/WB | ✅ | ⚠️ phải khoá bằng tay (AE/AF lock) |
| Ghép ra **sơ đồ để xem** | ✅ | ✅ |
| Ghép ra **toạ độ để kiểm tra** | ✅ | ❌ **không** |

Nói thẳng: ảnh điện thoại cầm tay đủ để **dựng sơ đồ sơ bộ và nhìn** — biết bo
có những gì, nằm đại khái ở đâu. **Không** đủ để làm chuẩn kiểm tra, vì sai số
khoảng cách và góc nghiêng trộn thẳng vào toạ độ mà không tách ra được.

Nếu mục tiêu là (a) xem thử ghép có ra gì không, hoặc (b) có sơ đồ nháp để gán
nhãn — thì cứ chụp, dùng được. Nếu là (c) làm chuẩn cho dây chuyền — thì cần
rig, dù rig ở đây chỉ là **một giá đỡ điện thoại cố định và bo trượt trên giấy
kẻ ô**.

### 8.6. Việc tôi làm được với bộ ảnh đó

Theo đúng thứ tự, và mỗi bước có thể dừng nếu bước trước hỏng:

1. Kiểm sáu điều kiện §8.4 trên chính bộ ảnh, **báo cái nào hỏng** trước khi
   ghép — ghép ảnh không đạt điều kiện chỉ tạo ra một sơ đồ trông như đúng.
2. Nắn các khung về một hệ toạ độ chung bằng điểm chung.
3. Chạy detect + 6.1 trên từng khung.
4. Ghép thành `InspectionMap`, mỗi linh kiện ghi rõ đến từ khung nào (§5.1).
5. **Đo bất đồng ở vùng chồng** (§6) — đây là con số nói sơ đồ đáng tin tới đâu,
   và nó đo được **mà không cần CAD**.
6. Xuất sơ đồ + báo cáo, kèm danh sách vùng không khung nào phủ.

---

### 8.7. Quy trình tại chỗ — làm theo đúng thứ tự này

Bo đang ở công ty, nên mục này viết để **cầm theo và làm một mạch**, không cần
hỏi lại.

**Chuẩn bị mang theo**

- Thước kẻ hoặc thước cặp (**bắt buộc** — xem điều 6 ở §8.4).
- Giấy kẻ ô hoặc băng dính giấy để đánh dấu vị trí dịch bo.
- Vật kê để giữ máy cố định: chồng sách, kẹp điện thoại, chân máy — bất cứ thứ
  gì giữ được **khoảng cách và góc không đổi** giữa các lần bấm.

**Bước 1 — đo bo, ghi vào §8.1** *(2 phút, nhưng không có thì mọi tính toán
sau đều treo)*

| đo gì | bằng gì |
|---|---|
| dài × rộng bo | thước |
| chiều cao linh kiện cao nhất | thước cặp, hoặc ước lượng theo tụ hoá cao nhất |
| bước chân nhỏ nhất | nhìn con IC chân dày nhất; SOIC là 1,27 mm, TSSOP 0,65 mm, QFP mịn 0,5 mm |
| có fiducial không | ba chấm tròn mạ đồng ở góc bo |

**Bước 2 — dựng chỗ chụp**

1. Đặt máy **vuông góc** với mặt bo. Cách kiểm rẻ nhất: nhìn bóng phản chiếu
   của ống kính trên một mặt phẳng bóng của bo — bóng nằm giữa khung là vuông.
2. Đặt bo trên giấy kẻ ô, đánh dấu vị trí ban đầu.
3. **Khoá AE/AF/WB.** Trên điện thoại: chạm giữ vào bo cho tới khi hiện
   "AE/AF Lock". Trên máy ảnh: chuyển sang M, ghi lại khẩu/tốc/ISO.
4. Chụp thử một tấm, **phóng to hết cỡ vào chỗ chân IC mịn nhất**. Không đọc
   được ranh giới từng chân thì hạ máy xuống gần hơn và tăng số khung — đây
   chính là vấn đề "chụp không nét" bạn nêu, và nó phải giải ở bước này chứ
   không giải được lúc ghép.

**Bước 3 — chụp**

- Dịch bo theo lưới, **chồng biên ~30%** (dư so với mức tối thiểu 15%; chồng
  thừa không hại gì, chồng thiếu thì phải quay lại).
- **Không đổi khoảng cách, không đổi zoom, không xoay bo** giữa các khung.
- Ít nhất **một khung có thước nằm trong ảnh**, đặt sát mặt bo cho cùng mặt
  phẳng tiêu.
- Đặt tên file theo thứ tự dịch: `r0c0.jpg`, `r0c1.jpg`, `r1c0.jpg`…

**Bước 4 — KIỂM NGAY, trước khi rời khỏi bo**

```bash
python scripts/check_capture_set.py <thư mục ảnh>
```

Nó so **trên vùng chồng của từng cặp khung**, nên nó phân biệt được "hai khung
khác nhau vì máy đổi thiết lập" với "hai khung khác nhau vì chụp chỗ khác":

| báo gì | sửa thế nào, ngay tại chỗ |
|---|---|
| lệch mức sáng | AE chưa khoá — khoá rồi chụp lại cả bộ |
| lệch màu | WB chưa khoá |
| chênh độ nét | một khung mất nét — chụp lại riêng khung đó |
| chồng biên thiếu | dịch bo ít hơn ở chỗ đó |
| khung không chồng với ai | thiếu khung bắc cầu, hoặc vùng đó trơn quá không có điểm nhận dạng |
| ảnh không cùng kích thước | zoom bị đổi giữa chừng |

Ra `ĐẠT` mới ra về. Ra `CHỤP LẠI` thì sửa đúng dòng nó chỉ rồi chạy lại — mỗi
vòng mất vài phút, còn phát hiện ở nhà thì mất cả buổi đi lại.

**Hai điều script KHÔNG kiểm được**, phải tự nhớ: (a) có thước trong ít nhất
một khung không, (b) máy có vuông góc không.

**Bước 5 — gửi tôi**

- Cả thư mục ảnh, **nguyên gốc, không crop, không chỉnh**.
- Bốn số ở §8.1.
- Chiều dài thật của vật chuẩn trong ảnh (ví dụ "vạch thước từ 0 đến 100 mm").
- Kết quả `check_capture_set.py` (copy màn hình cũng được).

---

## 9. Câu hỏi cần quyết
1. **Kích thước bo, chiều cao linh kiện cao nhất, bước chân nhỏ nhất** — điền
   vào §8.1. Không có ba số đó thì không tính được số khung.
2. **Mục tiêu của bộ ảnh sắp chụp là gì?** (a) xem thử, (b) sơ đồ nháp để gán
   nhãn, hay (c) chuẩn kiểm tra cho dây chuyền. (a) và (b) thì điện thoại cầm
   tay đủ; (c) thì phải có giá đỡ cố định — xem §8.5.


1. **Camera/bàn máy có dịch được board theo toạ độ đặt trước không?** §4.1 cần
   chụp cùng một board ở vài vị trí khác nhau. Dịch tay cũng được nhưng phải
   biết dịch bao nhiêu.
2. **Chiều cao linh kiện lấy từ đâu?** Không có CAD thì phải tra theo gói —
   tức phụ thuộc [kế hoạch package](ke_hoach_phan_nhom_package.md). Hoặc giả
   định cao cho mọi thứ, đổi lại vùng chính diện hẹp đi và cần nhiều khung hơn.
3. ~~Lề ROI hiện tại là bao nhiêu mm?~~ **ĐÃ ĐO:** trung vị **1,587 mm**, nhỏ
   nhất **0,736 mm** trên 28 pad của fixture. Đo bằng cách dịch ROI cho tới khi
   độ phủ tụt dưới 50%, nên đó là lề thật chứ không phải hằng số cấu hình.

---

Xem thêm: `Docs/thiet_ke/yeu_cau_phan_cung_camera.md` (46 µm/px, FOV, số khung
chụp) · `Docs/ke_hoach/ke_hoach_so_hoa_mach_pcb_aoi.md` (sơ đồ → CAD/netlist,
**giả định đã có ảnh Golden tốt** — kế hoạch này là tiền đề còn thiếu của nó).
