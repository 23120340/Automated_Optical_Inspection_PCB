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

## 8. Câu hỏi cần quyết

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
